# Copyright (C) 2021 - 2026 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Launch, connect, and session management for the Fluent REST transport.

Everything here is REST/HTTP — there is no gRPC in this stack.  The Fluent web
server is treated as a blackbox: we send it HTTP requests and read its replies.

* :class:`RestSolverSession` - wraps a :class:`FluentRestClient` plus the
  settings tree built by :func:`get_root`, and owns the session lifecycle.
* :func:`launch_webserver` - spawns Fluent locally, waits for an authenticated
  ping, and returns a connected session.
* :func:`connect_to_webserver` - connects to an already-running server given
  its ip, port, and auth token.

TLS certificates are user-supplied.  We discover them (we never create or
delete them); if found we use HTTPS, otherwise we warn once and use HTTP.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import secrets
import socket
import ssl
import subprocess
import time
import urllib.error
import urllib.request

from ansys.fluent.core.launcher.process_launch_string import get_fluent_exe_path
from ansys.fluent.core.rest._tls import build_ssl_context
from ansys.fluent.core.rest.client import FluentRestClient, FluentRestError
from ansys.fluent.core.solver.flobject import Group, get_root

logger = logging.getLogger(__name__)

_LOCALHOST = "127.0.0.1"

# Files a directory must contain to count as a usable TLS cert directory.
_CERT_FILES = ("webserver.crt", "webserver.key", "dh.pem")


# ---------------------------------------------------------------------------
# Helpers - port, token, scheme
# ---------------------------------------------------------------------------


def _get_free_port() -> int:
    """Return a free local TCP port.

    Binds to port 0 so the OS hands out an unused ephemeral port, then reads it
    back.  Two launches on one machine never collide.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((_LOCALHOST, 0))
            return sock.getsockname()[1]
    except OSError as exc:
        raise RuntimeError(f"Could not find a free local TCP port: {exc}") from exc


def _generate_auth_token() -> str:
    """Return a fresh 4-digit auth token (1000-9999) for this launch.

    One token per launch.  The raw number is never sent on the wire — the
    client transmits ``SHA-256(token)`` as the Bearer value.
    """
    token = str(secrets.randbelow(9000) + 1000)
    logger.debug("Generated per-launch auth token.")
    return token


def _bearer_header(auth_token: str) -> dict[str, str]:
    """Return the ``Authorization`` header for *auth_token*.

    The server expects the SHA-256 hash of the token (not the raw token),
    hashed the same way the REST client does.
    """
    token_hash = hashlib.sha256(auth_token.encode()).hexdigest()
    return {"Authorization": f"Bearer {token_hash}"}


def _scheme_for(ssl_context: ssl.SSLContext | None) -> str:
    """Return ``"https"`` when a TLS context is present, else ``"http"``."""
    return "https" if ssl_context else "http"


def _connection_kind(ssl_context: ssl.SSLContext | None) -> str:
    """Return a human label for the connection: secure HTTPS or plain HTTP."""
    return "HTTPS (secure)" if ssl_context else "HTTP (unsecured)"


# ---------------------------------------------------------------------------
# Helpers - TLS certificate discovery (never creates or deletes certs)
# ---------------------------------------------------------------------------


def _dir_has_certs(cert_dir: str | None) -> bool:
    """Return True if *cert_dir* exists and holds all required cert files."""
    if not cert_dir:
        return False
    return all(os.path.isfile(os.path.join(cert_dir, name)) for name in _CERT_FILES)


def _find_cert_dir(cert_dir: str | None) -> str | None:
    """Locate a usable TLS certificate directory, or return None.

    Search order: the explicit *cert_dir* argument, then the
    FLUENT_WEBSERVER_CERTIFICATE_ROOT environment variable (the same variable
    used to point Fluent at certs living under its install path).  We only
    discover user-supplied certs — we never create them.
    """
    candidates = (cert_dir, os.environ.get("FLUENT_WEBSERVER_CERTIFICATE_ROOT"))
    for candidate in candidates:
        if _dir_has_certs(candidate):
            return candidate
    return None


def _resolve_transport_security(
    cert_dir: str | None,
) -> tuple[str | None, ssl.SSLContext | None]:
    """Decide HTTPS vs HTTP from whatever certs the user already has.

    Returns ``(cert_dir, ssl_context)`` when certs are found (HTTPS), or
    ``(None, None)`` after a single warning when none are found (HTTP).
    Never generates or deletes certificates.
    """
    resolved = _find_cert_dir(cert_dir)
    if resolved:
        logger.info("TLS certificates found in %s — using HTTPS.", resolved)
        return resolved, build_ssl_context(resolved)

    logger.warning(
        "No TLS certificates found (checked the cert_dir argument and "
        "FLUENT_WEBSERVER_CERTIFICATE_ROOT). Starting Fluent in HTTP mode."
    )
    return None, None


def _get_fluent_exe(
    product_version: str | None = None,
    fluent_path: str | None = None,
) -> str:
    """Return the Fluent executable path (raises FileNotFoundError if missing)."""
    return str(
        get_fluent_exe_path(product_version=product_version, fluent_path=fluent_path)
    )


# ---------------------------------------------------------------------------
# Helpers - process lifecycle
# ---------------------------------------------------------------------------


def _spawn_fluent(
    fluent_exe: str,
    dimension: str,
    port: int,
    auth_token: str,
    cert_dir: str | None,
) -> subprocess.Popen:
    """Spawn the Fluent web server process.

    Injects the auth token (and cert directory, when present) through the
    environment.  Raises immediately if Fluent dies on startup.
    """
    launch_cmd = [fluent_exe, dimension, "-ws", f"-ws-port={port}"]
    logger.info("Launching Fluent: %s", launch_cmd)

    env = os.environ.copy()
    env["FLUENT_WEBSERVER_TOKEN"] = auth_token
    if cert_dir:
        env["FLUENT_WEBSERVER_CERTIFICATE_ROOT"] = cert_dir

    process = subprocess.Popen(launch_cmd, env=env)  # nosec B603 B607
    if process.poll() is not None:
        raise RuntimeError(
            f"Fluent exited immediately (rc={process.returncode}). "
            f"Command: {launch_cmd}"
        )
    return process


def _terminate_process(process: subprocess.Popen) -> None:
    """Stop *process* and reap it.

    Sends SIGTERM and waits up to 10s; if the process ignores that, sends
    SIGKILL and reaps it so it cannot linger as a zombie.  If the process has
    already exited there is nothing to do.
    """
    process_still_running = process.poll() is None
    if process_still_running:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _register_process_atexit(process: subprocess.Popen) -> None:
    """Terminate a leaked Fluent process at interpreter exit (backstop only).

    If the caller forgets ``session.exit()`` we still avoid leaving an orphaned
    Fluent process behind.  Never touches certificates.
    """

    def _cleanup() -> None:
        _terminate_process(process)

    atexit.register(_cleanup)


# ---------------------------------------------------------------------------
# Helpers - readiness probe (Option A: port open, then authenticated ping)
# ---------------------------------------------------------------------------


def _ping_once(
    ping_url: str,
    auth_token: str,
    ssl_context: ssl.SSLContext | None,
) -> bool:
    """Send one readiness ping and classify the result.

    Returns True when the server replies 200 (up and token accepted) and False
    when the port refuses the connection (server still booting).  Raises
    PermissionError on 401 (wrong token — waiting cannot fix it) and
    FluentRestError on any other reply (surface it, do not mask it).
    """
    request = urllib.request.Request(
        ping_url, method="POST", headers=_bearer_header(auth_token)
    )
    try:
        with urllib.request.urlopen(
            request, timeout=3, context=ssl_context
        ):  # nosec B310
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise PermissionError(
                "Fluent server rejected the auth token (HTTP 401)."
            ) from exc
        raise FluentRestError(
            exc.code, f"Unexpected reply while waiting for readiness: {exc.reason}"
        ) from exc
    except urllib.error.URLError:
        # Connection refused / not listening yet — server is still booting.
        return False


def _block_until_port_open(port: int, deadline: float) -> None:
    """Wait until *port* accepts TCP connections (Phase 1).

    Right after spawn the process is alive but has not bound the port yet.
    This just knocks on the port; it checks neither auth nor solver state.
    Raises TimeoutError if the port never opens — that means the server failed
    to start and the caller should relaunch.
    """
    logger.info("[wait] Phase 1 — waiting for TCP port %d to open...", port)
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((_LOCALHOST, port), timeout=2.0):
                logger.info("[wait] Port %d is open.", port)
                break
        except OSError:
            time.sleep(2)
    else:
        raise TimeoutError(f"Fluent web server on port {port} did not open in time.")


def _block_until_ping_ok(
    port: int,
    auth_token: str,
    deadline: float,
    ssl_context: ssl.SSLContext | None,
) -> None:
    """Wait until the authenticated ping returns 200 (Phase 2).

    A clean 200 proves the server is up AND the token is valid, so the settings
    tree built afterward needs no auth retry.  PermissionError (bad token) and
    FluentRestError (unexpected reply) propagate immediately from
    :func:`_ping_once`; a never-ready server raises TimeoutError here.
    """
    ping_url = f"{_scheme_for(ssl_context)}://{_LOCALHOST}:{port}/api/connection/ping"
    logger.info(
        "[wait] Phase 2 — connecting over %s on port %d...",
        _connection_kind(ssl_context),
        port,
    )
    while time.monotonic() < deadline:
        if _ping_once(ping_url, auth_token, ssl_context):
            logger.info("[wait] Fluent server is ready on port %d.", port)
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"Fluent server on port {port} was not ready in time.")


def _wait_for_server(
    port: int,
    auth_token: str,
    timeout: int = 120,
    ssl_context: ssl.SSLContext | None = None,
) -> None:
    """Block until Fluent is fully ready: port open, then ping accepted.

    Both phases share one deadline so the total wait never exceeds *timeout*.
    """
    deadline = time.monotonic() + timeout
    _block_until_port_open(port, deadline)
    _block_until_ping_ok(port, auth_token, deadline, ssl_context)


def _probe_server(
    base_url: str,
    auth_token: str,
    timeout: float = 5.0,
    ssl_context: ssl.SSLContext | None = None,
) -> bool:
    """Return True if the server answers a one-shot authenticated ping.

    Reachability check for :func:`connect_to_webserver`: 200 means reachable
    and authorized; 401 means reachable but wrong token (False); no reply means
    not reachable (False).  Pure check — never raises.
    """
    ping_url = f"{base_url}/api/connection/ping"
    request = urllib.request.Request(
        ping_url, method="POST", headers=_bearer_header(auth_token)
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=ssl_context
        ):  # nosec B310
            return True
    except urllib.error.HTTPError as exc:
        # The server answered, so it is reachable; only a 401 is a hard "no".
        return exc.code != 401
    except (urllib.error.URLError, OSError):
        return False


# ---------------------------------------------------------------------------
# RestSolverSession
# ---------------------------------------------------------------------------


class RestSolverSession:
    """Solver session that talks to Fluent over REST.

    Builds a :class:`FluentRestClient`, hands it to :func:`get_root` as the
    proxy, and exposes the resulting settings tree via :attr:`settings`.
    Reads, writes, and commands all travel over HTTP — there is no gRPC here.
    """

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: str | None = None,
        component: str = "fluent_1",
        version: str = "261",
        timeout: float = 30.0,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._client = FluentRestClient(
            base_url,
            auth_token=auth_token,
            component=component,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            ssl_context=ssl_context,
        )
        # No auth-retry loop: we reach here only after the launcher's ping (or
        # the connect probe) already proved the token, so get_root will not see
        # a transient 401.
        self._settings = get_root(self._client, version=version)
        self.ip: str | None = None
        self.port: int | None = None
        self.auth_token: str | None = auth_token
        self._process: subprocess.Popen | None = None

    @property
    def client(self) -> "FluentRestClient":
        """Return the underlying REST client for low-level access."""
        return self._client

    @property
    def settings(self) -> "Group":
        """Return the root of the solver settings tree."""
        return self._settings

    def read_case(self, file_name: str) -> None:
        """Read a Fluent case file (routes through the REST settings tree)."""
        logger.info("Reading case file: %s", file_name)
        self._settings.file.read_case(file_name=file_name)

    def read_case_data(self, file_name: str) -> None:
        """Read a Fluent case+data file (routes through the REST settings tree)."""
        logger.info("Reading case+data file: %s", file_name)
        self._settings.file.read_case_data(file_name=file_name)

    def read_data(self, file_name: str) -> None:
        """Read a Fluent data file (routes through the REST settings tree)."""
        logger.info("Reading data file: %s", file_name)
        self._settings.file.read_data(file_name=file_name)

    def exit(self) -> None:
        """Shut the server down gracefully, then ensure any local process is gone.

        Asks the server to stop via REST (``app/exit``).  For a launched
        session we then terminate the local process as a fallback; for a
        connected session there is no local process, so only the remote
        shutdown happens.  The fallback runs even if the server refused, so a
        launched Fluent process never leaks.
        """
        try:
            self._client.exit()
        finally:
            if self._process is not None:
                _terminate_process(self._process)
                self._process = None

    def __enter__(self) -> "RestSolverSession":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.exit()


# ---------------------------------------------------------------------------
# Public API - launchers
# ---------------------------------------------------------------------------


def launch_webserver(
    *,
    product_version: str | None = None,
    fluent_path: str | None = None,
    cert_dir: str | None = None,
    dimension: str = "3ddp",
    start_timeout: int = 60,
    component: str = "fluent_1",
    version: str = "261",
    timeout: float = 30.0,
    max_retries: int = 0,
    retry_delay: float = 1.0,
) -> RestSolverSession:
    """Launch a local Fluent process with the embedded web server.

    Uses the user's existing TLS certificates when found (HTTPS); otherwise
    warns once and starts in HTTP mode.  Generates a token, picks a free port,
    spawns Fluent, waits for an authenticated ping, then returns a connected
    session.  If anything fails after spawning, the spawned process is
    terminated before the error propagates.  Certificates are never created or
    deleted.

    Raises
    ------
    RuntimeError
        If no free port is available or Fluent exits immediately.
    FileNotFoundError
        If the Fluent executable cannot be located.
    PermissionError
        If the server rejects the auth token.
    TimeoutError
        If the server is not ready within *start_timeout* seconds.
    """
    auth_token = _generate_auth_token()
    resolved_cert_dir, ssl_ctx = _resolve_transport_security(cert_dir)
    port = _get_free_port()
    logger.info("Discovered free port %d for Fluent web server.", port)
    fluent_exe = _get_fluent_exe(
        product_version=product_version, fluent_path=fluent_path
    )

    process = _spawn_fluent(fluent_exe, dimension, port, auth_token, resolved_cert_dir)
    _register_process_atexit(process)

    # On any failure after spawning, terminate the process we started and let
    # the original error propagate. On success this block is skipped entirely.
    try:
        _wait_for_server(port, auth_token, timeout=start_timeout, ssl_context=ssl_ctx)
        base_url = f"{_scheme_for(ssl_ctx)}://{_LOCALHOST}:{port}"
        session = RestSolverSession(
            base_url,
            auth_token=auth_token,
            component=component,
            version=version,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            ssl_context=ssl_ctx,
        )
    except Exception as exc:
        logger.error(
            "Launch failed after spawning Fluent (pid=%d) — terminating.",
            process.pid,
        )
        _terminate_process(process)
        raise exc

    session.ip = _LOCALHOST
    session.port = port
    session._process = process
    return session


def connect_to_webserver(
    ip: str,
    port: int,
    auth_token: str,
    *,
    component: str = "fluent_1",
    version: str = "261",
    timeout: float = 30.0,
    max_retries: int = 0,
    retry_delay: float = 1.0,
    ca_cert: str | None = None,
) -> RestSolverSession:
    """Connect to an already-running Fluent REST server.

    Supply ip, port, and auth_token explicitly.  Scheme is auto-detected:
    HTTPS when *ca_cert* is given, HTTP otherwise.  A one-shot authenticated
    ping fails fast if the server is unreachable or the token is wrong.
    Calling ``exit()`` on the returned session shuts the remote server down via
    ``app/exit``.

    Future enhancement (kept explicit on purpose): Fluent writes a
    ``server_info-<session>.txt`` file with the port and token.  A later
    overload could read ip/port/token from that file so the user need not pass
    them by hand — the reachability probe below already isolates the connect
    step, so that change would be localized.

    Raises
    ------
    ConnectionError
        If the server does not answer the reachability probe.
    """
    ssl_ctx = build_ssl_context(ca_cert) if ca_cert else None
    scheme = "https" if ca_cert else "http"
    base_url = f"{scheme}://{ip}:{port}"

    if not _probe_server(
        base_url, auth_token, timeout=min(timeout, 5.0), ssl_context=ssl_ctx
    ):
        raise ConnectionError(
            f"Fluent web server at {base_url} did not answer the authenticated "
            f"ping (POST /api/connection/ping). Check that the server is running "
            f"on this ip/port and that auth_token is correct."
        )

    session = RestSolverSession(
        base_url,
        auth_token=auth_token,
        component=component,
        version=version,
        timeout=timeout,
        max_retries=max_retries,
        retry_delay=retry_delay,
        ssl_context=ssl_ctx,
    )
    session.ip = ip
    session.port = port
    # _process stays None -> exit() shuts the REMOTE server via app/exit.
    return session

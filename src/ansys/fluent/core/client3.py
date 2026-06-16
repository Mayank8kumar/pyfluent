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

"""Thin HTTP client for the Fluent Settings Service REST API.

This client is a **transport layer only**. It builds URLs, attaches
authentication, sends requests, and translates HTTP errors into
:class:`FluentRestError`. All business logic — what a 404 means for
your workflow, whether to retry on a specific status, how to interpret
a confirmation prompt — belongs in the calling layer.

The server is the single source of truth: this client never swallows,
reinterprets, or defaults away a server response.
"""

import hashlib
import json
import logging
import ssl
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_RETRYABLE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


# ------------------------------------------------------------------
# Error
# ------------------------------------------------------------------

class FluentRestError(RuntimeError):
    """HTTP error raised when a Fluent REST request fails.

    This class is the **single place** that understands how to interpret
    transport-level failures.  It knows which HTTP status codes come from
    the server vs. which originate from a broken connection, and it knows
    which failures are transient enough to be worth retrying.

    Attributes
    ----------
    status : int
        HTTP status code.  ``0`` means the request never reached the
        server (connection refused, reset, DNS failure, etc.).
    retryable : bool
        ``True`` when the failure is transient — a 502/503/504 gateway
        error or a connection-level ``OSError`` — and re-issuing the
        same request has a reasonable chance of succeeding.
    """

    _RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})

    def __init__(self, status: int, message: str, *, retryable: bool = False) -> None:
        self.status = status
        self.retryable = retryable
        super().__init__(f"HTTP {status}: {message}")

    @classmethod
    def from_transport(cls, exc: OSError) -> "FluentRestError":
        """Construct from a stdlib transport exception.

        ``urllib`` raises ``HTTPError`` (a subclass of ``OSError``) when
        the server replies with an error status, and plain ``OSError``
        when the connection itself fails.  This factory inspects the
        exception once and produces a fully-populated domain error.
        """
        if isinstance(exc, urllib.error.HTTPError):
            return cls(
                exc.code,
                cls._read_server_message(exc),
                retryable=exc.code in cls._RETRYABLE_STATUS_CODES,
            )
        return cls(0, cls._read_connection_message(exc), retryable=True)

    @staticmethod
    def _read_server_message(exc: urllib.error.HTTPError) -> str:
        """Extract the plain-text body the server sent with the error."""
        raw = exc.read().decode("utf-8", errors="replace")
        return raw.strip() or exc.reason

    @staticmethod
    def _read_connection_message(exc: OSError) -> str:
        """Produce a human-readable message from a connection failure."""
        return str(getattr(exc, "reason", exc))


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class FluentRestClient:
    """HTTP client for the Fluent DataModel REST API.

    Parameters
    ----------
    base_url : str
        Root URL of the Fluent REST server (e.g. ``"http://127.0.0.1:5000"``).
    auth_token : str, optional
        Raw bearer token. SHA-256 hashed before each request per server spec.
    component : str, optional
        DataModel component name. Defaults to ``"fluent_1"`` (solver).
    timeout : float, optional
        Socket timeout in seconds. Defaults to ``30.0``.
    max_retries : int, optional
        Automatic retries on transient failures. Defaults to ``0``.
    retry_delay : float, optional
        Base delay between retries (exponential back-off). Defaults to ``1.0``.
    ssl_context : ssl.SSLContext, optional
        Custom TLS context for HTTPS connections.
    """

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: str | None = None,
        component: str = "fluent_1",
        timeout: float = 30.0,
        max_retries: int = 0,
        retry_delay: float = 1.0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._component = component
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._ssl_context = ssl_context
        self._api_base = f"api/{component}"
        self._is_closed = False
        self._headers = self._make_auth_headers(auth_token)

    @staticmethod
    def _make_auth_headers(auth_token: str | None) -> dict[str, str]:
        """Pre-compute the headers that accompany every request.

        The token is SHA-256 hashed once at construction — not on every
        call — because it never changes for the lifetime of the client.
        """
        if not auth_token:
            return {}
        token_hash = hashlib.sha256(auth_token.encode()).hexdigest()
        return {"Authorization": f"Bearer {token_hash}"}

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_name(name: str) -> str:
        """Percent-encode a single URL segment (object name, command name)."""
        return urllib.parse.quote(name, safe="")

    @staticmethod
    def _encode_path(path: str) -> str:
        """Percent-encode each segment of a ``/``-delimited settings path."""
        return "/".join(
            FluentRestClient._encode_name(seg) for seg in path.split("/")
        )

    def _settings_endpoint(self, path: str) -> str:
        """Build the API endpoint for a settings *path*."""
        return f"{self._api_base}/{self._encode_path(path)}"

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    def _build_request(
        self,
        method: str,
        endpoint: str,
        body: Any = None,
    ) -> urllib.request.Request:
        """Assemble a :class:`urllib.request.Request` ready to send."""
        url = f"{self._base_url}/{endpoint}"
        headers = dict(self._headers)
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(
            url, data=data, headers=headers, method=method.upper()
        )

    def _send_once(self, req: urllib.request.Request) -> Any:
        """Execute one HTTP round-trip and decode the JSON response."""
        with urllib.request.urlopen(
            req, timeout=self._timeout, context=self._ssl_context
        ) as resp:  # nosec B310
            raw = resp.read()
            if not raw.strip():
                return None
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {}

    def _send(self, req: urllib.request.Request) -> Any:
        """Execute one HTTP call, raising :class:`FluentRestError` on failure."""
        try:
            return self._send_once(req)
        except OSError as exc:
            raise FluentRestError.from_transport(exc) from exc

    def _back_off(self, attempt: int) -> None:
        """Sleep with exponential back-off before the next retry."""
        time.sleep(self._retry_delay * (2**attempt))

    def _send_with_retry(self, req: urllib.request.Request, retries: int) -> Any:
        """Execute *req*, retrying up to *retries* times on transient failures.

        The loop covers the retry-eligible attempts.  After all retries
        are exhausted, a final ``_send`` runs with no exception handling —
        if it fails, the error propagates naturally.
        """
        for attempt in range(retries):
            try:
                return self._send(req)
            except FluentRestError as exc:
                if not exc.retryable:
                    raise
                self._back_off(attempt)
        return self._send(req)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        body: Any = None,
    ) -> Any:
        """Send an HTTP request, retrying transient failures on safe methods."""
        if self._is_closed:
            raise FluentRestError(0, "Session is closed")
        req = self._build_request(method, endpoint, body)
        retries = self._max_retries if method.upper() in _RETRYABLE_METHODS else 0
        return self._send_with_retry(req, retries)

    # ------------------------------------------------------------------
    # Settings API — discovery
    # ------------------------------------------------------------------

    def get_static_info(self) -> dict[str, Any]:
        """Return the full settings schema (GET ``static-info``)."""
        return self._request("GET", f"{self._api_base}/static-info")

    # ------------------------------------------------------------------
    # Settings API — read
    # ------------------------------------------------------------------

    def get_var(self, path: str) -> Any:
        """Read the value at *path* (POST ``get_var``)."""
        return self._request(
            "POST",
            f"{self._api_base}/get_var",
            body={"path": path.lstrip("/")},
        )

    def get_attrs(self, path: str, attrs: list[str], *, recursive: bool = False) -> Any:
        """Read attributes for *path* (POST ``get_attrs``).

        Uses the server's ``get_attrs`` endpoint which accepts a structured
        body — supporting ``attrs``, ``recursive``, and ``children`` — rather
        than the simpler GET query-parameter approach.
        """
        body: dict[str, Any] = {"path": path, "attrs": attrs, "recursive": recursive}
        return self._request("POST", f"{self._api_base}/get_attrs", body=body)

    def get_object_names(self, path: str) -> list[str]:
        """Return child object names at *path* (GET ``{path}``).

        The server returns either a JSON array or a dict keyed by name.
        Both shapes are normalised to a plain list.

        Raises
        ------
        FluentRestError
            Propagated as-is from the server (including 404 if the path
            does not exist — the caller decides what that means).
        """
        result = self._request("GET", self._settings_endpoint(path))
        return self._names_from(result)

    def get_list_size(self, path: str) -> int:
        """Return the element count of the list-object at *path* (GET ``{path}``).

        Raises
        ------
        FluentRestError
            Propagated as-is from the server.
        """
        result = self._request("GET", self._settings_endpoint(path))
        return self._size_from(result)

    # ------------------------------------------------------------------
    # Settings API — write
    # ------------------------------------------------------------------

    def set_var(self, path: str, value: Any) -> None:
        """Write *value* at *path* (PUT ``{path}``)."""
        self._request("PUT", self._settings_endpoint(path), body=value)

    def resize_list_object(self, path: str, size: int) -> None:
        """Resize the list-object at *path* to *size* elements (POST ``{path}``)."""
        self._request("POST", self._settings_endpoint(path), body={"new-size": size})

    # ------------------------------------------------------------------
    # Settings API — named-object CRUD
    # ------------------------------------------------------------------

    def create(self, path: str, name: str = "", properties: dict | None = None) -> Any:
        """Create a child object at *path* (POST ``{path}``)."""
        body = dict(properties) if properties else {}
        if name:
            body["name"] = name
        return self._request("POST", self._settings_endpoint(path), body=body)

    def delete(self, path: str, name: str) -> None:
        """Delete named object *name* at *path* (DELETE ``{path}/{name}``)."""
        encoded_name = self._encode_name(name)
        self._request("DELETE", f"{self._settings_endpoint(path)}/{encoded_name}")

    def rename(self, path: str, new: str, old: str) -> None:
        """Rename *old* to *new* at *path* (PUT ``{path}/{old}``)."""
        encoded_old = self._encode_name(old)
        self._request(
            "PUT",
            f"{self._settings_endpoint(path)}/{encoded_old}",
            body={"name": new},
        )

    def delete_child_objects(
        self,
        path: str,
        obj_type: str,
        child_names: list[str],
    ) -> None:
        """Delete specific named children of *obj_type* under *path*."""
        for name in child_names:
            self.delete(f"{path}/{obj_type}", name)

    def delete_all_child_objects(self, path: str, obj_type: str) -> None:
        """Delete every named child of *obj_type* under *path*."""
        names = self.get_object_names(f"{path}/{obj_type}")
        self.delete_child_objects(path, obj_type, names)

    # ------------------------------------------------------------------
    # Settings API — commands & queries
    # ------------------------------------------------------------------

    def _execute(self, path: str, name: str, **kwds) -> Any:
        """POST a command/query endpoint and return the response payload."""
        encoded_name = self._encode_name(name)
        return self._request(
            "POST",
            f"{self._settings_endpoint(path)}/{encoded_name}",
            body=kwds,
        )

    def execute_cmd(self, path: str, command: str, force: bool = False, **kwds) -> Any:
        """Execute *command* at *path*.

        When ``force=True`` the server skips its confirmation prompt.
        Defaults to ``False`` so the server's confirmation flow (409 →
        prompt → resend with ``force=True``) is respected by default.
        """
        encoded_command = self._encode_name(command)
        endpoint = f"{self._settings_endpoint(path)}/{encoded_command}"
        if force:
            endpoint += "?force=true"
        return self._request("POST", endpoint, body=kwds)

    def execute_query(self, path: str, query: str, **kwds) -> Any:
        """Execute *query* at *path* (POST ``{path}/{query}``)."""
        return self._execute(path, query, **kwds)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _send_exit_request(self) -> None:
        """Ask the server to shut down.

        A 403/409 means the server actively refused and is propagated.
        A dropped connection means the server is already going down.
        """
        try:
            self._request("POST", "api/app/exit")
        except FluentRestError as exc:
            if exc.status in (403, 409):
                logger.warning("Exit blocked (HTTP %d): %s", exc.status, exc)
                raise
        except OSError:
            pass

    def exit(self) -> None:
        """Request shutdown and mark the session closed."""
        if self._is_closed:
            return
        self._send_exit_request()
        self._is_closed = True
        logger.info("Fluent server terminated.")

    def __enter__(self) -> "FluentRestClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.exit()

    # ------------------------------------------------------------------
    # Response-shape interpreters
    # ------------------------------------------------------------------

    @staticmethod
    def _names_from(result: Any) -> list[str]:
        """Normalise a child-listing response to a plain list of names.

        The server returns either a JSON array ``["a", "b"]`` or a dict
        keyed by object name ``{"a": {...}, "b": {...}}``.
        """
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return list(result.keys())
        return []

    @staticmethod
    def _size_from(result: Any) -> int:
        """Extract an element count from a list-object response.

        A list-object reports its length directly; a named-object container
        may include an explicit ``size`` field or just its key count.
        """
        if isinstance(result, list):
            return len(result)
        if isinstance(result, dict):
            return result.get("size", len(result))
        return 0

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

"""Pure-Python REST client for the Fluent solver settings (DataModel API).

Connects to the Fluent embedded web server that exposes the solver settings
via a DataModel REST API.  The base path for all settings endpoints is
``/api/{component}/`` where *component* is ``"fluent_1"`` for a solver session
(``"fluent_meshing_1"`` for a meshing session).

API endpoints (from ``/openapi.json`` on a live Fluent server)
--------------------------------------------------------------

.. code-block:: text

    GET  /api/fluent_1/static-info
         Returns the full settings schema.

    POST /api/fluent_1/get_var
         body: { "path": "<path>" }
         Returns the current value at <path>.

    GET  /api/fluent_1/{dmpath}
         Returns the value / object at <dmpath>.

    PUT  /api/fluent_1/{dmpath}
         body: <json-value>
         Sets the value at <dmpath> (raw value, not wrapped).

    POST /api/fluent_1/{dmpath}
         body: { <command-args> }
         Executes a command at <dmpath>.

    DELETE /api/fluent_1/{path}
         Deletes the named object at <path>.

    GET  /api/fluent_1/{path}?attrs=attr1,attr2[&recursive=true]
         Returns attribute info for the setting at <path>.
         The server routes to ``getAttrs`` when the ``attrs`` query
         parameter is present.

Authentication
~~~~~~~~~~~~~~
Every request carries the header::

    Authorization: Bearer <sha256(auth_token)>

where *auth_token* is the password set when the Fluent session was started.

Error handling
~~~~~~~~~~~~~~
HTTP 4xx / 5xx responses raise :class:`FluentRestError`.
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
import warnings

logger = logging.getLogger(__name__)

# HTTP status codes eligible for automatic retry.
_RETRYABLE_STATUS_CODES = frozenset({502, 503, 504})

# Only idempotent methods are safe to re-issue after a transient failure.
_RETRYABLE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _validate_base_url(base_url: str) -> None:
    """Raise ValueError if *base_url* is not a usable http(s) URL.

    Fail-fast on a programmer mistake (missing scheme/host) instead of a
    confusing error deep inside urlopen later.
    """
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("base_url scheme must be http or https")
    if not parsed.netloc:
        raise ValueError("base_url must include host")


def _warn_if_token_sent_over_http(
    base_url: str,
    auth_token: str | None,
    ssl_context: ssl.SSLContext | None,
) -> None:
    """Warn once if a token would travel over unencrypted HTTP."""
    is_plain_http = urllib.parse.urlparse(base_url).scheme == "http"
    if auth_token and is_plain_http and ssl_context is None:
        warnings.warn(
            "auth_token is being sent over plain HTTP. "
            "Use https:// to protect credentials in transit.",
            stacklevel=3,
        )


class FluentRestError(RuntimeError):
    """Raised when the Fluent REST server returns an error response.

    Parameters
    ----------
    status : int
        HTTP status code.
    message : str
        Error detail from the response body, or the raw reason phrase.
    """

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(f"HTTP {status}: {message}")


class FluentRestClient:
    """Pure-Python HTTP client for the Fluent DataModel REST API.

    The public method signatures are intentionally identical to the duck-typed
    *flproxy* interface consumed by
    :func:`~ansys.fluent.core.solver.flobject.get_root`, so this client can be
    passed directly as *flproxy* to build the full settings tree over HTTP
    instead of gRPC.

    Parameters
    ----------
    base_url : str
        Root URL of the Fluent REST server, e.g. ``"http://127.0.0.1:<port>"``.
        A trailing slash is stripped automatically.
    auth_token : str, optional
        Raw bearer token (the password set when Fluent was started).  Before
        each request the token is SHA-256 hashed and sent as
        ``Authorization: Bearer <sha256(auth_token)>``.
    component : str, optional
        DataModel component name.  Defaults to ``"fluent_1"`` (solver).
        Use ``"fluent_meshing_1"`` for a meshing session.
    timeout : float, optional
        Socket timeout in seconds for every request.  Defaults to ``30.0``.
    max_retries : int, optional
        Maximum number of automatic retries on transient connection errors
        (``URLError``) or HTTP 502/503/504 responses.  Defaults to ``0``
        (no retries — fail immediately).
    retry_delay : float, optional
        Base delay in seconds between retries.  Uses exponential back-off:
        ``retry_delay * 2 ** attempt``.  Defaults to ``1.0``.

    Examples
    --------
    >>> from ansys.fluent.core.rest import FluentRestClient
    >>> client = FluentRestClient(
    ...     "http://127.0.0.1:<port>",
    ...     auth_token="<token>",
    ...     component="fluent_1",
    ... )
    >>> client.get_var("setup/models/energy/enabled")
    True
    >>> client.set_var("setup/models/energy/enabled", False)
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
        _validate_base_url(base_url)
        _warn_if_token_sent_over_http(base_url, auth_token, ssl_context)
        self._base_url = base_url.rstrip("/")
        self._auth_token = auth_token
        self._component = component
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._ssl_context = ssl_context
        # All DataModel endpoints live under this prefix, e.g. "api/fluent_1"
        self._api_base = f"api/{component}"
        self._is_closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, endpoint: str) -> str:
        """Build a full URL by joining *base_url* with *endpoint*.

        Parameters
        ----------
        endpoint : str
            Relative path, e.g. ``"api/fluent_1/static-info"``.

        Returns
        -------
        str
            Absolute URL.
        """
        return f"{self._base_url}/{endpoint}"

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        body: Any = None,
    ) -> Any:
        """Send an HTTP request and return the decoded JSON response body.

        Thin orchestration only: guard the closed state, then send — retrying
        a transient failure on idempotent methods.  Request assembly,
        transport, and the retry policy each live in their own helper.

        Returns
        -------
        dict
            Decoded JSON response, or ``{}`` for empty 2xx bodies.

        Raises
        ------
        FluentRestError
            For any HTTP 4xx/5xx response or connection-level failure.
        """
        if self._is_closed:
            raise FluentRestError(0, "Client is closed")

        req = self._build_request(method, endpoint, body)
        for attempt in range(self._max_retries + 1):
            try:
                return self._send_once(req)
            except FluentRestError as exc:
                if not self._should_retry(exc.status, method, attempt):
                    raise exc
                logger.warning(
                    "Transient failure (HTTP %d) on %s %s — retry %d/%d.",
                    exc.status,
                    method,
                    endpoint,
                    attempt + 1,
                    self._max_retries,
                )
                self._back_off(attempt)

    def _build_request(
        self,
        method: str,
        endpoint: str,
        body: Any = None,
    ) -> urllib.request.Request:
        """Assemble the request: URL, JSON body, content-type, and auth header."""
        headers: dict[str, str] = {}
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self._auth_token:
            token_hash = hashlib.sha256(self._auth_token.encode()).hexdigest()
            headers["Authorization"] = f"Bearer {token_hash}"
        return urllib.request.Request(
            self._url(endpoint), data=data, headers=headers, method=method.upper()
        )

    def _send_once(self, req: urllib.request.Request) -> Any:
        """Do one round-trip and decode JSON, mapping transport errors.

        Any HTTP status error or connection failure becomes a
        :class:`FluentRestError`, so callers handle one error type.
        """
        try:
            with urllib.request.urlopen(
                req, timeout=self._timeout, context=self._ssl_context
            ) as resp:  # nosec B310
                raw = resp.read()
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            raise FluentRestError(exc.code, self._error_detail(exc)) from exc
        except urllib.error.URLError as exc:
            raise FluentRestError(0, str(exc.reason)) from exc

    @staticmethod
    def _error_detail(exc: urllib.error.HTTPError) -> str:
        """Return the server's ``detail`` message, or its reason phrase."""
        try:
            return json.loads(exc.read()).get("detail", exc.reason)
        except (ValueError, AttributeError):
            return exc.reason

    def _should_retry(self, status: int, method: str, attempt: int) -> bool:
        """Return True only for a transient failure on an idempotent method."""
        return (
            attempt < self._max_retries
            and method.upper() in _RETRYABLE_METHODS
            and status in _RETRYABLE_STATUS_CODES
        )

    def _back_off(self, attempt: int) -> None:
        """Sleep with exponential back-off before the next retry."""
        time.sleep(self._retry_delay * (2**attempt))

    # ------------------------------------------------------------------
    # flobject proxy interface
    # ------------------------------------------------------------------

    def get_static_info(self) -> dict[str, Any]:
        """Return the full settings schema.

        Calls ``GET /api/{component}/static-info``.

        Returns
        -------
        dict[str, Any]
            A nested dict describing the settings tree structure, with keys
            such as ``"type"``, ``"children"``, ``"commands"``.
        """
        return self._request("GET", f"{self._api_base}/static-info")

    def get_var(self, path: str) -> Any:
        """Return the current value of the setting at *path*.

        Calls ``POST /api/{component}/get_var`` with body ``{"path": path}``.

        Parameters
        ----------
        path : str
            Slash-delimited settings path, e.g.
            ``"setup/models/energy/enabled"``.

        Returns
        -------
        Any
            The value at *path* — may be a bool, int, float, str, list, or
            dict (for group-level reads).

        Raises
        ------
        FluentRestError
            If the path does not exist (HTTP 404) or the server returns an
            error.
        """
        return self._request("POST", f"{self._api_base}/get_var", body={"path": path})

    def set_var(self, path: str, value: Any) -> None:
        """Set the value of the setting at *path*.

        Calls ``PUT /api/{component}/{path}`` with the value as the JSON body.
        The server expects the raw value directly, not wrapped in ``{"value": ...}``.

        Parameters
        ----------
        path : str
            Slash-delimited settings path.
        value : Any
            New value (bool, int, float, str, list, or dict).

        Raises
        ------
        FluentRestError
            If the server rejects the value (e.g. validation failure).
        """
        self._request("PUT", f"{self._api_base}/{path}", body=value)

    def get_attrs(self, path: str, attrs: list[str], recursive: bool = False) -> Any:
        """Return the requested attributes for the setting at *path*.

        Calls ``GET /api/{component}/{path}?attrs=attr1,attr2&recursive=true``.
        The server-side ``handleGet`` routes to ``getAttrs`` when the ``attrs``
        query parameter is present.

        Parameters
        ----------
        path : str
            Slash-delimited settings path.
        attrs : list[str]
            Attribute names to retrieve, e.g. ``["allowed-values"]``,
            ``["active?", "read-only?"]``.
        recursive : bool, optional
            If ``True``, include attributes of child nodes.  Defaults to
            ``False``.

        Returns
        -------
        dict
            A dict with an ``"attrs"`` key mapping to the requested
            attribute values, e.g.
            ``{"attrs": {"allowed-values": ["laminar", "k-epsilon", ...]}}``.

        Notes
        -----
        Attributes like ``active?`` and ``read-only?`` are solver-computed
        metadata and cannot be modified via :meth:`set_var`.
        """
        params = {"attrs": ",".join(attrs)}
        if recursive:
            params["recursive"] = "true"
        query = urllib.parse.urlencode(params)
        return self._request("GET", f"{self._api_base}/{path}?{query}")

    def get_object_names(self, path: str) -> list[str]:
        """Return the child named-object names at *path*.

        Calls ``GET /api/{component}/{path}`` and extracts the object names
        from the response dict keys.

        Parameters
        ----------
        path : str
            Path to a named-object container, e.g.
            ``"setup/boundary-conditions/velocity-inlet"``.

        Returns
        -------
        list[str]
            Sorted or insertion-order list of child names.  Returns ``[]``
            if the path does not exist (HTTP 404).

        Raises
        ------
        FluentRestError
            If the server returns an unexpected error.
        """
        try:
            result = self._request("GET", f"{self._api_base}/{path}")
        except FluentRestError as exc:
            if exc.status == 404:
                return []
            raise exc
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # Real Fluent returns named objects as dict with names as keys:
            # {"hot-inlet": {...}, "cold-inlet": {...}}
            return list(result.keys())
        return []

    def create(self, path: str, name: str) -> None:
        """Create a named child object *name* at *path*.

        Calls ``POST /api/{component}/{path}`` with body ``{"name": name}``.

        Parameters
        ----------
        path : str
            Path to the named-object container.
        name : str
            Name of the new child object.

        Raises
        ------
        FluentRestError
            If the server rejects the creation.
        """
        self._request("POST", f"{self._api_base}/{path}", body={"name": name})

    def delete(self, path: str, name: str, *, ignore_not_found: bool = False) -> None:
        """Delete the named child object *name* at *path*.

        Calls ``DELETE /api/{component}/{path}/{name}``.

        Parameters
        ----------
        path : str
            Path to the named-object container.
        name : str
            Name of the child object to delete.
        ignore_not_found : bool, optional
            If ``True``, silently ignore HTTP 404 (object already absent).
            Defaults to ``False`` for consistency with the gRPC proxy, but
            callers performing idempotent cleanup should pass ``True``.

        Raises
        ------
        FluentRestError
            If *ignore_not_found* is ``False`` and the object does not exist
            (HTTP 404), or on any other server error.
        """
        try:
            self._request("DELETE", f"{self._api_base}/{path}/{name}")
        except FluentRestError as exc:
            object_already_absent = ignore_not_found and exc.status == 404
            if not object_already_absent:
                raise exc
            # Object already gone — idempotent delete, nothing more to do.

    def rename(self, path: str, new: str, old: str) -> None:
        """Rename a child object at *path* from *old* to *new*.

        Calls ``PUT /api/{component}/{path}`` with body
        ``{"rename": {"new": new, "old": old}}``.

        Parameters
        ----------
        path : str
            Path to the named-object container.
        new : str
            New name for the child object.
        old : str
            Current name of the child object.

        Raises
        ------
        FluentRestError
            If the object *old* does not exist.
        """
        self._request(
            "PUT",
            f"{self._api_base}/{path}",
            body={"rename": {"new": new, "old": old}},
        )

    def delete_child_objects(
        self,
        path: str,
        obj_type: str,
        child_names: list[str],
    ) -> None:
        """Delete specific named children of *obj_type* under *path*.

        Calls ``DELETE /api/{component}/{path}/{obj_type}/{name}`` once for
        each entry in *child_names*.  This is the REST equivalent of the gRPC
        ``DeleteChildObjectsRequest`` with an explicit name list.

        Parameters
        ----------
        path : str
            Path to the parent container, e.g. ``"setup/boundary-conditions"``.
        obj_type : str
            Child object type (sub-container name), e.g. ``"velocity-inlet"``.
        child_names : list[str]
            Names of the child objects to delete.

        Raises
        ------
        FluentRestError
            If any individual delete fails (e.g. HTTP 404 — object not found).
        """
        for name in child_names:
            self.delete(f"{path}/{obj_type}", name)

    def delete_all_child_objects(self, path: str, obj_type: str) -> None:
        """Delete all named children of *obj_type* under *path*.

        Discovers children via :meth:`get_object_names` and then calls
        :meth:`delete_child_objects` for all of them.  This is the REST
        equivalent of the gRPC ``DeleteChildObjectsRequest`` with
        ``delete_all = True``.

        Parameters
        ----------
        path : str
            Path to the parent container, e.g. ``"setup/boundary-conditions"``.
        obj_type : str
            Child object type (sub-container name), e.g. ``"velocity-inlet"``.

        Raises
        ------
        FluentRestError
            If any individual delete fails.
        """
        names = self.get_object_names(f"{path}/{obj_type}")
        self.delete_child_objects(path, obj_type, names)

    def get_list_size(self, path: str) -> int:
        """Return the number of elements in the list-object at *path*.

        Calls ``GET /api/{component}/{path}`` and counts the entries.

        .. note::

            This method makes an independent ``GET`` request rather than
            delegating to :meth:`get_object_names` because it also handles
            list-objects that carry a ``"size"`` key and raw arrays, which
            ``get_object_names`` does not support.

        Parameters
        ----------
        path : str
            Path to a named-object container or list-object.

        Returns
        -------
        int
            Number of child objects.

        Raises
        ------
        FluentRestError
            Propagated as-is from the server (including 404 if the path does
            not exist — the caller decides what that means).
        """
        result = self._request("GET", f"{self._api_base}/{path}")
        if isinstance(result, list):
            return len(result)
        if isinstance(result, dict):
            # Explicit size field from list-objects
            if "size" in result:
                return result["size"]
            # Named-object containers: count the keys (object names)
            return len(result)
        return 0

    def resize_list_object(self, path: str, size: int) -> None:
        """Resize the list-object at *path* to *size* elements.

        Calls ``PUT /api/{component}/{path}`` with body ``{"size": size}``.

        Parameters
        ----------
        path : str
            Path to the list-object.
        size : int
            Desired number of elements.

        Raises
        ------
        FluentRestError
            If the server rejects the resize.
        """
        self._request("PUT", f"{self._api_base}/{path}", body={"size": size})

    def _execute(self, path: str, name: str, **kwds) -> Any:
        """Post a command or query and return the ``"reply"`` payload.

        The launcher's authenticated readiness ping already guarantees the
        solver is up before any command runs, so this issues a single request
        with no solver-readiness retry loop.
        """
        result = self._request("POST", f"{self._api_base}/{path}/{name}", body=kwds)
        return result.get("reply") if isinstance(result, dict) else result

    def execute_cmd(self, path: str, command: str, **kwds) -> Any:
        """Execute *command* at *path* with keyword arguments.

        Calls ``POST /api/{component}/{path}/{command}`` with body ``kwds``.
        Identical to :meth:`execute_query` at the transport level; both are
        required by the ``flobject`` proxy interface (``BaseCommand`` calls
        ``execute_cmd``, ``BaseQuery`` calls ``execute_query``).

        Parameters
        ----------
        path : str
            Path to the parent object containing the command.
        command : str
            Command name, e.g. ``"initialize"``.
        **kwds
            Arbitrary keyword arguments forwarded as the JSON request body.

        Returns
        -------
        Any
            The ``"reply"`` field from the response, or the raw response
            if no ``"reply"`` key is present.

        Raises
        ------
        FluentRestError
            If the server rejects the command (e.g. HTTP 409 conflict).
        """
        return self._execute(path, command, **kwds)

    def execute_query(self, path: str, query: str, **kwds) -> Any:
        """Execute *query* at *path* with keyword arguments.

        Calls ``POST /api/{component}/{path}/{query}`` with body ``kwds``.
        Identical to :meth:`execute_cmd` at the transport level; both are
        required by the ``flobject`` proxy interface (``BaseCommand`` calls
        ``execute_cmd``, ``BaseQuery`` calls ``execute_query``).

        Parameters
        ----------
        path : str
            Path to the parent object containing the query.
        query : str
            Query name, e.g. ``"get-zone-names"``.
        **kwds
            Arbitrary keyword arguments forwarded as the JSON request body.

        Returns
        -------
        Any
            The ``"reply"`` field from the response, or the raw response
            if no ``"reply"`` key is present.

        Raises
        ------
        FluentRestError
            If the server rejects the query.
        """
        return self._execute(path, query, **kwds)

    # ------------------------------------------------------------------
    # Additional proxy interface helpers (no server round-trip required)
    # ------------------------------------------------------------------

    def has_wildcard(self, name: str) -> bool:
        """Return ``True`` if *name* contains an ``fnmatch``-style wildcard.

        Recognised wildcard characters: ``*``, ``?``, ``[``.
        Performs the check locally — no server round-trip required.

        Parameters
        ----------
        name : str
            The name to check.

        Returns
        -------
        bool
            ``True`` if *name* contains a wildcard character.
        """
        return any(c in name for c in ("*", "?", "["))

    def is_interactive_mode(self) -> bool:
        """Return ``False`` always.

        The REST transport does not support interactive command prompts.
        Returning ``False`` prevents ``flobject.BaseCommand`` from calling
        :meth:`get_command_confirmation_prompt`, which is not meaningful
        over HTTP.

        Returns
        -------
        bool
            Always ``False``.
        """
        return False

    def get_command_confirmation_prompt(self, path: str, **kwargs) -> str:
        """Return an empty string — interactive prompts are not supported over REST.

        This method satisfies the *flproxy* interface contract required by
        ``flobject.BaseCommand``.  Since :meth:`is_interactive_mode` always
        returns ``False``, this method will never be called in practice.

        Returns
        -------
        str
            Always an empty string.
        """
        return ""

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def exit(self) -> None:
        """Ask the Fluent server to shut down, then mark this client closed.

        Sends ``POST api/app/exit``.  Safe to call more than once — the second
        call is a no-op.
        """
        if not self._is_closed:
            self._send_exit_request()
            self._is_closed = True
            logger.info("Fluent server shutdown requested.")

    def _send_exit_request(self) -> None:
        """Send the shutdown request, interpreting the server's answer.

        A 403/409 means the server actively refused (propagated so the caller
        knows); any other error status means the server is going down anyway;
        a dropped connection means it is already shutting down.
        """
        try:
            self._request("POST", "api/app/exit")
        except FluentRestError as exc:
            if exc.status in (403, 409):
                logger.warning("Server refused exit (HTTP %d): %s", exc.status, exc)
                raise exc
            # Other statuses: the server is shutting down regardless — done.
        except OSError:
            # Connection dropped mid-request: the server is already going down.
            logger.debug("Connection dropped during exit — server already down.")

    def __enter__(self) -> "FluentRestClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.exit()

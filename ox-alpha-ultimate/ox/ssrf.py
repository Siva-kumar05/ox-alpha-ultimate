"""SSRF-safe outbound HTTP for OX-ALPHA services.

Every server-side request to an operator/user-influenced URL must go through
this module.  Rules enforced (fail-closed):

* ``http``/``https`` schemes only — no file/ftp/data URLs.
* No credentials embedded in the URL.
* Hostname must not be localhost/local/internal by name.
* Every resolved IP must be globally routable — loopback, RFC1918 private,
  link-local (169.254.x, incl. cloud metadata), shared, multicast and
  reserved ranges are rejected.
* Optional host allowlist (``allowed_hosts``) for fixed-integration points
  such as the Telegram bot API.
* Redirects are re-validated, so a public URL cannot bounce the fetcher at
  an internal address.
* Responses are size-capped.

DNS lookups are cached briefly to keep high-frequency pollers cheap.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .core import SecurityError

_DNS_CACHE: dict[str, tuple[float, list[ipaddress.IPAddress]]] = {}
_DNS_TTL_SECONDS = 300.0

_BLOCKED_NAME_SUFFIXES = (
    ".localhost", ".local", ".internal", ".intranet", ".lan",
    ".home", ".corp", ".localdomain",
)


class SafeURLViolation(SecurityError):
    """Raised when a URL fails the SSRF guard. Callers must fail closed."""


def _resolve_host(host: str, port: int) -> list[ipaddress.IPAddress]:
    key = f"{host}:{port}"
    cached = _DNS_CACHE.get(key)
    now = time.monotonic()
    if cached and now - cached[0] < _DNS_TTL_SECONDS:
        return cached[1]
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise SafeURLViolation(f"dns resolution failed for {host!r}: {exc}") from exc
    if not infos:
        raise SafeURLViolation(f"no addresses resolved for {host!r}")
    ips: list[ipaddress.IPAddress] = []
    for info in infos:
        addr = info[4][0]
        try:
            ips.append(ipaddress.ip_address(addr))
        except ValueError as exc:
            raise SafeURLViolation(f"unparseable address {addr!r}") from exc
    _DNS_CACHE[key] = (now, ips)
    return ips


def assert_safe_url(url: str, allowed_hosts: set[str] | None = None) -> None:
    """Raise :class:`SafeURLViolation` unless ``url`` is safe to request."""
    if not isinstance(url, str) or not url.strip():
        raise SafeURLViolation("empty url")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SafeURLViolation(f"scheme {parsed.scheme!r} not allowed: {url!r}")
    if parsed.username or parsed.password:
        raise SafeURLViolation(f"credentials in url rejected: {url!r}")
    host = parsed.hostname
    if not host:
        raise SafeURLViolation(f"no hostname in url: {url!r}")
    host_lower = host.lower().rstrip(".")
    if (
        host_lower == "localhost"
        or host_lower.endswith(_BLOCKED_NAME_SUFFIXES)
    ):
        raise SafeURLViolation(f"blocked hostname {host!r}")
    if allowed_hosts is not None and host_lower not in {h.lower() for h in allowed_hosts}:
        raise SafeURLViolation(f"host {host!r} not in allowlist")

    # Literal IP in the URL: validate directly.
    try:
        literal = ipaddress.ip_address(host_lower)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise SafeURLViolation(f"non-global address {host!r} blocked")
        return

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    for ip in _resolve_host(host_lower, port):
        if not ip.is_global:
            raise SafeURLViolation(
                f"host {host!r} resolves to non-global address {ip} (private/loopback/reserved)"
            )


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-run the SSRF guard on every redirect target."""

    def __init__(self, allowed_hosts: set[str] | None):
        self._allowed_hosts = allowed_hosts
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        assert_safe_url(newurl, self._allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_DEFAULT_HEADERS = {
    "User-Agent": "ox-alpha-agent/2.0 (+local research agent)",
    "Accept": "application/rss+xml, application/xml, application/json, text/html;q=0.8, */*;q=0.5",
}


def _open(url: str, timeout: float, max_bytes: int, headers: dict[str, str] | None,
          allowed_hosts: set[str] | None, data: bytes | None):
    assert_safe_url(url, allowed_hosts)
    request = urllib.request.Request(url, data=data, headers={**_DEFAULT_HEADERS, **(headers or {})})
    opener = urllib.request.build_opener(_GuardedRedirectHandler(allowed_hosts))
    response = opener.open(request, timeout=timeout)
    payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise SafeURLViolation(f"response from {url!r} exceeds max_bytes={max_bytes}")
    return payload


def safe_fetch(
    url: str,
    *,
    timeout: float = 10.0,
    max_bytes: int = 2_000_000,
    headers: dict[str, str] | None = None,
    allowed_hosts: set[str] | None = None,
) -> bytes:
    """GET an external URL with full SSRF protection (including redirects)."""
    return _open(url, timeout, max_bytes, headers, allowed_hosts, data=None)


def safe_post_json(
    url: str,
    payload: dict,
    *,
    timeout: float = 10.0,
    headers: dict[str, str] | None = None,
    allowed_hosts: set[str] | None = None,
) -> dict:
    """POST a JSON body and parse the JSON response, with SSRF protection."""
    body = json.dumps(payload).encode("utf-8")
    merged = {"Content-Type": "application/json", **(headers or {})}
    raw = _open(url, timeout, 1_000_000, merged, allowed_hosts, data=body)
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise SafeURLViolation(f"non-JSON response from {url!r}") from exc

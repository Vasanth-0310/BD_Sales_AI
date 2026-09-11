"""
curl_cffi_adapter.py — HTTP Fetcher with Real Chrome TLS Fingerprinting
=======================================================================

Uses curl_cffi to make HTTP requests with a Chrome-like TLS handshake
(JA3/JA4 fingerprint).

Used for publicly-accessible pages on Cloudflare-protected sites
(e.g., Upwork job postings) where no login session is needed.
"""

import ipaddress
import socket
from datetime import datetime
from urllib.parse import urlparse

from src.common.logger import get_logger

logger = get_logger(__name__)


def _host_has_public_resolution(host: str, port_hint: int | None) -> bool:
    """True when the host resolves to at least one public, non-reserved IP.

    DNS-named hosts resolve via getaddrinfo; every returned address must NOT
    be private/loopback/link-local/multicast/reserved for the host to pass.
    """
    try:
        infos = socket.getaddrinfo(host, port_hint, proto=socket.IPPROTO_TCP)
    except OSError:
        return False
    if not infos:
        return False
    for _family, _type, _proto, _canonname, sockaddr in infos:
        try:
            a = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if (
            a.is_private or a.is_loopback or a.is_link_local
            or a.is_multicast or a.is_reserved
        ):
            return False
        # IPv4-mapped IPv6 (::ffff:10.0.0.1 style) — inspect the mapped pair.
        if a.version == 6 and a.ipv4_mapped is not None:
            m = a.ipv4_mapped
            if m.is_private or m.is_loopback or m.is_link_local:
                return False
    return True


def _assert_public_host_url(url: str) -> None:
    """SSRF guard: reject URLs whose host resolves to a private/loopback/
    link-local address, or whose scheme is not http(s).

    Called BEFORE any network I/O on the entry URL, and again by the caller
    on the post-redirect final URL — an attacker-controlled open redirect
    must not be able to turn the request onto the internal network.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Blocked non-http URL scheme: '{parsed.scheme}'")

    host = (parsed.hostname or "").strip("[]")
    if not host:
        raise ValueError(f"Blocked URL with no resolvable host: '{url}'")

    port_hint = parsed.port or (443 if parsed.scheme == "https" else 80)

    # 1. Literal-IP URLs are evaluated locally (no DNS involved).
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        blocked = (
            literal.is_private or literal.is_loopback or literal.is_link_local
            or literal.is_multicast or literal.is_reserved
            or (literal.version == 6 and literal.ipv4_mapped is not None
                and literal.ipv4_mapped.is_private)
        )
        if blocked:
            raise ValueError(
                f"Blocked non-public scrape target host: '{host}'"
            )
        return

    # 2. DNS-named host: require at least one public resolution.
    if not _host_has_public_resolution(host, port_hint):
        raise ValueError(
            f"Blocked non-public or unresolvable scrape target host: '{host}'"
        )


def _scoped_cookies(cookies: list[dict] | None, url: str) -> dict[str, str]:
    """
    Build a cookie jar containing only cookies whose domain scope matches
    the target host. Prevents cookies captured for third-party domains
    (e.g., SSO providers) from leaking to unrelated hosts.
    Expired cookies are dropped as well.
    """
    host = urlparse(url).netloc.split(":", 1)[0].lower()
    now = datetime.now().timestamp()

    jar: dict[str, str] = {}
    for c in cookies or []:
        name = c.get("name", "")
        value = c.get("value", "")
        if not name:
            continue

        cookie_domain = str(c.get("domain") or "").lower().lstrip(".")
        # Cookies with no recorded domain can't be scoped — keep them to
        # preserve behaviour for sessions captured without domain metadata.
        if cookie_domain and cookie_domain != host and not host.endswith("." + cookie_domain):
            continue

        expires = c.get("expires")
        if isinstance(expires, (int, float)) and expires > 0 and expires < now:
            continue

        jar[name] = value
    return jar


class CurlCFFIFetcher:
    """
    Lightweight HTTP fetcher using curl_cffi.
    Uses curl_cffi's Chrome impersonation mode for ordinary HTTP fetches.
    """

    async def fetch(self, url: str, cookies: list[dict] | None = None) -> tuple[str, int, str]:
        """
        Fetch a URL and return (html_content, status_code, final_url).

        Args:
            url:     The URL to fetch.
            cookies: Optional list of cookie dicts (from saved session).

        Returns:
            Tuple of (html_text, http_status_code, final_url_after_redirects).
        """
        try:
            from curl_cffi.requests import AsyncSession
        except ImportError:
            raise RuntimeError(
                "curl_cffi is not installed. Run: pip install curl_cffi"
            )

        # SSRF guard: the entry URL must target a public host.
        _assert_public_host_url(url)

        # Headers aligned with the impersonated browser version below —
        # Cloudflare cross-checks sec-ch-ua/UA against the TLS JA3/JA4
        # fingerprint, so a version mismatch is itself a bot signal.
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        headers = {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/avif,image/webp,image/apng,*/*;"
                "q=0.8,application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Cache-Control": "max-age=0",
            "Priority": "u=0, i",
            "Referer": origin + "/",
            "Sec-Ch-Ua": '"Chromium";v="136", "Google Chrome";v="136", "Not.A/Brand";v="99"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
        }

        logger.info(f"curl_cffi fetching: {url}")

        async with AsyncSession(impersonate="chrome136") as session:
            response = await session.get(
                url,
                headers=headers,
                cookies=_scoped_cookies(cookies, url) or None,
                timeout=30,
                allow_redirects=True,
            )

        # SSRF guard 2: an open redirect must not ferry the request onto the
        # internal network — re-validate the FINAL URL after all redirects.
        final_url = str(response.url)
        if final_url and final_url != url:
            try:
                _assert_public_host_url(final_url)
            except ValueError:
                logger.warning(
                    f"curl_cffi: blocked redirect to non-public host — "
                    f"final URL was {final_url}"
                )
                raise

        logger.info(
            f"curl_cffi fetch complete. Status: {response.status_code}, "
            f"Content: {len(response.text)} chars, Final URL: {response.url}"
        )
        return response.text, response.status_code, final_url

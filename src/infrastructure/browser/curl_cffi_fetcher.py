"""
curl_cffi_adapter.py — HTTP Fetcher with Real Chrome TLS Fingerprinting
=======================================================================

Uses curl_cffi to make HTTP requests with a Chrome-like TLS handshake
(JA3/JA4 fingerprint).

Used for publicly-accessible pages on Cloudflare-protected sites
(e.g., Upwork job postings) where no login session is needed.
"""


from datetime import datetime
from urllib.parse import urlparse

from src.common.logger import get_logger

logger = get_logger(__name__)


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

        logger.info(
            f"curl_cffi fetch complete. Status: {response.status_code}, "
            f"Content: {len(response.text)} chars, Final URL: {response.url}"
        )
        return response.text, response.status_code, str(response.url)

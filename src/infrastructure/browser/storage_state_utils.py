"""
Storage-state sanitisation for Playwright-based engines (CloakBrowser/Patchright).

Playwright's new_context(storage_state=...) strictly validates cookie.sameSite
against {Strict, Lax, None}, but cookies captured from other sources use
different vocabularies:

    - Chrome/CDP (nodriver capture):  "unspecified", "no_restriction", "lax"...
    - curl_cffi / http.cookiejar:     lowercase or missing entirely

Instead of crashing the whole context creation over one cookie, we normalise
known values and drop the field for unknown ones (Playwright defaults to Lax).
"""
from typing import Any

# Maps the various sameSite vocabularies to Playwright's accepted values.
_SAME_SITE_MAP = {
    "strict": "Strict",
    "lax": "Lax",
    "none": "None",
    "no_restriction": "None",   # Chrome CDP name for SameSite=None
    "unspecified": None,        # let Playwright default (Lax)
}

_VALID_SAME_SITE = {"Strict", "Lax", "None"}


def sanitize_storage_state(storage_state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a storage state safe to pass to Playwright's new_context().

    Only cookies/origins are carried over — extra keys saved alongside the
    session (e.g. nodriver_profile_dir) would be rejected by Playwright.
    """
    if not storage_state:
        return storage_state

    sanitized: dict[str, Any] = {
        "cookies": [
            sanitize_cookie(c)
            for c in (storage_state.get("cookies") or [])
            if isinstance(c, dict) and c.get("name") and c.get("value") is not None
        ],
        "origins": storage_state.get("origins") or [],
    }
    return sanitized


def sanitize_cookie(cookie: dict[str, Any]) -> dict[str, Any]:
    """Normalise a single cookie dict so Playwright accepts it."""
    cookie = dict(cookie)

    same_site = cookie.get("sameSite") or cookie.get("same_site")
    if same_site is not None:
        same_site = _SAME_SITE_MAP.get(str(same_site).strip().lower(), same_site)
        if same_site not in _VALID_SAME_SITE:
            same_site = None

    cookie.pop("same_site", None)
    if same_site is None:
        # Unknown/unspecified — omit so Playwright applies its default.
        cookie.pop("sameSite", None)
    else:
        cookie["sameSite"] = same_site

    return cookie

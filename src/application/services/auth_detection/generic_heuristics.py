from urllib.parse import urlparse
import re
from src.domain.enums.scrape_status import AuthStatus
from src.common.logger import get_logger

logger = get_logger(__name__)

# HTTP status codes that indicate authentication is required
_AUTH_STATUS_CODES = {401, 403}

# URL substrings that strongly suggest a login redirect
_AUTH_URL_KEYWORDS = ["login", "signin", "sign-in", "auth", "authenticate", "account/login"]

# Page text phrases indicating an auth wall
_AUTH_TEXT_SIGNALS = [
    "sign in to continue",
    "sign in to view",
    "log in to continue",
    "login to view",
    "authentication required",
    "please log in",
    "create an account",
    "register to view",
    "join to see",
]

# Minimum word count threshold. Pages with fewer words are likely auth-gated.
_MIN_WORD_COUNT = 80


class GenericHeuristics:
    """
    Layer 2 of the Authentication Detection Engine.
    Used as a fallback for unknown or unrecognized job portals.
    Always returns a definitive AuthStatus (never None).
    """

    def check(
        self,
        current_url: str,
        status_code: int,
        page_text: str,
    ) -> AuthStatus:
        """
        Run generic heuristics to determine auth requirement.

        Returns:
            AuthStatus.AUTH_REQUIRED or AuthStatus.PUBLIC.
        """
        page_text_lower = page_text.lower()

        # Check 1: HTTP status code.
        # F4: a 403 is frequently a WAF/bot-block (not an auth wall) — treating
        # it as AUTH_REQUIRED on its own used to cause the caller to DELETE the
        # user's valid session. Require corroboration for 403; 401 alone is
        # still definitive since it specifically means "unauthenticated".
        if status_code == 401:
            logger.debug("Generic heuristic: AUTH_REQUIRED via HTTP 401")
            return AuthStatus.AUTH_REQUIRED

        url_hit, text_hit = self._corroboration_signals(current_url, page_text_lower)

        if status_code == 403:
            if url_hit or text_hit:
                logger.debug(
                    f"Generic heuristic: AUTH_REQUIRED via HTTP 403 + corroboration "
                    f"(url_signal={url_hit}, text_signal={text_hit})"
                )
                return AuthStatus.AUTH_REQUIRED
            logger.debug(
                "Generic heuristic: HTTP 403 without login signals — likely a bot "
                "block, NOT auth. Treating as PUBLIC."
            )
            return AuthStatus.PUBLIC

        # Check 2: URL pattern (only check path and query, not domain name)
        word_count = len(page_text.split())
        has_text_signal = text_hit
        if not has_text_signal:
            parsed_url = urlparse(current_url)
            path_query_lower = (parsed_url.path + "?" + parsed_url.query).lower()
            for keyword in _AUTH_URL_KEYWORDS:
                if keyword == "auth":
                    # Ensure we don't match words like 'author'
                    if re.search(r'\bauth\b', path_query_lower):
                        logger.debug(f"Generic heuristic: AUTH_REQUIRED via URL keyword '{keyword}'")
                        return AuthStatus.AUTH_REQUIRED
                elif keyword in path_query_lower:
                    logger.debug(f"Generic heuristic: AUTH_REQUIRED via URL keyword '{keyword}'")
                    return AuthStatus.AUTH_REQUIRED

        # Check 3: Suspiciously low word count.
        # A sparse page alone must NOT trigger AUTH_REQUIRED — legitimately short
        # public posts exist, and a false positive here causes the caller to
        # invalidate valid saved sessions. Require corroboration from an
        # auth-text signal before deciding; otherwise treat as PUBLIC (the
        # pipeline's readability checks will reject truly empty pages later).
        if word_count < _MIN_WORD_COUNT:
            if has_text_signal:
                logger.debug(
                    f"Generic heuristic: AUTH_REQUIRED via low word count "
                    f"({word_count} < {_MIN_WORD_COUNT}) + auth text signal"
                )
                return AuthStatus.AUTH_REQUIRED
            logger.debug(
                f"Generic heuristic: sparse page ({word_count} words) without auth "
                f"signals — treating as PUBLIC."
            )
            return AuthStatus.PUBLIC

        # Check 4: Page text signals (only apply if the page isn't massive)
        # Auth walls generally don't have 400+ words. If a page is huge, a "please log in"
        # string is likely just a footer/sidebar button (e.g. "log in to save this job").
        if word_count < 400 and has_text_signal:
            for signal in _AUTH_TEXT_SIGNALS:
                if signal in page_text_lower:
                    logger.debug(f"Generic heuristic: AUTH_REQUIRED via text signal '{signal}'")
                    return AuthStatus.AUTH_REQUIRED

        logger.debug("Generic heuristic: Page is PUBLIC.")
        return AuthStatus.PUBLIC

    def _corroboration_signals(self, current_url: str, page_text_lower: str) -> tuple[bool, bool]:
        """Return (url_login_signal, text_login_signal) used to corroborate 403s."""
        parsed_url = urlparse(current_url)
        path_query_lower = (parsed_url.path + "?" + parsed_url.query).lower()
        url_hit = False
        for keyword in _AUTH_URL_KEYWORDS:
            if keyword == "auth":
                if re.search(r'\bauth\b', path_query_lower):
                    url_hit = True
                    break
            elif keyword in path_query_lower:
                url_hit = True
                break
        text_hit = any(signal in page_text_lower for signal in _AUTH_TEXT_SIGNALS)
        return url_hit, text_hit

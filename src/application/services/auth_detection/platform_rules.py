from typing import Optional
from src.domain.enums.scrape_status import AuthStatus
from src.common.logger import get_logger

logger = get_logger(__name__)

# Known login URL patterns per domain keyword
_PLATFORM_LOGIN_URLS: dict[str, str] = {
    "linkedin.com": "https://www.linkedin.com/login",
    "upwork.com": "https://www.upwork.com/ab/account-security/login",
    "indeed.com": "https://secure.indeed.com/auth",
}

# LinkedIn auth detection signals.
#
# Public LinkedIn job pages often include header/nav prompts like "Join now" and
# "Sign in" even when the actual job content is visible. Those phrases are weak
# signals and must not be treated as an auth wall by themselves.
_LINKEDIN_AUTH_URL_PATTERNS = ["/login", "/authwall", "/checkpoint"]
_LINKEDIN_STRONG_AUTH_DOM_SIGNALS = [
    "authwall",
    "sign in to continue",
    "sign in to view",
    "log in to continue",
    "join linkedin to view",
    "join now to see",
]
_LINKEDIN_PUBLIC_JOB_SIGNALS = [
    "about the job",
    "seniority level",
    "employment type",
    "job function",
    "industries",
    "applicants",
]

# Upwork auth detection signals
_UPWORK_AUTH_URL_PATTERNS = ["/ab/account-security/", "/login"]

# Indeed auth detection signals
_INDEED_AUTH_DOM_SIGNALS = ["sign in to view", "create a free account"]


class PlatformRules:
    """
    Layer 1 of the Authentication Detection Engine.
    Contains high-confidence, platform-specific detection rules for known job portals.
    Returns None if the domain doesn't match any known platform (falls through to Layer 2).
    """

    def check(
        self,
        domain: str,
        current_url: str,
        status_code: int,
        page_text_lower: str,
    ) -> Optional[AuthStatus]:
        """
        Run platform-specific rules.

        Returns:
            AuthStatus if a rule matches, None if the domain is unknown.
        """
        if "linkedin.com" in domain:
            return self._check_linkedin(current_url, page_text_lower)

        if "upwork.com" in domain:
            return self._check_upwork(current_url, status_code, page_text_lower)

        if "indeed.com" in domain:
            return self._check_indeed(page_text_lower)

        return None  # Unknown platform → fall through to generic heuristics

    def _check_linkedin(self, current_url: str, page_text_lower: str) -> AuthStatus:
        current_url_lower = current_url.lower()

        for pattern in _LINKEDIN_AUTH_URL_PATTERNS:
            if pattern in current_url_lower:
                logger.debug(f"LinkedIn auth detected via URL pattern: '{pattern}'")
                return AuthStatus.AUTH_REQUIRED

        if any(signal in page_text_lower for signal in _LINKEDIN_PUBLIC_JOB_SIGNALS):
            logger.debug("LinkedIn public job content detected.")
            return AuthStatus.PUBLIC

        for signal in _LINKEDIN_STRONG_AUTH_DOM_SIGNALS:
            if signal in page_text_lower:
                logger.debug(f"LinkedIn auth detected via DOM signal: '{signal}'")
                return AuthStatus.AUTH_REQUIRED

        return AuthStatus.PUBLIC

    def _check_upwork(self, current_url: str, status_code: int, page_text_lower: str) -> AuthStatus:
        # If we're on a Cloudflare challenge page, it's not an auth wall — just bot detection
        if "just a moment" in page_text_lower or "verifying" in page_text_lower:
            logger.debug("Upwork: Cloudflare challenge page detected — treating as PUBLIC (not auth wall).")
            return AuthStatus.PUBLIC

        # F4: 403 is usually Upwork's WAF/bot-block, not an auth wall — a bare
        # 403 must NOT cause the caller to delete the user's saved session.
        # Require a login URL pattern to corroborate 403; 401 alone stays
        # definitive (it specifically means "unauthenticated").
        if status_code == 401:
            logger.debug("Upwork auth detected via HTTP 401")
            return AuthStatus.AUTH_REQUIRED
        if status_code == 403:
            for pattern in _UPWORK_AUTH_URL_PATTERNS:
                if pattern in current_url:
                    logger.debug("Upwork auth detected via HTTP 403 + login URL pattern")
                    return AuthStatus.AUTH_REQUIRED
            logger.debug("Upwork: HTTP 403 without login URL pattern — bot block, treating as PUBLIC.")
            return AuthStatus.PUBLIC

        for pattern in _UPWORK_AUTH_URL_PATTERNS:
            if pattern in current_url:
                logger.debug(f"Upwork auth detected via URL pattern: '{pattern}'")
                return AuthStatus.AUTH_REQUIRED

        return AuthStatus.PUBLIC

    def _check_indeed(self, page_text_lower: str) -> AuthStatus:
        for signal in _INDEED_AUTH_DOM_SIGNALS:
            if signal in page_text_lower:
                logger.debug(f"Indeed auth detected via DOM signal: '{signal}'")
                return AuthStatus.AUTH_REQUIRED

        return AuthStatus.PUBLIC

    @staticmethod
    def get_login_url(domain: str) -> Optional[str]:
        """Return the known login URL for a platform domain, if available."""
        for key, url in _PLATFORM_LOGIN_URLS.items():
            if key in domain:
                return url
        return None

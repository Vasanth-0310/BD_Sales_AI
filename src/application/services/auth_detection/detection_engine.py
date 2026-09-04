from src.domain.enums.scrape_status import AuthStatus
from src.application.services.auth_detection.platform_rules import PlatformRules
from src.application.services.auth_detection.generic_heuristics import GenericHeuristics
from src.common.logger import get_logger

logger = get_logger(__name__)


class AuthDetectionEngine:
    """
    Orchestrates the two-layer authentication detection system.

    Layer 1: Platform-specific rules (high confidence, fast).
    Layer 2: Generic heuristics (universal fallback).

    This engine has zero I/O and no external dependencies — it only
    processes data passed in from the browser.
    """

    def __init__(self) -> None:
        self._platform_rules = PlatformRules()
        self._generic_heuristics = GenericHeuristics()

    def detect(
        self,
        domain: str,
        current_url: str,
        status_code: int,
        page_text: str,
    ) -> AuthStatus:
        """
        Determine whether the loaded page requires authentication.

        Args:
            domain: The registered domain (e.g., "linkedin.com").
            current_url: The current URL after navigation (may differ from original).
            status_code: HTTP status code of the page response.
            page_text: The cleaned plain text of the page.

        Returns:
            AuthStatus enum value.
        """
        logger.info(f"Running auth detection for domain: '{domain}' (HTTP {status_code})")

        page_text_lower = page_text.lower()

        # Layer 1: Try platform-specific rules first
        result = self._platform_rules.check(domain, current_url, status_code, page_text_lower)

        if result is not None:
            logger.info(f"Layer 1 (platform rule) result for '{domain}': {result.value}")
            return result

        # Layer 2: Fall back to generic heuristics
        result = self._generic_heuristics.check(current_url, status_code, page_text)
        logger.info(f"Layer 2 (generic heuristic) result for '{domain}': {result.value}")
        return result

    def get_login_url(self, domain: str) -> str | None:
        """Return the known login URL for the given domain, if any."""
        return self._platform_rules.get_login_url(domain)

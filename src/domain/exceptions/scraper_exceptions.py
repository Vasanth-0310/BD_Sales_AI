class ScraperException(Exception):
    """Base exception for all job scraper errors."""
    pass


class InvalidURLException(ScraperException):
    """Raised when the provided URL is malformed or cannot be parsed."""
    def __init__(self, url: str):
        super().__init__(f"Invalid or malformed URL: '{url}'")
        self.url = url


class AuthenticationRequiredException(ScraperException):
    """
    Raised when a page requires login and no valid session is available.
    Triggers the in-app headful browser login flow.
    """
    def __init__(self, domain: str, login_url: str | None = None):
        super().__init__(f"Authentication required for domain: '{domain}'")
        self.domain = domain
        self.login_url = login_url


class SessionExpiredException(ScraperException):
    """Raised when a stored session is found but is no longer valid on the target site."""
    def __init__(self, domain: str, user_id: str):
        super().__init__(f"Session expired for user '{user_id}' on '{domain}'")
        self.domain = domain
        self.user_id = user_id


class ExtractionFailedException(ScraperException):
    """Raised when the LLM fails to extract structured data from the page content."""
    def __init__(self, reason: str):
        super().__init__(f"AI extraction failed: {reason}")
        self.reason = reason


class RedirectLimitExceededException(ScraperException):
    """Raised if more than one external redirect is detected, preventing redirect loops."""
    def __init__(self, url: str):
        super().__init__(f"External redirect limit (1) exceeded at URL: '{url}'")
        self.url = url


class BrowserLaunchFailedException(ScraperException):
    """Raised when both CloakBrowser and the Patchright fallback fail to launch."""
    def __init__(self, reason: str):
        super().__init__(f"All browser backends failed to launch: {reason}")
        self.reason = reason

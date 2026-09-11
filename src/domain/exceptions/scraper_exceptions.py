class ScraperException(Exception):
    """Base exception for all job scraper errors."""
    pass


class ExtractionFailedException(ScraperException):
    """Raised when the LLM fails to extract structured data from the page content."""
    def __init__(self, reason: str):
        super().__init__(f"AI extraction failed: {reason}")
        self.reason = reason


class BrowserLaunchFailedException(ScraperException):
    """Raised when both CloakBrowser and the Patchright fallback fail to launch."""
    def __init__(self, reason: str):
        super().__init__(f"All browser backends failed to launch: {reason}")
        self.reason = reason

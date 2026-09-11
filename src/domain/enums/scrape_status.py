from enum import Enum


class AuthStatus(str, Enum):
    """Result of the authentication detection engine."""
    PUBLIC = "PUBLIC"
    AUTH_REQUIRED = "AUTH_REQUIRED"


class ScrapeStatus(str, Enum):
    """Lifecycle state of a scrape request."""
    SUCCESS = "SUCCESS"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    FAILED = "FAILED"


class SessionStatus(str, Enum):
    """Lifecycle state of a stored user session."""
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"

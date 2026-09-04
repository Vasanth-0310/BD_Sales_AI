from enum import Enum


class AuthStatus(str, Enum):
    """Result of the authentication detection engine."""
    PUBLIC = "PUBLIC"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    NO_SESSION = "NO_SESSION"


class ScrapeStatus(str, Enum):
    """Lifecycle state of a scrape request."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    SUCCESS = "SUCCESS"
    AUTH_REQUIRED = "AUTH_REQUIRED"
    FAILED = "FAILED"
    REDIRECT_FOLLOWED = "REDIRECT_FOLLOWED"


class SessionStatus(str, Enum):
    """Lifecycle state of a stored user session."""
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    INVALID = "INVALID"

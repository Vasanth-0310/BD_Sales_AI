from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from src.domain.enums.scrape_status import SessionStatus


@dataclass
class UserSession:
    """
    Domain entity representing an authenticated browser session
    for a specific user on a specific website/domain.
    Persisted in MongoDB via the ISessionStore port.
    """
    user_id: str
    website: str                            # e.g., "linkedin.com"
    storage_state: dict[str, Any]           # Full CloakBrowser/Patchright storage state
    status: SessionStatus = SessionStatus.ACTIVE

    id: Optional[str] = None
    cookies: list[dict] = field(default_factory=list)
    local_storage: dict[str, Any] = field(default_factory=dict)
    session_storage: dict[str, Any] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def mark_used(self) -> None:
        self.last_used = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def mark_expired(self) -> None:
        self.status = SessionStatus.EXPIRED
        self.updated_at = datetime.utcnow()

    def mark_invalid(self) -> None:
        self.status = SessionStatus.INVALID
        self.updated_at = datetime.utcnow()

    def update_state(self, storage_state: dict[str, Any]) -> None:
        # A Playwright capture (pool refresh) has no profile path; without
        # preserving it, one pool-based refresh would silently erase the
        # persistent-profile key and nodriver would never be used again.
        for key in ("nodriver_profile_dir", "profile_dir"):
            if key not in storage_state and key in self.storage_state:
                storage_state[key] = self.storage_state[key]
        self.storage_state = storage_state
        self.cookies = storage_state.get("cookies", [])
        self.updated_at = datetime.utcnow()
        self.status = SessionStatus.ACTIVE

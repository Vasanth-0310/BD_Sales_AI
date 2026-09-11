from abc import ABC, abstractmethod
from typing import Optional
from src.domain.entities.user_session import UserSession


class ISessionStore(ABC):
    """
    Port (Interface) for persisting and retrieving user browser sessions.
    Implemented by MongoDBSessionRepository in the infrastructure layer.
    """

    @abstractmethod
    async def get_session(self, user_id: str, domain: str) -> Optional[UserSession]:
        """
        Retrieve a stored session for the given user and domain.
        Returns None if no session exists.
        """
        ...

    @abstractmethod
    async def save_session(self, session: UserSession) -> None:
        """Persist a new or updated session to the store (upsert by user_id + domain)."""
        ...

    @abstractmethod
    async def delete_session(self, user_id: str, domain: str) -> None:
        """Remove a session from the store."""
        ...

    @abstractmethod
    async def get_all_active_sessions(self) -> list[UserSession]:
        """Return all sessions with ACTIVE status. Used by the refresh scheduler."""
        ...
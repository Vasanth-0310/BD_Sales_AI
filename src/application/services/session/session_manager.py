from typing import Any, Optional
from src.domain.entities.user_session import UserSession
from src.domain.interfaces.browser.i_browser import IBrowser
from src.domain.interfaces.session_store.i_session_store import ISessionStore
from src.common.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Application service managing the full lifecycle of user browser sessions.
    Delegates all persistence to ISessionStore and all browser operations to IBrowser.
    """

    def __init__(self, session_store: ISessionStore) -> None:
        self._store = session_store

    async def get_valid_session(self, user_id: str, domain: str) -> Optional[UserSession]:
        """
        Retrieve an ACTIVE session for the given user+domain.
        Returns None if no session exists or the session is expired/invalid.
        """
        from src.domain.enums.scrape_status import SessionStatus
        session = await self._store.get_session(user_id, domain)
        if session and session.status == SessionStatus.ACTIVE:
            logger.info(f"Valid session found for user '{user_id}' on '{domain}'.")
            return session
        
        # Fallback to default_user session if user-specific session does not exist.
        # WARNING: this shares one account's authenticated cookies across all
        # user_ids — gated behind settings.session_fallback_enabled so production
        # deployments can disable the cross-user credential leak.
        if user_id != "default_user":
            from src.common.config import settings as _settings
            if _settings.session_fallback_enabled:
                logger.warning(
                    f"No session for user '{user_id}' on '{domain}' — "
                    f"falling back to SHARED 'default_user' credentials."
                )
                default_session = await self._store.get_session("default_user", domain)
                if default_session and default_session.status == SessionStatus.ACTIVE:
                    logger.info(f"Valid fallback session found for user 'default_user' on '{domain}'.")
                    return default_session

        logger.info(f"No valid session for user '{user_id}' on '{domain}'.")
        return None

    async def capture_from_browser(
        self, browser: IBrowser, user_id: str, domain: str
    ) -> UserSession:
        """
        Capture the current browser session state after a manual login
        and build a UserSession entity.
        """
        logger.info(f"Capturing session from browser for user '{user_id}' on '{domain}'...")
        storage_state: dict[str, Any] = await browser.capture_session()

        session = UserSession(
            user_id=user_id,
            website=domain,
            storage_state=storage_state,
            cookies=storage_state.get("cookies", []),
        )
        logger.info("Session captured successfully.")
        return session

    async def persist(self, session: UserSession) -> None:
        """Save or update a session in the store."""
        await self._store.save_session(session)
        logger.info(f"Session persisted for user '{session.user_id}' on '{session.website}'.")

    async def capture_from_storage(
        self, user_id: str, domain: str, storage_state: dict
    ) -> UserSession:
        """
        Build a UserSession entity directly from a storage_state dict
        (already captured from the browser by the caller).
        """
        logger.info(f"Building session from storage state for user '{user_id}' on '{domain}'...")
        session = UserSession(
            user_id=user_id,
            website=domain,
            storage_state=storage_state,
            cookies=storage_state.get("cookies", []),
        )
        logger.info("Session built successfully.")
        return session

    async def invalidate(self, user_id: str, domain: str) -> None:
        """Delete a session from the store."""
        await self._store.delete_session(user_id, domain)
        logger.info(f"Session invalidated for user '{user_id}' on '{domain}'.")

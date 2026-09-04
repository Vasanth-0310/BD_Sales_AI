from src.domain.interfaces.browser.i_browser import IBrowser
from src.domain.interfaces.session_store.i_session_store import ISessionStore
from src.application.services.session.session_manager import SessionManager
from src.infrastructure.browser.browser_factory import BrowserFactory
from src.common.logger import get_logger

logger = get_logger(__name__)


class ManageSession:
    """
    Use case: Manages the full session lifecycle including the in-app login trigger.

    The in-app trigger flow:
        1. A headful browser window is launched (visible to the user).
        2. The browser navigates to the platform's login page.
        3. The application waits for the user to log in manually.
        4. Once the user completes login (detected by URL change), the session is captured.
        5. The session is saved to MongoDB.
    """

    def __init__(self, session_store: ISessionStore) -> None:
        self._session_manager = SessionManager(session_store)

    async def trigger_login_flow(
        self, user_id: str, domain: str, login_url: str
    ) -> dict:
        """
        Launch a headful browser for the user to manually authenticate.
        Waits until the login page URL changes (indicating successful login),
        then captures and saves the session.

        Args:
            user_id: The user initiating the login.
            domain: The domain to authenticate against (e.g., "linkedin.com").
            login_url: The login URL to navigate to.

        Returns:
            dict with success status and message.
        """
        logger.info(f"Triggering in-app login flow for user '{user_id}' on '{domain}'...")

        browser = None
        try:
            # Launch headful (visible) browser — user will interact with it.
            # Engine fallback chain applies (CloakBrowser -> Patchright -> nodriver).
            browser = await BrowserFactory.launch_browser(headless=False)
            await browser.navigate(login_url, wait_for_job_details=False)

            logger.info(
                f"Headful browser opened at '{login_url}'. "
                f"Waiting for user '{user_id}' to complete login..."
            )

            # Wait for navigation away from login page (user logged in)
            await self._wait_for_login_completion(browser, login_url)

            # Capture the authenticated session
            session = await self._session_manager.capture_from_browser(browser, user_id, domain)
            await self._session_manager.persist(session)

            logger.info(f"Login flow complete. Session saved for user '{user_id}' on '{domain}'.")
            return {"success": True, "domain": domain, "message": "Session captured and saved."}

        except Exception as e:
            logger.error(f"Login flow failed for user '{user_id}' on '{domain}': {e}")
            return {"success": False, "domain": domain, "message": str(e)}
        finally:
            if browser is not None:
                try:
                    await browser.close()
                except Exception as close_err:
                    logger.warning(f"Error closing login browser: {close_err}")

    async def _wait_for_login_completion(
        self, browser: IBrowser, login_url: str, poll_interval: float = 2.0, timeout: float = 300.0
    ) -> None:
        """
        Poll the browser URL every 2 seconds until it navigates away from the login page
        or a 5-minute timeout is reached.
        """
        import asyncio
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            current_url = await browser.get_current_url()
            # Consider login complete when the URL no longer contains login-related keywords
            if not any(kw in current_url for kw in ["login", "signin", "auth", "checkpoint"]):
                logger.info(f"Login detected. Current URL: {current_url}")
                return
            elapsed += poll_interval

        raise TimeoutError(f"Login flow timed out after {timeout}s. User did not complete login.")

    async def get_session(self, user_id: str, domain: str):
        return await self._session_manager.get_valid_session(user_id, domain)

    async def revoke(self, user_id: str, domain: str) -> None:
        await self._session_manager.invalidate(user_id, domain)

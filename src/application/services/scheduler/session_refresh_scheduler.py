import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.domain.entities.user_session import UserSession
from src.domain.interfaces.session_store.i_session_store import ISessionStore
from src.domain.enums.scrape_status import AuthStatus
from src.application.services.auth_detection.detection_engine import AuthDetectionEngine
from src.infrastructure.browser.browser_factory import BrowserFactory
from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger(__name__)


def _is_browser_verification_page(html: str) -> bool:
    sample = html[:4000].lower()
    return (
        "just a moment" in sample
        or "additional verification required" in sample
        or "cf-browser-verification" in sample
        or "challenge - upwork" in sample
        or "troubleshooting cloudflare errors" in sample
        or "verification successful. waiting for" in sample
        or "your ray id for this request" in sample
        or ("cloudflare" in sample and ("verifying" in sample or "ray id" in sample))
    )


class SessionRefreshScheduler:
    """
    Background APScheduler job that periodically validates all active sessions.
    Runs every N hours (configured via settings.session_refresh_interval_hours).
    """

    def __init__(self, session_store: ISessionStore) -> None:
        self._store = session_store
        self._auth_engine = AuthDetectionEngine()
        self._scheduler = AsyncIOScheduler()

    def start(self) -> None:
        """Register and start the session refresh cron job."""
        interval_hours = settings.session_refresh_interval_hours
        self._scheduler.add_job(
            self.refresh_all_active_sessions,
            trigger="interval",
            hours=interval_hours,
            id="session_refresh_job",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            misfire_grace_time=300,
        )
        self._scheduler.start()
        logger.info(f"Session refresh scheduler started (every {interval_hours}h).")

    def stop(self) -> None:
        """Cleanly shut down the scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Session refresh scheduler stopped.")

    async def refresh_all_active_sessions(self) -> None:
        """Fetch all active sessions and attempt to validate each one."""
        logger.info("Session refresh job started...")
        sessions = await self._store.get_all_active_sessions()
        logger.info(f"Found {len(sessions)} active session(s) to refresh.")

        for session in sessions:
            try:
                # Hard per-session deadline: a hung CDP call or a stuck website
                # would otherwise block this coroutine forever, and with
                # max_instances=1 APScheduler would then skip EVERY future
                # refresh cycle ("maximum number of running instances reached").
                await asyncio.wait_for(
                    self._refresh_single_session(session), timeout=120.0
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Session refresh TIMED OUT (120s) for user '{session.user_id}' "
                    f"on '{session.website}' — skipping."
                )
            except Exception as e:
                logger.error(
                    f"Error refreshing session for user '{session.user_id}' "
                    f"on '{session.website}': {e}"
                )

        logger.info("Session refresh job complete.")

    async def _refresh_single_session(self, session: UserSession) -> None:
        """
        Launch a browser with the stored session, navigate to the domain,
        and verify if the session is still valid.
        """
        # Attribute every log line emitted during this refresh to the session's
        # owner, so background AI logs show "which user did which action".
        from src.common.logger import set_log_context, reset_log_context

        user_token, action_token, section_token = set_log_context(
            user_id=session.user_id,
            action="Session Health Check",
            section="System",
        )
        try:
            await self._refresh_single_session_inner(session)
        finally:
            reset_log_context(user_token, action_token, section_token)

    async def _refresh_single_session_inner(self, session: UserSession) -> None:
        browser = None
        if session.storage_state and session.storage_state.get("nodriver_profile_dir"):
            from pathlib import Path

            raw_profile = session.storage_state.get("nodriver_profile_dir") or session.storage_state.get("profile_dir")
            profile_path = Path(str(raw_profile)).resolve()
            usable_profile = False
            try:
                # exists() is True for a FILE or an unreadable dir — iterdir()
                # would raise NotADirectoryError/PermissionError. Any such
                # failure simply means "no usable profile" (mirrors
                # scrape_job_url._has_usable_nodriver_profile).
                usable_profile = profile_path.is_dir() and any(profile_path.iterdir())
            except OSError as e:
                logger.warning(
                    f"Saved Chrome profile '{profile_path}' unreadable ({e}) — "
                    "falling back to shared pool."
                )
            if usable_profile:
                # MUST go through NodriverPool — Chrome locks the profile dir,
                # and a cold uc.start() would collide with the warm pooled
                # process and corrupt profile state.
                from src.infrastructure.browser.nodriver_pool import NodriverPool

                try:
                    browser = await NodriverPool.acquire(str(profile_path), session.storage_state)
                except Exception as e:
                    logger.warning(f"nodriver refresh launch failed ({e}). Falling back to shared pool.")
            else:
                logger.warning(
                    f"Saved Chrome profile is missing or empty: {profile_path}. "
                    "Falling back to shared pool."
                )
        if browser is None:
            # Cookie-based sessions refresh in a pool TAB (fast, and avoids
            # launching a second browser process which the CloakBrowser free
            # tier denies when the pool browser is already running). Pool tabs
            # are pool-managed: close() only closes the tab, never the browser.
            from src.infrastructure.browser.browser_pool import BrowserPool

            # RACE GUARD (atomic): if_idle=True performs the busy-check AND the
            # slot reservation under the pool's lock — a scrape starting in the
            # same instant can no longer slip between check and acquire, so the
            # refresh can never hijack an in-flight scrape's page state.
            # Returns None when the pool is busy → standalone browser instead.
            try:
                browser = await BrowserPool.acquire(
                    storage_state=session.storage_state, if_idle=True
                )
            except Exception as e:
                logger.warning(f"Pool acquire failed ({e}). Launching a standalone browser.")
                browser = None
            if browser is None:
                logger.info("Shared pool busy or unavailable — using standalone browser for refresh.")
            if browser is None:
                # skip_orphan_kill=True: the session's profile dir may be owned
                # by a live NodriverPool warm Chrome — a cold launch's orphan
                # sweep would taskkill it and crash concurrent user scrapes.
                browser = await BrowserFactory.launch_browser(
                    headless=settings.browser_cloak_headless,
                    storage_state=session.storage_state,
                    skip_orphan_kill=True,
                )
        try:
            url = f"https://{session.website}"
            status_code = await browser.navigate(url, wait_for_job_details=False)
            page_html = await browser.get_page_content()
            current_url = await browser.get_current_url()

            if _is_browser_verification_page(page_html):
                session.mark_used()
                await self._store.save_session(session)
                logger.warning(
                    f"Browser verification page encountered while refreshing "
                    f"'{session.website}'. Session kept unchanged."
                )
                return

            # Use auth detection engine to verify session validity
            from src.infrastructure.html_processing.html_cleaner import HTMLCleaner
            page_text = HTMLCleaner().clean(page_html)

            auth_status = self._auth_engine.detect(
                domain=session.website,
                current_url=current_url,
                status_code=status_code,
                page_text=page_text,
            )

            if auth_status == AuthStatus.PUBLIC:
                # Session is still valid — update the storage state
                updated_state = await browser.capture_session()
                session.update_state(updated_state)
                session.mark_used()
                await self._store.save_session(session)
                logger.info(
                    f"Login session for '{session.website}' verified successfully "
                    f"for user '{session.user_id}'."
                )
            else:
                # Session has expired
                session.mark_expired()
                await self._store.save_session(session)
                logger.warning(
                    f"Login session for '{session.website}' (user '{session.user_id}') "
                    f"has expired. The user must log in again."
                )
        finally:
            if browser is not None:
                try:
                    # TWO different pools mark adapters _pool_managed=True:
                    #   - BrowserPool (CloakBrowser/Patchright): adapter has
                    #     _page/_context → MUST use BrowserPool.release()
                    #     (adapter.close() skips the _active_tabs decrement).
                    #   - NodriverPool (real Chrome profile): adapter has
                    #     _tab/_browser and NO _page/_context — BrowserPool
                    #     .release() would AttributeError on _page (swallowed)
                    #     and never close the tab or re-hide the window,
                    #     leaking a visible Chrome every refresh cycle.
                    # Dispatch by _user_data_dir presence, mirroring
                    # scrape_job_url._close_quietly (the reference impl).
                    if getattr(browser, "_pool_managed", False) and not getattr(
                        browser, "_user_data_dir", None
                    ):
                        from src.infrastructure.browser.browser_pool import BrowserPool
                        await BrowserPool.release(browser)
                    else:
                        # NodriverAdapter pooled path: closes the tab via CDP,
                        # notifies NodriverPool (decrements tab count, re-hides
                        # window on last tab). Standalone: full teardown.
                        await browser.close()
                except Exception as e:
                    logger.warning(f"Error closing refresh browser: {e}")

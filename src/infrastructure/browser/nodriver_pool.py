"""
Persistent Nodriver Pool
------------------------
Keeps one warm nodriver Chrome process alive per persistent profile directory
for the lifetime of the server — the nodriver equivalent of BrowserPool.

Chrome locks a user-data-dir, so at most ONE process per profile can exist.
That has two consequences this pool handles:

    1. Acquire returns a NEW TAB (via CDP target.create_target), not a new
       browser — tab creation costs milliseconds vs. a 1-2s cold start.
    2. Anything that wants to drive a profile (scrapes, session refresh)
       MUST go through this pool — a second uc.start() on the same
       user-data-dir would fight the lock and corrupt profile state.

Session cookies are injected once at warm-up, not per acquire: a warm
browser carries FRESHER cookies after each scrape than the snapshot in
MongoDB, so re-injecting would overwrite live state with stale data.

Lifecycle:
    - Browsers are launched lazily on the first acquire for a profile.
    - adapter.close() (pool-managed) closes only the tab.
    - NodriverPool.shutdown() stops every Chrome process; wired into the
      FastAPI lifespan next to BrowserPool.shutdown().
"""

import asyncio
from pathlib import Path
from typing import Any

from src.common.config import settings
from src.common.logger import get_logger
from src.infrastructure.browser.window_hider import (
    ANTI_THROTTLING_ARGS,
    hide_windows_of_pids,
    hide_windows_when_visible,
    show_windows_of_pids,
)

logger = get_logger(__name__)


class _NodriverPool:
    """Manages one long-lived nodriver Chrome process per profile directory."""

    def __init__(self) -> None:
        self._browsers: dict[str, Any] = {}          # profile_dir -> uc.Browser
        self._profile_locks: dict[str, asyncio.Lock] = {}
        self._registry_lock = asyncio.Lock()
        self._active_tabs: dict[str, int] = {}       # profile_dir -> in-flight tabs
        self._hide_task: Any = None  # ref kept so GC can't cancel the hide task

    async def _lock_for(self, profile_dir: str) -> asyncio.Lock:
        async with self._registry_lock:
            return self._profile_locks.setdefault(profile_dir, asyncio.Lock())

    async def _launch(self, profile_dir: str, storage_state: dict[str, Any] | None) -> Any:
        """Start (or restart) the warm Chrome process for a profile."""
        import nodriver as uc
        from src.infrastructure.browser.nodriver_adapter import NODRIVER_BROWSER_ARGS, NodriverAdapter

        browser_args = list(NODRIVER_BROWSER_ARGS)
        if settings.browser_nodriver_headless:
            browser_args.append("--headless=new")
        else:
            # Headed Chrome kept fully functional while its window is hidden.
            browser_args.extend(ANTI_THROTTLING_ARGS)

        # Orphan cleanup: a crashed / reloaded server never runs shutdown(),
        # leaving a zombie Chrome that holds this profile's SingletonLock and
        # blocks every future launch. Kill any Chrome pinned to THIS profile
        # path before starting fresh (scoped to the full resolved path — the
        # same mechanism _kill_orphan_chrome_for_profile uses).
        await NodriverAdapter()._kill_orphan_chrome_for_profile(profile_dir)

        logger.info(f"NodriverPool: starting warm Chrome for profile {profile_dir}...")
        browser = await uc.start(
            headless=settings.browser_nodriver_headless,
            user_data_dir=profile_dir,
            browser_args=browser_args,
        )
        self._browsers[profile_dir] = browser

        # Hide the idle pool window from the taskbar — the browser stays
        # headed; only the window is invisible (see window_hider docs).
        if settings.browser_hide_windows and not settings.browser_nodriver_headless:
            try:
                pid = browser._process.pid
                self._hide_task = asyncio.create_task(
                    hide_windows_when_visible({pid})
                )
            except Exception as e:
                # A failed hide means a VISIBLE Chrome window that the user
                # will think is "never closing" — make this loud.
                logger.warning(f"NodriverPool: could not hide window for {profile_dir}: {e}")

        # Warm-up cookie injection — see module docstring for why only here.
        if storage_state and storage_state.get("cookies"):
            injector = NodriverAdapter()
            injector._browser = browser
            try:
                await injector._inject_cookies(storage_state["cookies"])
                logger.info(
                    f"NodriverPool: injected {len(storage_state['cookies'])} "
                    f"warm-up cookies for {profile_dir}."
                )
            except Exception as e:
                logger.warning(f"NodriverPool: warm-up cookie injection failed ({e}).")

        logger.info(f"NodriverPool: warm Chrome ready for {profile_dir}.")
        return browser

    async def _get_browser(self, profile_dir: str, storage_state: dict[str, Any] | None) -> Any:
        browser = self._browsers.get(profile_dir)
        if browser is not None:
            return browser
        return await self._launch(profile_dir, storage_state)

    async def acquire(self, profile_dir: str, storage_state: dict[str, Any] | None = None):
        """
        Return a pool-managed NodriverAdapter backed by a fresh tab in the
        warm Chrome process for this profile (launching it if needed).
        Caller must call adapter.close() when done — that closes only the tab.
        """
        from src.infrastructure.browser.nodriver_adapter import NodriverAdapter

        profile_dir = str(Path(profile_dir).resolve())
        lock = await self._lock_for(profile_dir)

        # Serialise launch and tab creation — concurrent create_target +
        # update_targets calls race inside nodriver's target bookkeeping.
        async with lock:
            browser = await self._get_browser(profile_dir, storage_state)
            try:
                tab = await browser.get("about:blank", new_tab=True)
            except Exception as e:
                # The warm process may have died (crash, OOM, user closed the
                # window). Drop it and retry once with a fresh start.
                logger.warning(
                    f"NodriverPool: tab creation failed for {profile_dir} ({e}). "
                    f"Restarting the warm Chrome."
                )
                self._browsers.pop(profile_dir, None)
                browser = await self._launch(profile_dir, storage_state)
                tab = await browser.get("about:blank", new_tab=True)

        adapter = NodriverAdapter()
        adapter._browser = browser
        adapter._tab = tab
        adapter._user_data_dir = profile_dir
        adapter._headless = settings.browser_nodriver_headless
        adapter._pool_managed = True

        # Track the in-flight tab BEFORE showing the window: with concurrent
        # scrapes on the same profile, an early close() must never hide the
        # window while another scrape is still using it.
        self._active_tabs[profile_dir] = self._active_tabs.get(profile_dir, 0) + 1

        # The pool window hides while idle — pop it back up for this scrape
        # so the browser is visible exactly while work is happening.
        if settings.browser_hide_windows and not settings.browser_nodriver_headless:
            try:
                show_windows_of_pids({browser._process.pid})
            except Exception as e:
                logger.debug(f"NodriverPool: could not show window before scrape: {e}")

        logger.info(
            f"NodriverPool: issued warm tab for {Path(profile_dir).name} "
            f"({self._active_tabs[profile_dir]} active)."
        )
        return adapter

    async def discard(self, profile_dir: str) -> None:
        """Throw away a broken warm browser so the next acquire() relaunches.

        Called when a pooled tab dies mid-navigation (ConnectionClosed) —
        the CDP websocket is dead but stop-on-close bookkeeping hasn't run,
        so without this every subsequent acquire reuses the zombie process.
        """
        profile_dir = str(Path(profile_dir).resolve())
        browser = self._browsers.pop(profile_dir, None)
        if browser is None:
            return
        # nodriver's Connection.aclose() only closes the client WebSocket —
        # it does NOT send Browser.close and does not terminate the
        # subprocess (the adapter's own close() needs a PID taskkill backstop
        # for the same reason). Without the backstop every discard leaks a
        # chrome.exe that holds the profile directory lock.
        pid = getattr(getattr(browser, "_process", None), "pid", None)
        try:
            await browser.aclose()
        except Exception:
            try:
                browser.stop()
            except Exception as e:
                logger.warning(f"NodriverPool: discard stop failed for {profile_dir}: {e}")
        if pid:
            import asyncio as _asyncio

            def _taskkill() -> None:
                import subprocess

                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True, timeout=5,
                )

            await _asyncio.to_thread(_taskkill)
        logger.info(f"NodriverPool: discarded broken warm Chrome for {profile_dir}.")

    def notify_tab_closed(self, profile_dir: str | None) -> None:
        """
        Account for a closed pooled tab. The idle window is re-hidden only when
        the LAST active tab for a profile closes — otherwise concurrent scrapes
        would hide each other's window mid-run.
        """
        if not profile_dir:
            return

        remaining = self._active_tabs.get(profile_dir, 1) - 1
        if remaining > 0:
            self._active_tabs[profile_dir] = remaining
            return

        self._active_tabs.pop(profile_dir, None)
        if settings.browser_hide_windows and not settings.browser_nodriver_headless:
            browser = self._browsers.get(profile_dir)
            if browser is not None:
                try:
                    from src.infrastructure.browser.window_hider import hide_windows_of_pids

                    hide_windows_of_pids({browser._process.pid})
                except Exception as e:
                    logger.debug(f"NodriverPool: could not re-hide idle window: {e}")

    async def shutdown(self) -> None:
        """Stop every warm Chrome process. Call on server shutdown."""
        logger.info(f"NodriverPool: shutting down {len(self._browsers)} warm Chrome process(es)...")
        for profile_dir, browser in list(self._browsers.items()):
            try:
                browser.stop()
            except Exception as e:
                logger.warning(f"NodriverPool: error stopping {profile_dir}: {e}")
        self._browsers.clear()
        self._profile_locks.clear()
        logger.info("NodriverPool: all warm Chrome processes stopped.")


# ── Module-level singleton ────────────────────────────────────────────────────
NodriverPool = _NodriverPool()

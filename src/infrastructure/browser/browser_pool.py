"""
Persistent Browser Pool
-----------------------
Keeps a single Chromium process alive for the lifetime of the server.

Instead of launching a full browser on every scrape request (5-8 second cold start),
this pool opens a new *tab* inside the already-running browser (~200ms).

The underlying engine is chosen via settings.browser_engine (default: CloakBrowser,
the source-patched stealth Chromium). Because CloakBrowser and Patchright both
expose a regular Playwright `Browser`, the pool falls back from CloakBrowser to
Patchright transparently if the patched binary is missing or the launch is denied.

Lifecycle:
    - The pool is a singleton, initialised lazily on the first request.
    - Call `BrowserPool.acquire()` to get an adapter wired to a new tab.
    - Call `BrowserPool.release(adapter)` when done — this closes only the tab, not the browser.
    - The pool shuts the underlying browser on server shutdown via `BrowserPool.shutdown()`.

If the underlying browser crashes or disconnects, the pool automatically re-initialises
on the next acquire() call so the server keeps running.
"""
import asyncio
from typing import Optional, Any

from patchright.async_api import async_playwright, Browser, Playwright

from src.common.config import settings
from src.common.logger import get_logger
from src.infrastructure.browser.window_hider import (
    ANTI_THROTTLING_ARGS,
    hide_windows_of_pids,
    hide_windows_when_visible,
    pids_with_cmdline_marker,
    show_windows_of_pids,
)

logger = get_logger(__name__)


class _BrowserPool:
    """Singleton that manages a single long-lived Chromium process."""

    def __init__(self) -> None:
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._engine: str = ""  # name of the engine actually running
        self._lock = asyncio.Lock()
        self._window_pids: set[int] = set()  # pool browser pids for show/hide
        self._active_tabs = 0  # in-flight tabs — window hides only at zero
        self._hide_task: Optional[Any] = None  # keep ref so GC can't kill it

    # Unique, harmless command-line marker: lets window_hider find this
    # browser's PID (Playwright doesn't expose it) to hide the idle window.
    _POOL_MARKER_ARG = "--bd-pool-instance=cloak"

    async def _launch_cloakbrowser(self) -> Browser:
        """Launch via CloakBrowser (patched Chromium). Raises on any failure."""
        import cloakbrowser

        browser = await cloakbrowser.launch_async(
            headless=settings.browser_cloak_headless,
            args=[*ANTI_THROTTLING_ARGS, self._POOL_MARKER_ARG],
        )
        self._engine = "cloakbrowser"
        self._playwright = None  # managed internally by CloakBrowser
        self._hide_pool_window()
        return browser

    async def _launch_patchright(self) -> Browser:
        """Launch via Patchright (stealth-patched Playwright)."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        browser = await self._playwright.chromium.launch(
            headless=settings.browser_cloak_headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                *ANTI_THROTTLING_ARGS,
                self._POOL_MARKER_ARG,
            ],
        )
        self._engine = "patchright"
        self._hide_pool_window()
        return browser

    def _hide_pool_window(self) -> None:
        """Hide the headed pool browser's window (browser stays headed).

        No-op when disabled or headless. Runs as a background task — the
        window appears asynchronously after launch and the hider polls for it.
        """
        if not settings.browser_hide_windows or settings.browser_cloak_headless:
            return
        import asyncio as _asyncio

        async def _hide() -> None:
            pids = await _asyncio.get_event_loop().run_in_executor(
                None, pids_with_cmdline_marker, self._POOL_MARKER_ARG
            )
            self._window_pids = pids
            if pids:
                await hide_windows_when_visible(pids)
            else:
                logger.warning("BrowserPool: no chrome.exe found with the pool marker — window not hidden.")

        try:
            self._hide_task = _asyncio.create_task(_hide())
        except Exception as e:
            logger.warning(f"BrowserPool: window-hiding task failed to start: {e}")

    def _show_window_if_hidden(self) -> None:
        """Pop the idle-hidden pool window back up for an active scrape."""
        if not settings.browser_hide_windows or settings.browser_cloak_headless:
            return
        if self._window_pids:
            try:
                show_windows_of_pids(self._window_pids)
            except Exception as e:
                logger.debug(f"BrowserPool: could not show window for scrape: {e}")

    def hide_window_now(self) -> None:
        """Hide the pool window again — called when a pooled tab is released."""
        if not settings.browser_hide_windows or settings.browser_cloak_headless:
            return
        if self._window_pids:
            try:
                hide_windows_of_pids(self._window_pids)
            except Exception as e:
                logger.debug(f"BrowserPool: could not hide window after scrape: {e}")

    async def _ensure_browser(self) -> Browser:
        """Start the browser if it is not already running."""
        async with self._lock:
            # Re-check inside the lock to avoid double-launch
            if self._browser and self._browser.is_connected():
                return self._browser

            # Orphan cleanup: uvicorn auto-reload, crashes, and taskkills skip
            # lifespan shutdown — pool Chromes (headed but window-hidden via
            # SW_HIDE) survive invisibly and accumulate in Task Manager. Kill
            # any stale marker processes BEFORE launching a fresh browser.
            self._kill_stale_pool_browsers()

            primary = settings.browser_engine.strip().lower()
            logger.info(f"BrowserPool: Starting persistent Chromium ({primary})...")

            if primary == "cloakbrowser":
                try:
                    self._browser = await self._launch_cloakbrowser()
                except Exception as e:
                    logger.warning(
                        f"BrowserPool: CloakBrowser launch failed ({e}). "
                        f"Falling back to Patchright."
                    )
                    self._browser = await self._launch_patchright()
            else:
                self._browser = await self._launch_patchright()

            logger.info(f"BrowserPool: {self._engine} Chromium is ready.")
            return self._browser

    def _kill_stale_pool_browsers(self) -> None:
        """Kill leftover pool Chromes from a previous crashed/reloaded process.

        Every pool launch carries the unique --bd-pool-instance=cloak marker,
        so anything found with that marker right now is an orphan we can
        safely kill (our own browser is guaranteed not running at this point).
        """
        import subprocess

        try:
            pids = pids_with_cmdline_marker(self._POOL_MARKER_ARG)
        except Exception as e:
            logger.debug(f"BrowserPool: could not scan for orphaned pool browsers: {e}")
            return
        if not pids:
            return
        killed = 0
        for pid in pids:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    timeout=5,
                )
                killed += 1
            except Exception:
                continue
        logger.warning(
            f"BrowserPool: cleaned up {killed} orphaned pool Chromium "
            f"process(es) from a previous run."
        )

    def _new_adapter(self):
        """Create an adapter matching the engine that is actually running."""
        if self._engine == "cloakbrowser":
            from src.infrastructure.browser.cloakbrowser_adapter import CloakBrowserAdapter
            return CloakBrowserAdapter()
        from src.infrastructure.browser.patchright_adapter import PatchrightAdapter
        return PatchrightAdapter()

    async def acquire(self, storage_state: dict[str, Any] | None = None):
        """
        Return an adapter backed by a fresh tab in the shared browser.
        The adapter's launch() is skipped — the browser is already running.
        Caller must call release(adapter) when done.
        """
        browser = await self._ensure_browser()

        context_kwargs: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 800},
            # Kept in sync with the Chrome/136 UA used by CurlCFFIFetcher so all
            # fetch paths present one consistent browser fingerprint.
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
            "java_script_enabled": True,
        }
        if storage_state:
            from src.infrastructure.browser.storage_state_utils import sanitize_storage_state

            context_kwargs["storage_state"] = sanitize_storage_state(storage_state)

        context = await browser.new_context(**context_kwargs)
        try:
            page = await context.new_page()
        except Exception:
            try:
                await context.close()
            except Exception:
                pass
            raise

        # Build an adapter and inject the already-running browser/context/page
        # so the adapter's navigate/get_page_content/etc. work exactly as before.
        adapter = self._new_adapter()
        adapter._playwright = self._playwright   # shared — must NOT be stopped on release
        adapter._browser = browser               # shared — must NOT be closed on release
        adapter._context = context               # owned by this tab — will be closed on release
        adapter._page = page                     # owned by this tab — will be closed on release
        adapter._pool_managed = True             # sentinel so close() knows it's pool-managed

        # Track the in-flight tab BEFORE showing the window: with concurrent
        # scrapes, an early release() must never hide the window while another
        # scrape is still using the pool.
        self._active_tabs += 1

        # Pop the idle-hidden pool window back up for this scrape (idempotent)
        self._show_window_if_hidden()

        logger.debug(f"BrowserPool: Issued new tab to caller ({self._active_tabs} active).")
        return adapter

    async def release(self, adapter) -> None:
        """
        Close only the tab (context + page) that was handed out by acquire().
        The underlying browser process is kept alive.
        """
        try:
            if adapter._page:
                await adapter._page.close()
        except Exception as e:
            logger.warning(f"BrowserPool: Error closing page: {e}")
        try:
            if adapter._context:
                await adapter._context.close()
            logger.debug("BrowserPool: Tab released and closed.")
        except Exception as e:
            logger.warning(f"BrowserPool: Error releasing tab: {e}")
        finally:
            # Only hide the window when the LAST active tab is done — otherwise
            # concurrent scrapes would hide each other's window mid-run.
            if self._active_tabs > 0:
                self._active_tabs -= 1
            if self._active_tabs == 0:
                self.hide_window_now()

    async def shutdown(self) -> None:
        """Stop the shared browser. Call this on server shutdown."""
        logger.info("BrowserPool: Shutting down persistent Chromium instance...")
        try:
            if self._browser:
                await self._browser.close()
            # Only Patchright holds an explicit playwright handle here;
            # CloakBrowser stops its own Playwright inside browser.close().
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.warning(f"BrowserPool: Error during shutdown: {e}")
        finally:
            self._browser = None
            self._playwright = None
            self._engine = ""
        logger.info("BrowserPool: Chromium shut down.")


# ── Module-level singleton ────────────────────────────────────────────────────
BrowserPool = _BrowserPool()

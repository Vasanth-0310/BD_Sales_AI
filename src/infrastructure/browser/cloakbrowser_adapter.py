import cloakbrowser

from typing import Any

from src.common.logger import get_logger
from src.infrastructure.browser.patchright_adapter import PatchrightAdapter

logger = get_logger(__name__)


class CloakBrowserAdapter(PatchrightAdapter):
    """
    Primary stealth browser: CloakBrowser (source-patched Chromium binary).

    CloakBrowser's `launch_async()` returns a regular Playwright `Browser`
    object driving a fingerprint-patched Chromium, so every Playwright-based
    method inherited from PatchrightAdapter (navigation, waits, JS injection,
    session capture) works unchanged. Only the launch and teardown differ.

    PatchrightAdapter remains the fallback engine; nodriver remains the
    dedicated engine for persistent-profile domains (Upwork).
    """

    def __init__(self) -> None:
        super().__init__()
        # Log inherited navigation methods under this engine's name so the
        # logs always show which browser actually served the request.
        self._log = logger

    async def launch(self, headless: bool = True, storage_state: dict[str, Any] | None = None) -> None:
        """Launch the patched CloakBrowser Chromium."""
        self._log.info(f"Launching CloakBrowser (headless={headless})...")

        # CloakBrowser manages its own Playwright instance internally and hooks
        # browser.close() to stop it — we must NOT hold or stop a playwright handle.
        self._playwright = None
        self._browser = await cloakbrowser.launch_async(headless=headless)

        context_kwargs: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 800},
            # Kept in sync with the Chrome/136 UA used by CurlCFFIFetcher and
            # BrowserPool so all fetch paths present one consistent fingerprint.
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

        try:
            self._context = await self._browser.new_context(**context_kwargs)
            self._page = await self._context.new_page()
        except Exception:
            # Context creation failed (bad storage state, OOM, ...) — the
            # browser is already running, so tear it down instead of leaking it.
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
            raise
        logger.info("CloakBrowser launched successfully.")

    async def close(self) -> None:
        """Close the browser and release all resources.

        If this adapter was handed out by BrowserPool (pool_managed=True),
        only the tab (page + context) is closed — the shared browser stays alive.
        For non-pooled instances, closing the browser also tears down the
        Playwright instance CloakBrowser manages internally.

        Each teardown step is isolated: a crashed page/context (very common
        on Cloudflare-blocked pages) must never skip browser.close(), or the
        whole stealth-Chromium + Playwright driver leaks.
        """
        logger.info("Closing CloakBrowser...")
        for step_name, step in (("page", self._page), ("context", self._context)):
            if step is None:
                continue
            try:
                await step.close()
            except Exception as e:
                logger.warning(f"CloakBrowser {step_name} close failed (continuing teardown): {e}")
        if not self._pool_managed and self._browser:
            # CloakBrowser hooks close() to stop its internal Playwright.
            try:
                await self._browser.close()
            except Exception as e:
                logger.warning(f"CloakBrowser browser close failed: {e}")
        self._page = None
        self._context = None
        logger.info("CloakBrowser closed.")

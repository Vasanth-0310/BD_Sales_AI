from typing import Type

from src.domain.interfaces.browser.i_browser import IBrowser
from src.domain.exceptions.scraper_exceptions import BrowserLaunchFailedException
from src.infrastructure.browser.cloakbrowser_adapter import CloakBrowserAdapter
from src.infrastructure.browser.patchright_adapter import PatchrightAdapter
from src.infrastructure.browser.nodriver_adapter import NodriverAdapter
from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger(__name__)

# Engine fallback chains, keyed by the configured primary engine.
# CloakBrowser needs the patched binary present (and a license for >1
# concurrent browser session); Patchright is the like-for-like Playwright
# substitute; nodriver is the CDP-direct last resort.
_ENGINE_CHAINS: dict[str, list[Type[IBrowser]]] = {
    "cloakbrowser": [CloakBrowserAdapter, PatchrightAdapter, NodriverAdapter],
    "patchright": [PatchrightAdapter, CloakBrowserAdapter, NodriverAdapter],
    "nodriver": [NodriverAdapter, PatchrightAdapter, CloakBrowserAdapter],
}

_PRIMARY_ENGINE = settings.browser_engine.strip().lower()
_ENGINE_CHAIN = _ENGINE_CHAINS.get(_PRIMARY_ENGINE, _ENGINE_CHAINS["cloakbrowser"])


class BrowserFactory:
    """
    Factory that returns IBrowser implementations with engine fallback.

    Priority (default, configurable via settings.browser_engine):
        1. CloakBrowser (primary — source-patched Chromium stealth browser)
        2. Patchright   (fallback — stealth-patched Playwright)
        3. nodriver     (last resort — CDP-direct browser control)

    This is the ONLY place in the codebase that knows about the engines.
    All other layers receive an IBrowser and are unaware of which backend is active.
    """

    @staticmethod
    def get_browser() -> IBrowser:
        """
        Return an uninitialized IBrowser instance for the primary engine.

        Prefer `launch_browser()` for automatic engine fallback at launch time.

        Raises:
            BrowserLaunchFailedException: If the primary engine cannot be constructed.
        """
        try:
            adapter = _ENGINE_CHAIN[0]()
            logger.info(f"BrowserFactory: primary engine is {type(adapter).__name__}.")
            return adapter
        except Exception as e:
            raise BrowserLaunchFailedException(
                reason=f"Primary browser engine failed to initialize: {e}"
            ) from e

    @staticmethod
    async def launch_browser(
        headless: bool = True,
        storage_state: dict | None = None,
        skip_orphan_kill: bool = False,
    ) -> IBrowser:
        """
        Return a LAUNCHED IBrowser, trying each engine in the fallback chain
        in order. The first engine whose launch succeeds wins; failures
        (missing binary, license denial, crash) fall through to the next.

        skip_orphan_kill: forwarded to engines that support it (nodriver) —
        must be True when the storage_state references a profile that a live
        NodriverPool warm Chrome may already own.

        Raises:
            BrowserLaunchFailedException: If every engine in the chain fails.
        """
        last_error: Exception | None = None
        for engine_cls in _ENGINE_CHAIN:
            adapter = engine_cls()
            try:
                # nodriver accepts the flag; other engines ignore unknown kwargs
                # via their own signature — pass explicitly only where supported.
                import inspect
                if "skip_orphan_kill" in inspect.signature(adapter.launch).parameters:
                    await adapter.launch(headless=headless, storage_state=storage_state,
                                         skip_orphan_kill=skip_orphan_kill)
                else:
                    await adapter.launch(headless=headless, storage_state=storage_state)
                logger.info(f"BrowserFactory: launched {engine_cls.__name__}.")
                return adapter
            except Exception as e:
                last_error = e
                logger.warning(
                    f"BrowserFactory: {engine_cls.__name__} launch failed ({e}). "
                    f"Trying next engine..."
                )
        raise BrowserLaunchFailedException(
            reason=f"All browser engines failed to launch. Last error: {last_error}"
        ) from last_error

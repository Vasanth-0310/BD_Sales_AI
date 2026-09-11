import re
from typing import Any, Optional
from patchright.async_api import async_playwright, Browser, BrowserContext, Page
from src.domain.interfaces.browser.i_browser import IBrowser
from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger(__name__)


class PatchrightAdapter(IBrowser):
    """
    Fallback browser implementation using Patchright (stealth-patched Playwright).
    Used when CloakBrowser is unavailable or fails to initialize.
    """

    def __init__(self) -> None:
        self._log = logger  # subclasses override with their own engine logger
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._pool_managed: bool = False  # True when owned by BrowserPool (shared browser)

    async def launch(self, headless: bool = True, storage_state: dict[str, Any] | None = None) -> None:
        """Launch a Patchright Chromium browser with stealth settings."""
        self._log.info(f"Launching Patchright (headless={headless})...")
        self._playwright = await async_playwright().start()

        self._browser = await self._playwright.chromium.launch(
            headless=headless,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        context_kwargs: dict[str, Any] = {
            "viewport": {"width": 1280, "height": 800},
            # Kept in sync with the Chrome/136 UA used by CurlCFFIFetcher and
            # BrowserPool so all fetch paths present one consistent fingerprint.
            # A captured session's user_agent OVERRIDES this: cf_clearance is
            # UA-bound and a mismatched UA makes Cloudflare re-challenge.
            "user_agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/136.0.0.0 Safari/537.36"
            ),
            "java_script_enabled": True,
        }
        session_ua = (storage_state or {}).get("user_agent")
        if isinstance(session_ua, str) and session_ua.strip():
            context_kwargs["user_agent"] = session_ua.strip()
        self._headless = headless

        if storage_state:
            from src.infrastructure.browser.storage_state_utils import sanitize_storage_state

            context_kwargs["storage_state"] = sanitize_storage_state(storage_state)

        try:
            self._context = await self._browser.new_context(**context_kwargs)
            self._page = await self._context.new_page()
        except Exception:
            # Context creation failed — close the browser we just started
            # so failed launches don't leak Chromium processes.
            try:
                await self._browser.close()
                await self._playwright.stop()
            except Exception:
                pass
            self._browser = None
            self._playwright = None
            raise
        self._log.info("Patchright browser launched successfully.")

    async def navigate(self, url: str, wait_for_job_details: bool = True) -> int:
        """Navigate to the given URL and return the HTTP response status code."""
        if not self._page:
            raise RuntimeError("Browser not launched. Call launch() first.")

        self._log.info(f"Navigating to: {url}")
        response = await self._page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=settings.browser_timeout_ms,
        )
        status = response.status if response else 200

        if wait_for_job_details:
            # Run the networkidle settle and the job-content selector race
            # CONCURRENTLY. Sequentially these cost up to 3s + 2s of waiting;
            # in parallel the page proceeds as soon as BOTH are settled, which is
            # dominated by whichever finishes last — typically ~2s on LinkedIn
            # where networkidle never fires but job content arrives early.
            _JOB_CONTENT_SELECTORS = [
                "#job-details",
                ".jobs-description",
                ".show-more-less-html__markup",
                "#jobDescriptionText",
                ".jobsearch-JobComponent",
                ".job-view-layout",
            ]

            async def _wait_for(selector: str) -> str | None:
                try:
                    await self._page.wait_for_selector(selector, timeout=4000)
                    return selector
                except Exception:
                    return None

            async def _wait_network_idle() -> None:
                try:
                    await self._page.wait_for_load_state("networkidle", timeout=3000)
                except Exception:
                    pass  # networkidle rarely fires on SPAs — DOM state is fine

            import asyncio as _asyncio
            idle_task = _asyncio.create_task(_wait_network_idle())
            selector_results = await _asyncio.gather(*[_wait_for(s) for s in _JOB_CONTENT_SELECTORS])
            matched = next((r for r in selector_results if r), None)
            # Don't abandon the idle wait entirely — give it whatever time the
            # selector race already bought us, then move on.
            try:
                await _asyncio.wait_for(_asyncio.shield(idle_task), timeout=1.0)
            except Exception:
                idle_task.cancel()
            if matched:
                self._log.info(f"Job content element detected: '{matched}'")
            else:
                # No known selector found — give the page a short extra moment to settle
                self._log.debug("No known job content selector matched. Waiting 2s for DOM to settle.")
                await self._page.wait_for_timeout(800)

            # Handle Cloudflare / Turnstile challenge pages.
            #
            # Cloudflare challenge pages return HTTP 403 initially — the real 200
            # only arrives *after* Turnstile verifies and the browser redirects.
            # So a 403 gets ONE chance: if a challenge is detected, wait for it
            # to auto-resolve. If no challenge is present (hard block / Ray-ID
            # page) or it doesn't resolve in time, bail IMMEDIATELY — waiting
            # for job content on a dead page burns 10+ seconds for nothing and
            # the engine loop should fall through to the next engine instead.
            if status in (403, 503):
                resolved = await self._wait_for_cloudflare_if_needed()
                if not resolved:
                    self._log.warning(
                        f"HTTP {status} returned and no Cloudflare challenge resolved. Bailing out."
                    )
                    return status

                # Challenge resolved — the redirect may have real content now.
                await self._wait_for_readable_content(timeout_ms=10000)
                visible = await self.get_visible_text() if self._page else ""
                if self._has_job_detail_content(visible):
                    self._log.info(
                        f"HTTP {status} initial response, but Cloudflare challenge resolved "
                        f"— proceeding with rendered content."
                    )
                    status = 200
                else:
                    self._log.warning(
                        f"HTTP {status}: challenge passed but content never became readable."
                    )
                    return status
            else:
                # 200 pages can still be "Just a moment..." interstitials.
                await self._wait_for_cloudflare_if_needed()
                await self._wait_for_readable_content(timeout_ms=10000)

            # Click any collapsed "Show more" sections
            await self._auto_expand_collapsed_sections()
            # Cap the post-click wait to just 3s (content should appear instantly if expanded)
            await self._wait_for_readable_content(timeout_ms=3000)
            
        self._log.info(f"Navigation complete. Status: {status}")
        return status


    async def _wait_for_cloudflare_if_needed(self) -> bool:
        """
        If the current page is a Cloudflare / Turnstile challenge, wait up to
        20 seconds for it to auto-verify and redirect to the real page.

        Detects two states:
          1. Pre-verification: page title contains 'just a moment',
             'verifying', or 'challenge - upwork'.
          2. Post-Turnstile: Turnstile widget verified but server hasn't
             responded yet — visible text contains
             'verification successful. waiting for'.

        Returns True only when a challenge was detected AND resolved.
        Interactive challenges ("Verify you are human" checkbox) cannot
        auto-resolve — callers should bail fast and let the next engine
        (nodriver with its cf_clearance profile) take over.
        """
        if not self._page:
            return False

        title = await self._page.title()
        title_lower = title.lower()
        is_challenge = (
            "just a moment" in title_lower
            or "verifying" in title_lower
            or "challenge - upwork" in title_lower
        )

        # Also check the page body for the post-Turnstile waiting state.
        # Turnstile may verify but the redirect hasn't happened yet — the
        # visible text literally says "Verification successful. Waiting for
        # <domain> to respond". One body read serves both checks below.
        body_text = ""
        try:
            body_text = (await self.get_visible_text() or "").lower()
        except Exception:
            pass
        if not is_challenge and "verification successful" in body_text and "waiting for" in body_text:
            is_challenge = True
            self._log.info(
                "Post-Turnstile waiting state detected "
                "('Verification successful. Waiting for …'). "
                "Waiting for server redirect (up to 20s)..."
            )

        if not is_challenge:
            return False

        if "verification successful" not in body_text:
            # Interactive challenges ("Verify you are human" — the checkbox
            # lives inside a Turnstile IFRAME, so body text may not show it)
            # need a human click. In a VISIBLE browser the user can solve it
            # in place — give a generous window instead of bailing after 20s.
            if not getattr(self, "_headless", True):
                self._log.info(
                    "Cloudflare challenge detected in a VISIBLE browser — waiting up to 90s. "
                    "If a 'Verify you are human' checkbox appears, solve it in the opened window..."
                )
            else:
                self._log.info("Cloudflare challenge detected. Waiting for auto-verification (up to 20s)...")

        challenge_timeout_ms = 90_000 if not getattr(self, "_headless", True) else 20_000

        try:
            # Wait until the title changes away from the Cloudflare page AND
            # the "Verification successful" text disappears from the body.
            await self._page.wait_for_function(
                "() => {"
                "  const t = document.title.toLowerCase();"
                "  if (t.includes('just a moment') || t.includes('verifying') "
                "      || t.includes('challenge - upwork')) return false;"
                "  const b = (document.body ? document.body.innerText : '').toLowerCase();"
                "  if (b.includes('verification successful') && b.includes('waiting for')) "
                "    return false;"
                "  return true;"
                "}",
                timeout=challenge_timeout_ms,
            )
            # Extra pause for the real page to fully render after redirect
            await self._page.wait_for_load_state("networkidle", timeout=8000)
            self._log.info(f"Cloudflare challenge passed. Now at: {self._page.url}")
            return True
        except Exception:
            self._log.warning("Cloudflare challenge did not resolve in time. Proceeding with current DOM.")
            return False

    async def _auto_expand_collapsed_sections(self) -> None:
        """Click all 'Show More' / 'Expand' buttons using a fast JS injection."""
        if not self._page:
            return

        script = """
        (() => {
            const cssSelectors = [
                '.show-more-less-html__button',
                '.jobs-description__footer-button',
                'button[aria-label*="Show more" i]',
                'button[aria-label*="Expand" i]',
                'button.show-more-button',
                'button[class*="show-more" i]',
                'button[class*="truncation" i]',
                'a[class*="show-more" i]',
                'span[class*="show-more" i]',
                '[data-ev-label*="more" i]',
            ];
            const textLabels = new Set(['more', 'show more', 'read more', 'view more', 'see more']);
            const clicked = new Set();
            
            function willNavigate(element) {
                const anchor = element.closest('a');
                if (!anchor) return false;
                const href = anchor.getAttribute('href');
                if (!href) return false;
                if (href.startsWith('#') || href.startsWith('javascript:')) return false;
                return true;
            }

            for (const sel of cssSelectors) {
                try {
                    for (const el of document.querySelectorAll(sel)) {
                        if (!clicked.has(el) && (el.offsetWidth || el.offsetHeight || el.getClientRects().length)) {
                            if (willNavigate(el)) continue;
                            el.click();
                            clicked.add(el);
                        }
                    }
                } catch (_) {}
            }

            for (const el of document.querySelectorAll('button, a, span, div, [role="button"]')) {
                if (clicked.has(el)) continue;
                if (willNavigate(el)) continue;
                
                const text = (el.innerText || el.textContent || el.getAttribute('aria-label') || '').trim().toLowerCase();
                const visible = !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
                if (visible && textLabels.has(text)) {
                    el.click();
                    clicked.add(el);
                }
            }
            return clicked.size;
        })()
        """
        try:
            clicked = await self._page.evaluate(script)
            if clicked:
                await self._page.wait_for_timeout(500)
                self._log.info(f"Expanded {clicked} collapsed section(s) via JS injection.")
        except Exception as e:
            self._log.debug(f"Auto-expand JS injection failed: {e}")

    async def get_page_content(self) -> str:
        """Return the full rendered HTML of the current page."""
        if not self._page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return await self._page.content()

    async def get_visible_text(self) -> str:
        """Return visible text from the current document body."""
        if not self._page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return await self._page.evaluate("document.body ? document.body.innerText : ''")

    async def get_current_url(self) -> str:
        """Return the current URL after any client-side redirects."""
        if not self._page:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._page.url

    async def capture_session(self) -> dict[str, Any]:
        """Capture the current browser storage state."""
        if not self._context:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return await self._context.storage_state()

    async def _wait_for_readable_content(self, timeout_ms: int = 10000) -> None:
        """Wait until visible text looks like a job page rather than an SPA shell."""
        if not self._page:
            return

        deadline_ms = min(max(timeout_ms, 3000), 30000)
        step_ms = 1000
        elapsed_ms = 0
        last_text = ""

        # Fast path DOM check for SPAs (LinkedIn, Upwork, Indeed)
        js_fast_check = """
            () => {
                const selectors = [
                    '#job-details', '.jobs-description', '.show-more-less-html__markup',
                    '#jobDescriptionText', '.jobsearch-JobComponent', '.job-view-layout',
                    '[data-test="job-description"]', '.job-description'
                ];
                for (let s of selectors) {
                    if (document.querySelector(s)) return true;
                }
                return false;
            }
        """

        while elapsed_ms < deadline_ms:
            try:
                # 1. Fast path: check DOM for exact job containers instantly
                if await self._page.evaluate(js_fast_check):
                    self._log.debug("Fast path DOM check succeeded.")
                    return
            except Exception:
                pass

            # 2. Slow path: check rendered text word count
            try:
                visible_text = await self.get_visible_text()
            except Exception:
                visible_text = ""

            if self._is_dead_end_page(visible_text):
                self._log.warning("Dead-end/interstitial page detected — stopping wait early.")
                return

            if self._has_job_detail_content(visible_text):
                return

            last_text = visible_text
            await self._trigger_lazy_render()
            await self._page.wait_for_timeout(step_ms)
            elapsed_ms += step_ms

        preview = re.sub(r"\s+", " ", last_text or "").strip()[:180]
        self._log.warning(
            "Timed out waiting for job-detail text. "
            f"Last visible text was {len(last_text or '')} chars: {preview!r}"
        )

    async def _trigger_lazy_render(self) -> None:
        if not self._page:
            return
        try:
            await self._page.evaluate(
                """
                (() => {
                    const height = Math.max(
                        document.body ? document.body.scrollHeight : 0,
                        document.documentElement ? document.documentElement.scrollHeight : 0
                    );
                    const firstStop = Math.min(height, Math.max(window.innerHeight || 900, 900));
                    window.scrollTo(0, firstStop);
                    setTimeout(() => window.scrollTo(0, 0), 150);
                    return true;
                })()
                """
            )
        except Exception as e:
            self._log.debug(f"Lazy-render scroll failed: {e}")

    @staticmethod
    def _has_job_detail_content(content: Any) -> bool:
        if content is None:
            return False

        text = re.sub(r"\s+", " ", str(content)).strip().lower()
        if text in ("", "undefined", "null", "none"):
            return False

        signals = (
            "about the job",
            "summary",
            "description",
            "responsibilities",
            "requirements",
            "qualifications",
            "skills",
            "employment type",
            "experience level",
            "hourly",
            "salary",
            "posted",
            "remote",
            "company",
        )
        signal_count = sum(1 for signal in signals if signal in text)
        word_count = len(text.split())
        if word_count >= 120 and signal_count >= 1:
            return True
        return word_count >= 35 and signal_count >= 2

    async def close(self) -> None:
        """Close the browser and release all resources.

        If this adapter was handed out by BrowserPool (pool_managed=True),
        only the tab (page + context) is closed — the shared browser stays alive.
        BrowserPool.release() handles this case; calling close() directly is safe.

        Each teardown step is isolated: a crashed page/context must never skip
        browser.close()/playwright.stop(), or the Chromium process leaks.
        """
        self._log.info("Closing Patchright browser...")
        for step_name, step in (("page", self._page), ("context", self._context)):
            if step is None:
                continue
            try:
                await step.close()
            except Exception as e:
                self._log.warning(f"Patchright {step_name} close failed (continuing teardown): {e}")
        if not self._pool_managed:
            # Only tear down the browser/playwright for non-pooled instances
            if self._browser:
                try:
                    await self._browser.close()
                except Exception as e:
                    self._log.warning(f"Patchright browser close failed: {e}")
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception as e:
                    self._log.warning(f"Patchright driver stop failed: {e}")
        self._page = None
        self._context = None
        self._log.info("Patchright browser closed.")

    @staticmethod
    def _is_dead_end_page(text: str | None) -> bool:
        if not text:
            return False
        sample = text[:1500].lower()

        dead_end_signals = (
            "we'll be right back",
            "this page is offline right now",
            "check back later",
            "just a moment",
        )
        word_count = len(text.split())
        has_dead_end = any(s in sample for s in dead_end_signals)
        return has_dead_end and word_count < 200

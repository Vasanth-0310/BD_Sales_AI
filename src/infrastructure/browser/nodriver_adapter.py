import asyncio
import re
import sys
from pathlib import Path
from typing import Any, Optional

import nodriver as uc
from nodriver import cdp

from src.common.config import settings
from src.domain.interfaces.browser.i_browser import IBrowser
from src.common.logger import get_logger

logger = get_logger(__name__)

# Launch args shared by NodriverAdapter.launch() and NodriverPool._launch()
# so a cold start and a pooled warm start behave identically.
NODRIVER_BROWSER_ARGS = [
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-infobars",
    "--disable-extensions",
    "--disable-popup-blocking",
    "--disable-session-crashed-bubble",  # Hides the "Restore Pages?" popup
    # nodriver's stop() hard-kills Chrome, so the profile's exit state
    # is always "crashed" and the next launch shows a "Restore pages?"
    # bubble. This flag (the modern replacement for the one above)
    # actually suppresses it on current Chrome builds.
    "--hide-crash-restore-bubble",
    "--password-store=basic",            # Prevent Chrome from hanging on system keychains
    "--use-mock-keychain",
    "--disable-gpu",                     # Render stability in headless environments
]


class NodriverAdapter(IBrowser):
    """
    Browser implementation using nodriver.
    Communicates directly with Chrome via the Chrome DevTools Protocol (CDP).
    """

    def __init__(self, user_data_dir: str | None = None) -> None:
        self._browser: Optional[uc.Browser] = None
        self._tab: Optional[uc.Tab] = None
        self._headless: bool = True
        self._user_data_dir = str(Path(user_data_dir).resolve()) if user_data_dir else None
        self._pool_managed: bool = False  # True when owned by NodriverPool
        self._chrome_pid: int | None = None  # captured at launch for force-kill backstop

    async def launch(
        self,
        headless: bool = True,
        storage_state: dict[str, Any] | None = None,
        skip_orphan_kill: bool = False,
    ) -> None:
        """Launch nodriver Chrome browser, optionally reusing a persistent profile.

        skip_orphan_kill: MUST be True when a cold launch could target a
        profile already owned by a live NodriverPool warm Chrome — the orphan
        kill runs BEFORE this flag could otherwise be set by the caller, and
        it would taskkill the pool's live process (critical timing bug).
        """
        self._headless = headless
        self._skip_orphan_kill = skip_orphan_kill
        profile_dir = self._resolve_profile_dir(storage_state)
        logger.info(f"Launching nodriver (headless={headless})...")
        if profile_dir:
            logger.info(f"Using persistent Chrome profile: {profile_dir}")
            if not self._skip_orphan_kill:
                # Clean up residual Chrome lock files from previous crashed/interrupted runs.
                # NEVER do this while a pool warm Chrome owns the profile —
                # deleting its SingletonLock lets a second Chrome process write
                # concurrently to the profile's SQLite DBs → "malformed disk image".
                for lock_name in ("lockfile", "SingletonLock", "SingletonSocket", "SingletonCookie"):
                    lock_file = Path(profile_dir) / lock_name
                    if lock_file.exists():
                        try:
                            lock_file.unlink(missing_ok=True)
                            logger.info(f"Cleaned up residual Chrome lock: {lock_name}")
                        except Exception as e:
                            logger.debug(f"Could not delete Chrome lock {lock_name}: {e}")

        # Build explicit browser args to guarantee headless on Windows
        # (nodriver's headless flag alone can be ignored on some Windows configs)
        browser_args = list(NODRIVER_BROWSER_ARGS)
        if headless:
            browser_args.append("--headless=new")

        # Kill orphaned Chrome processes that are locking this profile dir.
        # Previous Ctrl+C / force-quit of capture_session.py leaves zombie
        # Chrome processes that hold the profile lock, causing the next
        # uc.start() to fail with "Failed to connect to browser".
        # Runs regardless of headless — a zombie holds the lock either way.
        if profile_dir:
            await self._kill_orphan_chrome_for_profile(profile_dir)

        self._browser = await uc.start(
            headless=headless,
            user_data_dir=profile_dir,
            browser_args=browser_args,
            no_sandbox=True,
        )
        # Remember the Chrome PID immediately — needed as the force-kill
        # backstop in close() if nodriver's stop() silently no-ops.
        self._chrome_pid = getattr(
            getattr(self._browser, "_process", None), "pid", None
        ) or getattr(self._browser, "_process_pid", None)

        self._tab = await self._browser.get("about:blank")

        # Always inject the saved session cookies, even when a persistent
        # profile is used. nodriver's stop() hard-kills Chrome, which can
        # prevent the profile's on-disk cookie DB from being flushed — the
        # profile alone may be missing auth cookies (li_at etc.) that the
        # captured storage_state still has. Injecting merges both: the
        # profile keeps fingerprint/localStorage, the cookies restore auth.
        if storage_state and storage_state.get("cookies"):
            await self._inject_cookies(storage_state["cookies"])
            logger.info(f"Injected {len(storage_state['cookies'])} cookies into nodriver session.")

        logger.info("nodriver browser launched successfully.")

    async def navigate(self, url: str, wait_for_job_details: bool = True) -> int:
        """
        Navigate to the given URL.
        Note: nodriver does not natively expose HTTP response status codes.
        Returns 200 on success, 0 on failure.
        """
        if not self._browser or not self._tab:
            raise RuntimeError("Browser not launched. Call launch() first.")

        logger.info(f"nodriver navigating to: {url}")
        try:
            if self._pool_managed and self._tab:
                # Pool-managed: navigate the existing tab via CDP instead of
                # browser.get(url) which opens a NEW tab and leaks the old one.
                await self._tab.send(cdp.page.navigate(url=url))
                # Wait for the page to actually load before proceeding
                try:
                    await self._tab.sleep(0.5)
                    # Wait for readyState to be at least 'interactive'
                    for _ in range(20):  # up to ~10s
                        state = await self._tab.evaluate(
                            "document.readyState", return_by_value=True
                        )
                        state_val = self._extract_evaluate_value(state)
                        if state_val in ("interactive", "complete"):
                            break
                        await self._tab.sleep(0.5)
                except Exception:
                    pass  # Proceed with whatever DOM state we have
            else:
                # Non-pooled: browser.get() is fine — the whole browser is closed at the end
                self._tab = await self._browser.get(url)
            if wait_for_job_details:
                await self._wait_for_verification_page()
                await self._wait_for_readable_content(timeout_sec=10)
                await self._auto_expand_collapsed_sections()
                await self._wait_for_readable_content(timeout_sec=3)
            logger.info(f"Navigation complete. Current URL: {self._tab.url}")
            return 200
        except Exception as e:
            logger.error(f"nodriver navigation failed: {e}")
            return 0

    async def get_page_content(self) -> str:
        """Return the full rendered HTML of the current page."""
        if not self._tab:
            raise RuntimeError("Browser not launched. Call launch() first.")
        content = await self._tab.get_content()
        if self._is_invalid_content(content):
            content = await self._tab.evaluate(
                "document.documentElement ? document.documentElement.outerHTML : ''",
                return_by_value=True,
            )
            content = self._extract_evaluate_value(content)
        return str(content or "")

    async def get_visible_text(self) -> str:
        """Return visible text from the current document body."""
        if not self._tab:
            raise RuntimeError("Browser not launched. Call launch() first.")
        text = await self._tab.evaluate(
            "document.body ? document.body.innerText : ''",
            return_by_value=True,
        )
        text = self._extract_evaluate_value(text)
        return str(text or "")

    async def get_current_url(self) -> str:
        """Return the current URL after any client-side redirects."""
        if not self._tab:
            raise RuntimeError("Browser not launched. Call launch() first.")
        if self._tab.url:
            return self._tab.url
        current_url = await self._tab.evaluate("window.location.href", return_by_value=True)
        current_url = self._extract_evaluate_value(current_url)
        return str(current_url or "")

    async def capture_session(self) -> dict[str, Any]:
        """
        Capture the current browser session as a storage state dict.
        The persistent profile path is saved because cookies alone are often
        insufficient for sites with browser integrity checks.
        """
        if not self._browser:
            raise RuntimeError("Browser not launched. Call launch() first.")

        raw_cookies = await self._browser.cookies.get_all(requests_cookie_format=False)

        cookies_list = []
        for c in raw_cookies:
            same_site = c.same_site
            if hasattr(same_site, "value"):
                same_site = same_site.value
            elif same_site is not None:
                same_site = str(same_site)

            expires = c.expires
            if expires is not None:
                try:
                    expires = float(expires)
                except (TypeError, ValueError):
                    expires = None

            cookies_list.append({
                "name": str(c.name) if c.name else "",
                "value": str(c.value) if c.value else "",
                "domain": str(c.domain) if c.domain else "",
                "path": str(c.path) if c.path else "/",
                "expires": expires,
                "httpOnly": bool(c.http_only),
                "secure": bool(c.secure),
                "sameSite": same_site,
            })

        storage_state = {
            "cookies": cookies_list,
            "origins": [],
        }
        if self._user_data_dir:
            storage_state["nodriver_profile_dir"] = self._user_data_dir

        # Cloudflare binds cf_clearance to the UA that solved the challenge —
        # automation contexts MUST present the same UA or the cookie is
        # rejected and the site re-challenges.
        try:
            ua = await self._tab.evaluate("navigator.userAgent", return_by_value=True)
            ua = self._extract_evaluate_value(ua)
            if ua:
                storage_state["user_agent"] = str(ua)
        except Exception as e:
            logger.debug(f"Could not read navigator.userAgent during capture: {e}")

        logger.info(f"Session captured: {len(cookies_list)} cookies.")
        return storage_state

    async def close(self) -> None:
        """Close the nodriver browser and release all resources.

        If this adapter was handed out by NodriverPool (pool_managed=True),
        only the tab is closed — the shared Chrome process stays warm for
        the next scrape.
        """
        if self._pool_managed:
            logger.info("Closing pooled nodriver tab (keeping Chrome warm)...")
            if self._tab:
                try:
                    await self._tab.close()
                except Exception as e:
                    logger.debug(f"Error closing pooled tab: {e}")
            # Tab accounting goes back to the pool: it re-hides the idle
            # window only when the LAST tab for this profile closes, so
            # concurrent scrapes never hide each other's window.
            try:
                from src.infrastructure.browser.nodriver_pool import NodriverPool

                NodriverPool.notify_tab_closed(self._user_data_dir)
            except Exception as e:
                logger.debug(f"Could not notify NodriverPool of tab close: {e}")
            self._tab = None
            logger.info("Pooled nodriver tab closed.")
            return

        # Non-pooled (cold start): fully terminate the Chrome process.
        logger.info("Closing nodriver browser...")
        pid = None
        try:
            # Capture the PID BEFORE any teardown — once stop() runs, its
            # internal handles may be nulled (nodriver package bug) and we'd
            # lose the only handle to kill Chrome by force. Real attrs per
            # nodriver source: `_process` (subclass has pid) and `_process_pid`.
            pid = self._chrome_pid
            if not pid:
                proc = getattr(self._browser, "_process", None)
                pid = getattr(proc, "pid", None) or getattr(
                    self._browser, "_process_pid", None
                )
        except Exception:
            pid = None

        if self._browser:
            try:
                await self._browser.aclose()  # proper async variant of stop()
            except Exception:
                try:
                    self._browser.stop()
                except Exception as e:
                    logger.warning(f"nodriver stop() failed ({e}); forcing kill.")
            await asyncio.sleep(0.3)

            # Fallback: verify the process actually died. nodriver's stop()
            # can silently no-op (terminate loop nulls PIDs on first error),
            # so a taskkill by PID is the guaranteed backstop.
            if pid and self._process_alive(pid):
                try:
                    import subprocess

                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=5,
                    )
                    logger.warning(f"nodriver Chrome survived aclose — killed PID {pid}.")
                except Exception as e:
                    logger.warning(
                        f"nodriver Chrome may still be running (PID {pid}); "
                        f"taskkill failed: {e}"
                    )
        self._browser = None
        self._tab = None
        logger.info("nodriver browser closed.")

    @staticmethod
    def _process_alive(pid: int | None) -> bool:
        """Check whether a process ID is still running (Windows-safe)."""
        if not pid:
            return False
        try:
            import subprocess

            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return str(pid) in (result.stdout or "")
        except Exception:
            return False

    async def _inject_cookies(self, cookies: list[dict]) -> None:
        """Inject a list of cookie dicts into the browser via CDP."""
        if not self._browser:
            return

        cdp_cookies = []
        for c in cookies:
            try:
                expires = c.get("expires")
                if expires is not None and expires != -1:
                    try:
                        expires = cdp.network.TimeSinceEpoch(float(expires))
                    except (TypeError, ValueError):
                        expires = None
                else:
                    expires = None

                same_site = c.get("sameSite") or c.get("same_site")
                if same_site and isinstance(same_site, str):
                    try:
                        same_site = cdp.network.CookieSameSite.from_json(same_site)
                    except Exception:
                        same_site = None
                elif not isinstance(same_site, cdp.network.CookieSameSite):
                    same_site = None

                cdp_cookies.append(
                    cdp.network.CookieParam(
                        name=str(c.get("name", "")),
                        value=str(c.get("value", "")),
                        domain=str(c.get("domain")) if c.get("domain") else None,
                        path=str(c.get("path", "/")),
                        expires=expires,
                        secure=bool(c.get("secure", False)),
                        http_only=bool(c.get("httpOnly") or c.get("http_only", False)),
                        same_site=same_site,
                    )
                )
            except Exception as e:
                logger.warning(f"Skipping malformed cookie '{c.get('name', '?')}': {e}")

        if cdp_cookies:
            await self._browser.cookies.set_all(cdp_cookies)

    async def _kill_orphan_chrome_for_profile(self, profile_dir: str) -> None:
        """Kill any Chrome processes locking this profile directory."""
        # F5: when this adapter is a last-resort factory launch (not pool-
        # managed), killing by profile path would also murder the NodriverPool's
        # LIVE warm Chrome for that same profile plus any in-flight pooled tabs.
        # The subsequent cold uc.start() will simply fail on Chrome's own
        # SingletonLock instead — a clean, non-destructive outcome.
        if getattr(self, "_skip_orphan_kill", False):
            logger.debug("Skipping orphan-chrome cleanup (factory fallback path).")
            return
        if sys.platform != "win32" or not profile_dir:
            return
        try:
            import subprocess
            # Match on the FULL resolved profile path, not just the directory name —
            # a generic name like "profile" would otherwise kill unrelated Chrome
            # processes (including the user's own browser).
            resolved = str(Path(profile_dir).resolve())
            safe_marker = resolved.replace("\\", "*").replace("/", "*")

            def _kill_orphans_sync() -> None:
                result = subprocess.run(
                    [
                        "powershell", "-NoProfile", "-Command",
                        f"Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" "
                        f"| Where-Object {{ $_.CommandLine -like '*{safe_marker}*' "
                        f"-and $_.CommandLine -notlike '*{safe_marker}[0-9a-zA-Z_-]*' }} "
                        f"| Select-Object -ExpandProperty ProcessId",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                pids = {int(line) for line in result.stdout.split() if line.strip().isdigit()}
                for pid in pids:
                    logger.info(f"Killing orphan Chrome process (PID {pid}) locking profile: {safe_marker}")
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=5)

            # Blocking PowerShell round-trips (1-3s) must not freeze the event loop.
            await asyncio.to_thread(_kill_orphans_sync)
        except Exception as e:
            logger.debug(f"Failed to kill orphan Chrome for {profile_dir}: {e}")

    def _resolve_profile_dir(self, storage_state: dict[str, Any] | None) -> str | None:
        """Prefer an explicit adapter profile, then a usable profile path saved in storage state."""
        if self._user_data_dir:
            profile_dir = self._user_data_dir
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            return profile_dir

        if storage_state:
            raw_profile_dir = storage_state.get("nodriver_profile_dir") or storage_state.get("profile_dir")
            if raw_profile_dir:
                profile_path = Path(str(raw_profile_dir)).resolve()
                if profile_path.exists() and any(profile_path.iterdir()):
                    self._user_data_dir = str(profile_path)
                    return self._user_data_dir
                logger.warning(
                    f"Saved Chrome profile is missing or empty: {profile_path}. "
                    "Falling back to cookie injection."
                )

        return None

    async def _wait_for_verification_page(self) -> None:
        """
        Give browser-side verification pages time to complete after navigation.
        If they do not complete, the caller will handle the resulting page content.
        """
        if not self._tab:
            return

        timeout_sec = min(max(settings.browser_timeout_ms / 1000.0, 5.0), 30.0)
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout_sec:
            html = await self._tab.get_content()
            if self._is_invalid_content(html):
                await asyncio.sleep(1)
                continue
            
            sample = html[:4000].lower()
            
            if (
                "just a moment" not in sample
                and "additional verification required" not in sample
                and "cf-browser-verification" not in sample
                and "challenge - upwork" not in sample
                and "troubleshooting cloudflare errors" not in sample
                and "verification successful. waiting for" not in sample
                and "your ray id for this request" not in sample
                and not ("cloudflare" in sample and ("verifying" in sample or "ray id" in sample))
            ):
                return
            await asyncio.sleep(1)

    async def _wait_for_readable_content(self, timeout_sec: int | None = None) -> None:
        """Wait until the current document has useful visible job text."""
        if not self._tab:
            return

        if timeout_sec is None:
            timeout_sec = int(min(max(settings.browser_timeout_ms / 1000.0, 5.0), 30.0))
        start_time = asyncio.get_event_loop().time()
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

        while (asyncio.get_event_loop().time() - start_time) < timeout_sec:
            try:
                # 1. Fast path: check DOM for exact job containers instantly
                res = await self._tab.evaluate(js_fast_check, return_by_value=True)
                if self._extract_evaluate_value(res) is True:
                    logger.debug("Fast path DOM check succeeded.")
                    return
            except Exception:
                pass

            # 2. Slow path: check rendered text word count
            visible_text = await self.get_visible_text()

            # Fast-fail: if the page is an interstitial/error page, stop waiting immediately
            # instead of burning 30 seconds. Covers Upwork "We'll be right back",
            # unauthenticated nav shells, etc.
            if self._is_dead_end_page(visible_text):
                logger.warning("Dead-end/interstitial page detected — stopping wait early.")
                return

            if self._has_job_detail_content(visible_text):
                return

            html = await self.get_page_content()
            html_text = self._html_to_text(html)
            if self._has_job_detail_content(html_text):
                return

            last_text = visible_text or html_text
            await self._trigger_lazy_render()
            await asyncio.sleep(1)

        preview = re.sub(r"\s+", " ", last_text or "").strip()[:180]
        logger.warning(
            "Timed out waiting for job-detail text. "
            f"Last visible text was {len(last_text or '')} chars: {preview!r}"
        )

    async def _trigger_lazy_render(self) -> None:
        """Gently scroll the page to trigger lazy-rendered job sections."""
        if not self._tab:
            return

        script = """
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
        try:
            await self._tab.evaluate(script, return_by_value=True)
        except Exception as e:
            logger.debug(f"Lazy-render scroll failed: {e}")

    async def _auto_expand_collapsed_sections(self) -> None:
        """
        Click all collapsed-content / 'Show more' controls in a single JavaScript
        injection instead of querying and clicking them one-by-one in Python.
        """
        if not self._tab:
            return

        script = """
        (() => {
            const cssSelectors = [
                '.air3-truncation-btn',
                '.air3-truncation-btn-link',
                '[data-ev-label*="more" i]',
                '[data-ev-label*="expand" i]',
                'button[aria-label*="Show more" i]',
                'button[aria-label*="Expand" i]',
                'button[class*="show-more" i]',
                'button[class*="truncation" i]',
                'a[class*="show-more" i]',
                'a[class*="truncation" i]',
                'span[class*="show-more" i]',
                'span[class*="truncation" i]',
                '[data-test*="show-more" i]',
                'button.up-truncation-btn',
                'button.up-btn-link'
            ];

            let count = 0;
            const clicked = new Set();
            // Helper function to check if clicking an element will cause navigation
            function willNavigate(element) {
                const anchor = element.closest('a');
                if (!anchor) return false;
                const href = anchor.getAttribute('href');
                if (!href) return false;
                if (href.startsWith('#') || href.startsWith('javascript:')) return false;
                return true;
            }

            // Phase 1: Try CSS exact match patterns first
            const explicitElements = document.querySelectorAll(cssSelectors.join(', '));
            for (const el of explicitElements) {
                if (el.offsetParent !== null && !clicked.has(el)) {
                    if (willNavigate(el)) continue;
                    try { el.click(); count++; clicked.add(el); } catch (e) {}
                }
            }

            // Phase 2: Fallback to aggressive text matching on ALL clickable tags
            const textLabels = new Set(['more', 'show more', 'read more', 'view more', 'see more']);
            const clickableTags = document.querySelectorAll('button, a, span, div, [role="button"]');
            
            for (const el of clickableTags) {
                if (clicked.has(el)) continue;
                if (willNavigate(el)) continue;
                
                // For aggressive text match, only match direct text content, not huge containers
                // e.g., if a whole card has "see more" inside it, we don't want to click the card itself
                const text = (el.innerText || el.textContent || '').toLowerCase().trim();
                if (textLabels.has(text) && el.offsetParent !== null) {
                    try { el.click(); count++; clicked.add(el); } catch (e) {}
                }
            }
            
            return count;
        })();
        """

        try:
            clicked = await self._tab.evaluate(script)
            clicked = self._extract_evaluate_value(clicked)
            if clicked:
                await asyncio.sleep(0.5)
                logger.info(f"Expanded {clicked} collapsed section(s) via JS injection.")
        except Exception as e:
            logger.debug(f"Auto-expand JS injection failed: {e}")

    @staticmethod
    def _is_invalid_content(content: Any) -> bool:
        if content is None:
            return True
        text = str(content).strip().lower()
        if text in ("", "undefined", "null", "none"):
            return True
        text_without_tags = re.sub(r"<[^>]+>", " ", text)
        text_without_tags = re.sub(r"\s+", " ", text_without_tags).strip()
        
        # If the page only has a title but the body hasn't loaded (common in SPAs like Upwork),
        # the text will be very short. Wait until we have at least a few paragraphs of text.
        word_count = len(text_without_tags.split())
        return word_count < 30

    @staticmethod
    def _has_job_detail_content(content: Any) -> bool:
        """Return True when text looks like an actual job page, not just the title shell."""
        if content is None:
            return False

        text = re.sub(r"\s+", " ", str(content)).strip().lower()
        if text in ("", "undefined", "null", "none"):
            return False

        signals = (
            "summary",
            "description",
            "responsibilities",
            "requirements",
            "qualifications",
            "skills and expertise",
            "mandatory skills",
            "preferred qualifications",
            "hourly",
            "duration",
            "experience level",
            "project type",
            "remote job",
            "about the client",
            "posted",
            "less than",
            "$",
        )
        signal_count = sum(1 for signal in signals if signal in text)
        word_count = len(text.split())
        if word_count >= 120 and signal_count >= 1:
            return True
        return word_count >= 35 and signal_count >= 2

    @staticmethod
    def _html_to_text(content: Any) -> str:
        if content is None:
            return ""
        text = str(content)
        text = re.sub(r"<script\b[^<]*(?:(?!</script>)<[^<]*)*</script>", " ", text, flags=re.I)
        text = re.sub(r"<style\b[^<]*(?:(?!</style>)<[^<]*)*</style>", " ", text, flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _extract_evaluate_value(value: Any) -> Any:
        if isinstance(value, tuple) and value:
            value = value[0]
        if hasattr(value, "value") and value.value is not None:
            return value.value
        deep_value = getattr(value, "deep_serialized_value", None)
        if deep_value is not None and hasattr(deep_value, "value"):
            return deep_value.value
        if isinstance(value, dict) and "value" in value:
            return value["value"]
        return value

    @staticmethod
    def _is_dead_end_page(text: str | None) -> bool:
        """
        Return True if the visible text looks like an interstitial, error, or
        unauthenticated shell page — i.e., there is no point waiting for job
        content because this page will never have any.

        Examples: Upwork "We'll be right back", anonymous nav shells with
        "Log in Sign up" but no real content.
        """
        if not text:
            return False
        sample = text[:1500].lower()

        dead_end_signals = (
            "we'll be right back",
            "this page is offline right now",
            "check back later",
        )
        # Fire only if at least one hard dead-end signal is present
        # AND the text is too short to be a real job page
        word_count = len(text.split())
        has_dead_end = any(s in sample for s in dead_end_signals)
        return has_dead_end and word_count < 200


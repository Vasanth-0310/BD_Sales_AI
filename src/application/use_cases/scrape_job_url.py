from datetime import datetime
from pathlib import Path
from src.domain.interfaces.extractor.i_extractor import IExtractor
from src.domain.interfaces.session_store.i_session_store import ISessionStore
from src.domain.enums.scrape_status import AuthStatus
from src.domain.value_objects.domain_url import DomainURL
from src.application.dto.scrape_request_dto import ScrapeRequestDTO, JobResultDTO
from src.application.services.auth_detection.detection_engine import AuthDetectionEngine
from src.application.services.session.session_manager import SessionManager
from src.infrastructure.browser.browser_factory import BrowserFactory
from src.infrastructure.browser.browser_pool import BrowserPool
from src.infrastructure.browser.curl_cffi_fetcher import CurlCFFIFetcher
from src.infrastructure.html_processing.html_cleaner import HTMLCleaner
from src.infrastructure.ai.gemini_company_profiler import GeminiCompanyProfiler
from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger(__name__)

# HTTP statuses where retrying through a real browser may plausibly help
# (bot-blocking / transient server issues). Any other non-200 is treated as
# a definitive failure instead of burning a browser launch on it.
_RETRY_STATUS_CODES = {403, 408, 429, 500, 502, 503, 504}


def _is_cloudflare_page(html: str) -> bool:
    """Return True if the HTML looks like a Cloudflare or platform bot-verification page."""
    # Challenge markers are injected at the very top of the document; scan a
    # larger head slice than before so late-injected markers aren't missed.
    sample = html[:20000].lower()
    strong_markers = (
        "just a moment",
        "additional verification required",
        "cf-browser-verification",
        "challenge - upwork",          # Upwork's custom Turnstile page
        "troubleshooting cloudflare errors",
        "verification successful. waiting for",
    )
    if any(marker in sample for marker in strong_markers):
        return True
    # F3: "ray id" alone is NOT a challenge signal — every standard Cloudflare
    # ERROR page (404/410/5xx for dead links) carries it too. Requiring an
    # explicit verifying/challenge phrase prevents dead job postings from
    # being misclassified as bot-blocks (which previously produced a bogus
    # auth_required / "re-capture your session" response).
    return "cloudflare" in sample and (
        "verifying" in sample or "checking your browser" in sample
    )


class ScrapeJobURL:
    """
    Primary use case: Orchestrates the full job scraping pipeline for a single URL.

    Flow:
        1. Validate URL → DomainURL value object.
        2. Check session store for existing valid session.
        3a. Try curl_cffi for pages that do not require browser rendering.
        3b. If curl_cffi fails, fall back to browser automation:
            nodriver persistent profile first for session-captured domains
            (headed) -> CloakBrowser pool (headless) -> BrowserFactory.
        4. Run auth detection (browser path only).
        5. Clean HTML → detect external apply link → follow max 1 redirect.
        6. Extract structured job data via Gemini.
        7. Return JobResultDTO.
    """

    def __init__(
        self,
        extractor: IExtractor,
        session_store: ISessionStore,
        company_profiler: GeminiCompanyProfiler | None = None,
    ) -> None:
        self._extractor = extractor
        self._company_profiler = company_profiler or GeminiCompanyProfiler()
        self._session_manager = SessionManager(session_store)
        self._auth_engine = AuthDetectionEngine()
        self._html_cleaner = HTMLCleaner()

    async def execute(self, request: ScrapeRequestDTO) -> JobResultDTO:
        """
        Execute the scrape pipeline.

        Args:
            request: ScrapeRequestDTO with user_id and raw URL.

        Returns:
            JobResultDTO with extraction result or auth-required signal.
        """
        # Step 1: Validate URL
        try:
            domain_url = DomainURL(request.url)
        except ValueError:
            return JobResultDTO.failed(reason=f"Invalid URL: '{request.url}'")

        domain = domain_url.domain
       
        logger.info(f"Scrape requested by user '{request.user_id}' for domain '{domain}'")

        # Step 2: Check for an existing session.
        # F1: this sits OUTSIDE the structured-failure net below — a Mongo
        # outage or a corrupt session document must degrade gracefully to a
        # cookie-less scrape instead of hanging ~30s and raising a bare 500.
        try:
            session = await self._session_manager.get_valid_session(request.user_id, domain)
        except Exception as sess_err:
            logger.error(
                f"Session lookup failed for '{domain}' "
                f"([{type(sess_err).__name__}] {sess_err}). Continuing WITHOUT session."
            )
            session = None
        storage_state = session.storage_state if session else None
        session_cookies = session.cookies if session else []

        # ── Step 3a: Try curl_cffi first ─────────────────────────────────────────
        # curl_cffi handles pages that can be fetched without browser rendering.
        page_html = ""
        cleaned_curl_text = ""
        status_code = 200
        current_url = domain_url.normalized_url
        used_curl = False
        has_persistent_profile = self._has_usable_nodriver_profile(storage_state)
        requires_browser_rendering = (
            has_persistent_profile
            or "upwork.com" in domain
            or "linkedin.com" in domain
            or "indeed.com" in domain
        )

        try:
            if requires_browser_rendering:
                raise RuntimeError("browser-rendered page required for this domain/session")

            fetcher = CurlCFFIFetcher()
            page_html, status_code, current_url = await fetcher.fetch(
                domain_url.normalized_url,
                cookies=session_cookies,
            )

            if _is_cloudflare_page(page_html):
                logger.warning(f"curl_cffi got a Cloudflare page for '{domain}'. Falling back to browser.")
                raise RuntimeError("Cloudflare page returned to curl_cffi")

            # Trigger browser fallback if:
            # 1. The server returned a retry-worthy non-200 status (403, 429, 503...)
            #    — this means curl_cffi was likely blocked and a real browser may pass.
            # 2. Any other non-200 status is a definitive failure (e.g. 404/410) —
            #    scraping an error page would only produce garbage extraction.
            # 3. The cleaned content is nearly empty even on HTTP 200
            #    — this means the page is a JS-rendered SPA that curl_cffi can't render.
            if status_code != 200:
                if status_code in _RETRY_STATUS_CODES:
                    logger.warning(
                        f"curl_cffi got HTTP {status_code} for '{domain}' — likely blocked. "
                        f"Falling back to browser."
                    )
                    raise RuntimeError(f"HTTP {status_code} returned by curl_cffi; browser fallback required")
                logger.warning(
                    f"curl_cffi got HTTP {status_code} for '{domain}' — not retry-worthy. Failing."
                )
                return JobResultDTO.failed(
                    reason=f"'{domain}' returned HTTP {status_code}; the job page could not be fetched."
                )

            cleaned_curl_text = self._html_cleaner.clean(page_html)
            if len(cleaned_curl_text.split()) < 80:
                logger.warning(
                    f"curl_cffi got a near-empty page for '{domain}' "
                    f"({len(cleaned_curl_text.split())} words) — likely a JS-rendered SPA. Falling back to browser."
                )
                raise RuntimeError("JS-rendered SPA detected; browser fallback required")


            used_curl = True
            logger.info(
                f"curl_cffi fetch succeeded for '{domain}'. "
                f"Status: {status_code}, Content: {len(page_html)} chars"
            )

        except Exception as curl_err:
            logger.warning(f"curl_cffi failed ({curl_err}). Falling back to browser automation...")

        # ── Step 3b: Browser fallback (only if curl_cffi didn't work) ────────────
        if not used_curl:
            # F1: this call sits outside every structured-failure net. Its
            # engine loop handles launch/navigation errors internally, but an
            # unexpected escape (asyncio timeout, OS error on a malformed
            # profile path, etc.) must degrade to a structured FAILED — never
            # a raw HTTP 500.
            try:
                browser, fetch = await self._fetch_via_browsers(
                    url=domain_url.normalized_url,
                    domain=domain,
                    storage_state=storage_state,
                    has_persistent_profile=has_persistent_profile,
                )
            except Exception as fetch_err:
                logger.error(
                    f"Browser fallback crashed for '{domain}' "
                    f"([{type(fetch_err).__name__}] {fetch_err})."
                )
                return JobResultDTO.failed(
                    reason=(
                        f"Browser automation failed for '{domain}': "
                        f"[{type(fetch_err).__name__}] {fetch_err}"
                    )
                )
            if browser == "DEAD_LINK":
                # The posting is gone (HTTP 404/410) — not an auth problem.
                return JobResultDTO.failed(
                    reason=(
                        f"'{domain}' returned HTTP {fetch['status_code']}; "
                        f"the job posting appears to have been removed or expired."
                    )
                )
            if browser == "EXPIRED_REDIRECT":
                # LinkedIn redirected the expired posting to its job search.
                return JobResultDTO.failed(
                    reason=(
                        f"'{domain}' redirected this link to its job search — "
                        f"the posting has expired or been removed."
                    )
                )
            if fetch is None:
                login_url = self._auth_engine.get_login_url(domain)
                logger.warning(
                    f"All browser engines were blocked (Cloudflare/verification) "
                    f"for '{domain}'. Keeping session intact."
                )
                return JobResultDTO.auth_required(
                    domain=domain,
                    login_url=login_url,
                    message=(
                        f"'{domain}' showed a browser verification page on every engine. "
                        f"Your saved session was kept. Run `python capture_session.py`, "
                        f"open the same domain, complete the verification in the browser, "
                        f"press Enter only after the real page loads, then retry."
                    ),
                )

            cleaned_text = ""
            try:
                status_code = fetch["status_code"]
                page_html = fetch["page_html"]
                page_visible_text = fetch["visible_text"]
                current_url = fetch["current_url"]

                # The engine loop already produced the cleaned text (single
                # BeautifulSoup pass — don't re-parse 300KB of HTML here).
                cleaned_text = fetch.get("cleaned_text") or self._html_cleaner.clean(page_html)
                if self._word_count(page_visible_text) > self._word_count(cleaned_text):
                    cleaned_text = page_visible_text

                # ── Auth detection (browser path only) ──────────────────────────
                # Note: Cloudflare pages never reach here — the engine loop treats
                # them as "blocked" and retries with the next engine.
                auth_status = self._auth_engine.detect(domain, current_url, status_code, cleaned_text)

                if auth_status == AuthStatus.AUTH_REQUIRED:
                    login_url = self._auth_engine.get_login_url(domain)

                    if session:
                        if session.user_id == request.user_id:
                            await self._session_manager.invalidate(session.user_id, domain)
                            logger.warning(f"Session genuinely expired for '{domain}'. Invalidated.")
                            message = (
                                f"Your saved session for '{domain}' has expired. "
                                f"Run `python capture_session.py` to capture a new one."
                            )
                        else:
                            # This user was served a shared fallback session — it does not
                            # belong to them, so it must NOT be deleted on their behalf.
                            logger.warning(
                                f"Shared fallback session for '{domain}' looks expired — kept intact."
                            )
                            message = (
                                f"'{domain}' requires login. A shared saved session appears to "
                                f"have expired and was kept intact. Run `python capture_session.py` "
                                f"and log in under your own user id to save a fresh session."
                            )
                        return JobResultDTO.auth_required(
                            domain=domain,
                            login_url=login_url,
                            message=message,
                        )
                    else:
                        return JobResultDTO.auth_required(
                            domain=domain,
                            login_url=login_url,
                            message=(
                                f"'{domain}' requires login. "
                                f"Run `python capture_session.py` and log in manually "
                                f"to save your session. Then retry this request."
                            ),
                        )

                # Update session last_used — never let a transient Mongo blip
                # discard an already-successful fetch.
                if session:
                    session.mark_used()
                    try:
                        await self._session_manager.persist(session)
                    except Exception as persist_err:
                        logger.warning(
                            f"Session persist failed (non-fatal): [{type(persist_err).__name__}] {persist_err}"
                        )

            except Exception as e:
                # Auth detection / persistence / unexpected errors must surface as a
                # structured failure, not escape as an unhandled HTTP 500.
                logger.error(f"Scrape failed for '{request.url}': [{type(e).__name__}] {e}", exc_info=True)
                return JobResultDTO.failed(reason="Internal error: please check server logs.")
            finally:
                # Guarantees the pool tab (or the whole nodriver browser) is
                # released even when auth detection, persistence, or extraction
                # raises — pool-managed adapters close only the tab.
                await self._close_quietly(browser)

            # ── Extraction runs AFTER the browser tab has been released so slow
            # Gemini calls (extraction + company profiling) never hold pool capacity.
            final_url = current_url
            self._write_debug_text(domain, cleaned_text)
            if self._is_invalid_scrape_text(cleaned_text):
                return JobResultDTO.failed(
                    reason=(
                        f"Scraped page for '{domain}' did not contain readable job content. "
                        f"Final URL: {final_url}"
                    )
                )
            if self._is_non_job_interstitial(domain, cleaned_text):
                return JobResultDTO.failed(
                    reason=(
                        f"'{domain}' loaded an account/interstitial page instead of the job post. "
                        f"Final URL: {final_url or domain_url.normalized_url}"
                    )
                )
            try:
                job_details = await self._extractor.extract(cleaned_text)
                company_profile = await self._company_profiler.profile(job_details.company)
            except Exception as e:
                logger.error(f"Extraction failed for '{request.url}': [{type(e).__name__}] {e}", exc_info=True)
                return JobResultDTO.failed(reason="Internal error: please check server logs.")
            logger.info(f"Scrape successful (browser): '{job_details.title}' at '{job_details.company}'")
            return JobResultDTO.success(job_details=job_details, company_profile=company_profile)

        # ── Steps 4-6: curl_cffi path — extract directly from fetched HTML ───────
        try:
            # Same redirect identity check as the browser path: an expired
            # LinkedIn posting lands on the search page (200 OK, wrong job).
            if "expired_jd_redirect" in (current_url or "").lower():
                logger.warning(
                    f"curl_cffi was redirected to LinkedIn's job search "
                    f"(expired posting) for '{domain}'."
                )
                return JobResultDTO.failed(
                    reason=(
                        f"'{domain}' redirected this link to its job search — "
                        f"the posting has expired or been removed."
                    )
                )

            # For curl_cffi path, check if auth is needed (e.g. LinkedIn behind login)
            auth_status = self._auth_engine.detect(domain, current_url, status_code, cleaned_curl_text)

            if auth_status == AuthStatus.AUTH_REQUIRED:
                login_url = self._auth_engine.get_login_url(domain)
                if session:
                    if session.user_id == request.user_id:
                        await self._session_manager.invalidate(session.user_id, domain)
                        message = (
                            f"Your saved session for '{domain}' has expired. "
                            f"Run `python capture_session.py` to capture a new one."
                        )
                    else:
                        # Shared fallback session — never delete it on this user's behalf.
                        logger.warning(
                            f"Shared fallback session for '{domain}' looks expired — kept intact."
                        )
                        message = (
                            f"'{domain}' requires login. A shared saved session appears to "
                            f"have expired and was kept intact. Run `python capture_session.py` "
                            f"and log in under your own user id to save a fresh session."
                        )
                    return JobResultDTO.auth_required(
                        domain=domain,
                        login_url=login_url,
                        message=message,
                    )
                return JobResultDTO.auth_required(
                    domain=domain,
                    login_url=login_url,
                    message=(
                        f"'{domain}' requires login. "
                        f"Run `python capture_session.py` and log in manually "
                        f"to save your session. Then retry this request."
                    ),
                )

            if session:
                session.mark_used()
                try:
                    await self._session_manager.persist(session)
                except Exception as persist_err:
                    logger.warning(
                        f"Session persist failed (non-fatal): [{type(persist_err).__name__}] {persist_err}"
                    )

            # Extract from the fetched job detail page itself.
            cleaned_text = cleaned_curl_text
            self._write_debug_text(domain, cleaned_text)
            if self._is_invalid_scrape_text(cleaned_text):
                return JobResultDTO.failed(
                    reason=f"Fetched page for '{domain}' did not contain readable job content."
                )
            if self._is_non_job_interstitial(domain, cleaned_text):
                return JobResultDTO.failed(
                    reason=f"'{domain}' returned an interstitial page instead of the job post."
                )

            job_details = await self._extractor.extract(cleaned_text)
            company_profile = await self._company_profiler.profile(job_details.company)
            logger.info(f"Scrape successful (curl_cffi): '{job_details.title}' at '{job_details.company}'")
            return JobResultDTO.success(job_details=job_details, company_profile=company_profile)

        except Exception as e:
            logger.error(f"Scrape failed for '{request.url}': [{type(e).__name__}] {e}", exc_info=True)
            return JobResultDTO.failed(reason="Internal error: please check server logs.")

    @staticmethod
    def _write_debug_text(domain: str, cleaned_text: str) -> None:
        """Persist the exact text sent to the extractor when DEBUG is enabled."""
        if not settings.debug:
            return

        try:
            safe_domain = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in domain)
            debug_dir = Path(".debug_scrapes")
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_path = debug_dir / f"{safe_domain}_{timestamp}.txt"
            debug_path.write_text(cleaned_text, encoding="utf-8")
            logger.info(f"Saved cleaned scrape text for debugging: {debug_path}")
        except Exception as e:
            logger.debug(f"Failed to write cleaned scrape debug text: {e}")

    @staticmethod
    def _has_usable_nodriver_profile(storage_state: dict | None) -> bool:
        """Return True only when the saved nodriver profile path exists and has Chrome state."""
        if not storage_state:
            return False

        raw_profile_dir = storage_state.get("nodriver_profile_dir") or storage_state.get("profile_dir")
        if not raw_profile_dir:
            return False

        try:
            profile_path = Path(str(raw_profile_dir)).resolve()
            # F2: exists() can be True for a FILE or an unreadable directory
            # (sync placeholders, permission issues) — iterdir() would raise
            # and escape as a bare 500. Any such failure means "no usable
            # profile": fall through to the cookie-based engines instead.
            if not profile_path.is_dir():
                return False
            return any(profile_path.iterdir())
        except OSError as e:
            logger.warning(
                f"Saved nodriver profile '{raw_profile_dir}' unreadable ({e}) "
                f"— treating as unusable."
            )
            return False

    async def _fetch_via_browsers(
        self,
        url: str,
        domain: str,
        storage_state: dict | None,
        has_persistent_profile: bool,
    ):
        """Try browser engines in priority order until one returns readable content.

        Engine order:
        - With a persistent nodriver profile (session-captured domains like
          Upwork): nodriver (headed) FIRST — the real Chrome profile carries
          login state and cf_clearance through integrity checks. CloakBrowser
          pool is the fallback.
        - Without a profile: BrowserPool (CloakBrowser, headless) first —
          fast tab (~200ms), session injected via storage_state cookies.

        A Cloudflare/verification page or empty content marks the engine as
        blocked and the next engine is tried. Returns ``(browser, fetch)``
        on success or ``(None, None)`` when every engine was blocked.
        """
        from src.infrastructure.browser.nodriver_adapter import NodriverAdapter

        async def _pool_launcher():
            browser = await BrowserPool.acquire(storage_state=storage_state)
            logger.info("BrowserPool: Acquired tab from persistent Chromium instance.")
            return browser

        async def _nodriver_launcher():
            # Prefer a warm tab from the NodriverPool — a cold uc.start()
            # costs 1-2s and hard-kills Chrome afterwards (cookie-flush loss).
            # Chrome locks the profile dir, so ALL nodriver use of a profile
            # must go through the pool; only fall back to a cold start when
            # the pool itself fails.
            if storage_state:
                raw_profile = (
                    storage_state.get("nodriver_profile_dir")
                    or storage_state.get("profile_dir")
                )
                if raw_profile and Path(str(raw_profile)).resolve().exists():
                    try:
                        from src.infrastructure.browser.nodriver_pool import NodriverPool

                        return await NodriverPool.acquire(
                            str(Path(str(raw_profile)).resolve()), storage_state
                        )
                    except Exception as pool_err:
                        logger.warning(
                            f"NodriverPool acquire failed ({pool_err}). "
                            f"Cold-starting nodriver."
                        )
            browser = NodriverAdapter()
            await browser.launch(
                headless=settings.browser_nodriver_headless,
                storage_state=storage_state,
            )
            return browser

        async def _factory_launcher():
            browser = await BrowserFactory.launch_browser(
                headless=settings.browser_cloak_headless,
                storage_state=storage_state,
            )
            # F5: never let a last-resort cold launch taskkill the pool's warm
            # Chrome (it matches the same profile path).
            if isinstance(browser, NodriverAdapter):
                browser._skip_orphan_kill = True
            return browser

        engines: list[tuple[str, object]] = []
        if has_persistent_profile:
            engines.append(("nodriver-profile", _nodriver_launcher))
        engines.append(("cloakbrowser-pool", _pool_launcher))
        engines.append(("browser-factory", _factory_launcher))

        for engine_name, launcher in engines:
            browser = None
            try:
                browser = await launcher()
            except Exception as launch_err:
                logger.warning(
                    f"Engine '{engine_name}' failed to launch "
                    f"([{type(launch_err).__name__}] {launch_err}). Trying next engine..."
                )
                continue

            try:
                nav_status = await browser.navigate(url)
                page_html = await browser.get_page_content()
                visible_text = await self._get_browser_visible_text(browser)
                current_url = await browser.get_current_url()
                # Clean once here so the empty-content check below operates on
                # real text — raw HTML whitespace-splitting never looks "empty".
                cleaned_html = self._html_cleaner.clean(page_html)
                fetch = {
                    "status_code": nav_status,
                    "page_html": page_html,
                    "visible_text": visible_text,
                    "current_url": current_url,
                    "cleaned_text": cleaned_html,
                }
            except Exception as nav_err:
                logger.warning(
                    f"Engine '{engine_name}' navigation failed "
                    f"([{type(nav_err).__name__}] {nav_err}). Trying next engine..."
                )
                await self._close_quietly(browser)
                continue

            # ── Redirect identity check (LinkedIn expired postings) ────────
            # LinkedIn 200-redirects an EXPIRED job's /jobs/view/{id} URL to
            # its jobs-search page (?trk=expired_jd_redirect), where an
            # unrelated job is pre-selected. The final URL is the
            # deterministic, Google-guaranteed signal — no other engine will
            # do better, so abort immediately with a truthful FAILED.
            if "expired_jd_redirect" in (fetch["current_url"] or "").lower():
                logger.warning(
                    f"Engine '{engine_name}' was redirected to LinkedIn's job "
                    f"search (expired posting) for '{domain}'."
                )
                await self._close_quietly(browser)
                return "EXPIRED_REDIRECT", fetch

            blocked_reason = None
            if _is_cloudflare_page(fetch["page_html"]):
                blocked_reason = "a Cloudflare verification page"
            elif self._is_invalid_scrape_text(fetch["cleaned_text"]):
                blocked_reason = "empty/unreadable content"

            if blocked_reason:
                # F3: 404/410 from the browser means the posting itself is gone —
                # no other engine will do better. Return immediately so the caller
                # produces a clean FAILED instead of looping through every engine
                # and mislabeling a dead link as auth_required.
                if fetch["status_code"] in (404, 410):
                    logger.warning(
                        f"Engine '{engine_name}' got HTTP {fetch['status_code']} "
                        f"for '{domain}' — job posting appears to be gone."
                    )
                    await self._close_quietly(browser)
                    return "DEAD_LINK", fetch
                logger.warning(
                    f"Engine '{engine_name}' got {blocked_reason}. "
                    f"Trying next engine..."
                )
                self._dump_failed_html(domain, fetch["page_html"])
                await self._close_quietly(browser)
                continue

            logger.info(f"Engine '{engine_name}' fetched readable content.")
            return browser, fetch

        return None, None

    def _dump_failed_html(self, domain: str, page_html: str) -> None:
        """Save blocked/empty page HTML for debugging (why did the engine fail)."""
        try:
            debug_path = Path(".debug_scrapes")
            debug_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            debug_file = debug_path / f"failed_{domain}_{timestamp}.txt"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(self._html_cleaner.clean(page_html))
            logger.warning(f"Saved blocked scrape text for debugging: {debug_file}")
        except Exception as e:
            logger.debug(f"Failed to write blocked scrape debug text: {e}")

    async def _close_quietly(self, browser) -> None:
        """Close a browser adapter without letting cleanup errors escape.

        For pool-managed adapters (both CloakBrowser pool and NodriverPool),
        delegates to the pool's release() so the window is hidden again.
        """
        try:
            if getattr(browser, "_pool_managed", False):
                # Only nodriver adapters carry _user_data_dir — use getattr so
                # CloakBrowser/Patchright pool tabs (which lack the attribute)
                # reach BrowserPool.release() instead of raising here and
                # leaking the tab.
                if getattr(browser, "_user_data_dir", None):
                    # NodriverPool tab — close via the adapter (hides window)
                    await browser.close()
                else:
                    # CloakBrowser/Patchright pool tab — use pool release
                    from src.infrastructure.browser.browser_pool import BrowserPool
                    await BrowserPool.release(browser)
            elif hasattr(browser, "close"):
                await browser.close()
        except Exception as e:
            logger.warning(f"Error closing browser after scrape: {e}")

    @staticmethod
    def _is_invalid_scrape_text(value: str | None) -> bool:
        if value is None:
            return True
        text = str(value).strip()
        if text.lower() in ("", "undefined", "null", "none"):
            return True
        return len(text.split()) < 40

    @staticmethod
    async def _get_browser_visible_text(browser) -> str:
        getter = getattr(browser, "get_visible_text", None)
        if not getter:
            return ""
        try:
            return await getter()
        except Exception as e:
            logger.debug(f"Failed to read browser visible text: {e}")
            return ""

    @staticmethod
    def _word_count(value: str | None) -> int:
        if not value:
            return 0
        return len(str(value).split())

    @staticmethod
    def _is_non_job_interstitial(domain: str, text: str) -> bool:
        sample = text[:3000].lower()
        if "upwork.com" in domain:
            return (
                "we'll be right back" in sample
                or "this page is offline right now" in sample
                or "buy connects to apply" in sample and "summary" not in sample
            )
        if "linkedin.com" in domain:
            # Redirect markers for expired/closed job postings: LinkedIn
            # 200-redirects jobs/view/{id} of an EXPIRED posting to its jobs
            # SEARCH page (?trk=expired_jd_redirect), where a different,
            # unrelated job is pre-selected. Extracting that pane previously
            # returned a confident WRONG job as SUCCESS. Both markers were
            # verified against a live redirect capture.
            return (
                # NOTE: anchor past the apostrophe — LinkedIn renders "You're"
                # with a curly quote (U+2019) that a straight-quote match misses.
                "now using ai-powered job search" in sample
                or "no longer accepting applications" in sample
            )
        return False

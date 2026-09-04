"""
Mock end-to-end tests for the scrape pipeline — NO real network/browser/LLM.
Run: .venv\\Scripts\\python.exe scripts/mock_test_scrape.py
"""
import asyncio
import sys

sys.path.insert(0, ".")

from src.application.use_cases.scrape_job_url import ScrapeJobURL, _is_cloudflare_page
from src.application.dto.scrape_request_dto import ScrapeRequestDTO
from src.domain.value_objects.job_details import JobDetails
from src.domain.enums.scrape_status import AuthStatus, ScrapeStatus

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


JOB = JobDetails(
    title="Java Dev", company="ACME",
    required_proposal_questions=["Updated Resume11", "5 years exp needed"],
)


class FakeSessionManager:
    def __init__(self, session=None, boom=False):
        self.session = session
        self.boom = boom
        self.invalidated = None

    async def get_valid_session(self, user_id, domain):
        if self.boom:
            raise TimeoutError("mongo down")
        return self.session

    async def invalidate(self, user_id, domain):
        self.invalidated = user_id


class FakeSession:
    def __init__(self):
        from src.domain.entities.user_session import UserSession
        self.user_id = "default_user"
        self.website = "example.com"
        self.storage_state = {}
        self.cookies = []

    def mark_used(self):
        pass


class FakeAuthEngine:
    def __init__(self, status=AuthStatus.PUBLIC):
        self.status = status
        self.invalidated_checks = []

    def detect(self, domain, url, status_code, text):
        return self.status

    def get_login_url(self, domain):
        return f"https://{domain}/login"


class FakeExtractor:
    async def extract(self, text):
        from src.infrastructure.ai.gemini_extractor import _sanitize_proposal_questions
        return _sanitize_proposal_questions(JOB)


class FakeProfiler:
    async def profile(self, company):
        return None


class FakeFetcher:
    """curl_cffi stand-in: returns canned (html, status, url)."""

    def __init__(self, html="<html>job</html>", status=200):
        self.html = html
        self.status = status

    async def fetch(self, url, cookies=None):
        return self.html, self.status, url


class FakeBrowser:
    _pool_managed = False
    closed = False

    async def navigate(self, url, wait_for_job_details=True):
        return 200

    async def get_page_content(self):
        return "<html>" + "word " * 200 + "</html>"

    async def get_current_url(self):
        return "https://example.com/job"

    async def close(self):
        self.closed = True


def make_uc(fetcher, session_mgr, auth_engine=None):
    from src.infrastructure.html_processing.html_cleaner import HTMLCleaner

    uc = object.__new__(ScrapeJobURL)
    uc._session_manager = session_mgr
    uc._auth_engine = auth_engine or FakeAuthEngine()
    uc._extractor = FakeExtractor()
    uc._company_profiler = FakeProfiler()
    uc._html_cleaner = HTMLCleaner()
    return uc


async def run(uc, url="https://example.com/job/1"):
    return await uc.execute(ScrapeRequestDTO(user_id="u1", action="scrape", url=url))


async def main():
    import src.application.use_cases.scrape_job_url as mod

    # Force the browser path for all later tests — no real network in mocks.
    def _no_network_fetcher():
        raise RuntimeError("no network in mock test")
    mod.CurlCFFIFetcher = _no_network_fetcher

    print("== T1: normal curl_cffi success ==")
    fake = FakeFetcher("<html>" + "word " * 200 + "</html>", 200)
    uc = make_uc(fake, FakeSessionManager())
    orig = mod.CurlCFFIFetcher
    mod.CurlCFFIFetcher = lambda: fake
    r = await run(uc)
    mod.CurlCFFIFetcher = orig
    check("status SUCCESS", r.status == ScrapeStatus.SUCCESS, r.status)
    check("questions sanitized", r.job_details.required_proposal_questions == ["Updated Resume", "5 years exp needed"],
          str(r.job_details.required_proposal_questions))

    print("== T2: Mongo down -> graceful, no-session scrape still succeeds ==")
    fake2 = FakeFetcher("<html>" + "word " * 200 + "</html>", 200)
    uc2 = make_uc(fake2, FakeSessionManager(boom=True))
    orig = mod.CurlCFFIFetcher
    mod.CurlCFFIFetcher = lambda: fake2
    r2 = await run(uc2)
    mod.CurlCFFIFetcher = orig
    check("status SUCCESS despite Mongo outage", r2.status == ScrapeStatus.SUCCESS, r2.status)

    print("== T3: dead link (browser path 404) -> FAILED not AUTH_REQUIRED ==")
    results = {}

    class DeadBrowser(FakeBrowser):
        async def navigate(self, url, wait_for_job_details=True):
            return 404

        async def get_page_content(self):
            # CF error page: used to be misread as a challenge
            return ("<!DOCTYPE html><html><head><title>Attention Required</title></head>"
                    "<body>cloudflare Ray ID: 8abc something went wrong</body></html>")

    async def fake_fetch_via_browsers(self, **kw):
        return "DEAD_LINK", {"status_code": 404}

    uc3 = make_uc(None, FakeSessionManager())
    uc3._fetch_via_browsers = fake_fetch_via_browsers.__get__(uc3)
    r3 = await run(uc3)
    check("dead link -> FAILED", r3.status == ScrapeStatus.FAILED, r3.status)
    check("no re-capture advice", "capture_session" not in (r3.error_message or ""), r3.error_message)

    print("== T4: Cloudflare challenge detection precision ==")
    cf_error = "error 404 | cloudflare ray id: 8abc | performance by cloudflare"
    cf_challenge = "just a moment... please wait while we verify your browser"
    check("CF ERROR page NOT challenge", not _is_cloudflare_page(cf_error))
    check("CF CHALLENGE still detected", _is_cloudflare_page(cf_challenge))

    print("== T5: bare WAF 403 does NOT invalidate session ==")

    class SessionedMgr(FakeSessionManager):
        def __init__(self):
            super().__init__(session=None)
            s = FakeSession()
            s.user_id = "u1"  # own session
            self.session = s

    class QABrowser(FakeBrowser):
        """403 with verbose legit-looking page (passes >=80-word gate) but no login signals."""
        async def navigate(self, url, wait_for_job_details=True):
            return 403

    async def qab(self, **kw):
        b = QABrowser()
        return b, {
            "status_code": 403,
            "page_html": "<html></html>",
            "visible_text": ("We are sorry but your request looks unusual. Please retry later. " * 10),
            "current_url": "https://example.com/job/1",
            "cleaned_text": "Access denied this page is protected and your request looks unusual please try again shortly thanks " * 4,
        }

    mgr = SessionedMgr()
    # Use the REAL detection engine so the 403-corroboration fix is exercised
    # end-to-end (a fake would return AUTH_REQUIRED unconditionally).
    from src.application.services.auth_detection.detection_engine import AuthDetectionEngine
    uc5 = make_uc(None, mgr, auth_engine=AuthDetectionEngine())
    uc5._fetch_via_browsers = qab.__get__(uc5)
    r5 = await run(uc5)
    check("403-verbose handled without crash", r5 is not None)
    check("own session NOT invalidated on uncorroborated 403", mgr.invalidated is None,
          f"invalidated={mgr.invalidated}")

    print("== T6: genuine auth wall DOES invalidate own session ==")

    class AuthBrowser(QABrowser):
        async def navigate(self, url, wait_for_job_details=True):
            return 401

    async def ab(self, **kw):
        b = AuthBrowser()
        return b, {
            "status_code": 401,
            "page_html": "",
            "visible_text": "",
            "current_url": "https://example.com/login",
            "cleaned_text": "sign in to continue",
        }

    mgr6 = SessionedMgr()
    uc6 = make_uc(None, mgr6, auth_engine=FakeAuthEngine(status=AuthStatus.AUTH_REQUIRED))
    uc6._fetch_via_browsers = ab.__get__(uc6)
    r6 = await run(uc6)
    check("401 -> AUTH_REQUIRED", r6.status == ScrapeStatus.AUTH_REQUIRED, r6.status)
    check("own session invalidated on genuine auth wall", mgr6.invalidated == "u1",
          f"invalidated={mgr6.invalidated}")

    print("== T7: heuristic corroboration unit checks ==")
    from src.application.services.auth_detection.generic_heuristics import GenericHeuristics
    g = GenericHeuristics()
    long_waf_text = "your request looks automated please retry later " * 20
    check(
        "bare 403 verbose -> PUBLIC",
        g.check("https://x.com/job/1", 403, long_waf_text) == AuthStatus.PUBLIC,
    )
    check(
        "403 + login URL -> AUTH_REQUIRED",
        g.check("https://x.com/account/login?next=/job", 403, long_waf_text) == AuthStatus.AUTH_REQUIRED,
    )
    check(
        "bare 401 -> AUTH_REQUIRED",
        g.check("https://x.com/job/1", 401, long_waf_text) == AuthStatus.AUTH_REQUIRED,
    )

    print()
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())

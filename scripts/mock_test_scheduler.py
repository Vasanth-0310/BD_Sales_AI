"""
Mock tests for the SessionRefreshScheduler — no real Mongo/browsers/network.
Run: .venv\\Scripts\\python.exe scripts/mock_test_scheduler.py
"""
import asyncio
import sys
from datetime import datetime, timezone

sys.path.insert(0, ".")

from src.application.services.scheduler.session_refresh_scheduler import (
    SessionRefreshScheduler,
    _is_browser_verification_page,
)
from src.common.config import settings
from src.domain.entities.user_session import UserSession
from src.domain.enums.scrape_status import SessionStatus

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


class FakeStore:
    def __init__(self, sessions):
        self._sessions = sessions
        self.saved = []

    async def get_all_active_sessions(self):
        return list(self._sessions)

    async def save_session(self, session):
        self.saved.append(session)

    async def get_session(self, user_id, domain):
        return next((s for s in self._sessions if s.user_id == user_id), None)

    async def delete_session(self, user_id, domain):
        self._sessions = [s for s in self._sessions if s.user_id != user_id]


def make_session(user_id, domain, cookies=None):
    return UserSession(
        user_id=user_id,
        website=domain,
        storage_state={"cookies": cookies or [{"name": "sid", "value": "old"}]},
        cookies=cookies or [{"name": "sid", "value": "old"}],
    )


class FakeBrowser:
    """Pool-managed adapter stand-in (like BrowserPool.acquire returns)."""
    _pool_managed = True
    _page = None
    _context = None
    released = False
    closed_directly = False

    def __init__(self, html, status=200, url="https://example.com/"):
        self._html = html
        self._status = status
        self._url = url

    async def navigate(self, url, wait_for_job_details=False):
        return self._status

    async def get_page_content(self):
        return self._html

    async def get_current_url(self):
        return self._url

    async def capture_session(self):
        return {"cookies": [{"name": "sid", "value": "fresh"}]}

    async def close(self):
        self.closed_directly = True


class FakePoolAdapter(FakeBrowser):
    """Simulates a BrowserPool-issued tab adapter."""


async def setup_scheduler(monkeypool=True, sessions=None):
    store = FakeStore(sessions or [])
    sched = SessionRefreshScheduler(session_store=store)
    if monkeypool:
        import src.infrastructure.browser.browser_pool as bp

        class FakePool:
            _active_tabs = 0

            @staticmethod
            async def acquire(storage_state=None, if_idle: bool = False):
                raise NotImplementedError("patched per-test")

            @staticmethod
            async def release(adapter):
                adapter.released = True

        sched._fake_pool = FakePool
        bp.BrowserPool = FakePool
    return sched, store


HEALTHY_PAGE = "<html>" + "welcome to the dashboard feed content here " * 30 + "</html>"
VERIFY_PAGE = "<html><body>Just a moment... verifying you are human. cloudflare checking</body></html>"
LOGIN_PAGE = "<html><body>sign in to continue to your account</body></html>"


async def test_interval_config():
    print("== T1: cron interval is 3 hours ==")
    check("settings.session_refresh_interval_hours == 3",
          settings.session_refresh_interval_hours == 3,
          f"got {settings.session_refresh_interval_hours}")

    sched, store = await setup_scheduler()
    sched.start()
    try:
        job = sched._scheduler.get_job("session_refresh_job")
        check("job registered", job is not None)
        check("job interval == 3h",
              job.trigger.interval.total_seconds() == 3 * 3600,
              f"{job.trigger.interval}")
        check("max_instances == 1", job.max_instances == 1)
        check("coalesce enabled", job.coalesce is True)
        delta = (job.next_run_time - datetime.now(job.next_run_time.tzinfo)).total_seconds()
        check("next run ~3h away", 2.9 * 3600 < delta <= 3.01 * 3600, f"delta={delta}s")
    finally:
        sched.stop()


async def test_refresh_paths():
    print("== T2: healthy session gets cookies rotated + saved ==")
    s1 = make_session("u1", "example.com")
    sched, store = await setup_scheduler(sessions=[s1])

    fresh_browser = FakePoolAdapter(HEALTHY_PAGE)
    sched._fake_pool.acquire = staticmethod(
        lambda storage_state=None, if_idle=False: _ret(fresh_browser))
    _patch_pool_acquire(sched)

    await sched.refresh_all_active_sessions()
    check("session saved", len(store.saved) == 1, len(store.saved))
    check("cookies rotated to fresh",
          store.saved[0].cookies == [{"name": "sid", "value": "fresh"}],
          str(store.saved[0].cookies)[:80])
    check("still ACTIVE", store.saved[0].status == SessionStatus.ACTIVE)
    check("pool tab RELEASED (not adapter.close)", fresh_browser.released,
          f"released={fresh_browser.released} closed={fresh_browser.closed_directly}")

    print("== T3: verification page -> session kept but NOT falsely refreshed with new state ==")
    s2 = make_session("u2", "example.com")
    sched2, store2 = await setup_scheduler(sessions=[s2])
    vb = FakePoolAdapter(VERIFY_PAGE)
    sched2._fake_pool.acquire = staticmethod(
        lambda storage_state=None, if_idle=False: _ret(vb))
    _patch_pool_acquire(sched2)
    await sched2.refresh_all_active_sessions()
    check("session saved unchanged (kept)", len(store2.saved) == 1)
    check("state NOT overwritten with challenge-page state",
          store2.saved[0].cookies == [{"name": "sid", "value": "old"}],
          str(store2.saved[0].cookies)[:80])

    print("== T4: genuine login wall -> session marked EXPIRED ==")
    s3 = make_session("u3", "example.com")
    sched3, store3 = await setup_scheduler(sessions=[s3])
    lb = FakePoolAdapter(LOGIN_PAGE, url="https://example.com/login?next=/x")
    sched3._fake_pool.acquire = staticmethod(
        lambda storage_state=None, if_idle=False: _ret(lb))
    _patch_pool_acquire(sched3)
    await sched3.refresh_all_active_sessions()
    check("session marked EXPIRED",
          store3.saved and store3.saved[0].status == SessionStatus.EXPIRED,
          str([s.status for s in store3.saved]))

    print("== T5: verification-page detector precision ==")
    check("CF challenge detected", _is_browser_verification_page(VERIFY_PAGE))
    check("normal page not flagged", not _is_browser_verification_page(HEALTHY_PAGE))

    print("== T6: nodriver-pool adapter routes to close(), NOT BrowserPool.release ==")
    s4 = make_session("u4", "example.com")
    sched4, store4 = await setup_scheduler(sessions=[s4])

    class FakeNodriverPoolAdapter:
        """Mimics NodriverAdapter from NodriverPool.acquire: pool-managed,
        has _user_data_dir + _tab, NO _page/_context (Playwright attrs)."""
        _pool_managed = True
        _user_data_dir = r"C:\fake\profiles\default_user\example.com"
        _page = None
        _context = None
        close_called = False
        released = False

        async def navigate(self, url, wait_for_job_details=False):
            return 200

        async def get_page_content(self):
            return HEALTHY_PAGE

        async def get_current_url(self):
            return "https://example.com/"

        async def capture_session(self):
            return {"cookies": [{"name": "sid", "value": "fresh"}]}

        async def close(self):
            self.close_called = True  # pooled nodriver path: tab close + notify

    nb = FakeNodriverPoolAdapter()
    sched4._fake_pool.acquire = staticmethod(lambda storage_state=None, if_idle=False: _ret(nb))
    _patch_pool_acquire(sched4)
    await sched4.refresh_all_active_sessions()
    check("nodriver tab closed via close()", nb.close_called, f"closed={nb.close_called}")
    check("nodriver NOT sent to BrowserPool.release", not nb.released, f"released={nb.released}")
    check("session still refreshed", len(store4.saved) == 1, len(store4.saved))


def _ret(v):
    async def _inner():
        return v
    return _inner()


def _patch_pool_acquire(sched):
    """Point the scheduler's pool import at the fake pool type."""
    import src.infrastructure.browser.browser_pool as bp
    sched._orig_pool_cls = bp.BrowserPool

    class _Shim:
        _active_tabs = 0

        @staticmethod
        async def acquire(storage_state=None, if_idle: bool = False):
            return await sched._fake_pool.acquire(
                storage_state=storage_state, if_idle=if_idle
            )

        @staticmethod
        async def release(adapter):
            await sched._fake_pool.release(adapter)

    bp.BrowserPool = _Shim


async def main():
    await test_interval_config()
    await test_refresh_paths()
    print()
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())

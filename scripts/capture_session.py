"""
capture_session.py — Manual Session Capture Tool
=================================================

Run this script ONCE per platform to capture and save your browser session
to MongoDB. After that, the scraper reuses it automatically.

Usage:
    python capture_session.py

The script will ask for:
    - Login page URL  (e.g. https://www.linkedin.com/login)
    - Domain name     (e.g. linkedin.com)

A real browser window will open. Log in however you want
(password, 2FA, Google SSO — anything). When the URL changes away from
the login page, the session is captured and saved to MongoDB.
"""

import asyncio
import sys
import os
import warnings
from pathlib import Path

# Suppress harmless async transport cleanup warnings on script exit
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", message=".*closed pipe.*")

# Ensure the project root is on sys.path so src.* imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio

# ── Windows: must set ProactorEventLoop BEFORE any async imports ──────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
# ──────────────────────────────────────────────────────────────────────────────

from src.common.config import settings
from src.infrastructure.mongodb import connection as mongo
from src.infrastructure.mongodb.session_repository import MongoDBSessionRepository
from src.infrastructure.browser.nodriver_adapter import NodriverAdapter
from src.application.services.session.session_manager import SessionManager

# Login-related keywords — if the current URL contains any of these,
# we consider the user still on the login/auth page.
_LOGIN_KEYWORDS = ["login", "signin", "sign-in", "auth", "checkpoint", "challenge", "security"]

POLL_INTERVAL = 2.0   # seconds between URL checks
TIMEOUT       = 300.0 # 5 minutes max wait


def _profile_dir_for(user_id: str, domain: str) -> str:
    """Return a stable Chrome profile directory for this user/domain session."""
    safe_user_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in user_id)
    safe_domain = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in domain)
    profile_dir = Path(settings.browser_profile_root) / safe_user_id / safe_domain
    profile_dir.mkdir(parents=True, exist_ok=True)
    return str(profile_dir.resolve())


def _print_banner() -> None:
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║        BD Team Automation — Session Capture      ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("This tool opens a real browser so you can log in manually.")
    print("Your session is saved to MongoDB and reused by the scraper.")
    print()


def _prompt_inputs() -> tuple[str, str, str]:
    """Interactively ask the user for the login URL, domain, and capture method."""
    login_url = input("  Enter the login page URL : ").strip()
    domain    = input("  Enter the domain name    : ").strip()

    if not login_url or not domain:
        print("\n❌  Both URL and domain are required. Please try again.\n")
        sys.exit(1)

    # Basic cleanup — strip trailing slashes from domain
    domain = domain.rstrip("/").lower()
    # Strip protocol from domain if accidentally included
    domain = domain.replace("https://", "").replace("http://", "").split("/")[0]

    print("\nHow would you like to capture the session?")
    print("  1. Automated Browser (Opens a new window)")
    print("  2. Manual Cookie Paste (cookies only; less reliable)")
    method = input("\n  Select method (1/2): ").strip()
    return login_url, domain, method


async def _wait_for_user_done() -> None:
    """Waits for the user to press Enter in the terminal."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        input,
        "  ✋  Log in to the site in the browser window.\n"
        "      NOTE: If you see a 'Just a moment...' Cloudflare screen, just wait\n"
        "      for it to pass automatically — then log in normally.\n"
        "      When you are FULLY logged in, come back here and press Enter...",
    )


async def capture(login_url: str, domain: str, method: str, user_id: str = "default_user") -> None:
    """Full capture flow."""
    print(f"\n  Connecting to MongoDB...")
    await mongo.connect(settings.mongodb_uri, settings.mongodb_db_name)
    session_repo = MongoDBSessionRepository()
    session_mgr  = SessionManager(session_repo)
    print(f"  ✅  MongoDB connected.")

    if method == "2":
        print("\n  [Manual Cookie Paste]")
        print("  1. Open your OWN regular browser (Chrome/Edge/Firefox).")
        print(f"  2. Go to {login_url} and log in.")
        print("  3. Open Developer Tools (F12) -> Network tab.")
        print("  4. Refresh the page.")
        print("  5. Click on the first document request (e.g. 'jobs' or the page name).")
        print("  6. Scroll down to 'Request Headers' and find the 'cookie:' string.")
        print("  7. Right-click the cookie string value and copy it.")
        cookie_string = input("\n  Paste the raw cookie string here: ").strip()

        if not cookie_string:
            print("\n  ❌ No cookies provided. Exiting.")
            await mongo.disconnect()
            return
            
        # Parse raw cookie string into storage_state format
        cookies = []
        for pair in cookie_string.split(";"):
            pair = pair.strip()
            if "=" in pair:
                name, val = pair.split("=", 1)
                cookies.append({
                    "name": name,
                    "value": val,
                    "domain": domain,
                    "path": "/",
                    "sameSite": "Lax",
                    "secure": True,
                    "expires": -1
                })
        
        storage_state = {"cookies": cookies, "origins": []}
        session = await session_mgr.capture_from_storage(
            user_id=user_id,
            domain=domain,
            storage_state=storage_state,
        )
        await session_mgr.persist(session)
        print(f"\n╔══════════════════════════════════════════════════╗")
        print(f"║  ✅  Session for '{domain}' saved manually!      ")
        print(f"║      {len(cookies)} cookies parsed and saved.    ")
        print(f"║      The scraper will now use this session.      ")
        print(f"╚══════════════════════════════════════════════════╝\n")
        await mongo.disconnect()
        return

    # Method 1: Automated Browser
    profile_dir = _profile_dir_for(user_id, domain)
    browser = NodriverAdapter(user_data_dir=profile_dir)
    try:
        print(f"\n  Opening browser at: {login_url}")
        print(f"  Using browser profile: {profile_dir}")
        print(f"  👉  Log in to '{domain}' in the browser window that opens.")
        print(f"  ℹ️   If you see a 'Just a moment...' screen, wait for it to pass.\n")
        await browser.launch(headless=False)

        # Try to navigate to the login URL. If this fails (e.g. Chrome
        # fails to connect, network error), the browser is still open —
        # the user can manually paste the URL in the address bar.
        try:
            await browser.navigate(login_url, wait_for_job_details=False)
            # Give the page a moment to start rendering
            await asyncio.sleep(1)
        except Exception as nav_err:
            print(f"  ⚠️  Could not auto-navigate to the URL: {nav_err}")
            print(f"  👉  Please manually paste this URL in the browser address bar:")
            print(f"      {login_url}\n")

        await _wait_for_user_done()
        print()

        final_url = await browser.get_current_url()
        print(f"  ✅  Capturing session from: {final_url}")
        storage_state = await browser.capture_session()

        cookie_count = len(storage_state.get("cookies", []))
        if cookie_count == 0:
            print(f"\n  ⚠️  No cookies captured! Make sure you are fully logged in.")
            print(f"      Try running the script again.\n")
        else:
            session = await session_mgr.capture_from_storage(
                user_id=user_id,
                domain=domain,
                storage_state=storage_state,
            )
            await session_mgr.persist(session)

            print(f"\n╔══════════════════════════════════════════════════╗")
            print(f"║  ✅  Session for '{domain}' saved successfully!  ")
            print(f"║      {cookie_count} cookies captured.")
            print(f"║      The scraper will now use this session automatically.")
            print(f"╚══════════════════════════════════════════════════╝\n")

    except TimeoutError as e:
        print(f"\n  ❌  {e}")
        print("      Please run the script again and complete the login faster.\n")
    except Exception as e:
        print(f"\n  ❌  An error occurred: {e}\n")
    finally:
        await browser.close()
        await mongo.disconnect()


def main() -> None:
    _print_banner()
    login_url, domain, method = _prompt_inputs()
    print()
    asyncio.run(capture(login_url, domain, method))


if __name__ == "__main__":
    main()

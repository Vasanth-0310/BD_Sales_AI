"""Hide headed browser windows from the Windows taskbar.

The pool browsers stay HEADED (full Chrome rendering and fingerprint —
headless=False); only the window is hidden via ShowWindow(SW_HIDE), which
removes its taskbar button while the process keeps running normally.

Hidden (occluded) windows get throttled by Chrome by default, so callers
must launch with ANTI_THROTTLING_ARGS to keep rendering, timers, and
lazy-loading behaving exactly like a visible window.

Windows-only; every helper is a safe no-op on other platforms.
"""

import asyncio
import ctypes
import subprocess
import sys
from ctypes import WINFUNCTYPE, wintypes

from src.common.logger import get_logger

logger = get_logger(__name__)

SW_HIDE = 0
SW_SHOW = 5

# Keep hidden windows rendering at full fidelity. Without these, Chrome
# throttles timers and rAF for occluded windows and lazy content never loads.
ANTI_THROTTLING_ARGS = [
    "--disable-backgrounding-occluded-windows",
    "--disable-renderer-backgrounding",
    "--disable-background-timer-throttling",
]

if sys.platform == "win32":
    _user32 = ctypes.windll.user32
    _EnumWindowsProc = WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
else:
    _user32 = None
    _EnumWindowsProc = None


def _windows_of_pids(pids: set[int], visible_only: bool) -> list[int]:
    """Return top-level window handles owned by the given PIDs.

    Hidden windows are skipped when visible_only (they must be for hiding);
    showing must find them regardless of visibility.
    """
    if _user32 is None or not pids:
        return []

    targets: list[int] = []

    def _callback(hwnd: int, _lparam: int) -> bool:
        if visible_only and not _user32.IsWindowVisible(hwnd):
            return True
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value in pids:
            targets.append(hwnd)
        return True

    _user32.EnumWindows(_EnumWindowsProc(_callback), 0)
    return targets


def hide_windows_of_pids(pids: set[int]) -> int:
    """Hide every visible top-level window owned by the given PIDs."""
    hidden = 0
    for hwnd in _windows_of_pids(pids, visible_only=True):
        if _user32.ShowWindow(hwnd, SW_HIDE):
            hidden += 1
    return hidden


def show_windows_of_pids(pids: set[int]) -> int:
    """Un-hide (pop back up) the windows of the given PIDs."""
    if _user32 is None or not pids:
        return 0
    shown = 0
    for hwnd in _windows_of_pids(pids, visible_only=False):
        if _user32.ShowWindow(hwnd, SW_SHOW):
            shown += 1
    return shown


async def hide_windows_when_visible(pids: set[int], timeout_s: float = 8.0) -> int:
    """Poll until the process creates its first window, then hide it.

    A browser's window appears asynchronously after launch — this waits for
    it (up to timeout_s) so the taskbar button never lingers.
    """
    if _user32 is None or not pids:
        return 0

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        hidden = hide_windows_of_pids(pids)
        if hidden:
            logger.info(f"Hid {hidden} browser window(s) from the taskbar (pids={sorted(pids)}).")
            return hidden
        await asyncio.sleep(0.2)
    logger.warning(f"No visible window found for pids={sorted(pids)} to hide.")
    return 0


def pids_with_cmdline_marker(marker: str) -> set[int]:
    """Return PIDs of chrome.exe processes whose command line contains marker.

    Playwright-based engines (CloakBrowser/Patchright) don't expose their
    browser PID, so the pool launches them with a unique harmless argument
    (e.g. --bd-pool-instance) that we can match against here.
    """
    if sys.platform != "win32":
        return set()
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                f"Get-CimInstance Win32_Process -Filter \"Name='chrome.exe'\" "
                f"| Where-Object {{ $_.CommandLine -like '*{marker}*' }} "
                f"| Select-Object -ExpandProperty ProcessId",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return {int(line) for line in result.stdout.split() if line.strip().isdigit()}
    except Exception as e:
        logger.warning(f"Marker PID lookup failed for '{marker}': {e}")
        return set()

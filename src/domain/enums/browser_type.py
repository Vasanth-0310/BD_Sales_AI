from enum import Enum


class BrowserType(str, Enum):
    """Available browser automation backends."""
    PATCHRIGHT = "PATCHRIGHT"   # Primary — stealth-patched Playwright
    NODRIVER = "NODRIVER"       # Fallback — CDP-direct, no WebDriver binary

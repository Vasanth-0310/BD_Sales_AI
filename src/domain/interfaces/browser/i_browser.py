from abc import ABC, abstractmethod
from typing import Any


class IBrowser(ABC):
    """
    Port (Interface) for browser automation backends.
    Implemented by CloakBrowserAdapter (primary) and PatchrightAdapter (fallback).
    """

    @abstractmethod
    async def launch(self, headless: bool = True, storage_state: dict[str, Any] | None = None) -> None:
        """Launch the browser. Optionally inject a session storage state."""
        ...

    @abstractmethod
    async def navigate(self, url: str, wait_for_job_details: bool = True) -> int:
        """
        Navigate to the given URL.
        Returns the HTTP status code of the response.
        """
        ...

    @abstractmethod
    async def get_page_content(self) -> str:
        """Return the full rendered HTML of the current page."""
        ...

    @abstractmethod
    async def get_current_url(self) -> str:
        """Return the current URL (after any client-side redirects)."""
        ...

    @abstractmethod
    async def capture_session(self) -> dict[str, Any]:
        """Capture and return the current browser storage state (cookies, localStorage, etc.)."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the browser and release all resources."""
        ...

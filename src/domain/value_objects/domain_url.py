from urllib.parse import urlparse
from dataclasses import dataclass


@dataclass(frozen=True)
class DomainURL:
    """
    Immutable value object wrapping a raw URL string.
    Validates the URL and extracts the registered domain for session lookups.
    """
    raw_url: str

    def __post_init__(self) -> None:
        parsed = urlparse(self.raw_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: '{self.raw_url}'")

    @property
    def domain(self) -> str:
        """Returns the registered host (e.g., 'www.linkedin.com:443' → 'linkedin.com')."""
        netloc = urlparse(self.raw_url).netloc
        # Drop userinfo (user:pass@) if present
        host = netloc.rsplit("@", 1)[-1]
        # Drop the port (but never mangle bracketed IPv6 literals)
        if not host.startswith("["):
            host = host.split(":", 1)[0]
        # Lowercase for consistent session-key matching, then strip leading 'www.'
        return host.lower().removeprefix("www.")

    @property
    def normalized_url(self) -> str:
        """Returns a cleaned URL string safe for browser navigation."""
        url_str = self.raw_url.strip()
        # Convert LinkedIn search URLs with currentJobId parameter to direct job view URLs
        if "linkedin.com" in self.domain and "currentJobId=" in url_str:
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(url_str)
            qs = parse_qs(parsed.query)
            if "currentJobId" in qs and qs["currentJobId"]:
                job_id = qs["currentJobId"][0]
                return f"https://www.linkedin.com/jobs/view/{job_id}/"
        return url_str

    def __str__(self) -> str:
        return self.raw_url

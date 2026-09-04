from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from src.domain.enums.scrape_status import ScrapeStatus
from src.domain.value_objects.job_details import JobDetails


@dataclass
class Opportunity:
    """
    Domain entity representing a single job opportunity submitted by a user.
    Holds all structured job data extracted by the AI extraction layer.
    """
    user_id: str
    source_url: str
    status: ScrapeStatus = ScrapeStatus.PENDING

    # Set after scraping
    id: Optional[str] = None
    final_url: Optional[str] = None
    job_details: Optional[JobDetails] = None
    scraped_at: Optional[datetime] = None
    error_message: Optional[str] = None

    created_at: datetime = field(default_factory=datetime.utcnow)

    def mark_in_progress(self) -> None:
        self.status = ScrapeStatus.IN_PROGRESS

    def mark_success(self, job_details: JobDetails, final_url: str) -> None:
        self.status = ScrapeStatus.SUCCESS
        self.job_details = job_details
        self.final_url = final_url
        self.scraped_at = datetime.utcnow()

    def mark_auth_required(self) -> None:
        self.status = ScrapeStatus.AUTH_REQUIRED

    def mark_failed(self, reason: str) -> None:
        self.status = ScrapeStatus.FAILED
        self.error_message = reason

from dataclasses import dataclass
from typing import Optional
from src.domain.enums.scrape_status import ScrapeStatus
from src.domain.value_objects.job_details import JobDetails
from src.domain.value_objects.company_profile import CompanyProfile


@dataclass
class ScrapeRequestDTO:
    """Carries incoming scrape request data from the presentation layer."""
    user_id: str
    url: str
    action: str = "scrape"


@dataclass
class JobResultDTO:
    """
    Returned by ScrapeJobURL use case to the presentation layer.

    On success: status=SUCCESS, job_details is populated.
    On auth required: status=AUTH_REQUIRED, auth_required_domain is populated.
    On failure: status=FAILED, error_message is populated.
    """
    status: ScrapeStatus
    opportunity_id: Optional[str] = None
    job_details: Optional[JobDetails] = None
    company_profile: Optional[CompanyProfile] = None
    auth_required_domain: Optional[str] = None
    auth_required_login_url: Optional[str] = None
    error_message: Optional[str] = None
    message: Optional[str] = None

    @classmethod
    def success(cls, job_details: JobDetails, company_profile: CompanyProfile | None = None, opportunity_id: str | None = None) -> "JobResultDTO":
        return cls(status=ScrapeStatus.SUCCESS, job_details=job_details, company_profile=company_profile, opportunity_id=opportunity_id)

    @classmethod
    def auth_required(cls, domain: str, login_url: str | None = None, message: str | None = None) -> "JobResultDTO":
        return cls(
            status=ScrapeStatus.AUTH_REQUIRED,
            auth_required_domain=domain,
            auth_required_login_url=login_url,
            message=message,
        )

    @classmethod
    def failed(cls, reason: str) -> "JobResultDTO":
        return cls(status=ScrapeStatus.FAILED, error_message=reason)

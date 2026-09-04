from dataclasses import dataclass, field
from typing import Optional
from src.domain.value_objects.sales_enablement_result import SalesEnablementResult


@dataclass
class SalesEnablementRequestDTO:
    """Carries incoming sales enablement request data from the presentation layer."""
    job_details: str
    projects: list[dict] = field(default_factory=list)


@dataclass
class SalesEnablementResponseDTO:
    """
    Returned by the sales enablement use case to the presentation layer.

    On success: status=SUCCESS, result fields are populated.
    On failure: status=FAILED, error_message is populated.
    """
    status: str
    discovery_questions: Optional[list[str]] = None
    talking_points: Optional[list[str]] = None
    outreach_subject: Optional[str] = None
    outreach_template: Optional[str] = None
    error_message: Optional[str] = None

    @classmethod
    def success(cls, result: SalesEnablementResult) -> "SalesEnablementResponseDTO":
        return cls(
            status="SUCCESS",
            discovery_questions=list(result.discovery_questions),
            talking_points=list(result.talking_points),
            outreach_subject=result.outreach_subject,
            outreach_template=result.outreach_template,
        )

    @classmethod
    def failed(cls, reason: str) -> "SalesEnablementResponseDTO":
        return cls(status="FAILED", error_message=reason)

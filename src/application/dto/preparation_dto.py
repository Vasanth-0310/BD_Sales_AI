"""DTOs for the Technical Preparation pipeline."""

from dataclasses import dataclass, field
from typing import Optional
from src.domain.value_objects.technical_prep_result import TechnicalPrepResult


@dataclass
class TechnicalPrepRequestDTO:
    """Carries incoming Technical Preparation request data from the presentation layer.

    Attributes:
        job_details:      Plain text job description.
        variant_id:       UUID of the selected candidate variant (fetched from Qdrant).
        matching_skills:  Skills the candidate already has (from Step 3 Gemini result).
        missing_skills:   Skills the candidate lacks (from Step 3 Gemini result).
    """
    job_details: str
    variant_id: str
    matching_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)


@dataclass
class TechnicalPrepResponseDTO:
    """Returned by the technical prep use case to the presentation layer.

    On success: status=SUCCESS, result is populated with name, title, and guide.
    On failure: status=FAILED, error_message is populated.
    """
    status: str
    candidate_name: Optional[str] = None
    variant_title: Optional[str] = None
    result: Optional[TechnicalPrepResult] = None
    error_message: Optional[str] = None

    @classmethod
    def success(
        cls,
        candidate_name: str,
        variant_title: str,
        result: TechnicalPrepResult,
    ) -> "TechnicalPrepResponseDTO":
        return cls(
            status="SUCCESS",
            candidate_name=candidate_name,
            variant_title=variant_title,
            result=result,
        )

    @classmethod
    def failed(cls, reason: str) -> "TechnicalPrepResponseDTO":
        return cls(status="FAILED", error_message=reason)

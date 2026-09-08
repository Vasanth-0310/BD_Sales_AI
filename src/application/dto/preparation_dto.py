"""DTOs for the Technical Preparation pipeline."""

from dataclasses import dataclass, field
from typing import Optional
from src.domain.value_objects.technical_prep_result import (
    TechnicalPrepResult,
    BatchTechnicalPrepResult,
)


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
    user_id: str = ""   # Tenant scoping for the variant fetch (empty = no filter)
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


@dataclass
class CandidateInputDTO:
    """Per-candidate input for batch technical prep."""
    variant_id: str
    matching_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)


@dataclass
class BatchTechnicalPrepRequestDTO:
    """Carries batch tech prep request data from the presentation layer."""
    job_details: str
    user_id: str = ""   # Tenant scoping, forwarded to each per-candidate prep
    candidates: list[CandidateInputDTO] = field(default_factory=list)


@dataclass
class BatchTechnicalPrepResponseDTO:
    """Returned by the batch tech prep use case to the presentation layer."""
    status: str
    result: Optional[BatchTechnicalPrepResult] = None
    error_message: Optional[str] = None

    @classmethod
    def success(cls, result: BatchTechnicalPrepResult) -> "BatchTechnicalPrepResponseDTO":
        return cls(status="SUCCESS", result=result)

    @classmethod
    def failed(cls, reason: str) -> "BatchTechnicalPrepResponseDTO":
        return cls(status="FAILED", error_message=reason)

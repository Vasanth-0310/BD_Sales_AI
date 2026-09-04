from dataclasses import dataclass, field
from typing import Optional
from src.domain.value_objects.project_match_result import ProjectMatchResult


@dataclass
class IngestProjectDTO:
    """Carries incoming project ingest data from the presentation layer."""
    project_id: str                          # UUID
    user_id: str
    project_name: str
    domain: str
    techstacks: list[str] = field(default_factory=list)
    description: str = ""
    links: dict[str, str] = field(default_factory=dict)
    case_study_text: str = ""                # Extracted from docx or pdf


@dataclass
class ProjectMatchRequestDTO:
    """Carries incoming match request data from the presentation layer."""
    job_details: str                         # Plain text job description from the user


@dataclass
class ProjectMatchResponseDTO:
    """
    Returned by match use case to the presentation layer.

    On success: status=SUCCESS, matches is populated.
    On failure: status=FAILED, error_message is populated.
    """
    status: str                              # "SUCCESS" | "FAILED"
    matches: Optional[list[ProjectMatchResult]] = None
    error_message: Optional[str] = None

    @classmethod
    def success(cls, matches: list[ProjectMatchResult]) -> "ProjectMatchResponseDTO":
        return cls(status="SUCCESS", matches=matches)

    @classmethod
    def failed(cls, reason: str) -> "ProjectMatchResponseDTO":
        return cls(status="FAILED", error_message=reason)

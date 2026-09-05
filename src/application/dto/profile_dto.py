from dataclasses import dataclass, field
from typing import Optional
from src.domain.value_objects.profile_match_result import ProfileMatchResult


@dataclass
class ProjectInVariantDTO:
    """Schema for each project inside a variant."""
    project_id: str
    project_name: str
    domain: str
    tech_stack: list[str] = field(default_factory=list)
    links: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class VariantDTO:
    """Schema for a single variant in the ingest payload."""
    variant_id: str
    variant_title: str
    role: str
    experience_years: int
    no_of_projects: int
    tech_stacks: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    projects: list[ProjectInVariantDTO] = field(default_factory=list)


@dataclass
class IngestProfileDTO:
    """Top-level ingest schema (candidate info + list of variants)."""
    candidate_id: str
    candidate_name: str
    resource_status: str
    email: str
    education: str
    passout_year: int
    dob: str
    branch: str
    user_id: str = ""                        # Tenant ownership stored in Qdrant payload
    variants: list[VariantDTO] = field(default_factory=list)


@dataclass
class ProfileMatchRequestDTO:
    """Carries incoming match request data from the presentation layer."""
    job_details: str
    variant_id: Optional[str] = None   # When set, manual match path is taken
    user_id: str = ""                  # Tenant scoping for vector search (empty = no filter)


@dataclass
class ProfileMatchResponseDTO:
    """
    Returned by match use case to the presentation layer.

    On success: status=SUCCESS, matches is populated.
    On failure: status=FAILED, error_message is populated.
    """
    status: str
    matches: Optional[list[ProfileMatchResult]] = None
    error_message: Optional[str] = None

    @classmethod
    def success(cls, matches: list[ProfileMatchResult]) -> "ProfileMatchResponseDTO":
        return cls(status="SUCCESS", matches=matches)

    @classmethod
    def failed(cls, reason: str) -> "ProfileMatchResponseDTO":
        return cls(status="FAILED", error_message=reason)


@dataclass
class IngestProfileResponseDTO:
    """Returned by ingest use case to the presentation layer."""
    status: str
    candidate_id: Optional[str] = None
    variants_ingested: int = 0
    details: Optional[list[dict]] = None
    error_message: Optional[str] = None

    @classmethod
    def success(cls, candidate_id: str, details: list[dict]) -> "IngestProfileResponseDTO":
        return cls(
            status="SUCCESS",
            candidate_id=candidate_id,
            variants_ingested=len(details),
            details=details,
        )

    @classmethod
    def failed(cls, reason: str) -> "IngestProfileResponseDTO":
        return cls(status="FAILED", error_message=reason)

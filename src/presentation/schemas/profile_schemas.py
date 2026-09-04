from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ─── Ingest Schemas ───────────────────────────────────────────────────────────

class ResourceStatus(str, Enum):
    ON_BENCH = "On Bench"
    ON_PROJECT = "On Project"
    UPSKILLING = "Upskilling"
    PROPOSED_TO_CLIENT = "Proposed to Client"


class ProjectInVariantSchema(BaseModel):
    """A single project inside a variant."""
    project_id: str
    project_name: str
    domain: str
    tech_stack: list[str]
    links: dict[str, str] = Field(default_factory=dict)
    description: str


class VariantSchema(BaseModel):
    """A single variant in the ingest payload."""
    variant_id: str
    variant_title: str
    role: str
    experience_years: int
    no_of_projects: int
    tech_stacks: list[str]
    certifications: list[str] = Field(default_factory=list)
    projects: list[ProjectInVariantSchema] = Field(
        default_factory=list,
        max_length=10,
        description="Max 10 projects per variant",
    )


class CandidateProfileSchema(BaseModel):
    """The candidate profile data being ingested (no request metadata)."""
    candidate_id: str
    candidate_name: str
    resource_status: ResourceStatus = Field(..., description="Current status of the resource")
    email: str
    education: str
    passout_year: int
    dob: str = Field(..., description="Date of birth in YYYY-MM-DD format")
    branch: str
    variants: list[VariantSchema] = Field(
        ...,
        min_length=1,
        description="At least one variant must be provided",
    )


class IngestProfileRequest(BaseModel):
    """Request body for the profile ingestion endpoint.

    Envelope structure:
      - user_id / action  → request metadata (logging/attribution only)
      - profile           → the candidate payload stored in the vector DB
    """
    user_id: str
    action: str = Field(default="ingest_profile", description="The action to perform (default: 'ingest_profile')")
    profile: CandidateProfileSchema


class IngestVariantDetail(BaseModel):
    """Single variant detail in the ingest response."""
    variant_id: str
    variant_title: str


class IngestProfileResponse(BaseModel):
    """Response body for the profile ingestion endpoint."""
    status: str
    candidate_id: Optional[str] = None
    variants_ingested: Optional[int] = None
    details: Optional[list[IngestVariantDetail]] = None
    error_message: Optional[str] = None


# ─── Match Schemas ────────────────────────────────────────────────────────────

class ProfileMatchRequest(BaseModel):
    """Request body for the profile matching endpoint."""
    user_id: str
    job_details: str = Field(..., description="Plain text job description")
    action: str = Field(default="match_profiles", description="The action to perform (default: 'match_profiles')")

class ProfileMatchResultSchema(BaseModel):
    """A single match result in the response."""
    candidate_id: str
    candidate_name: str
    email: str = ""
    variant_id: str
    variant_title: str
    role: Optional[str] = None
    experience_years: int
    match_percentage: int = Field(..., ge=0, le=100)
    matching_skills: list[str]
    missing_skills: list[str]
    justification: str


class ProfileMatchResponse(BaseModel):
    """Response body for the profile matching endpoint."""
    status: str
    matches: list[ProfileMatchResultSchema] = Field(default_factory=list)
    error_message: Optional[str] = None


class DeleteProfileResponse(BaseModel):
    """Response body for the candidate profile delete endpoint."""
    status: str
    message: str

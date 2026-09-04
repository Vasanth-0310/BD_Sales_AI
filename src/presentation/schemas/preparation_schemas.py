"""Pydantic schemas for the Technical Preparation API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field


# â”€â”€â”€ Request Schemas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class CandidateInputSchema(BaseModel):
    """A single candidate's input in a batch tech prep request."""
    variant_id: str = Field(..., description="UUID of the candidate variant from Qdrant.")
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class TechnicalPrepPayloadSchema(BaseModel):
    """The generation inputs: JD + one or more candidate variants.

    ALWAYS a candidates list — send 1 candidate, get 1 prep guide back;
    send N, get N results (processed sequentially)."""
    job_details: str = Field(
        ...,
        description="Plain text job description (same one used in Steps 2 and 3).",
    )
    candidates: list[CandidateInputSchema] = Field(
        ...,
        min_length=1,
        max_length=10,
        description=(
            "List of 1-10 candidates, each with its own variant_id and skill "
            "gaps (matching/missing skills from the profile-match response). "
            "One preparation guide is generated per candidate."
        ),
    )


class TechnicalPrepRequest(BaseModel):
    """Request body for the technical preparation endpoint.

    Envelope structure:
      - user_id / action  -> request metadata (logging/attribution only)
      - payload           -> the generation inputs (always a candidates list)
    """
    user_id: str = Field(..., description="The ID of the user requesting the technical preparation guide.")
    action: str = Field(
            default="generate_technical_prep",
            description="The action to perform (default: 'generate_technical_prep')",
        )
    payload: TechnicalPrepPayloadSchema


# â”€â”€â”€ Response Schemas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class InterviewTopicSchema(BaseModel):
    """A single topic in the interview preparation guide."""
    topic: str = Field(..., description="Title of the preparation topic.")
    focus: str = Field(
        ...,
        description="'weakness' â€” candidate needs to learn/prepare this skill.",
    )
    questions: list[str] = Field(
        ...,
        description=(
            "3-5 study/preparation questions for this skill, calibrated to the "
            "years of experience required in the job description."
        ),
    )


# ___ Response (always per-candidate results) _________________________________

class CandidatePrepResultSchema(BaseModel):
    """Result for a single candidate."""
    variant_id: str
    status: str
    candidate_name: Optional[str] = None
    variant_title: Optional[str] = None
    technical_briefing_note: Optional[str] = None
    interview_preparation_guide: Optional[list[InterviewTopicSchema]] = None
    error_message: Optional[str] = None


class BatchTechnicalPrepResponse(BaseModel):
    """Response body for the technical preparation endpoint.
    results[] always contains one entry per requested candidate."""
    status: str
    results: list[CandidatePrepResultSchema] = Field(default_factory=list)
    error_message: Optional[str] = None



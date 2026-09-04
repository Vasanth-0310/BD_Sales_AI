"""Pydantic schemas for the Technical Preparation API endpoints."""

from typing import Optional
from pydantic import BaseModel, Field


# ─── Request Schemas ──────────────────────────────────────────────────────────

class TechnicalPrepPayloadSchema(BaseModel):
    """The generation inputs: JD + selected candidate variant + skill-gap data."""
    job_details: str = Field(
        ...,
        description="Plain text job description (same one used in Steps 2 and 3).",
    )
    variant_id: str = Field(
        ...,
        description=(
            "UUID of the selected candidate variant from the Step 3 response. "
            "The backend uses this to fetch the full candidate profile from Qdrant."
        ),
    )
    matching_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Skills the candidate already has — taken directly from the "
            "Step 3 (profile match) response. Used as the 20% strength focus."
        ),
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Skills the candidate lacks for this JD — taken directly from the "
            "Step 3 (profile match) response. Primary driver of prep topics (80%)."
        ),
    )


class TechnicalPrepRequest(BaseModel):
    """Request body for the technical preparation endpoint.

    Envelope structure:
      - user_id / action  -> request metadata (logging/attribution only)
      - payload           -> the generation inputs

    The frontend should send this after the user selects a candidate in Step 3.
    The matching_skills and missing_skills come directly from the Step 3 response.
    """
    user_id: str = Field(..., description="The ID of the user requesting the technical preparation guide.")
    action: str = Field(
            default="generate_technical_prep",
            description="The action to perform (default: 'generate_technical_prep')",
        )
    payload: TechnicalPrepPayloadSchema


# ─── Response Schemas ─────────────────────────────────────────────────────────

class InterviewTopicSchema(BaseModel):
    """A single topic in the interview preparation guide."""
    topic: str = Field(..., description="Title of the preparation topic.")
    focus: str = Field(
        ...,
        description="'weakness' — candidate needs to learn/prepare this skill.",
    )
    questions: list[str] = Field(
        ...,
        description=(
            "3-5 study/preparation questions for this skill, calibrated to the "
            "years of experience required in the job description."
        ),
    )


class TechnicalPrepResponse(BaseModel):
    """Response body for the technical preparation endpoint."""
    status: str
    candidate_name: Optional[str] = None
    variant_title: Optional[str] = None
    technical_briefing_note: Optional[str] = Field(
        default=None,
        description=(
            "A 2-3 sentence plain-English summary of what the interview will "
            "focus on, written for the candidate."
        ),
    )
    interview_preparation_guide: Optional[list[InterviewTopicSchema]] = Field(
        default=None,
        description="Ordered list of preparation topics (weaknesses).",
    )
    error_message: Optional[str] = None

from typing import Optional

from pydantic import BaseModel, field_validator


class ProfileMatchResult(BaseModel):
    """
    Immutable value object representing a single profile match result
    returned by the Gemini synthesis step.

    This is the structured output format that Gemini is instructed to
    produce for each matched candidate variant.
    """

    candidate_id: str
    candidate_name: str
    email: Optional[str] = None    # Populated from payload after Gemini scoring
    variant_id: str
    variant_title: str
    role: Optional[str] = None     # Populated from Qdrant payload after Gemini scoring
    experience_years: int
    match_percentage: int          # 0–100 (clamped)
    matching_skills: list[str]
    missing_skills: list[str]
    justification: str

    model_config = {"frozen": False}  # Allow post-construction enrichment

    @field_validator("match_percentage", mode="before")
    @classmethod
    def _clamp_percentage(cls, v):
        """Gemini occasionally emits out-of-range scores — never let them through."""
        try:
            v = int(v)
        except (TypeError, ValueError):
            return 0
        return max(0, min(100, v))

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

    @field_validator("experience_years", mode="before")
    @classmethod
    def _coerce_experience_years(cls, v):
        """Gemini emits messy experience values ("5+ years", "3-5", 4.5, null).
        A bare int field would reject them with a ValidationError and the
        synthesis salvage would SILENTLY DROP the candidate. Coerce instead:
        None → 0, "3-5" → upper bound, floats → floor, non-numeric → 0."""
        if v is None:
            return 0
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        text = str(v).strip()
        if not text:
            return 0
        import re as _re
        range_match = _re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
        if range_match:
            return int(range_match.group(2))
        num_match = _re.search(r"\d+", text)
        return int(num_match.group()) if num_match else 0

from pydantic import BaseModel, field_validator


class ProjectMatchResult(BaseModel):
    """
    Immutable value object representing a single project match result
    returned by the Gemini synthesis step.

    The match_score is automatically clamped to [0.0, 1.0] to guard
    against LLM hallucinating out-of-range values.
    """
    project_id: str
    project_name: str
    match_score: float                       # 0.0 – 1.0 (clamped)
    justification: str                       # Plain text, why it matched
    matched_evidence: list[str]              # Chunk texts used as evidence

    @field_validator("match_score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        """Ensure match_score is always within [0.0, 1.0]."""
        return max(0.0, min(1.0, v))

    model_config = {"frozen": True}

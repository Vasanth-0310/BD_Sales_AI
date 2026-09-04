from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class ClientInformation(BaseModel):
    """
    Structured contact/client block extracted from job postings.
    Maps directly to the 'Client Information' card in the UI.
    """
    name: Optional[str] = None       # e.g. "JAMES"
    company: Optional[str] = None    # e.g. "TechNova Solutions"
    role: Optional[str] = None       # e.g. "Chief Executive Officer"
    designation: Optional[str] = None # e.g. "CEO" (Abbreviated version of role)
    email: Optional[str] = None      # e.g. "james@technovasolutions.com"
    contact: Optional[str] = None    # e.g. "+13 456 483"
    location: Optional[str] = None   # e.g. "London, UK"

    model_config = {"frozen": True}


class JobDetails(BaseModel):
    """
    Immutable value object representing the structured data extracted
    from a job posting. Used as the Pydantic schema enforced by Gemini's
    structured output mode.
    """
    title: str
    domain: Optional[str] = None            # Business vertical e.g. "SaaS", "FinTech", "HealthTech", "E-commerce"
    company: Optional[str] = None            # null if company cannot be definitively identified
    location: Optional[str] = None           # null when the posting doesn't state one
    employment_type: Optional[str] = None
    industry: Optional[str] = None          # e.g. "Healthcare", "Fintech", "E-commerce"
    role: Optional[str] = None              # Standardized role name, e.g. "Mobile Developer (Cross-Platform)"
    experience: Optional[str] = None
    duration: Optional[str] = None          # Project/contract length, e.g. "3 to 6 months"
    level: Optional[Literal["JUNIOR", "INTERMEDIATE", "SENIOR", "EXPERT", "LEAD"]] = None
    salary: Optional[str] = None
    posted_at: Optional[str] = None       # Absolute calendar date, e.g. "August 14, 2026"
    required_skills: list[str] = []         # Must-have technical skills
    preferred_skills: list[str] = []        # Nice-to-have or bonus skills
    benefits: list[str] = []
    client_information: Optional[ClientInformation] = None  # Structured contact block
    apply_url: Optional[str] = None         # Direct application URL or portal link
    ai_job_summary: Optional[str] = None    # AI-generated narrative insight into the role
    required_proposal_questions: list[str] = []  # Questions the client requires in proposals

    @field_validator("level", mode="before")
    @classmethod
    def _normalize_level(cls, v):
        """Tolerate near-miss LLM outputs ('Senior', 'lead') by mapping to the enum."""
        if not isinstance(v, str):
            return v
        normalized = v.strip().upper()
        allowed = ("JUNIOR", "INTERMEDIATE", "SENIOR", "EXPERT", "LEAD")
        if normalized in allowed:
            return normalized
        # Common synonyms → canonical values; anything else becomes null
        synonyms = {
            "ENTRY": "JUNIOR", "ENTRY LEVEL": "JUNIOR", "ENTRY-LEVEL": "JUNIOR",
            "MID": "INTERMEDIATE", "MID LEVEL": "INTERMEDIATE", "MID-LEVEL": "INTERMEDIATE",
            "INTERMEDIATE/SENIOR": "SENIOR",
        }
        return synonyms.get(normalized)

    model_config = {"frozen": True}

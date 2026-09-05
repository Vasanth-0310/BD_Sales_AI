from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from src.domain.value_objects.location_info import LocationInfo, split_location


class ClientInformation(BaseModel):
    """
    Structured contact/client block extracted from job postings.
    Maps directly to the 'Client Information' card in the UI.
    """
    name: Optional[str] = None        # e.g. "JAMES"
    company: Optional[str] = None     # e.g. "TechNova Solutions"
    role: Optional[str] = None        # e.g. "Chief Executive Officer"
    designation: Optional[str] = None # e.g. "CEO" (Abbreviated version of role)
    email: Optional[str] = None       # e.g. "james@technovasolutions.com"
    contact: Optional[str] = None     # e.g. "+13 456 483"
    city: Optional[str] = None        # e.g. "San Francisco"
    state: Optional[str] = None       # e.g. "CA" or "California"
    country: Optional[str] = None     # e.g. "USA" or "United Kingdom"

    model_config = {"frozen": True}


# Keyword → canonical pay_period mapping (shared by the LLM prompt layer and
# the deterministic validator). Ordered: more specific phrases first.
_PAY_PERIOD_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("hourly", "Hourly"), ("/hr", "Hourly"), ("per hour", "Hourly"),
    ("an hour", "Hourly"), ("a hour", "Hourly"),
    ("yearly", "Yearly"), ("annual", "Yearly"), ("per year", "Yearly"),
    ("/year", "Yearly"), ("a year", "Yearly"),
    ("monthly", "Monthly"), ("per month", "Monthly"), ("/month", "Monthly"),
    ("a month", "Monthly"),
    ("weekly", "Weekly"), ("per week", "Weekly"), ("/week", "Weekly"),
    ("a week", "Weekly"),
    ("daily", "Daily"), ("per day", "Daily"), ("/day", "Daily"),
    ("a day", "Daily"),
    ("biweekly", "Biweekly"),
    ("fixed-price", "Fixed"), ("fixed price", "Fixed"),
)


def normalize_pay_period(value) -> Optional[str]:
    """Map any period wording ('annual', '/hr', 'per year', ...) to the
    canonical enum value. Unknown/None → None (never raises)."""
    if not isinstance(value, str):
        return None
    low = value.strip().lower()
    for kw, canon in _PAY_PERIOD_KEYWORDS:
        if kw in low:
            return canon
    # Exact enum match (case-insensitive) for values like "Hourly" itself
    allowed = ("Hourly", "Daily", "Weekly", "Biweekly", "Monthly", "Yearly", "Fixed")
    if low in (a.lower() for a in allowed):
        return next(a for a in allowed if a.lower() == low)
    return None


class SalaryInfo(BaseModel):
    """
    Structured, machine-processable salary breakdown extracted from a posting.
    Lets the backend filter/compare/aggregate without parsing free text.

    - min_pay/max_pay: floats in the NATIVE currency (no conversion) — e.g.
      30.0/60.0 for "$30.00 - $60.00 Hourly". Single fixed price → min_pay
      only, max_pay None.
    - pay_period: fixed enum so the frontend gets predictable values
      ("Hourly"/"Yearly"/... — never "per year"/"annual" leaks).
    - currency: ISO 4217 code (USD/INR/EUR/GBP/...) — never a symbol,
      since "$" alone is ambiguous between USD/CAD/AUD.
    - raw: the original salary text exactly as shown, for display.
    """
    min_pay: Optional[float] = None
    max_pay: Optional[float] = None
    pay_period: Optional[Literal[
        "Hourly", "Daily", "Weekly", "Biweekly", "Monthly", "Yearly", "Fixed"
    ]] = None
    currency: Optional[str] = None   # ISO 4217: USD, INR, EUR, GBP, CAD, AUD...
    raw: Optional[str] = None        # original text, e.g. "$30.00 - $60.00 Hourly"

    model_config = {"frozen": True}

    @field_validator("pay_period", mode="before")
    @classmethod
    def _normalize_pay_period(cls, v):
        """Bulletproof the enum: Gemini can output 'per year'/'annual' instead
        of the canonical 'Yearly' — a raw Literal rejection would fail the
        ENTIRE extraction. Unknown values degrade to None instead."""
        return normalize_pay_period(v)


class JobDetails(BaseModel):
    """
    Immutable value object representing the structured data extracted
    from a job posting. Used as the Pydantic schema enforced by Gemini's
    structured output mode.
    """
    title: str
    domain: Optional[str] = None            # Business vertical e.g. "SaaS", "FinTech", "HealthTech", "E-commerce"
    company: Optional[str] = None            # null if company cannot be definitively identified
    location: Optional[LocationInfo] = None  # {city, state, country, raw} — null when not stated
    employment_type: Optional[str] = None
    industry: Optional[str] = None          # e.g. "Healthcare", "Fintech", "E-commerce"
    role: Optional[str] = None              # Standardized role name, e.g. "Mobile Developer (Cross-Platform)"
    experience: Optional[str] = None
    duration: Optional[str] = None          # Project/contract length, e.g. "3 to 6 months"
    level: Optional[Literal["JUNIOR", "INTERMEDIATE", "SENIOR", "EXPERT", "LEAD"]] = None
    salary_info: Optional[SalaryInfo] = None  # Structured breakdown (min_pay/max_pay/period/currency)
    posted_at: Optional[str] = None       # Absolute calendar date, e.g. "August 14, 2026"
    required_skills: list[str] = []         # Must-have technical skills
    preferred_skills: list[str] = []        # Nice-to-have or bonus skills
    benefits: list[str] = []
    client_information: Optional[ClientInformation] = None  # Structured contact block
    apply_url: Optional[str] = None         # Direct application URL or portal link
    ai_job_summary: Optional[str] = None    # AI-generated narrative insight into the role
    required_proposal_questions: list[str] = []  # Questions the client requires in proposals

    @field_validator("location", mode="before")
    @classmethod
    def _coerce_location(cls, v):
        """Accept a plain string ('Bengaluru, Karnataka, India') OR a dict
        and always store the structured LocationInfo. Keeps both the free-text
        and structured-output paths safe."""
        return split_location(v)

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

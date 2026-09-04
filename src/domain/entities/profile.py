from dataclasses import dataclass, field


@dataclass
class ProjectInProfile:
    """
    Typed representation of a single project inside a profile variant.
    Enforces that all required fields (including domain) are always present.
    """
    project_id: str
    project_name: str
    domain: str
    tech_stack: list[str] = field(default_factory=list)
    links: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class ProfileVariant:
    """
    Domain entity representing a single variant of a candidate's profile.

    Each variant is a distinct "persona" (e.g., Full Stack Developer,
    Backend Developer) with its own tech stacks, certifications, and
    project list.  Each variant is independently embedded and stored
    as a single point in the Qdrant ``profile_variants`` collection.
    """

    # ── Candidate-level (common across all variants) ─────────────────
    candidate_id: str
    candidate_name: str
    email: str
    education: str
    passout_year: int
    dob: str
    branch: str

    # ── Variant-level ────────────────────────────────────────────────
    variant_id: str
    variant_title: str
    role: str
    experience_years: int
    no_of_projects: int
    tech_stacks: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    projects: list[ProjectInProfile] = field(default_factory=list)

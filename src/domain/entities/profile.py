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

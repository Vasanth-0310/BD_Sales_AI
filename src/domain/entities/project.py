from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Project:
    """
    Domain entity representing a project in the RAG system.
    Contains the core project metadata provided by the backend
    during the ingest process.
    """
    project_id: str                          # UUID
    user_id: str
    name: str
    domain: str
    techstacks: list[str]
    description: str
    links: dict[str, str] = field(default_factory=dict)
    case_study_text: Optional[str] = None    # Extracted from .docx or .pdf


@dataclass
class ProjectChunk:
    """
    Domain entity representing a single chunk of a project's case study.
    Each chunk is independently embedded and stored in Qdrant.
    Denormalized fields (project_name, domain, techstacks) are included
    so that chunk evidence can be passed directly to Gemini without
    requiring a separate lookup.
    """
    chunk_id: str                            # UUID, unique per chunk
    project_id: str                          # UUID, parent project reference
    user_id: str                             # Denormalized for access control
    project_name: str                        # Denormalized for fast retrieval
    domain: str                              # Denormalized for fast retrieval
    techstacks: list[str] = field(default_factory=list)  # Denormalized for fast retrieval
    text: str = ""                           # The chunk content
    token_count: int = 0                     # Used for debugging and limit validation
    sequence_index: int = 0                  # Order within the project's case study

from typing import Optional
from pydantic import BaseModel, Field


class ProjectDataSchema(BaseModel):
    """The project data being ingested (no request metadata)."""
    project_id: str
    project_name: str
    domain: str
    techstacks: list[str]
    description: str = Field(..., max_length=20_000)
    links: dict[str, str] = Field(default_factory=dict)


class IngestProjectRequest(BaseModel):
    """Schema for the JSON payload in the ingest endpoint.

    Envelope structure:
      - user_id / action  -> request metadata (logging/attribution only)
      - project           -> the project data stored in the vector DB
    """
    user_id: str
    action: str = Field(default="ingest_project", description="The action to perform (default: 'ingest_project')")
    project: ProjectDataSchema


class ProjectMatchRequest(BaseModel):
    """Request body for the project matching endpoint."""
    user_id: str
    job_details: str = Field(..., max_length=50_000)
    action: str = Field(default="match_projects", description="The action to perform (default: 'match_projects')")


class ProjectIngestResponse(BaseModel):
    """Response body for the project ingest endpoint.

    Keys MUST stay stable — the frontend depends on this exact shape.
    """
    project_id: str
    chunks_stored: int


class ProjectMatchResponse(BaseModel):
    """Response body for the project matching endpoint."""
    status: str
    matches: list[dict] = Field(default_factory=list)
    error_message: Optional[str] = None


class DeleteProjectResponse(BaseModel):
    """Response body for the project delete endpoint."""
    status: str
    message: str


# ─── Sales Enablement Schemas ─────────────────────────────────────────────────

class ProjectContextSchema(BaseModel):
    """A single project's context sent by the frontend for sales enablement generation."""
    project_name: str = Field(..., description="The project name (used as context only, not quoted in output)")
    domain: str = Field(..., description="The industry domain of the project")
    tech_stack: list[str] = Field(..., description="Technologies used in the project")
    description: str = Field(..., description="Brief description of what the project does")


class SalesEnablementPayloadSchema(BaseModel):
    """The generation context: job description plus 1-3 matched projects."""
    job_details: str = Field(..., max_length=50_000, description="Plain text job description")
    projects: list[ProjectContextSchema] = Field(
        default_factory=list,
        max_length=3,
        description="List of 0–3 matched projects to use as context",
    )


class SalesEnablementRequest(BaseModel):
    """Request body for the sales enablement generation endpoint.

    Envelope structure:
      - user_id / action  -> request metadata (logging/attribution only)
      - payload           -> the JD + matched projects used as generation context
    """
    user_id: str
    action: str = Field(default="generate_sales_enablement", description="The action to perform (default: 'generate_sales_enablement')")
    payload: SalesEnablementPayloadSchema


class SalesEnablementResponse(BaseModel):
    """Response body for the sales enablement generation endpoint."""
    status: str
    discovery_questions: Optional[list[str]] = None
    talking_points: Optional[list[str]] = None
    outreach_subject: Optional[str] = None
    outreach_template: Optional[str] = None
    error_message: Optional[str] = None


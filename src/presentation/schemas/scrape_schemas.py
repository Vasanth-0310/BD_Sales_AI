from typing import Optional
from pydantic import BaseModel, HttpUrl, Field


class ScrapeRequest(BaseModel):
    user_id: str
    action: str = Field(default="scrape", description="The action to perform (default: 'scrape')")
    url: HttpUrl


class ManualEntryJobSchema(BaseModel):
    """The manually-entered job details being analyzed (no request metadata)."""
    company_name: str = Field(..., description="The name of the hiring company")
    company_website: str = Field(default="", description="The company's website URL")
    job_title: str = Field(..., description="The title of the job role")
    experience: str = Field(default="", description="Experience required")
    job_description: str = Field(..., max_length=50_000, description="The full job description text")
    additional_notes: Optional[str] = Field(default=None, max_length=20_000, description="Any additional context provided by the user")


class ManualEntryRequest(BaseModel):
    """Request body for the manual entry endpoint.

    Envelope structure:
      - user_id / action  -> request metadata (logging/attribution only)
      - job               -> the manually-entered job details being analyzed
    """
    user_id: str = Field(..., description="The ID of the user submitting the job details")
    action: str = Field(default="analyze_manual_entry", description="The action to perform (default: 'analyze_manual_entry')")
    job: ManualEntryJobSchema


class ScrapeResponse(BaseModel):
    status: str
    platform: Optional[str] = None
    job_details: Optional[dict] = None
    company_profile: Optional[dict] = None
    auth_required: Optional[bool] = None
    auth_domain: Optional[str] = None
    auth_login_url: Optional[str] = None
    error_message: Optional[str] = None
    message: Optional[str] = None  # Human-readable instruction (e.g. run capture_session.py)

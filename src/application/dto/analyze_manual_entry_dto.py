from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalyzeManualEntryDTO:
    user_id: str
    company_name: str
    company_website: str
    job_title: str
    experience: str
    job_description: str
    additional_notes: Optional[str]
    action: str = "analyze_manual_entry"

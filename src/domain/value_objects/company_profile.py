from typing import Optional
from pydantic import BaseModel


class CompanyProfile(BaseModel):
    """
    AI-generated company profile populated by Gemini Google Search grounding.
    Fields match exactly what is displayed in the Company Profile UI card.
    All fields are optional -- if Gemini cannot find reliable information, it returns null.
    """
    overview: Optional[str] = None            # 2-3 sentence company description
    industry: Optional[str] = None            # e.g. "Technology / Mobile Software"
    products_services: Optional[str] = None   # Main products or services
    headquarters: Optional[str] = None        # e.g. "Remote / US Timezone Preferred"
    location: Optional[str] = None            # Country or region, e.g. "UK"

    model_config = {"frozen": True}

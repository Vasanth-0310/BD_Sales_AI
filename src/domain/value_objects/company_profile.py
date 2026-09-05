from typing import Optional
from pydantic import BaseModel, field_validator

from src.domain.value_objects.location_info import LocationInfo, split_location


class CompanyProfile(BaseModel):
    """
    AI-generated company profile populated by Gemini Google Search grounding.
    Fields match exactly what is displayed in the Company Profile UI card.
    All fields are optional -- if Gemini cannot find reliable information, it returns null.
    """
    overview: Optional[str] = None            # 2-3 sentence company description
    industry: Optional[str] = None            # e.g. "Technology / Mobile Software"
    products_services: Optional[str] = None   # Main products or services
    headquarters: Optional[LocationInfo] = None  # {city, state, country, raw} HQ location
    location: Optional[str] = None            # Country or region, e.g. "UK"

    model_config = {"frozen": True}

    @field_validator("headquarters", mode="before")
    @classmethod
    def _coerce_headquarters(cls, v):
        """Accept a string ('Milpitas, California, USA') or a dict and always
        store the structured LocationInfo."""
        return split_location(v)

import re
from typing import Any, Optional

from pydantic import BaseModel


# Work-mode values are not a place — they stay only in `raw`.
_REMOTE_TOKENS = (
    "remote", "work from home", "wfh", "anywhere", "fully remote",
    "100% remote", "telecommute", "global", "worldwide",
    "hybrid", "onsite", "on-site", "on site", "in-office", "in office",
    "distributed", "flexible",
)

# A compact set so 2-part strings like "Bengaluru, India" resolve the last
# segment as country rather than state. Unknown values follow position rules.
_KNOWN_COUNTRIES = {
    "india", "usa", "us", "u.s.", "u.s.a.", "united states",
    "united states of america", "uk", "u.k.", "united kingdom", "england",
    "canada", "australia", "germany", "france", "china", "japan",
    "singapore", "uae", "united arab emirates", "dubai", "saudi arabia",
    "qatar", "oman", "kuwait", "bahrain", "malaysia", "indonesia",
    "vietnam", "thailand", "philippines", "south korea", "korea",
    "new zealand", "ireland", "netherlands", "holland", "belgium",
    "switzerland", "sweden", "norway", "denmark", "finland", "poland",
    "portugal", "spain", "italy", "russia", "brazil", "mexico",
    "argentina", "chile", "peru", "colombia", "costa rica", "panama",
    "nigeria", "kenya", "south africa", "egypt", "israel", "turkey",
    "pakistan", "bangladesh", "sri lanka", "nepal",
}

# Postal-code + ISO country tail, e.g. "600091 IN".
_PINCODE_COUNTRY_RE = re.compile(r"^(\d{4,7})\s*([A-Za-z]{2})$")

# Compound separators: the string is a multi-location/region list, not one place.
_COMPOUND_SEPARATORS = (";", " / ", " and ", "|")


def _is_country(token: str) -> bool:
    t = token.strip().lower().rstrip(".")
    return t in _KNOWN_COUNTRIES or (
        len(t) == 2 and t.isalpha() and t.upper() == t
    )


def _split_text(text: str) -> dict:
    """Deterministic position-based split of a raw location string.

    Returns a dict with any of city/state/country (possibly empty). Rules:
      - remote/work-mode or compound ("India and Qatar", "A; B / C") -> {}
      - pincode tail ("600091 IN") -> country ISO code, token consumed
      - purely numeric segments (plain pincodes) -> dropped
      - 1 part -> country; 2 parts -> city + country/state; 3+ -> city, ..., country
    """
    low = text.lower()
    if any(tok in low for tok in _REMOTE_TOKENS):
        return {}
    if any(sep in low for sep in _COMPOUND_SEPARATORS):
        return {}

    parts: list[str] = []
    tail_country: Optional[str] = None
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            continue
        pin_match = _PINCODE_COUNTRY_RE.match(part)
        if pin_match:
            tail_country = pin_match.group(2).upper()
            continue
        parts.append(part)

    if not parts:
        return {"country": tail_country} if tail_country else {}

    city = state = country = None
    if len(parts) == 1:
        country = parts[0]
    elif len(parts) == 2:
        city = parts[0]
        if _is_country(parts[1]):
            country = parts[1]
        else:
            state = parts[1]
    elif len(parts) == 3:
        city, state, country = parts
    else:
        city = parts[0]
        country = parts[-1]
        state = ", ".join(parts[1:-1])

    if country is None and tail_country is not None:
        country = tail_country

    return {"city": city, "state": state, "country": country}


def split_location(value: Any) -> Optional["LocationInfo"]:
    """Coerce any incoming location representation into a LocationInfo.

      - None            -> None
      - str             -> deterministically split into components (raw kept)
      - dict            -> normalized; any null component is BACKFILLED from a
                           deterministic re-split of 'raw' (Gemini sometimes
                           returns the object with raw set but components null)
      - LocationInfo    -> returned as-is
    """
    if value is None:
        return None
    if isinstance(value, LocationInfo):
        return value

    if isinstance(value, dict):
        clean = {
            k: (str(value.get(k)).strip()
                if value.get(k) and str(value.get(k)).strip() else None)
            for k in ("city", "state", "country")
        }
        raw = value.get("raw")
        raw = (str(raw).strip() if raw and str(raw).strip() else None)

        # Backfill null components from the raw text — never override a value
        # the LLM actually provided.
        if raw is not None and any(v is None for v in clean.values()):
            guess = _split_text(raw)
            clean = {k: clean[k] if clean[k] else guess.get(k)
                     for k in clean}

        # Fully empty object (Gemini emitting {} for "no location") -> null.
        if raw is None and not any(clean.values()):
            return None
        if raw is None:
            raw = ", ".join(str(v) for v in clean.values() if v) or None

        return LocationInfo(raw=raw, **clean)

    text = str(value).strip()
    if not text:
        return None

    guess = _split_text(text)
    return LocationInfo(
        city=guess.get("city"),
        state=guess.get("state"),
        country=guess.get("country"),
        raw=text,
    )


class LocationInfo(BaseModel):
    """
    Structured geographic breakdown for job locations and company HQs.

    - city/state/country: null when the source text doesn't provide them;
      never guessed or inferred.
    - raw: the original location string verbatim, for display and for
      non-place values like "Remote" or "Bengaluru; Delhi; Mumbai".
    """
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    raw: Optional[str] = None

    model_config = {"frozen": True}

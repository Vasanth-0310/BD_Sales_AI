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


# 2-letter tokens that are countries but NOT US-state abbreviations.
# Ambiguous overlaps (CA, CO, ID, IN, NE, GA, ...) are intentionally
# excluded: position rules then classify them as state, matching the
# dominant "City, ST" convention in job postings.
_ISO2_UNAMBIGUOUS = {
    "ae", "at", "au", "be", "bh", "br", "ch", "cl", "cn", "cr", "cz",
    "dk", "eg", "es", "fi", "fr", "gb", "gr", "hk", "hr", "hu", "ie",
    "it", "jp", "ke", "kr", "kw", "lk", "lt", "lu", "lv", "ma", "mx",
    "my", "ng", "nl", "no", "np", "nz", "om", "pe", "ph", "pk", "pl",
    "pt", "qa", "ro", "rs", "ru", "sa", "se", "sg", "si", "sk", "th",
    "tn", "tr", "tw", "ua", "us", "uy", "uz", "vn", "za",
}

# Curated fact table for the metros job platforms overwhelmingly use:
# city (lowercased, incl. common aliases) → (state, country). Applied ONLY
# to null components — declared values are never overwritten, and unknown
# cities stay null (no guessing beyond this static table).
_KNOWN_CITY_GEO: dict[str, tuple[Optional[str], str]] = {
    "bengaluru": ("Karnataka", "India"), "bangalore": ("Karnataka", "India"),
    "mumbai": ("Maharashtra", "India"), "bombay": ("Maharashtra", "India"),
    "chennai": ("Tamil Nadu", "India"), "madras": ("Tamil Nadu", "India"),
    "hyderabad": ("Telangana", "India"),
    "pune": ("Maharashtra", "India"), "poona": ("Maharashtra", "India"),
    "kolkata": ("West Bengal", "India"), "calcutta": ("West Bengal", "India"),
    "kochi": ("Kerala", "India"), "cochin": ("Kerala", "India"),
    "gurugram": ("Haryana", "India"), "gurgaon": ("Haryana", "India"),
    "noida": ("Uttar Pradesh", "India"), "greater noida": ("Uttar Pradesh", "India"),
    "delhi": ("Delhi", "India"), "new delhi": ("Delhi", "India"),
    "ahmedabad": ("Gujarat", "India"), "jaipur": ("Rajasthan", "India"),
    "lucknow": ("Uttar Pradesh", "India"), "indore": ("Madhya Pradesh", "India"),
    "chandigarh": ("Punjab", "India"), "coimbatore": ("Tamil Nadu", "India"),
    "vijayawada": ("Andhra Pradesh", "India"), "visakhapatnam": ("Andhra Pradesh", "India"),
    "thiruvananthapuram": ("Kerala", "India"), "trivandrum": ("Kerala", "India"),
    "london": (None, "United Kingdom"),
    "new york": ("New York", "United States"), "san francisco": ("California", "United States"),
    "dubai": (None, "United Arab Emirates"), "singapore": (None, "Singapore"),
    "berlin": (None, "Germany"), "sydney": (None, "Australia"),
    "toronto": ("Ontario", "Canada"), "doha": (None, "Qatar"),
}


def _apply_city_facts(city: Optional[str], state: Optional[str],
                      country: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Fill null state/country from the curated city table.

    Table facts are deterministic (a major city belongs to exactly one
    country) — this is data resolution, not guessing. Declared values win.
    """
    if not city or (state and country):
        return state, country
    facts = _KNOWN_CITY_GEO.get(city.strip().lower())
    if not facts:
        return state, country
    table_state, table_country = facts
    if state is None and table_state:
        state = table_state
    if country is None:
        country = table_country
    return state, country


def _is_country(token: str) -> bool:
    t = token.strip().lower().rstrip(".")
    return t in _KNOWN_COUNTRIES or (len(t) == 2 and t in _ISO2_UNAMBIGUOUS)


def _scrub_components(clean: dict) -> dict:
    """Validate LLM-declared components — *junk states get rejected*, not kept.

    The trust policy "declared values are never overridden" assumed the LLM
    produce sane components; real scrapes proved otherwise. This keeps the
    spirit of that policy for **valid** values while catching the garbage:
      - state containing a multi-city comma list (≥2 known cities) → null
      - state that is actually a country token → null
      - city that is actually a country token → moved to country (if empty)
    """
    import re as _re

    city, state, country = clean.get("city"), clean.get("state"), clean.get("country")

    # 1. state with ≥2 known cities -> that's a multi-location list, not a state.
    if state:
        city_tokens = _re.findall(
            r"[A-Za-z][A-Za-z\s']+", state
        )
        if sum(1 for t in city_tokens if t.strip().lower() in _KNOWN_CITY_GEO) >= 2:
            state = None
            # for a multi-list even a declared single city is not THE city
            city = None

    # 2. state with country tokens mixed in ("India, Karnataka") -> keep the
    #    non-country segments only ("Karnataka").
    if state:
        segs = [s.strip() for s in state.split(",") if s.strip()]
        kept = [s for s in segs if not _is_country(s)]
        if not kept:
            state = None
        elif len(kept) != len(segs):
            state = ", ".join(kept)

    # 3. city holding a country name -> relocate into country when empty
    if city and _is_country(city):
        if not country:
            country = city
        city = None

    # Re-derive state/country from the city table after scrubbing.
    if city and (state is None or country is None):
        facts = _KNOWN_CITY_GEO.get(city.strip().lower())
        if facts:
            if state is None and facts[0]:
                state = facts[0]
            if country is None:
                country = facts[1]

    return {"city": city, "state": state, "country": country}


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

    # Deduplicate repeated segments ("Bengaluru, India, Karnataka, India" is
    # the model's country-resolution leaking into raw) and drop known-country
    # tokens from the middle so they can never land in `state`.
    seen_parts: set[str] = set()
    deduped: list[str] = []
    for part in parts:
        key = part.lower()
        if key in seen_parts:
            continue
        seen_parts.add(key)
        deduped.append(part)
    parts = deduped
    middle_parts = [p for p in parts[1:-1] if not _is_country(p)]

    # ── MULTI-LOCATION DETECTION ────────────────────────────────────────
    # Naukri/LinkedIn job lists often name SEVERAL cities (7-city lists) —
    # squeezing them into one city/state/country triple produces garbage
    # like state="Mumbai, New Delhi, Hyderabad, ...". When two or more
    # segments are unambiguous known cities, this is a multi-city listing:
    # keep only structured facts that are TRUE for the WHOLE list (country,
    # when every known city agrees) and keep the remaining detail in raw.
    known_city_hits = [p for p in parts if p.strip().lower() in _KNOWN_CITY_GEO]
    if len(known_city_hits) >= 2:
        listed_cities = {_KNOWN_CITY_GEO[p.strip().lower()][1]
                         for p in known_city_hits}
        shared_country = next(iter(listed_cities)) if len(listed_cities) == 1 else None
        if shared_country is None and tail_country is not None:
            shared_country = tail_country
        return {"city": None, "state": None, "country": shared_country}

    city = state = country = None
    if len(parts) == 1:
        if _is_country(parts[0]):
            country = parts[0]
        else:
            city = parts[0]
    elif len(parts) == 2:
        city = parts[0]
        if _is_country(parts[1]):
            country = parts[1]
        else:
            state = parts[1]
    else:
        city = parts[0]
        country = parts[-1] if _is_country(parts[-1]) else None
        if not country:
            # No stated country — try pincode tail, then the city table.
            pass
        state = ", ".join(middle_parts) if middle_parts else None

    if country is None and tail_country is not None:
        country = tail_country

    # Curated city facts: a stated major city pins its state/country
    # deterministically when the components weren't in the source text.
    state, country = _apply_city_facts(city, state, country)

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
        def _clean_str(v: Any) -> Optional[str]:
            # Only accept real strings — never stringify lists/ints that a
            # misbehaving LLM put in a component field.
            if isinstance(v, str):
                s = v.strip()
                return s or None
            return None

        clean = {k: _clean_str(value.get(k)) for k in ("city", "state", "country")}
        raw = _clean_str(value.get("raw"))

        # Backfill null components from the raw text — never override a value
        # the LLM actually provided.
        if raw is not None and any(v is None for v in clean.values()):
            guess = _split_text(raw)
            clean = {k: clean[k] if clean[k] else guess.get(k)
                     for k in clean}

        # Scrub declared-but-junk components (multi-city states, country
        # tokens appearing in state/city) — the "never override" policy is
        # honored for valid values; garbage values are rejected.
        clean = _scrub_components(clean)

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

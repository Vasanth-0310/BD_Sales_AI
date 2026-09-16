"""Helpers for turning verbose job descriptions into compact text."""

import json
import re
from typing import Any


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "not", "no", "nor", "so", "yet", "both", "either", "neither",
    "this", "that", "these", "those", "we", "our", "you", "your",
    "they", "their", "it", "its", "as", "if", "than", "then", "when",
    "who", "which", "what", "how", "all", "each", "any", "such",
    "also", "more", "most", "very", "just", "about", "up", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "looking", "high", "growth", "new", "previous", "highly", "preferred",
    "required", "including", "use", "using", "build", "building",
    "manage", "managing", "role", "team", "years", "year", "experience",
    "title", "domain", "company", "location", "employment", "type",
    "industry", "duration", "level", "salary", "posted", "skills",
    "benefits", "client", "information", "apply", "url", "summary",
    "questions", "proposal", "null", "none", "remote", "full", "time",
}

_SHORT_TECH_TERMS = {
    "ai", "ml", "qa", "ui", "ux", "go", "c", "r",
    "c#", "f#", "js", "ts", "ci", "cd",
}
_TOKEN_RE = re.compile(
    r"(?:\.[A-Za-z][A-Za-z0-9]*|[A-Za-z][A-Za-z0-9+#.]*)",
    re.IGNORECASE,
)


def extract_jd_keywords(jd_text: str, max_keywords: int = 40) -> str:
    """Extract a bounded set of high-signal terms for Qdrant MatchText."""

    source_parts = _structured_keyword_sources(jd_text)
    if not source_parts:
        source_parts = [jd_text]

    keywords: list[str] = []
    seen: set[str] = set()
    for part in source_parts:
        for token in _TOKEN_RE.findall(str(part)):
            # Preserve leading dots in technologies such as .NET while
            # removing ordinary surrounding punctuation.
            normalized = token.strip(",;:()[]{}'\"").strip()
            if not normalized:
                continue

            key = normalized.lower()
            if key in _STOPWORDS:
                continue
            if len(normalized) < 3 and key not in _SHORT_TECH_TERMS:
                continue
            if key in seen:
                continue

            seen.add(key)
            keywords.append(normalized)
            if len(keywords) >= max_keywords:
                return " ".join(keywords)

    return " ".join(keywords)


def build_retrieval_query(jd_text: str) -> str:
    """Build a compact, importance-aware query for embeddings and BM25.

    Retrieval must not be driven by a raw JSON request where company, city and
    a long preferred-skill tail carry the same weight as the central stack.
    Repeating explicit required skills gives lexical and embedding retrieval a
    deterministic signal without inventing requirements.  Final critical vs
    supporting judgement still happens evidence-first in the synthesizer.
    """
    data = _parse_json_object(jd_text)
    if not data:
        return truncate_text(jd_text, 6000)

    parts: list[str] = []
    for key in ("title", "role", "domain", "industry", "experience", "level"):
        value = data.get(key)
        if value:
            parts.append(str(value))

    required = _as_text_list(data.get("required_skills") or data.get("skills"))
    preferred = _as_text_list(data.get("preferred_skills"))
    # Required terms appear three times, preferred terms once.  This affects
    # recall only; it is never presented as fabricated candidate evidence.
    parts.extend(required * 3)
    parts.extend(preferred)

    for key in (
        "description", "job_description", "responsibilities", "requirements",
        "qualifications", "duties",
    ):
        value = data.get(key)
        if value:
            parts.append(truncate_text(value, 1400))

    return truncate_text(" ".join(parts), 6000)


def compact_job_details(jd_text: str, summary_chars: int = 700) -> str:
    """Return a shorter JD string for LLM prompts while preserving key fields.

    Non-JSON JDs are capped at 6,000 chars (was 1,800): requirements and
    skills frequently sit in the bottom half of long postings, and 1,800
    characters silently amputates them before the LLM ever sees them.
    """
    data = _parse_json_object(jd_text)
    if not data:
        return truncate_text(jd_text, 6000)

    keep_keys = (
        "title",
        "role",
        "level",
        "experience",
        "domain",
        "industry",
        # Preserve source sections used to classify critical versus supporting
        # requirements. Scoring must not rely on an incomplete skill list.
        "description",
        "job_description",
        "responsibilities",
        "requirements",
        "qualifications",
        "duties",
        # BD personalization hooks: the cold email/discovery questions can
        # address the client's company, region (timezone overlap) and
        # engagement duration — strong outsourcing-pitch signals.
        "company",
        "location",
        "duration",
        "required_skills",
        "preferred_skills",
        "ai_job_summary",
    )
    compact: dict[str, Any] = {}
    for key in keep_keys:
        value = data.get(key)
        if value in (None, "", [], {}):
            continue
        if key in {
            "description",
            "job_description",
            "responsibilities",
            "requirements",
            "qualifications",
            "duties",
            "ai_job_summary",
        }:
            value = truncate_text(value, 1200 if key != "ai_job_summary" else summary_chars)
        compact[key] = value

    return json.dumps(compact, ensure_ascii=True)


def truncate_text(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip() + "..."


def _structured_keyword_sources(jd_text: str) -> list[str]:
    data = _parse_json_object(jd_text)
    if not data:
        return []

    parts: list[str] = []
    # Preserve source order: required skills are emitted before preferred
    # terms, so the keyword cap cannot crowd core requirements out.
    for key in ("required_skills", "skills", "tech_stacks", "preferred_skills"):
        value = data.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif value:
            parts.append(str(value))

    for key in ("title", "role", "domain", "industry", "level"):
        value = data.get(key)
        if value:
            parts.append(str(value))

    # Structured skill arrays are occasionally incomplete.  Include original
    # duty text in keyword recall as well as in the later LLM prompt.
    for key in (
        "description", "job_description", "responsibilities", "requirements",
        "qualifications", "duties",
    ):
        value = data.get(key)
        if value:
            parts.append(str(value))

    summary = data.get("ai_job_summary")
    if summary and len(parts) < 8:
        parts.append(str(summary))

    return parts


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return [str(value)] if value else []

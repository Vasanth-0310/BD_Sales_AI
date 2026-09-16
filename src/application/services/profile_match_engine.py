"""Deterministic, evidence-first profile matching.

This module deliberately makes eligibility and scoring independent of model
output.  An LLM may later improve a shortlisted explanation, but it never
decides whether a person is recommended or changes their score.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from src.domain.value_objects.profile_match_result import ProfileMatchResult


_SOFT_SKILL_TERMS = {
    "communication", "analytical", "teamwork", "documentation",
    "problem solving", "leadership", "collaboration", "adaptability",
}
_NON_TECHNICAL_TERMS = _SOFT_SKILL_TERMS | {
    "english", "spanish", "french", "german", "hindi", "tamil",
}
_OPERATIONAL_PHRASE_TERMS = {
    "latency", "timeout", "timeouts", "retry", "retries", "tracing",
    "transaction", "browser", "logging", "correlation", "alerting",
    "monitoring", "failure", "failures", "support",
    "analysis", "profiling", "dump", "dumps", "pool", "pools",
    "management", "troubleshooting", "tuning", "investigation",
}
_GENERIC_DOMAIN_TERMS = {
    "and", "the", "services", "service", "software", "business",
    "consulting", "information", "technology", "technologies", "it",
}
_PLAIN_TEXT_NON_SKILL_TOKENS = {
    "a", "an", "and", "at", "for", "in", "of", "or", "the", "to", "with",
    "senior", "junior", "developer", "engineer", "role", "skills", "required",
    "experience", "year", "years", "minimum", "least", "plus", "must", "should",
}

_ROLE_MARKERS = {
    "backend": {
        "backend", "back end", "java", "spring", "api", "server",
        "python", "django", "flask", "fastapi", "dotnet", ".net",
    },
    "frontend": {"frontend", "front end", "react", "angular", "next.js", "vue"},
    "data": {"data engineer", "data science", "machine learning", "analytics", "etl"},
    "devops": {"devops", "sre", "platform engineer", "cloud engineer", "kubernetes"},
    "qa": {"qa", "quality assurance", "test engineer", "sdet", "testing"},
    "mobile": {"android", "ios", "flutter", "react native", "mobile"},
}

_DIRECT_EQUIVALENTS = {
    "react": {"react.js", "reactjs"},
    "next.js": {"nextjs", "next js"},
    "javascript": {"js"},
    "typescript": {"ts"},
    "node.js": {"node", "nodejs"},
    "kubernetes": {"k8s"},
    "postgresql": {"postgres"},
    "rest api": {"restful api", "rest apis", "restful web api"},
    "spring boot": {"springboot"},
    "mongodb": {"mongo db", "mongo"},
    "artificial intelligence": {"ai"},
    "machine learning": {"ml"},
    "data analysis": {"data analytics", "data analyst"},
    "software testing": {"testing", "quality assurance", "qa", "test cases"},
    "quality assurance engineering": {"quality assurance", "qa", "qa engineer"},
    "automation testing": {"test automation", "automation", "selenium", "webdriver"},
    "manual testing": {"manual test", "manual qa"},
    "regression testing": {"regression test"},
    "full-stack development": {"full stack", "fullstack"},
}


@dataclass(frozen=True)
class JobRequirementPlan:
    role_family: str | None
    critical: list[str]
    supporting: list[str]
    preferred: list[str]
    minimum_years: int | None
    domain: str


def _normalise(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _skill_key(value: object) -> str:
    """Compare technology spellings without punctuation/spacing differences."""
    return re.sub(r"[^a-z0-9]+", "", _normalise(value))


def _is_nontechnical_requirement(value: object) -> bool:
    """Exclude soft skills and spoken languages from technical coverage."""
    normalized = _normalise(value)
    if normalized in _NON_TECHNICAL_TERMS:
        return True
    return any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", normalized)
        for term in _NON_TECHNICAL_TERMS
    )


def _equivalent_terms(requirement: str) -> set[str]:
    """Return a bidirectional controlled-equivalence group for one skill."""
    key = _skill_key(requirement)
    for canonical, aliases in _DIRECT_EQUIVALENTS.items():
        group = {canonical, *aliases}
        if any(_skill_key(item) == key for item in group):
            return {_normalise(item) for item in group}
    return {_normalise(requirement)}


def _plain_text_requirements(text: str) -> list[str]:
    """Keep recognised multi-word technologies intact for pasted JDs."""
    source = _normalise(text)
    phrases = set(_DIRECT_EQUIVALENTS)
    # Single-word core technologies must be discovered too.  Previously Java
    # in a prose JD was appended after multi-word phrases, so a Java role
    # could incorrectly make Spring/REST/Kubernetes the only core skills.
    phrases.update(
        marker for markers in _ROLE_MARKERS.values() for marker in markers
        if marker not in {"api", "server", "backend", "back end", "frontend", "front end", "mobile"}
    )
    phrases.update(
        {
            "spring boot", "spring mvc", "machine learning", "rest api",
            "react native", "data analysis", "quality assurance engineering",
            "manual testing", "automation testing", "regression testing",
            "websphere", "weblogic", "jvm profiling", "oracle sql",
            "distributed tracing", "microservices architecture",
        }
    )
    found: list[tuple[int, int, str]] = []
    for phrase in phrases:
        for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", source):
            found.append((match.start(), match.end(), phrase))
            break
    # At the same location prefer a complete name ("Spring Boot") to its
    # component marker ("Spring"), and do not retain overlapping names.
    found.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    accepted: list[tuple[int, int, str]] = []
    for candidate in found:
        if any(candidate[0] < end and start < candidate[1] for start, end, _ in accepted):
            continue
        accepted.append(candidate)
    accepted.sort(key=lambda item: item[0])
    requirements = [phrase for _, _, phrase in accepted]
    # Preserve useful single-token technologies not covered by the phrase map.
    covered_positions = [(start, end) for start, end, _ in accepted]
    for token_match in re.finditer(r"(?:\.[A-Za-z][A-Za-z0-9]*|[A-Za-z][A-Za-z0-9+#.]*)", text):
        token = token_match.group().rstrip(".")
        if (
            not any(start <= token_match.start() and token_match.end() <= end for start, end in covered_positions)
            and token.lower() not in {_normalise(item) for item in requirements}
            and not _is_nontechnical_requirement(token)
            and token.lower() not in _PLAIN_TEXT_NON_SKILL_TOKENS
        ):
            requirements.append(token)
        if len(requirements) >= 30:
            break
    return requirements


def _parse_jd(job_details: str) -> dict:
    try:
        parsed = json.loads(job_details)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        # Plain-text JDs are supported by the public endpoint.  Preserve a
        # bounded requirement inventory rather than silently scoring everybody
        # against an empty list.
        text = str(job_details or "")
        return {
            # The opening part of a prose JD normally carries its role title.
            # This gives role-family detection a reliable signal without
            # treating the whole responsibility list as a title.
            "title": re.split(r"[\n.!?]", text, maxsplit=1)[0][:180],
            "description": text,
            "required_skills": _plain_text_requirements(text),
        }


def _role_family(jd: dict) -> str | None:
    source = " ".join(str(jd.get(key) or "") for key in ("title", "role")).lower()
    scores = {
        family: sum(marker in source for marker in markers)
        for family, markers in _ROLE_MARKERS.items()
    }
    family, score = max(scores.items(), key=lambda item: item[1])
    return family if score else None


def _minimum_years(jd: dict) -> int | None:
    # Only consume the structured experience field; never mistake company age
    # or project duration inside a prose description for a job requirement.
    match = re.search(r"\d+", str(jd.get("experience") or ""))
    if not match and jd.get("description"):
        match = re.search(
            r"\b(\d+)\s*\+?\s*(?:years?|yrs?)\b",
            str(jd.get("description") or ""),
            re.IGNORECASE,
        )
    return int(match.group(1) if match and match.lastindex else match.group()) if match else None


def build_job_requirement_plan(job_details: str) -> JobRequirementPlan:
    jd = _parse_jd(job_details)
    required = [str(item).strip() for item in jd.get("required_skills") or [] if str(item).strip()]
    preferred = [str(item).strip() for item in jd.get("preferred_skills") or [] if str(item).strip()]
    family = _role_family(jd)

    # Core skills are technical requirements explicitly aligned to the role.
    # Preserve the JD's order as a tie-breaker; it usually reflects priority.
    technical = [skill for skill in required if not _is_nontechnical_requirement(skill)]

    def _is_primary_capability(skill: str) -> bool:
        normalized = _normalise(skill)
        # Responsibilities such as latency investigation and browser tracing
        # matter, but they do not define whether someone is a Java developer.
        if any(term in normalized for term in _OPERATIONAL_PHRASE_TERMS):
            return False
        # A named platform/tool is meaningful even when it is specialised for
        # a role (for example WebSphere in a Java performance position).  The
        # explicit operational exclusion above prevents duties from becoming
        # core merely because we need three core slots.
        return bool(normalized)

    aligned = [skill for skill in technical if _is_primary_capability(skill)]
    critical: list[str] = []
    for skill in aligned:
        if skill not in critical:
            critical.append(skill)
        # Three primary capabilities mirror how a recruiter reads a JD. Extra
        # requirements remain meaningful but should not turn a partial, real
        # match into a false zero.
        if len(critical) == 3:
            break
    return JobRequirementPlan(
        role_family=family,
        critical=critical,
        supporting=[skill for skill in technical if skill not in critical],
        preferred=[skill for skill in preferred if not _is_nontechnical_requirement(skill)],
        minimum_years=_minimum_years(jd),
        domain=str(jd.get("domain") or "").strip(),
    )


def _evidence_text(payload: dict) -> tuple[str, str]:
    tech: list[str] = [str(item) for item in payload.get("tech_stacks") or []]
    prose: list[str] = []
    for project in payload.get("projects") or []:
        tech.extend(str(item) for item in project.get("tech_stack") or [])
        prose.append(str(project.get("description") or ""))
    tech.extend(str(item) for item in payload.get("certifications") or [])
    return _normalise(" ".join(tech)), _normalise(" ".join(prose))


def _is_evidenced(
    requirement: str,
    tech_text: str,
    prose_text: str,
    role_text: str,
) -> bool:
    term = _normalise(requirement)
    candidates = _equivalent_terms(term)
    # A skill stack is explicit evidence. Prose needs a bounded word match.
    explicit_evidence = any(
        re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", tech_text)
        or re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", prose_text)
        for candidate in candidates
    )
    if explicit_evidence:
        return True
    # A declared professional title is legitimate evidence for broad role
    # capabilities only. It must not claim a specific tool such as Selenium,
    # Django, or automation on the candidate's behalf.
    return term in {"software testing", "quality assurance engineering", "full-stack development"} and any(
        re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", role_text)
        for candidate in candidates
    )


def _role_is_aligned(
    plan: JobRequirementPlan,
    payload: dict,
    critical_matches: int,
    prose_text: str,
) -> bool:
    if not plan.role_family:
        return True
    title = _normalise(f"{payload.get('role', '')} {payload.get('variant_title', '')}")
    candidate_families = {
        family for family, markers in _ROLE_MARKERS.items()
        if any(marker in title for marker in markers)
    }
    if not candidate_families or plan.role_family in candidate_families:
        return True
    # A frontend/QA/data/etc. title can legitimately describe a transition,
    # but language or database keywords alone do not prove backend delivery.
    # Require a project to explicitly state that the person built backend
    # services, not merely consumed an API from a frontend or test suite.
    if plan.role_family == "backend":
        return bool(
            critical_matches >= max(1, math.ceil(len(plan.critical) / 2))
            and re.search(r"\b(developed|built|implemented|created|designed)\b", prose_text)
            and re.search(
                r"\b(back\s*end|backend|server[ -]?side|rest(?:ful)?\s+api|microservices?)\b",
                prose_text,
            )
        )
    # Other role-family transitions need project-level evidence for the target
    # family as well. A global skill list alone is not enough to demonstrate a
    # frontend/data/DevOps/mobile delivery transition.
    markers = _ROLE_MARKERS.get(plan.role_family, set())
    project_evidence = " ".join(
        " ".join(str(item) for item in project.get("tech_stack") or [])
        + " " + str(project.get("description") or "")
        for project in payload.get("projects") or []
    ).lower()
    return (
        critical_matches >= max(1, math.ceil(len(plan.critical) / 2))
        and any(marker in project_evidence for marker in markers)
    )


def profile_role_priority(plan: JobRequirementPlan, payload: dict) -> int:
    """Rank direct-role profiles ahead of full-stack transition profiles.

    Lower is preferred.  This is a ranking preference, not fabricated
    evidence: a profile must still pass ``evaluate_profile`` independently.
    """
    if not plan.role_family:
        return 1
    title = _normalise(f"{payload.get('role', '')} {payload.get('variant_title', '')}")
    if re.search(r"\bfull[ -]?stack\b", title):
        return 1
    candidate_families = {
        family for family, markers in _ROLE_MARKERS.items()
        if any(marker in title for marker in markers)
    }
    if plan.role_family in candidate_families:
        return 0
    # A cross-family profile can only be eligible with explicit project
    # transition evidence. Keep it after direct and full-stack profiles.
    return 2


def _domain_score(plan: JobRequirementPlan, payload: dict) -> float:
    domain = _normalise(plan.domain)
    if not domain:
        return 0.0
    domain_tokens = {
        token for token in re.findall(r"[a-z0-9]+", domain)
        if len(token) > 2 and token not in _GENERIC_DOMAIN_TERMS
    }
    if not domain_tokens:
        return 0.0
    project_domains = " ".join(
        str(project.get("domain") or "") for project in payload.get("projects") or []
    ).lower()
    return 15.0 if domain_tokens.intersection(re.findall(r"[a-z0-9]+", project_domains)) else 0.0


def _experience_score(minimum_years: int | None, value: object) -> float:
    try:
        years = int(float(value))
    except (TypeError, ValueError):
        years = 0
    if minimum_years is None:
        return 0.0
    if years >= minimum_years:
        return 25.0
    return 12.0 if years == minimum_years - 1 else 0.0


def _weighted_technical(plan: JobRequirementPlan, matching: set[str]) -> float:
    categories = ((plan.critical, 45.0), (plan.supporting, 12.0), (plan.preferred, 3.0))
    active = [(items, weight) for items, weight in categories if items]
    if not active:
        return 0.0
    total_weight = sum(weight for _, weight in active)
    return sum(
        (sum(item in matching for item in items) / len(items)) * 60.0 * weight / total_weight
        for items, weight in active
    )


def evaluate_profile(plan: JobRequirementPlan, payload: dict) -> ProfileMatchResult:
    tech_text, prose_text = _evidence_text(payload)
    role_text = _normalise(f"{payload.get('role', '')} {payload.get('variant_title', '')}")
    all_requirements = plan.critical + plan.supporting + plan.preferred
    matching = [
        skill for skill in all_requirements
        if _is_evidenced(skill, tech_text, prose_text, role_text)
    ]
    critical_matches = sum(skill in matching for skill in plan.critical)
    # A real match on a primary skill is enough to be considered. Missing
    # primary skills and a different role are reflected in score and wording,
    # rather than being converted into an artificial 0% rejection.
    required_critical = 1 if plan.critical else 0
    # A different declared role may still be a valid transition, but only
    # when the profile's projects explicitly prove delivery in the target
    # family. This prevents frontend/QA profiles from being recommended for a
    # backend role merely because they list Node.js, JavaScript, or Java.
    role_transition_ok = _role_is_aligned(plan, payload, critical_matches, prose_text)
    eligible = critical_matches >= required_critical and role_transition_ok
    technical = _weighted_technical(plan, set(matching))
    domain = _domain_score(plan, payload)
    experience = _experience_score(plan.minimum_years, payload.get("experience_years", 0))
    # A JD without a stated experience threshold or a meaningful domain must
    # not cap every candidate below 100%.  Normalize only across dimensions
    # the JD actually makes assessable; the underlying evidence remains the
    # same and no unverified points are invented.
    available_weight = 60.0
    if plan.minimum_years is not None:
        available_weight += 25.0
    if {
        token for token in re.findall(r"[a-z0-9]+", _normalise(plan.domain))
        if len(token) > 2 and token not in _GENERIC_DOMAIN_TERMS
    }:
        available_weight += 15.0
    score = int(round((technical + domain + experience) * 100.0 / available_weight)) if eligible else 0
    missing = [skill for skill in all_requirements if skill not in matching]
    evidence = ", ".join(matching) if matching else "no listed job requirements"
    gaps = ", ".join(missing[:5])
    years = payload.get("experience_years", 0)
    experience_sentence = (
        f"The profile reports {years} years, which meets the stated {plan.minimum_years}-year minimum."
        if plan.minimum_years is not None and experience == 25.0
        else f"The profile reports {years} years; the role requests at least {plan.minimum_years} years."
        if plan.minimum_years is not None
        else f"The profile reports {years} years of experience."
    )
    if eligible:
        domain_sentence = (
            f"Its project history also shows {plan.domain} domain evidence."
            if domain > 0 else "No explicit matching domain evidence is stated in the profile."
        )
        justification = (
            f"Relevant because the profile explicitly evidences {evidence}. "
            f"{experience_sentence} {domain_sentence}"
            + (f" Remaining gaps in the supplied profile: {gaps}." if gaps else "")
        )
        if not _role_is_aligned(plan, payload, critical_matches, prose_text):
            justification += (
                " The current role differs from the job family; no explicit "
                "project transition evidence is stated."
            )
    else:
        role_sentence = (
            " The current role differs from the job family and the supplied projects do not explicitly show a transition into this work."
            if not _role_is_aligned(plan, payload, critical_matches, prose_text) else ""
        )
        justification = (
            f"Not recommended for this role: the profile explicitly evidences {evidence}, "
            f"but lacks evidence for core requirements such as {gaps or 'the required core stack'}."
            f" {experience_sentence}{role_sentence}"
        )
    return ProfileMatchResult(
        candidate_id=str(payload.get("candidate_id") or ""),
        candidate_name=str(payload.get("candidate_name") or ""),
        email=str(payload.get("email") or ""),
        variant_id=str(payload.get("variant_id") or ""),
        variant_title=str(payload.get("variant_title") or ""),
        role=str(payload.get("role") or ""),
        experience_years=payload.get("experience_years", 0),
        match_percentage=score,
        matching_skills=matching,
        missing_skills=missing,
        justification=justification,
    )


def no_qualified_profiles_reason(
    plan: JobRequirementPlan,
    evaluated: list[ProfileMatchResult],
) -> str:
    """Explain an empty recommendation result without exposing score mechanics."""
    core = ", ".join(plan.critical) or "the role's stated core requirements"
    if not evaluated:
        return (
            f"No profiles were available to evaluate for this role. The job requires evidence of {core}."
        )
    core_set = set(plan.critical)
    closest = max(
        evaluated,
        key=lambda result: (
            sum(skill in core_set for skill in result.matching_skills),
            result.match_percentage,
            result.experience_years,
        ),
    )
    evidence = ", ".join(closest.matching_skills) or "no stated required skills"
    gaps = ", ".join(closest.missing_skills[:5]) or "the remaining core requirements"
    return (
        f"No profile qualifies for this role. The available profiles do not provide enough explicit "
        f"evidence of the core stack: {core}. The closest profile is {closest.candidate_name} "
        f"({closest.role or closest.variant_title}), which evidences {evidence} but lacks evidence "
        f"for {gaps}."
    )

"""Infrastructure adapter that synthesises project match results via Gemini.

This module implements :class:`ISynthesizerPort` by sending deduplicated,
project-grouped chunk evidence together with a job description to the
Gemini LLM.  The model is instructed to return structured JSON that is
parsed and validated into :class:`ProjectMatchResult` value objects.
"""

import asyncio
import json
import re
import time
from collections import defaultdict

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import settings
from src.common.jd_text import compact_job_details, extract_jd_keywords, truncate_text
from src.common.logger import get_logger
from src.domain.exceptions.rag_exceptions import SynthesisError
from src.domain.interfaces.rag.i_synthesizer_port import ISynthesizerPort
from src.domain.value_objects.project_match_result import ProjectMatchResult
from src.domain.value_objects.sales_enablement_result import (
    SalesEnablementResult,
    SalesEnablementOutput,
)
from src.domain.value_objects.profile_match_result import ProfileMatchResult
from src.domain.value_objects.technical_prep_result import (
    TechnicalPrepResult,
    InterviewTopic,
    TechnicalPrepOutput,
)
from src.infrastructure.metrics.metrics_repository import MetricsRepository

logger = get_logger(__name__)

# A full profile pool can contain 30 candidates. One structured JSON response
# for all of them can exceed the model output ceiling and be truncated midway
# through a string. Score every candidate, but in complete bounded batches.
_PROFILE_MATCH_BATCH_SIZE = 10


class ProfileMatchLLMResult(BaseModel):
    """Private Gemini contract; public API output remains unchanged."""

    candidate_id: str
    candidate_name: str
    variant_id: str
    variant_title: str
    experience_years: int = 0
    match_percentage: float = 0
    matching_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    justification: str = ""
    # Job-first inventory fields.  These are consumed and verified internally,
    # never exposed by the existing API response schema.
    critical_requirements: list[str] = Field(default_factory=list)
    supporting_requirements: list[str] = Field(default_factory=list)
    preferred_requirements: list[str] = Field(default_factory=list)
    domain_score: float = Field(default=0, ge=0, le=15)


class ProfileRequirementInventory(BaseModel):
    """One validated JD requirement allocation, shared by every batch."""

    critical_requirements: list[str] = Field(default_factory=list)
    supporting_requirements: list[str] = Field(default_factory=list)
    preferred_requirements: list[str] = Field(default_factory=list)


class ProjectMatchLLMResult(BaseModel):
    """Private project-score contract; public response schema is unchanged."""

    project_id: str
    project_name: str
    match_score: float = 0
    justification: str = ""
    matched_evidence: list[str] = Field(default_factory=list)
    technical_score: float = Field(default=0, ge=0, le=60)
    domain_score: float = Field(default=0, ge=0, le=20)
    evidence_quality_score: float = Field(default=0, ge=0, le=20)

# ------------------------------------------------------------------ #
# System prompt for the Gemini synthesis call
# ------------------------------------------------------------------ #

# The project matcher must make the same distinction as profile matching:
# core requirements matter more than incidental keyword overlap. This active
# contract also makes evidence provenance enforceable by the parser below.
_PROJECT_SYSTEM_INSTRUCTION = """\
You are an evidence-first technical recruiter AI selecting the most relevant
past projects for a supplied job description (JD). Accuracy and non-fabrication
are more important than returning a high score.

The input contains one JD and evidence grouped by project. Each project has a
project_id, project_name, domain, tech stacks, and retrieved evidence chunks.
Use only the supplied JD and that project's supplied evidence. Never use
outside knowledge, another project's evidence, retrieval rank, project order,
project-name familiarity, or profile verbosity as evidence.

JOB-FIRST ANALYSIS (do this once for all projects)
1. Read the complete JD, including description, responsibilities,
   qualifications, required_skills, preferred_skills, domain, industry,
   experience, and level. Treat ai_job_summary as context only: it must not
   create a requirement absent from explicit JD text or structured fields.
2. Build one fixed requirement inventory before reviewing projects:
   - CRITICAL: competencies essential to the JD's central work, normally the
     2-5 skills repeatedly emphasized or required by core duties.
   - SUPPORTING: useful requirements that support the role but are not core.
   - PREFERRED: explicitly optional, desirable, bonus, plus, or nice-to-have.
   Do not make a skill critical merely because it appears in a list, and do not
   treat a soft skill as technical unless the JD makes it central.
3. Record the JD's explicit domain and industry. If absent, treat it as
   unspecified; do not invent one. Apply the same inventory to every project.

EVIDENCE-ONLY MATCHING (anti-hallucination rules)
- A JD skill matches only when that project's tech stacks or evidence chunks
  explicitly show the skill or a defensible direct equivalent. Project domain
  alone does not prove a skill.
- Semantic matching is allowed only for same technology/direct equivalents,
  such as React/React.js/ReactJS, REST APIs/RESTful Web APIs,
  Selenium WebDriver/Selenium, Node.js/Node, K8s/Kubernetes, Golang/Go,
  Postgres/PostgreSQL, and JS/JavaScript when unambiguous.
- Related is not equivalent: React Native is not React; testing is not
  automatically unit testing; SQL is not automatically PostgreSQL
  administration. When uncertain, mark the skill not evidenced.
- An evidence chunk supports a claim only when its text actually states the
  technology, work, outcome, or domain. Never infer architecture, scale,
  outcomes, or domain from a technology name.
- matched_evidence MUST contain only exact, verbatim strings copied from the
  supplied Evidence Chunks for that same project. Do not paraphrase, shorten,
  combine, or invent evidence text.

EXACT PROJECT SCORE: TECHNICAL 60 + DOMAIN 20 + EVIDENCE QUALITY 20 = 100
Technical score (60 points): critical coverage = 45 points, supporting
coverage = 12 points, preferred coverage = 3 points. Divide each category's
points equally among its non-empty JD requirements. If a category is empty,
redistribute its points proportionally to the non-empty categories. A skill
receives points only when explicitly evidenced or defensibly equivalent;
partial/unclear evidence receives zero. Never count a JD requirement twice.

Domain score (20 points): award points only for explicit evidence that the
project worked in the same or closely comparable domain and comparable
business context. Use 20/20 for direct, explicit domain evidence; 10/20 for
closely adjacent explicit domain evidence; 0/20 when unrelated or not
evidenced. Do not infer domain from technology names.

Evidence quality score (20 points): award 20/20 only when the supplied chunks
explicitly describe relevant implementation or outcomes; 10/20 when they name
relevant skills but provide little detail; 0/20 when there is no relevant
evidence. Do not reward a project for unsupported claims.

Compute final match_score = technical_score + domain_score + evidence_quality_score,
then divide by 100 to produce a float from 0.0 to 1.0. Do not adjust after
arithmetic for overall impression. Show the component scores, category
coverage, total, and a per-skill evidence map in the justification.

INTERNAL REQUIRED FIELDS (not part of the public API): technical_score (0-60),
domain_score (0-20), and evidence_quality_score (0-20).  The application
recomputes match_score from these fields.

OUTPUT RULES
- Return only projects represented in the supplied evidence.
- project_id must be copied exactly from the input.
- project_name must be copied exactly from the input for that project.
- justification must name only facts supported by the JD or that project's
  evidence, and must state when evidence is sparse or a requirement is not
  evidenced.
- A score above 0.90 requires nearly all critical skills, explicit comparable
  domain evidence, and detailed relevant implementation evidence.
- Projects with identical JD evidence, domain evidence, and evidence quality
  must receive identical component and final scores. Do not use unrelated
  metadata as a tie-breaker.

Return the top 3 relevant projects, sorted by match_score descending. If fewer
than 3 projects are supplied, return all of them. Score every supplied project
honestly before selecting the top 3; weak or unrelated projects may score 0.0.
"""

# ------------------------------------------------------------------ #
# System prompt for the Sales Enablement call
# ------------------------------------------------------------------ #
_BD_SYSTEM_INSTRUCTION_TEMPLATE = """\
You are an expert Business Development Manager at a software outsourcing company.
You will be given a Job Description (JD) and a list of past projects our company
has delivered.

Your task is to generate FOUR types of sales enablement content.

CRITICAL RULES — you MUST follow these without exception:
1. You are writing from the perspective of a Business Development Executive, NOT a developer.
2. Do NOT explicitly mention or quote any of the internal project names provided to you.
   Use the context of those projects (domain, tech stack, outcome) but NEVER their name.
3. The tone must be professional but concise throughout.
4. All content must be highly specific to the JD — avoid generic phrases.

Generate the following:

1. discovery_questions: A list of 4-5 business-focused qualification questions
   a BD Executive would ask the client to better understand the project scope, timelines,
   team structure, and overall goals. Keep the questions conversational and high-level;
   avoid deeply technical or architectural questions that a developer would ask.
   ANTI-GENERIC RULE (critical): every question MUST be anchored to a SPECIFIC
   element of THIS job description — its domain, one of its required skills,
   its stated experience level, its location/engagement model, or an unusual
   detail in the posting. A question that could be copy-pasted onto ANY other
   job posting (e.g. "What is your timeline?", "What is your budget?") is
   INVALID. Each question must be ≤2 sentences and end with a question mark.
   A reader who never saw the JD must not be able to claim the question applies
   to their own role.

2. talking_points: A list of 4-6 specific talking points that the BD can use when pitching
   to this client. Each point must reference our specific relevant technical experience
   (drawn from the projects provided) WITHOUT naming those projects.
   Example format: "Our team has delivered [specific type of feature/system] for clients
   in the [domain] space, which directly aligns with your requirement for [JD requirement]."
   PER-PROJECT ATTRIBUTION RULE: draw each point from a DIFFERENT project where
   possible (do not reuse one project's context for every point), and make the
   link explicit: the point must state (a) the capability we delivered, (b) the
   domain or technology context, and (c) the exact JD requirement it answers.
   Each talking point ≤2 sentences. Do NOT invent capabilities absent from the
   provided project context.

3. outreach_subject: A short, catchy, and professional subject line for the cold email.
   It should be relevant to the JD (e.g., mentioning the role or tech stack) and designed
   to get a high open rate. Do NOT include placeholder brackets like [Company].
   Do NOT use ALL-CAPS words, emojis, or clickbait phrasing.

4. outreach_paragraphs: The cold email as a JSON array of EXACTLY 6 paragraph strings
   (the adapter joins them with blank lines — you do NOT need to insert \\n\\n yourself).

   STRUCTURE (exactly 6 paragraphs, in this order):
   Paragraph 1: "Hi,"
   Paragraph 2: "I came across your requirement for a [exact job title from JD] and wanted to connect from __COMPANY_NAME__."
   Paragraph 3: "We have experienced [role type] with strong expertise in [list the required skills from the JD, comma separated]. They also have hands-on experience with [list the preferred/additional skills from the JD that are relevant]."
   Paragraph 4: "Our developers are available remotely on a contract basis with flexibility to quickly integrate into your team."
   Paragraph 5: "I would be happy to share profiles for your review or discuss how we can support your team. Would you be available for a quick call this week?"
   Paragraph 6: "Looking forward to your response."

   STRICT RULES for outreach_paragraphs:
   - Do NOT use "Dear [Name]" — always start with "Hi,"
   - Do NOT add any signature block or "Best regards" closing
   - Do NOT mention any project names
   - Keep each paragraph SHORT — do not add extra sentences or elaboration
   - Replace [role type] with the normalized role from the JD (e.g. "full stack developers", "backend engineers")
   - Skills in paragraph 3 must come directly from the JD's required_skills and preferred_skills
   - Use ONLY the exact skill names that appear in the JD — no substitutes,
     no paraphrases, no added technologies the JD did not mention
"""


class GeminiSynthesizerAdapter(ISynthesizerPort):
    """Concrete :class:`ISynthesizerPort` backed by the Gemini LLM.

    Deduplicates and groups chunk evidence by project, builds a
    structured prompt, and requests structured JSON output that is
    validated with Pydantic.
    """

    def __init__(self, metrics: MetricsRepository | None = None) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model: str = settings.gemini_model
        self._fallback_model: str = settings.gemini_fallback_model.strip()
        self._metrics = metrics
        self._disable_thinking = settings.gemini_disable_thinking
        # Brand name comes from settings — never hardcoded in the prompt body.
        self._bd_instruction = _BD_SYSTEM_INSTRUCTION_TEMPLATE.replace(
            "__COMPANY_NAME__", settings.company_name
        )
        logger.info(
            "GeminiSynthesizerAdapter initialised with model=%s",
            self._model,
        )

    # ------------------------------------------------------------------
    # Gemini call helpers (shared by all four synthesis methods)
    # ------------------------------------------------------------------

    async def _generate_hedged(self, contents, config):
        """
        Call Gemini with thinking disabled and request hedging.

        Same serving-route latency variance as the scrape path applies here
        (3s vs 22s observed for similar prompts): if the first call hasn't
        answered within gemini_hedge_delay_s, a second identical call fires
        in parallel and the first response wins.
        """
        if self._disable_thinking and getattr(config, "thinking_config", None) is None:
            try:
                config = config.model_copy(
                    update={"thinking_config": types.ThinkingConfig(thinking_budget=0)}
                )
            except Exception:
                pass  # SDK doesn't support it — proceed without

        async def _call_model(model_name: str):
            try:
                return await self._client.aio.models.generate_content(
                    model=model_name, contents=contents, config=config
                )
            except Exception as e:
                if "thinking" in str(e).lower() and self._disable_thinking:
                    logger.warning("Synthesizer: thinking_config rejected (%s). Retrying without it...", e)
                    self._disable_thinking = False
                    try:
                        config2 = config.model_copy(update={"thinking_config": None})
                    except Exception:
                        config2 = config
                    return await self._client.aio.models.generate_content(
                        model=model_name, contents=contents, config=config2
                    )
                raise

        async def _call():
            try:
                return await _call_model(self._model)
            except Exception as exc:
                message = str(exc).lower()
                capacity_error = any(
                    marker in message
                    for marker in ("503", "unavailable", "429", "resource_exhausted", "rate limit")
                )
                if not capacity_error or not self._fallback_model or self._fallback_model == self._model:
                    raise
                logger.warning(
                    "Gemini primary model unavailable; retrying with fallback model=%s",
                    self._fallback_model,
                )
                fallback_response = await _call_model(self._fallback_model)
                logger.info(
                    "Gemini synthesis completed with fallback model=%s",
                    self._fallback_model,
                )
                return fallback_response

        hedge_delay = settings.gemini_hedge_delay_s
        primary = asyncio.ensure_future(_call())
        if hedge_delay <= 0:
            # 0 disables hedging — still bound the single call so a hung
            # socket can't stall forever.
            _call_timeout = settings.gemini_call_timeout_s
            return await asyncio.wait_for(
                asyncio.shield(primary),
                timeout=_call_timeout if _call_timeout > 0 else None,
            )
        try:
            return await asyncio.wait_for(asyncio.shield(primary), timeout=hedge_delay)
        except asyncio.TimeoutError:
            logger.info("Synthesizer: Gemini call exceeded %.0fs — hedging with a parallel request...", hedge_delay)
        except Exception:
            raise

        secondary = asyncio.ensure_future(_call())
        # Bound the WHOLE hedged pair — previously asyncio.wait had no
        # deadline, so two simultaneously-hung calls stalled the request
        # forever. The loser-wait keeps its own (generous) bound.
        call_timeout = settings.gemini_call_timeout_s
        loser_wait_timeout = max(call_timeout * 2, 30) if call_timeout > 0 else None
        overall_deadline = (
            hedge_delay + loser_wait_timeout + 5
            if hedge_delay > 0 and loser_wait_timeout else None
        )
        try:
            done, pending = await asyncio.wait(
                {primary, secondary},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=overall_deadline,
            )
            if not done:
                # Both calls blew the total deadline — hang, not slowness.
                raise TimeoutError(
                    "Both hedged Gemini calls exceeded the overall deadline"
                )
            if not pending:
                # Both finished before the waiter resumed — next(iter(pending))
                # would raise StopIteration here. Prefer a successful result.
                ok = [t for t in done if not t.exception()]
                if ok:
                    return ok[0].result()
                raise next(iter(done)).exception()
            winner = next(iter(done))
            loser = next(iter(pending))
            if winner.exception():
                # First-finisher failed — wait for the other one, bounded so a
                # hung request can't stall the pipeline forever.
                logger.warning(
                    "Synthesizer: hedged call failed (%r); awaiting the other request...",
                    winner.exception(),
                )
                return await asyncio.wait_for(
                    asyncio.shield(loser), timeout=loser_wait_timeout
                )
            loser.cancel()
            return winner.result()
        finally:
            primary.cancel()
            secondary.cancel()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate_chunks(chunk_evidence: list[dict]) -> list[dict]:
        """Remove chunks whose text content has already been seen.

        Args:
            chunk_evidence: Raw list of chunk dicts from the retrieval
                pipeline.

        Returns:
            A list with duplicate texts removed (first occurrence kept).
        """
        # Identical boilerplate can legitimately occur in two different
        # projects.  De-duplicating by text alone erased the later project's
        # only evidence before provenance validation.
        seen: set[tuple[str, str]] = set()
        unique: list[dict] = []
        for chunk in chunk_evidence:
            text = chunk.get("text", "")
            key = (str(chunk.get("project_id", "")), str(text))
            if key not in seen:
                seen.add(key)
                unique.append(chunk)
        return unique

    @staticmethod
    def _group_by_project(chunks: list[dict]) -> dict[str, list[dict]]:
        """Group chunks by their ``project_id``.

        Args:
            chunks: Deduplicated chunk dicts.

        Returns:
            A mapping from project_id → list of chunk dicts.
        """
        groups: dict[str, list[dict]] = defaultdict(list)
        for chunk in chunks:
            pid = chunk.get("project_id", "")
            groups[pid].append(chunk)
        return dict(groups)

    @staticmethod
    def _build_evidence_section(
        grouped: dict[str, list[dict]],
    ) -> str:
        """Format grouped chunks into a human-readable evidence section.

        Args:
            grouped: Project-grouped chunk dicts.

        Returns:
            A multi-line string suitable for the LLM prompt.
        """
        parts: list[str] = []
        for project_id, chunks in grouped.items():
            first = chunks[0]
            
            tech = first.get('techstacks') or first.get('tech_stack') or []
            if isinstance(tech, list):
                tech = ", ".join(tech)
            elif not tech:
                tech = "N/A"
                
            header = (
                f"--- Project ID: {project_id} ---\n"
                f"Project Name : {first.get('project_name', 'N/A')}\n"
                f"Domain       : {first.get('domain', 'N/A')}\n"
                f"Tech Stacks  : {tech}\n"
                f"Evidence Chunks:"
            )
            chunk_texts = "\n".join(
                f"  [{i + 1}] {truncate_text(c.get('text', ''), 700)}"
                for i, c in enumerate(chunks[:3])
            )
            parts.append(f"{header}\n{chunk_texts}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def synthesize(
        self,
        job_details: str,
        chunk_evidence: list[dict],
    ) -> list[ProjectMatchResult]:
        """Evaluate chunk evidence against a job description via Gemini.

        The method performs the following steps:

        1. Deduplicate chunks by text content.
        2. Group remaining chunks by ``project_id``.
        3. Build a structured prompt with job details and evidence.
        4. Call the Gemini LLM with structured JSON output.
        5. Parse and validate the response with Pydantic.
        6. Sort results by ``match_score`` descending.

        Args:
            job_details: The full job details dict from the scraper.
            chunk_evidence: List of chunk dicts (with project metadata)
                retrieved from the vector search stage.

        Returns:
            A list of :class:`ProjectMatchResult` value objects sorted
            by ``match_score`` descending.

        Raises:
            SynthesisError: When the LLM call fails, returns invalid
                JSON, or the response cannot be validated.
        """
        start = time.perf_counter()
        try:
            # 1 — Deduplicate
            unique_chunks = self._deduplicate_chunks(chunk_evidence)
            logger.info(
                "synthesize  |  raw_chunks=%d  unique_chunks=%d",
                len(chunk_evidence),
                len(unique_chunks),
            )

            # 2 — Group by project
            grouped = self._group_by_project(unique_chunks)
            logger.info(
                "synthesize  |  projects_with_evidence=%d",
                len(grouped),
            )

            # 3 — Build prompt
            evidence_section = self._build_evidence_section(grouped)
            compact_jd = compact_job_details(job_details)
            user_prompt = (
                "## Job Details\n"
                f"{compact_jd}\n\n"
                "## Evidence by Project\n"
                f"{evidence_section}"
            )

            # 4 — Call Gemini
            logger.info("[Synthesizer] [synthesize_projects] Sending to Gemini (model=%s)...", self._model)
            _gemini_start = time.perf_counter()
            logger.debug(
                "Gemini project-match prompt prepared  |  jd_chars=%d  projects=%d",
                len(compact_jd), len(grouped),
            )
            response = await self._generate_hedged(user_prompt, config=types.GenerateContentConfig(
                    system_instruction=_PROJECT_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=list[ProjectMatchLLMResult],
                    temperature=0.1,
                ),
            )
            _gemini_elapsed = time.perf_counter() - _gemini_start
            logger.info("[Synthesizer] [synthesize_projects] Response in %.2fs", _gemini_elapsed)
            _usage = response.usage_metadata
            _pt = getattr(_usage, "prompt_token_count", 0) or 0
            _ct = getattr(_usage, "candidates_token_count", 0) or 0
            logger.info("[Synthesizer] [synthesize_projects] Tokens: prompt=%d completion=%d total=%d", _pt, _ct, _pt + _ct)
            if self._metrics:
                try:
                    await self._metrics.increment(self._model, _pt, _ct, "synthesize_projects")
                except Exception as metrics_err:
                    logger.warning("Metrics persist failed (non-fatal): %s", metrics_err)

            raw_text = response.text
            logger.debug("Gemini project-match response received  |  chars=%d", len(raw_text))
            logger.info(
                "synthesize  |  raw_response_length=%d",
                len(raw_text),
            )

            # 5 — Parse & validate
            # Deduplicate matched_evidence inside each raw dict BEFORE constructing
            # the frozen ProjectMatchResult — frozen instances cannot be mutated.
            raw_results: list[dict] = json.loads(raw_text)

            # Enforce provenance at the boundary. The model is only allowed to
            # return projects and evidence that were actually supplied to it;
            # this prevents fabricated project names, IDs, and evidence quotes
            # from reaching the API response. Evidence is compared against the
            # exact truncated text shown in the prompt.
            project_names = {
                project_id: str(chunks[0].get("project_name", "N/A"))
                for project_id, chunks in grouped.items()
                if chunks
            }
            allowed_evidence = {
                project_id: {
                    truncate_text(chunk.get("text", ""), 700)
                    for chunk in chunks
                    if chunk.get("text")
                }
                for project_id, chunks in grouped.items()
            }

            def _verified_evidence(project_id: str, value: object) -> str | None:
                """Resolve model evidence to the exact supplied chunk text.

                Gemini may include the prompt's ``[n]`` label or quote only a
                verbatim excerpt. Return the canonical stored chunk so the
                public result never contains model-invented evidence.
                """
                if not isinstance(value, str):
                    return None
                candidate = re.sub(r"^\s*\[\d+\]\s*", "", value).strip()
                if not candidate:
                    return None
                for supplied in allowed_evidence.get(project_id, set()):
                    if candidate == supplied or candidate in supplied or supplied in candidate:
                        return supplied
                return None

            verified_results: list[dict] = []
            for item in raw_results:
                project_id = item.get("project_id")
                if project_id not in project_names:
                    logger.warning(
                        "Dropping project-match item with unknown project_id=%s",
                        project_id,
                    )
                    continue

                # ``match_score`` is model output only.  Recompute the public
                # score from the bounded private component fields below, so a
                # plausible-looking but unsupported total cannot affect rank.
                try:
                    technical_score = float(item.get("technical_score", 0))
                    domain_score = float(item.get("domain_score", 0))
                    evidence_quality_score = float(
                        item.get("evidence_quality_score", 0)
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "Dropping project-match item with invalid component score | project_id=%s",
                        project_id,
                    )
                    continue
                if not (
                    0.0 <= technical_score <= 60.0
                    and 0.0 <= domain_score <= 20.0
                    and 0.0 <= evidence_quality_score <= 20.0
                ):
                    logger.warning(
                        "Dropping project-match item with out-of-range component score | project_id=%s",
                        project_id,
                    )
                    continue

                # Canonicalize identity from retrieved data, never from the
                # model's free-form copy of the project name.
                item["project_name"] = project_names[project_id]
                evidence = item.get("matched_evidence") or []
                seen: set[str] = set()
                verified_evidence: list[str] = []
                for ev in evidence:
                    resolved = _verified_evidence(project_id, ev)
                    if resolved and resolved not in seen:
                        seen.add(resolved)
                        verified_evidence.append(resolved)
                item["matched_evidence"] = verified_evidence
                # A non-zero project score without any verifiable chunk
                # evidence is unsupported. Keep the result shape unchanged,
                # but make the score safely non-matching.  Otherwise the
                # model's total is ignored in favour of verified arithmetic.
                if not item["matched_evidence"]:
                    item["match_score"] = 0.0
                else:
                    item["match_score"] = round(
                        (technical_score + domain_score + evidence_quality_score) / 100,
                        4,
                    )
                    item["justification"] = (
                        f"{item.get('justification', '').strip()} "
                        "Verified score: "
                        f"technical {technical_score:g}/60, "
                        f"domain {domain_score:g}/20, "
                        f"evidence {evidence_quality_score:g}/20."
                    ).strip()
                verified_results.append(item)

            raw_results = verified_results

            # Salvage: one malformed item from Gemini must not kill the whole
            # match response — drop invalid items and keep the valid ones.
            results: list[ProjectMatchResult] = []
            for item in raw_results:
                try:
                    results.append(ProjectMatchResult(**item))
                except Exception as exc:
                    logger.warning(
                        "Dropping invalid ProjectMatchResult item (%s): %s",
                        exc,
                        item.get("project_name", item.get("project_id", "?")),
                    )
            if raw_results and not results:
                raise SynthesisError(
                    reason="All match results failed schema validation"
                )

            # 6 — Sort descending by match_score
            results.sort(key=lambda r: r.match_score, reverse=True)

            elapsed = time.perf_counter() - start
            logger.info(
                "synthesize completed in %.3fs  |  results=%d  top_score=%.3f",
                elapsed,
                len(results),
                results[0].match_score if results else 0.0,
            )
            return results

        except SynthesisError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "synthesize failed after %.3fs: %s",
                elapsed,
                exc,
            )
            raise SynthesisError(
                reason=f"Gemini synthesis call failed. Details logged.",
            ) from exc

    # ------------------------------------------------------------------
    # Sales Enablement Generation
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_sales_enablement(
        self,
        job_details: str,
        projects: list[dict],
    ) -> SalesEnablementResult:
        """Generate sales enablement content (discovery questions, talking points,
        and a formal BD outreach email) from the JD and matched project context.

        Args:
            job_details: Plain text job description.
            projects: List of project dicts. Used as internal context only —
                      project names must NOT appear in the generated output.

        Returns:
            A SalesEnablementResult value object.

        Raises:
            SynthesisError: When the LLM call fails or returns invalid data.
        """
        start = time.perf_counter()
        try:
            # Build the project context section (capped — the BD prompt gains
            # nothing from 10 project dumps but pays for them in tokens).
            _MAX_SE_PROJECTS = 5
            compact_jd = compact_job_details(job_details)
            project_sections = []
            for i, project in enumerate(projects[:_MAX_SE_PROJECTS], 1):
                # Accept both key spellings: the API schema uses "tech_stack"
                # but project-match payloads (Qdrant) use "techstacks" —
                # frontends forwarding match results may send either.
                tech = project.get("tech_stack") or project.get("techstacks") or []
                if isinstance(tech, list):
                    tech = ", ".join(tech)
                project_sections.append(
                    f"Project {i}:\n"
                    f"  Domain      : {project.get('domain', 'N/A')}\n"
                    f"  Tech Stack  : {tech}\n"
                    f"  Description : {truncate_text(project.get('description', 'N/A'), 650)}"
                )
            project_context = "\n\n".join(project_sections)

            user_prompt = (
                "## Job Description\n"
                f"{compact_jd}\n\n"
                "## Our Past Project Context (for your reference only — DO NOT quote project names)\n"
                f"{project_context}"
            )

            logger.info(
                "[Synthesizer] [generate_sales_enablement] Sending to Gemini (model=%s) | projects=%d",
                self._model, len(projects),
            )
            _gemini_start = time.perf_counter()
            response = await self._generate_hedged(user_prompt, config=types.GenerateContentConfig(
                    system_instruction=self._bd_instruction,
                    response_mime_type="application/json",
                    response_schema=SalesEnablementOutput,
                    temperature=0.4,
                ),
            )
            _gemini_elapsed = time.perf_counter() - _gemini_start
            logger.info("[Synthesizer] [generate_sales_enablement] Response in %.2fs", _gemini_elapsed)
            _usage = response.usage_metadata
            _pt = getattr(_usage, "prompt_token_count", 0) or 0
            _ct = getattr(_usage, "candidates_token_count", 0) or 0
            logger.info("[Synthesizer] [generate_sales_enablement] Tokens: prompt=%d completion=%d total=%d", _pt, _ct, _pt + _ct)
            if self._metrics:
                try:
                    await self._metrics.increment(self._model, _pt, _ct, "generate_sales_enablement")
                except Exception as metrics_err:
                    logger.warning("Metrics persist failed (non-fatal): %s", metrics_err)

            raw_result: dict = json.loads(response.text)

            # The email arrives as an explicit paragraph array — join it here so
            # paragraph breaks can never be lost to LLM formatting (the old
            # regex-repair hack silently failed whenever Gemini paraphrased).
            paragraphs = [
                str(p).strip()
                for p in (raw_result.get("outreach_paragraphs") or [])
                if str(p).strip()
            ]

            result = SalesEnablementResult(
                discovery_questions=raw_result.get("discovery_questions") or [],
                talking_points=raw_result.get("talking_points") or [],
                outreach_subject=raw_result.get("outreach_subject", ""),
                outreach_template="\n\n".join(paragraphs),
            )

            # Programmatic leak guard: internal project names must never appear
            # in BD-facing content — enforce the prompt rule in code as well.
            project_names = [
                str(p.get("project_name") or p.get("name") or "").strip()
                for p in projects
            ]
            leak_targets = [result.outreach_template, result.outreach_subject]
            leak_targets += result.talking_points
            leak_targets += result.discovery_questions
            for name in filter(None, project_names):
                pattern = re.compile(
                    rf"(?<!\w){re.escape(name)}(?!\w)",
                    flags=re.IGNORECASE,
                )
                if any(pattern.search(t) for t in leak_targets):
                    logger.warning(
                        "[Synthesizer] Internal project name '%s' leaked into sales "
                        "enablement output — redacting.",
                        name,
                    )
                    # Case-INSENSITIVE redaction: detection ignores case, so
                    # replacement must too — a case-sensitive replace() on
                    # "acmeportal" vs "AcmePortal" redacts 0 characters and
                    # leaks the internal name to the client.
                    result = result.model_copy(update={
                        field: pattern.sub("our recent work", value)
                        for field, value in (
                            ("outreach_template", result.outreach_template),
                            ("outreach_subject", result.outreach_subject),
                        )
                    } | {
                        "talking_points": [
                            pattern.sub("our recent work", tp)
                            for tp in result.talking_points
                        ],
                        # Discovery questions quote client context too — they
                        # leak internal names just as easily as the email does.
                        "discovery_questions": [
                            pattern.sub("our recent work", q)
                            for q in result.discovery_questions
                        ],
                    })

            elapsed = time.perf_counter() - start
            logger.info(
                "generate_sales_enablement completed in %.3fs  |  "
                "questions=%d  talking_points=%d",
                elapsed,
                len(result.discovery_questions),
                len(result.talking_points),
            )
            return result

        except SynthesisError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "generate_sales_enablement failed after %.3fs: %s",
                elapsed,
                exc,
            )
            raise SynthesisError(
                reason=f"Gemini sales enablement call failed. Details logged.",
            ) from exc

    # ------------------------------------------------------------------
    # Profile Matching Synthesis
    # ------------------------------------------------------------------

    # The original prompt above is retained in history for easier review;
    # this assignment is the active profile-matching contract.
    _PROFILE_SYSTEM_INSTRUCTION = """\
You are an evidence-first technical recruiter AI. Evaluate candidate variants
against the supplied job description (JD). Accuracy and non-fabrication are
more important than producing a high score.

The input contains a JD and candidate data. For each candidate return the
requested ProfileMatchResult fields. Use only facts present in that candidate's
input. Never use outside knowledge, another candidate's data, retrieval rank,
profile order, candidate name, availability, or profile verbosity as evidence.

JOB-FIRST ANALYSIS (do this once before reviewing any candidate)
1. Read the complete supplied JD, including description, responsibilities,
   qualifications, required_skills, preferred_skills, domain, industry,
   experience, and level.
   Treat ai_job_summary as a convenience summary only; it is not authoritative
   evidence and must never create a requirement that is absent from the JD's
   explicit text or structured requirement fields.
2. Create one fixed requirement inventory shared by every candidate:
   - CRITICAL: essential competencies needed for the role's central work,
     normally the 2-5 skills repeatedly emphasized or required by core duties.
   - SUPPORTING: useful technical/professional requirements that support the
     role but are not its core.
   - PREFERRED: explicitly optional, desirable, bonus, plus, or nice-to-have.
   Do not make a skill critical only because it appears in a list. Do not turn
   a soft skill into a critical technical skill unless the JD makes it central.
3. Record the JD's explicit domain and explicit experience/seniority target.
   If either is absent, record it as unspecified; do not invent a target.
   Keep this classification and allocation identical for all candidates.
   When the user prompt supplies a `Canonical JD Requirement Inventory`, it
   has already been validated from this JD. Use it exactly; do not reclassify,
   add, remove, or move requirements between its categories.

EVIDENCE-ONLY MATCHING (anti-hallucination rules)
- A skill is MATCHED only when explicitly evidenced in this candidate's
  tech_stacks, project tech_stack, project description, certifications, or
  stated experience. A title, seniority, domain, or adjacent technology is not
  evidence by itself.
- Semantic matching is allowed only for a defensible same-technology or direct
  equivalent, and the profile evidence must be named in the justification.
  Examples: React/React.js/ReactJS, REST APIs/RESTful Web APIs,
  Selenium WebDriver/Selenium, Node.js/Node, K8s/Kubernetes, Golang/Go,
  Postgres/PostgreSQL, and JS/JavaScript when unambiguous.
- Related is not equivalent: React Native is not React; testing is not
  automatically unit testing; SQL is not automatically PostgreSQL
  administration. If uncertain, classify the requirement as NOT EVIDENCED.
- A project domain alone does not prove a skill. A project description counts
  only when it explicitly states the relevant work or technology.
- Missing evidence means "not evidenced", not that the candidate definitely
  cannot do it. Never invent skills, years, outcomes, certifications,
  responsibilities, project scale, equivalences, or domain experience.

EXACT SCORE: TECHNICAL 60 + DOMAIN 15 + EXPERIENCE 25 = 100
Technical score (60 points):
- CRITICAL coverage: 45 points (75% of technical score).
- SUPPORTING coverage: 12 points (20%).
- PREFERRED coverage: 3 points (5%).
Within each non-empty category, divide its points equally among that category's
JD requirements. If a category is empty, redistribute its points
proportionally to the non-empty categories. A requirement receives its full
allocation only when explicitly evidenced or defensibly equivalent. Partial or
unclear evidence receives zero. Never count one JD requirement twice.

Domain score (15 points): award points only for explicit evidence of the same
or closely comparable domain and comparable project complexity. Do not infer
domain from a technology name. If domain evidence is absent, award 0/15.

Experience score (25 points) is calculated by the application from the
canonical profile experience_years and the JD's explicit years target. Do not
invent years or alter the supplied value.

The application calculates the technical allocation and final percentage. Do
not adjust match_percentage for intuition; return it only as a provisional
number and provide the requirement inventory below.

OUTPUT REQUIREMENTS
- matching_skills: JD skill names with explicit evidence. When wording differs,
  include the profile wording in parentheses. Never list inferred skills.
- missing_skills: JD skills not explicitly evidenced; this communicates a gap
  in supplied evidence, not a definite inability.
- justification: concise but auditable, maximum 60 words. Include the
  critical/supporting/preferred split, named evidence, technical/domain/
  experience components, arithmetic total, and any hard requirement not
  evidenced. If data is sparse, say so plainly.
- A 90%+ score requires nearly all critical skills to be evidenced, relevant
  domain evidence, and the explicit experience requirement to be met or closely
  supported.
- Candidates with identical evidence across technical skills, domain, and
experience must receive identical component and final scores. Do not use
availability or any unrelated metadata in scoring.

INTERNAL REQUIRED FIELDS (these are not public API fields)
- critical_requirements: exact JD required-skill names classified as critical.
  Keep this to the 2-5 central requirements unless fewer are listed.
- supporting_requirements: remaining exact JD required-skill names.
- preferred_requirements: exact JD preferred-skill names.
- domain_score: a numeric 0-15 score based only on explicit domain evidence.
The same requirement inventory must be returned for every candidate.

Return a JSON array containing exactly one entry per supplied candidate, sorted by
match_percentage descending. You MUST score every candidate that appears in the input —
do not omit, merge, or skip any candidate regardless of how low its score is.
"""

    async def _build_profile_requirement_inventory(
        self,
        job_details: str,
    ) -> dict[str, list[str]]:
        """Classify one JD once so scores are comparable across batches."""
        compact_jd = compact_job_details(job_details)
        prompt = (
            "## Job Description\n"
            f"{compact_jd}\n\n"
            "Classify only exact items from required_skills: choose the 2-5 "
            "central technical requirements as critical; place every other "
            "required skill in supporting_requirements. Copy preferred_skills "
            "to preferred_requirements. Do not add, rename, infer, or omit any "
            "listed skill. Return JSON only."
        )
        try:
            response = await self._generate_hedged(
                prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ProfileRequirementInventory,
                    temperature=0,
                    max_output_tokens=1024,
                ),
            )
            raw = json.loads(response.text)
            return ProfileRequirementInventory.model_validate(raw).model_dump()
        except Exception as exc:
            # The fallback cannot hallucinate: it is canonicalized from exact
            # JD lists below, with all requirements retained as supporting.
            logger.warning(
                "Profile JD requirement classification failed; using deterministic "
                "JD-only fallback: %s",
                exc,
            )
            return {
                "critical_requirements": [],
                "supporting_requirements": [],
                "preferred_requirements": [],
            }

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def synthesize_profile_matches(
        self,
        job_details: str,
        variant_payloads: list[dict],
        requirement_inventory: dict[str, list[str]] | None = None,
    ) -> list[ProfileMatchResult]:
        """Evaluate candidate variant payloads against a JD using Gemini.

        Args:
            job_details: Plain text job description.
            variant_payloads: List of full variant payload dicts from Qdrant.

        Returns:
            List of ProfileMatchResult sorted by match_percentage descending.

        Raises:
            SynthesisError: When the LLM call fails or returns invalid data.
        """
        start = time.perf_counter()

        # Establish the JD allocation before any candidate batches are sent.
        # Every candidate in this request must use exactly the same inventory.
        if requirement_inventory is None:
            requirement_inventory = await self._build_profile_requirement_inventory(
                job_details
            )

        # Do not reduce the retrieval pool to work around model output limits:
        # score each submitted variant in a separate complete JSON batch, then
        # merge the verified results. Sequential batches avoid a burst of
        # concurrent Gemini calls and preserve the existing retry behaviour.
        if len(variant_payloads) > _PROFILE_MATCH_BATCH_SIZE:
            logger.info(
                "[Synthesizer] [synthesize_profile_matches] Splitting %d variants into batches of %d",
                len(variant_payloads),
                _PROFILE_MATCH_BATCH_SIZE,
            )
            combined_results: list[ProfileMatchResult] = []
            for batch_start in range(0, len(variant_payloads), _PROFILE_MATCH_BATCH_SIZE):
                batch = variant_payloads[
                    batch_start : batch_start + _PROFILE_MATCH_BATCH_SIZE
                ]
                logger.info(
                    "[Synthesizer] Profile-match batch %d-%d of %d",
                    batch_start + 1,
                    batch_start + len(batch),
                    len(variant_payloads),
                )
                combined_results.extend(
                    await self.synthesize_profile_matches(
                        job_details,
                        batch,
                        requirement_inventory=requirement_inventory,
                    )
                )
            combined_results.sort(key=lambda result: result.match_percentage, reverse=True)
            logger.info(
                "[Synthesizer] Profile-match batches completed | variants=%d results=%d elapsed=%.2fs",
                len(variant_payloads),
                len(combined_results),
                time.perf_counter() - start,
            )
            return combined_results

        try:
            # Build the candidate profiles section
            compact_jd = compact_job_details(job_details)
            variant_sections = []
            for i, payload in enumerate(variant_payloads, 1):
                tech = payload.get("tech_stacks", [])
                if isinstance(tech, list):
                    tech_str = ", ".join(tech)
                else:
                    tech_str = str(tech)

                certs = payload.get("certifications", [])
                certs_str = "; ".join(certs) if certs else "None"

                projects_text = ""
                projects = payload.get("projects") or []
                for j, proj in enumerate(projects[:4], 1):
                    p_tech = proj.get("tech_stack", [])
                    if isinstance(p_tech, list):
                        p_tech = ", ".join(p_tech)
                    projects_text += (
                        f"    Project {j}: {proj.get('project_name', 'N/A')}\n"
                        f"      Domain: {proj.get('domain', 'N/A')}\n"
                        f"      Tech: {p_tech}\n"
                        f"      Description: {truncate_text(proj.get('description', 'N/A'), 450)}\n"
                    )
                if len(projects) > 4:
                    projects_text += f"    Additional projects omitted: {len(projects) - 4}\n"

                variant_sections.append(
                    f"Candidate {i}:\n"
                    + f"  candidate_id: {payload.get('candidate_id')}\n"
                    f"  candidate_name: {payload.get('candidate_name')}\n"
                    f"  resource_status: {payload.get('resource_status', 'N/A')}\n"
                    f"  variant_id: {payload.get('variant_id')}\n"
                    f"  variant_title: {payload.get('variant_title')}\n"
                    f"  experience_years: {payload.get('experience_years')}\n"
                    f"  tech_stacks: {tech_str}\n"
                    f"  certifications: {certs_str}\n"
                    f"  projects:\n{projects_text}"
                )

            candidates_context = "\n".join(variant_sections)

            user_prompt = (
                "## Job Description\n"
                f"{compact_jd}\n\n"
                "## Canonical JD Requirement Inventory\n"
                f"{json.dumps(requirement_inventory, ensure_ascii=True)}\n\n"
                "## Candidate Profiles\n"
                f"{candidates_context}"
            )

            logger.info(
                "[Synthesizer] [synthesize_profile_matches] Sending to Gemini (model=%s) | variants=%d",
                self._model, len(variant_payloads),
            )
            logger.debug(
                "Gemini profile-match prompt prepared  |  jd_chars=%d  variants=%d",
                len(compact_jd), len(variant_payloads),
            )
            _gemini_start = time.perf_counter()
            response = await self._generate_hedged(user_prompt, config=types.GenerateContentConfig(
                    system_instruction=self._PROFILE_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=list[ProfileMatchLLMResult],
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            _gemini_elapsed = time.perf_counter() - _gemini_start
            logger.info("[Synthesizer] [synthesize_profile_matches] Response in %.2fs", _gemini_elapsed)
            _usage = response.usage_metadata
            _pt = getattr(_usage, "prompt_token_count", 0) or 0
            _ct = getattr(_usage, "candidates_token_count", 0) or 0
            logger.info("[Synthesizer] [synthesize_profile_matches] Tokens: prompt=%d completion=%d total=%d", _pt, _ct, _pt + _ct)
            if self._metrics:
                try:
                    await self._metrics.increment(self._model, _pt, _ct, "synthesize_profile_matches")
                except Exception as metrics_err:
                    logger.warning("Metrics persist failed (non-fatal): %s", metrics_err)

            raw_text = response.text
            logger.debug("Gemini profile-match response received  |  chars=%d", len(raw_text))
            logger.info(
                "synthesize_profile_matches  |  raw_response_length=%d",
                len(raw_text),
            )

            raw_results: list[dict] = json.loads(raw_text)

            # Enforce profile provenance at the boundary just as project
            # matching already does.  An LLM response is not allowed to alter
            # identity, years, title, or invent a skill outside the supplied
            # JD.  Canonical payload values win over model-generated copies.
            payload_by_variant = {
                str(payload.get("variant_id")): payload
                for payload in variant_payloads
                if isinstance(payload, dict) and payload.get("variant_id")
            }
            jd_data: dict = {}
            try:
                parsed_jd = json.loads(job_details)
                jd_data = parsed_jd if isinstance(parsed_jd, dict) else {}
                jd_skills = (
                    list(jd_data.get("required_skills") or [])
                    + list(jd_data.get("preferred_skills") or [])
                )
            except Exception:
                # The public endpoint also accepts a pasted plain-text JD.
                # Preserve lexical requirements for deterministic scoring;
                # Gemini still performs the semantic/evidence judgment.
                jd_skills = extract_jd_keywords(job_details).split()
                jd_data = {
                    "required_skills": jd_skills,
                    "experience": job_details,
                }
            canonical_skills = {str(skill).strip().lower(): str(skill) for skill in jd_skills if skill}
            required_skills = [
                str(skill) for skill in (jd_data.get("required_skills") or [])
            ]
            preferred_skills = [
                str(skill) for skill in (jd_data.get("preferred_skills") or [])
            ]

            def _canonical_requirement_list(values: object, allowed: set[str]) -> list[str]:
                if not isinstance(values, list):
                    return []
                output: list[str] = []
                for value in values:
                    skill = canonical_skills.get(str(value).strip().lower())
                    if skill and skill in allowed and skill not in output:
                        output.append(skill)
                return output

            # Requirement classification belongs to the JD, not to an LLM
            # result batch. Canonicalize the one preflight inventory strictly
            # against exact JD skills before scoring anybody.
            required_set = set(required_skills)
            critical_requirements = _canonical_requirement_list(
                requirement_inventory.get("critical_requirements"), required_set,
            )
            critical_requirements = critical_requirements[:5]
            supporting_requirements = [
                skill for skill in required_skills if skill not in critical_requirements
            ]
            if not critical_requirements and required_skills:
                critical_requirements = required_skills[:5]
                supporting_requirements = required_skills[5:]
            preferred_requirements = _canonical_requirement_list(
                requirement_inventory.get("preferred_requirements"),
                set(preferred_skills),
            ) or preferred_skills

            def _technical_score(matching: list[str]) -> float:
                categories = [
                    (critical_requirements, 45.0),
                    (supporting_requirements, 12.0),
                    (preferred_requirements, 3.0),
                ]
                active = [(requirements, weight) for requirements, weight in categories if requirements]
                if not active:
                    return 0.0
                total_weight = sum(weight for _, weight in active)
                return sum(
                    (sum(skill in matching for skill in requirements) / len(requirements))
                    * (60.0 * weight / total_weight)
                    for requirements, weight in active
                )

            # The structured experience field can be "2 - 4" or "2 Junior",
            # not only "2 years". Its first number is the minimum target.
            experience_text = str(jd_data.get("experience") or "").strip()
            experience_match = re.search(r"\d+", experience_text)
            required_years = int(experience_match.group()) if experience_match else None

            def _experience_score(value: object) -> float:
                try:
                    years = int(float(value))
                except (TypeError, ValueError):
                    years = 0
                if required_years is None:
                    return 0.0
                if years >= required_years:
                    return 25.0
                if years == required_years - 1:
                    return 12.0
                return 0.0

            target_title = " ".join(
                str(jd_data.get(key) or "") for key in ("title", "role")
            ).lower()
            is_backend_target = bool(
                re.search(r"\b(java|back\s*end|backend)\b", target_title)
            )

            def _has_backend_delivery_evidence(payload: dict) -> bool:
                """A QA title needs actual delivery evidence, not language overlap."""
                tech_values = list(payload.get("tech_stacks") or [])
                projects = payload.get("projects") or []
                descriptions: list[str] = []
                for project in projects:
                    tech_values.extend(project.get("tech_stack") or [])
                    descriptions.append(str(project.get("description") or ""))
                tech_text = " ".join(map(str, tech_values)).lower()
                if re.search(
                    r"\b(spring\s*boot|spring\s*mvc|hibernate|jpa|"
                    r"microservices?|kafka|quarkus|dropwizard)\b",
                    tech_text,
                ):
                    return True
                project_text = " ".join(descriptions).lower()
                return bool(
                    re.search(r"\b(developed|built|implemented|created|designed)\b", project_text)
                    and re.search(
                        r"\b(back\s*end|backend|server[ -]?side|rest(?:ful)?\s+api|microservices?)\b",
                        project_text,
                    )
                )

            def _is_role_ineligible(payload: dict) -> bool:
                candidate_role = " ".join(
                    str(payload.get(key) or "") for key in ("role", "variant_title")
                ).lower()
                is_qa_role = bool(
                    re.search(r"\b(qa|quality assurance|test(?:ing)?|sdet)\b", candidate_role)
                )
                return is_backend_target and is_qa_role and not _has_backend_delivery_evidence(payload)

            verified_results: list[dict] = []
            scored_variant_ids: set[str] = set()
            for item in raw_results:
                if not isinstance(item, dict):
                    continue
                variant_id = str(item.get("variant_id") or "").strip()
                payload = payload_by_variant.get(variant_id)
                if payload is None:
                    logger.warning("Dropping profile-match item with unknown variant_id=%s", variant_id)
                    continue
                if variant_id in scored_variant_ids:
                    logger.warning(
                        "Dropping duplicate Gemini profile-match item for variant_id=%s",
                        variant_id,
                    )
                    continue
                scored_variant_ids.add(variant_id)

                def _canonical_skill_list(values: object) -> list[str]:
                    if not isinstance(values, list):
                        return []
                    output: list[str] = []
                    seen_skills: set[str] = set()
                    for value in values:
                        text = str(value).strip()
                        # "React Query (TanStack Query)" maps to the exact JD
                        # term while preserving equivalent evidence in the
                        # justification only.
                        base = text.split("(", 1)[0].strip().lower()
                        canonical = canonical_skills.get(base)
                        if canonical and canonical not in seen_skills:
                            seen_skills.add(canonical)
                            output.append(canonical)
                    return output

                matching = _canonical_skill_list(item.get("matching_skills"))
                all_requirements = (
                    critical_requirements + supporting_requirements + preferred_requirements
                )
                missing = [skill for skill in all_requirements if skill not in matching]
                technical_score = _technical_score(matching)
                try:
                    domain_score = float(item.get("domain_score", 0))
                except (TypeError, ValueError):
                    domain_score = 0.0
                domain_score = max(0.0, min(15.0, domain_score))
                experience_score = _experience_score(payload.get("experience_years", 0))
                final_score = int(round(technical_score + domain_score + experience_score))
                role_ineligible = _is_role_ineligible(payload)
                if role_ineligible:
                    # A Java/SQL keyword match does not establish that a QA
                    # engineer is qualified for a backend delivery role. Keep
                    # the result auditable, but prevent it from becoming a
                    # recommendation unless the profile proves the transition.
                    logger.info(
                        "Profile role gate: %s excluded from backend recommendation "
                        "because QA role lacks explicit backend delivery evidence",
                        payload.get("candidate_name", variant_id),
                    )
                    final_score = 0
                # Identity and experience always originate from the Qdrant
                # payload, never from free-form LLM output.
                item.update({
                    "candidate_id": str(payload.get("candidate_id", "")),
                    "candidate_name": str(payload.get("candidate_name", "")),
                    "variant_id": variant_id,
                    "variant_title": str(payload.get("variant_title", "")),
                    "experience_years": payload.get("experience_years", 0),
                    "matching_skills": matching,
                    "missing_skills": missing,
                    "match_percentage": final_score,
                })
                role_gate_note = (
                    " Role gate: QA-labelled profile has no explicit backend "
                    "delivery evidence; not eligible for this backend role."
                    if role_ineligible else ""
                )
                item["justification"] = (
                    f"{str(item.get('justification') or '').strip()}{role_gate_note} "
                    f"Verified score: Technical {technical_score:.1f}/60, "
                    f"Domain {domain_score:.1f}/15, Experience {experience_score:.1f}/25, "
                    f"Total {final_score}/100."
                ).strip()
                verified_results.append(item)

            # Backfill: create zero-score entries for any candidates Gemini
            # omitted so no candidate silently disappears from the pipeline.
            missing_vids = set(payload_by_variant.keys()) - scored_variant_ids
            if missing_vids:
                logger.warning(
                    "[Synthesizer] Gemini omitted %d/%d candidates — backfilling with zero scores",
                    len(missing_vids), len(payload_by_variant),
                )
                all_requirements = (
                    critical_requirements + supporting_requirements + preferred_requirements
                )
                for vid in missing_vids:
                    p = payload_by_variant[vid]
                    experience_score = _experience_score(p.get("experience_years", 0))
                    backfill_score = int(round(experience_score))
                    verified_results.append({
                        "candidate_id": str(p.get("candidate_id", "")),
                        "candidate_name": str(p.get("candidate_name", "")),
                        "variant_id": vid,
                        "variant_title": str(p.get("variant_title", "")),
                        "experience_years": p.get("experience_years", 0),
                        "matching_skills": [],
                        "missing_skills": list(all_requirements),
                        "match_percentage": backfill_score,
                        "justification": (
                            "Gemini did not return a score for this candidate. "
                            f"Backfilled with experience-only score: {backfill_score}/100."
                        ),
                    })
                    logger.info(
                        "[Synthesizer] Backfilled: %s (%s) → %d%%",
                        p.get("candidate_name", "?"), vid, backfill_score,
                    )

            raw_results = verified_results

            # Salvage: drop malformed items instead of failing the whole
            # profile-match response (same policy as project matching).
            results: list[ProfileMatchResult] = []
            for item in raw_results:
                try:
                    results.append(ProfileMatchResult(**item))
                except Exception as exc:
                    logger.warning(
                        "Dropping invalid ProfileMatchResult item (%s): %s",
                        exc,
                        item.get("candidate_name", item.get("variant_id", "?")),
                    )
            if raw_results and not results:
                raise SynthesisError(
                    reason="All profile match results failed schema validation"
                )

            # Sort by match_percentage descending
            results.sort(key=lambda r: r.match_percentage, reverse=True)

            elapsed = time.perf_counter() - start
            logger.info(
                "synthesize_profile_matches completed in %.3fs  |  results=%d",
                elapsed,
                len(results),
            )
            return results

        except SynthesisError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "synthesize_profile_matches failed after %.3fs: %s",
                elapsed,
                exc,
            )
            raise SynthesisError(
                reason=f"Gemini profile synthesis failed. Details logged.",
            ) from exc

    # ------------------------------------------------------------------
    # Technical Preparation Synthesis
    # ------------------------------------------------------------------

    _TECH_PREP_SYSTEM_INSTRUCTION = """\
You are a Senior Technical Interview Coach. Your task is to prepare a specific
developer for an upcoming client interview.

You will receive:
1. A job description describing exactly what the client needs — including the
   years of experience the client expects.
2. A candidate profile including:
   - Their name, title, and years of experience
   - Their full tech stack and certifications
   - A list of their past projects (name, domain, tech used, description)
   - matching_skills: skills they ALREADY HAVE (20% focus)
   - missing_skills: skills they LACK for this JD (80% focus — these are priorities)

Your output MUST follow this JSON structure exactly:
{
  "technical_briefing_note": "<2-3 sentence paragraph>",
  "interview_preparation_guide": [
    {"topic": "<topic name>", "focus": "weakness", "questions": ["<question 1>", "<question 2>", "<question 3>"]}
  ]
}

Rules:
- technical_briefing_note: Address the candidate by name. Acknowledge their strengths
  briefly, then clearly state what the interview will focus on (the weakness areas).
  Example: "Since [Name] already has strong REST API and JavaScript skills, the
  client interview will focus primarily on React Native and Redux — areas that
  require targeted preparation."
- interview_preparation_guide: Look at the bulleted list of `missing_skills`. You MUST generate exactly ONE separate topic entry for EACH individual bullet point in that list (focus="weakness"). Do NOT mash or combine multiple bullet points into a single topic. If there are N missing skills, output exactly N topics. Do NOT include any strength topics. If missing_skills is "None", return an empty list [].
- questions: For EACH topic, generate 3-5 preparation/study questions covering what
  the candidate needs to learn about that skill. The questions ARE the study guide —
  each question tells the candidate exactly what to go and learn.
- EXPERIENCE CALIBRATION (critical): The difficulty and depth of every question MUST
  match the years of experience demanded in the JOB DESCRIPTION (not the candidate's
  own experience). Read the JD's stated requirement:
  * Junior level (0-2 years): fundamentals, definitions, basic syntax, simple use cases.
  * Mid level (3-5 years): practical application, debugging scenarios, integration,
    common patterns and pitfalls.
  * Senior level (6+ years): architecture decisions, trade-offs, performance at scale,
    security, mentoring/leadership angles, "how would you design X" style questions.
- Questions must be specific to THIS job description and the skill in question —
  reference the actual requirements in the JD where possible. Do NOT be generic.
- Do NOT include skills in the guide that are not relevant to the JD.
"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def generate_technical_prep(
        self,
        job_details: str,
        candidate_context: dict,
    ) -> TechnicalPrepResult:
        """Generate a targeted Technical Preparation guide via Gemini.

        Args:
            job_details: Plain text job description.
            candidate_context: Combined dict of the candidate's Qdrant profile
                data merged with matching_skills / missing_skills from Step 3.

        Returns:
            TechnicalPrepResult containing the briefing note and guide topics.

        Raises:
            SynthesisError: When the LLM call fails or returns invalid data.
        """
        start = time.perf_counter()
        try:
            # Build past projects section — capped and truncated like the
            # profile-match prompt (a prep guide gains nothing from 12 full
            # project descriptions, but pays for them in latency and tokens).
            _MAX_PREP_PROJECTS = 6
            projects_text = ""
            projects = candidate_context.get("projects") or []
            for i, proj in enumerate(projects[:_MAX_PREP_PROJECTS], 1):
                p_tech = proj.get("tech_stack", [])
                if isinstance(p_tech, list):
                    p_tech = ", ".join(p_tech)
                projects_text += (
                    f"  Project {i}: {proj.get('project_name', 'N/A')}\n"
                    f"    Domain: {proj.get('domain', 'N/A')}\n"
                    f"    Tech: {p_tech}\n"
                    f"    Description: {truncate_text(proj.get('description', 'N/A'), 450)}\n"
                )
            if len(projects) > _MAX_PREP_PROJECTS:
                projects_text += (
                    f"  Additional projects omitted: {len(projects) - _MAX_PREP_PROJECTS}\n"
                )

            tech_stacks = candidate_context.get("tech_stacks", [])
            tech_str = ", ".join(tech_stacks) if isinstance(tech_stacks, list) else str(tech_stacks)

            certs = candidate_context.get("certifications", [])
            certs_str = "; ".join(certs) if certs else "None"

            matching_list = candidate_context.get("matching_skills", [])
            missing_list = candidate_context.get("missing_skills", [])
            
            matching = "\n".join(f"    - {s}" for s in matching_list) if matching_list else "    - None"
            missing = "\n".join(f"    - {s}" for s in missing_list) if missing_list else "    - None"

            user_prompt = (
                "## Job Description\n"
                f"{job_details}\n\n"
                "## Candidate Profile\n"
                f"  Name: {candidate_context.get('candidate_name')}\n"
                f"  Title: {candidate_context.get('variant_title')}\n"
                f"  Experience: {candidate_context.get('experience_years')} years\n"
                f"  Tech Stacks: {tech_str}\n"
                f"  Certifications: {certs_str}\n"
                f"  Past Projects:\n{projects_text}\n"
                "## Skill Gap Analysis (from Step 3)\n"
                f"  matching_skills (candidate HAS these):\n{matching}\n"
                f"  missing_skills (candidate LACKS these):\n{missing}\n"
            )

            logger.info(
                "[Synthesizer] [generate_technical_prep] Sending to Gemini (model=%s) | candidate=%s",
                self._model, candidate_context.get("candidate_name"),
            )
            logger.debug("Gemini Tech Prep Input Prompt:\n%s", user_prompt)
            _gemini_start = time.perf_counter()
            response = await self._generate_hedged(user_prompt, config=types.GenerateContentConfig(
                    system_instruction=self._TECH_PREP_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=TechnicalPrepOutput,
                    temperature=0.3,
                ),
            )
            _gemini_elapsed = time.perf_counter() - _gemini_start
            logger.info("[Synthesizer] [generate_technical_prep] Response in %.2fs", _gemini_elapsed)
            _usage = response.usage_metadata
            _pt = getattr(_usage, "prompt_token_count", 0) or 0
            _ct = getattr(_usage, "candidates_token_count", 0) or 0
            logger.info("[Synthesizer] [generate_technical_prep] Tokens: prompt=%d completion=%d total=%d", _pt, _ct, _pt + _ct)
            if self._metrics:
                try:
                    await self._metrics.increment(self._model, _pt, _ct, "generate_technical_prep")
                except Exception as metrics_err:
                    logger.warning("Metrics persist failed (non-fatal): %s", metrics_err)

            raw_text = response.text
            logger.debug("Gemini Tech Prep Raw Output:\n%s", raw_text)

            data = json.loads(raw_text)

            # Salvage: schema enforcement keeps Gemini honest, but validate
            # item-by-item anyway so one bad topic never kills the guide.
            guide: list[InterviewTopic] = []
            for item in data.get("interview_preparation_guide", []) or []:
                try:
                    questions_raw = item.get("questions") or []
                    questions = [str(q).strip() for q in questions_raw if str(q).strip()]
                    if not item.get("topic") or not questions:
                        raise ValueError("topic or questions missing/empty")
                    guide.append(InterviewTopic(
                        topic=item["topic"],
                        focus=item["focus"],
                        questions=questions,
                    ))
                except Exception as exc:
                    logger.warning(
                        "Dropping invalid InterviewTopic item (%s): %s",
                        exc,
                        item.get("topic", "?"),
                    )
            if data.get("interview_preparation_guide") and not guide:
                raise SynthesisError(
                    reason="All interview topics failed schema validation"
                )

            result = TechnicalPrepResult(
                technical_briefing_note=data.get("technical_briefing_note", ""),
                interview_preparation_guide=guide,
            )

            elapsed = time.perf_counter() - start
            logger.info(
                "generate_technical_prep completed in %.3fs  |  topics=%d",
                elapsed,
                len(guide),
            )
            return result

        except SynthesisError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "generate_technical_prep failed after %.3fs: %s", elapsed, exc
            )
            raise SynthesisError(
                reason=f"Gemini technical prep generation failed. Details logged.",
            ) from exc

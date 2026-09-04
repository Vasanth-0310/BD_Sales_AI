"""Infrastructure adapter that synthesises project match results via Gemini.

This module implements :class:`ISynthesizerPort` by sending deduplicated,
project-grouped chunk evidence together with a job description to the
Gemini LLM.  The model is instructed to return structured JSON that is
parsed and validated into :class:`ProjectMatchResult` value objects.
"""

import asyncio
import json
import time
from collections import defaultdict

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from src.common.config import settings
from src.common.jd_text import compact_job_details, truncate_text
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

# ------------------------------------------------------------------ #
# System prompt for the Gemini synthesis call
# ------------------------------------------------------------------ #
_SYSTEM_INSTRUCTION = """\
You are an expert technical recruiter AI.  Your task is to evaluate a
company's past projects against a new job opportunity and determine which
projects are the most relevant.

You will receive:
1. A plain text string describing the job opportunity (title, required skills,
   description, etc.).
2. Evidence chunks grouped by project.  Each group contains the
   project_name, domain, tech stacks, and the actual text chunks that
   were retrieved as evidence of relevance.

Your output MUST be a JSON array of objects, each with:
- project_id   (str)   — the project UUID
- project_name (str)   — the project name
- match_score  (float) — relevance score between 0.0 and 1.0
- justification (str)  — concise explanation of why this project matches
- matched_evidence (list[str]) — the chunk texts you found most relevant

Return the TOP 3 most relevant projects, sorted by match_score
descending.  If fewer than 3 projects exist, return all of them.
Score EVERY provided project honestly — give genuinely weak matches the low
score they deserve (the caller filters anything below a relevance threshold).
If the evidence chunks contradict a project's summary, trust the evidence.
Be precise and objective — do not inflate scores.
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

2. talking_points: A list of 4-6 specific talking points that the BD can use when pitching
   to this client. Each point must reference our specific relevant technical experience
   (drawn from the projects provided) WITHOUT naming those projects.
   Example format: "Our team has delivered [specific type of feature/system] for clients
   in the [domain] space, which directly aligns with your requirement for [JD requirement]."

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

        async def _call():
            try:
                return await self._client.aio.models.generate_content(
                    model=self._model, contents=contents, config=config
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
                        model=self._model, contents=contents, config=config2
                    )
                raise

        hedge_delay = settings.gemini_hedge_delay_s
        primary = asyncio.ensure_future(_call())
        try:
            return await asyncio.wait_for(asyncio.shield(primary), timeout=hedge_delay)
        except asyncio.TimeoutError:
            logger.info("Synthesizer: Gemini call exceeded %.0fs — hedging with a parallel request...", hedge_delay)
        except Exception:
            raise

        secondary = asyncio.ensure_future(_call())
        try:
            done, pending = await asyncio.wait(
                {primary, secondary}, return_when=asyncio.FIRST_COMPLETED
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
                    asyncio.shield(loser), timeout=settings.gemini_call_timeout_s
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
        seen: set[str] = set()
        unique: list[dict] = []
        for chunk in chunk_evidence:
            text = chunk.get("text", "")
            if text not in seen:
                seen.add(text)
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
            logger.debug(f"Gemini Project Match Input Prompt:\n{user_prompt}")
            response = await self._generate_hedged(user_prompt, config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=list[ProjectMatchResult],
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
                await self._metrics.increment(self._model, _pt, _ct, "synthesize_projects")

            raw_text = response.text
            logger.debug(f"Gemini Project Match Raw Output JSON:\n{raw_text}")
            logger.info(
                "synthesize  |  raw_response_length=%d",
                len(raw_text),
            )

            # 5 — Parse & validate
            # Deduplicate matched_evidence inside each raw dict BEFORE constructing
            # the frozen ProjectMatchResult — frozen instances cannot be mutated.
            raw_results: list[dict] = json.loads(raw_text)
            for item in raw_results:
                evidence = item.get("matched_evidence", [])
                seen: set[str] = set()
                item["matched_evidence"] = [
                    ev for ev in evidence
                    if ev not in seen and not seen.add(ev)
                ]

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
                reason=f"Gemini synthesis call failed: {exc}",
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
                await self._metrics.increment(self._model, _pt, _ct, "generate_sales_enablement")

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
            for name in filter(None, project_names):
                if len(name) >= 4 and any(name.lower() in t.lower() for t in leak_targets):
                    logger.warning(
                        "[Synthesizer] Internal project name '%s' leaked into sales "
                        "enablement output — redacting.",
                        name,
                    )
                    result = result.model_copy(update={
                        field: value.replace(name, "our recent work")
                        for field, value in (
                            ("outreach_template", result.outreach_template),
                            ("outreach_subject", result.outreach_subject),
                        )
                    } | {
                        "talking_points": [tp.replace(name, "our recent work") for tp in result.talking_points],
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
                reason=f"Gemini sales enablement call failed: {exc}",
            ) from exc

    # ------------------------------------------------------------------
    # Profile Matching Synthesis
    # ------------------------------------------------------------------

    _PROFILE_SYSTEM_INSTRUCTION = """\
You are an expert technical recruiter AI. Your task is to evaluate candidate
profile variants against a job description and determine the best matches.

You will receive:
1. A plain text job description (title, required skills, experience, etc.).
2. A list of candidate variant profiles. Each variant contains:
   - candidate_id, candidate_name, variant_id, variant_title
   - experience_years, tech_stacks, certifications
   - A list of past projects with project_name, domain, tech_stack, and description

For EACH variant, you must produce:
- candidate_id (str) — the candidate's ID
- candidate_name (str) — the candidate's name
- variant_id (str) — the variant ID
- variant_title (str) — the variant title
- experience_years (int) — years of experience
- match_percentage (int) — a score from 0 to 100 representing how well this
  variant matches the JD. Be precise and objective.
  SCORING RUBRIC — compute the score from these weighted factors, NOT gut feel:
  * Tech stack overlap (~60%): (JD required skills the candidate HAS) ÷
    (total JD required skills). This anchors the score — e.g. matching 3 of
    6 required skills must land near 50% before other factors adjust it.
    Two candidates with the SAME skill overlap MUST get the SAME base score.
  * Domain/industry relevance (~25%): adjust up/down from the base within
    ±15 points based on how closely the candidate's project domains match
    the JD's industry.
  * Experience level alignment (~15%): LEVELS map to years as
    JUNIOR=0-2, INTERMEDIATE=3-5, SENIOR=6-9, EXPERT/LEAD=10+.
    Penalise ONLY when the gap is clear: a candidate with far fewer years
    than the JD demands, or heavily over-qualified, scores lower — state
    the gap in the justification. If the JD states NO explicit years AND
    no level, do NOT penalise on experience at all.
  * Project complexity and relevance: fold into the domain adjustment.
- matching_skills (list[str]) — specific skills from the JD that this candidate HAS
- missing_skills (list[str]) — specific skills from the JD that this candidate LACKS
- justification (str) — a concise 2-3 sentence explanation of the match/mismatch.
  Reference specific projects or certifications as evidence. If a variant's
  profile data is sparse, say so plainly — do NOT invent evidence to fill gaps.

Availability/resource_status is NOT part of the match score — score purely on
skills, experience and domain fit.

Return a JSON array sorted by match_percentage descending.
Be precise and objective — do not inflate scores. A 90%+ match should only be
given when the candidate has nearly all required skills AND relevant domain experience.
CONSISTENCY CHECK before responding: if two candidates have identical
matching_skills and missing_skills lists, they MUST receive the same
match_percentage — never differentiate candidates by score when their
skill evidence is identical (mention any other differences in the
justification instead).
"""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def synthesize_profile_matches(
        self,
        job_details: str,
        variant_payloads: list[dict],
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
                projects = payload.get("projects", [])
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
                "## Candidate Profiles\n"
                f"{candidates_context}"
            )

            logger.info(
                "[Synthesizer] [synthesize_profile_matches] Sending to Gemini (model=%s) | variants=%d",
                self._model, len(variant_payloads),
            )
            logger.debug(f"Gemini Profile Match Input Prompt:\n{user_prompt}")
            _gemini_start = time.perf_counter()
            response = await self._generate_hedged(user_prompt, config=types.GenerateContentConfig(
                    system_instruction=self._PROFILE_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=list[ProfileMatchResult],
                    temperature=0.2,
                ),
            )
            _gemini_elapsed = time.perf_counter() - _gemini_start
            logger.info("[Synthesizer] [synthesize_profile_matches] Response in %.2fs", _gemini_elapsed)
            _usage = response.usage_metadata
            _pt = getattr(_usage, "prompt_token_count", 0) or 0
            _ct = getattr(_usage, "candidates_token_count", 0) or 0
            logger.info("[Synthesizer] [synthesize_profile_matches] Tokens: prompt=%d completion=%d total=%d", _pt, _ct, _pt + _ct)
            if self._metrics:
                await self._metrics.increment(self._model, _pt, _ct, "synthesize_profile_matches")

            raw_text = response.text
            logger.debug(f"Gemini Profile Match Raw Output JSON:\n{raw_text}")
            logger.info(
                "synthesize_profile_matches  |  raw_response_length=%d",
                len(raw_text),
            )

            raw_results: list[dict] = json.loads(raw_text)

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
                reason=f"Gemini profile synthesis failed: {exc}",
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
            projects = candidate_context.get("projects", [])
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
                await self._metrics.increment(self._model, _pt, _ct, "generate_technical_prep")

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
                reason=f"Gemini technical prep generation failed: {exc}",
            ) from exc

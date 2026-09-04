import datetime
import time
from google import genai
from google.genai import types
from src.domain.interfaces.extractor.i_extractor import IExtractor
from src.domain.value_objects.job_details import JobDetails
from src.domain.exceptions.scraper_exceptions import ExtractionFailedException
from src.common.config import settings
from src.common.logger import get_logger
from src.infrastructure.metrics.metrics_repository import MetricsRepository

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are an expert, highly thorough job data extraction AI.
Your task is to parse raw text from a job posting web page and extract a complete, detailed JSON representation.

Rules for extraction:
1. Extract ALL explicitly mentioned details without skipping information.
 2. 'title': Job title exactly as stated.
    TITLE HARD RULE — the page text frequently BEGINS with the logged-in viewer's own
    account/agency/organization name followed by platform chrome, BEFORE the real job
    headline (e.g. text starting "UMA Microapp\nGeneral writing project for All New
    beginners - Virtual Assistance\n..." → the title is "General writing project for
    All New beginners", NOT "UMA Microapp"). The viewer identity is the READER, never
    the job title. The real title is the job headline that follows the chrome —
    typically appearing right before a "Posted" line or a "Summary"/description
    section. If a first line also appears above unrelated account notices (tax,
    Connects balance, membership), it is viewer chrome: skip it and take the headline.
3. 'domain': The business vertical or industry domain this role falls under. This is NOT the same as industry.
   Generate a short, specific business vertical label such as: "SaaS", "FinTech", "HealthTech", "E-commerce",
   "EdTech", "Logistics", "HRTech", "PropTech", "AI / ML Platform", "CyberSecurity", "Gaming", "Media & Entertainment",
   "Supply Chain", "Enterprise Software", "Retail Tech", "Travel Tech", "LegalTech", "CleanTech", "InsurTech".
   Infer from the job title, company type, and description. Always return a value — do NOT return null.
4. 'company': The hiring company name — return null unless you are CERTAIN.
   CRITICAL — VIEWER IDENTITY TRAP: freelance platforms like Upwork show the LOGGED-IN VIEWER'S OWN
   account/agency name at the very top of the page, next to account notices (tax/TDS messages, Connects
   balance, membership prompts, wallet info). That name is the READER, not the hirer. NEVER extract a
   company name from the top of the page, from anything near account/tax/payment notices, or from the
   first line of the text. On Upwork, the client's real identity (if shown at all) appears ONLY under
   the "About the client" section — and it is very often completely hidden (showing just country, hire
   rate, member-since). When the client's company is not explicitly displayed there or in the job body,
   you MUST return null. Do not give job platform names like "LinkedIn" or "Indeed" as the company either.
5. 'industry': The industry or domain this role falls under (e.g. "Healthcare IT", "Fintech", "E-commerce", "SaaS"). Infer from the job context if not explicitly stated.
6. 'role': A standardized, clean role label — different from the raw job title. Normalize it (e.g. "Mobile Developer (Cross-Platform)", "Full Stack Engineer", "Data Scientist"). Strip company names or seniority noise from the title if needed.
7. 'location': The job location ONLY if explicitly stated in the posting (city, country, or "Remote"). Return null if the posting does not state a location — do NOT guess or infer one.
8. 'employment_type': The employment arrangement if explicitly stated (e.g. "Contract", "Full-time", "Part-time", "Hourly", "Freelance"). Return null if not mentioned.
9. 'duration': The project or contract length if mentioned (e.g. "3 to 6 months", "12 months", "ongoing"). Return null if not mentioned.
10. 'level': The seniority level. Must be exactly one of: JUNIOR, INTERMEDIATE, SENIOR, EXPERT, or LEAD. Infer from experience requirements, salary, or title if not explicitly stated.
11. 'posted_at': Provide the exact calendar date the job was published (e.g. "August 14, 2026"). Use the 'Current UTC time' provided in the prompt to calculate the exact absolute date if the webpage only shows a relative time (like "14 hours ago" or "2 days ago"). Do NOT output relative times like "14 hours ago" or "just now". Do NOT output raw ISO timestamps. Return null if not present or if you cannot calculate it.
    POSTED_AT HARD RULES — violating these makes the output INVALID:
    - posted_at may ONLY come from an explicit posting indicator visible on the page:
      a relative timestamp ("Posted 3 days ago", "Just posted", "30+ days ago"),
      a literal date string ("Posted on August 14, 2026"), or a date-type field.
    - If NO such posting indicator exists on the page, posted_at MUST be null.
      NEVER guess, estimate, or construct a date from any other page content.
    - NEVER copy a date that appears inside the job description body, company
      information, benefits, or requirements sections — those are NOT posting
      dates (e.g. an application deadline, a start date, or a date in a project
      description is NOT when the job was posted).
    - Sanity check before outputting: a job posting older than ~3 months is
      implausible; if your computed date is older than 3 months before the
      Current UTC time, output null instead.
    Worked example: the page shows no "Posted X ago" text and no literal posting
    date anywhere → the ONLY correct output is "posted_at": null.
12. 'required_skills': Extract the must-have, expected, or mandatory technical skills (languages, frameworks, libraries, databases, clouds, tools). If a specific skills section or tech stack is mentioned (even under broad headers like 'What we're looking for', 'What you'll do', 'Skills', 'Requirements', or 'Tech Stack'), extract those as required skills. e.g. ['React Native', 'REST APIs', 'PostgreSQL', 'Java', 'Python']. Do not invent or generalize skills.
13. 'preferred_skills': Extract technical skills that are explicitly mentioned as nice-to-have, optional, bonus, or preferred (e.g. 'Preferred', 'Nice to have', 'Bonus', 'Plus', 'Advantage'). If the job description does not clearly separate preferred/optional skills from the main expected skills, put all technical skills in 'required_skills' and leave 'preferred_skills' as an empty list [].
14. 'benefits': Extract any perk, healthcare, 401k, remote flexibility, or compensation benefit mentioned.
15. 'experience': Extract required years of experience if mentioned (e.g. '5+ years', '3-5 years').
16. 'salary': Extract pay rate, compensation range, or salary ONLY if numerical figures or exact amounts are mentioned (e.g. "$19.00 - $40.00 Hourly", "$100k"). Do NOT extract vague terms like "Competitive salary", "Negotiable", or "DOE". Return null if no numerical salary is found. NEVER mix freelance/engagement metadata (project type, connects required, client activity, payment status) into this field — salary must contain compensation figures only.
17. 'client_information': Extract the contact or client details block if present in the posting. This is a nested object with these fields — only populate fields that are explicitly stated:
    - 'name': The contact person's name (e.g. "JAMES")
    - 'company': The contact's company name (e.g. "TechNova Solutions")
    - 'role': The contact person's job title, exactly as stated (e.g. "Chief Executive Officer", "Human Resource Advisor")
    - 'designation': An abbreviated or acronym version of the 'role' if it can be shortened (e.g. "CEO", "HR Advisor", "TA"). If the role cannot be shortened meaningfully, leave null or duplicate the role.
    - 'email': Email address (e.g. "james@technovasolutions.com")
    - 'contact': Phone number or other contact info (e.g. "+13 456 483")
    - 'location': The contact's location (e.g. "London, UK")
    If no client contact information is found in the posting, return null for the entire client_information object.
18. 'apply_url': Extract any direct application URL or portal link if present. If not found, return null.
19. If a field is not present in the text, return an empty list [] for lists, or null for optional string fields. Do not return empty strings "". Do not fabricate details.
20. 'ai_job_summary': Generate a sharp, insightful 1 short paragraph narrative summary of this role — written as if advising a job seeker. Go beyond listing facts. Highlight what the client truly prioritizes, identify the most critical must-have skills, and flag anything uniquely important or unusual about this posting. Refer to the hirer as "the client" unless the 'company' field holds a confirmed company name — NEVER open the summary with an account/agency name taken from the top of the page. **CRITICAL**: Use ONLY plain text. Do NOT use markdown formatting, do NOT use stars or asterisks for bolding. Just plain unformatted text.
21. 'required_proposal_questions': Extract the exact list of questions the client requires applicants to answer when submitting a proposal or application. Return each question as a separate string in the list. If no such questions are found, return [].

FINAL SELF-CHECK BEFORE RESPONDING (mandatory):
- 'title': if the first line(s) of the text are the viewer's own account/agency/organization name or platform chrome, the title MUST be the job headline that FOLLOWS them — never the chrome itself.
- 'posted_at': if the page contains NO explicit posting indicator ("Posted X ago" or a literal posting date), it MUST be null. A fabricated date = invalid response.
- If you are not ≥95% certain a value appears in the source text, return null for that field.
- Never fabricate dates, names, emails, phone numbers, or salary figures. An honest null is always correct; an invented value is always wrong."""

_USER_PROMPT_TEMPLATE = """Current UTC time: {current_time}

Extract all job details from the following posting:

---
{job_text}
---

Return the data as a structured JSON object according to the schema."""

import asyncio
import re

# Exact-match UI chrome lines seen in cleaned page text (LinkedIn/Upwork/Indeed
# headers, footers, prompts). Dropping them cuts prompt tokens without ever
# touching job content — none of these phrases appear in real JD prose.
_BOILERPLATE_LINES = {
    "page header", "linkedin logo", "sign in", "join now", "dismiss",
    "show more", "show less", "welcome back", "not you?", "forgot password?",
    "new user?", "learn more", "get started", "download app",
    "add to wishlist", "report this job", "save job", "saved job",
    "easy apply", "apply now", "sign up", "log in", "menu", "search",
    "notifications", "messaging", "home", "jobs", "my items",
    "about", "accessibility", "help center", "privacy policy", "terms of service",
    "cookie policy", "your california privacy choices", "copyright policy",
    "feedback", "skip to main content", "skip to job description",
    "how hiring works", "browse jobs", "company insights",
    "reveal insights on any company", "explore other companies",
    "sign in to see more", "join conversation", "post a job",
    "here", "this link", "click here",
}

# Logged-in account/organization identity rendered by the platform's global
# navigation (e.g. Upwork's ng-sidebar-nav shows the account's org name on
# EVERY page — it leaked into extraction as the job title). These are read
# from settings so an org/name change only needs a .env edit, not a code
# change. Comma-separated, compared case-insensitively as exact line matches.
_CHROME_ACCOUNT_NAMES = tuple(
    n.strip().lower()
    for n in (settings.boilerplate_account_names or "").split(",")
    if n.strip()
)
_BOILERPLATE_LINES |= set(_CHROME_ACCOUNT_NAMES)

# Regex-based removal for multi-line / parameterised UI chrome that exact-line
# matching can't catch. Currently covers the logged-in Upwork header block
# (TDS tax notice, Connects/bid prompts) — none of it is job content, and the
# account name sitting next to it was being mis-extracted as the hiring company.
_BOILERPLATE_PATTERNS = tuple(re.compile(p, re.I) for p in (
    r"^in compliance with tax law",
    r"tax at deduction source",
    r"^add your pan",
    r"^faqs\.?$",
    r"upgrade your membership",
    r"^available connects:",
    r"^send a proposal for:",
    r"you'll need connects to bid",
    r"^learn more$",
    r"activate windows",
))


def _strip_boilerplate(text: str) -> str:
    """Drop exact-match UI chrome lines and known boilerplate patterns."""
    lines = text.splitlines()
    drop: set[int] = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() in _BOILERPLATE_LINES or any(
            p.search(stripped) for p in _BOILERPLATE_PATTERNS
        ):
            drop.add(i)
            # The logged-in viewer's own account/agency name sits directly
            # ABOVE these notice blocks (e.g. Upwork prints the user's name
            # right before the TDS tax notice). Drop that preceding short
            # standalone line too — Gemini kept mis-taking it for the hiring
            # company. Only short, non-sentence lines are eligible so real
            # job content is never eaten.
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            if j >= 0 and j not in drop:
                prev = lines[j].strip()
                if len(prev) <= 60 and not prev.endswith((".", ":", "?", "!")):
                    drop.add(j)
    cleaned = "\n".join(l for i, l in enumerate(lines) if i not in drop)
    # Collapse runs of blank lines left behind by removals
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned


def _sanitize_proposal_questions(job_details: "JobDetails") -> "JobDetails":
    """Clean UI noise out of required_proposal_questions.

    Upwork renders each screening question with numeric fragments attached
    (question index / '1 answer required' counters). When the page text is
    flattened those digits get glued onto the question text by the LLM
    (e.g. 'Updated Resume11'). Strip trailing/leading digit runs and drop
    entries that end up empty or purely numeric.
    """
    if not job_details.required_proposal_questions:
        return job_details
    cleaned: list[str] = []
    for q in job_details.required_proposal_questions:
        q = re.sub(r"[\s\d]+$", "", q)          # trailing whitespace/digits
        # Leading digits ONLY when they are UI noise: glued onto a word
        # ('11Updated') or a numbered-list marker ('1. Question').
        # A real sentence like '5 years experience needed' is preserved.
        q = re.sub(r"^\d{1,3}(?=[A-Za-z])", "", q)
        q = re.sub(r"^\d{1,3}[\.\)\-:]\s*", "", q)
        q = q.strip()
        if q and not q.isdigit():
            cleaned.append(q)
    return job_details.model_copy(
        update={"required_proposal_questions": cleaned}
    )


class GeminiExtractor(IExtractor):
    """
    Concrete implementation of IExtractor using Google Gemini.
    Uses structured output mode with a Pydantic schema to guarantee
    a valid JobDetails JSON response every time.
    """

    def __init__(self, metrics: MetricsRepository | None = None) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        self._metrics = metrics
        self._disable_thinking = settings.gemini_disable_thinking

    async def extract(self, cleaned_text: str) -> JobDetails:
        """
        Send cleaned job page text to Gemini and return a structured JobDetails object.

        Args:
            cleaned_text: The plain text of the job page after HTML cleaning.

        Returns:
            JobDetails value object.

        Raises:
            ExtractionFailedException: If Gemini fails or returns invalid data.
        """
        original_len = len(cleaned_text)
        cleaned_text = _strip_boilerplate(cleaned_text)
        if len(cleaned_text) != original_len:
            logger.info(
                f"[GeminiExtractor] [STEP 1] Input text: {original_len} chars — "
                f"boilerplate strip → {len(cleaned_text)} chars"
            )
        else:
            logger.info(f"[GeminiExtractor] [STEP 1] Input text: {len(cleaned_text)} chars")
        current_time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        # Head+tail truncation: proposal questions and benefits typically live
        # at the very END of a posting — a head-only slice loses them.
        max_chars = 25000
        if len(cleaned_text) > max_chars:
            tail_chars = 3000
            logger.warning(
                f"[GeminiExtractor] Input text {len(cleaned_text)} chars exceeds "
                f"{max_chars} — truncating (keeping head + last {tail_chars} chars)."
            )
            job_text = (
                cleaned_text[: max_chars - tail_chars]
                + "\n\n[...middle truncated...]\n\n"
                + cleaned_text[-tail_chars:]
            )
        else:
            job_text = cleaned_text
        prompt = _USER_PROMPT_TEMPLATE.format(
            current_time=current_time,
            job_text=job_text,
        )

        try:
            logger.info(f"[GeminiExtractor] [STEP 2] Sending prompt to Gemini (model={self._model})...")
            step2_start = time.perf_counter()

            response = await self._generate_with_retry(prompt)

            elapsed = time.perf_counter() - step2_start
            logger.info(f"[GeminiExtractor] [STEP 3] Response received in {elapsed:.2f}s")

            # ── Token usage logging ────────────────────────────────────────
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)
            logger.info(
                f"[GeminiExtractor] [STEP 4] Token usage: "
                f"prompt={prompt_tokens} | completion={completion_tokens} | total={total_tokens}"
            )

            # ── Persist metrics ────────────────────────────────────────────
            # F6: metrics persistence must NEVER fail an extraction that
            # already succeeded — a locked/read-only metrics file is not a
            # scrape failure. Failures are logged and swallowed.
            if self._metrics:
                try:
                    await self._metrics.increment(
                        model=self._model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        operation="extract_job",
                    )
                except Exception as metrics_err:
                    logger.warning(
                        f"[GeminiExtractor] Metrics persist failed (non-fatal): {metrics_err}"
                    )

            job_details = _sanitize_proposal_questions(
                JobDetails.model_validate_json(response.text)
            )
            logger.info(
                f"[GeminiExtractor] [STEP 5] Extraction complete: "
                f"'{job_details.title}' at '{job_details.company}'"
            )
            return job_details

        except Exception as e:
            logger.error(f"[GeminiExtractor] Extraction failed: {e}")
            raise ExtractionFailedException(reason=str(e)) from e

    async def _generate_with_retry(self, prompt: str):
        """
        Call Gemini with hedging. Identical prompts have been observed taking
        7s or 30s depending on Google's serving route, so if the first call
        hasn't answered within gemini_hedge_delay_s, a second identical call
        fires in parallel and the first response wins. The loser is cancelled.
        This bounds latency near the FAST route's time instead of the slow one.
        """
        hedge_delay = settings.gemini_hedge_delay_s
        call_timeout = settings.gemini_call_timeout_s

        primary = asyncio.ensure_future(self._call_gemini(prompt))
        if hedge_delay > 0:
            try:
                return await asyncio.wait_for(asyncio.shield(primary), timeout=hedge_delay)
            except asyncio.TimeoutError:
                logger.info(
                    f"[GeminiExtractor] First call exceeded {hedge_delay}s — "
                    f"hedging with a parallel request..."
                )
            except Exception:
                raise  # real error (not slowness) — no point hedging
        else:
            # F7: "0 disables" hedging per config docs — but the primary call
            # still needs an overall deadline or a hung socket stalls forever.
            return await asyncio.wait_for(asyncio.shield(primary), timeout=call_timeout)

        secondary = asyncio.ensure_future(self._call_gemini(prompt))
        # F7: bound the WHOLE hedged pair — previously asyncio.wait had no
        # deadline, so two simultaneously-hung calls stalled the scrape
        # forever. The loser-wait keeps its own (generous) bound.
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
                # hung request can't stall the scrape forever.
                logger.warning(
                    f"[GeminiExtractor] Hedged call failed ({winner.exception()!r}); "
                    f"awaiting the other request..."
                )
                result = await asyncio.wait_for(
                    asyncio.shield(loser), timeout=loser_wait_timeout
                )
            else:
                loser.cancel()
                result = winner.result()
            return result
        finally:
            primary.cancel()
            secondary.cancel()

    async def _call_gemini(self, prompt: str):
        """Single Gemini call with thinking disabled when supported.

        Retried by tenacity on HARD failures (503 overload, network errors) —
        the hedging in _generate_with_retry covers slow routes, not errors.
        """
        config_kwargs: dict = {}
        if self._disable_thinking:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        config = types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=JobDetails,
            temperature=0.1,
            **config_kwargs,
        )
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            reraise=True,
        )
        async def _attempt() -> types.GenerateContentResponse:
            return await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )

        try:
            return await _attempt()
        except Exception as e:
            if "thinking" in str(e).lower() and self._disable_thinking:
                logger.warning(
                    f"[GeminiExtractor] Model rejected thinking_config ({e}). "
                    f"Retrying without it..."
                )
                self._disable_thinking = False
                return await self._call_gemini(prompt)
            raise

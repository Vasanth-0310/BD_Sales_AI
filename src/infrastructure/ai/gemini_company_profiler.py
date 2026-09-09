import asyncio
import contextvars
import time
from google import genai
from google.genai import types
from src.domain.value_objects.company_profile import CompanyProfile
from src.common.config import settings
from src.common.logger import get_logger
from src.infrastructure.metrics.metrics_repository import MetricsRepository

logger = get_logger(__name__)

# How many DuckDuckGo results to feed into Gemini as context
_DDG_MAX_RESULTS = 3

# Company profiles rarely change — cache them so repeat scrapes of the same
# company skip both the DDG search and the Gemini call (saves ~20s per repeat).
_PROFILE_CACHE_TTL_SEC = 7 * 24 * 3600  # 7 days
_PROFILE_CACHE_MAX = 500

_COMPANY_SEARCH_PROMPT = """You are a professional business intelligence researcher.

Your task is to generate a structured company profile for "{company_name}".

You have access to web search snippets below. Use them as your primary source of truth.
If the snippets contain enough information about a field, use that.
If the snippets are sparse or don't cover a field, use your own knowledge about this company to fill it in.
Only return null for a field if you genuinely have no information at all — either from the snippets or your own knowledge.

ENTITY MATCHING (critical):
First verify that the snippets are actually about a company called "{company_name}".
If the snippets appear to be about a DIFFERENT company that merely has a similar name
(e.g. the search matched a better-known company sharing part of the name), you MUST
return null for every field — a wrong profile for a similarly-named company is worse
than no profile at all.

--- SEARCH SNIPPETS ---
{search_snippets}
--- END SNIPPETS ---

Generate a structured company profile with ONLY the following fields:
- overview: A concise 2-3 sentence description of what the company does and its mission.
- industry: The primary industry or sector (e.g. "Technology / Mobile Software", "Healthcare IT", "SaaS / Fintech", "BPO").
- products_services: The main products or services they offer. Keep it concise (1-2 sentences).
- headquarters: The HQ location as a structured object with four fields: 'raw' (the location text exactly as found), 'city', 'state', 'country'. Populate only the components actually present. Example: for HQ "Milpitas, California, USA" return raw="Milpitas, California, USA", city="Milpitas", state="California", country="USA". For work-mode-only values like "Remote / US Timezone Preferred" set only 'raw'. Return null if HQ is unknown.
- location: The country or region the company primarily operates in (e.g. "UK", "USA", "Romania", "Global").

Return valid JSON only."""


class GeminiCompanyProfiler:
    """
    Generates a structured CompanyProfile by:
    1. Fetching live web snippets via DuckDuckGo (free, no quota).
    2. Feeding those snippets into a standard Gemini prompt (unrestricted free tier).

    This completely avoids the restrictive Google Search Grounding quota.
    """

    def __init__(self, metrics: MetricsRepository | None = None) -> None:
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._metrics = metrics
        self._profile_cache: dict[str, tuple[float, CompanyProfile]] = {}
        self._disable_thinking = settings.gemini_disable_thinking

    def _search_company(self, company_name: str) -> str:
        """
        Uses DuckDuckGo to fetch top search results for the company name.
        Returns a formatted string of snippets for Gemini to read.
        """
        query = f"{company_name} company overview headquarters industry"
        snippets = []
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=_DDG_MAX_RESULTS))
                for r in results:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    href = r.get("href", "")
                    snippets.append(f"Source: {href}\nTitle: {title}\nSnippet: {body}")
        except ImportError:
            logger.warning("ddgs is not installed; skipping company profile search.")
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed for '{company_name}': {e}")

        if not snippets:
            return "No search results found."
        return "\n\n".join(snippets)

    async def profile(self, company_name: str) -> CompanyProfile | None:
        """
        Search the web for the given company name and return a structured CompanyProfile.
        Returns None if profiling fails or company_name is empty/unknown.
        """
        if not company_name or company_name.lower() in ("not specified", "unknown", "n/a", "confidential", ""):
            logger.info(f"[GeminiCompanyProfiler] Skipping profile for unresolved name: '{company_name}'")
            return None

        cache_key = company_name.strip().lower()
        cached = self._profile_cache.get(cache_key)
        if cached:
            cached_at, cached_profile = cached
            if time.time() - cached_at < _PROFILE_CACHE_TTL_SEC:
                logger.info(
                    f"[GeminiCompanyProfiler] Cache hit for '{company_name}' "
                    f"(age {int((time.time() - cached_at) / 3600)}h) — skipping search + Gemini."
                )
                return cached_profile
            self._profile_cache.pop(cache_key, None)

        logger.info(f"[GeminiCompanyProfiler] [STEP 1] Starting company profile for: '{company_name}'")

        # Step 2: Get live search snippets from DuckDuckGo (free, no quota).
        # Hard 6s cap — the search is a nice-to-have context boost, never
        # worth blocking the scrape on a slow/timeout search.
        try:
            # copy_context so logs inside the worker thread still carry the
            # request's user_id/action (threads don't inherit ContextVars).
            ctx = contextvars.copy_context()
            search_snippets = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: ctx.run(self._search_company, company_name)
                ),
                timeout=6,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[GeminiCompanyProfiler] Web search timed out for '{company_name}' "
                f"(6s) — profiling from model knowledge only."
            )
            search_snippets = "No search results found."
        except Exception as e:
            logger.warning(
                f"[GeminiCompanyProfiler] Web search failed for '{company_name}': {e} — "
                f"profiling from model knowledge only."
            )
            search_snippets = "No search results found."
        logger.info(
            f"[GeminiCompanyProfiler] [STEP 2] Search snippets for '{company_name}': "
            f"{'none (model knowledge only)' if search_snippets == 'No search results found.' else 'retrieved'}"
        )

        # Step 3: Ask Gemini to synthesize the snippets into a structured profile
        prompt = _COMPANY_SEARCH_PROMPT.format(
            company_name=company_name,
            search_snippets=search_snippets,
        )

        try:
            logger.info(f"[GeminiCompanyProfiler] [STEP 3] Sending to Gemini (model={settings.gemini_model})...")
            step3_start = time.perf_counter()

            response = await self._generate_with_retry(prompt)
            elapsed = time.perf_counter() - step3_start
            logger.info(f"[GeminiCompanyProfiler] [STEP 4] Response received in {elapsed:.2f}s")

            # ── Token usage logging ───────────────────────────────────────
            usage = response.usage_metadata
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)
            logger.info(
                f"[GeminiCompanyProfiler] [STEP 5] Token usage: "
                f"prompt={prompt_tokens} | completion={completion_tokens} | total={total_tokens}"
            )

            if self._metrics:
                try:
                    await self._metrics.increment(
                        model=settings.gemini_model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        operation="profile_company",
                    )
                except Exception as metrics_err:
                    logger.warning(f"Metrics persist failed (non-fatal): {metrics_err}")

            profile = CompanyProfile.model_validate_json(response.text)

            if len(self._profile_cache) >= _PROFILE_CACHE_MAX:
                # Simple eviction: drop the oldest entries.
                oldest = sorted(self._profile_cache.items(), key=lambda kv: kv[1][0])
                for k, _ in oldest[: _PROFILE_CACHE_MAX // 10]:
                    self._profile_cache.pop(k, None)
            self._profile_cache[cache_key] = (time.time(), profile)

            logger.info(f"[GeminiCompanyProfiler] [STEP 6] Profile complete for: '{company_name}'")
            return profile

        except Exception as e:
            logger.warning(f"[GeminiCompanyProfiler] Profiling failed for '{company_name}': {e}")
            return None

    async def _generate_with_retry(self, prompt: str):
        """
        Call Gemini with hedging — see GeminiExtractor for rationale. This
        prompt is tiny, so latency is pure serving-route variance.
        """
        hedge_delay = settings.gemini_hedge_delay_s

        primary = asyncio.ensure_future(self._call_gemini(prompt))
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
            logger.info(
                f"[GeminiCompanyProfiler] First call exceeded {hedge_delay}s — "
                f"hedging with a parallel request..."
            )
        except Exception:
            raise

        secondary = asyncio.ensure_future(self._call_gemini(prompt))
        # Bound the WHOLE hedged pair — previously asyncio.wait had no
        # deadline, so two simultaneously-hung calls stalled the profile
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
                logger.warning(
                    f"[GeminiCompanyProfiler] Hedged call failed ({winner.exception()!r}); "
                    f"awaiting the other request..."
                )
                return await asyncio.wait_for(
                    asyncio.shield(loser), timeout=loser_wait_timeout
                )
            loser.cancel()
            return winner.result()
        finally:
            primary.cancel()
            secondary.cancel()

    async def _call_gemini(self, prompt: str):
        config_kwargs: dict = {}
        if self._disable_thinking:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CompanyProfile,
            temperature=0.1,
            **config_kwargs,
        )
        try:
            return await self._client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            if "thinking" in str(e).lower() and self._disable_thinking:
                logger.warning(f"[GeminiCompanyProfiler] thinking_config rejected ({e}). Retrying without it...")
                self._disable_thinking = False
                return await self._call_gemini(prompt)
            raise

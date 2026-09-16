import time
import uuid
import asyncio
from src.domain.interfaces.rag.i_embedding_port import IEmbeddingPort
from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.domain.interfaces.rag.i_synthesizer_port import ISynthesizerPort
from src.domain.exceptions.rag_exceptions import RAGBaseException
from src.common.bm25_rescorer import BM25Rescorer, rrf_merge
from src.common.jd_text import build_retrieval_query, extract_jd_keywords
from src.common.config import settings
from src.application.services.profile_match_engine import (
    build_job_requirement_plan,
    evaluate_profile,
    no_qualified_profiles_reason,
    profile_role_priority,
)
from src.application.dto.profile_dto import (
    ProfileMatchRequestDTO,
    ProfileMatchResponseDTO,
)
from src.common.logger import get_logger
from src.infrastructure.evaluation import pipeline_tracer as _tracer

logger = get_logger(__name__)

# Retrieval is deliberately wider than the final result. A relevant profile
# must reach Gemini before it can be scored; top-10 retrieval plus an 8-item
# synthesis pool was dropping valid candidates before semantic evaluation.
_PROFILE_RETRIEVAL_TOP_K = 30
_GEMINI_POOL_SIZE = 30

# Lower number = preferred when RRF scores tie exactly.
_STATUS_PRIORITY = {
    "On Bench": 1,
    "Proposed to Client": 2,
    "Upskilling": 3,
    "On Project": 4,
}


class MatchProfilesUseCase:
    """
    Application use case that orchestrates the full profile matching pipeline:

    1. Embed JD → query vector
    2. Dense search on profile_variants → Top 10
    3. Keyword (BM25) search on profile_variants → Top 10
    4. Merge & BM25 Rescore + RRF → Top 10
    5. Send full variant payloads + JD to Gemini
    6. Gemini scores each → match_percentage, matching_skills, missing_skills
    7. Deduplicate by candidate_id AFTER Gemini (keep best variant per person)
    8. Return Top 5
    """

    def __init__(
        self,
        embedding_port: IEmbeddingPort,
        vector_store_port: IVectorStorePort,
        synthesizer_port: ISynthesizerPort,
    ) -> None:
        self._embedding_port = embedding_port
        self._vector_store_port = vector_store_port
        self._synthesizer_port = synthesizer_port

    async def execute(self, dto: ProfileMatchRequestDTO) -> ProfileMatchResponseDTO:
        total_start = time.perf_counter()
        logger.info("======== PROFILE MATCHING PIPELINE STARTED ========")

        try:
            job_details = dto.job_details

            # Empty JD would fail late inside the embedding call with a
            # cryptic "empty Part" 400 — reject it here with a clear message.
            if not job_details or not job_details.strip():
                return ProfileMatchResponseDTO.failed(
                    reason="job_details must not be empty."
                )

            retrieval_query = build_retrieval_query(job_details)
            # A JSON JD with unfamiliar field names can legitimately produce
            # no curated query parts. Never forward an empty embedding request
            # to Gemini; fall back to the supplied content for safe recall.
            if not retrieval_query.strip():
                retrieval_query = job_details.strip()
            logger.info(
                "[STEP 1] Query text ready  |  raw_chars=%d  retrieval_chars=%d",
                len(job_details), len(retrieval_query),
            )

            # ── Early-exit: Manual variant selection ─────────────────────
            # Only enter the manual path when variant_id is a valid UUID.
            # Any other value (None, "", "null", "string", random text) falls
            # through to the normal auto-match pipeline.
            if dto.variant_id and self._is_valid_uuid(dto.variant_id.strip()):
                return await self._execute_manual_match(
                    job_details, dto.variant_id.strip(), user_id=dto.user_id or None
                )

            # ── Step 2: Embed JD ─────────────────────────────────────────
            logger.info("[STEP 2] Embedding JD via Gemini")
            embed_start = time.perf_counter()
            query_vector = await self._embedding_port.embed_query(retrieval_query)
            embed_time = time.perf_counter() - embed_start
            logger.info("[STEP 2] JD embedding completed in %.2fs", embed_time)

            # ── Step 3: Hybrid search (Dense + Keyword) in parallel ──────
            logger.info("[STEP 3] Hybrid retrieval (Dense + Keyword) on profile_variants")
            stage1_start = time.perf_counter()

            # Extract focused keywords from the JD for Qdrant's MatchText.
            # Qdrant MatchText does an AND on every token — passing the whole
            # JD (96+ tokens) guarantees 0 results.  We extract short, high-
            # signal terms (tech names, role titles) and pass them one at a time
            # as separate OR conditions inside the adapter.
            # Keep the original structured JD here. Passing the flattened
            # embedding query loses field boundaries and can omit skills stored
            # under alternate structured keys.
            keyword_query = self._extract_keywords(job_details)
            logger.debug("[STEP 3] Keyword query terms: %s", keyword_query)

            dense_results, keyword_results = await asyncio.gather(
                self._vector_store_port.search_profile_variants_dense(
                    query_vector,
                    top_k=_PROFILE_RETRIEVAL_TOP_K,
                    user_id=dto.user_id or None,
                ),
                self._vector_store_port.search_profile_variants_keyword(
                    keyword_query,
                    top_k=_PROFILE_RETRIEVAL_TOP_K,
                    user_id=dto.user_id or None,
                ),
            )

            retrieval_time = time.perf_counter() - stage1_start
            
            # Log retrieved items (skip malformed/legacy points missing variant_id)
            dense_vids = [r["payload"]["variant_id"] for r in dense_results
                          if isinstance(r.get("payload"), dict) and r["payload"].get("variant_id")]
            keyword_vids = [r["payload"]["variant_id"] for r in keyword_results
                            if isinstance(r.get("payload"), dict) and r["payload"].get("variant_id")]
            logger.debug(f"Retrieved Dense Variant IDs: {dense_vids}")
            logger.debug(f"Retrieved Keyword Variant IDs: {keyword_vids}")

            logger.info(
                "[STEP 3] Retrieval completed in %.2fs  |  dense=%d  keyword=%d",
                retrieval_time,
                len(dense_results),
                len(keyword_results),
            )

            # ── Step 4: Merge & deduplicate by variant_id ────────────────
            logger.info("[STEP 4] Merging and deduplicating retrieval results")
            candidates_map: dict[str, dict] = {}
            for result in dense_results:
                payload = result.get("payload") or {}
                vid = payload.get("variant_id")
                if vid:
                    candidates_map[vid] = result
            for result in keyword_results:
                payload = result.get("payload") or {}
                vid = payload.get("variant_id")
                if vid and vid not in candidates_map:
                    candidates_map[vid] = result

            candidates = list(candidates_map.values())
            logger.info(
                "[STEP 4] %d unique variants after merge", len(candidates)
            )
            plan = build_job_requirement_plan(job_details)

            if not candidates:
                hint = no_qualified_profiles_reason(plan, [])
                logger.info("No candidates found. %s", hint)
                return ProfileMatchResponseDTO(
                    status="SUCCESS", matches=[], error_message=hint,
                )

            # ── Step 5: BM25 Rescore + RRF ───────────────────────────────
            logger.info("[STEP 5] BM25 Rescoring")
            rrf_start = time.perf_counter()
            bm25_scores = BM25Rescorer.rescore(
                candidates,
                retrieval_query,
                # combined_text (built at ingest) already contains title,
                # stacks, certifications and project text — the extra fields
                # cover variants ingested before combined_text existed.
                text_fields=("combined_text", "variant_title"),
                list_fields=("tech_stacks", "certifications"),
                id_field="variant_id",
            )

            dense_ranking = [
                (r["payload"]["variant_id"], r["score"])
                for r in sorted(dense_results, key=lambda x: x["score"], reverse=True)
                if isinstance(r.get("payload"), dict) and r["payload"].get("variant_id")
            ]
            bm25_ranking = [
                (vid, score)
                for vid, score in sorted(
                    bm25_scores.items(), key=lambda x: x[1], reverse=True
                )
            ]

            logger.debug(f"RRF Input - Dense Ranking: {dense_ranking}")
            logger.debug(f"RRF Input - BM25 Ranking: {bm25_ranking}")

            logger.info("[STEP 6] RRF merge (relevance primary, status tiebreak)")
            rrf_results = rrf_merge(dense_ranking, bm25_ranking, k=60)

            # Relevance is PRIMARY: rank by fused RRF score descending.
            # Resource status is only a TIEBREAKER — an On Bench variant no
            # longer outranks a far better-matching On Project variant.
            def sort_key(item):
                vid, score = item
                payload = candidates_map.get(vid, {}).get("payload", {})
                priority = _STATUS_PRIORITY.get(payload.get("resource_status", ""), 5)
                return (-score, priority)

            rrf_results.sort(key=sort_key)

            # Evaluate every retrieved variant locally, then retain each
            # person's best-evidenced variant.  Selecting the first retrieved
            # variant used to hide a person's relevant Java/React/etc. project
            # when another, weaker variant happened to rank first in RRF.
            if not rrf_results:
                rrf_results = [
                    (vid, 0.0) for vid in candidates_map
                ]
            previews_by_variant = {}
            best_variant_by_candidate: dict[str, tuple[str, int, int, int]] = {}
            role_priority_by_variant: dict[str, int] = {}
            for rank, (vid, _score) in enumerate(rrf_results):
                payload = candidates_map.get(vid, {}).get("payload") or {}
                cid = payload.get("candidate_id", "") or vid
                preview = evaluate_profile(plan, payload)
                previews_by_variant[vid] = preview
                role_priority = profile_role_priority(plan, payload)
                role_priority_by_variant[vid] = role_priority
                candidate = (vid, preview.match_percentage, rank, role_priority)
                existing = best_variant_by_candidate.get(cid)
                if existing is None or (
                    candidate[3], -candidate[1], candidate[2]
                ) < (
                    existing[3], -existing[1], existing[2]
                ):
                    best_variant_by_candidate[cid] = candidate

            selected = sorted(
                best_variant_by_candidate.values(),
                # For a frontend/backend JD, direct-role profiles are the
                # primary pool. Full-stack profiles are considered next, then
                # explicit cross-role transitions. Score orders each tier.
                key=lambda item: (item[3], -item[1], item[2]),
            )[:_GEMINI_POOL_SIZE]
            top_variant_ids = [vid for vid, _score, _rank, _priority in selected]

            rrf_time = time.perf_counter() - rrf_start

            logger.debug(f"RRF Output - Full Results (ID, Score): {rrf_results}")

            logger.info(
                "[STEP 6] RRF completed in %.2fs  |  top %d variant_ids=%s",
                rrf_time,
                len(top_variant_ids),
                top_variant_ids,
            )
            logger.info(
                "[STEP 6] Pool diversity: %d people from %d best-evidence variants  |  "
                "rrf_total=%d",
                len(best_variant_by_candidate),
                len(top_variant_ids),
                len(rrf_results),
            )

            # ── Step 7: Collect full payloads for the Gemini pool ────────
            logger.info("[STEP 7] Building one JD plan and verifying candidate evidence")
            gemini_payloads = []
            for vid in top_variant_ids:
                if vid not in candidates_map:
                    continue
                payload = dict(candidates_map[vid]["payload"])  # shallow copy
                gemini_payloads.append(payload)
                logger.debug("Profile payload prepared  |  variant_id=%s", payload.get("variant_id"))

            # ── Step 8: Gemini synthesis ─────────────────────────────────
            match_results = [previews_by_variant[payload["variant_id"]] for payload in gemini_payloads]
            eligible_results = [result for result in match_results if result.match_percentage > 0]
            eligible_results.sort(key=lambda result: result.match_percentage, reverse=True)
            logger.info(
                "[STEP 7] Evidence evaluation complete | role_family=%s critical=%s eligible=%d/%d",
                plan.role_family or "unspecified", plan.critical,
                len(eligible_results), len(match_results),
            )
            # Do not expose a free-form model explanation unless every claim
            # can be separately verified. The deterministic justification is
            # already auditable, so matching completes without a generative
            # request or a Gemini availability dependency.
            logger.info(
                "[STEP 8] Deterministic finalization; no generative scoring or explanation call",
            )
            gemini_time = 0.0
            # ── RAGAS capture (background, non-blocking) ─────────────────
            # Build context strings from gemini_payloads — the exact data the
            # synthesiser formatted and sent to Gemini — so RAGAS can measure
            # whether the retrieval phase surfaced relevant profiles.
            # ── Enrich results with payload metadata ────────────────────

            # Gemini only returns what we put in the prompt. Metadata is in payload.
            variant_meta_map: dict[str, dict] = {
                p["variant_id"]: {
                    "email": p.get("email", ""),
                    "role": p.get("role", ""),
                    "resource_status": p.get("resource_status", ""),
                }
                for p in gemini_payloads
                if isinstance(p, dict) and p.get("variant_id")
            }
            for result in match_results:
                meta = variant_meta_map.get(result.variant_id, {})
                result.email = meta.get("email", "")
                result.role = meta.get("role", "")

            # ── Step 9: Deduplicate by candidate_id AFTER Gemini ─────────
            # If same candidate appears with multiple variants, keep only the
            # variant with the highest match_percentage.
            logger.info("[STEP 9] Deduplicating by candidate_id (post-Gemini)")
            seen_candidates: dict[str, int] = {}  # candidate_id -> index in deduped list
            deduped = []
            for result in match_results:
                # Use variant_id as a unique key when candidate_id is missing
                # so different unknown variants don't collapse into one bucket.
                cid = result.candidate_id or result.variant_id
                if cid not in seen_candidates:
                    seen_candidates[cid] = len(deduped)
                    deduped.append(result)
                else:
                    # Already seen — keep the one with higher match_percentage
                    existing_idx = seen_candidates[cid]
                    existing = deduped[existing_idx]
                    if result.match_percentage > existing.match_percentage:
                        logger.info(
                            "[STEP 9] Dedup: %s variant %s (%d%%) replaced by variant %s (%d%%)",
                            result.candidate_name, existing.variant_id,
                            existing.match_percentage, result.variant_id,
                            result.match_percentage,
                        )
                        deduped[existing_idx] = result
                    else:
                        logger.info(
                            "[STEP 9] Dedup: %s variant %s (%d%%) dropped (kept %s at %d%%)",
                            result.candidate_name, result.variant_id,
                            result.match_percentage, existing.variant_id,
                            existing.match_percentage,
                        )

            logger.info(
                "[STEP 9] %d unique candidates after deduplication (from %d variants)",
                len(deduped),
                len(match_results),
            )
            for r in deduped:
                logger.info(
                    "[STEP 9] Scored: %s (%s) → %d%%  |  variant=%s",
                    r.candidate_name, r.candidate_id, r.match_percentage, r.variant_id,
                )

            # ── Step 9.5: Re-sort deterministically (mirror match_projects) ──
            # Gemini is asked to return sorted output, but its ordering is not
            # guaranteed — same flaw fixed in match_projects. Python re-sorts
            # by match_percentage so the top-5 slice is always correct, with
            # the availability status as an EXACT-percentage tiebreaker
            # (On Bench wins ties — the same policy as the RRF sort above).
            def _tiebreak_key(r):
                meta = variant_meta_map.get(r.variant_id, {})
                priority = _STATUS_PRIORITY.get(meta.get("resource_status", ""), 5)
                return (
                    role_priority_by_variant.get(r.variant_id, 1),
                    -r.match_percentage,
                    priority,
                )

            deduped.sort(key=_tiebreak_key)

            # ── Step 9.6: Minimum-quality threshold ────────────────────
            # Weak matches below this percentage are not meaningful
            # recommendations — mirrors project_match_min_score. Set to 0
            # in settings to disable the filter entirely.
            min_pct = settings.profile_match_min_percentage
            pre_filter_pool = list(deduped)  # closest-match hint needs the unfiltered pool
            if min_pct > 0:
                before = len(deduped)
                filtered_out = [r for r in deduped if r.match_percentage < min_pct]
                deduped = [r for r in deduped if r.match_percentage >= min_pct]
                if filtered_out:
                    logger.info(
                        "[STEP 9.6] %d candidate(s) below min threshold (%d%%) filtered; "
                        "%d remain",
                        len(filtered_out), min_pct, len(deduped),
                    )
                    for r in filtered_out:
                        logger.info(
                            "[STEP 9.6] Filtered: %s → %d%% (below %d%%)",
                            r.candidate_name, r.match_percentage, min_pct,
                        )

            # ── Step 10: Return Top 5 unique candidates ────────────────
            top_matches = deduped[:5]

            # RAGAS must evaluate one answer against that candidate's own
            # evidence. Capturing a mixed ranking creates false faithfulness
            # and context-precision failures, so only final API matches are
            # recorded, one sample per candidate.
            payload_by_variant = {
                str(payload.get("variant_id")): payload
                for payload in gemini_payloads
                if isinstance(payload, dict) and payload.get("variant_id")
            }
            for match in top_matches:
                payload = payload_by_variant.get(str(match.variant_id), {})
                contexts = [
                    (
                        f"Candidate: {payload.get('candidate_name', match.candidate_name)}\n"
                        f"Title: {payload.get('variant_title', '')}\n"
                        f"Experience: {payload.get('experience_years', '')} years\n"
                        f"Tech stacks: {', '.join(payload.get('tech_stacks', []) or [])}\n"
                        f"Certifications: {', '.join(payload.get('certifications', []) or []) or 'None'}\n"
                        f"Profile evidence: {str(payload.get('combined_text', ''))[:1000]}"
                    )
                ]
                for project in (payload.get("projects") or [])[:3]:
                    if not isinstance(project, dict):
                        continue
                    project_tech = project.get("tech_stack", []) or []
                    contexts.append(
                        f"Project: {project.get('project_name', '')}\n"
                        f"Domain: {project.get('domain', '')}\n"
                        f"Tech stacks: {', '.join(project_tech) if isinstance(project_tech, list) else project_tech}\n"
                        f"Evidence: {str(project.get('description', ''))[:800]}"
                    )
                asyncio.create_task(
                    _tracer.capture_profile_sample(
                        question=job_details,
                        retrieval_contexts=contexts,
                        answer=(
                            f"{match.candidate_name} ({match.match_percentage}%): "
                            f"{match.justification}"
                        ),
                        retrieval_meta={
                            "format": "candidate_evidence_v2",
                            "candidate_id": match.candidate_id,
                            "variant_id": match.variant_id,
                            "dense_count": len(dense_results),
                            "keyword_count": len(keyword_results),
                            "merged_count": len(candidates),
                            "rrf_total": len(rrf_results),
                        },
                        generation_meta={
                            "returned_match_percentage": match.match_percentage,
                            "sent_to_gemini": len(gemini_payloads),
                            "scored_by_gemini": len(match_results),
                        },
                    )
                )

            total_time = time.perf_counter() - total_start
            logger.info(
                "======== PROFILE MATCHING COMPLETE in %.2fs ========\n"
                "Stats: Embed=%.2fs | Retrieval=%.2fs | RRF=%.2fs | Gemini=%.2fs",
                total_time,
                embed_time,
                retrieval_time,
                rrf_time,
                gemini_time,
            )

            if not top_matches:
                hint = no_qualified_profiles_reason(plan, pre_filter_pool)
                logger.info("[STEP 10] No evidence-qualified profile found: %s", hint)
                return ProfileMatchResponseDTO(
                    status="SUCCESS",
                    matches=[],
                    error_message=hint,
                )

            return ProfileMatchResponseDTO.success(top_matches)

        except RAGBaseException as e:
            logger.error("Profile matching pipeline failed: %s", e)
            return ProfileMatchResponseDTO.failed(reason=str(e))
        except Exception as e:
            logger.error("Unexpected error in profile matching: %s", e, exc_info=True)
            return ProfileMatchResponseDTO.failed(reason="Internal error: please check server logs.")


    @staticmethod
    def _extract_keywords(jd_text: str) -> str:
        return extract_jd_keywords(jd_text, max_keywords=40)

    @staticmethod
    def _is_valid_uuid(value: str) -> bool:
        """Return True only if value is a properly formatted UUID.
        Rejects plain strings like 'null', 'none', 'string', etc.
        """
        try:
            uuid.UUID(value)
            return True
        except (ValueError, AttributeError):
            return False

    async def _execute_manual_match(
        self, job_details: str, variant_id: str, user_id: str = ""
    ) -> ProfileMatchResponseDTO:
        """
        Manual path: skip all retrieval (embed / dense / keyword / BM25 / RRF).
        Fetch one specific variant from Qdrant by variant_id, run Gemini
        synthesis on it alone, return a single-item matches list.

        Intentionally bypasses profile_match_min_percentage — manual selection
        is a human decision; the threshold must not override the user's choice.
        """
        logger.info(
            "======== MANUAL PROFILE MATCH STARTED | variant_id=%s ========",
            variant_id,
        )
        start = time.perf_counter()

        try:
            # Step 1: Fetch the variant payload directly from Qdrant
            logger.info("[STEP 1] Fetching variant from Qdrant | variant_id=%s", variant_id)
            payload = await self._vector_store_port.fetch_profile_variant_by_id(
                variant_id, user_id=user_id,
            )

            if payload is None:
                logger.warning("[STEP 1] variant_id=%s not found in Qdrant", variant_id)
                return ProfileMatchResponseDTO.failed(
                    reason=f"Variant '{variant_id}' not found in the knowledge base."
                )

            logger.info(
                "[STEP 1] Variant fetched | candidate=%s  title=%s",
                payload.get("candidate_name", "unknown"),
                payload.get("variant_title", ""),
            )

            # Step 2: score from stored evidence. Manual selection returns the
            # requested profile even when it is ineligible, but it never gets a
            # fabricated score or skill match from a model.
            logger.info("[STEP 2] Building JD plan and verifying selected profile evidence")
            result = evaluate_profile(build_job_requirement_plan(job_details), payload)

            logger.info(
                "======== MANUAL PROFILE MATCH COMPLETE in %.2fs | match=%d%% ========",
                time.perf_counter() - start,
                result.match_percentage,
            )
            return ProfileMatchResponseDTO.success([result])

        except RAGBaseException as e:
            logger.error("Manual profile match failed: %s", e)
            return ProfileMatchResponseDTO.failed(reason=str(e))
        except Exception as e:
            logger.error("Unexpected error in manual profile match: %s", e, exc_info=True)
            return ProfileMatchResponseDTO.failed(reason="Internal error: please check server logs.")

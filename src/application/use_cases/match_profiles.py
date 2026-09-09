import time
import uuid
import asyncio
from src.domain.interfaces.rag.i_embedding_port import IEmbeddingPort
from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.domain.interfaces.rag.i_synthesizer_port import ISynthesizerPort
from src.domain.exceptions.rag_exceptions import RAGBaseException
from src.infrastructure.db.qdrant.bm25_rescorer import BM25Rescorer, rrf_merge
from src.common.jd_text import extract_jd_keywords
from src.common.config import settings
from src.application.dto.profile_dto import (
    ProfileMatchRequestDTO,
    ProfileMatchResponseDTO,
)
from src.common.logger import get_logger

logger = get_logger(__name__)

# How many variants to send to Gemini. Post-Gemini dedup by candidate_id
# shrinks the list (a strong candidate's variants cluster together), so the
# pool must be larger than the 5 unique candidates we want to return.
_GEMINI_POOL_SIZE = 8

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

            logger.info("[STEP 1] Query text ready: %d chars", len(job_details))

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
            query_vector = await self._embedding_port.embed_query(job_details)
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
            keyword_query = self._extract_keywords(job_details)
            logger.debug("[STEP 3] Keyword query terms: %s", keyword_query)

            dense_results, keyword_results = await asyncio.gather(
                self._vector_store_port.search_profile_variants_dense(
                    query_vector, top_k=10, user_id=dto.user_id or None,
                ),
                self._vector_store_port.search_profile_variants_keyword(
                    keyword_query, top_k=10, user_id=dto.user_id or None,
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

            if not candidates:
                logger.info("No candidates found. Returning empty matches.")
                return ProfileMatchResponseDTO.success([])

            # ── Step 5: BM25 Rescore + RRF ───────────────────────────────
            logger.info("[STEP 5] BM25 Rescoring")
            rrf_start = time.perf_counter()
            bm25_scores = BM25Rescorer.rescore(
                candidates,
                job_details,
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

            # Diversify the Gemini pool BEFORE synthesis: without a per-candidate
            # cap, the top-8 RRF variants can be 8 variants of 3 people, and the
            # user sees fewer than 5 unique candidates. Max 2 variants per
            # candidate also stops Gemini from spending tokens scoring
            # duplicate people.
            _PER_CANDIDATE_POOL_CAP = 2
            per_candidate_count: dict[str, int] = {}
            top_variant_ids = []
            overflow_ids = []
            for vid, _score in rrf_results:
                payload = candidates_map.get(vid, {}).get("payload") or {}
                cid = payload.get("candidate_id", "")
                if cid and per_candidate_count.get(cid, 0) >= _PER_CANDIDATE_POOL_CAP:
                    overflow_ids.append(vid)
                    continue
                if cid:
                    per_candidate_count[cid] = per_candidate_count.get(cid, 0) + 1
                top_variant_ids.append(vid)
                if len(top_variant_ids) >= _GEMINI_POOL_SIZE:
                    break

            # Pool-diversity top-up: with the per-candidate cap, duplicate-
            # heavy results can fill fewer than 8 slots (e.g. 4 candidates ×
            # 2 variants). Top up from the overflow queue — extra variants of
            # already-pooled candidates are still worth scoring if slots are
            # otherwise wasted.
            if len(top_variant_ids) < _GEMINI_POOL_SIZE and overflow_ids:
                for vid in overflow_ids:
                    if len(top_variant_ids) >= _GEMINI_POOL_SIZE:
                        break
                    top_variant_ids.append(vid)
                logger.info(
                    "[STEP 6] Pool topped up from overflow to %d variant(s) "
                    "(diversity cap left slots unused).",
                    len(top_variant_ids),
                )

            rrf_time = time.perf_counter() - rrf_start

            logger.debug(f"RRF Output - Full Results (ID, Score): {rrf_results}")

            logger.info(
                "[STEP 6] RRF completed in %.2fs  |  top %d variant_ids=%s",
                rrf_time,
                len(top_variant_ids),
                top_variant_ids,
            )

            # ── Step 7: Collect full payloads for the Gemini pool ────────
            logger.info("[STEP 7] Collecting full payloads for Gemini")
            gemini_payloads = []
            for vid in top_variant_ids:
                if vid not in candidates_map:
                    continue
                payload = dict(candidates_map[vid]["payload"])  # shallow copy
                gemini_payloads.append(payload)
                logger.debug(
                    "Variant Payload [ID %s]: %s",
                    payload.get("variant_id"),
                    payload.get("variant_title"),
                )

            # ── Step 8: Gemini synthesis ─────────────────────────────────
            logger.info(
                "[STEP 8] Sending %d variant payloads to Gemini for scoring",
                len(gemini_payloads),
            )
            gemini_start = time.perf_counter()
            match_results = await self._synthesizer_port.synthesize_profile_matches(
                job_details=job_details,
                variant_payloads=gemini_payloads,
            )
            gemini_time = time.perf_counter() - gemini_start
            logger.info(
                "[STEP 8] Gemini synthesis completed in %.2fs  |  results=%d",
                gemini_time,
                len(match_results),
            )



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
                cid = result.candidate_id
                if cid not in seen_candidates:
                    seen_candidates[cid] = len(deduped)
                    deduped.append(result)
                else:
                    # Already seen — keep the one with higher match_percentage
                    existing_idx = seen_candidates[cid]
                    if result.match_percentage > deduped[existing_idx].match_percentage:
                        deduped[existing_idx] = result

            logger.info(
                "[STEP 9] %d unique candidates after deduplication (from %d variants)",
                len(deduped),
                len(match_results),
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
                return (-r.match_percentage, priority)

            deduped.sort(key=_tiebreak_key)

            # ── Step 9.6: Minimum-quality threshold ────────────────────
            # Weak matches below this percentage are not meaningful
            # recommendations — mirrors project_match_min_score. Set to 0
            # in settings to disable the filter entirely.
            min_pct = settings.profile_match_min_percentage
            pre_filter_pool = list(deduped)  # closest-match hint needs the unfiltered pool
            if min_pct > 0:
                before = len(deduped)
                deduped = [r for r in deduped if r.match_percentage >= min_pct]
                if len(deduped) < before:
                    logger.info(
                        "[STEP 9.6] %d candidate(s) below min threshold (%d%%) filtered; "
                        "%d remain",
                        before - len(deduped), min_pct, len(deduped),
                    )

            # ── Step 10: Return Top 5 unique candidates ────────────────
            top_matches = deduped[:5]

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
                # Empty result with an explanation — a blank list tells the
                # BD user nothing. Name the closest candidate so they know the
                # KB WAS searched and WHY nothing qualified (e.g. "JD needs
                # .NET; KB has no .NET profiles").
                if pre_filter_pool:
                    best = pre_filter_pool[0]
                    if min_pct > 0:
                        hint = (
                            f"No candidates met the minimum match threshold ({min_pct}%). "
                            f"Closest match: {best.candidate_name} at "
                            f"{best.match_percentage}% (variant '{best.variant_title}'). "
                            "The knowledge base may not contain candidates with this "
                            "JD's core stack — consider ingesting more profiles."
                        )
                    else:
                        hint = (
                            "Gemini returned no candidate results for this JD. "
                            "The knowledge base may not contain relevant profiles."
                        )
                    logger.info(
                        "[STEP 10] Empty result — closest candidate was %s at %d%%",
                        best.candidate_name, best.match_percentage,
                    )
                else:
                    hint = (
                        "No candidate variants were found in the knowledge base "
                        "for this job description. Ingest candidate profiles first."
                    )
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

            # Step 2: Gemini synthesis (reuses same synthesizer, single-item list)
            logger.info("[STEP 2] Sending variant to Gemini for scoring")
            gemini_start = time.perf_counter()
            match_results = await self._synthesizer_port.synthesize_profile_matches(
                job_details=job_details,
                variant_payloads=[payload],
            )
            logger.info(
                "[STEP 2] Gemini completed in %.2fs", time.perf_counter() - gemini_start
            )

            if not match_results:
                return ProfileMatchResponseDTO.failed(
                    reason="Gemini returned no result for the selected variant."
                )

            # Step 3: Enrich result with metadata from payload
            result = match_results[0]
            result.email = payload.get("email", "")
            result.role = payload.get("role", "")

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


import time
import asyncio
from src.domain.interfaces.rag.i_embedding_port import IEmbeddingPort
from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.domain.interfaces.rag.i_synthesizer_port import ISynthesizerPort
from src.domain.exceptions.rag_exceptions import RAGBaseException
from src.infrastructure.db.qdrant.bm25_rescorer import BM25Rescorer, rrf_merge
from src.application.dto.project_dto import ProjectMatchRequestDTO, ProjectMatchResponseDTO
from src.common.jd_text import extract_jd_keywords
from src.common.config import settings
from src.common.logger import get_logger

logger = get_logger(__name__)

# Fair evidence allocation for Stage 3: no single project may hog the chunk
# evidence (which would starve other Stage-1 winners and bias Gemini's scores
# toward the most verbose project, not the most relevant one).
_MAX_EVIDENCE_PER_PROJECT = 3
_MAX_EVIDENCE_TOTAL = 12


class MatchProjectsUseCase:
    """
    Application use case that orchestrates the full project matching pipeline:
    Stage 1: Hybrid summary search (Dense + Keyword → BM25 rescore → RRF) → Top 5
    Stage 2: Dense chunk search filtered to Top 5 (pool of 15)
    Stage 3: Balanced evidence selection (max 3/project, 12 total) →
             Gemini synthesis (LLM-as-Reranker) → threshold filter → Top 3
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

    async def execute(self, dto: ProjectMatchRequestDTO) -> ProjectMatchResponseDTO:
        """
        Execute the full retrieval + synthesis pipeline.
        """
        total_start = time.perf_counter()
        logger.info("================ MATCHING PIPELINE STARTED ================")

        try:
            job_details = dto.job_details

            # Empty JD would fail late inside the embedding call with a
            # cryptic "empty Part" 400 — reject it here with a clear message.
            if not job_details or not job_details.strip():
                return ProjectMatchResponseDTO.failed(
                    reason="job_details must not be empty."
                )

            # ── Step 1: Query text ────────────────
            logger.info("[STEP 1] Starting query text setup")
            query_text = job_details
            logger.info(f"[STEP 1] Query text ready: {len(query_text)} chars")

            # ── Step 2: Embed query ──────────────────────────────────────────
            logger.info("[STEP 2] Starting query embedding via Gemini")
            embed_start = time.perf_counter()
            query_vector = await self._embedding_port.embed_query(query_text)
            embed_time = time.perf_counter() - embed_start
            logger.info(f"[STEP 2] Query embedding completed in {embed_time:.2f}s")

            # ── Step 3: Stage 1 — Hybrid search on summaries (parallel) ──────
            logger.info("[STEP 3] Starting Stage 1 hybrid retrieval (Dense + Keyword)")
            stage1_start = time.perf_counter()

            keyword_query = self._extract_keywords(query_text)
            logger.debug("[STEP 3] Keyword query terms: %s", keyword_query)

            dense_results, keyword_results = await asyncio.gather(
                self._vector_store_port.search_summaries_dense(
                    query_vector, top_k=10, user_id=dto.user_id or None,
                ),
                self._vector_store_port.search_summaries_keyword(
                    keyword_query, top_k=10, user_id=dto.user_id or None,
                ),
            )

            retrieval_time = time.perf_counter() - stage1_start
            logger.info(f"[STEP 3] Stage 1 retrieval completed in {retrieval_time:.2f}s")
            
            # Log retrieved items
            dense_pids = [r["payload"]["project_id"] for r in dense_results
                          if isinstance(r.get("payload"), dict) and r["payload"].get("project_id")]
            keyword_pids = [r["payload"]["project_id"] for r in keyword_results
                            if isinstance(r.get("payload"), dict) and r["payload"].get("project_id")]
            logger.debug(f"Retrieved Dense Project IDs: {dense_pids}")
            logger.debug(f"Retrieved Keyword Project IDs: {keyword_pids}")
            logger.info(f"[STEP 3] Retrieved {len(dense_results)} dense results and {len(keyword_results)} keyword results")

            logger.info("[STEP 4] Starting candidate deduplication and merging")
            candidates_map: dict[str, dict] = {}
            for result in dense_results:
                pid = (result.get("payload") or {}).get("project_id")
                if pid:
                    candidates_map[pid] = result
            for result in keyword_results:
                pid = (result.get("payload") or {}).get("project_id")
                if pid and pid not in candidates_map:
                    candidates_map[pid] = result
            
            candidates = list(candidates_map.values())
            logger.info(f"[STEP 4] Candidate merging completed. {len(candidates)} unique candidates found")

            if not candidates:
                logger.info("No candidates found in Stage 1. Returning empty matches.")
                return ProjectMatchResponseDTO.success([])

            # BM25 rescore
            logger.info("[STEP 5] Starting BM25 Rescoring phase")
            rrf_start = time.perf_counter()
            bm25_scores = BM25Rescorer.rescore(candidates, query_text)
            logger.info(f"[STEP 5] BM25 Rescoring completed for {len(bm25_scores)} candidates")

            logger.info("[STEP 6] Preparing rankings for RRF")
            dense_ranking = [
                (r["payload"]["project_id"], r["score"])
                for r in sorted(dense_results, key=lambda x: x["score"], reverse=True)
                if isinstance(r.get("payload"), dict) and r["payload"].get("project_id")
            ]
            bm25_ranking = [
                (pid, score)
                for pid, score in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            logger.debug(f"RRF Input - Dense Ranking: {dense_ranking}")
            logger.debug(f"RRF Input - BM25 Ranking: {bm25_ranking}")
            
            logger.info("[STEP 7] Starting RRF (Reciprocal Rank Fusion) merge")
            rrf_results = rrf_merge(dense_ranking, bm25_ranking, k=60)
            top5_project_ids = [pid for pid, _ in rrf_results[:5]]

            rrf_time = time.perf_counter() - rrf_start
            logger.info(f"[STEP 7] RRF merge completed in {rrf_time:.2f}s")
            logger.debug(f"RRF Output - Full Results (ID, Score): {rrf_results}")
            logger.info(f"[STEP 7] Top 5 projects selected for deep chunk retrieval: {top5_project_ids}")

            if not top5_project_ids:
                return ProjectMatchResponseDTO.success([])

            # ── Step 4: Stage 2 — Dense search on chunks ─────────────────────
            logger.info(f"[STEP 8] Starting Stage 2 dense chunk retrieval for Top 5 projects")
            stage2_start = time.perf_counter()
            # Per-project queries (top_k=3 each) instead of one global top-15:
            # a single dominant project's chunks would otherwise consume the
            # global limit and starve the other 4 candidates of any evidence.
            per_project = await asyncio.gather(*(
                self._vector_store_port.search_chunks_dense(
                    query_vector, project_ids=[pid], top_k=3,
                    user_id=dto.user_id or None,
                )
                for pid in top5_project_ids
            ))
            chunk_results: list[dict] = []
            for chunks in per_project:
                chunk_results.extend(chunks)
            stage2_time = time.perf_counter() - stage2_start
            logger.info(f"[STEP 8] Stage 2 chunk retrieval completed in {stage2_time:.2f}s")
            logger.info(f"[STEP 8] Retrieved a total of {len(chunk_results)} evidence chunks")

            if not chunk_results:
                logger.info("No chunks found in Stage 2. Returning empty matches.")
                return ProjectMatchResponseDTO.success([])

            logger.info("[STEP 9] Formatting chunk evidence for LLM (balanced per project)")
            chunk_evidence = self._balance_evidence(chunk_results)
            for i, chunk in enumerate(chunk_evidence):
                preview = str(chunk.get("text", ""))[:100].replace("\n", " ")
                logger.debug(f"Evidence Chunk {i+1} [Proj {chunk.get('project_id')}]: {preview}...")

            # ── Step 5: Stage 3 — Gemini synthesis ───────────────────────────
            logger.info("[STEP 10] Starting Stage 3 Gemini LLM Synthesis and Reranking")
            gemini_start = time.perf_counter()
            match_results = await self._synthesizer_port.synthesize(
                job_details=job_details,
                chunk_evidence=chunk_evidence,
            )
            gemini_time = time.perf_counter() - gemini_start
            logger.info(f"[STEP 10] Gemini synthesis completed in {gemini_time:.2f}s")
            logger.info(f"[STEP 10] LLM generated {len(match_results)} match justifications")

            # ── Step 11: Filter weak matches + return Top 3 ─────────────────
            # Any project scoring below the threshold is not a meaningful
            # match for this JD. Gemini itself labels these as "no overlap" —
            # this threshold prevents forcing irrelevant results to the frontend.
            min_score = settings.project_match_min_score
            meaningful = [r for r in match_results if r.match_score >= min_score]

            if not meaningful:
                logger.info(
                    "[STEP 11] All %d results scored below threshold (%.2f). "
                    "Returning empty matches.",
                    len(match_results),
                    min_score,
                )
                return ProjectMatchResponseDTO.success([])

            # Sort by score descending — Gemini's output order is not
            # guaranteed to be ranked, and the top-3 slice below must keep
            # the genuinely best-scoring projects.
            meaningful.sort(key=lambda r: r.match_score, reverse=True)
            top3 = meaningful[:3]
            logger.info(
                "[STEP 11] %d/%d results passed threshold  |  returning top %d",
                len(meaningful),
                len(match_results),
                len(top3),
            )

            total_time = time.perf_counter() - total_start
            logger.info(
                f"================ PIPELINE COMPLETE in {total_time:.2f}s ================\n"
                f"Stats: Embed={embed_time:.2f}s | Stage1={retrieval_time:.2f}s | "
                f"RRF={rrf_time:.2f}s | Stage2={stage2_time:.2f}s | Gemini={gemini_time:.2f}s"
            )

            return ProjectMatchResponseDTO.success(top3)

        except RAGBaseException as e:
            logger.error(f"RAG pipeline failed: {e}")
            return ProjectMatchResponseDTO.failed(reason=str(e))
        except Exception as e:
            logger.error(f"Unexpected error in match pipeline: {e}", exc_info=True)
            return ProjectMatchResponseDTO.failed(reason="Internal error: please check server logs.")

    @staticmethod
    def _balance_evidence(chunk_results: list[dict]) -> list[dict]:
        """
        Select evidence chunks fairly across projects, round-robin by score.

        Dense chunk search returns the global top-k, so one project with many
        similar chunks can crowd out the other Stage-1 winners entirely. This
        gives every retrieved project a share of the evidence budget (up to
        _MAX_EVIDENCE_PER_PROJECT each, _MAX_EVIDENCE_TOTAL overall) while
        still preferring higher-scoring chunks.
        """
        by_project: dict[str, list[dict]] = {}
        for r in sorted(chunk_results, key=lambda x: x["score"], reverse=True):
            pid = (r.get("payload") or {}).get("project_id")
            if not pid:
                continue  # malformed/legacy point — skip rather than crash
            by_project.setdefault(pid, []).append(r["payload"])

        selected: list[dict] = []
        for round_idx in range(_MAX_EVIDENCE_PER_PROJECT):
            for pid in by_project:
                chunks = by_project[pid]
                if round_idx < len(chunks):
                    selected.append(chunks[round_idx])
                    if len(selected) >= _MAX_EVIDENCE_TOTAL:
                        return selected
        return selected

    @staticmethod
    def _extract_keywords(jd_text: str) -> str:
        return extract_jd_keywords(jd_text, max_keywords=40)

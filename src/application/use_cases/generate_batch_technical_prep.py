"""Use case: Generate Technical Preparation guides for multiple candidates."""

import time

from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.domain.interfaces.rag.i_synthesizer_port import ISynthesizerPort
from src.domain.value_objects.technical_prep_result import (
    CandidatePrepResult,
    BatchTechnicalPrepResult,
)
from src.application.dto.preparation_dto import (
    BatchTechnicalPrepRequestDTO,
    BatchTechnicalPrepResponseDTO,
    TechnicalPrepRequestDTO,
)
from src.application.use_cases.generate_technical_prep import GenerateTechnicalPrepUseCase
from src.common.logger import get_logger

logger = get_logger(__name__)


class GenerateBatchTechnicalPrepUseCase:
    """
    Batch wrapper around GenerateTechnicalPrepUseCase.

    Delegates each candidate to the single-prep use case — no duplication
    of Qdrant fetch, context-building, or Gemini call logic. Single-prep
    behaviour is guaranteed identical by construction.

    Steps:
    1. Validate job_details + candidates list
    2. For each candidate, call single-prep use case (sequential — avoids
       Gemini 429 rate-limit storms)
    3. Map each TechnicalPrepResponseDTO → CandidatePrepResult
    4. If ALL candidates failed → top-level FAILED
       Otherwise → top-level SUCCESS with per-candidate statuses
    """

    def __init__(
        self,
        vector_store_port: IVectorStorePort,
        synthesizer_port: ISynthesizerPort,
    ) -> None:
        self._single_prep = GenerateTechnicalPrepUseCase(
            vector_store_port=vector_store_port,
            synthesizer_port=synthesizer_port,
        )

    async def execute(
        self, dto: BatchTechnicalPrepRequestDTO
    ) -> BatchTechnicalPrepResponseDTO:
        total_start = time.perf_counter()
        logger.info(
            "======== BATCH TECHNICAL PREP PIPELINE STARTED | candidates=%d ========",
            len(dto.candidates),
        )

        # ── Step 1: Validate ────────────────────────────────────────────
        if not dto.job_details or not dto.job_details.strip():
            return BatchTechnicalPrepResponseDTO.failed("job_details must not be empty.")
        if not dto.candidates:
            return BatchTechnicalPrepResponseDTO.failed("candidates list must not be empty.")

        # ── Step 2: Sequential loop — delegate to single use case ───────
        results: list[CandidatePrepResult] = []

        for idx, candidate in enumerate(dto.candidates):
            logger.info(
                "[CANDIDATE %d/%d] variant_id=%s",
                idx + 1, len(dto.candidates), candidate.variant_id,
            )
            candidate_start = time.perf_counter()

            try:
                single_dto = TechnicalPrepRequestDTO(
                    job_details=dto.job_details,
                    variant_id=candidate.variant_id,
                    matching_skills=candidate.matching_skills,
                    missing_skills=candidate.missing_skills,
                )
                r = await self._single_prep.execute(single_dto)

                if r.status == "SUCCESS":
                    results.append(CandidatePrepResult(
                        variant_id=candidate.variant_id,
                        status="SUCCESS",
                        candidate_name=r.candidate_name,
                        variant_title=r.variant_title,
                        technical_briefing_note=(
                            r.result.technical_briefing_note if r.result else None
                        ),
                        interview_preparation_guide=(
                            r.result.interview_preparation_guide if r.result else []
                        ),
                    ))
                else:
                    results.append(CandidatePrepResult(
                        variant_id=candidate.variant_id,
                        status="FAILED",
                        error_message=r.error_message,
                    ))

            except Exception as e:
                logger.error(
                    "[CANDIDATE %d/%d] Unexpected error: %s",
                    idx + 1, len(dto.candidates), e,
                    exc_info=True,
                )
                # No internal details to the client — full error is logged.
                results.append(CandidatePrepResult(
                    variant_id=candidate.variant_id,
                    status="FAILED",
                    error_message="Internal error: please check server logs.",
                ))

            logger.info(
                "[CANDIDATE %d/%d] completed in %.2fs | status=%s",
                idx + 1, len(dto.candidates),
                time.perf_counter() - candidate_start,
                results[-1].status,
            )

        # ── Step 3: Post-loop — ALL candidates failed → top-level FAILED ─
        if results and all(r.status == "FAILED" for r in results):
            logger.error("All %d candidates failed.", len(results))
            return BatchTechnicalPrepResponseDTO.failed(
                f"All {len(results)} candidates failed. "
                f"First error: {results[0].error_message}"
            )

        batch_result = BatchTechnicalPrepResult(results=results)
        logger.info(
            "======== BATCH TECHNICAL PREP COMPLETE in %.2fs | "
            "succeeded=%d failed=%d ========",
            time.perf_counter() - total_start,
            batch_result.succeeded,
            batch_result.failed,
        )
        return BatchTechnicalPrepResponseDTO.success(batch_result)

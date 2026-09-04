"""Use case: Generate Technical Preparation guide for a selected candidate."""

import time
from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.domain.interfaces.rag.i_synthesizer_port import ISynthesizerPort
from src.domain.exceptions.rag_exceptions import RAGBaseException
from src.application.dto.preparation_dto import (
    TechnicalPrepRequestDTO,
    TechnicalPrepResponseDTO,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


class GenerateTechnicalPrepUseCase:
    """
    Application use case that orchestrates the Technical Preparation pipeline.

    Steps:
    1. Validate the incoming DTO (job_details, variant_id, skills)
    2. Fetch the candidate's full profile from Qdrant using variant_id
    3. Merge the fetched profile with the skill gap data from Step 3
    4. Send the combined context + JD to Gemini to generate:
       - technical_briefing_note (interview focus summary)
       - interview_preparation_guide (topics: 80% gaps, 20% strengths)
    5. Return a structured response DTO
    """

    def __init__(
        self,
        vector_store_port: IVectorStorePort,
        synthesizer_port: ISynthesizerPort,
    ) -> None:
        self._vector_store_port = vector_store_port
        self._synthesizer_port = synthesizer_port

    async def execute(
        self, dto: TechnicalPrepRequestDTO,
    ) -> TechnicalPrepResponseDTO:
        total_start = time.perf_counter()
        logger.info("======== TECHNICAL PREP PIPELINE STARTED ========")

        try:
            # ── Step 1: Validate inputs ───────────────────────────────────
            if not dto.job_details or not dto.job_details.strip():
                return TechnicalPrepResponseDTO.failed(
                    reason="job_details must not be empty."
                )
            if not dto.variant_id or not dto.variant_id.strip():
                return TechnicalPrepResponseDTO.failed(
                    reason="variant_id must not be empty."
                )

            logger.info(
                "[STEP 1] Input validated  |  variant_id=%s  "
                "matching=%d  missing=%d  jd_len=%d",
                dto.variant_id,
                len(dto.matching_skills or []),
                len(dto.missing_skills or []),
                len(dto.job_details),
            )

            # ── Step 2: Fetch candidate profile from Qdrant ───────────────
            logger.info(
                "[STEP 2] Fetching candidate profile from Qdrant  |  "
                "variant_id=%s",
                dto.variant_id,
            )
            fetch_start = time.perf_counter()
            profile_payload = await self._vector_store_port.fetch_profile_variant_by_id(
                dto.variant_id
            )
            fetch_time = time.perf_counter() - fetch_start

            if profile_payload is None:
                logger.warning(
                    "[STEP 2] variant_id=%s not found in Qdrant", dto.variant_id
                )
                return TechnicalPrepResponseDTO.failed(
                    reason=f"Candidate variant '{dto.variant_id}' not found."
                )

            candidate_name = profile_payload.get("candidate_name", "the candidate")
            variant_title = profile_payload.get("variant_title", "")
            logger.info(
                "[STEP 2] Profile fetched in %.3fs  |  candidate=%s  title=%s",
                fetch_time,
                candidate_name,
                variant_title,
            )

            # ── Step 3: Merge profile + skill gaps into candidate_context ──
            logger.info("[STEP 3] Building combined candidate context")
            candidate_context = {
                # Identity
                "candidate_name": candidate_name,
                "variant_title": variant_title,
                "experience_years": profile_payload.get("experience_years", 0),
                # Tech background
                "tech_stacks": profile_payload.get("tech_stacks", []),
                "certifications": profile_payload.get("certifications", []),
                "projects": profile_payload.get("projects", []),
                # Skill gap analysis from Step 3 (Gemini profile match result)
                "matching_skills": dto.matching_skills,
                "missing_skills": dto.missing_skills,
            }
            logger.info(
                "[STEP 3] Context built  |  projects=%d  "
                "matching_skills=%d  missing_skills=%d",
                len(candidate_context["projects"]),
                len(candidate_context["matching_skills"]),
                len(candidate_context["missing_skills"]),
            )

            # ── Step 4: Call Gemini to generate the preparation guide ──────
            logger.info(
                "[STEP 4] Calling Gemini  |  candidate=%s", candidate_name
            )
            gemini_start = time.perf_counter()
            result = await self._synthesizer_port.generate_technical_prep(
                job_details=dto.job_details,
                candidate_context=candidate_context,
            )
            gemini_time = time.perf_counter() - gemini_start
            logger.info(
                "[STEP 4] Gemini completed in %.3fs  |  topics=%d",
                gemini_time,
                len(result.interview_preparation_guide),
            )

            total_time = time.perf_counter() - total_start
            logger.info(
                "======== TECHNICAL PREP PIPELINE COMPLETE in %.2fs ========\n"
                "Stats: Qdrant=%.3fs | Gemini=%.3fs",
                total_time,
                fetch_time,
                gemini_time,
            )

            return TechnicalPrepResponseDTO.success(
                candidate_name=candidate_name,
                variant_title=variant_title,
                result=result,
            )

        except RAGBaseException as e:
            logger.error("Technical prep pipeline failed: %s", e)
            return TechnicalPrepResponseDTO.failed(reason=str(e))
        except Exception as e:
            logger.error(
                "Unexpected error in technical prep pipeline: %s", e, exc_info=True
            )
            return TechnicalPrepResponseDTO.failed(
                reason=f"Internal error: {str(e)}"
            )

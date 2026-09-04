import time
from src.domain.interfaces.rag.i_synthesizer_port import ISynthesizerPort
from src.domain.exceptions.rag_exceptions import RAGBaseException
from src.application.dto.sales_enablement_dto import (
    SalesEnablementRequestDTO,
    SalesEnablementResponseDTO,
)
from src.common.logger import get_logger

logger = get_logger(__name__)


class GenerateSalesEnablementUseCase:
    """
    Application use case that orchestrates the Sales Enablement generation pipeline.

    Accepts a job description and a list of matched project contexts, then
    calls the Gemini LLM to generate:
    - Discovery Questions for the BD team to qualify the lead
    - Talking Points for the BD pitch (without naming internal projects)
    - A formal Outreach Email template written from a BD perspective
    """

    def __init__(self, synthesizer_port: ISynthesizerPort) -> None:
        self._synthesizer_port = synthesizer_port

    async def execute(
        self,
        dto: SalesEnablementRequestDTO,
    ) -> SalesEnablementResponseDTO:
        """
        Execute the sales enablement generation pipeline.

        Args:
            dto: Contains the job_details string and list of project context dicts.

        Returns:
            SalesEnablementResponseDTO with discovery_questions, talking_points,
            and outreach_template on success, or error_message on failure.
        """
        total_start = time.perf_counter()
        logger.info("======== SALES ENABLEMENT PIPELINE STARTED ========")
        logger.info(
            "[STEP 1] Input received | jd_len=%d chars | projects=%d",
            len(dto.job_details),
            len(dto.projects),
        )

        try:
            if not dto.job_details or not dto.job_details.strip():
                return SalesEnablementResponseDTO.failed(
                    reason="job_details must not be empty."
                )

            if not dto.projects:
                return SalesEnablementResponseDTO.failed(
                    reason="At least one project must be provided."
                )

            logger.info("[STEP 2] Calling Gemini to generate sales enablement content")
            gemini_start = time.perf_counter()
            result = await self._synthesizer_port.generate_sales_enablement(
                job_details=dto.job_details,
                projects=dto.projects,
            )
            gemini_time = time.perf_counter() - gemini_start
            logger.info(
                "[STEP 2] Gemini generation completed in %.2fs | "
                "questions=%d | talking_points=%d",
                gemini_time,
                len(result.discovery_questions),
                len(result.talking_points),
            )

            total_time = time.perf_counter() - total_start
            logger.info(
                "======== SALES ENABLEMENT PIPELINE COMPLETE in %.2fs ========",
                total_time,
            )

            return SalesEnablementResponseDTO.success(result)

        except RAGBaseException as e:
            logger.error("Sales enablement pipeline failed: %s", e)
            return SalesEnablementResponseDTO.failed(reason=str(e))
        except Exception as e:
            logger.error("Unexpected error in sales enablement pipeline: %s", e, exc_info=True)
            return SalesEnablementResponseDTO.failed(reason="Internal error: please check server logs.")

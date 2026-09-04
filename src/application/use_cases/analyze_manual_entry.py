import time
from src.domain.interfaces.extractor.i_extractor import IExtractor
from src.application.dto.analyze_manual_entry_dto import AnalyzeManualEntryDTO
from src.application.dto.scrape_request_dto import JobResultDTO
from src.infrastructure.ai.gemini_company_profiler import GeminiCompanyProfiler
from src.common.logger import get_logger

logger = get_logger(__name__)


class AnalyzeManualEntry:
    """
    Use case to extract and structure job details from manually entered text.
    Uses the same extractor and profiler as the scrape pipeline to guarantee
    identical output structures.
    """

    def __init__(self, extractor: IExtractor, profiler: GeminiCompanyProfiler | None = None) -> None:
        self._extractor = extractor
        self._profiler = profiler or GeminiCompanyProfiler()

    async def execute(self, dto: AnalyzeManualEntryDTO) -> JobResultDTO:
        logger.info(f"======== MANUAL JOB ENTRY PIPELINE STARTED ========")
        start_time = time.perf_counter()

        try:
            # 1. Format the manual fields into a single text block for the LLM
            # We explicitly label the fields to give the LLM clear context
            parts = []
            if dto.company_name:
                parts.append(f"Company Name: {dto.company_name}")
            if dto.company_website:
                parts.append(f"Company Website: {dto.company_website}")
            if dto.job_title:
                parts.append(f"Job Title: {dto.job_title}")
            if dto.experience:
                parts.append(f"Experience Required: {dto.experience}")
            if dto.job_description:
                parts.append(f"Job Description:\n{dto.job_description}")
            if dto.additional_notes:
                parts.append(f"Additional Context/Notes:\n{dto.additional_notes}")
            
            cleaned_text = "\n\n".join(parts)

            # 2. Extract job details using Gemini
            logger.info("Extracting structured job details using Gemini...")
            job_details = await self._extractor.extract(cleaned_text)

            # 3. Generate company profile
            company_profile = None
            company_name_to_profile = job_details.company or dto.company_name
            if company_name_to_profile:
                logger.info(f"Generating company profile for '{company_name_to_profile}'...")
                try:
                    company_profile = await self._profiler.profile(company_name_to_profile)
                except Exception as e:
                    logger.warning(f"Company profiling failed for '{company_name_to_profile}': {e}")
                    # We continue even if profiling fails
            
            total_time = time.perf_counter() - start_time
            logger.info(f"======== MANUAL ENTRY PIPELINE COMPLETE in {total_time:.2f}s ========")
            
            return JobResultDTO.success(
                job_details=job_details,
                company_profile=company_profile
            )

        except Exception as e:
            logger.error(f"Manual entry analysis failed: {e}", exc_info=True)
            return JobResultDTO.failed(reason=str(e))

from fastapi import APIRouter, Depends, Request

from src.application.use_cases.scrape_job_url import ScrapeJobURL
from src.application.dto.scrape_request_dto import ScrapeRequestDTO
from src.domain.enums.scrape_status import ScrapeStatus
from src.domain.interfaces.extractor.i_extractor import IExtractor
from src.infrastructure.mongodb.session_repository import MongoDBSessionRepository
from src.presentation.schemas.scrape_schemas import (
    ScrapeRequest,
    ScrapeResponse,
    ManualEntryRequest,
)
from src.application.use_cases.analyze_manual_entry import AnalyzeManualEntry
from src.application.dto.analyze_manual_entry_dto import AnalyzeManualEntryDTO
from src.common.logger import (
    get_logger,
    set_log_context,
    reset_log_context,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Scraper"])


def get_session_store() -> MongoDBSessionRepository:
    return MongoDBSessionRepository()


def get_extractor(request: Request) -> IExtractor:
    return request.app.state.extractor


def get_company_profiler(request: Request):
    return getattr(request.app.state, "company_profiler", None)


def get_scrape_use_case(
    session_store: MongoDBSessionRepository = Depends(get_session_store),
    extractor: IExtractor = Depends(get_extractor),
    company_profiler=Depends(get_company_profiler),
) -> ScrapeJobURL:
    return ScrapeJobURL(
        extractor=extractor,
        session_store=session_store,
        company_profiler=company_profiler,
    )


def get_analyze_manual_entry_use_case(
    extractor: IExtractor = Depends(get_extractor),
    company_profiler=Depends(get_company_profiler),
) -> AnalyzeManualEntry:
    return AnalyzeManualEntry(
        extractor=extractor,
        profiler=company_profiler,
    )


# =============================================================================
# SCRAPE ENDPOINT
# =============================================================================

@router.post(
    "/scrape",
    response_model=ScrapeResponse,
    response_model_exclude_none=True,
    summary="Scrape a job posting URL",
)
async def scrape_job(
    body: ScrapeRequest,
    use_case: ScrapeJobURL = Depends(get_scrape_use_case),
) -> ScrapeResponse:
    """
    Accepts a job posting URL, runs the full scraping pipeline,
    and returns structured job data.

    User ID and action are placed into request-scoped logging context so
    every downstream log generated during this request automatically carries
    the same user_id and action.
    """

    # -------------------------------------------------------------------------
    # The endpoint itself defines the authoritative audit action.
    # We do not trust body.action for the audit context.
    # -------------------------------------------------------------------------
    user_id = body.user_id

    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Scrape Job Posting",
        section="Scraper",
    )

    try:
        logger.info(
            f"Job posting scrape requested for '{body.url}'"
        )


        request_dto = ScrapeRequestDTO(
            user_id=user_id,
            action=body.action,
            url=str(body.url),
        )

        result = await use_case.execute(request_dto)

        from src.domain.value_objects.domain_url import DomainURL

        try:
            raw_domain = DomainURL(str(body.url)).domain
            parts = raw_domain.split(".")
            # Generic second-level labels — the brand label sits one position
            # further left for these (covers co.uk/com.au/ac.uk/edu.au/gov.in).
            _generic_second_level = {"co", "com", "org", "net", "ac", "edu", "gov"}

            if len(parts) >= 3 and parts[-2] in _generic_second_level:
                platform = parts[-3]
            elif len(parts) >= 2:
                platform = parts[-2]
            else:
                platform = raw_domain

        except ValueError:
            platform = None

        if result.status == ScrapeStatus.SUCCESS:
            from src.domain.value_objects.company_profile import (
                CompanyProfile as _CompanyProfile,
            )

            company_profile_data = (
                result.company_profile.model_dump()
                if result.company_profile
                else _CompanyProfile().model_dump()
            )

            logger.info(
                "Scrape request completed successfully"
            )

            return ScrapeResponse(
                status=result.status.value,
                platform=platform,
                job_details=(
                    result.job_details.model_dump()
                    if result.job_details
                    else None
                ),
                company_profile=company_profile_data,
            )

        if result.status == ScrapeStatus.AUTH_REQUIRED:
            logger.warning(
                "Scrape request requires authentication"
            )

            return ScrapeResponse(
                status=result.status.value,
                platform=platform,
                auth_required=True,
                auth_domain=result.auth_required_domain,
                auth_login_url=result.auth_required_login_url,
                message=result.message,
            )

        logger.warning(
            "Scrape request failed"
        )

        return ScrapeResponse(
            status=result.status.value,
            platform=platform,
            error_message=result.error_message,
        )

    finally:
        # Restore the previous context after this request finishes.
        #
        # This prevents one user's context from leaking into another request.
        reset_log_context(user_token, action_token, section_token)


# =============================================================================
# MANUAL ENTRY ENDPOINT
# =============================================================================

@router.post(
    "/manual-entry",
    response_model=ScrapeResponse,
    response_model_exclude_none=True,
    summary="Analyze a manually entered job posting",
)
async def analyze_manual_entry(
    body: ManualEntryRequest,
    use_case: AnalyzeManualEntry = Depends(get_analyze_manual_entry_use_case),
) -> ScrapeResponse:
    """
    Accepts manually entered job details, extracts missing information via
    Gemini, and returns structured job data identical to the scrape endpoint.

    User ID and action are placed into request-scoped logging context so
    every downstream log generated during this request automatically carries
    the same user_id and action.
    """

    user_id = body.user_id
    job = body.job

    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Analyze Manual Job Entry",
        section="Scraper",
    )

    try:
        logger.info(
            f"Manual job entry received — '{job.job_title}' at '{job.company_name}'"
        )

        request_dto = AnalyzeManualEntryDTO(
            user_id=user_id,
            action=body.action,
            company_name=job.company_name,
            company_website=job.company_website,
            job_title=job.job_title,
            experience=job.experience,
            job_description=job.job_description,
            additional_notes=job.additional_notes,
        )

        result = await use_case.execute(request_dto)

        if result.status == ScrapeStatus.SUCCESS:
            from src.domain.value_objects.company_profile import (
                CompanyProfile as _CompanyProfile,
            )

            company_profile_data = (
                result.company_profile.model_dump()
                if result.company_profile
                else _CompanyProfile().model_dump()
            )

            logger.info(
                "Audit event: manual entry request completed successfully"
            )

            return ScrapeResponse(
                status=result.status.value,
                platform="manual",
                job_details=(
                    result.job_details.model_dump()
                    if result.job_details
                    else None
                ),
                company_profile=company_profile_data,
            )

        logger.warning(
            "Audit event: manual entry request failed"
        )

        return ScrapeResponse(
            status=result.status.value,
            platform="manual",
            error_message=result.error_message,
        )

    finally:
        reset_log_context(user_token, action_token, section_token)
from fastapi import APIRouter, Depends, HTTPException, Request

from src.application.dto.profile_dto import (
    IngestProfileDTO,
    ProfileMatchRequestDTO,
    VariantDTO,
    ProjectInVariantDTO,
)
from src.application.use_cases.ingest_profile import IngestProfileUseCase
from src.application.use_cases.match_profiles import MatchProfilesUseCase
from src.application.use_cases.delete_profile import DeleteProfileUseCase, CandidateNotFoundException
from src.domain.exceptions.rag_exceptions import VectorStoreError
from src.infrastructure.ai.cached_embedding_adapter import CachedEmbeddingAdapter
from src.infrastructure.ai.gemini_synthesizer_adapter import GeminiSynthesizerAdapter
from src.infrastructure.db.qdrant.qdrant_vector_store_adapter import QdrantVectorStoreAdapter
from src.presentation.schemas.profile_schemas import (
    IngestProfileRequest,
    IngestProfileResponse,
    IngestVariantDetail,
    ProfileMatchRequest,
    ProfileMatchResponse,
    ProfileMatchResultSchema,
    DeleteProfileResponse,
)
from src.common.logger import (
    get_logger,
    set_log_context,
    reset_log_context,
)


logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/profiles", tags=["Profiles"])


# ─── Dependency Injection ─────────────────────────────────────────────────────

def get_vector_store(request: Request) -> QdrantVectorStoreAdapter:
    return request.app.state.vector_store


def get_embedding_port(request: Request) -> CachedEmbeddingAdapter:
    return request.app.state.embedding_port


def get_synthesizer(request: Request) -> GeminiSynthesizerAdapter:
    return request.app.state.synthesizer


def get_ingest_profile_use_case(
    embedding_port: CachedEmbeddingAdapter = Depends(get_embedding_port),
    vector_store: QdrantVectorStoreAdapter = Depends(get_vector_store),
) -> IngestProfileUseCase:
    return IngestProfileUseCase(
        embedding_port=embedding_port,
        vector_store_port=vector_store,
    )


def get_match_profiles_use_case(
    embedding_port: CachedEmbeddingAdapter = Depends(get_embedding_port),
    vector_store: QdrantVectorStoreAdapter = Depends(get_vector_store),
    synthesizer: GeminiSynthesizerAdapter = Depends(get_synthesizer),
) -> MatchProfilesUseCase:
    return MatchProfilesUseCase(
        embedding_port=embedding_port,
        vector_store_port=vector_store,
        synthesizer_port=synthesizer,
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestProfileResponse,
    response_model_exclude_none=True,
    summary="Ingest a candidate profile with variants into the RAG vector store",
)
async def ingest_profile(
    body: IngestProfileRequest,
    use_case: IngestProfileUseCase = Depends(get_ingest_profile_use_case),
) -> IngestProfileResponse:

    """
    Accepts a candidate profile payload containing one or more variants.
    Each variant is independently embedded and stored as a single point
    in the Qdrant `profile_variants` collection.

    Supports both bulk ingestion (all variants at once) and single
    variant updates (send just the updated variant).
    """

    user_id = body.user_id
    profile = body.profile

    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Add Candidate Profile",
        section="Profiles",
    )

    try:
        logger.info(
            "Adding candidate '%s' with %d profile variant(s) to the knowledge base",
            profile.candidate_name,
            len(profile.variants),
        )

        # Build DTOs
        variant_dtos = []

        for v in profile.variants:
            project_dtos = [
                ProjectInVariantDTO(
                    project_id=p.project_id,
                    project_name=p.project_name,
                    domain=p.domain,
                    tech_stack=p.tech_stack,
                    links=p.links,
                    description=p.description,
                )
                for p in v.projects
            ]

            variant_dtos.append(
                VariantDTO(
                    variant_id=v.variant_id,
                    variant_title=v.variant_title,
                    role=v.role,
                    experience_years=v.experience_years,
                    no_of_projects=v.no_of_projects,
                    tech_stacks=v.tech_stacks,
                    certifications=v.certifications,
                    projects=project_dtos,
                )
            )

        dto = IngestProfileDTO(
            candidate_id=profile.candidate_id,
            candidate_name=profile.candidate_name,
            resource_status=profile.resource_status.value,
            email=profile.email,
            education=profile.education,
            passout_year=profile.passout_year,
            dob=profile.dob,
            branch=profile.branch,
            user_id=user_id,
            variants=variant_dtos,
        )

        result = await use_case.execute(dto)

        details = None

        if result.details:
            details = [
                IngestVariantDetail(
                    variant_id=d["variant_id"],
                    variant_title=d["variant_title"],
                )
                for d in result.details
            ]

        return IngestProfileResponse(
            status=result.status,
            candidate_id=result.candidate_id,
            variants_ingested=result.variants_ingested,
            details=details,
            error_message=result.error_message,
        )

    finally:
        reset_log_context(user_token, action_token, section_token)

@router.post(
    "/match",
    response_model=ProfileMatchResponse,
    response_model_exclude_none=True,
    summary="Match candidate profiles against a job description",
)
async def match_profiles(
    body: ProfileMatchRequest,
    use_case: MatchProfilesUseCase = Depends(get_match_profiles_use_case),
) -> ProfileMatchResponse:
    """
    Accepts a plain text job description and runs the full profile matching
    pipeline:
    1. Hybrid retrieval (Dense + Keyword) on profile_variants
    2. BM25 Rescore + RRF (relevance primary, resource status as tiebreaker)
    3. Gemini LLM scoring → match_percentage, matching_skills, missing_skills
    4. Candidate deduplication (best variant per candidate)
    5. Returns up to 5 unique candidates
    """

    user_id = body.user_id

    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Match Candidates to Job",
        section="Profiles",
    )

    try:
        logger.info(
            "Searching for candidates matching the job description "
            "(%d characters)",
            len(body.job_details),
        )

        dto = ProfileMatchRequestDTO(
            job_details=body.job_details,
            variant_id=body.variant_id,
            user_id=user_id,
        )

        result = await use_case.execute(dto)

        matches = [
            ProfileMatchResultSchema(
                candidate_id=m.candidate_id,
                candidate_name=m.candidate_name,
                email=m.email if hasattr(m, "email") else "",
                variant_id=m.variant_id,
                variant_title=m.variant_title,
                role=m.role if hasattr(m, "role") else None,
                experience_years=m.experience_years,
                match_percentage=m.match_percentage,
                matching_skills=m.matching_skills,
                missing_skills=m.missing_skills,
                justification=m.justification,
            )
            for m in result.matches
        ] if result.matches else []

        return ProfileMatchResponse(
            status=result.status,
            matches=matches,
            error_message=result.error_message,
        )

    finally:
        reset_log_context(user_token, action_token, section_token)
# ─── Delete Candidate Profile ──────────────────────────────────────────────

def get_delete_profile_use_case(
    vector_store: QdrantVectorStoreAdapter = Depends(get_vector_store),
) -> DeleteProfileUseCase:
    return DeleteProfileUseCase(vector_store_port=vector_store)


@router.delete(
    "/candidates/{candidate_id}",
    response_model=DeleteProfileResponse,
    summary="Delete all profile variants for a candidate",
)
async def delete_candidate_profiles(
    user_id: str,
    candidate_id: str,
    action: str = "delete_profiles",
    use_case: DeleteProfileUseCase = Depends(get_delete_profile_use_case),
) -> DeleteProfileResponse:
    """
    Permanently deletes all profile variants for a given candidate_id
    from the vector store.

    Returns 404 if the candidate_id does not exist.
    """

    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Remove Candidate",
        section="Profiles",
    )

    try:
        logger.info(
            "Removing candidate profile '%s' from the knowledge base",
            candidate_id,
        )

        count = await use_case.execute(candidate_id)

        return DeleteProfileResponse(
            status="SUCCESS",
            message=(
                f"All {count} profile variant(s) for candidate "
                f"'{candidate_id}' have been deleted."
            ),
        )

    except CandidateNotFoundException as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
    except VectorStoreError as exc:
        # Qdrant unreachable / retries exhausted — a service outage, not a
        # client error. Log full details, return a clean 503 without
        # leaking internals.
        logger.error("Qdrant unavailable during profile delete: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is temporarily unavailable. Please retry shortly.",
        )

    finally:
        reset_log_context(user_token, action_token, section_token)
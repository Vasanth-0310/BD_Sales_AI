"""API router for Technical Preparation endpoints."""

from fastapi import APIRouter, Depends, Request

from src.application.dto.preparation_dto import TechnicalPrepRequestDTO
from src.application.use_cases.generate_technical_prep import GenerateTechnicalPrepUseCase
from src.infrastructure.ai.gemini_synthesizer_adapter import GeminiSynthesizerAdapter
from src.infrastructure.db.qdrant.qdrant_vector_store_adapter import QdrantVectorStoreAdapter
from src.presentation.schemas.preparation_schemas import (
    TechnicalPrepRequest,
    TechnicalPrepResponse,
    InterviewTopicSchema,
)
from src.common.logger import (
    get_logger,
    set_log_context,
    reset_log_context,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/preparations", tags=["Technical Preparation"])


# ─── Dependency Injection ─────────────────────────────────────────────────────

def get_vector_store(request: Request) -> QdrantVectorStoreAdapter:
    return request.app.state.vector_store


def get_synthesizer(request: Request) -> GeminiSynthesizerAdapter:
    return request.app.state.synthesizer


def get_technical_prep_use_case(
    vector_store: QdrantVectorStoreAdapter = Depends(get_vector_store),
    synthesizer: GeminiSynthesizerAdapter = Depends(get_synthesizer),
) -> GenerateTechnicalPrepUseCase:
    return GenerateTechnicalPrepUseCase(
        vector_store_port=vector_store,
        synthesizer_port=synthesizer,
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/technical",
    response_model=TechnicalPrepResponse,
    summary="Generate Technical Preparation Guide",
    description=(
        "Generates a targeted interview preparation guide for a selected candidate. "
        "Fetches the candidate's full profile from Qdrant using the variant_id, "
        "then combines it with the skill gap analysis from Step 3 to produce a "
        "structured preparation guide focused 80% on skill gaps and 20% on strengths."
    ),
)
async def generate_technical_prep(
    body: TechnicalPrepRequest,
    use_case: GenerateTechnicalPrepUseCase = Depends(get_technical_prep_use_case),
) -> TechnicalPrepResponse:

    user_id = body.user_id
    payload = body.payload

    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Generate Interview Prep Guide",
        section="Technical Preparation",
    )

    try:
        logger.info(
            "Generating interview preparation guide for candidate variant '%s' "
            "(%d skill gap(s) to cover)",
            payload.variant_id,
            len(payload.missing_skills or []),
        )

        dto = TechnicalPrepRequestDTO(
            job_details=payload.job_details,
            variant_id=payload.variant_id,
            matching_skills=payload.matching_skills,
            missing_skills=payload.missing_skills,
        )

        result_dto = await use_case.execute(dto)

        if result_dto.status == "FAILED":
            return TechnicalPrepResponse(
                status="FAILED",
                error_message=result_dto.error_message,
            )

        guide = None

        if (
            result_dto.result
            and result_dto.result.interview_preparation_guide
        ):
            guide = [
                InterviewTopicSchema(
                    topic=t.topic,
                    focus=t.focus,
                    questions=t.questions,
                )
                for t in result_dto.result.interview_preparation_guide
            ]

        return TechnicalPrepResponse(
            status="SUCCESS",
            candidate_name=result_dto.candidate_name,
            variant_title=result_dto.variant_title,
            technical_briefing_note=(
                result_dto.result.technical_briefing_note
                if result_dto.result
                else None
            ),
            interview_preparation_guide=guide,
        )

    finally:
        reset_log_context(user_token, action_token, section_token)
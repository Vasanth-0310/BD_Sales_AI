"""API router for Technical Preparation endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Request

from src.application.dto.preparation_dto import (
    BatchTechnicalPrepRequestDTO,
    CandidateInputDTO,
)
from src.application.use_cases.generate_batch_technical_prep import (
    GenerateBatchTechnicalPrepUseCase,
)
from src.infrastructure.ai.gemini_synthesizer_adapter import GeminiSynthesizerAdapter
from src.infrastructure.db.qdrant.qdrant_vector_store_adapter import QdrantVectorStoreAdapter
from src.presentation.schemas.preparation_schemas import (
    TechnicalPrepRequest,
    BatchTechnicalPrepResponse,
    InterviewTopicSchema,
    CandidatePrepResultSchema,
)
from src.common.logger import (
    get_logger,
    set_log_context,
    reset_log_context,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/preparations", tags=["Technical Preparation"])


# â”€â”€â”€ Dependency Injection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def get_vector_store(request: Request) -> QdrantVectorStoreAdapter:
    return request.app.state.vector_store


def get_synthesizer(request: Request) -> GeminiSynthesizerAdapter:
    return request.app.state.synthesizer


def get_batch_prep_use_case(
    vector_store: QdrantVectorStoreAdapter = Depends(get_vector_store),
    synthesizer: GeminiSynthesizerAdapter = Depends(get_synthesizer),
) -> GenerateBatchTechnicalPrepUseCase:
    return GenerateBatchTechnicalPrepUseCase(
        vector_store_port=vector_store,
        synthesizer_port=synthesizer,
    )


# â”€â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post(
    "/technical",
    response_model=BatchTechnicalPrepResponse,
    response_model_exclude_none=True,
    summary="Generate Technical Preparation Guides (1-10 candidates)",
    description=(
        "payload.candidates is ALWAYS a list: send 1 candidate to get 1 prep "
        "guide back, send up to 10 to get one guide per candidate. Candidates "
        "are processed sequentially. Partial failure is handled: a candidate "
        "that fails (variant not found, Gemini error) gets status FAILED with "
        "an error_message while the others still succeed. Top-level FAILED "
        "only when ALL candidates fail."
    ),
)
async def generate_technical_prep(
    body: TechnicalPrepRequest,
    batch_use_case: GenerateBatchTechnicalPrepUseCase = Depends(get_batch_prep_use_case),
) -> BatchTechnicalPrepResponse:

    user_id = body.user_id
    payload = body.payload

    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Generate Interview Prep Guide",
        section="Technical Preparation",
    )

    try:
        logger.info(
            "Generating interview preparation guides for %d candidate(s) "
            "(%d skill gap(s) on first candidate)",
            len(payload.candidates),
            len(payload.candidates[0].missing_skills or []),
        )

        dto = BatchTechnicalPrepRequestDTO(
            job_details=payload.job_details,
            candidates=[
                CandidateInputDTO(
                    variant_id=c.variant_id,
                    matching_skills=c.matching_skills,
                    missing_skills=c.missing_skills,
                )
                for c in payload.candidates
            ],
        )

        result_dto = await batch_use_case.execute(dto)

        if result_dto.status == "FAILED":
            return BatchTechnicalPrepResponse(
                status="FAILED",
                error_message=result_dto.error_message,
            )

        batch = result_dto.result
        return BatchTechnicalPrepResponse(
            status="SUCCESS",
            results=[
                CandidatePrepResultSchema(
                    variant_id=r.variant_id,
                    status=r.status,
                    candidate_name=r.candidate_name,
                    variant_title=r.variant_title,
                    technical_briefing_note=r.technical_briefing_note,
                    interview_preparation_guide=[
                        InterviewTopicSchema(
                            topic=t.topic,
                            focus=t.focus,
                            questions=t.questions,
                        )
                        for t in (r.interview_preparation_guide or [])
                    ] or None,
                    error_message=r.error_message,
                )
                for r in batch.results
            ],
        )

    finally:
        reset_log_context(user_token, action_token, section_token)

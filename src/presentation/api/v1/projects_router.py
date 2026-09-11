import json

from fastapi import APIRouter, Depends, HTTPException, Request

from src.application.dto.project_dto import (
    IngestProjectDTO,
    ProjectMatchRequestDTO,
)
from src.application.dto.sales_enablement_dto import SalesEnablementRequestDTO
from src.application.use_cases.ingest_project import IngestProjectUseCase
from src.application.use_cases.match_projects import MatchProjectsUseCase
from src.application.use_cases.generate_sales_enablement import GenerateSalesEnablementUseCase
from src.application.use_cases.delete_project import DeleteProjectUseCase, ProjectNotFoundException
from src.domain.exceptions.rag_exceptions import RAGBaseException, VectorStoreError
from src.infrastructure.ai.cached_embedding_adapter import CachedEmbeddingAdapter
from src.infrastructure.ai.gemini_synthesizer_adapter import GeminiSynthesizerAdapter
from src.infrastructure.db.qdrant.qdrant_vector_store_adapter import QdrantVectorStoreAdapter
from src.infrastructure.db.qdrant.semantic_chunker import SemanticChunker
from src.presentation.schemas.project_schemas import (
    IngestProjectRequest,
    ProjectIngestResponse,
    ProjectMatchRequest,
    ProjectMatchResponse,
    DeleteProjectResponse,
    SalesEnablementRequest,
    SalesEnablementResponse,
)
from src.common.logger import (
    get_logger,
    set_log_context,
    reset_log_context,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/projects", tags=["Projects"])


# ─── Dependency Injection (reads from app.state singletons) ───────────────────

def get_vector_store(request: Request) -> QdrantVectorStoreAdapter:
    return request.app.state.vector_store


def get_embedding_port(request: Request) -> CachedEmbeddingAdapter:
    return request.app.state.embedding_port


def get_synthesizer(request: Request) -> GeminiSynthesizerAdapter:
    return request.app.state.synthesizer


def get_chunker(request: Request) -> SemanticChunker:
    return request.app.state.chunker


def get_ingest_use_case(
    embedding_port: CachedEmbeddingAdapter = Depends(get_embedding_port),
    vector_store: QdrantVectorStoreAdapter = Depends(get_vector_store),
    chunker: SemanticChunker = Depends(get_chunker),
) -> IngestProjectUseCase:
    return IngestProjectUseCase(
        embedding_port=embedding_port,
        vector_store_port=vector_store,
        chunker=chunker,
    )


def get_match_use_case(
    embedding_port: CachedEmbeddingAdapter = Depends(get_embedding_port),
    vector_store: QdrantVectorStoreAdapter = Depends(get_vector_store),
    synthesizer: GeminiSynthesizerAdapter = Depends(get_synthesizer),
) -> MatchProjectsUseCase:
    return MatchProjectsUseCase(
        embedding_port=embedding_port,
        vector_store_port=vector_store,
        synthesizer_port=synthesizer,
    )


def get_sales_enablement_use_case(
    synthesizer: GeminiSynthesizerAdapter = Depends(get_synthesizer),
) -> GenerateSalesEnablementUseCase:
    return GenerateSalesEnablementUseCase(synthesizer_port=synthesizer)


# ─── Endpoints ────────────────────────────────────────────────────────────────
@router.post(
    "/ingest",
    summary="Ingest a project into the RAG vector store",
    response_model=ProjectIngestResponse,
    response_model_exclude_none=True,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["payload", "case_study"],
                        "properties": {
                            "payload": {
                                "type": "object",
                                "description": "Request envelope: metadata + project data",
                                "properties": {
                                    "user_id": {"type": "string"},
                                    "action": {
                                        "type": "string",
                                        "default": "ingest_project",
                                    },
                                    "project": {
                                        "type": "object",
                                        "description": "Project's Data",
                                        "properties": {
                                            "project_id": {"type": "string"},
                                            "project_name": {"type": "string"},
                                            "domain": {"type": "string"},
                                            "techstacks": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "description": {
                                                "type": "string",
                                            },
                                            "links": {
                                                "type": "object",
                                                "additionalProperties": {"type": "string"},
                                                "nullable": True,
                                            },
                                        },
                                        "required": [
                                            "project_id",
                                            "project_name",
                                            "domain",
                                            "techstacks",
                                            "description",
                                        ],
                                    },
                                },
                                "required": ["user_id", "action", "project"],
                            },
                            "case_study": {
                                "type": "string",
                                "format": "binary",
                                "description": "Project case study file (.docx or .pdf)",
                            },
                        },
                    },
                    "encoding": {
                        "payload": {"contentType": "application/json"},
                    },
                }
            },
        }
    },
)
async def ingest_project(
    request: Request,
    use_case: IngestProjectUseCase = Depends(get_ingest_use_case),
) -> ProjectIngestResponse:
    """
    Accepts multipart/form-data with:
    - **payload**: A JSON object containing all project metadata.
    - **case_study**: The project case study file (.docx or .pdf).
    """

    # Parse multipart form — F2 fix: a malformed/broken multipart body raises
    # Starlette internals; net it here so it's a clean 400, never a raw 500.
    try:
        form = await request.form()
    except Exception as e:
        logger.error("Failed to parse multipart form: %s", e)
        raise HTTPException(
            status_code=400,
            detail="Invalid or malformed multipart form data.",
        )

    # ── Parse JSON payload ────────────────────────────────────────────────────
    raw_payload = form.get("payload")

    if raw_payload is None:
        raise HTTPException(
            status_code=400,
            detail="Missing required field: 'payload'",
        )

    try:
        if isinstance(raw_payload, str):
            data = json.loads(raw_payload)
        else:
            data = json.loads(await raw_payload.read())

        req = IngestProjectRequest.model_validate(data)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payload: {e}",
        )

    user_id = req.user_id
    project = req.project

    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Add Project",
        section="Projects",
    )

    try:
        logger.info(
            f"Adding project '{project.project_name}' to the knowledge base"
        )

        # ── Parse file ─────────────────────────────────────────────
        case_study = form.get("case_study")

        if case_study is None:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: 'case_study'",
            )

        # A plain text form field with this name arrives as str, not
        # UploadFile — reject cleanly instead of AttributeError-crashing.
        if not hasattr(case_study, "read") or not hasattr(case_study, "filename"):
            raise HTTPException(
                status_code=400,
                detail="Field 'case_study' must be a file upload, not a text field.",
            )

        filename = case_study.filename or ""

        ext = (
            filename.rsplit(".", 1)[-1].lower()
            if "." in filename
            else ""
        )

        if ext not in ("docx", "pdf"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: '.{ext}'. "
                    "Only .docx and .pdf are accepted."
                ),
            )

        # Cap the read WITHOUT buffering the whole upload: read at most
        # max+1 bytes — anything longer is rejected before it can eat RAM.
        _MAX_CASE_STUDY_BYTES = 20 * 1024 * 1024  # 20 MB
        file_bytes = await case_study.read(_MAX_CASE_STUDY_BYTES + 1)
        if len(file_bytes) > _MAX_CASE_STUDY_BYTES:
            await case_study.close()
            raise HTTPException(
                status_code=413,
                detail="Case study file too large (max 20 MB).",
            )

        # ── Build DTO ──────────────────────────────────────────────
        dto = IngestProjectDTO(
            project_id=project.project_id,
            user_id=req.user_id,
            project_name=project.project_name,
            domain=project.domain,
            techstacks=project.techstacks,
            description=project.description,
            links=project.links if project.links else {},
        )

        # ── Execute ────────────────────────────────────────────────
        result = await use_case.execute(
            dto,
            file_bytes,
            filename,
        )

        return ProjectIngestResponse(
            project_id=result["project_id"],
            chunks_stored=result["chunks_stored"],
        )

    except RAGBaseException as e:
        logger.error(f"Ingest failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Ingest failed due to an internal error. Check server logs for details.",
        )

    finally:
        # Release the spooled temp-file handle — on Windows an unclosed
        # UploadFile locks its %TEMP% file until process exit.
        if case_study is not None and hasattr(case_study, "close"):
            try:
                await case_study.close()
            except Exception:
                pass
        reset_log_context(user_token, action_token, section_token)

@router.post(
    "/match",
    response_model=ProjectMatchResponse,
    response_model_exclude_none=True,
    summary="Match projects against a job description",
)
async def match_projects(
    body: ProjectMatchRequest,
    use_case: MatchProjectsUseCase = Depends(get_match_use_case),
) -> ProjectMatchResponse:
    """
    Accepts a job_details payload, runs the full 3-stage retrieval pipeline
    (Hybrid Summary → Dense Chunks → Gemini Synthesis), and returns
    the top 3 project matches with scores and justifications.
    """
    user_id = body.user_id

    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Match Projects to Job",
        section="Projects",
    )
    try:
        logger.info(
            f"Searching knowledge base for projects matching the job description "
            f"({len(body.job_details)} characters)"
        )

        dto = ProjectMatchRequestDTO(job_details=body.job_details, user_id=user_id)
        result = await use_case.execute(dto)

        return ProjectMatchResponse(
            status=result.status,
            matches=[m.model_dump() for m in result.matches] if result.matches else [],
            error_message=result.error_message,
        )
    finally:
        reset_log_context(user_token, action_token, section_token)


@router.post(
    "/sales-enablement",
    response_model=SalesEnablementResponse,
    response_model_exclude_none=True,
    summary="Generate Sales Enablement content for a job opportunity",
)
async def generate_sales_enablement(
    body: SalesEnablementRequest,
    use_case: GenerateSalesEnablementUseCase = Depends(get_sales_enablement_use_case),
) -> SalesEnablementResponse:
    """
    Accepts a job description and up to 3 matched project contexts.
    Calls Gemini to generate:
    - **Discovery Questions**: Questions for the BD team to ask the client.
    - **Talking Points**: BD pitch points based on our technical experience (no project names).
    - **Outreach Template**: A formal cold email from a BD Executive perspective.
    """
    user_id = body.user_id
    payload = body.payload
    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Generate Sales Content",
        section="Projects",
    )
    try:
        logger.info(
            "Preparing sales content using %d matched project(s) for the job description",
            len(payload.projects),
        )

        dto = SalesEnablementRequestDTO(
            job_details=payload.job_details,
            projects=[p.model_dump() for p in payload.projects],
        )

        result = await use_case.execute(dto)

        return SalesEnablementResponse(
            status=result.status,
            discovery_questions=result.discovery_questions,
            talking_points=result.talking_points,
            outreach_subject=result.outreach_subject,
            outreach_template=result.outreach_template,
            error_message=result.error_message,
        )
    finally:
        reset_log_context(user_token, action_token, section_token)


# ─── Delete Project ───────────────────────────────────────────────────────────

def get_delete_project_use_case(
    vector_store: QdrantVectorStoreAdapter = Depends(get_vector_store),
) -> DeleteProjectUseCase:
    return DeleteProjectUseCase(vector_store_port=vector_store)


@router.delete(
    "/{project_id}",
    response_model=DeleteProjectResponse,
    summary="Delete a project and all its data",
)
async def delete_project(
    user_id: str,
    project_id: str,
    action: str = "delete_project",
    use_case: DeleteProjectUseCase = Depends(get_delete_project_use_case),
) -> DeleteProjectResponse:
    """
    Permanently deletes a project's summary vector and all its chunk vectors
    from the vector store.

    Returns 404 if the project_id does not exist.
    """
    user_token, action_token, section_token = set_log_context(
        user_id=user_id,
        action="Remove Project",
        section="Projects",
    )
    logger.info("Removing project '%s' from the knowledge base", project_id)
    try:
        await use_case.execute(project_id, user_id)
        return DeleteProjectResponse(
            status="SUCCESS",
            message=f"Project '{project_id}' and all its data have been deleted.",
        )
    except ProjectNotFoundException as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        # Destructive-path guard: blank user_id is a client error, not a 500.
        raise HTTPException(status_code=400, detail=str(exc))
    except VectorStoreError as exc:
        # Qdrant unreachable / retries exhausted — a service outage, not a
        # client error. Log full details, return a clean 503 without
        # leaking internals.
        logger.error("Qdrant unavailable during project delete: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is temporarily unavailable. Please retry shortly.",
        )
    finally:
        reset_log_context(user_token, action_token, section_token)

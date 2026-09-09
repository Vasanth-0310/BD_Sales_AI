import time
import uuid
import asyncio
import contextvars
from src.domain.entities.project import Project, ProjectChunk
from src.domain.interfaces.rag.i_embedding_port import IEmbeddingPort
from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.domain.exceptions.rag_exceptions import (
    RAGBaseException,
    DocumentExtractionError,
)
from src.infrastructure.file_processing.document_extractor import DocumentExtractor
from src.infrastructure.db.qdrant.semantic_chunker import SemanticChunker
from src.application.dto.project_dto import IngestProjectDTO
from src.common.logger import get_logger
from src.common.backup_service import BackupService


logger = get_logger(__name__)


class IngestProjectUseCase:
    """
    Application use case that orchestrates the full project ingestion pipeline:
    1. Extract text from the uploaded case study document (.docx / .pdf)
    2. Build a rich summary text and embed it
    3. Semantically chunk the case study text
    4. Batch embed all chunks
    5. Upsert everything into Qdrant
    """

    def __init__(
        self,
        embedding_port: IEmbeddingPort,
        vector_store_port: IVectorStorePort,
        chunker: SemanticChunker,
    ) -> None:
        self._embedding_port = embedding_port
        self._vector_store_port = vector_store_port
        self._chunker = chunker

    async def execute(
        self,
        dto: IngestProjectDTO,
        file_bytes: bytes,
        filename: str,
    ) -> dict:
        total_start = time.perf_counter()
        logger.info(f"================ INGEST PIPELINE STARTED ================")
        logger.info(f"[STEP 0] Initializing ingest for project_id='{dto.project_id}', name='{dto.project_name}'")

        try:
            # ── Step 1: Extract case study text (CPU-heavy: pdfplumber/docx) ──
            # Run in a thread so the event loop keeps serving other requests.
            logger.info("[STEP 1] Starting case study text extraction")
            extract_start = time.perf_counter()
            loop = asyncio.get_running_loop()
            # copy_context so logs inside the worker thread still carry the
            # request's user_id/action (threads don't inherit ContextVars).
            ctx = contextvars.copy_context()
            case_study_text = await loop.run_in_executor(
                None, lambda: ctx.run(DocumentExtractor.extract_text, file_bytes, filename)
            )
            logger.info(f"[STEP 1] Document extraction completed in {time.perf_counter() - extract_start:.2f}s")
            logger.info(f"[STEP 1] Extracted a total of {len(case_study_text)} characters")

            # ── Step 2: Build Project entity ─────────────────────────────────
            logger.info("[STEP 2] Building Project domain entity")
            project = Project(
                project_id=dto.project_id,
                user_id=dto.user_id,
                name=dto.project_name,
                domain=dto.domain,
                techstacks=dto.techstacks,
                description=dto.description,
                links=dto.links,
                case_study_text=case_study_text,
            )
            logger.info("[STEP 2] Project entity built successfully")

            # ── Step 3: Embed summary AND chunk the case study in parallel ───
            # The summary embedding is network-bound (Gemini) while semantic
            # chunking is CPU-bound (MiniLM) — they don't depend on each other.
            summary_text = (
                f"{project.name} | Domain: {project.domain} | "
                f"Stack: {', '.join(project.techstacks)} | {project.description}"
            )
            logger.info(f"[STEP 3] Summary text prepared ({len(summary_text)} chars)")

            logger.info("[STEP 3/4] Embedding summary and chunking case study in parallel...")
            parallel_start = time.perf_counter()
            summary_vector, raw_chunks = await asyncio.gather(
                self._embedding_port.embed_document(summary_text),
                loop.run_in_executor(None, lambda: ctx.run(self._chunker.chunk, case_study_text)),
            )
            logger.info(
                f"[STEP 3/4] Parallel embed+chunk completed in "
                f"{time.perf_counter() - parallel_start:.2f}s "
                f"({len(raw_chunks)} semantic chunks)"
            )

            # ── Step 4: Format chunks ────────────────────────────────────────
            project_chunks: list[ProjectChunk] = []
            for i, raw_chunk in enumerate(raw_chunks):
                project_chunks.append(ProjectChunk(
                    chunk_id=str(uuid.uuid4()),
                    project_id=project.project_id,
                    user_id=project.user_id,
                    project_name=project.name,
                    domain=project.domain,
                    techstacks=project.techstacks,
                    text=raw_chunk["text"],
                    token_count=raw_chunk["token_count"],
                    sequence_index=i,
                ))
            logger.info(f"[STEP 5] Successfully formatted {len(project_chunks)} chunks")

            # ── Step 5: Batch embed all chunks ───────────────────────────────
            chunk_texts = [c.text for c in project_chunks]
            if chunk_texts:
                logger.info(f"[STEP 6] Sending {len(chunk_texts)} chunks to Gemini for batch embedding")
                batch_start = time.perf_counter()
                chunk_vectors = await self._embedding_port.embed_documents_batch(chunk_texts)
                logger.info(f"[STEP 6] Batch embedding completed in {time.perf_counter() - batch_start:.2f}s")
            else:
                logger.info("[STEP 6] No chunks to embed (empty case study)")
                chunk_vectors = []

            # ── Step 6: Delete old data and upsert new ───────────────────────
            # Chunks are written BEFORE the summary — the summary is what Stage 1
            # retrieval keys on, so it acts as the commit point: if the flow dies
            # mid-way we're left with orphan chunks (harmless) rather than a
            # summary pointing at zero evidence (silently bad matches).
            logger.info(f"[STEP 7] Connecting to Qdrant vector store")
            upsert_start = time.perf_counter()

            try:
                logger.info(f"[STEP 7.1] Deleting any existing data for project '{project.project_id}'")
                await self._vector_store_port.delete_project(
                    project.project_id, user_id=getattr(dto, "user_id", None) or None,
                )

                # From this point on the project exists nowhere — retry each
                # write once before giving up so a transient Qdrant blip
                # doesn't permanently destroy the previously stored project.
                last_exc: Exception | None = None
                for attempt in range(2):
                    try:
                        if project_chunks:
                            logger.info(f"[STEP 7.2] Upserting {len(project_chunks)} chunk vectors to Qdrant")
                            await self._vector_store_port.upsert_project_chunks(project_chunks, chunk_vectors)

                        logger.info(f"[STEP 7.3] Upserting summary vector to Qdrant (commit point)")
                        await self._vector_store_port.upsert_project_summary(project, summary_vector)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        logger.warning(f"[STEP 7] Qdrant write failed (attempt {attempt + 1}/2): {exc}")
                if last_exc is not None:
                    raise last_exc
            except Exception as exc:
                logger.error("[STEP 7] Qdrant operation failed, writing to DLQ")
                BackupService.backup_failed_project(
                    project_id=dto.project_id,
                    project_name=dto.project_name,
                    error_msg=str(exc),
                    summary_text=summary_text,
                    chunks_count=len(project_chunks)
                )
                raise  # Re-raise to fail the request gracefully

            logger.info(f"[STEP 7] All Qdrant operations completed in {time.perf_counter() - upsert_start:.2f}s")

            total_time = time.perf_counter() - total_start
            logger.info(f"================ INGEST COMPLETE in {total_time:.2f}s ================")
            logger.info(f"Final Status: project_id='{dto.project_id}', chunks_stored={len(project_chunks)}")

            return {
                "project_id": dto.project_id,
                "chunks_stored": len(project_chunks),
            }

        except RAGBaseException:
            raise
        except Exception as e:
            # Wrap unexpected failures (corrupt files, chunker crashes, etc.)
            # so the router returns a clean error instead of an unhandled 500.
            logger.error(f"Ingest pipeline failed: [{type(e).__name__}] {e}", exc_info=True)
            raise DocumentExtractionError(reason=f"Ingest failed: [{type(e).__name__}] {e}") from e

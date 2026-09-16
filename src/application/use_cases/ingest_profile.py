import time
from dataclasses import asdict

from src.domain.entities.profile import ProjectInProfile
from src.domain.interfaces.rag.i_embedding_port import IEmbeddingPort
from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.domain.exceptions.rag_exceptions import RAGBaseException
from src.application.dto.profile_dto import (
    IngestProfileDTO,
    IngestProfileResponseDTO,
)
from src.common.logger import get_logger
from src.infrastructure.backup_service import BackupService
from src.common.jd_text import truncate_text


logger = get_logger(__name__)


class IngestProfileUseCase:
    """
    Application use case that orchestrates the profile variant ingestion pipeline:
    1. Loop through each variant in the payload
    2. Build a natural-language summary text for embedding
    3. Build a combined_text field for BM25 keyword search
    4. Embed the summary text via Gemini
    5. Upsert variant into Qdrant (using variant_id as point ID)
    """

    def __init__(
        self,
        embedding_port: IEmbeddingPort,
        vector_store_port: IVectorStorePort,
    ) -> None:
        self._embedding_port = embedding_port
        self._vector_store_port = vector_store_port

    async def execute(self, dto: IngestProfileDTO) -> IngestProfileResponseDTO:
        total_start = time.perf_counter()
        logger.info("======== PROFILE INGEST PIPELINE STARTED ========")
        logger.info(
            "[STEP 0] candidate_id=%s  name='%s'  variants=%d",
            dto.candidate_id,
            dto.candidate_name,
            len(dto.variants),
        )

        try:
            if not dto.variants:
                return IngestProfileResponseDTO.failed(
                    reason="No variants provided in the payload."
                )

            # ── Steps 1-4 (per variant, no I/O): build texts + payloads ──
            prepared: list[dict] = []
            for variant_dto in dto.variants:
                summary_text = self._build_summary_text(dto, variant_dto)
                combined_text = self._build_combined_text(variant_dto)

                projects_typed = []
                for p in variant_dto.projects:
                    projects_typed.append(ProjectInProfile(
                        project_id=p.project_id,
                        project_name=p.project_name,
                        domain=p.domain,
                        tech_stack=p.tech_stack,
                        links=p.links,
                        description=p.description,
                    ))

                payload = {
                    # Candidate-level
                    "candidate_id": dto.candidate_id,
                    "candidate_name": dto.candidate_name,
                    "user_id": dto.user_id or None,  # Tenant ownership (legacy points lack it)
                    "resource_status": dto.resource_status,
                    "email": dto.email,
                    "education": dto.education,
                    "passout_year": dto.passout_year,
                    "dob": dto.dob,
                    "branch": dto.branch,
                    # Variant-level
                    "variant_id": variant_dto.variant_id,
                    "variant_title": variant_dto.variant_title,
                    "role": variant_dto.role,
                    "experience_years": variant_dto.experience_years,
                    "no_of_projects": variant_dto.no_of_projects,
                    "tech_stacks": variant_dto.tech_stacks,
                    "certifications": variant_dto.certifications,
                    "projects": [asdict(p) for p in projects_typed],
                    # BM25 fields (for keyword search)
                    "combined_text": combined_text,
                    "tech_stacks_text": " ".join(variant_dto.tech_stacks),
                }
                # A stored user_id=null is NOT matched by the tenant filter's
                # is_empty clause in all Qdrant versions — omit the key
                # entirely so unowned points stay legacy-tolerant.
                if not payload.get("user_id"):
                    payload.pop("user_id", None)
                prepared.append({
                    "variant_id": variant_dto.variant_id,
                    "variant_title": variant_dto.variant_title,
                    "summary_text": summary_text,
                    "payload": payload,
                })

            # ── Step 5: Batch-embed ALL variant summaries in one call ────
            # Sequential per-variant embedding was N Gemini round-trips;
            # the batch API does it in one (with per-item cache hits free).
            batch_start = time.perf_counter()
            vectors = await self._embedding_port.embed_documents_batch(
                [p["summary_text"] for p in prepared]
            )
            logger.info(
                "[BATCH] Embedded %d variant summaries in %.2fs",
                len(vectors),
                time.perf_counter() - batch_start,
            )

            # ── Step 6: Upsert each variant to Qdrant ────────────────────
            ingested_details: list[dict] = []
            ingested_variant_ids: list[str] = []
            for item, vector in zip(prepared, vectors):
                variant_start = time.perf_counter()
                try:
                    await self._vector_store_port.upsert_profile_variant(
                        variant_id=item["variant_id"],
                        vector=vector,
                        payload=item["payload"],
                    )
                    ingested_variant_ids.append(item["variant_id"])
                except Exception as exc:
                    logger.error(
                        "[VARIANT %s] Upsert failed, writing to DLQ (%d variant(s) "
                        "already committed before this failure)",
                        item["variant_id"], len(ingested_variant_ids),
                    )
                    BackupService.backup_failed_profile(
                        candidate_id=dto.candidate_id,
                        variant_id=item["variant_id"],
                        payload=item["payload"],
                        error_msg=str(exc)
                    )
                    raise  # Re-raise to fail the request gracefully

                logger.info(
                    "[VARIANT %s] '%s' ingested in %.2fs",
                    item["variant_id"],
                    item["variant_title"],
                    time.perf_counter() - variant_start,
                )
                ingested_details.append({
                    "variant_id": item["variant_id"],
                    "variant_title": item["variant_title"],
                })

            total_time = time.perf_counter() - total_start

            # Zombie-variant reconciliation: a re-ingest where the candidate
            # removed or renamed variants in the source system must not leave
            # stale variants matching queries forever. Runs AFTER all upserts
            # succeeded — a failed ingest reconciles nothing. GATED OFF for
            # partial payloads (differential sync) — deleting "stale" variants
            # that simply weren't part of this partial payload would destroy
            # the candidate's existing data (zero-deletion violation).
            if ingested_variant_ids and dto.reconcile_variants:
                try:
                    removed = await self._vector_store_port.reconcile_profile_variants(
                        candidate_id=dto.candidate_id,
                        keep_variant_ids=ingested_variant_ids,
                        user_id=dto.user_id or None,
                    )
                    if removed:
                        logger.info(
                            "[RECONCILE] Removed %d stale variant(s) for candidate %s",
                            removed, dto.candidate_id,
                        )
                except Exception as reconcile_err:
                    logger.warning(
                        "Variant reconciliation failed (non-fatal): %s", reconcile_err
                    )

            logger.info(
                "======== PROFILE INGEST COMPLETE in %.2fs  |  variants=%d ========",
                total_time,
                len(ingested_details),
            )

            return IngestProfileResponseDTO.success(
                candidate_id=dto.candidate_id,
                details=ingested_details,
            )

        except RAGBaseException as e:
            logger.error("Profile ingest failed: %s", e)
            return IngestProfileResponseDTO.failed(reason=str(e))
        except Exception as e:
            logger.error("Unexpected error in profile ingest: %s", e, exc_info=True)
            return IngestProfileResponseDTO.failed(reason="Internal error: please check server logs.")

    @staticmethod
    def _build_summary_text(dto: IngestProfileDTO, variant_dto) -> str:
        """Build a natural-language summary for the embedding model.

        Uses prose format instead of pipe-delimited to produce
        significantly better semantic embeddings.
        """
        tech_str = ", ".join(variant_dto.tech_stacks) if variant_dto.tech_stacks else "N/A"

        parts = [
            f"{dto.candidate_name} is a {variant_dto.variant_title} (Role: {variant_dto.role}) "
            f"with {variant_dto.experience_years} years of experience.",
            f"Core technologies include {tech_str}.",
        ]

        # Add project summaries
        if variant_dto.projects:
            project_parts = []
            for p in variant_dto.projects:
                p_tech = ", ".join(p.tech_stack) if p.tech_stack else ""
                domain_str = f" in the {p.domain} domain" if p.domain else ""
                project_parts.append(
                    f"{p.project_name}{domain_str} using {p_tech}: "
                    f"{truncate_text(p.description, 800)}"
                )
            parts.append(
                "Project experience includes " + "; ".join(project_parts) + "."
            )

        # Add certifications
        if variant_dto.certifications:
            parts.append(
                "Certifications: " + "; ".join(variant_dto.certifications) + "."
            )

        # The embedding provider has an 8,000-character input ceiling.  Keep
        # title and stack first, then bounded project evidence, rather than
        # silently truncating an arbitrary tail inside the adapter.
        return truncate_text(" ".join(parts), 7_500)

    @staticmethod
    def _build_combined_text(variant_dto) -> str:
        """Build the combined_text field used for BM25 keyword search.

        Concatenates all searchable text: variant_title + tech_stacks +
        certifications + all project descriptions.
        """
        parts = [
            variant_dto.variant_title,
            variant_dto.role,
            " ".join(variant_dto.tech_stacks),
            " ".join(variant_dto.certifications),
        ]

        for p in variant_dto.projects:
            parts.append(p.project_name)
            parts.append(p.domain)
            parts.append(" ".join(p.tech_stack))
            parts.append(p.description)

        return " ".join(parts)

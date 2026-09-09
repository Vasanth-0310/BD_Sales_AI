from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.common.logger import get_logger

logger = get_logger(__name__)


class CandidateNotFoundException(Exception):
    """Raised when a candidate_id does not exist in the vector store."""
    def __init__(self, candidate_id: str) -> None:
        super().__init__(f"Candidate '{candidate_id}' not found in the database.")
        self.candidate_id = candidate_id


class VariantNotFoundException(Exception):
    """Raised when a variant_id does not exist or does not belong to the candidate."""
    def __init__(self, variant_id: str, candidate_id: str) -> None:
        super().__init__(
            f"Variant '{variant_id}' not found for candidate '{candidate_id}'."
        )
        self.variant_id = variant_id
        self.candidate_id = candidate_id


class DeleteProfileUseCase:
    """
    Deletes profile variants from the vector store.

    Supports two modes:
    - If variant_id is provided: deletes only that single variant
      (cross-checked against candidate_id for safety).
    - If variant_id is omitted: deletes ALL variants for the candidate.

    Raises CandidateNotFoundException / VariantNotFoundException for 404.
    Returns the number of variants deleted.
    """

    def __init__(self, vector_store_port: IVectorStorePort) -> None:
        self._vector_store = vector_store_port

    async def execute(
        self,
        candidate_id: str,
        user_id: str = "",
        variant_id: str | None = None,
    ) -> int:
        user = user_id or None

        # ── Single variant deletion ──────────────────────────────────
        if variant_id:
            logger.info(
                "[DeleteProfile] Deleting single variant_id='%s' "
                "for candidate_id='%s'",
                variant_id, candidate_id,
            )
            deleted = await self._vector_store.delete_profile_variant_by_id(
                variant_id=variant_id,
                candidate_id=candidate_id,
                user_id=user,
            )
            if not deleted:
                logger.warning(
                    "[DeleteProfile] variant_id='%s' not found for "
                    "candidate_id='%s' — returning 404",
                    variant_id, candidate_id,
                )
                raise VariantNotFoundException(variant_id, candidate_id)

            logger.info(
                "[DeleteProfile] variant_id='%s' deleted successfully.",
                variant_id,
            )
            return 1

        # ── All variants deletion (existing behavior) ────────────────
        logger.info(
            "[DeleteProfile] Checking existence of candidate_id='%s'",
            candidate_id,
        )

        exists = await self._vector_store.check_candidate_exists(candidate_id, user)
        if not exists:
            logger.warning(
                "[DeleteProfile] candidate_id='%s' not found — returning 404",
                candidate_id,
            )
            raise CandidateNotFoundException(candidate_id)

        logger.info(
            "[DeleteProfile] Deleting all variants for candidate_id='%s'...",
            candidate_id,
        )
        count = await self._vector_store.delete_profiles_by_candidate_id(candidate_id, user)
        logger.info(
            "[DeleteProfile] candidate_id='%s' — %d variant(s) deleted.",
            candidate_id, count,
        )
        return count


"""Use cases deleting candidate profiles / profile variants."""

from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.common.logger import get_logger

logger = get_logger(__name__)


class CandidateNotFoundException(Exception):
    """Raised when a candidate_id does not exist in the vector store."""
    def __init__(self, candidate_id: str) -> None:
        super().__init__(f"Candidate '{candidate_id}' not found in the database.")
        self.candidate_id = candidate_id


class VariantNotFoundException(Exception):
    """Raised when a variant_id does not exist or does not belong to the tenant."""
    def __init__(self, variant_id: str) -> None:
        super().__init__(f"Variant '{variant_id}' not found in the database.")
        self.variant_id = variant_id


class DeleteProfileUseCase:
    """
    Deletes a candidate's ENTIRE profile data from the vector store —
    the profile itself along with ALL of its variants.
    """

    def __init__(self, vector_store_port: IVectorStorePort) -> None:
        self._vector_store = vector_store_port

    async def execute(self, candidate_id: str, user_id: str = "") -> int:
        # DESTRUCTIVE path: blank user_id would disable tenant isolation
        # entirely (cross-tenant deletion) — refuse rather than degrade.
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required for delete operations.")
        user = user_id.strip()

        logger.info("[DeleteProfile] Checking existence of candidate_id='%s'", candidate_id)

        exists = await self._vector_store.check_candidate_exists(candidate_id, user)
        if not exists:
            logger.warning("[DeleteProfile] candidate_id='%s' not found — returning 404", candidate_id)
            raise CandidateNotFoundException(candidate_id)

        logger.info("[DeleteProfile] Deleting all variants for candidate_id='%s'...", candidate_id)
        count = await self._vector_store.delete_profiles_by_candidate_id(candidate_id, user)
        logger.info("[DeleteProfile] candidate_id='%s' — %d variant(s) deleted.", candidate_id, count)
        return count


class DeleteProfileVariantUseCase:
    """
    Deletes a SINGLE profile variant by its variant_id alone.

    No candidate_id is required — the variant_id IS the Qdrant point ID, so
    it addresses exactly one variant. Tenant ownership is still enforced
    (a variant owned by another user raises VariantNotFoundException).
    """

    def __init__(self, vector_store_port: IVectorStorePort) -> None:
        self._vector_store = vector_store_port

    async def execute(self, variant_id: str, user_id: str = "") -> int:
        # DESTRUCTIVE path: blank user_id would disable tenant isolation
        # entirely (cross-tenant deletion) — refuse rather than degrade.
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required for delete operations.")

        logger.info("[DeleteVariant] Deleting variant_id='%s'", variant_id)

        user = user_id.strip()
        deleted = await self._vector_store.delete_profile_variant_by_id(
            variant_id=variant_id,
            candidate_id=None,      # not provided — no cross-check
            user_id=user,
        )
        if not deleted:
            logger.warning(
                "[DeleteVariant] variant_id='%s' not found or not owned by the "
                "requesting tenant — returning 404",
                variant_id,
            )
            raise VariantNotFoundException(variant_id)

        logger.info("[DeleteVariant] variant_id='%s' deleted successfully.", variant_id)
        return 1

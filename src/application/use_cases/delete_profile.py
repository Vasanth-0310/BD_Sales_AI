from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.common.logger import get_logger

logger = get_logger(__name__)


class CandidateNotFoundException(Exception):
    """Raised when a candidate_id does not exist in the vector store."""
    def __init__(self, candidate_id: str) -> None:
        super().__init__(f"Candidate '{candidate_id}' not found in the database.")
        self.candidate_id = candidate_id


class DeleteProfileUseCase:
    """
    Deletes all profile variants for a candidate from the vector store.
    Raises CandidateNotFoundException if the candidate_id does not exist (404).
    Returns the number of variants deleted.
    """

    def __init__(self, vector_store_port: IVectorStorePort) -> None:
        self._vector_store = vector_store_port

    async def execute(self, candidate_id: str) -> int:
        logger.info("[DeleteProfile] Checking existence of candidate_id='%s'", candidate_id)

        exists = await self._vector_store.check_candidate_exists(candidate_id)
        if not exists:
            logger.warning("[DeleteProfile] candidate_id='%s' not found — returning 404", candidate_id)
            raise CandidateNotFoundException(candidate_id)

        logger.info("[DeleteProfile] Deleting all variants for candidate_id='%s'...", candidate_id)
        count = await self._vector_store.delete_profiles_by_candidate_id(candidate_id)
        logger.info("[DeleteProfile] candidate_id='%s' — %d variant(s) deleted.", candidate_id, count)
        return count

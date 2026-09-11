from src.domain.interfaces.rag.i_vector_store_port import IVectorStorePort
from src.common.logger import get_logger

logger = get_logger(__name__)


class ProjectNotFoundException(Exception):
    """Raised when a project_id does not exist in the vector store."""
    def __init__(self, project_id: str) -> None:
        super().__init__(f"Project '{project_id}' not found in the database.")
        self.project_id = project_id


class DeleteProjectUseCase:
    """
    Deletes a project (summary + all chunks) from the vector store.
    Raises ProjectNotFoundException if the project_id does not exist (404).
    """

    def __init__(self, vector_store_port: IVectorStorePort) -> None:
        self._vector_store = vector_store_port

    async def execute(self, project_id: str, user_id: str = "") -> None:
        # DESTRUCTIVE path: blank user_id would disable tenant isolation
        # entirely (cross-tenant deletion) — refuse rather than degrade.
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required for delete operations.")

        user = user_id.strip()
        logger.info("[DeleteProject] Checking existence of project_id='%s'", project_id)

        # Tenant scoping: a project owned by another user looks "not found".
        exists = await self._vector_store.check_project_exists(project_id, user)
        if not exists:
            logger.warning("[DeleteProject] project_id='%s' not found — returning 404", project_id)
            raise ProjectNotFoundException(project_id)

        logger.info("[DeleteProject] Deleting project_id='%s' from vector store...", project_id)
        await self._vector_store.delete_project(project_id, user)
        logger.info("[DeleteProject] project_id='%s' deleted successfully.", project_id)

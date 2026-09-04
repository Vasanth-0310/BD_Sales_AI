from datetime import datetime
from typing import Optional
from pymongo import ASCENDING
from src.domain.entities.user_session import UserSession
from src.domain.enums.scrape_status import SessionStatus
from src.domain.interfaces.session_store.i_session_store import ISessionStore
from src.infrastructure.mongodb.connection import get_database
from src.common.logger import get_logger

logger = get_logger(__name__)

_COLLECTION = "user_sessions"


def _to_document(session: UserSession) -> dict:
    return {
        "user_id": session.user_id,
        "website": session.website,
        "storage_state": session.storage_state,
        "cookies": session.cookies,
        "local_storage": session.local_storage,
        "session_storage": session.session_storage,
        "status": session.status.value,
        "created_at": session.created_at,
        "updated_at": session.updated_at,
        "last_used": session.last_used,
        "expires_at": session.expires_at,
    }


def _from_document(doc: dict) -> UserSession:
    return UserSession(
        id=str(doc.get("_id", "")),
        user_id=doc["user_id"],
        website=doc["website"],
        storage_state=doc.get("storage_state", {}),
        cookies=doc.get("cookies", []),
        local_storage=doc.get("local_storage", {}),
        session_storage=doc.get("session_storage", {}),
        status=SessionStatus(doc.get("status", SessionStatus.ACTIVE.value)),
        created_at=doc.get("created_at", datetime.utcnow()),
        updated_at=doc.get("updated_at", datetime.utcnow()),
        last_used=doc.get("last_used", datetime.utcnow()),
        expires_at=doc.get("expires_at"),
    )


class MongoDBSessionRepository(ISessionStore):
    """
    MongoDB implementation of ISessionStore using Motor async driver.
    Collection: user_sessions
    """

    @property
    def _collection(self):
        return get_database()[_COLLECTION]

    async def ensure_indexes(self) -> None:
        """Create required indexes on startup."""
        await self._collection.create_index(
            [("user_id", ASCENDING), ("website", ASCENDING)],
            unique=True,
            name="idx_user_website_unique",
        )
        await self._collection.create_index(
            [("status", ASCENDING)],
            name="idx_status",
        )
        logger.info("MongoDB session indexes ensured.")

    async def get_session(self, user_id: str, domain: str) -> Optional[UserSession]:
        doc = await self._collection.find_one({"user_id": user_id, "website": domain})
        if not doc:
            return None
        return _from_document(doc)

    async def save_session(self, session: UserSession) -> None:
        doc = _to_document(session)
        created_at = doc.pop("created_at")
        # Preserve the original creation timestamp on updates: created_at must
        # only be set on insert ($setOnInsert), otherwise every save overwrites
        # it and the audit trail is lost.
        await self._collection.update_one(
            {"user_id": session.user_id, "website": session.website},
            {
                "$set": doc,
                "$setOnInsert": {"created_at": created_at},
            },
            upsert=True,
        )
        logger.info(f"Session saved for user '{session.user_id}' on '{session.website}'.")

    async def delete_session(self, user_id: str, domain: str) -> None:
        await self._collection.delete_one({"user_id": user_id, "website": domain})
        logger.info(f"Session deleted for user '{user_id}' on '{domain}'.")

    async def get_all_active_sessions(self) -> list[UserSession]:
        cursor = self._collection.find({"status": SessionStatus.ACTIVE.value})
        docs = await cursor.to_list(length=None)
        return [_from_document(doc) for doc in docs]

    async def mark_session_expired(self, user_id: str, domain: str) -> None:
        await self._collection.update_one(
            {"user_id": user_id, "website": domain},
            {"$set": {"status": SessionStatus.EXPIRED.value, "updated_at": datetime.utcnow()}},
        )
        logger.info(f"Session marked EXPIRED for user '{user_id}' on '{domain}'.")

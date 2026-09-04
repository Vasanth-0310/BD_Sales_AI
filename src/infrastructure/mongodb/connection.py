from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from src.common.logger import get_logger

logger = get_logger(__name__)

_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def connect(uri: str, db_name: str) -> None:
    """Initialize the Motor async MongoDB client."""
    global _client, _database
    _client = AsyncIOMotorClient(uri)
    _database = _client[db_name]

    # Verify connection
    await _client.admin.command("ping")
    logger.info("Database connection established.")


async def disconnect() -> None:
    """Close the MongoDB client connection."""
    global _client, _database
    if _client:
        _client.close()
        _client = None
        _database = None
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """Return the active database instance."""
    if _database is None:
        raise RuntimeError("MongoDB is not connected. Call connect() first.")
    return _database

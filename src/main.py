import sys
import asyncio
import logging

# Fix for Windows: Patchright/Playwright requires ProactorEventLoop to create subprocesses.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import time

from src.infrastructure.mongodb.connection import connect, disconnect, get_database
from src.infrastructure.mongodb.session_repository import MongoDBSessionRepository
from src.application.services.scheduler.session_refresh_scheduler import SessionRefreshScheduler
from src.infrastructure.browser.browser_pool import BrowserPool
from src.infrastructure.browser.nodriver_pool import NodriverPool
from src.presentation.api.v1.scraper_router import router as scraper_router
from src.presentation.api.v1.projects_router import router as projects_router
from src.presentation.api.v1.profiles_router import router as profiles_router
from src.presentation.api.v1.preparations_router import router as preparations_router
from src.presentation.api.v1.dashboard_router import router as dashboard_router
from src.common.config import settings
from src.common.logger import get_logger

# ─── RAG Singletons ──────────────────────────────────────────────────────────
from src.infrastructure.db.qdrant.semantic_chunker import SemanticChunker
from src.infrastructure.db.qdrant.qdrant_vector_store_adapter import QdrantVectorStoreAdapter
from src.infrastructure.ai.gemini_embedding_adapter import GeminiEmbeddingAdapter
from src.infrastructure.ai.cached_embedding_adapter import CachedEmbeddingAdapter
from src.infrastructure.ai.gemini_synthesizer_adapter import GeminiSynthesizerAdapter
from src.infrastructure.ai.gemini_extractor import GeminiExtractor
from src.infrastructure.ai.gemini_company_profiler import GeminiCompanyProfiler
from src.infrastructure.metrics.metrics_repository import MetricsRepository

logger = get_logger(__name__)

# ─── Scheduler (module-level to allow clean shutdown) ───────────────────────
_scheduler: SessionRefreshScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    global _scheduler

    from src.common.logger import set_log_context, reset_log_context

    # Ensure Windows event loop policy allows subprocesses
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    def _system_log(message: str, level: int = logging.INFO) -> None:
        """Emit a lifecycle event visible to the frontend under System."""
        tokens = set_log_context(None, "Server Status", "System")
        try:
            logger.log(level, message)
        finally:
            reset_log_context(*tokens)

    # ── STARTUP ──────────────────────────────────────────────────────────────
    _system_log(f"'{settings.app_name}' service is starting...")

    # Connect to MongoDB
    await connect(uri=settings.mongodb_uri, db_name=settings.mongodb_db_name)

    # Ensure database indexes
    repo = MongoDBSessionRepository()
    await repo.ensure_indexes()

    # Start the session refresh scheduler
    _scheduler = SessionRefreshScheduler(session_store=repo)
    _scheduler.start()

    # ── RAG Singletons: initialize once, store on app.state ──────────────────
    logger.info("Initializing RAG components...")

    # 0. MetricsRepository — shared singleton, injected into all AI adapters
    app.state.metrics = MetricsRepository()

    # 1. SemanticChunker — loads all-MiniLM-L6-v2 (~90MB) once
    app.state.chunker = SemanticChunker()

    # 2. Qdrant vector store — opens persistent async connection
    app.state.vector_store = QdrantVectorStoreAdapter()

    # 3. Gemini embedding (inner) + MongoDB-cached wrapper
    _inner_embedding = GeminiEmbeddingAdapter()
    app.state.embedding_port = CachedEmbeddingAdapter(
        inner=_inner_embedding,
        db=get_database(),
    )

    # 4. Gemini synthesizer — receives metrics for token tracking
    app.state.synthesizer = GeminiSynthesizerAdapter(metrics=app.state.metrics)

    # 5. GeminiExtractor and GeminiCompanyProfiler singletons with metrics
    app.state.extractor = GeminiExtractor(metrics=app.state.metrics)
    app.state.company_profiler = GeminiCompanyProfiler(metrics=app.state.metrics)

    _system_log("AI components loaded successfully.")
    _system_log("Service started and ready to accept requests.")
    yield

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    _system_log("Service is shutting down...")

    if _scheduler:
        _scheduler.stop()

    # Gracefully shut down the persistent browser pools if they were started
    await BrowserPool.shutdown()
    await NodriverPool.shutdown()

    await disconnect()

    # ── Windows ProactorEventLoop transport drain ─────────────────────────
    # On Windows, browser subprocesses leave asyncio ProactorSubprocessTransport
    # handles open. Without this drain, Python's GC tries to __del__ them after
    # the event loop closes → noisy "unclosed transport / I/O operation on closed
    # pipe" ResourceWarnings on every shutdown.
    # Yielding control twice gives the ProactorEventLoop time to process all
    # pending transport-close callbacks before the loop exits.
    if sys.platform == "win32":
        try:
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        except Exception:
            pass

    _system_log("Service shut down complete.")


# ─── FastAPI Application ─────────────────────────────────────────────────────
app = FastAPI(
    title="Job Scraper",
    description=(
        "AI-powered universal job scraper. Supports any job portal URL "
        "with hybrid authentication detection and per-user session management."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Middleware ───────────────────────────────────────────────────────────────
# NOTE: allow_origins=["*"] combined with allow_credentials=True is rejected
# by browsers. Wildcard origins are only safe WITHOUT credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time
    logger.info(f"==> [ENDPOINT TIMING] {request.method} {request.url.path} took {process_time:.2f}s total")
    return response

# ─── Routers ─────────────────────────────────────────────────────────────────
app.include_router(scraper_router)
app.include_router(projects_router)
app.include_router(profiles_router)
app.include_router(preparations_router)
app.include_router(dashboard_router)


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "ok", "app": settings.app_name, "env": settings.app_env}


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Run from project root: python -m src.main
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=settings.debug)

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Ignore unknown .env keys instead of crashing startup — a new key
        # (e.g. for a script) must never take the whole API server down.
        extra="ignore",
    )

    # Application
    app_name: str = "universal-job-scraper"
    app_env: str = "development"
    debug: bool = True
    # Brand name used in BD-generated cold outreach. Change here — never in prompts.
    company_name: str = "Soft Suave"

    # MongoDB
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "job_scraper_db"

    # PostgreSQL (used by scripts/sync_postgres_to_qdrant.py)
    postgres_url: str = ""

    # Gemini AI
    gemini_api_key: str = ""
    gemini_model: str = ""
    # Used when the primary Gemini model is temporarily overloaded or rate
    # limited. Keep this as a separately configurable model for deployments
    # with different model access.
    gemini_fallback_model: str = "gemini-3.6-flash"
    # Soft timeout per Gemini call on the scrape path — a call slower than this
    # is cancelled and retried once (Gemini latency varies 3s-22s for the same
    # prompt; the retry usually lands on a fast route). 0 disables.
    gemini_call_timeout_s: int = 12
    # Hedged requests: if a Gemini call hasn't finished within this delay, fire
    # a second identical call in parallel and keep whichever completes first.
    # Cures Gemini's serving-route latency roulette (7s vs 30s for the same
    # prompt). Costs extra tokens only when the first call is slow. 0 disables.
    gemini_hedge_delay_s: int = 6
    # Disable Gemini "thinking" tokens on speed-critical calls (extraction,
    # profiling). Falls back automatically if the model rejects the setting.
    gemini_disable_thinking: bool = True

    # Qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_summary_collection: str = ""        # e.g. projects_summaries
    qdrant_chunks_collection: str = ""         # e.g. projects_chunks
    qdrant_profile_variants_collection: str = ""  # e.g. profile_variants
    qdrant_vector_size: int = 1536             # gemini-embedding-001 output dim
    # Shared search scope for the internal BD talent/project library.  This is
    # an access boundary only: it must never influence retrieval scores or LLM
    # matching.  Set a unique value per customer/workspace in multi-tenant
    # deployments, then run the non-destructive workspace backfill script.
    rag_workspace_id: str = "softsuave"
    # Upper bound for the unscored Qdrant MatchText candidate scan.  The
    # application subsequently applies BM25/RRF; a small scroll page here can
    # otherwise hide a valid result before it is ranked.
    rag_keyword_candidate_limit: int = 300

    # Project matching
    # Below this fused match score a project is not meaningful enough to pitch.
    # Kept well above 0: the reranker is told to return the top 3, so the floor
    # only bites when even the "best" projects are weak matches for the JD.
    project_match_min_score: float = 0.35

    # Profile matching
    # Candidates scoring below this percentage are filtered out before the
    # top-5 slice (mirrors project_match_min_score). 0 disables the filter.
    # A candidate who evidences a primary job skill is a useful partial match;
    # eligibility prevents unrelated profiles from passing, so do not discard
    # relevant developing candidates merely because they lack secondary skills.
    profile_match_min_percentage: int = 15

    # RAGAS evaluation
    # Set RAGAS_CAPTURE_ENABLED=false in .env to stop saving evaluation samples
    # to disk (e.g. for privacy-sensitive deployments). When enabled, every live
    # profile/project match call appends one sample to data/eval_samples.jsonl;
    # run scripts/evaluate_pipeline.py to score the collected samples.
    ragas_capture_enabled: bool = True
    # Offline RAGAS evaluation limits. Production samples may contain dozens
    # of retrieved profiles; constrain judge input and retries so an eval run
    # cannot appear to hang indefinitely or exhaust the Gemini quota.
    ragas_max_contexts: int = 5
    ragas_context_max_chars: int = 800
    ragas_answer_max_chars: int = 1800
    ragas_judge_model: str = "gemini-3.1-flash-lite"
    ragas_call_timeout_s: int = 90
    ragas_max_retries: int = 1
    ragas_max_workers: int = 4

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug_value(cls, value: object) -> object:
        """Accept conventional deployment labels supplied through DEBUG."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"development", "dev"}:
                return True
        return value


    # Scheduler
    session_refresh_interval_hours: int = 3
    # SECURITY: when True, a user with no session falls back to the shared
    # "default_user" session (its cookies are reused across users). Only keep
    # enabled in single-tenant/demo environments.
    session_fallback_enabled: bool = True

    # Browser
    # Primary stealth engine for the shared pool / generic scraping path.
    # One of: "cloakbrowser" | "patchright" | "nodriver".
    # The engine order is CloakBrowser -> Patchright -> nodriver; this setting
    # only changes which engine is tried first.
    browser_engine: str = "cloakbrowser"
    # Browsers run HEADED (visible process, window hidden while idle) — headed
    # Chrome passes Cloudflare/verification checks far more reliably than
    # headless. Keep False unless debugging a specific headless issue.
    browser_cloak_headless: bool = False
    browser_nodriver_headless: bool = False
    browser_timeout_ms: int = 30000
    browser_profile_root: str = ".browser_profiles"
    # Keep the headed pool browsers' windows hidden from the taskbar
    # (they still run fully headed — only the window is invisible).
    # Set False to watch the pool browsers during debugging.
    browser_hide_windows: bool = True

    # Logged-in account/organization names rendered by platform navigation
    # chrome (e.g. Upwork's sidebar shows the account's org name on every
    # page). Comma-separated. These lines are stripped before LLM extraction
    # so they can never be mistaken for job titles or company names.
    boilerplate_account_names: str = "UMA Microapp"


settings = Settings()

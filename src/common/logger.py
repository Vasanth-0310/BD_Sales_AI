import logging
import re
import sys
from contextvars import ContextVar
from typing import Optional


# -----------------------------------------------------------------------------
# Credential scrubbing
# -----------------------------------------------------------------------------
# Defense-in-depth: even if a log call forgets to redact, nothing secret ever
# reaches the log file or console. Applied to BOTH the formatted message and
# exception text (tracebacks frequently embed full connection strings).
# -----------------------------------------------------------------------------

_SCRUB_PATTERNS = [
    # mongodb+srv://user:password@host  /  postgres://user:pass@host etc.
    (re.compile(r"(\w[\w.\-]*):([^@\s/]{3,})@"), r"\1:***@"),
    # Gemini API keys (AIza...)
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{10,}"), "***GEMINI-KEY***"),
    # api_key / api-key / token / password = value   (query strings, JSON)
    (
        re.compile(
            r"((?:api[-_]?key|token|password|secret|authorization)"
            r"[\"']?\s*[:=]\s*[\"']?)"
            r"([^\s\"'&,}]{6,})",
            re.IGNORECASE,
        ),
        r"\1***",
    ),
]


def _scrub(text: str) -> str:
    for pattern, replacement in _SCRUB_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class CredentialScrubberFilter(logging.Filter):
    """Remove credentials from every record before it is written anywhere."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _scrub(str(record.msg))
        if record.args:
            try:
                record.args = tuple(
                    _scrub(a) if isinstance(a, str) else a for a in record.args
                )
            except Exception:
                record.args = None
        # Scrub tracebacks too — driver exceptions often embed full
        # connection strings (mongodb://user:pass@host/...).
        if record.exc_info:
            try:
                fmt = logging.Formatter()
                record.exc_text = _scrub(fmt.formatException(record.exc_info))
                record.exc_info = None  # already rendered + scrubbed
            except Exception:
                pass
        return True


# -----------------------------------------------------------------------------
# Request-scoped logging context
# -----------------------------------------------------------------------------
# These values are automatically available to every logger running inside
# the current async request.
#
# Example:
#   user_id = "123"
#   action = "scrape"
#
# Any downstream logger will automatically include:
#   user_id=123 | action=scrape
#
# When there is no active user request (startup, shutdown, scheduler, etc.),
# the values remain None.
# -----------------------------------------------------------------------------

_user_id_context: ContextVar[Optional[str]] = ContextVar(
    "user_id_context",
    default=None,
)

_action_context: ContextVar[Optional[str]] = ContextVar(
    "action_context",
    default=None,
)

# Business-facing section (Scraper / Projects / Profiles / Technical
# Preparation / Dashboard / Health / System). Lines WITHOUT a section are
# internal plumbing and are hidden from the frontend AI Logs view.
_section_context: ContextVar[Optional[str]] = ContextVar(
    "section_context",
    default=None,
)


def set_log_context(
    user_id: Optional[str],
    action: Optional[str],
    section: Optional[str] = None,
):
    """
    Set user/action/section context for the current async request.

    Returns tokens that MUST be passed to reset_log_context()
    after the request completes.
    """
    user_token = _user_id_context.set(user_id)
    action_token = _action_context.set(action)
    section_token = _section_context.set(section)

    return user_token, action_token, section_token


def reset_log_context(*tokens) -> None:
    """
    Restore the previous logging context.
    Accepts the 3 tokens returned by set_log_context().
    """
    _user_id_context.reset(tokens[0])
    _action_context.reset(tokens[1])
    if len(tokens) > 2:
        _section_context.reset(tokens[2])


class RequestContextFilter(logging.Filter):
    """
    Inject request-scoped user_id and action into every LogRecord.

    This means existing logger calls such as:

        logger.info("Checking session...")

    automatically become:

        user_id=123 | action=scrape | Checking session...
    """

    def filter(self, record: logging.LogRecord) -> bool:
        user_id = _user_id_context.get()
        action = _action_context.get()
        section = _section_context.get()

        # Store values on the LogRecord so the formatter can use them.
        record.user_id = user_id if user_id is not None else "-"
        record.action = action if action is not None else "-"
        # NOTE: LogRecord already has a built-in `.module` attribute (source
        # filename), so the business section is stored as `record.section`.
        record.section = section if section is not None else "-"

        return True


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Usage:
        logger = get_logger(__name__)
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Prevent messages from being handled again by the root logger.
    logger.propagate = False

    # -------------------------------------------------------------------------
    # Console Handler
    # -------------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # -------------------------------------------------------------------------
    # File Handler
    # -------------------------------------------------------------------------
    file_handler = logging.FileHandler(
        "rag_pipeline.log",
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)

    # -------------------------------------------------------------------------
    # Formatter
    # -------------------------------------------------------------------------
    #
    # Every log line now has:
    #
    # timestamp
    # level
    # module
    # user_id
    # action
    # message
    #
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | "
            "%(levelname)-8s | "
            "%(name)s | "
            "user_id=%(user_id)s | "
            "action=%(action)s | "
            "section=%(section)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # -------------------------------------------------------------------------
    # Request Context Filter
    # -------------------------------------------------------------------------
    context_filter = RequestContextFilter()
    scrubber = CredentialScrubberFilter()

    for handler in (console_handler, file_handler):
        handler.addFilter(context_filter)
        handler.addFilter(scrubber)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
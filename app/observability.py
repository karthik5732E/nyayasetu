# =============================================================================
# observability.py — LangSmith Tracing & Observability
# Wraps all major operations with LangSmith tracing.
# Sign up for free: https://smith.langchain.com
# =============================================================================

import os
import logging
from functools import wraps
from typing import Callable, Any

from langsmith import Client, traceable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangSmith Client
# ---------------------------------------------------------------------------

_langsmith_client: Client = None

def get_langsmith_client() -> Client:
    """Get or create the LangSmith client."""
    global _langsmith_client
    if _langsmith_client is None:
        api_key = os.environ.get("LANGSMITH_API_KEY", "")
        if api_key and api_key != "your_langsmith_api_key_here":
            _langsmith_client = Client(api_key=api_key)
            logger.info("LangSmith client initialized")
        else:
            logger.warning("LANGSMITH_API_KEY not set — tracing disabled")
    return _langsmith_client


def is_tracing_enabled() -> bool:
    """Check if LangSmith tracing is enabled."""
    api_key = os.environ.get("LANGSMITH_API_KEY", "")
    return bool(api_key and api_key != "your_langsmith_api_key_here")


# ---------------------------------------------------------------------------
# Trace Decorators (already used in nodes.py via @traceable)
# Additional helpers for API-level tracing
# ---------------------------------------------------------------------------

def trace_api_call(endpoint: str):
    """Decorator to trace FastAPI endpoint calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not is_tracing_enabled():
                return await func(*args, **kwargs)

            client = get_langsmith_client()
            project = os.environ.get("LANGSMITH_PROJECT", "nyaya-setu")

            @traceable(
                name=f"api_{endpoint}",
                run_type="chain",
                project_name=project,
                extra={"endpoint": endpoint}
            )
            async def traced(*a, **kw):
                return await func(*a, **kw)

            return await traced(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Manual Trace Helper
# ---------------------------------------------------------------------------

async def trace_run(name: str, inputs: dict, outputs: dict, metadata: dict = None):
    """Manually log a trace to LangSmith."""
    if not is_tracing_enabled():
        return

    try:
        client = get_langsmith_client()
        if client:
            # Create a run in LangSmith
            client.create_run(
                name=name,
                run_type="chain",
                inputs=inputs,
                outputs=outputs,
                extra=metadata or {}
            )
    except Exception as e:
        logger.error(f"Failed to log trace: {e}")


# ---------------------------------------------------------------------------
# Startup: Verify LangSmith connection
# ---------------------------------------------------------------------------

def init_observability():
    """Initialize observability on app startup."""
    client = get_langsmith_client()
    if client:
        try:
            projects = list(client.list_projects())
            logger.info(f"LangSmith connected. Projects: {len(projects)}")
        except Exception as e:
            logger.warning(f"LangSmith connection check failed: {e}")
    else:
        logger.info("LangSmith tracing is disabled. Set LANGSMITH_API_KEY to enable.")

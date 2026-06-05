"""
langfuse_setup.py — Langfuse observability integration (v4 compatible)
=======================================================================
Drop in project root (same folder as agents.py, retriever.py etc.)

Degrades gracefully when Langfuse is not installed / keys not set.
The pipeline runs identically without tracing — no errors, no changes.

Required .env additions:
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com   # or http://localhost:3000

Optional:
    LANGFUSE_ENABLED=false   # disable without removing keys

Langfuse v4 API (what you have installed):
    - Import:      from langfuse.langchain import CallbackHandler
    - Constructor: CallbackHandler() — no args, reads env vars automatically
    - Credentials: initialise Langfuse(public_key=...) once, then reuse
    - Metadata:    session_id, trace_name etc. go in the invoke() config dict,
                   NOT on the handler object
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level Langfuse client — created once, reused across all requests.
# This is the v4 pattern: initialise the client with credentials once,
# then CallbackHandler() picks it up automatically with no constructor args.
_langfuse_client = None


def langfuse_enabled() -> bool:
    """
    True only when:
      1. LANGFUSE_ENABLED is not "false"
      2. LANGFUSE_PUBLIC_KEY is set
      3. LANGFUSE_SECRET_KEY is set
      4. langfuse package is importable
    """
    if os.getenv("LANGFUSE_ENABLED", "true").lower() == "false":
        return False
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        return False
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        return False
    try:
        import langfuse  # noqa: F401
        return True
    except ImportError:
        logger.warning(
            "langfuse package not installed — tracing disabled. "
            "Fix: pip install langfuse"
        )
        return False


def _get_langfuse_client():
    """
    Returns a singleton Langfuse client, creating it on first call.

    v4 pattern: initialise Langfuse() once with credentials, then
    CallbackHandler() will automatically find and use this client.
    Doing this once at module level avoids repeated initialisation overhead.
    """
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key = os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key = os.getenv("LANGFUSE_SECRET_KEY"),
            host       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        return _langfuse_client
    except Exception as e:
        logger.warning(f"Langfuse client initialisation failed: {e}")
        return None


def get_callback_handler(
    session_id: Optional[str] = None,
    trace_name: Optional[str] = None,
    metadata:   Optional[dict] = None,
    user_id:    Optional[str] = None,
):
    """
    Creates a fresh Langfuse CallbackHandler for one pipeline run.

    Returns (handler, config_extras) where config_extras is a dict you
    merge into your LangGraph invoke config to attach session/trace metadata.

    v4 pattern:
      handler      = CallbackHandler()          # no constructor args
      invoke_config = {
          "callbacks": [handler],
          "metadata": {                          # session metadata goes here
              "langfuse_session_id": session_id,
              "langfuse_user_id":    user_id,
          },
          "run_name": trace_name,
      }
      await _graph.ainvoke(state, config=invoke_config)

    Returns (None, {}) if Langfuse is disabled/unavailable — callers check
    for None and skip tracing, pipeline runs identically either way.
    """
    if not langfuse_enabled():
        return None, {}

    try:
        # Ensure the Langfuse client is initialised with credentials first
        client = _get_langfuse_client()
        if client is None:
            return None, {}

        from langfuse.langchain import CallbackHandler

        # v4: CallbackHandler takes no constructor args —
        # it automatically uses the Langfuse client we initialised above
        handler = CallbackHandler()

        # Build the config dict to pass into ainvoke(config=...)
        # This is how v4 attaches session_id, trace name etc. to the trace
        config_extras: dict = {"callbacks": [handler]}

        lf_metadata: dict = {}
        if session_id:
            lf_metadata["langfuse_session_id"] = session_id
        if user_id:
            lf_metadata["langfuse_user_id"] = user_id
        if metadata:
            lf_metadata.update(metadata)
        if lf_metadata:
            config_extras["metadata"] = lf_metadata
        if trace_name:
            config_extras["run_name"] = trace_name

        return handler, config_extras

    except Exception as e:
        logger.warning(f"Langfuse handler creation failed (non-critical): {e}")
        return None, {}


def flush_handler(handler) -> None:
    """
    Flush buffered Langfuse events after a pipeline run.

    v4: flush on the global client, not the handler directly.
    Falls back to handler.flush() for safety.
    Safe to call with handler=None.
    """
    if handler is None:
        return
    # Prefer flushing via the global client (v4 recommended way)
    if _langfuse_client is not None:
        try:
            _langfuse_client.flush()
            return
        except Exception as e:
            logger.warning(f"Langfuse client flush failed (non-critical): {e}")
    # Fallback: try flushing the handler directly
    try:
        handler.flush()
    except Exception as e:
        logger.warning(f"Langfuse handler flush failed (non-critical): {e}")
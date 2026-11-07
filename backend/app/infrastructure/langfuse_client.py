import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _BACKEND_ROOT.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv(_BACKEND_ROOT / ".env")


class _PrivateObservation:
    """Prevent generation prompts and completions from reaching the SDK."""

    def __init__(self, current):
        self.current = current

    def update(self, **kwargs: Any) -> None:
        if "output" in kwargs:
            kwargs["output"] = "[REDACTED]"
        self.current.update(**kwargs)


class _NoopObservation:
    def update(self, **kwargs: Any) -> None:
        return None


def tracing_enabled() -> bool:
    """True when credentials exist and this process should export traces.

    Pytest sets PYTEST_CURRENT_TEST. Unit tests stay offline unless
    LANGFUSE_TRACING_IN_TESTS is explicitly turned on.
    """
    if os.getenv("PYTEST_CURRENT_TEST") and os.getenv(
        "LANGFUSE_TRACING_IN_TESTS", ""
    ).strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    flag = os.getenv("LANGFUSE_TRACING_ENABLED", "true").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def get_langfuse():
    """Return the Langfuse client when credentials are present."""
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise RuntimeError(
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY before using Langfuse."
        )
    from langfuse import get_client

    return get_client()


def flush_langfuse() -> None:
    if not tracing_enabled():
        return
    try:
        get_langfuse().flush()
    except Exception:
        return


@contextmanager
def observation(**kwargs: Any):
    """Start a Langfuse observation, or a no-op when tracing is off."""
    if not tracing_enabled():
        yield _NoopObservation()
        return
    try:
        client = get_langfuse()
        is_generation = kwargs.get("as_type") == "generation"
        if is_generation:
            kwargs["input"] = "[REDACTED]"
        context = client.start_as_current_observation(**kwargs)
    except Exception:
        yield _NoopObservation()
        return
    current = context.__enter__()
    try:
        yield _PrivateObservation(current) if is_generation else current
    except Exception as exc:
        try:
            current.update(level="ERROR", status_message=type(exc).__name__)
        except Exception:
            pass
        raise
    finally:
        # Do not let SDK exception recording export sensitive exception messages.
        context.__exit__(None, None, None)

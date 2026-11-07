import pytest

from app.infrastructure.langfuse_client import flush_langfuse, get_langfuse, tracing_enabled


def test_langfuse_connection() -> None:
    if not tracing_enabled():
        pytest.skip("Langfuse tracing is disabled for this process.")
    client = get_langfuse()
    with client.start_as_current_observation(
        as_type="span",
        name="langfuse-connection-test",
        input={"message": "Testing Langfuse connection"},
    ) as span:
        span.update(output={"status": "Langfuse tracing is working"})
    flush_langfuse()


if __name__ == "__main__":
    client = get_langfuse()
    with client.start_as_current_observation(
        as_type="span",
        name="langfuse-connection-test",
        input={"message": "Testing Langfuse connection"},
    ) as span:
        span.update(output={"status": "Langfuse tracing is working"})
    client.flush()
    print("Test observation submitted.")

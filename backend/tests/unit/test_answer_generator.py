from app.domain.retrieval.models import EvidenceSource
from app.infrastructure.answer_generation.openai_compatible_generator import (
    OpenAICompatibleAnswerGenerator,
)


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "The semester begins in July. [S1]"}}]}


class FakeHttp:
    def __init__(self):
        self.payload = None
        self.url = None

    def post(self, url, **kwargs):
        self.url = url
        self.payload = kwargs
        return FakeResponse()


def test_answer_generator_accepts_gpt_oss_full_width_citation_brackets():
    pattern = OpenAICompatibleAnswerGenerator.CITATION_PATTERN
    assert pattern.findall("Supported by source 【S1】 and [S2].") == ["S1", "S2"]


def test_openai_compatible_answer_generator_extracts_citations():
    http = FakeHttp()
    generator = OpenAICompatibleAnswerGenerator(
        base_url="https://api.example.test/v1/",
        api_key="test-key",
        model_id="test-model",
        http_client=http,
    )
    evidence = [
        EvidenceSource(
            evidence_id="S1",
            chunk_id="chunk-1",
            source_file="calendar.pdf",
            page_number=1,
            excerpt="The semester begins in July.",
            score=0.9,
        )
    ]

    result = generator.generate("When does the semester begin?", evidence)

    assert result.citations == ["S1"]
    assert result.answer.endswith("[S1]")
    assert http.url == "https://api.example.test/v1/chat/completions"
    assert http.payload["json"]["model"] == "test-model"
    assert http.payload["json"]["temperature"] == 0.2
    assert len(http.payload["json"]["messages"]) == 2
    assert http.payload["json"]["messages"][0]["role"] == "system"
    assert "untrusted" in http.payload["json"]["messages"][0]["content"].lower()
    assert "[S1]" in http.payload["json"]["messages"][1]["content"]
    assert OpenAICompatibleAnswerGenerator.CITATION_PATTERN.findall("[W1] [S1]") == ["W1", "S1"]


def test_invalid_usage_does_not_discard_completion(monkeypatch):
    from contextlib import contextmanager

    from app.infrastructure.answer_generation import openai_compatible_generator as module

    updates = []

    class Observation:
        def update(self, **kwargs):
            updates.append(kwargs)

    @contextmanager
    def observation(**kwargs):
        yield Observation()

    monkeypatch.setattr(module, "observation", observation)

    class Response(FakeResponse):
        def json(self):
            return {
                **super().json(),
                "usage": {
                    "prompt_tokens": "invalid",
                    "completion_tokens": "12",
                    "total_tokens": {},
                },
            }

    class Http:
        def post(self, *args, **kwargs):
            return Response()

    generator = OpenAICompatibleAnswerGenerator(
        "https://example.test", "key", "model", http_client=Http()
    )
    assert generator._complete([]) == "The semester begins in July. [S1]"
    assert updates == [
        {"output": "The semester begins in July. [S1]", "usage_details": {"output": 12}}
    ]

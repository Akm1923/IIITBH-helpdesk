import json

import pytest

from app.application.evaluation.llm_judge import LlmJudge, parse_judge_response


def test_parse_judge_response_reads_fenced_json() -> None:
    raw = '```json\n{"score": 0.5, "reason": "Partial match."}\n```'

    score, reason = parse_judge_response(raw)

    assert score == 0.5
    assert reason == "Partial match."


def test_parse_judge_response_scales_percentage_scores() -> None:
    score, _reason = parse_judge_response('{"score": 80, "reason": "Mostly correct."}')

    assert score == 0.8


def test_judge_returns_zero_when_the_model_output_is_not_json() -> None:
    judge = LlmJudge(lambda _name, _prompt: "not json")

    evaluation = judge.correctness(
        input={"question": "What is the course code?"},
        output={"answer": "CS203", "contexts": ["CS203 is the code."]},
        expected_output="CS203",
    )

    assert evaluation.name == "correctness"
    assert evaluation.value == 0.0
    assert "Judge failed" in evaluation.comment


def test_judge_passes_the_reference_answer_to_the_model() -> None:
    seen = {}

    def complete(name: str, prompt: str) -> str:
        seen["name"] = name
        seen["prompt"] = prompt
        return json.dumps({"score": 1, "reason": "The course code matches."})

    evaluation = LlmJudge(complete).correctness(
        input={"question": "What is the course code?"},
        output={"answer": "CS203 [S1]", "contexts": ["Object Oriented Programming is CS203."]},
        expected_output="CS203",
    )

    assert seen["name"] == "judge-correctness"
    assert "CS203" in seen["prompt"]
    assert evaluation.value == 1.0


def test_faithfulness_includes_retrieved_context() -> None:
    seen = {}

    def complete(_name: str, prompt: str) -> str:
        seen["prompt"] = prompt
        return json.dumps({"score": 1, "reason": "Supported."})

    LlmJudge(complete).faithfulness(
        input={"question": "What is the course code?"},
        output={"answer": "CS203", "contexts": ["The code is CS203."]},
    )

    assert "The code is CS203." in seen["prompt"]


def test_parse_judge_response_rejects_text_without_json() -> None:
    with pytest.raises(ValueError):
        parse_judge_response("I cannot score this.")

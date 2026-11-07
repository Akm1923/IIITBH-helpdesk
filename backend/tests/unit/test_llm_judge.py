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


def test_judge_returns_no_score_when_the_model_output_is_not_json() -> None:
    judge = LlmJudge(lambda _name, _prompt: "not json")

    evaluation = judge.correctness(
        input={"question": "What is the course code?"},
        output={"answer": "CS203", "contexts": ["CS203 is the code."]},
        expected_output="CS203",
    )

    assert evaluation.name == "correctness"
    assert evaluation.value is None
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


@pytest.mark.parametrize(
    "value", [None, True, "bad", [], {}, float("nan"), float("inf"), -float("inf")]
)
def test_parse_judge_rejects_invalid_scores(value):
    with pytest.raises(ValueError):
        parse_judge_response(json.dumps({"score": value}))


def test_parse_judge_rejects_missing_score():
    with pytest.raises(ValueError):
        parse_judge_response('{"reason": "missing"}')


def test_judge_catches_key_error():
    def complete(*args):
        raise KeyError("score")

    assert LlmJudge(complete).score("correctness", "prompt").value is None


def test_mean_excludes_failures_and_counts_them():
    from app.application.evaluation.llm_judge import mean_score

    scores = mean_score(
        item_results=[
            {
                "evaluations": [
                    {"name": "correctness", "value": 0.8},
                    {"name": "correctness", "value": None},
                    {"name": "faithfulness", "value": None},
                ]
            }
        ]
    )
    values = {score.name: score.value for score in scores}
    assert values["mean_correctness"] == 0.8
    assert values["failed_correctness"] == 1
    assert values["mean_faithfulness"] is None
    assert values["failed_faithfulness"] == 1


@pytest.mark.parametrize("value, expected", [("0.5", 0.5), (80, 0.8), (-3, 0.0), (101, 1.0)])
def test_parse_preserves_valid_scaling_and_clamping(value, expected):
    assert parse_judge_response(json.dumps({"score": value}))[0] == expected

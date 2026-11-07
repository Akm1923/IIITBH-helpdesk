import json
import re
from collections.abc import Callable

from langfuse import Evaluation

from app.infrastructure.answer_generation.openai_compatible_generator import (
    AnswerGenerationError,
    OpenAICompatibleAnswerGenerator,
)

JudgeComplete = Callable[[str, str], str]

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_response(raw: str) -> tuple[float, str]:
    """Read a judge score in [0, 1] and a short reason from model text."""
    match = _JSON_OBJECT.search(raw)
    if match is None:
        raise ValueError("Judge response did not contain a JSON object.")
    data = json.loads(match.group(0))
    score = float(data["score"])
    if score > 1:
        score = score / 100 if score <= 100 else 1.0
    score = min(1.0, max(0.0, score))
    reason = str(data.get("reason") or "").strip()[:800]
    return score, reason


def _question_text(item_input) -> str:
    if isinstance(item_input, dict):
        return str(item_input.get("question") or "")
    return "" if item_input is None else str(item_input)


def _expected_text(expected_output) -> str:
    if isinstance(expected_output, dict):
        return str(expected_output.get("answer") or "")
    return "" if expected_output is None else str(expected_output)


def _answer_text(output) -> str:
    if isinstance(output, dict):
        return str(output.get("answer") or "")
    return "" if output is None else str(output)


def _contexts(output) -> list[str]:
    if not isinstance(output, dict):
        return []
    contexts = output.get("contexts") or []
    return [str(item) for item in contexts if str(item).strip()]


class LlmJudge:
    """Score RAG outputs with the same chat model used for answers."""

    def __init__(self, complete: JudgeComplete) -> None:
        self.complete = complete

    @classmethod
    def from_generator(cls, generator: OpenAICompatibleAnswerGenerator) -> "LlmJudge":
        def complete(name: str, prompt: str) -> str:
            return generator._complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an evaluation judge for an IIIT Bhagalpur RAG assistant. "
                            "Return only a JSON object with numeric score from 0 to 1 and a short reason. "
                            "Do not add markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.0,
                name=name,
            )

        return cls(complete)

    def score(self, name: str, prompt: str) -> Evaluation:
        try:
            raw = self.complete("judge-" + name, prompt)
            value, reason = parse_judge_response(raw)
        except (AnswerGenerationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            return Evaluation(
                name=name,
                value=0.0,
                comment="Judge failed: " + type(exc).__name__,
            )
        return Evaluation(name=name, value=value, comment=reason or None)

    def evaluators(self) -> list:
        return [self.correctness, self.faithfulness, self.relevance]

    def correctness(self, *, input, output, expected_output=None, **kwargs) -> Evaluation:
        prompt = (
            "Score whether the candidate answer contains the reference fact and does not contradict it. "
            "Ignore citation markers such as [S1] or [W1]. Extra correct explanation is acceptable. "
            "Use 1 for a match, 0 for a miss or contradiction, and a value between them for a partial match.\n\n"
            f"QUESTION:\n{_question_text(input)}\n\n"
            f"REFERENCE ANSWER:\n{_expected_text(expected_output)}\n\n"
            f"CANDIDATE ANSWER:\n{_answer_text(output)}"
        )
        return self.score("correctness", prompt)

    def faithfulness(self, *, input, output, **kwargs) -> Evaluation:
        contexts = _contexts(output)
        context_block = "\n\n".join(contexts) if contexts else "(no retrieved context)"
        prompt = (
            "Score whether the candidate answer is supported by the retrieved context. "
            "A statement that the information was not found is faithful when the context does not contain the fact. "
            "Claims that are absent from the context are unfaithful. World knowledge does not count as support. "
            "Use 1 when the answer stays within the context, and 0 when it adds unsupported facts.\n\n"
            f"QUESTION:\n{_question_text(input)}\n\n"
            f"RETRIEVED CONTEXT:\n{context_block}\n\n"
            f"CANDIDATE ANSWER:\n{_answer_text(output)}"
        )
        return self.score("faithfulness", prompt)

    def relevance(self, *, input, output, **kwargs) -> Evaluation:
        prompt = (
            "Score whether the candidate answer addresses the question. "
            "Use 1 when it answers the question, 0 when it is off-topic or empty, "
            "and a value between them when it is only partly on topic.\n\n"
            f"QUESTION:\n{_question_text(input)}\n\n"
            f"CANDIDATE ANSWER:\n{_answer_text(output)}"
        )
        return self.score("relevance", prompt)


def mean_score(*, item_results, **kwargs) -> list[Evaluation]:
    """Average each item-level judge score across the experiment run."""
    buckets: dict[str, list[float]] = {
        "correctness": [],
        "faithfulness": [],
        "relevance": [],
    }
    for result in item_results:
        evaluations = result["evaluations"] if isinstance(result, dict) else result.evaluations
        for evaluation in evaluations:
            if isinstance(evaluation, dict):
                name, value = evaluation.get("name"), evaluation.get("value")
            else:
                name, value = evaluation.name, evaluation.value
            if name in buckets and isinstance(value, (int, float)):
                buckets[name].append(float(value))
    summaries = []
    for name, values in buckets.items():
        average = sum(values) / len(values) if values else 0.0
        summaries.append(
            Evaluation(
                name="mean_" + name,
                value=average,
                comment=f"Average {name} across {len(values)} items.",
            )
        )
    return summaries

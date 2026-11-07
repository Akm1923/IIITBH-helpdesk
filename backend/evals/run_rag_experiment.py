"""Upload the Q&A workbook to Langfuse and score the live RAG pipeline.

Run from the backend directory:

    py -3 evals/run_rag_experiment.py --limit 5
    py -3 evals/run_rag_experiment.py --sync-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.application.evaluation.llm_judge import LlmJudge, mean_score
from app.application.evaluation.qa_dataset import DATASET_NAME, load_qa_pairs
from app.application.query.query_service import QueryService
from app.core.config import get_settings
from app.infrastructure.answer_generation.factory import create_answer_generator
from app.infrastructure.embeddings.factory import create_embedding_client
from app.infrastructure.langfuse_client import flush_langfuse, get_langfuse
from app.infrastructure.vectorstores.qdrant_store import QdrantVectorStore
from app.infrastructure.web_search.ddgs_search import DDGSWebSearch


def sync_dataset(client, pairs) -> None:
    try:
        client.create_dataset(
            name=DATASET_NAME,
            description="Golden IIIT Bhagalpur questions and reference answers for RAG evaluation.",
        )
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" not in message and "409" not in message:
            raise
    for pair in pairs:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=pair.dataset_item_id,
            input={"question": pair.question},
            expected_output=pair.answer,
            metadata={"pair_id": pair.pair_id},
        )


def build_service(settings):
    if not settings.local_embedding_base_url or not settings.qdrant_url:
        raise SystemExit(
            "Set LOCAL_EMBEDDING_BASE_URL and QDRANT_URL before running the RAG evaluation."
        )
    embedding_client = create_embedding_client(settings)
    vector_store = QdrantVectorStore.from_settings(settings)
    try:
        answer_generator = create_answer_generator(settings)
    except ValueError:
        vector_store.client.close()
        embedding_client.close()
        raise
    if answer_generator is None:
        vector_store.client.close()
        embedding_client.close()
        raise SystemExit(
            "Answer generation is not configured. Set GROQ_API_KEY before scoring answers."
        )
    service = QueryService(
        embedding_client=embedding_client,
        vector_store=vector_store,
        answer_generator=answer_generator,
        expected_dimension=settings.embedding_dimension,
        web_search=DDGSWebSearch(settings.web_search_max_results)
        if settings.web_search_enabled
        else None,
        web_search_enabled=settings.web_search_enabled,
    )
    return service, embedding_client, vector_store, answer_generator


def rag_output(service: QueryService, question: str) -> dict:
    result = service.query(question)
    return {
        "answer": result.answer,
        "status": result.status,
        "insufficient_evidence": result.insufficient_evidence,
        "contexts": [source.excerpt for source in result.sources],
        "sources": [
            {
                "evidence_id": source.evidence_id,
                "source_file": source.source_file,
                "score": source.score,
                "used_in_answer": source.used_in_answer,
            }
            for source in result.sources
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the IIITBH RAG pipeline in Langfuse.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N dataset items.")
    parser.add_argument("--sync-only", action="store_true", help="Upload the workbook and do not call the model.")
    parser.add_argument("--run-name", default=None, help="Langfuse experiment run name.")
    parser.add_argument("--max-concurrency", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1.")

    pairs = load_qa_pairs()
    client = get_langfuse()
    sync_dataset(client, pairs)
    print(f"Synced {len(pairs)} items to dataset {DATASET_NAME}.")
    if args.sync_only:
        flush_langfuse()
        return

    settings = get_settings()
    service, embedding_client, vector_store, answer_generator = build_service(settings)
    judge = LlmJudge.from_generator(answer_generator)

    def task(*, item, **kwargs):
        raw_input = item.input
        question = raw_input["question"] if isinstance(raw_input, dict) else str(raw_input)
        return rag_output(service, question)

    try:
        dataset = client.get_dataset(DATASET_NAME)
        if args.limit is not None:
            dataset.items = list(dataset.items)[: args.limit]
        result = dataset.run_experiment(
            name="IIITBH RAG evaluation",
            run_name=args.run_name,
            description="Retrieval plus answer generation scored by an LLM judge.",
            task=task,
            evaluators=judge.evaluators(),
            run_evaluators=[mean_score],
            max_concurrency=args.max_concurrency,
            metadata={"dataset": DATASET_NAME, "items": str(len(dataset.items))},
        )
        print(result.format(include_item_results=True))
        if getattr(result, "experiment_url", None):
            print(result.experiment_url)
    finally:
        flush_langfuse()
        vector_store.client.close()
        embedding_client.close()
        answer_generator.http.close()


if __name__ == "__main__":
    main()

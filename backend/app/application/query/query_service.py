import re
from datetime import datetime
from uuid import uuid4

from langfuse import propagate_attributes

from app.application.query.guardrails import validate_question
from app.domain.retrieval.models import EvidenceSource, GeneratedAnswer, QueryResult
from app.infrastructure.langfuse_client import observation, tracing_enabled
from app.ports.answer_generator import AnswerGenerator
from app.ports.embedding_client import EmbeddingClient
from app.ports.vector_store import VectorStore


class InvalidCitationError(RuntimeError):
    pass


CURRENT_PATTERNS = (
    r"\b(current|currently|present|latest|recent|recently|today|now|this year|as of|up to date|updated)\b",
    r"\b(who is|who's)\s+(the\s+)?(director|registrar|dean|hod|head of department|chairperson|chairman)\b",
    r"\b(current|present|new)\s+(director|registrar|dean|hod|head of department|chairperson|chairman)\b",
    r"\b(name of (the )?(director|registrar|dean|hod))\b",
    r"\b(upcoming|next)\s+(event|exam|holiday|deadline|notice|convocation|admission|schedule)\b",
    r"\b(latest|new)\s+(notice|announcement|circular|notification|tender|recruitment)\b",
)


def needs_live_search(question: str) -> bool:
    return any(re.search(pattern, question, re.IGNORECASE) for pattern in CURRENT_PATTERNS)


def _trace_sources(sources: list[EvidenceSource]) -> list[dict]:
    return [
        {
            "evidence_id": source.evidence_id,
            "source_type": source.source_type,
            "source_file": source.source_file,
            "page_number": source.page_number,
            "score": source.score,
            "used_in_answer": source.used_in_answer,
            "excerpt": source.excerpt[:800],
        }
        for source in sources
    ]


class QueryService:
    """Hybrid institutional RAG + live web retrieval with query and answer guardrails."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        answer_generator: AnswerGenerator | None = None,
        embedding_batch_size: int = 16,
        expected_dimension: int = 1024,
        web_search=None,
        web_search_enabled: bool = True,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.answer_generator = answer_generator
        self.embedding_batch_size = embedding_batch_size
        self.expected_dimension = expected_dimension
        self.web_search = web_search
        self.web_search_enabled = web_search_enabled

    def query(
        self,
        question: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> QueryResult:
        attributes = (
            propagate_attributes(trace_name="iiitbh-rag-query", tags=["rag", "iiitbh"])
            if tracing_enabled()
            else None
        )
        if attributes is None:
            return self._traced_query(question, top_k, score_threshold)
        with attributes:
            return self._traced_query(question, top_k, score_threshold)

    def _traced_query(
        self,
        question: str,
        top_k: int,
        score_threshold: float | None,
    ) -> QueryResult:
        with observation(
            as_type="span",
            name="rag-query",
            input={
                "question": question,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
        ) as current:
            result = self._query(
                question=question,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            current.update(
                output={
                    "query_id": result.query_id,
                    "status": result.status,
                    "answer": result.answer,
                    "insufficient_evidence": result.insufficient_evidence,
                    "sources": _trace_sources(result.sources),
                }
            )
            return result

    def _query(
        self, question: str, top_k: int = 5, score_threshold: float | None = None
    ) -> QueryResult:
        normalized_question = validate_question(question)
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20.")

        retrieval_query = normalized_question
        if self.answer_generator is not None and hasattr(
            self.answer_generator, "reformulate_query"
        ):
            retrieval_query = (
                self.answer_generator.reformulate_query(normalized_question) or normalized_question
            )

        embedding_kwargs: dict = {
            "as_type": "embedding",
            "name": "embed-query",
            "input": {"text": retrieval_query},
        }
        embedding_model = getattr(self.embedding_client, "model", None)
        if isinstance(embedding_model, str) and embedding_model.strip():
            embedding_kwargs["model"] = embedding_model
        with observation(**embedding_kwargs) as embed_obs:
            query_vector = self.embedding_client.embed_query(retrieval_query)
            embed_obs.update(output={"dimension": len(query_vector)})
        if len(query_vector) != self.expected_dimension:
            raise ValueError(
                f"Query embedding dimension {len(query_vector)} does not match configured embedding dimension {self.expected_dimension}."
            )
        with observation(
            as_type="retriever",
            name="institutional-retrieval",
            input={
                "query": retrieval_query,
                "top_k": top_k,
                "score_threshold": score_threshold,
            },
        ) as retrieval_obs:
            retrieved = self.vector_store.search(
                query_vector=query_vector, top_k=top_k, score_threshold=score_threshold
            )
            retrieval_obs.update(
                output={
                    "count": len(retrieved),
                    "hits": [
                        {
                            "source_file": item.source_file,
                            "document_id": item.document_id,
                            "page_number": item.page_number,
                            "score": item.score,
                            "excerpt": item.text[:800],
                        }
                        for item in retrieved
                    ],
                }
            )
        query_id = str(uuid4())

        sources = [
            EvidenceSource(
                evidence_id=f"S{index}",
                chunk_id=item.point_id,
                source_file=item.source_file,
                document_id=item.document_id,
                page_number=item.page_number,
                excerpt=item.text,
                score=item.score,
                extraction_method=item.extraction_method,
                metadata=item.metadata,
                source_type="institutional",
            )
            for index, item in enumerate(retrieved, start=1)
        ]

        web_error = False
        web_search_attempted = (
            self.web_search_enabled
            and self.web_search is not None
            and needs_live_search(normalized_question)
        )
        if web_search_attempted:
            live_query = retrieval_query + " " + datetime.now().astimezone().date().isoformat()
            try:
                with observation(
                    as_type="tool",
                    name="web-search",
                    input={"query": live_query},
                ) as web_obs:
                    web_results = self.web_search.search(live_query)
                    web_obs.update(
                        output={
                            "count": len(web_results),
                            "results": [
                                {
                                    "title": item.get("title"),
                                    "url": item.get("url"),
                                    "snippet": str(item.get("snippet", ""))[:800],
                                }
                                for item in web_results
                            ],
                        }
                    )
                if not web_results:
                    web_error = True
                for index, item in enumerate(web_results, start=1):
                    sources.append(
                        EvidenceSource(
                            evidence_id=f"W{index}",
                            chunk_id="web:" + item["url"],
                            source_file=item["title"],
                            excerpt=item["snippet"],
                            score=1.0,
                            source_url=item["url"],
                            source_type="web",
                            metadata={"domain": item["url"].split("/")[2]},
                        )
                    )
            except Exception:
                # Keep institutional RAG available if a search provider is temporarily down.
                web_error = True

        if not sources:
            note = " Live web search was unavailable." if web_error else ""
            return QueryResult(
                query_id=query_id,
                status="no_evidence",
                answer="I couldn't find relevant evidence in the indexed IIIT Bhagalpur documents or live search."
                + note,
                insufficient_evidence=True,
                sources=[],
            )

        if self.answer_generator is None:
            return QueryResult(
                query_id=query_id,
                status="evidence_only",
                answer=None,
                insufficient_evidence=False,
                sources=sources,
            )

        answer_question = normalized_question
        if web_error:
            answer_question += "\n\nCurrent-information safeguard: live web search returned no usable results or could not be reached. Do not state an office-holder, current notice, deadline, schedule or other time-sensitive fact as verified current. Explicitly say live verification was unavailable and distinguish any dated institute-document evidence."
        generated: GeneratedAnswer = self.answer_generator.generate(answer_question, sources)
        valid_ids = {source.evidence_id for source in sources}
        invalid_ids = set(generated.citations) - valid_ids
        if invalid_ids:
            raise InvalidCitationError(
                "Answer model referenced evidence IDs that were not retrieved: "
                + ", ".join(sorted(invalid_ids))
            )
        if not generated.answer.strip() or not generated.citations:
            return QueryResult(
                query_id=query_id,
                status="insufficient_evidence",
                answer="The retrieved evidence does not provide enough verified information to answer confidently. Please check the cited official source or contact the institute.",
                insufficient_evidence=True,
                sources=sources,
            )

        cited_ids = set(generated.citations)
        cited_sources = [
            source.model_copy(update={"used_in_answer": source.evidence_id in cited_ids})
            for source in sources
        ]
        return QueryResult(
            query_id=query_id,
            status="answered",
            answer=generated.answer,
            insufficient_evidence=False,
            sources=cited_sources,
        )

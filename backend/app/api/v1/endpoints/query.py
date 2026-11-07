from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_query_service
from app.application.query.query_service import InvalidCitationError, QueryService
from app.infrastructure.answer_generation.openai_compatible_generator import AnswerGenerationError
from app.infrastructure.embeddings.huggingface_bge_m3 import EmbeddingServiceError
from app.infrastructure.langfuse_client import flush_langfuse
from app.infrastructure.vectorstores.qdrant_store import VectorStoreError
from app.schemas.query import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Retrieve IIIT Bhagalpur evidence and answer a question",
)
def query_documents(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    try:
        result = service.query(
            question=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
        )
    except InvalidCitationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The answer provider returned an invalid citation reference.",
        ) from exc
    except (AnswerGenerationError, EmbeddingServiceError, VectorStoreError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="A retrieval dependency failed. Check backend logs.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        flush_langfuse()
    return QueryResponse.model_validate(result.model_dump())

import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel

from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def get_rag_service() -> RAGService:
    """
    Dependency injector for RAGService.
    Allows easier testing and future configuration.
    """
    return RAGService()


class Message(BaseModel):
    """Represents a single chat message."""
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    """Request payload for chat endpoint."""
    question: str
    document_filter: Optional[str] = None
    history: Optional[List[Message]] = None  # Avoid mutable default


class ChatResponse(BaseModel):
    """Response returned to the client."""
    answer: str
    sources: List[str]


@router.post("/ask", response_model=ChatResponse)
async def ask_question(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Handles user question against stored documents using RAG.
    Supports optional document filtering and conversation context.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    try:
        history_dicts = [
            {"role": msg.role, "content": msg.content}
            for msg in request.history or []
        ]

        result = rag_service.query(
            user_question=request.question,
            document_filter=request.document_filter,
            history=history_dicts
        )

        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except RuntimeError as e:
        logger.error(f"Service runtime error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )

    except Exception:
        logger.exception("Unhandled error processing chat request.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error."
        )
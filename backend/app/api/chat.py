from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import logging

from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

rag_service = RAGService()

class ChatRequest(BaseModel):
    question: str
    document_filter : Optional[str] = None  # Optional filter for document source, e.g., filename

class ChatResponse(BaseModel):
    answer: str
    sources : list[str]

@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    """
    Submits a question to the AI, enforcing answers based strictly on uploaded documents.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty."
        )

    try:
        result = rag_service.query(request.question, request.document_filter)
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
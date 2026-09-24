from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import logging

from app.services.pdf_service import PDFService
from app.services.text_service import TextService
from app.services.vector_service import VectorStoreService

router = APIRouter(prefix="/documents", tags=["documents"])

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 5 * 1024 * 1024

# Initialize vector service once to maintain database connection pooling
vector_service = VectorStoreService()

class UploadResponse(BaseModel):
    filename: str
    character_count: int
    total_chunks: int
    first_chunk_preview: str

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Processes a PDF file, extracts text, generates chunks, 
    and persists vector embeddings to ChromaDB.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are supported."
        )

    try:
        file_bytes = await file.read()

        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File size exceeds the 5MB limit."
            )

        if not file_bytes.startswith(b"%PDF"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File signature validation failed."
            )

        extracted_text = PDFService.extract_text(file_bytes)
        
        chunks = TextService.split_text(extracted_text)
        
        if chunks:
            vector_service.store_chunks(chunks, file.filename)

        preview = chunks[0][:300] + "..." if chunks and len(chunks[0]) > 300 else (chunks[0] if chunks else "")

        return UploadResponse(
            filename=file.filename,
            character_count=len(extracted_text),
            total_chunks=len(chunks),
            first_chunk_preview=preview
        )

    except ValueError as e:
        logger.warning(f"Validation error processing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )
    except RuntimeError as e:
        logger.error(f"Database error processing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector database service is currently unavailable."
        )
    except Exception:
        logger.exception(f"Unhandled exception processing {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
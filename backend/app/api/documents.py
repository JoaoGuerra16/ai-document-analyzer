from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
import logging

from app.services.pdf_service import PDFService

router = APIRouter(prefix="/documents", tags=["documents"])

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


class UploadResponse(BaseModel):
    filename: str
    character_count: int
    preview: str


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file, validate its type and size,
    and extract its text content.
    """

    # Validate content type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only PDF files are supported."
        )

    try:
        file_bytes = await file.read()

        # Validate file size
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large. Maximum allowed size is 5MB."
            )

        # Optional: basic PDF signature validation
        if not file_bytes.startswith(b"%PDF"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF file."
            )

        # Extract text using service layer
        extracted_text = PDFService.extract_text(file_bytes)

        return UploadResponse(
            filename=file.filename,
            character_count=len(extracted_text),
            preview=(
                extracted_text[:300] + "..."
                if len(extracted_text) > 300
                else extracted_text
            )
        )

    except ValueError as e:
        logger.warning(f"Validation error processing file {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

    except Exception as e:
        logger.exception(f"Unexpected error processing file {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
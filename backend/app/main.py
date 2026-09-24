import logging
from fastapi import FastAPI
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

app = FastAPI(
    title="AI Document Analyzer API",
    description="RAG System backend utilizing ChromaDB and Gemini",
    version="1.0.0"
)

app.include_router(documents_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


@app.get("/api/health")
def check_health():
    return {
        "status": "operational",
        "database": "ChromaDB pending initialization",
        "ai_service": "offline"
    }
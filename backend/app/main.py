from fastapi import FastAPI
from app.api.documents import router as documents_router

app = FastAPI(
    title="AI Document Analyzer API",
    description="RAG System backend utilizing ChromaDB and Gemini",
    version="1.0.0"
)

# Register endpoint routers
app.include_router(documents_router, prefix="/api")

@app.get("/api/health")
def check_health():
    return {
        "status": "operational",
        "database": "ChromaDB pending initialization",
        "ai_service": "offline"
    }
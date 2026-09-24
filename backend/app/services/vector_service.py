import logging
from typing import List
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    Handles vector database operations using local ChromaDB and Gemini embeddings.
    """

    def __init__(self):
        self.persist_directory = "./chroma_data"
        self.collection_name = "document_embeddings"
        
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing from environment variables.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=settings.GEMINI_API_KEY
        )

    def store_chunks(self, chunks: List[str], source_filename: str) -> None:
        """
        Generates embeddings for text chunks and persists them to the local vector store.
        """
        if not chunks:
            return

        try:
            # Attach source metadata to each chunk for filtering and citation purposes
            metadatas = [{"source": source_filename} for _ in chunks]
            
            # Initialize vector store and persist data
            Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                metadatas=metadatas,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )
            
            logger.info(f"Successfully vectorized and stored {len(chunks)} chunks for {source_filename}.")
            
        except Exception as e:
            logger.error(f"Vector storage process failed for {source_filename}")
            raise RuntimeError("Failed to store document vectors in the database.")
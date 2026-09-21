import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextService:
    """
    Utilities for preparing text before vectorization (chunking).
    """

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200

    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> List[str]:
        """
        Splits a large text into smaller, meaningful chunks using LangChain.
        
        Args:
            text (str): The raw text to be split.
            chunk_size (int): Maximum number of characters per chunk.
            chunk_overlap (int): Number of characters to overlap between chunks.
            
        Returns:
            List[str]: A list of text chunks ready for vectorization.
        """

        if not text or not text.strip():
            logger.warning("Attempted to split empty text.")
            return []

        try:
            # Preserve semantic structure when splitting (paragraphs > sentences > words)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False
            )

            chunks = splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks.")

            return chunks

        except Exception:
            logger.exception("Failed to split text")
            raise ValueError("Error during text chunking")
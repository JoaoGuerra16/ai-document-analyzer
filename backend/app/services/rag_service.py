import logging
from typing import List, Dict, Optional

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """
    Handles retrieval-augmented generation (RAG) operations:
    - retrieves relevant document chunks
    - injects context into prompt
    - generates answer using LLM
    """

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing from environment variables.")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=settings.GEMINI_API_KEY
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            google_api_key=settings.GEMINI_API_KEY
        )

        # Persistent vector store
        self.vector_store = Chroma(
            persist_directory="./chroma_data",
            embedding_function=self.embeddings,
            collection_name="document_embeddings"
        )

        system_prompt = (
            "You are a professional assistant.\n"
            "Use ONLY the provided context to answer the question.\n"
            "If the answer is not in the context, say it clearly.\n\n"
            "Conversation:\n{chat_history}\n\n"
            "Context:\n{context}\n\n"
            "Answer concisely and professionally."
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def query(
        self,
        user_question: str,
        document_filter: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> dict:
        """
        Executes RAG pipeline:
        1. Retrieve relevant chunks
        2. Inject context + history
        3. Generate answer
        """
        try:
            # Configure retriever dynamically
            search_kwargs = {"k": 3}
            if document_filter:
                search_kwargs["filter"] = {"source": document_filter}

            retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)

            # Retrieve documents
            docs = retriever.invoke(user_question)

            if not docs:
                return {
                    "answer": "No relevant information found in the documents.",
                    "sources": []
                }

            # Build context string
            context_text = "\n\n".join(doc.page_content for doc in docs)

            # Format recent conversation history (last 4 messages)
            if history:
                recent = history[-4:]
                chat_history = "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in recent
                )
            else:
                chat_history = "No previous conversation."

            # Generate answer
            answer = self.chain.invoke({
                "context": context_text,
                "chat_history": chat_history,
                "input": user_question
            })

            # Extract unique sources
            sources = list({
                doc.metadata.get("source", "Unknown")
                for doc in docs
            })

            return {
                "answer": answer,
                "sources": sources
            }

        except Exception:
            logger.exception("RAG query execution failed")
            raise RuntimeError("Failed to generate an answer from the document database.")
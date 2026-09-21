import logging
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    """
    Service responsible for handling retrieval and generation (QA) operations
    using modern LangChain Core architecture.
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
        
        self.vector_store = Chroma(
            persist_directory="./chroma_data",
            embedding_function=self.embeddings,
            collection_name="document_embeddings"
        )
        
        system_prompt = (
            "You are a professional corporate assistant. Use the following retrieved context to answer the user's question.\n"
            "If you do not know the answer based strictly on the context provided, state clearly that the information is not available in the documents. Do not hallucinate or make up information.\n\n"
            "Context:\n{context}"
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        self.generation_chain = self.prompt | self.llm | StrOutputParser()

    def query(self, user_question: str, document_filter: str = None) -> dict:
        """
        Retrieves relevant documents and generates an answer, optionally applying metadata filters.
        """
        try:
            search_kwargs = {"k": 3}
            if document_filter:
                search_kwargs["filter"] = {"source": document_filter}
                logger.info(f"Applying metadata filter for source: {document_filter}")

            dynamic_retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)

            docs = dynamic_retriever.invoke(user_question)
        
            if not docs:
                return {
                    "answer": "No relevant information found in the specified document.",
                    "sources": []
                }

            context_text = "\n\n".join(doc.page_content for doc in docs)
            
            answer = self.generation_chain.invoke({
                "context": context_text,
                "input": user_question
            })
            
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))
            
            return {
                "answer": answer,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"RAG query execution failed: {e}")
            raise RuntimeError("Failed to generate an answer from the document database.")
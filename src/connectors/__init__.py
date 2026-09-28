from src.connectors.base import RAGConnector, RAGResponse, RetrievedChunk
from src.connectors.http import HTTPRAGConnector

__all__ = ["HTTPRAGConnector", "LangChainRAGConnector", "QdrantRAGConnector", "RAGConnector", "RAGResponse",
           "RetrievedChunk"]


def __getattr__(name: str):
    # qdrant_client and langchain_core are slow to import; only load them when asked for
    if name == "QdrantRAGConnector":
        from src.connectors.qdrant import QdrantRAGConnector
        return QdrantRAGConnector
    if name == "LangChainRAGConnector":
        from src.connectors.langchain import LangChainRAGConnector
        return LangChainRAGConnector
    raise AttributeError(name)

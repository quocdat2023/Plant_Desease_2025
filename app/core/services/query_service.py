from typing import List
from ..models.document import Document
from ..repositories.index_repository import IndexRepository
from ...handlers.faiss_handler import FaissHandler
from ...handlers.bm25_handler import BM25Handler
from ...handlers.hybrid_handler import HybridHandler

class QueryService:
    def __init__(self, index_repo: IndexRepository):
        self.index_repo = index_repo
    
    def create_query_handler(self, strategy: str) -> HybridHandler | FaissHandler | BM25Handler:
        if strategy == "hybrid":
            return HybridHandler(self.index_repo)
        elif strategy == "faiss":
            return FaissHandler(self.index_repo)
        elif strategy == "bm25":
            return BM25Handler(self.index_repo)
        raise ValueError(f"Unknown query strategy: {strategy}")
    
    def query(self, query: str, k: int = 5, doc_type: str = "banan", strategy: str = "hybrid") -> List[Document]:
        handler = self.create_query_handler(strategy)
        return handler.query(query, k, doc_type)
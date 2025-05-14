from .query_handler import QueryHandler
from .faiss_handler import FaissHandler
from .bm25_handler import BM25Handler
from ..core.models.document import Document
from ..core.repositories.index_repository import IndexRepository
from typing import List

class HybridHandler(QueryHandler):
    def __init__(self, index_repo: IndexRepository, faiss_weight: float = 0.7, bm25_weight: float = 0.7):
        self.faiss_handler = FaissHandler(index_repo)
        self.bm25_handler = BM25Handler(index_repo)
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight
    
    def query(self, query: str, k: int, doc_type: str) -> List[Document]:
        faiss_results = self.faiss_handler.query(query, k, doc_type)
        bm25_results = self.bm25_handler.query(query, k, doc_type)
        
        faiss_scores = {res.id: 1.0 - res.distance for res in faiss_results if res.distance is not None}
        bm25_scores = {res.id: res.score for res in bm25_results if res.score is not None}
        
        combined_results = {}
        all_ids = set(faiss_scores.keys()).union(bm25_scores.keys())
        max_bm25_score = max(bm25_scores.values(), default=1.0) or 1.0
        
        for id in all_ids:
            faiss_score = faiss_scores.get(id, 0.0)
            bm25_score = bm25_scores.get(id, 0.0) / max_bm25_score
            combined_score = self.faiss_weight * faiss_score + self.bm25_weight * bm25_score
            combined_results[id] = combined_score
        
        sorted_ids = sorted(combined_results, key=combined_results.get, reverse=True)[:k]
        results = []
        for id in sorted_ids:
            for res in faiss_results + bm25_results:
                if res.id == id:
                    results.append(res)
                    break
        
        return results
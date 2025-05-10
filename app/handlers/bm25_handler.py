from nltk.tokenize import word_tokenize
from .query_handler import QueryHandler
from ..core.models.document import Document
from ..core.repositories.index_repository import IndexRepository
import numpy as np
from typing import List

class BM25Handler(QueryHandler):
    def __init__(self, index_repo: IndexRepository):
        self.index_repo = index_repo
    
    def query(self, query: str, k: int, doc_type: str) -> List[Document]:
        bm25 = self.index_repo.get_bm25_index(doc_type)
        metadata = self.index_repo.get_metadata(doc_type)
        
        tokenized_query = word_tokenize(query.lower())
        scores = bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        results = []
        
        for idx in top_k_indices:
            if scores[idx] >= 0.0 and 0 <= idx < len(metadata["ids"]):
                meta = metadata["metadata"][idx]
                if doc_type is None or meta.get("type") == doc_type:
                    results.append(Document(
                        id=metadata["ids"][idx],
                        text=metadata["texts"][idx],
                        metadata=meta,
                        score=float(scores[idx]),
                        case_summary=meta.get("case_summary"),
                        legal_issues=meta.get("legal_issues"),
                        court_reasoning=meta.get("court_reasoning"),
                        decision=meta.get("decision"),
                        relevant_laws=meta.get("relevant_laws")
                    ))
        
        return results[:k]
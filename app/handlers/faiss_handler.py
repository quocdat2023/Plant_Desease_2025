import numpy as np
from .query_handler import QueryHandler
from ..core.models.document import Document
from ..core.repositories.index_repository import IndexRepository
from typing import List

class FaissHandler(QueryHandler):
    def __init__(self, index_repo: IndexRepository):
        self.index_repo = index_repo
    
    def query(self, query: str, k: int, doc_type: str) -> List[Document]:
        embeddings = self.index_repo.get_embeddings()
        faiss_index = self.index_repo.get_faiss_index(doc_type)
        metadata = self.index_repo.get_metadata(doc_type)
        
        query_emb = embeddings.encode([query], convert_to_numpy=True)
        distances, indices = faiss_index.search(query_emb, k)
        results = []
        
        for dist, i in zip(distances[0], indices[0]):
            if 0 <= i < len(metadata["ids"]):
                meta = metadata["metadata"][i]
                if doc_type is None or meta.get("type") == doc_type:
                    results.append(Document(
                        id=metadata["ids"][i],
                        text=metadata["texts"][i],
                        metadata=meta,
                        distance=float(dist),
                        case_summary=meta.get("case_summary"),
                        legal_issues=meta.get("legal_issues"),
                        court_reasoning=meta.get("court_reasoning"),
                        decision=meta.get("decision"),
                        relevant_laws=meta.get("relevant_laws")
                    ))
        
        return results[:k]
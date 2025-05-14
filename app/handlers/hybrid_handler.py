from .query_handler import QueryHandler
from .faiss_handler import FaissHandler
from .bm25_handler import BM25Handler
from ..core.models.document import Document
from ..core.repositories.index_repository import IndexRepository
from typing import List, Dict

class HybridHandler(QueryHandler):
    def __init__(self, index_repo: IndexRepository, faiss_weight: float = 0.6, bm25_weight: float = 0.4):
        self.faiss_handler = FaissHandler(index_repo)
        self.bm25_handler = BM25Handler(index_repo)
        self.faiss_weight = faiss_weight
        self.bm25_weight = bm25_weight

    def query(self, query: str, k: int, doc_type: str) -> List[Document]:
        # Lấy kết quả từ cả hai phương pháp
        faiss_results = self.faiss_handler.query(query, k * 2, doc_type)  # Lấy thêm để dự phòng
        bm25_results = self.bm25_handler.query(query, k * 2, doc_type)

        # Tạo dictionary ánh xạ ID -> Document
        document_map: Dict[str, Document] = {}
        for res in faiss_results + bm25_results:
            if res.id not in document_map:
                document_map[res.id] = res

        # Chuẩn hóa điểm Faiss về [0, 1]
        faiss_scores = {}
        for res in faiss_results:
            if res.distance is not None:
                similarity = 1.0 - res.distance  # Giả sử sử dụng cosine similarity
                normalized_score = (similarity + 1) / 2  # Đưa về khoảng [0, 1]
                faiss_scores[res.id] = normalized_score

        # Chuẩn hóa điểm BM25 và xử lý chia cho 0
        bm25_scores = {res.id: res.score for res in bm25_results if res.score is not None}
        max_bm25 = max(bm25_scores.values(), default=0)
        max_bm25_score = max_bm25 if max_bm25 > 0 else 1.0

        # Kết hợp điểm số
        combined_scores: Dict[str, float] = {}
        all_ids = set(faiss_scores.keys()).union(bm25_scores.keys())
        
        for doc_id in all_ids:
            faiss_score = faiss_scores.get(doc_id, 0.0)
            bm25_score = (bm25_scores.get(doc_id, 0.0) / max_bm25_score) if max_bm25_score != 0 else 0.0
            combined_score = (self.faiss_weight * faiss_score) + (self.bm25_weight * bm25_score)
            combined_scores[doc_id] = combined_score

        # Sắp xếp và chọn top k
        sorted_ids = sorted(combined_scores, key=lambda x: combined_scores[x], reverse=True)[:k]
        
        # Tạo kết quả cuối cùng
        results = []
        for doc_id in sorted_ids:
            results.append(document_map[doc_id])
            
        return results
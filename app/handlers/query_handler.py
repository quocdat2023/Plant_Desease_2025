from abc import ABC, abstractmethod
from typing import List
from ..core.models.document import Document

class QueryHandler(ABC):
    @abstractmethod
    def query(self, query: str, k: int, doc_type: str) -> List[Document]:
        pass
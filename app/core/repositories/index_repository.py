import faiss
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from .metadata_repository import MetadataRepository
from ...config.settings import Config

logger = logging.getLogger(__name__)

class IndexRepository:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IndexRepository, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        Config().validate()  # Validate GEMINI_API_KEYS
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(Config.INDEX_PATH)
        logger.info(f"FAISS index loaded: {self.faiss_index.ntotal} documents")
        
        # Load metadata
        self.metadata_repo = MetadataRepository()
        self.metadata_dict = self.metadata_repo.load_metadata(Config.METADATA_PATH)
        self.summarized_metadata_dict = self.metadata_repo.load_metadata(Config.SUMMARIZED_METADATA_PATH)
        
        # Initialize BM25 indices
        self.bm25_banan = BM25Okapi([word_tokenize(text.lower()) for text in self.metadata_dict["texts"]])
        self.bm25_banan_sum = BM25Okapi([word_tokenize(text.lower()) for text in self.summarized_metadata_dict["texts"]])
        logger.info("BM25 indices initialized")
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_faiss_index(self, doc_type: str):
        return self.faiss_index if doc_type == "banan_sum" else self.faiss_index
    
    def get_bm25_index(self, doc_type: str):
        if doc_type == "banan":
            return self.bm25_banan
        elif doc_type == "banan_sum":
            return self.bm25_banan_sum
        return self.bm25_banan  # Default to banan if doc_type is unknown
    
    def get_metadata(self, doc_type: str):
        if doc_type == "banan":
            return self.metadata_dict
        elif doc_type == "banan_sum":
            return self.summarized_metadata_dict
        return self.metadata_dict  # Default to metadata_dict if doc_type is unknown
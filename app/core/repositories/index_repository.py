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
        self.embeddings = SentenceTransformer('hiieu/halong_embedding')
        
        # Load FAISS indices
        self.faiss_index = faiss.read_index(Config.INDEX_PATH)
        self.faiss_anle_index = faiss.read_index(Config.ANLE_INDEX_PATH)
        logger.info(f"FAISS indices loaded: {self.faiss_index.ntotal} (banan), {self.faiss_anle_index.ntotal} (anle)")
        
        # Load metadata
        self.metadata_repo = MetadataRepository()
        self.metadata_dict = self.metadata_repo.load_metadata(Config.METADATA_PATH)
        self.summarized_metadata_dict = self.metadata_repo.load_metadata(Config.SUMMARIZED_METADATA_PATH)
        self.anle_metadata_dict = self.metadata_repo.load_metadata(Config.ANLE_METADATA_PATH)
        
        # Initialize BM25 indices
        self.bm25_banan = BM25Okapi([word_tokenize(text.lower()) for text in self.metadata_dict["texts"]])
        self.bm25_banan_sum = BM25Okapi([word_tokenize(text.lower()) for text in self.summarized_metadata_dict["texts"]])
        self.bm25_anle = BM25Okapi([word_tokenize(text.lower()) for text in self.anle_metadata_dict["texts"]])
        logger.info("BM25 indices initialized")
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_faiss_index(self, doc_type: str):
        return self.faiss_anle_index if doc_type == "anle" else self.faiss_index
    
    def get_bm25_index(self, doc_type: str):
        if doc_type == "banan":
            return self.bm25_banan
        elif doc_type == "banan_sum":
            return self.bm25_banan_sum
        return self.bm25_anle
    
    def get_metadata(self, doc_type: str):
        if doc_type == "banan":
            return self.metadata_dict
        elif doc_type == "banan_sum":
            return self.summarized_metadata_dict
        return self.anle_metadata_dict
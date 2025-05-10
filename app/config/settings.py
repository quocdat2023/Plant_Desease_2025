import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "")
    INDEX_PATH = "source/faiss_index"
    METADATA_PATH = "source/faiss_metadata.pkl"
    SUMMARIZED_METADATA_PATH = "source/summarized_faiss_metadata.pkl"
    ANLE_INDEX_PATH = "source/faiss_index_anle.index"
    ANLE_METADATA_PATH = "source/metadata_anle.pkl"
    
    @staticmethod
    def validate():
        if not Config.GEMINI_API_KEYS:
            raise RuntimeError("Missing GEMINI_API_KEYS in environment!")
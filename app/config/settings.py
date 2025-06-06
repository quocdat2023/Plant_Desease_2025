import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS", "")
    INDEX_PATH = "source/index_plant.faiss"
    METADATA_PATH = "source/faiss_metadata_30_05.pkl"
    SUMMARIZED_METADATA_PATH = "source/summarized_faiss_metadata.pkl"

    def __init__(self):
        self.validate()  # Gọi validate khi khởi tạo

    @staticmethod
    def validate():
        if not Config.GEMINI_API_KEYS:
            raise RuntimeError("Missing GEMINI_API_KEYS in environment!")
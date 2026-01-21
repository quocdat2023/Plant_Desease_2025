import pickle
import joblib
import pandas as pd
from ultralytics import YOLO
from config.config import Config
import faiss
import logging
import os

class ModelStore:
    def __init__(self):
        self.yolo_model = None
        self.scaler_plant = None
        self.label_encoder_plant = None
        self.logistic_plant = None
        self.fertilizer_model = None
        self.label_encoders_fertilizer = {}
        self.faiss_index = None
        self.embeddings_data = None

    def load_models(self):
        print("Loading models...")
        # YOLO
        if os.path.exists(Config.YOLO_MODEL_PATH):
            self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)
            print("YOLO loaded.")
        else:
            print(f"Warning: YOLO model not found at {Config.YOLO_MODEL_PATH}")

        # Plant Recommendation Models
        try:
            if os.path.exists(Config.SCALER_PLANT):
                with open(Config.SCALER_PLANT, 'rb') as f:
                    self.scaler_plant = pickle.load(f)
            if os.path.exists(Config.LABEL_ENCODER_PLANT):
                with open(Config.LABEL_ENCODER_PLANT, 'rb') as f:
                    self.label_encoder_plant = pickle.load(f)
            if os.path.exists(Config.LOGISTIC_PLANT):
                with open(Config.LOGISTIC_PLANT, 'rb') as f:
                    self.logistic_plant = pickle.load(f)
            print("Plant recommendation models loaded.")
        except Exception as e:
            print(f"Error loading plant models: {e}")

        # Fertilizer Models
        try:
            if os.path.exists(Config.FERTILIZER_MODEL):
                self.fertilizer_model = joblib.load(Config.FERTILIZER_MODEL)
                
                # Load encoders
                for column in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
                     p = os.path.join(Config.MODELS_DIR, f'label_encoder_{column}.pkl')
                     if os.path.exists(p):
                        self.label_encoders_fertilizer[column] = joblib.load(p)
                print("Fertilizer models loaded.")
        except Exception as e:
            print(f"Error loading fertilizer models: {e}")

        # FAISS
        try:
             if os.path.exists(Config.FAISS_INDEX_PATH):
                 self.faiss_index = faiss.read_index(Config.FAISS_INDEX_PATH)
             if os.path.exists(Config.EMBEDDINGS_DATA_PATH):
                 with open(Config.EMBEDDINGS_DATA_PATH, 'rb') as f:
                     self.embeddings_data = pickle.load(f)
             print("FAISS loaded.")
        except Exception as e:
             print(f"Error loading FAISS: {e}")

# Global instance
model_store = ModelStore()

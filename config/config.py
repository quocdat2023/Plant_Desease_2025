import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key'
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Static & Templates
    STATIC_FOLDER = os.path.join(BASE_DIR, 'app', 'static')
    UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    # Data & Models
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
    CROPS_DIR = os.path.join(DATA_DIR, 'crops')
    
    # Specific Model Paths
    YOLO_MODEL_PATH = os.path.join(MODELS_DIR, 'trained_yolo11s_cls.pt')
    
    # Recommender Models
    MODELS_PLANT_DIR = MODELS_DIR
    SCALER_PLANT = os.path.join(MODELS_PLANT_DIR, 'scaler_plant.pkl')
    LABEL_ENCODER_PLANT = os.path.join(MODELS_PLANT_DIR, 'label_encoder_plant.pkl')
    LOGISTIC_PLANT = os.path.join(MODELS_PLANT_DIR, 'LogisticRegresion_plant.pkl')
    
    FERTILIZER_MODEL = os.path.join(MODELS_PLANT_DIR, 'multioutput_xgboost_fertilizer_model.pkl')
    
    # FAISS
    FAISS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, 'index_plant.faiss')
    EMBEDDINGS_DATA_PATH = os.path.join(EMBEDDINGS_DIR, 'index_plant.pkl')
    
    # JSON Data
    CROP_DATA_PATH = os.path.join(CROPS_DIR, 'crop_data.json')

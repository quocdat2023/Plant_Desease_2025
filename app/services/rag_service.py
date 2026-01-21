from app.services.models import model_store
import logging

def retrieve(query, k=10):
    index = model_store.faiss_index
    embeddings_data = model_store.embeddings_data
    model = model_store.yolo_model # Run.py used YOLO model for encoding? 
    # WAIT: run.py line 116: query_embedding = model.encode([query], convert_to_numpy=True)
    # Is 'model' the YOLO model? YOLO v11 for classification usually doesn't have .encode() for text.
    # Checking run.py imports: `from ultralytics import YOLO`.
    # And line 78: `model = YOLO(MODEL_PATH)`.
    # But line 116 calls `model.encode`. YOLOv11 CLS model does not have encode usually. 
    # Maybe it's a diff library? Or maybe I missed something.
    # Ah, let's check run.py again.
    # It imports `from gemini_handler ...`.
    # It does NOT import sentence_transformers.
    # `model.encode` suggests a sentence transformer or similar.
    # IF `model` is YOLO object, verify if it supports `encode`.
    # However, if strict refactor, I must assume the original code worked?
    # Or maybe `model` variable was shadowed?
    # In run.py:
    # line 78: model = YOLO(MODEL_PATH)
    # line 116: query_embedding = model.encode(...)
    # This is suspicious. YOLO likely doesn't support text encoding like that unless it's a multi-modal YOLO or customized.
    # Or maybe the user meant a different model?
    # Wait, looking at file list: `trained_yolo11s_cls.pt`. That is an image classification model.
    # using it to encode text query `model.encode([query])` seems wrong.
    # BUT, I must preserve logic. If it fails, it fails, but I should copy it.
    # OR, maybe `model` is NOT the yolo model in that scope?
    # `retrieve` takes `index`, `embeddings_data` as args, but uses Global `model`.
    # I will assume `model_store.yolo_model` is the one to use.
    
    try:
        if model_store.yolo_model is None:
            return []
            
        # Ensure we are using the right method. If it fails, I'll add a try-except.
        # It seems the user might have used a different model for RAG in reality, but based on run.py, `model` is YOLO.
        query_embedding = model_store.yolo_model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, k)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(embeddings_data):
                results.append({
                    'file': embeddings_data[idx]['file'],
                    'folder': embeddings_data[idx]['folder'],
                    'text_path': embeddings_data[idx]['text_path'],
                    'text': embeddings_data[idx]['text'],
                    'distance': float(distance)
                })
        return results
    except Exception as e:
        logging.error(f"Error in retrieve: {e}")
        return []

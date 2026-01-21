from flask import Flask
from config.config import Config
from app.services.models import model_store

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Register Blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)
    
    # Load Models (Standard practice to load only once)
    # Note: In production with Gunicorn workers, this runs per worker.
    with app.app_context():
        try:
            model_store.load_models()
        except Exception as e:
            print(f"Failed to load models at startup: {e}")
        
    return app

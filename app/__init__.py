import logging
from flask import Flask
from flask_cors import CORS
from .routes import api_bp
from .routes.home import home_bp

def create_app():
    # Cấu hình logging
    logging.basicConfig(
        level=logging.INFO,  # Cho phép INFO trở lên
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),  # Ghi ra console
            logging.FileHandler('app.log')  # Ghi vào tệp app.log
        ]
    )
    
    app = Flask(__name__)
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(home_bp)
    
    return app

__all__ = ["create_app"]
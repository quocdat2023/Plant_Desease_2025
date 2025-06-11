# import logging
# from flask import Flask
# from flask_cors import CORS
# from .routes import api_bp
# from .routes.home import home_bp

# def create_app():
#     # Cấu hình logging
#     logging.basicConfig(
#         level=logging.INFO,  # Cho phép INFO trở lên
#         format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
#         handlers=[
#             logging.StreamHandler(),  # Ghi ra console
#             logging.FileHandler('app.log')  # Ghi vào tệp app.log
#         ]
#     )
    
#     app = Flask(__name__)
#     CORS(app)
    
#     # Register blueprints
#     app.register_blueprint(api_bp, url_prefix='/api')
#     app.register_blueprint(home_bp)
    
#     return app

# __all__ = ["create_app"]

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
    
    # Hardcode a secure secret key (48-character random string)
    app.secret_key = 'b9c8a7d6e5f43210fedcba9876543210abcdef9876543210'
    
    # Debug: Confirm secret key is set
    logger = logging.getLogger(__name__)
    logger.info(f"Secret Key Set: {app.secret_key}")
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(home_bp)
    
    return app

__all__ = ["create_app"]
from flask import Flask
from app.controllers.detection_controller import detection_bp
import os

def create_app():
    app = Flask(__name__,
                template_folder='views/templates',
                static_folder='views/static')    
    
    # Asegurarse de que existe el directorio static
    os.makedirs(os.path.join(app.root_path, 'views/static'), exist_ok=True)
    
    # Registrar blueprints
    app.register_blueprint(detection_bp)
    
    return app
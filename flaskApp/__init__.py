from flask import Flask
from flask_cors import CORS
from flaskApp.config import Config
from flaskApp.views import webhook_bp, web_bp
def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Enable CORS
    CORS(app)
    # Register blueprints
    app.register_blueprint(webhook_bp)
    app.register_blueprint(web_bp)
        
    return app


app = create_app()
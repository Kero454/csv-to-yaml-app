import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_cors import CORS
from config import Config

db = SQLAlchemy()
migrate = Migrate()
login = LoginManager()
login.login_view = 'auth.login'
login.login_message = 'Please log in to access this page.'

def create_app(config_class=Config):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_class)
    
    # Enable CORS for local dev and server deployment
    CORS(app, supports_credentials=True, origins=[
        'http://localhost:5000',
        'http://127.0.0.1:5000',
        'http://localhost:*',
        'http://127.0.0.1:*',
        'http://10.1.65.251:*',
        'http://smartgridwks6:*',
    ])

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    db.init_app(app)
    migrate.init_app(app, db)
    login.init_app(app)

    # Register blueprints
    from app.routes import web as routes_blueprint
    app.register_blueprint(routes_blueprint)

    from app.auth import auth_bp as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')

    from . import models

    return app


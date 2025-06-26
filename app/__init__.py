"""
CSV to YAML Converter Application Package

This package contains the core functionality for the CSV to YAML Converter.
"""

import os
from flask import Flask
from werkzeug.middleware.proxy_fix import ProxyFix

def create_app(config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load default configuration
    app.config.from_object('config.DefaultConfig')
    
    # Load environment-specific configuration
    env = os.environ.get('FLASK_ENV', 'development')
    if env == 'production':
        app.config.from_object('config.ProductionConfig')
    elif env == 'testing':
        app.config.from_object('config.TestingConfig')
    
    # Load any additional configuration passed in
    if config is not None:
        app.config.update(config)
    
    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Configure logging
    from app.utils.logging import configure_logging
    configure_logging(app)
    
    # Register blueprints and extensions
    register_blueprints(app)
    register_extensions(app)
    
    # Add middleware
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    return app

def register_blueprints(app):
    """Register Flask blueprints."""
    # Import blueprints here to avoid circular imports
    from app.api.v1 import api_bp as api_v1_bp
    
    # Register API v1 blueprint
    app.register_blueprint(api_v1_bp, url_prefix='/api/v1')

def register_extensions(app):
    """Register Flask extensions."""
    # Initialize extensions here
    pass

import os
from flask import Flask, jsonify, request

def create_app(test_config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__, instance_relative_config=True, template_folder='../templates')

    # --- Configuration ---
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        UPLOAD_FOLDER=os.environ.get('UPLOAD_FOLDER', os.path.join(app.instance_path, 'uploads')),
        MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16 MB
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.update(test_config)

    # --- Ensure the instance folder exists ---
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # --- Ensure the upload folder exists ---
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'])
    except OSError:
        pass

    # --- Register Blueprints ---
    from . import routes
    app.register_blueprint(routes.web)

    # --- Health Check ---
    @app.route('/health')
    def health_check():
        return jsonify({'status': 'healthy'}), 200

    print("App created successfully")
    return app

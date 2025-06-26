"""
CSV to YAML Converter Application

This is the main application module for the CSV to YAML Converter.
It provides a web interface and REST API for converting CSV files to YAML format.
"""

import os
import csv
import yaml
import pandas as pd
import uuid
import shutil
import logging
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from flask import (
    Flask, render_template, request, jsonify, 
    send_from_directory, redirect, url_for, g, 
    has_request_context, current_app
)
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_restx import Api, Resource, fields

# Configure logging
def configure_logging(app):
    """Configure application logging."""
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(app.root_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up file handler
    log_file = os.path.join(logs_dir, 'app.log')
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=1024 * 1024 * 10,  # 10MB
        backupCount=10
    )
    
    # Set formatter
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s '
        '[in %(pathname)s:%(lineno)d] [%(request_id)s]'
    )
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handler
    root_logger.addHandler(file_handler)
    
    # Set application logger level based on config
    app_logger = logging.getLogger(__name__)
    app_logger.setLevel(
        logging.DEBUG if app.config.get('DEBUG', False) else logging.INFO
    )
    
    return app_logger

# Create Flask application
app = Flask(__name__)


class RequestIdFilter(logging.Filter):
    """Add request ID to log records."""
    def filter(self, record):
        record.request_id = getattr(g, 'request_id', 'no-request')
        return True


# Configure logging after app creation
app.logger = configure_logging(app)
app.logger.addFilter(RequestIdFilter())
# Trust X-Forwarded-* headers when running behind a reverse proxy
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Request logging middleware
@app.before_request
def before_request():
    """Process each request before it's handled by a view."""
    # Skip logging for health checks
    if request.path in ['/health', '/readiness', '/favicon.ico']:
        return
    
    # Generate a unique request ID if not provided
    g.start_time = time.time()
    g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    
    # Log request details
    extra = {
        'method': request.method,
        'path': request.path,
        'ip': request.remote_addr,
        'user_agent': request.user_agent.string,
        'params': dict(request.args),
        'content_type': request.content_type,
        'request_id': g.request_id
    }
    
    # Log request body for non-GET requests if it's not too large
    if request.method != 'GET' and request.content_length and request.content_length < 1024 * 10:  # 10KB
        try:
            if request.is_json:
                extra['request_body'] = request.get_json()
            elif request.content_type == 'application/x-www-form-urlencoded':
                extra['form_data'] = dict(request.form)
        except Exception as e:
            app.logger.warning(f'Could not log request body: {str(e)}')
    
    app.logger.info('Request started', extra=extra)

@app.after_request
def after_request(response):
    """Process each response before it's sent to the client."""
    # Skip logging for health checks
    if request.path in ['/health', '/readiness', '/favicon.ico']:
        return response
    
    # Calculate request duration
    duration = (time.time() - g.start_time) * 1000  # Convert to milliseconds
    
    # Log response details
    extra = {
        'method': request.method,
        'path': request.path,
        'status': response.status_code,
        'duration_ms': round(duration, 2),
        'content_type': response.content_type,
        'content_length': response.content_length or 0,
        'request_id': getattr(g, 'request_id', 'unknown')
    }
    
    # Log response body for errors
    if response.status_code >= 400 and response.content_length and response.content_length < 1024:  # 1KB
        try:
            if response.is_json:
                extra['response_body'] = response.get_json()
            elif response.content_type == 'text/plain':
                extra['response_body'] = response.get_data(as_text=True)[:500]  # First 500 chars
        except Exception as e:
            app.logger.warning(f'Could not log response body: {str(e)}')
    
    # Log at appropriate level based on status code
    if 500 <= response.status_code <= 599:
        app.logger.error('Request completed with server error', extra=extra)
    elif 400 <= response.status_code <= 499:
        app.logger.warning('Request completed with client error', extra=extra)
    else:
        app.logger.info('Request completed successfully', extra=extra)
    
    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Add request ID to response headers
    if hasattr(g, 'request_id'):
        response.headers['X-Request-ID'] = g.request_id
    
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all uncaught exceptions and log them."""
    # Get request context if available
    request_info = {}
    if has_request_context():
        request_info = {
            'path': request.path,
            'method': request.method,
            'endpoint': request.endpoint or 'unknown',
            'url': request.url,
            'remote_addr': request.remote_addr,
            'user_agent': request.user_agent.string if request.user_agent else None,
            'referrer': request.referrer,
            'content_type': request.content_type,
            'is_json': request.is_json,
            'is_secure': request.is_secure,
            'headers': dict(request.headers)
        }
    
    # Log the exception with full context
    app.logger.error(
        f'Unhandled exception: {str(e)}',
        exc_info=True,
        extra={
            'request_id': getattr(g, 'request_id', 'no-request'),
            'error_type': e.__class__.__name__,
            'error_details': str(e),
            'request': request_info
        }
    )
    
    # Return a JSON response for API errors
    if has_request_context() and (request.path.startswith('/api/') or request.is_json):
        response = {
            'error': 'Internal server error',
            'request_id': getattr(g, 'request_id', 'unknown'),
            'status': 500,
            'path': request.path,
            'method': request.method
        }
        
        # Include more details in development
        if app.config.get('DEBUG', False):
            response['error_details'] = str(e)
            response['error_type'] = e.__class__.__name__
        
        return jsonify(response), 500
        
    # For non-API requests, render an error page
    return render_template('error.html', 
                         error='An unexpected error occurred',
                         status_code=500,
                         request_id=getattr(g, 'request_id', 'unknown')), 500

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def ensure_user_dirs(username, experiment):
    """
    Ensure user, experiment, and architecture directories exist.
    Returns: (base_dir, experiment_dir, architecture_dir)
    """
    try:
        # Create base directory structure: Users/username/Experiment/experiment/
        base_dir = Path(app.config['BASE_UPLOAD_FOLDER']) / username / 'Experiment' / experiment
        architecture_dir = base_dir / 'Architecture'
        
        # Create directories with parents as needed
        base_dir.mkdir(parents=True, exist_ok=True)
        architecture_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify directories were created and are writable
        if not base_dir.exists() or not base_dir.is_dir() or not os.access(str(base_dir), os.W_OK):
            raise Exception(f"Failed to create or access directory: {base_dir}")
            
        if not architecture_dir.exists() or not architecture_dir.is_dir() or not os.access(str(architecture_dir), os.W_OK):
            raise Exception(f"Failed to create or access directory: {architecture_dir}")
        
        return str(base_dir), str(architecture_dir)
        
    except Exception as e:
        print(f"Error in ensure_user_dirs: {str(e)}")
        raise

def csv_to_yaml(csv_file_path):
    """
    Convert a CSV file to YAML format.
    
    This function reads a CSV file, converts it to a list of dictionaries,
    and then dumps it to a YAML-formatted string.
    
    Args:
        csv_file_path (str): Path to the CSV file to convert
        
    Returns:
        tuple: (yaml_string, error_message) - If successful, yaml_string contains the YAML data
               and error_message is None. If an error occurs, yaml_string is None and 
               error_message contains the error details.
    """
    try:
        # Verify the file exists and is readable
        if not os.path.exists(csv_file_path):
            return None, f"File not found: {csv_file_path}"
            
        if not os.access(csv_file_path, os.R_OK):
            return None, f"No read permission for file: {csv_file_path}"
        
        # Try different encodings and delimiters
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252']
        delimiters = [',', ';', '\t', '|']
        
        df = None
        
        # First, try with automatic delimiter detection
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file_path, encoding=encoding, engine='python')
                if len(df.columns) > 1:  # Successfully read with multiple columns
                    break
            except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
        
        # If automatic detection failed or we only have one column, try different delimiters
        if df is None or len(df.columns) <= 1:
            for encoding in encodings:
                for delimiter in delimiters:
                    try:
                        df = pd.read_csv(
                            csv_file_path, 
                            delimiter=delimiter, 
                            encoding=encoding, 
                            engine='python',
                            on_bad_lines='warn'
                        )
                        if len(df.columns) > 1:  # Successfully read with multiple columns
                            break
                    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
                        continue
                if df is not None and len(df.columns) > 1:
                    break
        
        # If we still couldn't parse the CSV
        if df is None or len(df.columns) <= 1:
            return None, "Could not parse CSV file. Please check the file format and delimiter."
        
        # Clean up column names (remove leading/trailing spaces and convert to string)
        df.columns = df.columns.astype(str).str.strip()
        
        # Convert NaN/None values to empty strings for better YAML output
        df = df.fillna('')
        
        # Convert to list of dictionaries (one per row)
        data = df.to_dict(orient='records')
        
        # Generate YAML with safe_dump to prevent potential security issues
        yaml_str = yaml.safe_dump(
            data,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
            width=float('inf'),  # Prevent line wrapping
            encoding='utf-8',
            explicit_start=True,
            explicit_end=True
        )
        
        # Decode bytes to string if needed
        if isinstance(yaml_str, bytes):
            yaml_str = yaml_str.decode('utf-8')
        
        return yaml_str, None
        
    except Exception as e:
        error_msg = f"Error converting CSV to YAML: {str(e)}"
        current_app.logger.error(error_msg, exc_info=True)
        return None, error_msg

@app.route('/health')
def health_check():
    """Health check endpoint for Kubernetes liveness and readiness probes."""
    try:
        # Check if the upload directory is accessible
        upload_dir = app.config.get('UPLOAD_FOLDER', '/tmp')
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
        
        # Verify write permission
        test_file = os.path.join(upload_dir, '.healthcheck')
        with open(test_file, 'w') as f:
            f.write('ok')
        os.remove(test_file)
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'environment': app.config.get('FLASK_ENV', 'development')
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@app.route('/readiness')
def readiness_check():
    """Readiness check endpoint for Kubernetes."""
    return health_check()

def register_blueprints(app):
    """Register Flask blueprints."""
    # Import blueprints here to avoid circular imports
    from app.api.v1 import init_api
    
    # Initialize API if enabled
    if app.config.get('ENABLE_API', True):
        api = init_api(app)
        app.logger.info('API endpoints registered')
    
    # Register web UI routes if enabled
    if app.config.get('ENABLE_WEB_UI', True):
        @app.route('/')
        def index():
            """Render the main application page."""
            return render_template('index.html')
        
        app.logger.info('Web UI routes registered')
    
    return app

# Register blueprints
register_blueprints(app)

@app.route('/upload/architecture', methods=['POST'])
def upload_architecture():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    username = request.form.get('username', '').strip()
    experiment = request.form.get('experiment', '').strip()
    
    if not username or not experiment:
        return jsonify({'error': 'Username and experiment name are required'}), 400
    
    # Create the architecture directory if it doesn't exist
    try:
        # Create the full path: Users/username/Experiment/experiment/Architecture
        base_dir = os.path.join('Users', username, 'Experiment', experiment)
        arch_dir = os.path.join(base_dir, 'Architecture')
        
        # Create parent directories if they don't exist
        os.makedirs(arch_dir, exist_ok=True)
        
        # Ensure the directory was created and is writable
        if not os.path.isdir(arch_dir):
            return jsonify({'error': f'Failed to create directory: {arch_dir}'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Error creating directory: {str(e)}'}), 500
    
    uploaded_files = []
    
    for file in request.files.getlist('files[]'):
        if file.filename == '':
            continue
            
        try:
            # Sanitize the filename
            filename = secure_filename(file.filename)
            if not filename:
                continue
                
            # Ensure the filename is safe and doesn't contain path traversal
            if '..' in filename or filename.startswith('/') or '~' in filename:
                continue
                
            # Create the full path to save the file
            file_path = os.path.join(arch_dir, filename)
            
            # Save the file
            file.save(file_path)
            
            # Verify the file was saved
            if os.path.exists(file_path):
                uploaded_files.append(filename)
            else:
                print(f"Warning: File {filename} was not saved successfully")
                
        except Exception as e:
            print(f"Error processing file {file.filename}: {str(e)}")
            continue
    
    if not uploaded_files:
        return jsonify({
            'error': 'No files were uploaded. Please check file names and try again.',
            'uploaded_count': 0
        }), 400
    
    return jsonify({
        'message': f'Successfully uploaded {len(uploaded_files)} file(s)',
        'uploaded_files': uploaded_files,
        'uploaded_count': len(uploaded_files)
    })
    if errors:
        response['warnings'] = errors
    
    return jsonify(response)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    username = request.form.get('username', '').strip()
    experiment = request.form.get('experiment', '').strip()
    
    # Validate inputs
    if not username or not experiment:
        return jsonify({'error': 'Username and experiment are required'}), 400
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Please upload a CSV file.'}), 400
        
    try:
        # Create necessary directories and get paths
        base_dir, _ = ensure_user_dirs(username, experiment)
        
        # Save the uploaded CSV file with the experiment name
        csv_filename = f"{experiment}.csv"
        csv_path = Path(base_dir) / csv_filename
        
        # Save the uploaded file
        file.save(str(csv_path))
        
        # Verify the file was saved
        if not csv_path.exists() or not csv_path.is_file():
            return jsonify({'error': 'Failed to save the uploaded file'}), 500
        
        # Convert CSV to YAML
        yaml_data, error = csv_to_yaml(str(csv_path))
        if error:
            return jsonify({'error': f'Error converting CSV to YAML: {error}'}), 500
        
        # Save YAML with a consistent naming pattern
        yaml_filename = f"config_{experiment}.yaml"
        yaml_path = Path(base_dir) / yaml_filename
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_data)
        
        # Verify the YAML file was saved
        if not yaml_path.exists() or not yaml_path.is_file():
            return jsonify({'error': 'Failed to save the generated YAML file'}), 500
        
        # Create a download URL for the YAML file
        download_url = url_for('download_file', 
                             username=username, 
                             experiment=experiment, 
                             filename=yaml_filename)
        
        return jsonify({
            'message': 'File successfully processed',
            'yaml': yaml_data,
            'download_url': download_url,
            'csv_filename': csv_filename,
            'yaml_filename': yaml_filename,
            'experiment': experiment
        })
        
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/download/<path:username>/<path:experiment>/<path:filename>')
def download_file(username, experiment, filename):
    try:
        # Security: Sanitize the username, experiment, and filename
        username = secure_filename(username)
        experiment = secure_filename(experiment)
        filename = secure_filename(filename)
        
        if not username or not experiment or not filename:
            return jsonify({'error': 'Invalid parameters'}), 400
        
        # Build the file path using the new directory structure
        file_path = Path('Users') / username / 'Experiment' / experiment / filename
        
        # Convert to absolute path for security checks
        abs_file_path = file_path.resolve()
        
        # Ensure the path is within the allowed directory to prevent directory traversal
        allowed_dir = Path('Users').resolve()
        if not str(abs_file_path).startswith(str(allowed_dir)):
            return jsonify({'error': 'Access denied'}), 403
        
        # Check if file exists and is a file (not a directory)
        if not abs_file_path.exists() or not abs_file_path.is_file():
            return jsonify({'error': 'File not found'}), 404
        
        # Get the directory and filename separately for send_from_directory
        dir_path = abs_file_path.parent
        file_name = abs_file_path.name
        
        # Send the file for download
        return send_from_directory(
            directory=dir_path,
            path=file_name,
            as_attachment=True,
            download_name=filename  # This ensures the original filename is used for download
        )
        
    except Exception as e:
        print(f"Error in download_file: {str(e)}")
        return jsonify({'error': 'Failed to download file'}), 500

if __name__ == '__main__':
    # Create base upload directory if it doesn't exist
    Path(app.config['BASE_UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

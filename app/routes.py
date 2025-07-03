"""
Application routes.
"""

from flask import jsonify, request, send_file, url_for, g, render_template, redirect, current_app
from werkzeug.utils import secure_filename
import os
import csv
import yaml
from pathlib import Path
from datetime import datetime
from flask import Blueprint

# Create a blueprint for web routes
web = Blueprint('web', __name__)

# Get the app instance from the current_app proxy
def get_app():
    return current_app._get_current_object()

# Note: The upload directory is configured via `current_app.config['UPLOAD_FOLDER']` inside create_app.
# We no longer hard-code a separate Users path here.

from flask import session, redirect, url_for

@web.route('/')
def index():
    return redirect(url_for('web.username'))

@web.route('/username', methods=['GET', 'POST'])
def username():
    if request.method == 'POST':
        session['user_name'] = request.form['user_name'].strip()
        return redirect(url_for('web.experiment'))
    return render_template('username.html')

@web.route('/experiment', methods=['GET', 'POST'])
def experiment():
    if 'user_name' not in session:
        return redirect(url_for('web.username'))
    if request.method == 'POST':
        session['experiment_name'] = request.form['experiment_name'].strip()
        return redirect(url_for('web.upload_config'))
    return render_template('experiment.html')

@web.route('/upload_config', methods=['GET', 'POST'])
def upload_config():
    if 'user_name' not in session or 'experiment_name' not in session:
        return redirect(url_for('web.username'))
    if request.method == 'POST':
        config_file = request.files.get('config_file')
        if not config_file or config_file.filename == '':
            return render_template('upload_config.html', error='Please upload a configuration file.')
        session['config_filename'] = config_file.filename
        session['config_bytes'] = config_file.read()
        return redirect(url_for('web.upload_csv'))
    return render_template('upload_config.html')

@web.route('/upload_csv', methods=['GET', 'POST'])
def upload_csv():
    if 'user_name' not in session or 'experiment_name' not in session or 'config_bytes' not in session:
        return redirect(url_for('web.username'))
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        if not csv_file or csv_file.filename == '' or not csv_file.filename.lower().endswith('.csv'):
            return render_template('upload_csv.html', error='Please upload a valid CSV file.')
        session['csv_filename'] = csv_file.filename
        # Save CSV to temp session storage (not disk yet)
        session['csv_bytes'] = csv_file.read()
        return redirect(url_for('web.upload_archs'))
    return render_template('upload_csv.html')

@web.route('/upload_archs', methods=['GET', 'POST'])
def upload_archs():
    if 'user_name' not in session or 'experiment_name' not in session or 'csv_bytes' not in session or 'config_bytes' not in session:
        return redirect(url_for('web.username'))
    if request.method == 'POST':
        arch_files = request.files.getlist('arch_files')
        if not arch_files or all(f.filename == '' for f in arch_files):
            return render_template('upload_archs.html', error='Please upload at least one architecture file.')
        # Prepare folder structure
        safe_user = secure_filename(session['user_name'])
        safe_exp = secure_filename(session['experiment_name'])
        csv_filename = secure_filename(session['csv_filename'])
        config_filename = secure_filename(session['config_filename'])
        csv_base = os.path.splitext(csv_filename)[0]
        base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
        user_dir = os.path.join(base_upload, safe_user)
        data_dir = os.path.join(user_dir, 'Data', csv_base)
        exp_dir = os.path.join(user_dir, safe_exp)
        arch_dir = os.path.join(exp_dir, 'Architecture')
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(arch_dir, exist_ok=True)

        # Save experiment config file
        exp_config_path = os.path.join(exp_dir, config_filename)
        with open(exp_config_path, 'wb') as f:
            f.write(session['config_bytes'])

        # Save CSV file
        csv_path = os.path.join(data_dir, csv_filename)
        with open(csv_path, 'wb') as f:
            f.write(session['csv_bytes'])
        # Save architecture files
        arch_saved = []
        for arch_file in arch_files:
            if arch_file and arch_file.filename:
                arch_filename = secure_filename(arch_file.filename)
                arch_path = os.path.join(arch_dir, arch_filename)
                arch_file.save(arch_path)
                arch_saved.append(arch_filename)
        # Create experiment config YAML
        config_path = os.path.join(exp_dir, f"config_{safe_exp}.yaml")
        config_data = {
            'user': session['user_name'],
            'experiment': session['experiment_name'],
            'experiment_config_file': config_filename,
            'csv_file': csv_filename,
            'architecture_files': arch_saved,
            'timestamp': datetime.utcnow().isoformat()
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        session.pop('csv_bytes', None)
        session.pop('csv_filename', None)
        session.pop('config_bytes', None)
        session.pop('config_filename', None)
        return redirect(url_for('web.done'))
    return render_template('upload_archs.html')

@web.route('/done')
def done():
    return render_template('done.html')


@web.route('/files')
def list_files():
    """Lists all uploaded files, organized by user."""
    upload_folder = current_app.config['UPLOAD_FOLDER']
    base_path = os.path.join(upload_folder, 'Users')
    
    if not os.path.exists(base_path):
        return render_template('files.html', files_by_user={})

    files_by_user = {}
    for user in sorted(os.listdir(base_path)):
        user_path = os.path.join(base_path, user)
        if os.path.isdir(user_path):
            files_by_user[user] = []
            for dirpath, _, filenames in os.walk(user_path):
                for filename in sorted(filenames):
                    relative_dir = os.path.relpath(dirpath, base_path)
                    # Use forward slashes for URL path compatibility
                    files_by_user[user].append(os.path.join(relative_dir, filename).replace('\\', '/'))
    
    return render_template('files.html', files_by_user=files_by_user)


@web.route('/uploads/<path:filepath>')
def serve_upload(filepath):
    """Serves an uploaded file securely."""
    base_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    file_abs_path = os.path.abspath(os.path.join(base_path, filepath))
    
    if not file_abs_path.startswith(os.path.abspath(base_path)):
        return "Forbidden", 403
        
    return send_file(file_abs_path)


# (upload_experiment route removed - flow is now multi-step, handled by the above routes)


@web.route('/health')
def health_check():
    """Health check endpoint for Kubernetes."""
    try:
        # Check filesystem access
        try:
            test_file = os.path.join(current_app.config['UPLOAD_FOLDER'], '.healthcheck')
            with open(test_file, 'w') as f:
                f.write('healthcheck')
            os.remove(test_file)
            fs_status = 'ok'
        except Exception as e:
            current_app.logger.error(f'Filesystem health check failed: {str(e)}')
            fs_status = 'error'
        
        return jsonify({
            'status': 'healthy' if fs_status == 'ok' else 'degraded',
            'timestamp': datetime.utcnow().isoformat(),
            'version': '1.0.0',
            'services': {
                'filesystem': fs_status,
                'api': 'ok'
            }
        }), 200 if fs_status == 'ok' else 503
    except Exception as e:
        current_app.logger.error(f'Health check failed: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'error',
            'error': str(e) if current_app.debug else 'Internal server error',
            'timestamp': datetime.utcnow().isoformat()
        }), 500

@web.route('/readiness')
def readiness_check():
    """Readiness check endpoint for Kubernetes."""
    try:
        # Add any additional readiness checks here
        # For example, check database connection, external services, etc.
        # if not check_database_connection():
        #     return jsonify({'status': 'not ready', 'reason': 'Database unavailable'}), 503
        
        return jsonify({
            'status': 'ready',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        current_app.logger.error(f'Readiness check failed: {str(e)}', exc_info=True)
        return jsonify({
            'status': 'not ready',
            'error': str(e) if current_app.debug else 'Service not ready',
            'timestamp': datetime.utcnow().isoformat()
        }), 503

# Add more routes here as needed

# Error handlers
@web.errorhandler(404)
def not_found_error(error):
    # If it's an API request, return JSON
    if request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    # Otherwise, render an HTML error page
    return render_template('error.html', error='Page not found', code=404), 404

@web.errorhandler(500)
def internal_error(error):
    current_app.logger.error(f'Server error: {str(error)}', exc_info=True)
    # If it's an API request, return JSON
    if request.path.startswith('/api/'):
        return jsonify({
            'error': 'Internal server error',
            'message': str(error) if current_app.debug else 'An unexpected error occurred'
        }), 500
    # Otherwise, render an HTML error page
    return render_template('error.html', 
                         error='Internal server error', 
                         message=str(error) if current_app.debug else 'An unexpected error occurred',
                         code=500), 500

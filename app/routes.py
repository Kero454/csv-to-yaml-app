"""
Application routes.
"""

from flask import (
    Blueprint, flash, redirect, render_template, request, session, url_for,
    current_app, send_file, jsonify
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import csv
import yaml
from datetime import datetime
from app import db

# Create a blueprint for web routes
web = Blueprint('routes', __name__)

@web.route('/')
@web.route('/index')
def index():
    if current_user.is_authenticated:
        return render_template('index.html', title='Home')
    return redirect(url_for('auth.login'))

@web.route('/experiment', methods=['GET', 'POST'])
@login_required
def experiment():
    if request.method == 'POST':
        session['experiment_name'] = request.form['experiment_name'].strip()
        return redirect(url_for('routes.upload_config'))
    return render_template('experiment.html')

@web.route('/upload_config', methods=['GET', 'POST'])
@login_required
def upload_config():
    if 'experiment_name' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    if request.method == 'POST':
        config_file = request.files.get('config_file')
        if not config_file or config_file.filename == '':
            flash('Please upload a configuration file.')
            return redirect(request.url)
        session['config_filename'] = secure_filename(config_file.filename)
        session['config_bytes'] = config_file.read()
        return redirect(url_for('routes.upload_csv'))
    return render_template('upload_config.html')

@web.route('/upload_csv', methods=['GET', 'POST'])
@login_required
def upload_csv():
    if 'experiment_name' not in session or 'config_bytes' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        if not csv_file or csv_file.filename == '' or not csv_file.filename.lower().endswith('.csv'):
            flash('Please upload a valid CSV file.')
            return redirect(request.url)
        session['csv_filename'] = secure_filename(csv_file.filename)
        session['csv_bytes'] = csv_file.read()
        return redirect(url_for('routes.upload_archs'))
    return render_template('upload_csv.html')

@web.route('/upload_archs', methods=['GET', 'POST'])
@login_required
def upload_archs():
    if 'experiment_name' not in session or 'csv_bytes' not in session or 'config_bytes' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    if request.method == 'POST':
        arch_files = request.files.getlist('arch_files')
        if not arch_files or all(f.filename == '' for f in arch_files):
            flash('Please upload at least one architecture file.')
            return redirect(request.url)
        
        safe_user = secure_filename(current_user.username)
        safe_exp = secure_filename(session['experiment_name'])
        csv_filename = session['csv_filename']
        config_filename = session['config_filename']
        csv_base = os.path.splitext(csv_filename)[0]
        
        base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
        user_dir = os.path.join(base_upload, safe_user)
        data_dir = os.path.join(user_dir, 'Data', csv_base)
        exp_dir = os.path.join(user_dir, safe_exp)
        arch_dir = os.path.join(exp_dir, 'Architecture')
        
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(arch_dir, exist_ok=True)

        exp_config_path = os.path.join(exp_dir, config_filename)
        with open(exp_config_path, 'wb') as f:
            f.write(session['config_bytes'])

        csv_path = os.path.join(data_dir, csv_filename)
        with open(csv_path, 'wb') as f:
            f.write(session['csv_bytes'])
            
        arch_saved = []
        for arch_file in arch_files:
            if arch_file and arch_file.filename:
                arch_filename = secure_filename(arch_file.filename)
                arch_path = os.path.join(arch_dir, arch_filename)
                arch_file.save(arch_path)
                arch_saved.append(arch_filename)
                
        config_path = os.path.join(exp_dir, f"config_{safe_exp}.yaml")
        config_data = {
            'user': current_user.username,
            'experiment': session['experiment_name'],
            'experiment_config_file': config_filename,
            'csv_file': csv_filename,
            'architecture_files': arch_saved,
            'timestamp': datetime.utcnow().isoformat()
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
            
        session.pop('csv_bytes', None)
        session.pop('csv_filename', None)
        session.pop('config_bytes', None)
        session.pop('config_filename', None)
        session.pop('experiment_name', None)
        
        flash('Your files have been uploaded and processed successfully!')
        return redirect(url_for('routes.done'))
        
    return render_template('upload_archs.html')

@web.route('/done')
@login_required
def done():
    return render_template('done.html')

@web.route('/files')
@login_required
def list_files():
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
                    files_by_user[user].append(os.path.join(relative_dir, filename).replace('\\', '/'))
    
    return render_template('files.html', files_by_user=files_by_user)

@web.route('/uploads/<path:filepath>')
@login_required
def serve_upload(filepath):
    base_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    file_abs_path = os.path.abspath(os.path.join(base_path, filepath))
    
    if not file_abs_path.startswith(os.path.abspath(base_path)):
        return "Forbidden", 403
        
    return send_file(file_abs_path)

@web.route('/health')
def health_check():
    """Liveness check for Kubernetes."""
    return jsonify({'status': 'healthy'}), 200

@web.route('/ready')
def readiness_check():
    """Readiness check for Kubernetes."""
    db_status = 'ok'
    fs_status = 'ok'
    try:
        db.session.execute('SELECT 1')
    except Exception as e:
        current_app.logger.error(f'Database readiness check failed: {str(e)}')
        db_status = 'error'

    try:
        test_file = os.path.join(current_app.config['UPLOAD_FOLDER'], '.readycheck')
        with open(test_file, 'w') as f:
            f.write('readycheck')
        os.remove(test_file)
    except Exception as e:
        current_app.logger.error(f'Filesystem readiness check failed: {str(e)}')
        fs_status = 'error'

    if db_status == 'ok' and fs_status == 'ok':
        return jsonify({'status': 'ready'}), 200
    else:
        return jsonify({
            'status': 'not ready',
            'services': {
                'database': db_status,
                'filesystem': fs_status
            }
        }), 503

# Error handlers
@web.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found', code=404), 404

@web.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    current_app.logger.error(f'Server error: {error}', exc_info=True)
    return render_template('error.html', error='Internal server error', code=500), 500


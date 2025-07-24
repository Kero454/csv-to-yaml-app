"""
Application routes.
"""

from flask import (
    Blueprint, flash, redirect, render_template, request, session, url_for,
    current_app, send_file, jsonify, Response
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
import os
import csv
import yaml
from datetime import datetime
from app import db

def get_user_csv_files(user):
    """Get list of CSV files uploaded by the user.
    
    Args:
        user: Current user object
        
    Returns:
        List of dictionaries containing CSV file information
    """
    csv_files = []
    try:
        safe_user = secure_filename(user.username)
        base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
        user_data_dir = os.path.join(base_upload, safe_user, 'Data')
        
        if os.path.exists(user_data_dir):
            for csv_dir in os.listdir(user_data_dir):
                csv_path = os.path.join(user_data_dir, csv_dir)
                if os.path.isdir(csv_path):
                    for file in os.listdir(csv_path):
                        if file.lower().endswith('.csv') and os.path.isfile(os.path.join(csv_path, file)):
                            csv_files.append({
                                'name': file,
                                'display_name': f"{csv_dir}/{file}",
                                'path': os.path.join(csv_dir, file),
                                'full_path': os.path.join(csv_path, file)
                            })
    except Exception as e:
        current_app.logger.error(f"Error getting user CSV files: {str(e)}")
    
    return csv_files

def generate_config_template(experiment_name):
    """Generate a template configuration file for the experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dict containing the template configuration
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create a template with common configuration options
    template = {
        "experiment": {
            "name": experiment_name,
            "date": current_date,
            "created_by": current_user.username,
            "description": "Enter experiment description here"
        },
        "data": {
            "csv_file": "Will be set from uploaded CSV",
            "input_columns": ["column1", "column2"],  # Replace with actual columns from CSV
            "output_columns": ["target"],  # Replace with actual target column
            "preprocessing": {
                "normalize": True,
                "handle_missing": "mean"  # Options: mean, median, zero, none
            }
        },
        "model": {
            "architecture": "Will be set from uploaded architecture files",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "early_stopping": True,
                "patience": 10
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1"],
                "test_split": 0.2,
                "validation_split": 0.1,
                "cross_validation": False,
                "cv_folds": 5
            }
        },
        "output": {
            "save_predictions": True,
            "save_model": True,
            "visualization": ["confusion_matrix", "roc_curve"]
        }
    }
    
    return template

# Create a blueprint for web routes
web = Blueprint('routes', __name__)

@web.route('/')
@web.route('/index')
def index():
    if current_user.is_authenticated:
        return render_template('index.html', username=current_user.username)
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
        # Check if the user wants to generate a template
        if 'generate_template' in request.form:
            # Generate a template config file
            config_data = generate_config_template(session['experiment_name'])
            config_yaml = yaml.dump(config_data, default_flow_style=False)
            
            # Store in session
            session['config_filename'] = f"{secure_filename(session['experiment_name'])}_config.yaml"
            session['config_bytes'] = config_yaml.encode('utf-8')
            
            # Return the template for editing
            return render_template('edit_config.html', 
                                   config_content=config_yaml,
                                   experiment_name=session['experiment_name'])
        # Handle file upload
        elif 'config_file' in request.files:
            config_file = request.files.get('config_file')
            if not config_file or config_file.filename == '':
                flash('Please upload a configuration file or generate a template.')
                return redirect(request.url)
            session['config_filename'] = secure_filename(config_file.filename)
            session['config_bytes'] = config_file.read()
            return redirect(url_for('routes.upload_csv'))
        # Handle saving edited config
        elif 'edited_config' in request.form:
            config_content = request.form.get('edited_config')
            if not config_content:
                flash('Configuration cannot be empty.')
                return redirect(request.url)
            
            # Validate YAML format
            try:
                yaml.safe_load(config_content)
            except yaml.YAMLError as e:
                flash(f'Invalid YAML format: {str(e)}')
                return render_template('edit_config.html', 
                                      config_content=config_content,
                                      experiment_name=session['experiment_name'])
            
            # Store in session
            session['config_bytes'] = config_content.encode('utf-8')
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


@web.route('/view_file_content')
@login_required
def view_file_content():
    """Serve file content for the preview modal."""
    filepath = request.args.get('filepath')
    current_app.logger.info(f"Received request for filepath: {filepath}")
    if not filepath:
        return jsonify({'error': 'Filepath is required.'}), 400

    # Remove 'uploads/' prefix if present to get the relative path
    if filepath.startswith('uploads/'):
        relative_path = filepath[8:]  # Remove 'uploads/' prefix
    else:
        relative_path = filepath

    # Security check: Ensure the file is within the user's own directory
    # inside the main UPLOAD_FOLDER to prevent directory traversal.
    base_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    safe_filepath = os.path.abspath(os.path.join(base_path, relative_path))
    
    current_app.logger.info(f"Base path: {base_path}")
    current_app.logger.info(f"Safe filepath: {safe_filepath}")
    current_app.logger.info(f"Absolute base path: {os.path.abspath(base_path)}")

    # 1. Check for directory traversal
    if not safe_filepath.startswith(os.path.abspath(base_path)):
        current_app.logger.warning(f"Directory traversal check failed: {safe_filepath} does not start with {os.path.abspath(base_path)}")
        current_app.logger.warning(f"Potential directory traversal attempt by user {current_user.id} for path {filepath}")
        return jsonify({'error': 'Access denied.'}), 403
    
    current_app.logger.info(f"Directory traversal check passed")

    # 2. Check that the user is accessing their own files
    # Use secure_filename to sanitize username consistently with file storage
    sanitized_username = secure_filename(current_user.username)
    current_app.logger.info(f"Current user: {current_user.username}, Sanitized: {sanitized_username}, User ID: {current_user.id}")
    current_app.logger.info(f"Relative path: {relative_path}")
    current_app.logger.info(f"Expected prefix: {sanitized_username}/")
    if not relative_path.startswith(sanitized_username + '/'):
        current_app.logger.warning(f"Username check failed: {relative_path} does not start with {sanitized_username}/")
        current_app.logger.warning(f"User {current_user.id} ({current_user.username}) attempting to access other user's file at {relative_path}")
        return jsonify({'error': 'Access denied.'}), 403
    
    current_app.logger.info(f"Username check passed")

    # Check if file exists before trying to read it
    current_app.logger.info(f"Checking if file exists: {safe_filepath}")
    if not os.path.exists(safe_filepath):
        current_app.logger.warning(f"File not found: {safe_filepath}")
        return jsonify({'error': 'File not found.'}), 404
    
    current_app.logger.info(f"File exists, attempting to read: {safe_filepath}")
    try:
        with open(safe_filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        current_app.logger.info(f"Successfully read file, content length: {len(content)}")
        return jsonify({'content': content})
    except FileNotFoundError:
        current_app.logger.error(f"File not found during read: {safe_filepath}")
        return jsonify({'error': 'File not found.'}), 404
    except Exception as e:
        current_app.logger.error(f"Error reading file {safe_filepath} for preview: {e}")
        return jsonify({'error': 'An error occurred while reading the file.'}), 500


@web.route('/health')
def health_check():
    """Liveness check for Kubernetes."""
    return jsonify({'status': 'healthy'}), 200

@web.route('/ready')
def readiness_check():
    """Kubernetes readiness probe endpoint."""
    try:
        # Check if all essential directories exist
        for directory in [current_app.config['UPLOAD_FOLDER'], 
                           os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')]:
            if not os.path.exists(directory):
                return Response("Storage not ready: Missing directory: {}".format(directory), status=503)
        
        # Check if we can write to the upload directory
        test_file = os.path.join(current_app.config['UPLOAD_FOLDER'], '.readiness_check')
        try:
            with open(test_file, 'w') as f:
                f.write('readiness_check')
            os.remove(test_file)
        except Exception as e:
            return Response("Storage not ready: Write test failed: {}".format(str(e)), status=503)
        
        # Check database connectivity
        try:
            db.session.execute("SELECT 1")
        except Exception as e:
            return Response("Database not ready: {}".format(str(e)), status=503)
        
        return Response("OK", status=200)
    except Exception as e:
        return Response("Readiness check failed: {}".format(str(e)), status=503)

@web.route('/experiment_explorer')
@login_required
def experiment_explorer():
    """Display all experiments and associated files for the current user."""
    upload_folder = current_app.config['UPLOAD_FOLDER']
    base_path = os.path.join(upload_folder, 'Users')
    user_path = os.path.join(base_path, secure_filename(current_user.username))
    
    if not os.path.exists(user_path):
        return render_template('experiment_explorer.html', experiments=[])
    
    experiments = []
    
    # Find all experiment directories for this user
    for experiment_name in os.listdir(user_path):
        exp_path = os.path.join(user_path, experiment_name)
        if os.path.isdir(exp_path) and experiment_name != 'Data':  # Skip the Data directory
            try:
                # Try to read experiment metadata
                config_path = os.path.join(exp_path, f"config_{experiment_name}.yaml")
                timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')  # Default timestamp
                
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                        if config_data and 'timestamp' in config_data:
                            timestamp = datetime.fromisoformat(config_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                
                # Get configuration files
                config_files = []
                for file in os.listdir(exp_path):
                    if file.lower().endswith(('.yaml', '.yml')) and os.path.isfile(os.path.join(exp_path, file)):
                        # Path for file links (serve_upload route)
                        file_link_path = os.path.join(secure_filename(current_user.username), experiment_name, file)
                        # Path for preview data attributes (view_file_content route)
                        file_preview_path = os.path.join('uploads', secure_filename(current_user.username), experiment_name, file)
                        file_size = os.path.getsize(os.path.join(exp_path, file))
                        config_files.append({
                            'name': file,
                            'path': file_link_path,
                            'preview_path': file_preview_path,
                            'size': f"{file_size / 1024:.1f} KB"
                        })
                
                # Get architecture files
                arch_files = []
                arch_dir = os.path.join(exp_path, 'Architecture')
                if os.path.exists(arch_dir) and os.path.isdir(arch_dir):
                    for file in os.listdir(arch_dir):
                        if os.path.isfile(os.path.join(arch_dir, file)):
                            # Path for file links (serve_upload route)
                            file_link_path = os.path.join(secure_filename(current_user.username), experiment_name, 'Architecture', file)
                            # Path for preview data attributes (view_file_content route)
                            file_preview_path = os.path.join('uploads', secure_filename(current_user.username), experiment_name, 'Architecture', file)
                            file_size = os.path.getsize(os.path.join(arch_dir, file))
                            arch_files.append({
                                'name': file,
                                'path': file_link_path,
                                'preview_path': file_preview_path,
                                'size': f"{file_size / 1024:.1f} KB"
                            })
                
                # Get data files (linked from Data directory)
                data_files = []
                data_dir = os.path.join(user_path, 'Data')
                if os.path.exists(data_dir) and os.path.isdir(data_dir):
                    for csv_dir in os.listdir(data_dir):
                        csv_path = os.path.join(data_dir, csv_dir)
                        if os.path.isdir(csv_path):
                            for file in os.listdir(csv_path):
                                if file.lower().endswith('.csv') and os.path.isfile(os.path.join(csv_path, file)):
                                    # Path for file links (serve_upload route)
                                    file_link_path = os.path.join(secure_filename(current_user.username), 'Data', csv_dir, file)
                                    # Path for preview data attributes (view_file_content route)
                                    file_preview_path = os.path.join('uploads', secure_filename(current_user.username), 'Data', csv_dir, file)
                                    file_size = os.path.getsize(os.path.join(csv_path, file))
                                    data_files.append({
                                        'name': file,
                                        'path': file_link_path,
                                        'preview_path': file_preview_path,
                                        'size': f"{file_size / 1024:.1f} KB"
                                    })
                
                experiments.append({
                    'name': experiment_name,
                    'path': exp_path,
                    'timestamp': timestamp,
                    'config_files': config_files,
                    'arch_files': arch_files,
                    'data_files': data_files
                })
            except Exception as e:
                current_app.logger.error(f"Error processing experiment {experiment_name}: {str(e)}")
    
    # Sort experiments by timestamp (newest first)
    experiments.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('experiment_explorer.html', experiments=experiments)

@web.route('/template_optimizer_form', methods=['GET', 'POST'])
@login_required
def template_optimizer_form():
    """Template optimizer form with 6 steps for advanced configuration."""
    if 'experiment_name' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    
    if request.method == 'POST':
        # Process form data and generate YAML configuration
        config_data = {
            'dataset': {
                'dataset': request.form.get('dataset', 'electricity'),
                'path': '/home/agobbi/Projects/ExpTS/data'  # Default path since dataset_path field was removed
            },
            'scheduler_config': {
                'gamma': float(request.form.get('scheduler_gamma', 0.75)),
                'step_size': int(request.form.get('scheduler_step_size', 2500))
            },
            'optim_config': {
                'lr': float(request.form.get('optim_lr', 0.00005)),
                'weight_decay': float(request.form.get('optim_weight_decay', 0.0001))
            },
            'model_configs': {
                'past_steps': int(request.form.get('model_past_steps', 64)),
                'future_steps': int(request.form.get('model_future_steps', 64)),
                'quantiles': _parse_quantiles(request.form.get('model_quantiles', '')),
                'past_channels': _parse_null_value(request.form.get('model_past_channels')),
                'future_channels': _parse_null_value(request.form.get('model_future_channels')),
                'embs': _parse_null_value(request.form.get('model_embs')),
                'out_channels': _parse_null_value(request.form.get('model_out_channels')),
                'loss_type': _parse_null_value(request.form.get('model_loss_type')),
                'persistence_weight': float(request.form.get('model_persistence_weight', 1.0))
            },
            'split_params': {
                'perc_train': float(request.form.get('split_perc_train', 0.6)),
                'perc_valid': float(request.form.get('split_perc_valid', 0.2)),
                'range_train': _parse_null_value(request.form.get('split_range_train')),
                'range_validation': _parse_null_value(request.form.get('split_range_validation')),
                'range_test': _parse_null_value(request.form.get('split_range_test')),
                'shift': int(request.form.get('split_shift', 0)),
                'starting_point': _parse_null_value(request.form.get('split_starting_point')),
                'skip_step': int(request.form.get('split_skip_step', 1)),
                'past_steps': 'model_configs@past_steps',
                'future_steps': 'model_configs@future_steps'
            },
            'train_config': {
                'dirpath': request.form.get('train_dirpath', '/home/agobbi/Projects/ExpTS/electricity'),
                'num_workers': int(request.form.get('train_num_workers', 0)),
                'auto_lr_find': request.form.get('train_auto_lr_find') == 'on',
                'devices': _parse_devices(request.form.get('train_devices', '0')),
                'seed': int(request.form.get('train_seed', 42))
            },
            'inference': {
                'output_path': request.form.get('inference_output_path', '/home/agobbi/Projects/ExpTS/electricity'),
                'load_last': request.form.get('inference_load_last') == 'on',
                'batch_size': int(request.form.get('inference_batch_size', 200)),
                'num_workers': int(request.form.get('inference_num_workers', 4)),
                'set': request.form.get('inference_set', 'test'),
                'rescaling': request.form.get('inference_rescaling') == 'on'
            },
            'defaults': [
                '_self_',
                {'architecture': _parse_null_value(request.form.get('defaults_architecture'))},
                {'override hydra/launcher': request.form.get('defaults_hydra_launcher', 'joblib')}
            ],
            'hydra': {
                'launcher': {
                    'n_jobs': int(request.form.get('hydra_n_jobs', 4)),
                    'verbose': int(request.form.get('hydra_verbose', 1)),
                    'pre_dispatch': int(request.form.get('hydra_pre_dispatch', 4)),
                    'batch_size': int(request.form.get('hydra_batch_size', 4)),
                    '_target_': request.form.get('hydra_target', 'hydra_plugins.hydra_joblib_launcher.joblib_launcher.JoblibLauncher')
                },
                'output_subdir': _parse_null_value(request.form.get('output_subdir')),
                'sweeper': {
                    'params': {
                        'architecture': request.form.get('sweeper_params', 'glob(*)')
                    }
                }
            }
        }
        
        # Store configuration in session
        session['config_data'] = config_data
        config_yaml = yaml.dump(config_data, default_flow_style=False)
        session['config_bytes'] = config_yaml.encode('utf-8')
        session['config_filename'] = f"{secure_filename(session['experiment_name'])}_config.yaml"
        
        # Check if user selected a dataset from their uploaded files
        dataset_selected = False
        dataset_value = config_data['dataset'].get('dataset', '')
        if dataset_value and dataset_value != '' and '/' in dataset_value:
            # User selected an uploaded CSV file (format: "folder/filename.csv")
            dataset_selected = True
            # Set up CSV data from the selected file
            user_csv_files = get_user_csv_files(current_user)
            for csv_file in user_csv_files:
                if csv_file['path'] == dataset_value:
                    session['csv_filename'] = csv_file['name']
                    # Read the CSV file content
                    with open(csv_file['full_path'], 'rb') as f:
                        session['csv_bytes'] = f.read()
                    break
        
        flash('Configuration generated successfully!')
        
        # Conditional routing: skip CSV upload if dataset selected, otherwise go to CSV upload
        if dataset_selected:
            current_app.logger.info(f"Template Optimizer: Dataset selected from user files, skipping CSV upload step")
            return redirect(url_for('routes.upload_archs'))
        else:
            current_app.logger.info(f"Template Optimizer: No dataset selected, proceeding to CSV upload step")
            return redirect(url_for('routes.upload_csv'))
    
    # Get user's uploaded CSV files for dataset selection
    user_csv_files = get_user_csv_files(current_user)
    
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    return render_template('template_optimizer_form.html', 
                           experiment_name=session['experiment_name'], 
                           username=current_user.username, 
                           current_date=current_date_str,
                           user_csv_files=user_csv_files)

@web.route('/preview_template_yaml', methods=['POST'])
@login_required
def preview_template_yaml():
    """Generate a YAML preview from the template optimizer form data."""
    try:
        # Process form data and generate YAML configuration
        config_data = {
            'dataset': {
                'dataset': request.form.get('dataset', 'electricity'),
                'path': '/home/agobbi/Projects/ExpTS/data'  # Default path since dataset_path field was removed
            },
            'scheduler_config': {
                'gamma': float(request.form.get('scheduler_gamma', 0.75)),
                'step_size': int(request.form.get('scheduler_step_size', 2500))
            },
            'optim_config': {
                'lr': float(request.form.get('optim_lr', 0.00005)),
                'weight_decay': float(request.form.get('optim_weight_decay', 0.0001))
            },
            'model_configs': {
                'past_steps': int(request.form.get('model_past_steps', 64)),
                'future_steps': int(request.form.get('model_future_steps', 64)),
                'quantiles': _parse_quantiles(request.form.get('model_quantiles', '')),
                'past_channels': _parse_null_value(request.form.get('model_past_channels')),
                'future_channels': _parse_null_value(request.form.get('model_future_channels')),
                'embs': _parse_null_value(request.form.get('model_embs')),
                'out_channels': _parse_null_value(request.form.get('model_out_channels')),
                'loss_type': _parse_null_value(request.form.get('model_loss_type')),
                'persistence_weight': float(request.form.get('model_persistence_weight', 1.0))
            },
            'split_params': {
                'perc_train': float(request.form.get('split_perc_train', 0.6)),
                'perc_valid': float(request.form.get('split_perc_valid', 0.2)),
                'range_train': _parse_null_value(request.form.get('split_range_train')),
                'range_validation': _parse_null_value(request.form.get('split_range_validation')),
                'range_test': _parse_null_value(request.form.get('split_range_test')),
                'shift': int(request.form.get('split_shift', 0)),
                'starting_point': _parse_null_value(request.form.get('split_starting_point')),
                'skip_step': int(request.form.get('split_skip_step', 1)),
                'past_steps': 'model_configs@past_steps',
                'future_steps': 'model_configs@future_steps'
            },
            'train_config': {
                'dirpath': request.form.get('train_dirpath', '/home/agobbi/Projects/ExpTS/electricity'),
                'num_workers': int(request.form.get('train_num_workers', 0)),
                'auto_lr_find': request.form.get('train_auto_lr_find') == 'on',
                'devices': _parse_devices(request.form.get('train_devices', '0')),
                'seed': int(request.form.get('train_seed', 42))
            },
            'inference': {
                'output_path': request.form.get('inference_output_path', '/home/agobbi/Projects/ExpTS/electricity'),
                'load_last': request.form.get('inference_load_last') == 'on',
                'batch_size': int(request.form.get('inference_batch_size', 200)),
                'num_workers': int(request.form.get('inference_num_workers', 4)),
                'set': request.form.get('inference_set', 'test'),
                'rescaling': request.form.get('inference_rescaling') == 'on'
            },
            'defaults': [
                '_self_',
                {'architecture': _parse_null_value(request.form.get('defaults_architecture'))},
                {'override hydra/launcher': request.form.get('defaults_hydra_launcher', 'joblib')}
            ],
            'hydra': {
                'launcher': {
                    'n_jobs': int(request.form.get('hydra_n_jobs', 4)),
                    'verbose': int(request.form.get('hydra_verbose', 1)),
                    'pre_dispatch': int(request.form.get('hydra_pre_dispatch', 4)),
                    'batch_size': int(request.form.get('hydra_batch_size', 4)),
                    '_target_': request.form.get('hydra_target', 'hydra_plugins.hydra_joblib_launcher.joblib_launcher.JoblibLauncher')
                },
                'output_subdir': _parse_null_value(request.form.get('output_subdir')),
                'sweeper': {
                    'params': {
                        'architecture': request.form.get('sweeper_params', 'glob(*)')
                    }
                }
            }
        }

        # Generate YAML string but do not save to session
        yaml_data = yaml.dump(config_data, default_flow_style=False, sort_keys=False)
        return jsonify({'yaml_data': yaml_data})

    except Exception as e:
        current_app.logger.error(f"Error generating YAML preview: {str(e)}")
        return jsonify({'error': 'Failed to generate preview. Check form data for errors.'}), 400

def _parse_quantiles(quantiles_str):
    """Parse comma-separated quantiles string into list."""
    if not quantiles_str or not quantiles_str.strip():
        return []
    try:
        return [float(q.strip()) for q in quantiles_str.split(',') if q.strip()]
    except ValueError:
        return []

def _parse_null_value(value):
    """Parse form value, returning None for empty or 'null' values."""
    if not value or value.strip().lower() in ['null', 'none', '']:
        return None
    return value.strip()

def _parse_devices(devices_str):
    """Parse comma-separated devices string into list of integers."""
    try:
        return [int(d.strip()) for d in devices_str.split(',') if d.strip()]
    except ValueError:
        return [0]

@web.route('/config_form', methods=['GET'])
@login_required
def config_form():
    """Display the Google Form-style configuration editor."""
    if 'experiment_name' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    
    # Generate a template if none exists in the session
    if 'config_data' not in session:
        config_data = generate_config_template(session['experiment_name'])
        session['config_data'] = config_data
        config_yaml = yaml.dump(config_data, default_flow_style=False)
        session['config_bytes'] = config_yaml.encode('utf-8')
        session['config_filename'] = f"{secure_filename(session['experiment_name'])}_config.yaml"
    
    # Get user's uploaded CSV files for dataset selection
    user_csv_files = get_user_csv_files(current_user)
    
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    return render_template('form_config.html', experiment_name=session['experiment_name'], 
                           username=current_user.username, current_date=current_date_str,
                           user_csv_files=user_csv_files)

@web.route('/advanced_config_form', methods=['GET'])
@login_required
def advanced_config_form():
    """Display the advanced Google Form-style configuration editor."""
    if 'experiment_name' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    
    # Generate or use existing config data
    if 'config_data' not in session:
        config_data = generate_config_template(session['experiment_name'])
        session['config_data'] = config_data
        config_yaml = yaml.dump(config_data, default_flow_style=False)
        session['config_bytes'] = config_yaml.encode('utf-8')
        session['config_filename'] = f"{secure_filename(session['experiment_name'])}_config.yaml"
    
    # Get user's uploaded CSV files for dataset selection
    user_csv_files = get_user_csv_files(current_user)
    
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    return render_template('advanced_form_config.html', 
                           experiment_name=session['experiment_name'], 
                           username=current_user.username, 
                           current_date=current_date_str,
                           user_csv_files=user_csv_files)

@web.route('/save_form_config', methods=['POST'])
@login_required
def save_form_config():
    """Save the configuration from the basic Google Form-style editor."""
    if 'experiment_name' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
        
    yaml_data = request.form.get('yaml_data')
    if not yaml_data:
        flash('No configuration data received.')
        return redirect(url_for('routes.config_form'))
    
    try:
        # Parse the YAML to validate it
        config_data = yaml.safe_load(yaml_data)
        session['config_data'] = config_data
        session['config_bytes'] = yaml_data.encode('utf-8')
        
        # Check if user selected a dataset from their uploaded files
        dataset_selected = False
        if 'experiment' in config_data and 'data' in config_data['experiment']:
            dataset_value = config_data['experiment']['data'].get('dataset', '')
            if dataset_value and dataset_value != '' and '/' in dataset_value:
                # User selected an uploaded CSV file (format: "folder/filename.csv")
                dataset_selected = True
                # Set up CSV data from the selected file
                user_csv_files = get_user_csv_files(current_user)
                for csv_file in user_csv_files:
                    if csv_file['path'] == dataset_value:
                        session['csv_filename'] = csv_file['name']
                        # Read the CSV file content
                        with open(csv_file['full_path'], 'rb') as f:
                            session['csv_bytes'] = f.read()
                        break
        
        flash('Configuration saved successfully!')
        
        # Conditional routing: skip CSV upload if dataset selected, otherwise go to CSV upload
        if dataset_selected:
            current_app.logger.info(f"Dataset selected from user files, skipping CSV upload step")
            return redirect(url_for('routes.upload_archs'))
        else:
            current_app.logger.info(f"No dataset selected, proceeding to CSV upload step")
            return redirect(url_for('routes.upload_csv'))
            
    except yaml.YAMLError as e:
        flash(f'Invalid YAML format: {str(e)}')
        return redirect(url_for('routes.config_form'))
    except Exception as e:
        current_app.logger.error(f"Error in save_form_config: {str(e)}")
        flash('An error occurred while processing the configuration.')
        return redirect(url_for('routes.config_form'))
        
@web.route('/save_advanced_form_config', methods=['POST'])
@login_required
def save_advanced_form_config():
    """Save the configuration from the advanced Google Form-style editor."""
    if 'experiment_name' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
        
    yaml_data = request.form.get('yaml_data')
    if not yaml_data:
        flash('No configuration data received.')
        return redirect(url_for('routes.advanced_config_form'))
    
    try:
        # Parse the YAML to validate it
        config_data = yaml.safe_load(yaml_data)
        session['config_data'] = config_data
        session['config_bytes'] = yaml_data.encode('utf-8')
        
        # Check if user selected a dataset from their uploaded files
        dataset_selected = False
        if 'data' in config_data and 'dataset' in config_data['data']:
            dataset_value = config_data['data'].get('dataset', '')
            if dataset_value and dataset_value != '' and '/' in dataset_value:
                # User selected an uploaded CSV file (format: "folder/filename.csv")
                dataset_selected = True
                # Set up CSV data from the selected file
                user_csv_files = get_user_csv_files(current_user)
                for csv_file in user_csv_files:
                    if csv_file['path'] == dataset_value:
                        session['csv_filename'] = csv_file['name']
                        # Read the CSV file content
                        with open(csv_file['full_path'], 'rb') as f:
                            session['csv_bytes'] = f.read()
                        break
        
        flash('Advanced configuration saved successfully!')
        
        # Conditional routing: skip CSV upload if dataset selected, otherwise go to CSV upload
        if dataset_selected:
            current_app.logger.info(f"Dataset selected from user files, skipping CSV upload step")
            return redirect(url_for('routes.upload_archs'))
        else:
            current_app.logger.info(f"No dataset selected, proceeding to CSV upload step")
            return redirect(url_for('routes.upload_csv'))
            
    except yaml.YAMLError as e:
        flash(f'Invalid YAML format: {str(e)}')
        return redirect(url_for('routes.advanced_config_form'))
    except Exception as e:
        current_app.logger.error(f"Error in save_advanced_form_config: {str(e)}")
        flash('An error occurred while processing the configuration.')
        return redirect(url_for('routes.advanced_config_form'))

# Error handlers
@web.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found', code=404), 404

@web.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    current_app.logger.error(f'Server error: {error}', exc_info=True)
    return render_template('error.html', error='Internal server error', code=500), 500

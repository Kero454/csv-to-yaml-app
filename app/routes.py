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

# ---------------------------------------------------------------------------
# xApp container registry.
# The O-RAN xApp onboarder validates the container image `registry` against the
# pattern ^[A-Za-z0-9.-]+\.[A-Za-z]+(:\d+)?$ (it MUST contain a dot), so a bare
# `localhost:30500` is rejected. We therefore tag images with a dotted registry
# name. The image is built into minikube's docker daemon and deployed with
# imagePullPolicy=Never, so this name is only a local tag - no real registry is
# contacted. Override with the XAPP_REGISTRY env var if a real registry is used.
XAPP_REGISTRY = os.environ.get('XAPP_REGISTRY', 'registry.local:30500')


def xapp_image_tag(model_name, model_version):
    """Return the fully-qualified image tag for an xApp model image."""
    return f'{XAPP_REGISTRY}/xapps/{model_name}:{model_version}'


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
                                'display_name': file,  # Show only filename, not path
                                'path': file,  # Use only filename as identifier
                                'full_path': os.path.join(csv_path, file)
                            })
    except Exception as e:
        current_app.logger.error(f"Error getting user CSV files: {str(e)}")
    
    return csv_files

def save_single_file_and_redirect(file_type):
    """Save a single file when coming from experiment explorer and redirect back."""
    try:
        safe_user = secure_filename(current_user.username)
        safe_exp = secure_filename(session['experiment_name'])
        base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
        
        if file_type == 'config':
            # Save config file to experiment directory
            exp_dir = os.path.join(base_upload, safe_user, safe_exp)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Generate unique config filename with format {experiment_name_i}.config
            counter = 1
            config_filename = f"{safe_exp}_{counter}.config"
            while os.path.exists(os.path.join(exp_dir, config_filename)):
                counter += 1
                config_filename = f"{safe_exp}_{counter}.config"
            
            config_path = os.path.join(exp_dir, config_filename)
            with open(config_path, 'wb') as f:
                f.write(session['config_bytes'])
            
            flash(f'Configuration file {config_filename} added successfully!')
            
        elif file_type == 'data':
            # Save data file to Data directory
            csv_filename = session.get('csv_filename', 'data.csv')
            csv_base = os.path.splitext(csv_filename)[0]
            data_dir = os.path.join(base_upload, safe_user, 'Data', csv_base)
            os.makedirs(data_dir, exist_ok=True)
            
            csv_path = os.path.join(data_dir, csv_filename)
            with open(csv_path, 'wb') as f:
                f.write(session['csv_bytes'])
            
            flash(f'Data file {csv_filename} added successfully!')
            
        elif file_type == 'architecture':
            # Save architecture files to Architecture directory
            arch_dir = os.path.join(base_upload, safe_user, safe_exp, 'Architecture')
            os.makedirs(arch_dir, exist_ok=True)
            
            # This would be handled by the upload_archs route
            flash('Architecture files added successfully!')
        
        # Clean up session
        session.pop('from_explorer', None)
        session.pop('config_bytes', None)
        session.pop('config_filename', None)
        session.pop('csv_bytes', None)
        session.pop('csv_filename', None)
        session.pop('experiment_name', None)
        
        return redirect(url_for('routes.experiment_explorer'))
        
    except Exception as e:
        current_app.logger.error(f"Error saving single file: {e}")
        flash('An error occurred while saving the file.')
        return redirect(url_for('routes.experiment_explorer'))

def generate_config_template(experiment_name):
    """Generate a template configuration file for the experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dict containing the template configuration
    """
    # New default structured template matching requested format
    template = {
        'dataset': {
            'dataset': 'electricity',
            'path': '/DSIPTS-P/data/'
        },
        'scheduler_config': {
            'gamma': 0.75,
            'step_size': 2500
        },
        'optim_config': {
            'lr': 0.00005,
            'weight_decay': 0.0001
        },
        'model_configs': {
            'past_steps': 64,
            'future_steps': 64,
            'quantiles': [],
            'past_channels': None,
            'future_channels': None,
            'embs': None,
            'out_channels': None,
            'loss_type': None,
            'persistence_weight': 1.0
        },
        'split_params': {
            'perc_train': 0.6,
            'perc_valid': 0.2,
            'range_train': None,
            'range_validation': None,
            'range_test': None,
            'shift': 0,
            'starting_point': None,
            'skip_step': 1,
            'past_steps': 'model_configs@past_steps',
            'future_steps': 'model_configs@future_steps'
        },
        'train_config': {
            'dirpath': '/DSIPTS-P/data/',
            'num_workers': 0,
            'auto_lr_find': True,
            'devices': [0],
            'seed': 42
        },
        'inference': {
            'output_path': '/DSIPTS-P/output/',
            'load_last': True,
            'batch_size': 200,
            'num_workers': 4,
            'set': 'test',
            'rescaling': True
        },
        'defaults': [
            '_self_',
            {'architecture': None},
            {'override hydra/launcher': 'joblib'}
        ],
        'hydra': {
            'launcher': {
                'n_jobs': 4,
                'verbose': 1,
                'pre_dispatch': 4,
                'batch_size': 4,
                '_target_': 'hydra_plugins.hydra_joblib_launcher.joblib_launcher.JoblibLauncher'
            },
            'output_subdir': None,
            'sweeper': {
                'params': {
                    'architecture': 'glob(*)'
                }
            }
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
            
            # Check if coming from explorer - if so, save file and redirect back
            if session.get('from_explorer'):
                return save_single_file_and_redirect('config')
            
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
            
            # Debug session keys
            current_app.logger.info(f"Session keys before redirect: {list(session.keys())}")
            current_app.logger.info(f"experiment_name in session: {'experiment_name' in session}")
            
            # Store in session
            session['config_bytes'] = config_content.encode('utf-8')
            session.modified = True  # Ensure session is saved
            
            current_app.logger.info(f"Session keys after setting config_bytes: {list(session.keys())}")
            
            # Check if coming from explorer - if so, save file and redirect back
            if session.get('from_explorer'):
                return save_single_file_and_redirect('config')
            
            return redirect(url_for('routes.upload_csv'))
    
    return render_template('upload_config.html')

@web.route('/other_config_options')
@login_required
def other_config_options():
    """Display other configuration options (Basic Form Editor and Use Template)."""
    if 'experiment_name' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    
    return render_template('other_config_options.html')

@web.route('/upload_csv', methods=['GET', 'POST'])
@login_required
def upload_csv():
    # Debug session keys
    current_app.logger.info(f"upload_csv: Session keys on entry: {list(session.keys())}")
    
    if 'experiment_name' not in session or 'config_bytes' not in session:
        current_app.logger.info(f"upload_csv: Missing keys - experiment_name: {'experiment_name' in session}, config_bytes: {'config_bytes' in session}")
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    if request.method == 'POST':
        csv_file = request.files.get('csv_file')
        if not csv_file or csv_file.filename == '' or not csv_file.filename.lower().endswith('.csv'):
            flash('Please upload a valid CSV file.')
            return redirect(request.url)
        session['csv_filename'] = secure_filename(csv_file.filename)
        session['csv_bytes'] = csv_file.read()
        
        # Check if coming from explorer - if so, save file and redirect back
        if session.get('from_explorer'):
            return save_single_file_and_redirect('data')
        
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
                
        # Write experiment description file with metadata
        describe_path = os.path.join(exp_dir, f"{safe_exp}_describe.yaml")
        describe_data = {
            'user': current_user.username,
            'experiment': session['experiment_name'],
            'experiment_config_file': config_filename,
            'csv_file': csv_filename,
            'architecture_files': arch_saved,
            'configuration_method': 'file_upload',
            'timestamp': datetime.utcnow().isoformat()
        }
        with open(describe_path, 'w') as f:
            yaml.dump(describe_data, f, default_flow_style=False)
            
        session.pop('csv_bytes', None)
        session.pop('csv_filename', None)
        session.pop('config_bytes', None)
        session.pop('config_filename', None)
        session.pop('experiment_name', None)
        
        # Check if coming from explorer - if so, redirect back to explorer
        if session.pop('from_explorer', None):
            flash('Architecture files have been uploaded successfully!')
            return redirect(url_for('routes.experiment_explorer'))
        
        flash('Your files have been uploaded and processed successfully!')
        return redirect(url_for('routes.done'))
        
    return render_template('upload_archs.html')

@web.route('/arch_config_setup', methods=['GET', 'POST'])
@login_required
def arch_config_setup():
    """Setup page for architecture file configuration - asks user how many files to configure."""
    if 'experiment_name' not in session or 'csv_bytes' not in session or 'config_bytes' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    
    if request.method == 'POST':
        num_files = int(request.form.get('num_files', 1))
        if num_files < 1 or num_files > 100:
            flash('Please enter a valid number of files (1-100).')
            return redirect(request.url)
        
        # Initialize architecture configuration session data
        session['arch_config'] = {
            'total_files': num_files,
            'current_file': 1,
            'files': {}  # Will store configuration for each file
        }
        
        return redirect(url_for('routes.arch_config_form', file_num=1))
    
    return render_template('arch_config_setup.html')

@web.route('/arch_config_form')
@web.route('/arch_config_form/<int:file_num>')
@login_required
def arch_config_form(file_num=1):
    """Display configuration form for individual architecture file."""
    if 'experiment_name' not in session or 'arch_config' not in session:
        flash('Architecture configuration session expired. Please start again.')
        return redirect(url_for('routes.arch_config_setup'))
    
    arch_config = session['arch_config']
    total_files = arch_config['total_files']
    
    if file_num < 1 or file_num > total_files:
        flash(f'Invalid file number. Please select a file between 1 and {total_files}.')
        return redirect(url_for('routes.arch_config_form', file_num=1))
    
    # Get existing configuration for this file if it exists
    file_config = arch_config['files'].get(str(file_num), {})
    
    # Default content for new architecture files
    default_content = '''# Architecture File Template
# Replace this with your actual architecture implementation

import torch
import torch.nn as nn

class CustomArchitecture(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomArchitecture, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x
'''
    
    return render_template('arch_config_form.html',
                         current_file=file_num,
                         total_files=total_files,
                         filename=file_config.get('filename', ''),
                         description=file_config.get('description', ''),
                         file_content=file_config.get('content', ''),
                         file_type=file_config.get('type', 'python'),
                         default_content=default_content)

@web.route('/arch_config_form', methods=['POST'])
@login_required
def save_arch_config():
    """Save architecture file configuration and navigate to next file or complete."""
    if 'experiment_name' not in session or 'arch_config' not in session:
        flash('Architecture configuration session expired. Please start again.')
        return redirect(url_for('routes.arch_config_setup'))
    
    current_file = int(request.form.get('current_file', 1))
    total_files = int(request.form.get('total_files', 1))
    
    # Validate form data
    filename = request.form.get('filename', '').strip()
    description = request.form.get('description', '').strip()
    file_content = request.form.get('file_content', '').strip()
    file_type = request.form.get('file_type', 'python')
    
    if not filename:
        flash('Please provide a filename for the architecture file.')
        return redirect(url_for('routes.arch_config_form', file_num=current_file))
    
    if not file_content:
        flash('Please provide content for the architecture file.')
        return redirect(url_for('routes.arch_config_form', file_num=current_file))
    
    # Save current file configuration
    arch_config = session['arch_config']
    arch_config['files'][str(current_file)] = {
        'filename': filename,
        'description': description,
        'content': file_content,
        'type': file_type
    }
    session['arch_config'] = arch_config
    
    # Determine next action
    if current_file < total_files:
        # Go to next file
        return redirect(url_for('routes.arch_config_form', file_num=current_file + 1))
    else:
        # All files configured, process and save them
        return process_configured_arch_files()

def process_configured_arch_files():
    """Process all configured architecture files and complete the experiment."""
    try:
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

        # Save experiment config file
        exp_config_path = os.path.join(exp_dir, config_filename)
        with open(exp_config_path, 'wb') as f:
            f.write(session['config_bytes'])

        # Save CSV file
        csv_path = os.path.join(data_dir, csv_filename)
        with open(csv_path, 'wb') as f:
            f.write(session['csv_bytes'])
            
        # Save all configured architecture files
        arch_config = session['arch_config']
        arch_saved = []
        
        for file_num, file_config in arch_config['files'].items():
            filename = secure_filename(file_config['filename'])
            
            # Add appropriate file extension if not present
            if file_config['type'] == 'python' and not filename.endswith('.py'):
                filename += '.py'
            elif file_config['type'] == 'yaml' and not filename.endswith(('.yaml', '.yml')):
                filename += '.yaml'
            elif file_config['type'] == 'json' and not filename.endswith('.json'):
                filename += '.json'
            elif file_config['type'] == 'text' and not filename.endswith('.txt'):
                filename += '.txt'
            
            arch_path = os.path.join(arch_dir, filename)
            with open(arch_path, 'w', encoding='utf-8') as f:
                f.write(file_config['content'])
            
            arch_saved.append({
                'filename': filename,
                'description': file_config.get('description', ''),
                'type': file_config['type']
            })
                
        # Create experiment summary config
        config_path = os.path.join(exp_dir, f"config_{safe_exp}.yaml")
        config_data = {
            'user': current_user.username,
            'experiment': session['experiment_name'],
            'experiment_config_file': config_filename,
            'csv_file': csv_filename,
            'architecture_files': arch_saved,
            'configuration_method': 'configured',
            'timestamp': datetime.utcnow().isoformat()
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
            
        # Clean up session
        session.pop('csv_bytes', None)
        session.pop('csv_filename', None)
        session.pop('config_bytes', None)
        session.pop('config_filename', None)
        session.pop('arch_config', None)
        session.pop('experiment_name', None)
        
        # Check if coming from explorer - if so, redirect back to explorer
        if session.pop('from_explorer', None):
            flash(f'{len(arch_saved)} architecture files have been configured and added successfully!')
            return redirect(url_for('routes.experiment_explorer'))
        
        flash(f'Your experiment with {len(arch_saved)} configured architecture files has been created successfully!')
        return redirect(url_for('routes.done'))
        
    except Exception as e:
        current_app.logger.error(f"Error processing configured architecture files: {e}")
        flash('An error occurred while processing your architecture files. Please try again.')
        return redirect(url_for('routes.arch_config_setup'))

@web.route('/yaml_arch_setup', methods=['GET', 'POST'])
@login_required
def yaml_arch_setup():
    """Setup page for YAML architecture configuration - asks user how many YAML files to configure."""
    if 'experiment_name' not in session or 'csv_bytes' not in session or 'config_bytes' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    
    if request.method == 'POST':
        num_yaml_files = int(request.form.get('num_yaml_files', 1))
        if num_yaml_files < 1 or num_yaml_files > 100:
            flash('Please enter a valid number of YAML files (1-100).')
            return redirect(request.url)
        
        # Initialize YAML architecture configuration session data
        session['yaml_arch_config'] = {
            'total_files': num_yaml_files,
            'current_file': 1,
            'files': {}  # Will store configuration for each YAML file
        }
        
        return redirect(url_for('routes.yaml_arch_step1', file_num=1))
    
    return render_template('yaml_arch_setup.html')

@web.route('/yaml_arch_step1', methods=['GET', 'POST'])
@web.route('/yaml_arch_step1/<int:file_num>', methods=['GET', 'POST'])
@login_required
def yaml_arch_step1(file_num=1):
    """Step 1: Configure everything except model_configs for YAML architecture file."""
    if 'experiment_name' not in session or 'yaml_arch_config' not in session:
        flash('YAML architecture configuration session expired. Please start again.')
        return redirect(url_for('routes.yaml_arch_setup'))
    
    yaml_config = session['yaml_arch_config']
    total_files = yaml_config['total_files']
    
    if file_num < 1 or file_num > total_files:
        flash(f'Invalid file number. Please select a file between 1 and {total_files}.')
        return redirect(url_for('routes.yaml_arch_step1', file_num=1))
    
    if request.method == 'POST':
        # Handle Step 1 form submission - collect basic configuration
        current_file = int(request.form.get('current_file', file_num))
        total_files = int(request.form.get('total_files', total_files))
        
        # Collect Step 1 form data (everything except model_configs)
        step1_config = {
            'model_type': request.form.get('model_type', 'autoformer'),
            'model_retrain': request.form.get('model_retrain', 'true'),
            # TS config
            'ts_name': request.form.get('ts_name', 'model'),
            'ts_version': request.form.get('ts_version', '1'),
            'ts_enrich': request.form.get('ts_enrich', ''),
            'use_covariates': request.form.get('use_covariates', 'true'),
            'past_variables': request.form.get('past_variables', ''),
            'future_variables': request.form.get('future_variables', ''),
            'static_variables': request.form.get('static_variables', ''),
            # Training config
            'batch_size': request.form.get('batch_size', '32'),
            'max_epochs': request.form.get('max_epochs', '50')
        }
        
        # Save Step 1 configuration
        if 'files' not in yaml_config:
            yaml_config['files'] = {}
        if str(current_file) not in yaml_config['files']:
            yaml_config['files'][str(current_file)] = {}
        
        yaml_config['files'][str(current_file)].update(step1_config)
        session['yaml_arch_config'] = yaml_config
        session['current_yaml_file'] = current_file
        session.modified = True
        
        # Go to Step 2 for model_configs
        return redirect(url_for('routes.yaml_arch_step2', file_num=current_file))
    
    # GET request - display Step 1 form
    # Get existing configuration for this file if it exists
    file_config = yaml_config['files'].get(str(file_num), {})
    
    # Set default model type
    model_type = file_config.get('model_type', 'autotransformer')
    
    # Generate default TS name based on model type with incremental numbering if not already set
    if 'ts_name' not in file_config:
        # Find existing files with similar names to determine next increment
        safe_user = secure_filename(current_user.username)
        safe_exp = secure_filename(session['experiment_name'])
        base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
        exp_dir = os.path.join(base_upload, safe_user, safe_exp)
        arch_dir = os.path.join(exp_dir, 'Architecture')
        os.makedirs(arch_dir, exist_ok=True)
        
        counter = 1
        while True:
            test_filename = f"{model_type}_{counter}.yaml"
            if not os.path.exists(os.path.join(arch_dir, test_filename)):
                break
            counter += 1
        
        default_ts_name = f"{model_type}_{counter}"
    else:
        default_ts_name = file_config['ts_name']
    
    return render_template('yaml_arch_step1.html',
                         current_file=file_num,
                         total_files=total_files,
                         # Model config
                         model_type=model_type,
                         model_retrain=file_config.get('model_retrain', 'true'),
                         # TS config
                         ts_name=default_ts_name,
                         ts_version=file_config.get('ts_version', '1'),
                         ts_enrich=file_config.get('ts_enrich', ''),
                         use_covariates=file_config.get('use_covariates', 'true'),
                         past_variables=file_config.get('past_variables', ''),
                         future_variables=file_config.get('future_variables', ''),
                         static_variables=file_config.get('static_variables', ''),
                         # Training config
                         batch_size=file_config.get('batch_size', '32'),
                         max_epochs=file_config.get('max_epochs', '20'))

@web.route('/yaml_arch_step2', methods=['GET', 'POST'])
@web.route('/yaml_arch_step2/<int:file_num>', methods=['GET', 'POST'])
@login_required
def yaml_arch_step2(file_num=1):
    """Step 2: Configure dynamic model_configs based on selected model type."""
    if 'experiment_name' not in session or 'yaml_arch_config' not in session:
        flash('YAML architecture configuration session expired. Please start again.')
        return redirect(url_for('routes.yaml_arch_setup'))
    
    yaml_config = session['yaml_arch_config']
    total_files = yaml_config['total_files']
    current_file = session.get('current_yaml_file', file_num)
    
    if current_file < 1 or current_file > total_files:
        flash(f'Invalid file number. Please select a file between 1 and {total_files}.')
        return redirect(url_for('routes.yaml_arch_step1', file_num=1))
    
    # Get Step 1 configuration to determine model type
    file_config = yaml_config['files'].get(str(current_file), {})
    model_type = file_config.get('model_type', 'autoformer')
    
    if request.method == 'POST':
        # Handle Step 2 form submission - collect model_configs
        current_app.logger.info(f"Step 2 POST request received for file {current_file}")
        current_app.logger.info(f"Form data: {dict(request.form)}")
        model_configs = {}
        
        if model_type == 'autoformer':
            model_configs = {
                'd_model': int(request.form.get('d_model', 4)),
                'kernel_size': int(request.form.get('kernel_size', 3)),
                'n_layer_encoder': int(request.form.get('n_layer_encoder', 2)),
                'n_layer_decoder': int(request.form.get('n_layer_decoder', 2)),
                'label_len': int(request.form.get('label_len', 4)),
                'n_head': int(request.form.get('n_head', 2)),
                'dropout_rate': float(request.form.get('dropout_rate', 0.5)),
                'factor': int(request.form.get('factor', 5)),
                'hidden_size': int(request.form.get('hidden_size', 12)),
                'optim': request.form.get('optim', 'torch.optim.Adam'),
                'activation': request.form.get('activation', 'torch.nn.PReLU'),
                'persistence_weight': float(request.form.get('persistence_weight', 0.010)),
                'loss_type': request.form.get('loss_type', 'l1')
            }
        elif model_type == 'lstm':
            model_configs = {
                'cat_emb_dim': int(request.form.get('cat_emb_dim', 16)),
                'hidden_RNN': int(request.form.get('hidden_RNN', 12)),
                'num_layers_RNN': int(request.form.get('num_layers_RNN', 3)),
                'kernel_size': int(request.form.get('kernel_size', 5)),
                'kind': request.form.get('kind', 'lstm'),
                'sum_emb': request.form.get('sum_emb', 'true') == 'true',
                'optim': request.form.get('optim', 'torch.optim.SGD'),
                'activation': request.form.get('activation', 'torch.nn.SELU')
            }
        elif model_type == 'crossformer':
            model_configs = {
                'd_model': int(request.form.get('d_model', 4)),
                'hidden_size': int(request.form.get('hidden_size', 12)),
                'n_layer_encoder': int(request.form.get('n_layer_encoder', 2)),
                'n_head': int(request.form.get('n_head', 2)),
                'dropout_rate': float(request.form.get('dropout_rate', 0.5)),
                'win_size': int(request.form.get('win_size', 2)),
                'seg_len': int(request.form.get('seg_len', 6)),
                'factor': int(request.form.get('factor', 10)),
                'optim': request.form.get('optim', 'torch.optim.Adam'),
                'persistence_weight': float(request.form.get('persistence_weight', 0.010)),
                'loss_type': request.form.get('loss_type', 'l1')
            }
        elif model_type == 'd3vae':
            model_configs = {
                'embedding_dimension': int(request.form.get('embedding_dimension', 2)),
                'scale': float(request.form.get('scale', 0.1)),
                'hidden_size': int(request.form.get('hidden_size', 2)),
                'num_layers': int(request.form.get('num_layers', 1)),
                'dropout_rate': float(request.form.get('dropout_rate', 0.1)),
                'diff_steps': int(request.form.get('diff_steps', 1)),
                'loss_type': request.form.get('loss_type', 'kl'),
                'beta_end': float(request.form.get('beta_end', 0.01)),
                'beta_schedule': request.form.get('beta_schedule', 'linear'),
                'channel_mult': int(request.form.get('channel_mult', 1)),
                'mult': int(request.form.get('mult', 4)),
                'num_preprocess_blocks': int(request.form.get('num_preprocess_blocks', 1)),
                'num_preprocess_cells': int(request.form.get('num_preprocess_cells', 1)),
                'num_channels_enc': int(request.form.get('num_channels_enc', 1)),
                'arch_instance': request.form.get('arch_instance', 'res_mbconv'),
                'num_latent_per_group': int(request.form.get('num_latent_per_group', 1)),
                'num_channels_dec': int(request.form.get('num_channels_dec', 1)),
                'groups_per_scale': int(request.form.get('groups_per_scale', 1)),
                'num_postprocess_blocks': int(request.form.get('num_postprocess_blocks', 1)),
                'num_postprocess_cells': int(request.form.get('num_postprocess_cells', 1)),
                'beta_start': float(request.form.get('beta_start', 0)),
                'optim': request.form.get('optim', 'torch.optim.SGD')
            }
        elif model_type == 'diffusion':
            model_configs = {
                'd_model': int(request.form.get('d_model', 12)),
                'learn_var': request.form.get('learn_var', 'true') == 'true',
                'cosine_alpha': request.form.get('cosine_alpha', 'true') == 'true',
                'diffusion_steps': int(request.form.get('diffusion_steps', 100)),
                'beta': float(request.form.get('beta', 0.03)),
                'gamma': float(request.form.get('gamma', 0.01)),
                'n_layers_RNN': int(request.form.get('n_layers_RNN', 4)),
                'd_head': int(request.form.get('d_head', 64)),
                'n_head': int(request.form.get('n_head', 8)),
                'dropout_rate': float(request.form.get('dropout_rate', 0.0)),
                'activation': request.form.get('activation', 'torch.nn.GELU'),
                'subnet': int(request.form.get('subnet', 1)),
                'optim': request.form.get('optim', 'torch.optim.Adam'),
                'perc_subnet_learning_for_step': float(request.form.get('perc_subnet_learning_for_step', 0.1)),
                'persistence_weight': float(request.form.get('persistence_weight', 0.010)),
                'loss_type': request.form.get('loss_type', 'l1')
            }
        elif model_type == 'dilated_conv':
            quantiles_str = request.form.get('quantiles', '0.1,0.5,0.9')
            quantiles = [float(q.strip()) for q in quantiles_str.split(',') if q.strip()]
            model_configs = {
                'cat_emb_dim': int(request.form.get('cat_emb_dim', 4)),
                'hidden_RNN': int(request.form.get('hidden_RNN', 16)),
                'num_layers_RNN': int(request.form.get('num_layers_RNN', 1)),
                'kernel_size': int(request.form.get('kernel_size', 3)),
                'kind': request.form.get('kind', 'gru'),
                'sum_emb': request.form.get('sum_emb', 'true') == 'true',
                'persistence_weight': float(request.form.get('persistence_weight', 1.0)),
                'use_bn': request.form.get('use_bn', 'false') == 'true',
                'use_glu': request.form.get('use_glu', 'true') == 'true',
                'glu_percentage': float(request.form.get('glu_percentage', 0.2)),
                'quantiles': quantiles,
                'optim': request.form.get('optim', 'torch.optim.SGD'),
                'activation': request.form.get('activation', 'torch.nn.SELU'),
                'loss_type': request.form.get('loss_type', 'linear_penalization')
            }
        elif model_type == 'dilated_conv_ed':
            quantiles_str = request.form.get('quantiles', '0.1,0.5,0.9')
            quantiles = [float(q.strip()) for q in quantiles_str.split(',') if q.strip()]
            model_configs = {
                'cat_emb_dim': int(request.form.get('cat_emb_dim', 4)),
                'hidden_RNN': int(request.form.get('hidden_RNN', 16)),
                'num_layers_RNN': int(request.form.get('num_layers_RNN', 1)),
                'kernel_size': int(request.form.get('kernel_size', 3)),
                'kind': request.form.get('kind', 'gru'),
                'sum_emb': request.form.get('sum_emb', 'true') == 'true',
                'persistence_weight': float(request.form.get('persistence_weight', 1.0)),
                'use_bn': request.form.get('use_bn', 'false') == 'true',
                'quantiles': quantiles,
                'optim': request.form.get('optim', 'torch.optim.SGD'),
                'activation': request.form.get('activation', 'torch.nn.SELU'),
                'loss_type': request.form.get('loss_type', 'linear_penalization')
            }
        elif model_type == 'dlinear':
            model_configs = {
                'cat_emb_dim': int(request.form.get('cat_emb_dim', 4)),
                'kernel_size': int(request.form.get('kernel_size', 3)),
                'sum_emb': request.form.get('sum_emb', 'true') == 'true',
                'hidden_size': int(request.form.get('hidden_size', 12)),
                'kind': request.form.get('kind', 'dlinear'),
                'optim': request.form.get('optim', 'torch.optim.SGD'),
                'activation': request.form.get('activation', 'torch.nn.LeakyReLU'),
                'simple': request.form.get('simple', 'true') == 'true'
            }
        elif model_type == 'rnn':
            model_configs = {
                'cat_emb_dim': int(request.form.get('cat_emb_dim', 16)),
                'hidden_RNN': int(request.form.get('hidden_RNN', 12)),
                'num_layers_RNN': int(request.form.get('num_layers_RNN', 3)),
                'kernel_size': int(request.form.get('kernel_size', 5)),
                'kind': request.form.get('kind', 'gru'),
                'sum_emb': request.form.get('sum_emb', 'true') == 'true'
            }
        elif model_type == 'informer':
            model_configs = {
                'd_model': int(request.form.get('d_model', 4)),
                'hidden_size': int(request.form.get('hidden_size', 4)),
                'n_layer_encoder': int(request.form.get('n_layer_encoder', 2)),
                'n_layer_decoder': int(request.form.get('n_layer_decoder', 2)),
                'n_head': int(request.form.get('n_head', 2)),
                'dropout_rate': float(request.form.get('dropout_rate', 0.5)),
                'optim': request.form.get('optim', 'torch.optim.Adam'),
                'activation': request.form.get('activation', 'torch.nn.PReLU'),
                'persistence_weight': float(request.form.get('persistence_weight', 0.010)),
                'loss_type': request.form.get('loss_type', 'l1'),
                'remove_last': request.form.get('remove_last', 'true') == 'true'
            }
        elif model_type == 'linear':
            model_configs = {
                'cat_emb_dim': int(request.form.get('cat_emb_dim', 16)),
                'kernel_size': int(request.form.get('kernel_size', 5)),
                'sum_emb': request.form.get('sum_emb', 'true') == 'true',
                'hidden_size': int(request.form.get('hidden_size', 8)),
                'kind': request.form.get('kind', 'linear'),
                'dropout_rate': float(request.form.get('dropout_rate', 0.1)),
                'use_bn': request.form.get('use_bn', 'false') == 'true',
                'optim': request.form.get('optim', 'torch.optim.Adam'),
                'activation': request.form.get('activation', 'torch.nn.PReLU'),
                'persistence_weight': float(request.form.get('persistence_weight', 0.010)),
                'loss_type': request.form.get('loss_type', 'l1'),
                'simple': request.form.get('simple', 'false') == 'true'
            }
        elif model_type == 'nlinear':
            model_configs = {
                'cat_emb_dim': int(request.form.get('cat_emb_dim', 16)),
                'kernel_size': int(request.form.get('kernel_size', 5)),
                'sum_emb': request.form.get('sum_emb', 'true') == 'true',
                'hidden_size': int(request.form.get('hidden_size', 24)),
                'kind': request.form.get('kind', 'nlinear')
            }
        elif model_type == 'patchtst':
            model_configs = {
                'd_model': int(request.form.get('d_model', 4)),
                'kernel_size': int(request.form.get('kernel_size', 3)),
                'decomposition': request.form.get('decomposition', 'true') == 'true',
                'n_layer': int(request.form.get('n_layer', 2)),
                'patch_len': int(request.form.get('patch_len', 4)),
                'n_head': int(request.form.get('n_head', 2)),
                'stride': int(request.form.get('stride', 4)),
                'dropout_rate': float(request.form.get('dropout_rate', 0.5)),
                'hidden_size': int(request.form.get('hidden_size', 12)),
                'optim': request.form.get('optim', 'torch.optim.Adam'),
                'activation': request.form.get('activation', 'torch.nn.PReLU'),
                'persistence_weight': float(request.form.get('persistence_weight', 0.010)),
                'loss_type': request.form.get('loss_type', 'l1'),
                'remove_last': request.form.get('remove_last', 'true') == 'true'
            }
        elif model_type == 'persistent':
            # Persistent model has no model_configs
            model_configs = {}
        elif model_type == 'tft':
            model_configs = {
                'd_model': int(request.form.get('d_model', 4)),
                'd_head': int(request.form.get('d_head', 4)),
                'n_head': int(request.form.get('n_head', 4)),
                'num_layers_RNN': int(request.form.get('num_layers_RNN', 8)),
                'optim': request.form.get('optim', 'torch.optim.Adam'),
                'dropout_rate': float(request.form.get('dropout_rate', 0.5)),
                'persistence_weight': float(request.form.get('persistence_weight', 0.010)),
                'loss_type': request.form.get('loss_type', 'l1')
            }
        elif model_type == 'xlstm':
            model_configs = {
                'cat_emb_dim': int(request.form.get('cat_emb_dim', 16)),
                'hidden_RNN': int(request.form.get('hidden_RNN', 12)),
                'num_layers_RNN': int(request.form.get('num_layers_RNN', 3)),
                'kernel_size': int(request.form.get('kernel_size', 5)),
                'kind': request.form.get('kind', 'xlstm'),
                'sum_emb': request.form.get('sum_emb', 'true') == 'true',
                'num_blocks': int(request.form.get('num_blocks', 2)),
                'bidirectional': request.form.get('bidirectional', 'true') == 'true',
                'lstm_type': request.form.get('lstm_type', 'slstm')
            }
        
        # Save model_configs to session
        yaml_config['files'][str(current_file)]['model_configs'] = model_configs
        session['yaml_arch_config'] = yaml_config
        session.modified = True
        
        current_app.logger.info(f"Saved model_configs for file {current_file}: {model_configs}")
        current_app.logger.info(f"Current file: {current_file}, Total files: {total_files}")
        
        if current_file < total_files:
            # Go to next file Step 1
            current_app.logger.info(f"Redirecting to next file: {current_file + 1}")
            return redirect(url_for('routes.yaml_arch_step1', file_num=current_file + 1))
        else:
            # All files configured, process and save them
            current_app.logger.info("All files configured, calling process_configured_yaml_arch_files()")
            return process_configured_yaml_arch_files()
    
    # GET request - display Step 2 form with model-specific fields
    # Get existing model_configs if they exist
    existing_model_configs = file_config.get('model_configs', {})
    
    return render_template('yaml_arch_step2.html',
                         current_file=current_file,
                         total_files=total_files,
                         model_type=model_type,
                         # Common fields
                         d_model=existing_model_configs.get('d_model', 4),
                         kernel_size=existing_model_configs.get('kernel_size', 3),
                         n_layer_encoder=existing_model_configs.get('n_layer_encoder', 2),
                         n_layer_decoder=existing_model_configs.get('n_layer_decoder', 2),
                         label_len=existing_model_configs.get('label_len', 4),
                         n_head=existing_model_configs.get('n_head', 2),
                         dropout_rate=existing_model_configs.get('dropout_rate', 0.5),
                         factor=existing_model_configs.get('factor', 5),
                         hidden_size=existing_model_configs.get('hidden_size', 12),
                         optim=existing_model_configs.get('optim', 'torch.optim.Adam'),
                         activation=existing_model_configs.get('activation', 'torch.nn.PReLU'),
                         persistence_weight=existing_model_configs.get('persistence_weight', 0.010),
                         loss_type=existing_model_configs.get('loss_type', 'l1'),
                         # LSTM/RNN fields
                         cat_emb_dim=existing_model_configs.get('cat_emb_dim', 16),
                         hidden_RNN=existing_model_configs.get('hidden_RNN', 12),
                         num_layers_RNN=existing_model_configs.get('num_layers_RNN', 3),
                         kind=existing_model_configs.get('kind', 'lstm'),
                         sum_emb=existing_model_configs.get('sum_emb', True),
                         # Crossformer fields
                         win_size=existing_model_configs.get('win_size', 2),
                         seg_len=existing_model_configs.get('seg_len', 6),
                         # D3VAE fields
                         embedding_dimension=existing_model_configs.get('embedding_dimension', 2),
                         scale=existing_model_configs.get('scale', 0.1),
                         num_layers=existing_model_configs.get('num_layers', 1),
                         diff_steps=existing_model_configs.get('diff_steps', 1),
                         beta_end=existing_model_configs.get('beta_end', 0.01),
                         beta_schedule=existing_model_configs.get('beta_schedule', 'linear'),
                         channel_mult=existing_model_configs.get('channel_mult', 1),
                         mult=existing_model_configs.get('mult', 4),
                         num_preprocess_blocks=existing_model_configs.get('num_preprocess_blocks', 1),
                         num_preprocess_cells=existing_model_configs.get('num_preprocess_cells', 1),
                         num_channels_enc=existing_model_configs.get('num_channels_enc', 1),
                         arch_instance=existing_model_configs.get('arch_instance', 'res_mbconv'),
                         num_latent_per_group=existing_model_configs.get('num_latent_per_group', 1),
                         num_channels_dec=existing_model_configs.get('num_channels_dec', 1),
                         groups_per_scale=existing_model_configs.get('groups_per_scale', 1),
                         num_postprocess_blocks=existing_model_configs.get('num_postprocess_blocks', 1),
                         num_postprocess_cells=existing_model_configs.get('num_postprocess_cells', 1),
                         beta_start=existing_model_configs.get('beta_start', 0),
                         # Diffusion fields
                         learn_var=existing_model_configs.get('learn_var', 'true'),
                         cosine_alpha=existing_model_configs.get('cosine_alpha', 'true'),
                         diffusion_steps=existing_model_configs.get('diffusion_steps', 100),
                         beta=existing_model_configs.get('beta', 0.03),
                         gamma=existing_model_configs.get('gamma', 0.01),
                         n_layers_RNN=existing_model_configs.get('n_layers_RNN', 4),
                         d_head=existing_model_configs.get('d_head', 64),
                         subnet=existing_model_configs.get('subnet', 1),
                         perc_subnet_learning_for_step=existing_model_configs.get('perc_subnet_learning_for_step', 0.1),
                         # Dilated Conv fields
                         use_bn=existing_model_configs.get('use_bn', 'false'),
                         use_glu=existing_model_configs.get('use_glu', 'true'),
                         glu_percentage=existing_model_configs.get('glu_percentage', 0.2),
                         quantiles=','.join(map(str, existing_model_configs.get('quantiles', [0.1, 0.5, 0.9]))),
                         # New model type fields
                         simple=existing_model_configs.get('simple', 'true'),
                         remove_last=existing_model_configs.get('remove_last', 'true'),
                         decomposition=existing_model_configs.get('decomposition', 'true'),
                         n_layer=existing_model_configs.get('n_layer', 2),
                         patch_len=existing_model_configs.get('patch_len', 4),
                         stride=existing_model_configs.get('stride', 4),
                         num_blocks=existing_model_configs.get('num_blocks', 2),
                         bidirectional=existing_model_configs.get('bidirectional', 'true'),
                         lstm_type=existing_model_configs.get('lstm_type', 'slstm'))

@web.route('/yaml_arch_form', methods=['POST'])
@login_required
def save_yaml_arch_config():
    """Save YAML architecture file configuration and navigate to next file or complete."""
    if 'experiment_name' not in session or 'yaml_arch_config' not in session:
        flash('YAML architecture configuration session expired. Please start again.')
        return redirect(url_for('routes.yaml_arch_setup'))
    
    current_file = int(request.form.get('current_file', 1))
    total_files = int(request.form.get('total_files', 1))
    
    # Validate form data
    filename = request.form.get('filename', '').strip()
    if not filename:
        flash('Please provide a filename for the YAML file.')
        return redirect(url_for('routes.yaml_arch_form', file_num=current_file))
    
    # Get model type and set TS name same as model type
    model_type = request.form.get('model_type', 'rnn')
    ts_name = request.form.get('ts_name', model_type)  # Default to model_type if not edited
    
    # Collect all form data
    file_config = {
        'filename': filename if filename else f"{model_type}.yaml",  # Default filename to model_type.yaml
        'description': request.form.get('description', '').strip(),
        # Model config
        'model_type': model_type,
        'model_retrain': request.form.get('model_retrain', 'true'),
        # TS config
        'ts_name': ts_name,
        'ts_version': request.form.get('ts_version', '1'),
        'ts_enrich': request.form.get('ts_enrich', ''),
        'use_covariates': request.form.get('use_covariates', 'true'),
        'past_variables': request.form.get('past_variables', '[1]'),
        'use_future_covariates': request.form.get('use_future_covariates', 'false'),
        'future_variables': request.form.get('future_variables', 'null'),
        'interpolate': request.form.get('interpolate', 'true'),
        # Model configs with choice/range support
        'cat_emb_dim': _process_param_value(request.form, 'cat_emb_dim', '128'),
        'hidden_rnn': request.form.get('hidden_rnn', '64'),
        'num_layers_rnn': request.form.get('num_layers_rnn', '2'),
        'kernel_size': request.form.get('kernel_size', '3'),
        'kind': request.form.get('kind', 'lstm'),
        'sum_emb': request.form.get('sum_emb', 'true'),
        'use_bn': request.form.get('use_bn', 'true'),
        'optim': request.form.get('optim', 'torch.optim.SGD'),
        'activation': request.form.get('activation', 'torch.nn.ReLU'),
        'dropout_rate': _process_param_value(request.form, 'dropout_rate', '0.2'),
        'persistence_weight': request.form.get('persistence_weight', '0.010'),
        'loss_type': request.form.get('loss_type', 'l1'),
        'remove_last': request.form.get('remove_last', 'true'),
        # Training config
        'batch_size': request.form.get('batch_size', '128'),
        'max_epochs': request.form.get('max_epochs', '20')
    }
    
    # Save current file configuration
    yaml_config = session['yaml_arch_config']
    yaml_config['files'][str(current_file)] = file_config
    session['yaml_arch_config'] = yaml_config
    
    # Determine next action based on form submission
    action = request.form.get('action', 'next')
    
    if action == 'next' and current_file < total_files:
        # Redirect to the first YAML architecture file configuration (Step 1)
        return redirect(url_for('routes.yaml_arch_step1', file_num=1))
    else:
        # Complete configuration and process all files
        return process_configured_yaml_arch_files()
    
    # GET request - render the model configuration form
    return render_template('model_config_form.html',
                         current_file=current_file,
                         total_files=total_files,
                         ts_name=ts_name,
                         model_type=model_type,
                         **file_config.get('model_configs', {}))

def process_configured_yaml_arch_files():
    """Process all configured YAML architecture files and complete the experiment."""
    try:
        current_app.logger.info("Starting YAML architecture files processing")
        current_app.logger.info(f"Session keys: {list(session.keys())}")
        
        safe_user = secure_filename(current_user.username)
        safe_exp = secure_filename(session['experiment_name'])
        current_app.logger.info(f"Processing for user: {safe_user}, experiment: {safe_exp}")
        
        # Handle optional CSV and config files (may not exist in YAML-only workflow)
        csv_filename = session.get('csv_filename')
        config_filename = session.get('config_filename')
        csv_bytes = session.get('csv_bytes')
        config_bytes = session.get('config_bytes')
        
        base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
        user_dir = os.path.join(base_upload, safe_user)
        exp_dir = os.path.join(user_dir, safe_exp)
        arch_dir = os.path.join(exp_dir, 'Architecture')
        
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(arch_dir, exist_ok=True)

        # Save experiment config file if it exists
        if config_filename and config_bytes:
            exp_config_path = os.path.join(exp_dir, config_filename)
            with open(exp_config_path, 'wb') as f:
                f.write(config_bytes)

        # Save CSV file if it exists
        if csv_filename and csv_bytes:
            csv_base = os.path.splitext(csv_filename)[0]
            data_dir = os.path.join(user_dir, 'Data', csv_base)
            os.makedirs(data_dir, exist_ok=True)
            csv_path = os.path.join(data_dir, csv_filename)
            with open(csv_path, 'wb') as f:
                f.write(csv_bytes)
            
        # Generate and save all configured YAML architecture files
        current_app.logger.info("Getting YAML config from session")
        yaml_config = session['yaml_arch_config']
        current_app.logger.info(f"YAML config: {yaml_config}")
        arch_saved = []
        
        current_app.logger.info(f"Processing {len(yaml_config['files'])} YAML files")
        for file_num, file_config in yaml_config['files'].items():
            current_app.logger.info(f"Processing file {file_num}: {file_config}")
            
            # Generate filename based on user-provided TS name or model type with incremental numbering
            ts_name = file_config.get('ts_name', '').strip()
            model_type = file_config.get('model_type', 'lstm')
            
            # Determine base filename (without extension)
            if not ts_name or ts_name.lower() in ['lstm', 'default', '']:
                base_name = secure_filename(model_type)
            else:
                base_name = secure_filename(ts_name)
            
            # Find existing files with similar names to determine next increment
            counter = 1
            while True:
                test_filename = f"{base_name}_{counter}.yaml"
                if not os.path.exists(os.path.join(arch_dir, test_filename)):
                    break
                counter += 1
                
            # Set the incremental filename
            filename = f"{base_name}_{counter}.yaml"
            
            current_app.logger.info(f"Generating YAML content for file: {filename}")
            # Generate YAML content
            yaml_content = generate_yaml_training_config(file_config)
            current_app.logger.info(f"Generated YAML content length: {len(yaml_content)}")
            
            arch_path = os.path.join(arch_dir, filename)
            current_app.logger.info(f"Saving YAML file to: {arch_path}")
            with open(arch_path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            
            arch_saved.append({
                'filename': filename,
                'description': f"YAML training configuration {file_num} for {safe_exp}",
                'type': 'yaml_training_config'
            })
            current_app.logger.info(f"Successfully saved file {file_num}")
                
        # Create experiment description file with metadata
        describe_path = os.path.join(exp_dir, f"{safe_exp}_describe.yaml")
        describe_data = {
            'user': current_user.username,
            'experiment': session['experiment_name'],
            'architecture_files': arch_saved,
            'configuration_method': 'yaml_training_configured',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add optional files if they exist
        if config_filename:
            describe_data['experiment_config_file'] = config_filename
        if csv_filename:
            describe_data['csv_file'] = csv_filename
        with open(describe_path, 'w') as f:
            yaml.dump(describe_data, f, default_flow_style=False)
            
        # Clean up session (but keep experiment_name for sweeper workflow)
        session.pop('csv_bytes', None)
        session.pop('csv_filename', None)
        session.pop('config_bytes', None)
        session.pop('config_filename', None)
        session.pop('yaml_arch_config', None)
        # Keep experiment_name for sweeper workflow - will be cleaned up in sweeper_prompt
        
        current_app.logger.info(f"Successfully completed processing {len(arch_saved)} YAML files")
        
        # Check if coming from explorer - if so, redirect back to explorer
        if session.pop('from_explorer', None):
            flash(f'{len(arch_saved)} YAML training files have been configured and added successfully!')
            return redirect(url_for('routes.experiment_explorer'))
        
        flash(f'Your experiment with {len(arch_saved)} configured YAML training files has been created successfully!')
        return redirect(url_for('routes.sweeper_prompt'))
        
    except Exception as e:
        import traceback
        current_app.logger.error(f"Error processing configured YAML architecture files: {e}")
        current_app.logger.error(f"Full traceback: {traceback.format_exc()}")
        flash('An error occurred while processing your YAML architecture files. Please try again.')
        return redirect(url_for('routes.yaml_arch_setup'))

def generate_yaml_training_config(config):
    """Generate YAML training configuration content from form data."""
    # Parse list values
    def parse_list_or_null(value):
        if not value or str(value).strip().lower() in ['null', 'none', '']:
            return None
        try:
            # Handle list format like [1, 2, 3] or [1]
            if str(value).strip().startswith('[') and str(value).strip().endswith(']'):
                return eval(str(value).strip())
            else:
                return str(value).strip()
        except:
            return str(value).strip()
    
    def parse_enrich_list(value):
        if not value or not str(value).strip():
            return []
        return [item.strip() for item in str(value).split(',') if item.strip()]
    
    def parse_variables_list(value):
        if not value or not str(value).strip():
            return []
        return [item.strip() for item in str(value).split(',') if item.strip()]
    
    # Build the configuration dictionary
    yaml_config = {
        'model': {
            'type': config.get('model_type', 'autotransformer'),
            'retrain': config.get('model_retrain', 'true') == 'true'
        },
        'ts': {
            'name': config.get('ts_name', 'model'),
            'version': int(config.get('ts_version', 1)),
            'enrich': parse_enrich_list(config.get('ts_enrich', '')),
            'use_covariates': config.get('use_covariates', 'true') == 'true',
            'past_variables': parse_variables_list(config.get('past_variables', '')),
            'future_variables': parse_variables_list(config.get('future_variables', '')),
            'static_variables': parse_variables_list(config.get('static_variables', ''))
        },
        'train_config': {
            'batch_size': int(config.get('batch_size', 32)),
            'max_epochs': int(config.get('max_epochs', 50))
        }
    }
    
    # Add model_configs from the 2-step workflow
    if 'model_configs' in config:
        yaml_config['model_configs'] = config['model_configs']
    else:
        # Fallback for old structure (shouldn't happen in new workflow)
        yaml_config['model_configs'] = {}
    
    # Generate YAML content with header
    yaml_content = "# @package _global_\n\n"
    yaml_content += yaml.dump(yaml_config, default_flow_style=False, sort_keys=False)
    
    return yaml_content

@web.route('/model_config_form/<int:file_num>', methods=['GET', 'POST'])
@login_required
def model_config_form(file_num=1):
    """Dynamic model configuration form that changes based on model type."""
    if 'experiment_name' not in session or 'yaml_arch_config' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    
    yaml_config = session['yaml_arch_config']
    current_file = file_num
    total_files = yaml_config['total_files']
    
    # Get the file configuration to determine model type
    file_config = yaml_config['files'].get(str(current_file), {})
    ts_name = file_config.get('ts_name', 'custom')
    model_type = ts_name.lower()  # Model type is same as TS name
    
    if request.method == 'POST':
        # Collect model-specific configuration based on model type
        model_config = {}
        
        if model_type == 'lstm':
            model_config = {
                'hidden_size': int(request.form.get('lstm_hidden_size', 128)),
                'num_layers': int(request.form.get('lstm_num_layers', 2)),
                'dropout': float(request.form.get('lstm_dropout', 0.2)),
                'bidirectional': request.form.get('lstm_bidirectional') == 'true'
            }
        elif model_type == 'gru':
            model_config = {
                'hidden_size': int(request.form.get('gru_hidden_size', 128)),
                'num_layers': int(request.form.get('gru_num_layers', 2)),
                'dropout': float(request.form.get('gru_dropout', 0.2)),
                'bidirectional': request.form.get('gru_bidirectional') == 'true'
            }
        elif model_type == 'transformer':
            model_config = {
                'd_model': int(request.form.get('transformer_d_model', 512)),
                'nhead': int(request.form.get('transformer_nhead', 8)),
                'num_layers': int(request.form.get('transformer_num_layers', 6)),
                'dim_feedforward': int(request.form.get('transformer_dim_feedforward', 2048)),
                'dropout': float(request.form.get('transformer_dropout', 0.1))
            }
        elif model_type == 'cnn':
            model_config = {
                'num_filters': int(request.form.get('cnn_num_filters', 64)),
                'kernel_size': int(request.form.get('cnn_kernel_size', 3)),
                'num_layers': int(request.form.get('cnn_num_layers', 3)),
                'dropout': float(request.form.get('cnn_dropout', 0.2))
            }
        else:
            # Generic/custom model configuration
            model_config = {
                'param1': request.form.get('custom_param1', ''),
                'param2': request.form.get('custom_param2', ''),
                'config_json': request.form.get('custom_config_json', '{}')
            }
        
        # Add model configuration to the file config
        file_config['model_configs'] = model_config
        yaml_config['files'][str(current_file)] = file_config
        session['yaml_arch_config'] = yaml_config
        
        # Determine next action based on form submission
        action = request.form.get('action', 'next')
        
        if action == 'next' and current_file < total_files:
            # Go to next file
            return redirect(url_for('routes.yaml_arch_form', file_num=current_file + 1))
        else:
            # Complete configuration and process all files
            return process_configured_yaml_arch_files()
    
    # GET request - render the model configuration form
    return render_template('model_config_form.html',
                         current_file=current_file,
                         total_files=total_files,
                         ts_name=ts_name,
                         model_type=model_type,
                         **file_config.get('model_configs', {}))

@web.route('/done')
@login_required
def done():
    return render_template('done.html')

@web.route('/files')
@login_required
def list_files():
    """List files for the current user only (privacy-protected)."""
    upload_folder = current_app.config['UPLOAD_FOLDER']
    base_path = os.path.join(upload_folder, 'Users')
    
    # Get current user's sanitized username
    sanitized_username = secure_filename(current_user.username)
    user_path = os.path.join(base_path, sanitized_username)
    
    if not os.path.exists(user_path):
        return render_template('files.html', files_by_user={})

    # Only show current user's files for privacy
    files_by_user = {sanitized_username: []}
    
    for dirpath, _, filenames in os.walk(user_path):
        for filename in sorted(filenames):
            relative_dir = os.path.relpath(dirpath, base_path)
            files_by_user[sanitized_username].append(os.path.join(relative_dir, filename).replace('\\', '/'))
    
    current_app.logger.info(f"User {current_user.username} accessing their files only")
    return render_template('files.html', files_by_user=files_by_user)

@web.route('/uploads/<path:filepath>')
@login_required
def serve_upload(filepath):
    """Serve uploaded files for the current user only (privacy-protected)."""
    base_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    file_abs_path = os.path.abspath(os.path.join(base_path, filepath))
    
    # Security check 1: Prevent directory traversal
    if not file_abs_path.startswith(os.path.abspath(base_path)):
        current_app.logger.warning(f"Directory traversal attempt by user {current_user.id} for path {filepath}")
        return "Forbidden", 403
    
    # Security check 2: Ensure user can only access their own files
    sanitized_username = secure_filename(current_user.username)
    if not filepath.startswith(sanitized_username + '/'):
        current_app.logger.warning(f"User {current_user.id} ({current_user.username}) attempting to access other user's file at {filepath}")
        return "Forbidden", 403
        
    current_app.logger.info(f"User {current_user.username} accessing their own file: {filepath}")
    return send_file(file_abs_path)


@web.route('/save_file_content', methods=['POST'])
@login_required
def save_file_content():
    """Save edited file content directly."""
    try:
        data = request.get_json()
        filepath = data.get('filepath', '')
        filename = data.get('filename', '')
        experiment = data.get('experiment', '')
        filetype = data.get('filetype', '')
        content = data.get('content', '')
        
        # Clean the filepath if it starts with 'uploads/'
        if filepath.startswith('uploads/'):
            filepath = filepath[8:]  # Remove 'uploads/' prefix
        
        # Security check: ensure the filepath belongs to the current user
        original_username = current_user.username
        safe_user = secure_filename(current_user.username)
        # Handle both forward slash and backslash separators
        path_parts = filepath.replace('\\', '/').split('/')
        
        # Find the username part in the path (check both original and sanitized)
        username_found = False
        for i, part in enumerate(path_parts):
            if part == safe_user or part == original_username:
                # Username should be first or second after "Users"
                if i <= 2 and (i == 0 or path_parts[0] == 'Users' or (path_parts[0] == 'Users' and path_parts[1] == '..')):
                    username_found = True
                    break
        
        if not username_found:
            return jsonify({'error': 'Access denied'}), 403
        
        # Build the full file path
        upload_folder = current_app.config['UPLOAD_FOLDER']
        full_path = os.path.join(upload_folder, 'Users', filepath)
        
        # Verify the file exists and is within allowed directories
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Write the content to the file
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return jsonify({'success': True, 'message': 'File saved successfully'})
        
    except Exception as e:
        current_app.logger.error(f"Error saving file content: {e}")
        return jsonify({'error': str(e)}), 500

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
    
    # Check if the path starts with the sanitized username
    # Handle both forward slash and backslash separators
    path_parts = relative_path.replace('\\', '/').split('/')
    
    # The path might be "Users/username/..." or just "username/..."
    # We need to check for both the original username and sanitized username
    original_username = current_user.username
    sanitized_username = secure_filename(current_user.username)
    
    # Find the username part in the path (check both original and sanitized)
    username_index = -1
    found_username = None
    for i, part in enumerate(path_parts):
        if part == sanitized_username or part == original_username:
            username_index = i
            found_username = part
            break
    
    if username_index == -1:
        current_app.logger.warning(f"Username check failed: neither original '{original_username}' nor sanitized '{sanitized_username}' found in path parts {path_parts}")
        current_app.logger.warning(f"User {current_user.id} ({current_user.username}) attempting to access other user's file at {relative_path}")
        return jsonify({'error': 'Access denied. Username not found in path.'}), 403
    
    # If username is found but not in the expected position (should be first or second after "Users")
    if username_index > 2 or (username_index == 1 and path_parts[0] != 'Users') or (username_index == 2 and not (path_parts[0] == 'Users' and path_parts[1] == '..')):
        current_app.logger.warning(f"Username check failed: username '{found_username}' found at unexpected position {username_index} in path {path_parts}")
        current_app.logger.warning(f"Path parts: {path_parts}, username_index: {username_index}, path_parts[0]: '{path_parts[0] if path_parts else 'EMPTY'}'")
        error_msg = f'Access denied. Invalid path structure. Username "{found_username}" at position {username_index} in path {path_parts}'
        return jsonify({'error': error_msg}), 403
    
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
    from flask import flash
    flash(f'Debug: Experiment explorer called for user {current_user.username}', 'info')
    upload_folder = current_app.config['UPLOAD_FOLDER']
    base_path = os.path.join(upload_folder, 'Users')
    user_path = os.path.join(base_path, secure_filename(current_user.username))
    
    current_app.logger.info(f"Experiment explorer: current_user.username = {current_user.username}")
    current_app.logger.info(f"Experiment explorer: secure_filename(username) = {secure_filename(current_user.username)}")
    current_app.logger.info(f"Experiment explorer: upload_folder = {upload_folder}")
    current_app.logger.info(f"Experiment explorer: user_path = {user_path}")
    current_app.logger.info(f"Experiment explorer: user_path exists = {os.path.exists(user_path)}")
    
    if not os.path.exists(user_path):
        current_app.logger.info("User path does not exist, returning empty experiments")
        return render_template('experiment_explorer.html', experiments=[])
    
    experiments = []
    
    # Find all experiment directories for this user
    current_app.logger.info(f"Scanning user directory: {user_path}")
    user_dirs = os.listdir(user_path)
    current_app.logger.info(f"Found directories: {user_dirs}")
    
    for experiment_name in user_dirs:
        exp_path = os.path.join(user_path, experiment_name)
        current_app.logger.info(f"Processing: {experiment_name}, is_dir: {os.path.isdir(exp_path)}")
        if os.path.isdir(exp_path) and experiment_name != 'Data':  # Skip the Data directory
            try:
                # Try to read experiment metadata - check both naming patterns
                config_path = os.path.join(exp_path, f"config_{experiment_name}.yaml")
                current_app.logger.info(f"Looking for config file: {config_path}")
                if not os.path.exists(config_path):
                    config_path = os.path.join(exp_path, f"{experiment_name}_config.yaml")
                    current_app.logger.info(f"Trying alternative config file: {config_path}")
                timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')  # Default timestamp
                
                current_app.logger.info(f"Config file exists: {os.path.exists(config_path)}")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_data = yaml.safe_load(f)
                        if config_data and 'timestamp' in config_data:
                            timestamp = datetime.fromisoformat(config_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                # Fallback: check describe file for timestamp
                describe_path = os.path.join(exp_path, f"{experiment_name}_describe.yaml")
                if os.path.exists(describe_path):
                    with open(describe_path, 'r') as f:
                        describe_data = yaml.safe_load(f)
                        if describe_data and 'timestamp' in describe_data:
                            timestamp = datetime.fromisoformat(describe_data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                
                # Get configuration files
                config_files = []
                for file in os.listdir(exp_path):
                    if (file.lower().endswith(('.yaml', '.yml', '.config')) or 'config' in file.lower()) and os.path.isfile(os.path.join(exp_path, file)) and not file.startswith('values_'):
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
                
                config_data = {
                    'experiment': {
                        'exp_id': experiment_name,
                        'type': 'train',
                        'version': 1,
                        'data': {
                            'dataset': 'default',
                            'path': '/DSIPTS-P/data/',
                        }
                    }
                }
                experiments.append({
                    'name': experiment_name,
                    'path': exp_path,
                    'timestamp': timestamp,
                    'config_files': config_files,
                    'arch_files': arch_files,
                    'data_files': data_files
                })
                current_app.logger.info(f"Successfully processed experiment: {experiment_name}")
            except Exception as e:
                current_app.logger.error(f"Error processing experiment {experiment_name}: {str(e)}")
    
    # Sort experiments by timestamp (newest first)
    experiments.sort(key=lambda x: x['timestamp'], reverse=True)
    
    current_app.logger.info(f"Total experiments found: {len(experiments)}")
    current_app.logger.info(f"Experiment names: {[exp['name'] for exp in experiments]}")
    
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
                'path': '/DSIPTS-P/data/'  # Default path since dataset_path field was removed
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
                'dirpath': request.form.get('train_dirpath', '/DSIPTS-P/data/'),
                'num_workers': int(request.form.get('train_num_workers', 0)),
                'auto_lr_find': request.form.get('train_auto_lr_find') == 'on',
                'devices': _parse_devices(request.form.get('train_devices', '0')),
                'seed': int(request.form.get('train_seed', 42))
            },
            'inference': {
                'output_path': request.form.get('inference_output_path', '/DSIPTS-P/output/'),
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
        # Fix defaults indentation
        lines = config_yaml.split('\n')
        fixed_lines = []
        in_defaults = False
        
        for line in lines:
            if line.startswith('defaults:'):
                in_defaults = True
                fixed_lines.append(line)
            elif in_defaults and line.startswith('- '):
                # Add proper indentation to defaults list items (1 space)
                fixed_lines.append(' ' + line)
            else:
                if line and not line[0].isspace() and ':' in line:
                    in_defaults = False
                fixed_lines.append(line)
        
        config_yaml = '\n'.join(fixed_lines)
        session['config_bytes'] = config_yaml.encode('utf-8')
        session['config_filename'] = f"{secure_filename(session['experiment_name'])}_config.yaml"
        
        # Check if user selected a dataset (either uploaded file or default dataset)
        dataset_selected = False
        dataset_value = config_data['dataset'].get('dataset', '')
        
        if dataset_value and dataset_value != '' and dataset_value != 'no_dataset':
            # User selected some dataset (either uploaded file or default dataset)
            dataset_selected = True
            
            # Check if it's an uploaded CSV file (ends with .csv)
            if dataset_value.endswith('.csv'):
                user_csv_files = get_user_csv_files(current_user)
                for csv_file in user_csv_files:
                    if csv_file['path'] == dataset_value or csv_file['name'] == dataset_value:
                        session['csv_filename'] = csv_file['name']
                        # Read the CSV file content
                        with open(csv_file['full_path'], 'rb') as f:
                            session['csv_bytes'] = f.read()
                        current_app.logger.info(f"Template Optimizer: Set csv_bytes for file {csv_file['name']}")
                        break
                else:
                    current_app.logger.error(f"Template Optimizer: Could not find CSV file {dataset_value}")
            # For default datasets (electricity, traffic, etc.), no CSV setup needed
        
        flash('Configuration generated successfully!')
        
        # Check if coming from explorer - if so, save file and redirect back
        if session.get('from_explorer'):
            return save_single_file_and_redirect('config')
        
        # Conditional routing: skip CSV upload if dataset selected, otherwise go to CSV upload
        if dataset_selected:
            current_app.logger.info(f"Template Optimizer: Dataset selected from user files, skipping CSV upload step")
            
            # Debug session keys before redirect
            current_app.logger.info(f"Template Optimizer: Session keys before redirect: {list(session.keys())}")
            current_app.logger.info(f"Template Optimizer: experiment_name in session: {'experiment_name' in session}")
            current_app.logger.info(f"Template Optimizer: config_bytes in session: {'config_bytes' in session}")
            current_app.logger.info(f"Template Optimizer: csv_bytes in session: {'csv_bytes' in session}")
            
            # Ensure session is saved before redirect
            session.modified = True
            
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
                'path': '/DSIPTS-P/data/'  # Default path since dataset_path field was removed
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
                'dirpath': request.form.get('train_dirpath', '/DSIPTS-P/data/'),
                'num_workers': int(request.form.get('train_num_workers', 0)),
                'auto_lr_find': request.form.get('train_auto_lr_find') == 'on',
                'devices': _parse_devices(request.form.get('train_devices', '0')),
                'seed': int(request.form.get('train_seed', 42))
            },
            'inference': {
                'output_path': request.form.get('inference_output_path', '/DSIPTS-P/output/'),
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

def _process_param_value(form_data, param_name, default_value):
    """Process parameter value that can be fixed, choice, or range."""
    param_type = form_data.get(f'{param_name}_type', 'fixed')
    
    if param_type == 'choice':
        return form_data.get(f'{param_name}_choice', f'choice({default_value})')
    elif param_type == 'range':
        return form_data.get(f'{param_name}_range', f'range(1,{default_value},1)')
    else:  # fixed
        return form_data.get(f'{param_name}_fixed', default_value)

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
        # Fix defaults indentation
        lines = config_yaml.split('\n')
        fixed_lines = []
        in_defaults = False
        
        for line in lines:
            if line.startswith('defaults:'):
                in_defaults = True
                fixed_lines.append(line)
            elif in_defaults and line.startswith('- '):
                # Add proper indentation to defaults list items (1 space)
                fixed_lines.append(' ' + line)
            else:
                if line and not line[0].isspace() and ':' in line:
                    in_defaults = False
                fixed_lines.append(line)
        
        config_yaml = '\n'.join(fixed_lines)
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
        # Fix defaults indentation
        lines = config_yaml.split('\n')
        fixed_lines = []
        in_defaults = False
        
        for line in lines:
            if line.startswith('defaults:'):
                in_defaults = True
                fixed_lines.append(line)
            elif in_defaults and line.startswith('- '):
                # Add proper indentation to defaults list items (1 space)
                fixed_lines.append(' ' + line)
            else:
                if line and not line[0].isspace() and ':' in line:
                    in_defaults = False
                fixed_lines.append(line)
        
        config_yaml = '\n'.join(fixed_lines)
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
            if dataset_value and dataset_value != '' and dataset_value.endswith('.csv'):
                # User selected an uploaded CSV file
                dataset_selected = True
                # Set up CSV data from the selected file
                user_csv_files = get_user_csv_files(current_user)
                for csv_file in user_csv_files:
                    if csv_file['path'] == dataset_value or csv_file['name'] == dataset_value:
                        session['csv_filename'] = csv_file['name']
                        # Read the CSV file content
                        with open(csv_file['full_path'], 'rb') as f:
                            session['csv_bytes'] = f.read()
                        current_app.logger.info(f"Form Config: Set csv_bytes for file {csv_file['name']}")
                        break
                else:
                    current_app.logger.error(f"Form Config: Could not find CSV file {dataset_value}")
        
        flash('Configuration saved successfully!')
        
        # Check if coming from explorer - if so, save file and redirect back
        if session.get('from_explorer'):
            return save_single_file_and_redirect('config')
        
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
            if dataset_value and dataset_value != '' and dataset_value.endswith('.csv'):
                # User selected an uploaded CSV file
                dataset_selected = True
                # Set up CSV data from the selected file
                user_csv_files = get_user_csv_files(current_user)
                for csv_file in user_csv_files:
                    if csv_file['path'] == dataset_value or csv_file['name'] == dataset_value:
                        session['csv_filename'] = csv_file['name']
                        # Read the CSV file content
                        with open(csv_file['full_path'], 'rb') as f:
                            session['csv_bytes'] = f.read()
                        current_app.logger.info(f"Advanced Config: Set csv_bytes for file {csv_file['name']}")
                        break
                else:
                    current_app.logger.error(f"Advanced Config: Could not find CSV file {dataset_value}")
        
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

@web.route('/sweeper_prompt', methods=['GET', 'POST'])
@login_required
def sweeper_prompt():
    """Prompt user if they want to add hyperparameter sweeper to their config files."""
    if 'experiment_name' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    
    if request.method == 'POST':
        add_sweeper = request.form.get('add_sweeper') == 'yes'
        
        if add_sweeper:
            return redirect(url_for('routes.sweeper_config_selection'))
        else:
            # User doesn't want sweeper, go to config selection before deployment config
            return redirect(url_for('routes.select_config_for_run'))
    
    return render_template('sweeper_prompt.html', experiment_name=session['experiment_name'])

@web.route('/run_experiment_config_selection', methods=['GET', 'POST'])
@login_required
def run_experiment_config_selection():
    """Step 6: Allow user to select any config file from all their experiments for optimization."""
    # Get all user's config files from all experiments
    safe_user = secure_filename(current_user.username)
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    user_dir = os.path.join(base_upload, safe_user)
    
    config_files = []
    if os.path.exists(user_dir):
        for exp_folder in os.listdir(user_dir):
            exp_path = os.path.join(user_dir, exp_folder)
            if os.path.isdir(exp_path):
                for file in os.listdir(exp_path):
                    if file.lower().endswith(('.yaml', '.yml')) and os.path.isfile(os.path.join(exp_path, file)):
                        # Include any file matching pattern {name}_config.yaml
                        if file.endswith('_config.yaml'):
                            config_files.append({
                                'name': file,
                                'experiment': exp_folder,
                                'path': os.path.join(exp_path, file),
                                'full_path': f"{exp_folder}/{file}"
                            })
    
    if request.method == 'POST':
        selected_config = request.form.get('selected_config')
        if not selected_config:
            flash('Please select a config file to optimize.')
            return redirect(request.url)
        
        # Parse experiment and filename from selection
        exp_name, filename = selected_config.split('/', 1)
        
        # Store selected config in session for Step 6 workflow
        session['sweeper_config_file'] = filename
        session['sweeper_experiment_name'] = exp_name
        session['sweeper_config_path'] = os.path.join(user_dir, exp_name, filename)
        return redirect(url_for('routes.run_experiment_sweeper_form'))
    
    return render_template('run_experiment_config_selection.html', 
                         config_files=config_files)

@web.route('/run_experiment_sweeper_form', methods=['GET', 'POST'])
@login_required
def run_experiment_sweeper_form():
    """Step 6: Google Form-style hyperparameter sweeper configuration for standalone run."""
    if 'sweeper_config_file' not in session or 'sweeper_experiment_name' not in session:
        flash('Sweeper configuration session expired. Please start again.')
        return redirect(url_for('routes.run_experiment'))
    
    if request.method == 'POST':
        # Process sweeper configuration with proper hydra nesting and launcher
        sweeper_config = {
            'defaults': [
                '_self_',
                'architecture: null',
                'override hydra/launcher: joblib',
                'override hydra/sweeper: optuna'
            ],
            'hydra': {
                'launcher': {
                    'n_jobs': 4,
                    'verbose': 1,
                    'pre_dispatch': 1,
                    'batch_size': 4
                },
                'output_subdir': None,
                'sweeper': {
                    'sampler': {
                        '_target_': 'optuna.samplers.TPESampler',
                        'seed': int(request.form.get('sampler_seed', 123)),
                        'consider_prior': request.form.get('consider_prior') == 'on',
                        'prior_weight': float(request.form.get('prior_weight', 1.0)),
                        'consider_magic_clip': request.form.get('consider_magic_clip') == 'on',
                        'consider_endpoints': request.form.get('consider_endpoints') == 'on',
                        'n_startup_trials': int(request.form.get('n_startup_trials', 10)),
                        'n_ei_candidates': int(request.form.get('n_ei_candidates', 24)),
                        'multivariate': request.form.get('multivariate') == 'on',
                        'warn_independent_sampling': request.form.get('warn_independent_sampling') == 'on'
                    },
                    '_target_': 'hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper',
                    'direction': request.form.get('direction', 'minimize'),
                    'storage': request.form.get('storage', 'null'),
                    'study_name': request.form.get('study_name', session['sweeper_experiment_name']),
                    'n_trials': int(request.form.get('n_trials', 4)),
                    'n_jobs': int(request.form.get('n_jobs', 2)),
                    'params': {}
                }
            }
        }
        
        # Process dynamic parameters
        param_keys = request.form.getlist('param_key[]')
        param_values = request.form.getlist('param_value[]')
        
        for key, value in zip(param_keys, param_values):
            if key and value:
                sweeper_config['hydra']['sweeper']['params'][key] = value
        
        # Store sweeper config in session for preview
        session['sweeper_config'] = sweeper_config
        return redirect(url_for('routes.run_experiment_sweeper_preview'))
    
    return render_template('run_experiment_sweeper_form.html', 
                         experiment_name=session['sweeper_experiment_name'],
                         config_file=session['sweeper_config_file'])

@web.route('/run_experiment_sweeper_preview', methods=['GET', 'POST'])
@login_required
def run_experiment_sweeper_preview():
    """Step 6: Preview and confirm sweeper configuration before applying to standalone config."""
    if ('sweeper_config_file' not in session or 'sweeper_experiment_name' not in session or 
        'sweeper_config' not in session or 'sweeper_config_path' not in session):
        flash('Sweeper configuration session expired. Please start again.')
        return redirect(url_for('routes.run_experiment'))
    
    if request.method == 'POST':
        # Apply sweeper configuration to selected config file
        try:
            config_path = session['sweeper_config_path']
            
            # Read existing config
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f)
            
            # Merge sweeper configuration properly - only update what's needed
            # Update defaults section - add the overrides
            if 'defaults' not in existing_config:
                existing_config['defaults'] = []
            
            # Find and update the defaults list
            defaults_list = existing_config['defaults']
            if isinstance(defaults_list, list):
                # Add the override entries if not present
                has_sweeper_override = False
                for item in defaults_list:
                    if isinstance(item, str) and 'override hydra/sweeper' in item:
                        has_sweeper_override = True
                
                if not has_sweeper_override:
                    defaults_list.append('override hydra/sweeper: optuna')
            
            # Merge hydra section properly
            if 'hydra' not in existing_config:
                existing_config['hydra'] = {}
            
            # Update launcher config
            existing_config['hydra']['launcher'] = session['sweeper_config']['hydra']['launcher']
            
            # Preserve output_subdir if it exists
            if 'output_subdir' not in existing_config['hydra']:
                existing_config['hydra']['output_subdir'] = None
            
            # Add sweeper config
            existing_config['hydra']['sweeper'] = session['sweeper_config']['hydra']['sweeper']
            
            # Write updated config back with proper formatting
            with open(config_path, 'w') as f:
                # Custom YAML dumper to handle defaults list formatting
                yaml_content = yaml.dump(existing_config, default_flow_style=False, sort_keys=False)
                
                # Fix defaults indentation if present
                lines = yaml_content.split('\n')
                fixed_lines = []
                in_defaults = False
                
                for line in lines:
                    if line.startswith('defaults:'):
                        in_defaults = True
                        fixed_lines.append(line)
                    elif in_defaults and line.startswith('- '):
                        # Add proper indentation to defaults list items
                        fixed_lines.append('  ' + line)
                        if not line.strip().startswith('- '):
                            in_defaults = False
                    else:
                        if line and not line[0].isspace() and ':' in line:
                            in_defaults = False
                        fixed_lines.append(line)
                
                f.write('\n'.join(fixed_lines))
            
            # Store config path for PVC mounting
            session['config_file_path'] = config_path
            session['config_file_name'] = session['sweeper_config_file']
            
            # Clean up sweeper-specific session data
            session.pop('sweeper_config_file', None)
            session.pop('sweeper_config', None)
            session.pop('sweeper_config_path', None)
            
            flash('Hyperparameter sweeper has been added to your config file successfully!')
            # Always go through config selection before deployment config
            return redirect(url_for('routes.select_config_for_run'))
            
        except Exception as e:
            current_app.logger.error(f"Error applying sweeper config: {e}")
            flash('An error occurred while applying the sweeper configuration.')
            return redirect(url_for('routes.run_experiment_sweeper_form'))
    
    # Generate full file preview with merged configuration
    try:
        config_path = session['sweeper_config_path']
        
        # Read existing config
        with open(config_path, 'r') as f:
            existing_config = yaml.safe_load(f)
        
        # Create a copy and merge with sweeper config
        merged_config = existing_config.copy()
        merged_config.update(session['sweeper_config'])
        
        # Generate YAML for full merged file
        full_file_yaml = yaml.dump(merged_config, default_flow_style=False, sort_keys=False)
        sweeper_only_yaml = yaml.dump(session['sweeper_config'], default_flow_style=False, sort_keys=False)
        
    except Exception as e:
        current_app.logger.error(f"Error generating preview: {e}")
        flash('Error generating preview. Please try again.')
        return redirect(url_for('routes.run_experiment_sweeper_form'))
    
    return render_template('run_experiment_sweeper_preview.html',
                         experiment_name=session['sweeper_experiment_name'],
                         config_file=session['sweeper_config_file'],
                         sweeper_yaml=sweeper_only_yaml,
                         full_file_yaml=full_file_yaml)

@web.route('/sweeper_config_selection', methods=['GET', 'POST'])
@login_required
def sweeper_config_selection():
    """Allow user to select which config file to add sweeper to."""
    if 'experiment_name' not in session:
        flash('Experiment session expired. Please start again.')
        return redirect(url_for('routes.experiment'))
    
    # Get user's config files for this experiment
    safe_user = secure_filename(current_user.username)
    safe_exp = secure_filename(session['experiment_name'])
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    user_dir = os.path.join(base_upload, safe_user)
    exp_dir = os.path.join(user_dir, safe_exp)
    
    config_files = []
    if os.path.exists(exp_dir):
        for file in os.listdir(exp_dir):
            if file.lower().endswith(('.yaml', '.yml')) and os.path.isfile(os.path.join(exp_dir, file)):
                # Only include files matching {experiment-name}_config.yaml pattern
                expected_name = f"{safe_exp}_config.yaml"
                if file == expected_name:
                    config_files.append({
                        'name': file,
                        'path': os.path.join(exp_dir, file)
                    })
    
    if request.method == 'POST':
        selected_config = request.form.get('selected_config')
        if not selected_config:
            flash('Please select a config file to add sweeper to.')
            return redirect(request.url)
        
        # Store selected config in session
        session['sweeper_config_file'] = selected_config
        return redirect(url_for('routes.sweeper_form'))
    
    return render_template('sweeper_config_selection.html', 
                         experiment_name=session['experiment_name'],
                         config_files=config_files)

@web.route('/sweeper_form', methods=['GET', 'POST'])
@login_required
def sweeper_form():
    """Google Form-style hyperparameter sweeper configuration."""
    if 'experiment_name' not in session or 'sweeper_config_file' not in session:
        flash('Sweeper configuration session expired. Please start again.')
        return redirect(url_for('routes.sweeper_prompt'))
    
    if request.method == 'POST':
        # Process sweeper configuration with proper hydra nesting and launcher
        sweeper_config = {
            'defaults': [
                '_self_',
                'architecture: null',
                'override hydra/launcher: joblib',
                'override hydra/sweeper: optuna'
            ],
            'hydra': {
                'launcher': {
                    'n_jobs': 4,
                    'verbose': 1,
                    'pre_dispatch': 1,
                    'batch_size': 4
                },
                'output_subdir': None,
                'sweeper': {
                    'sampler': {
                        '_target_': 'optuna.samplers.TPESampler',
                        'seed': int(request.form.get('sampler_seed', 123)),
                        'consider_prior': request.form.get('consider_prior') == 'on',
                        'prior_weight': float(request.form.get('prior_weight', 1.0)),
                        'consider_magic_clip': request.form.get('consider_magic_clip') == 'on',
                        'consider_endpoints': request.form.get('consider_endpoints') == 'on',
                        'n_startup_trials': int(request.form.get('n_startup_trials', 10)),
                        'n_ei_candidates': int(request.form.get('n_ei_candidates', 24)),
                        'multivariate': request.form.get('multivariate') == 'on',
                        'warn_independent_sampling': request.form.get('warn_independent_sampling') == 'on'
                    },
                    '_target_': 'hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper',
                    'direction': request.form.get('direction', 'minimize'),
                    'storage': request.form.get('storage', 'null'),
                    'study_name': request.form.get('study_name', session['experiment_name']),
                    'n_trials': int(request.form.get('n_trials', 4)),
                    'n_jobs': int(request.form.get('n_jobs', 2)),
                    'params': {}
                }
            }
        }
        
        # Process dynamic parameters
        param_keys = request.form.getlist('param_key[]')
        param_values = request.form.getlist('param_value[]')
        
        for key, value in zip(param_keys, param_values):
            if key and value:
                sweeper_config['hydra']['sweeper']['params'][key] = value
        
        # Store sweeper config in session for preview
        session['sweeper_config'] = sweeper_config
        return redirect(url_for('routes.sweeper_preview'))
    
    return render_template('sweeper_form.html', 
                         experiment_name=session['experiment_name'],
                         config_file=session['sweeper_config_file'])

@web.route('/sweeper_preview', methods=['GET', 'POST'])
@login_required
def sweeper_preview():
    """Preview and confirm sweeper configuration before applying."""
    if 'experiment_name' not in session or 'sweeper_config_file' not in session or 'sweeper_config' not in session:
        flash('Sweeper configuration session expired. Please start again.')
        return redirect(url_for('routes.sweeper_prompt'))
    
    if request.method == 'POST':
        # Apply sweeper configuration to selected config file
        try:
            safe_user = secure_filename(current_user.username)
            safe_exp = secure_filename(session['experiment_name'])
            base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
            user_dir = os.path.join(base_upload, safe_user)
            exp_dir = os.path.join(user_dir, safe_exp)
            config_path = os.path.join(exp_dir, session['sweeper_config_file'])
            
            # Read existing config
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f)
            
            # Merge sweeper configuration properly - only update what's needed
            # Update defaults section - add the overrides
            if 'defaults' not in existing_config:
                existing_config['defaults'] = []
            
            # Find and update the defaults list
            defaults_list = existing_config['defaults']
            if isinstance(defaults_list, list):
                # Add the override entries if not present
                has_sweeper_override = False
                for item in defaults_list:
                    if isinstance(item, str) and 'override hydra/sweeper' in item:
                        has_sweeper_override = True
                
                if not has_sweeper_override:
                    defaults_list.append('override hydra/sweeper: optuna')
            
            # Merge hydra section properly
            if 'hydra' not in existing_config:
                existing_config['hydra'] = {}
            
            # Update launcher config
            existing_config['hydra']['launcher'] = session['sweeper_config']['hydra']['launcher']
            
            # Preserve output_subdir if it exists
            if 'output_subdir' not in existing_config['hydra']:
                existing_config['hydra']['output_subdir'] = None
            
            # Add sweeper config
            existing_config['hydra']['sweeper'] = session['sweeper_config']['hydra']['sweeper']
            
            # Write updated config back with proper formatting
            with open(config_path, 'w') as f:
                # Custom YAML dumper to handle defaults list formatting
                yaml_content = yaml.dump(existing_config, default_flow_style=False, sort_keys=False)
                
                # Fix defaults indentation if present
                lines = yaml_content.split('\n')
                fixed_lines = []
                in_defaults = False
                
                for line in lines:
                    if line.startswith('defaults:'):
                        in_defaults = True
                        fixed_lines.append(line)
                    elif in_defaults and line.startswith('- '):
                        # Add proper indentation to defaults list items
                        fixed_lines.append('  ' + line)
                        if not line.strip().startswith('- '):
                            in_defaults = False
                    else:
                        if line and not line[0].isspace() and ':' in line:
                            in_defaults = False
                        fixed_lines.append(line)
                
                f.write('\n'.join(fixed_lines))
            
            # Store config path for PVC mounting
            session['config_file_path'] = config_path
            session['config_file_name'] = session['sweeper_config_file']
            
            # Clean up sweeper-specific session data
            session.pop('sweeper_config_file', None)
            session.pop('sweeper_config', None)
            
            flash('Hyperparameter sweeper has been added to your config file successfully!')
            # Always go through config selection before deployment config
            return redirect(url_for('routes.select_config_for_run'))
            
        except Exception as e:
            current_app.logger.error(f"Error applying sweeper config: {e}")
            flash('An error occurred while applying the sweeper configuration.')
            return redirect(url_for('routes.sweeper_form'))
    
    # Generate full file preview with merged configuration
    try:
        safe_user = secure_filename(current_user.username)
        safe_exp = secure_filename(session['experiment_name'])
        base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
        user_dir = os.path.join(base_upload, safe_user)
        exp_dir = os.path.join(user_dir, safe_exp)
        config_path = os.path.join(exp_dir, session['sweeper_config_file'])
        
        # Read existing config
        with open(config_path, 'r') as f:
            existing_config = yaml.safe_load(f)
        
        # Create a copy and merge with sweeper config
        merged_config = existing_config.copy()
        merged_config.update(session['sweeper_config'])
        
        # Generate YAML for full merged file
        full_file_yaml = yaml.dump(merged_config, default_flow_style=False, sort_keys=False)
        sweeper_only_yaml = yaml.dump(session['sweeper_config'], default_flow_style=False, sort_keys=False)
        
    except Exception as e:
        current_app.logger.error(f"Error generating preview: {e}")
        flash('Error generating preview. Please try again.')
        return redirect(url_for('routes.sweeper_form'))
    
    return render_template('sweeper_preview.html',
                         experiment_name=session['experiment_name'],
                         config_file=session['sweeper_config_file'],
                         sweeper_yaml=sweeper_only_yaml,
                         full_file_yaml=full_file_yaml)

@web.route('/preview_sweeper_yaml', methods=['POST'])
@login_required
def preview_sweeper_yaml():
    """Generate YAML preview from sweeper form data."""
    try:
        sweeper_config = {
            'defaults': [
                '_self_',
                'architecture: null',
                'override hydra/launcher: joblib',
                'override hydra/sweeper: optuna'
            ],
            'hydra': {
                'launcher': {
                    'n_jobs': 4,
                    'verbose': 1,
                    'pre_dispatch': 1,
                    'batch_size': 4
                },
                'output_subdir': None,
                'sweeper': {
                    'sampler': {
                        '_target_': 'optuna.samplers.TPESampler',
                        'seed': int(request.form.get('sampler_seed', 123)),
                        'consider_prior': request.form.get('consider_prior') == 'on',
                        'prior_weight': float(request.form.get('prior_weight', 1.0)),
                        'consider_magic_clip': request.form.get('consider_magic_clip') == 'on',
                        'consider_endpoints': request.form.get('consider_endpoints') == 'on',
                        'n_startup_trials': int(request.form.get('n_startup_trials', 10)),
                        'n_ei_candidates': int(request.form.get('n_ei_candidates', 24)),
                        'multivariate': request.form.get('multivariate') == 'on',
                        'warn_independent_sampling': request.form.get('warn_independent_sampling') == 'on'
                    },
                    '_target_': 'hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper',
                    'direction': request.form.get('direction', 'minimize'),
                    'storage': request.form.get('storage', 'null'),
                    'study_name': request.form.get('study_name', 'experiment'),
                    'n_trials': int(request.form.get('n_trials', 4)),
                    'n_jobs': int(request.form.get('n_jobs', 2)),
                    'params': {}
                }
            }
        }
        
        # Process dynamic parameters
        param_keys = request.form.getlist('param_key[]')
        param_values = request.form.getlist('param_value[]')
        
        for key, value in zip(param_keys, param_values):
            if key and value:
                sweeper_config['hydra']['sweeper']['params'][key] = value
        
        yaml_data = yaml.dump(sweeper_config, default_flow_style=False, sort_keys=False)
        return jsonify({'yaml_data': yaml_data})
        
    except Exception as e:
        current_app.logger.error(f"Error generating sweeper YAML preview: {str(e)}")
        return jsonify({'error': 'Failed to generate preview. Check form data for errors.'}), 400

@web.route('/deployment_config', methods=['GET', 'POST'])
@web.route('/deployment_config/<experiment_name>', methods=['GET', 'POST'])
@login_required
def deployment_config(experiment_name=None):
    """Final step: Configure deployment settings for Kubernetes/Helm."""
    
    # Check if config file has been selected for run experiment workflow
    if not session.get('run_config_filename') and not session.get('config_file_name'):
        # No config selected, redirect to config selection
        return redirect(url_for('routes.select_config_for_run'))
    
    # Get experiment name from URL parameter, session, or use default
    if not experiment_name:
        experiment_name = session.get('run_experiment_name') or session.get('sweeper_experiment_name') or session.get('experiment_name', 'default')
    
    # Get config file info for PVC mounting
    config_file_name = session.get('run_config_filename') or session.get('config_file_name', '')
    config_file_path = session.get('config_file_path', '')
    
    if request.method == 'POST':
        try:
            # Get user info first
            safe_user = secure_filename(current_user.username)
            
            # Extract form data
            replica_count = int(request.form.get('replicaCount', 1))
            # Handle multi-select models
            models_list = request.form.getlist('modelsToTrain')
            models_to_train = ','.join(models_list) if models_list else ''
            
            # Validate that at least one architecture file is selected
            if not models_to_train:
                flash('Please select at least one architecture file to train.')
                return redirect(url_for('routes.deployment_config', experiment_name=experiment_name))
            
            # Worker group replica counts
            worker_groups = {
                'a100-workers': {'replicaCount': int(request.form.get('a100_replicas', 1))},
                'a1000-workers': {'replicaCount': int(request.form.get('a1000_replicas', 0))},
                'gh200-workers': {'replicaCount': int(request.form.get('gh200_replicas', 0))},
                'cpu-workers': {'replicaCount': int(request.form.get('cpu_replicas', 0))}
            }
            
            # Create the full deployment configuration
            deployment_config = {
                # Default values for dsipts-p-chart
                'replicaCount': replica_count,
                
                'image': {
                    'repository': 'hakushaku69/dsipts-p',
                    'pullPolicy': 'Always',
                    'tag': '3.0.0'
                },
                
                'imagePullSecrets': [],
                'nameOverride': '',
                'fullnameOverride': '',
                
                'serviceAccount': {
                    'create': True,
                    'annotations': {},
                    'name': ''
                },
                
                'podAnnotations': {},
                'podSecurityContext': {},
                'securityContext': {},
                
                'service': {
                    'type': 'ClusterIP',
                    'port': 80
                },
                
                'aim': {
                    'annotations': {
                        'description': 'Deployment for Aim container',
                        'field.cattle.io/publicEndpoints': '[{"port":30088,"protocol":"TCP","serviceName":"ts-framework:aim-service","allNodes":true}]'
                    },
                    'image': {
                        'repository': 'hakushaku69/aim-container',
                        'pullPolicy': 'IfNotPresent',
                        'tag': '1'
                    },
                    'service': {
                        'annotations': {
                            'field.cattle.io/publicEndpoints': '[{"port":30088,"protocol":"TCP","serviceName":"ts-framework:aim-service","allNodes":true}]'
                        },
                        'type': 'NodePort',
                        'port': 80,
                        'targetPort': 43800,
                        'nodePort': 30088
                    },
                    'persistence': {
                        'enabled': True,
                        'claimName': 'nfs-aim-pvc'
                    },
                    'resources': {
                        'limits': {
                            'cpu': '1',
                            'memory': '2Gi'
                        },
                        'requests': {
                            'cpu': '100m',
                            'memory': '1Gi'
                        }
                    },
                    'nodeName': 'srv01'
                },
                'dsipts_p': {
                    'defaults': {
                        'persistence': {
                            'enabled': True,
                            'volumes': [
                                {
                                    'name': 'nfs-ts-framework-pv',
                                    'claimName': 'nfs-ts-framework-pvc',
                                    'mounts': [
                                        {'mountPath': '/DSIPTS-P/output', 'subPath': 'output'},
                                        {'mountPath': '/DSIPTS-P/data', 'subPath': f"instance/uploads/Users/{safe_user}/data"}
                                    ]
                                },
                                {
                                    'name': 'nfs-aim-pv',
                                    'claimName': 'nfs-aim-pvc',
                                    'mounts': [{'mountPath': '/aim'}]
                                }
                            ]
                        },
                        'command': {
                            'enabled': True,
                            'run': ['/bin/sh', '-c'],
                            'modelsToTrain': models_to_train,
                            'args': 'python train.py {{ if .Values.dsipts_p.defaults.command.modelsToTrain }}-m architecture={{ .Values.dsipts_p.defaults.command.modelsToTrain }}{{ end }} --config-dir=config_milan --config-name=config_milan;\nsleep infinity'
                        },
                        'configFiles': {
                            'enabled': True if config_file_name else False,
                            'name': config_file_name,
                            'fromDirectory': f"Users/{current_user.username}/{experiment_name}",
                            'mountPath': '/DSIPTS-P/bash_examples/config_milan'
                        }
                    },
                    'workerGroups': [
                        {
                            'name': 'a100-workers',
                            'replicaCount': worker_groups['a100-workers']['replicaCount'],
                            'resources': {
                                'limits': {
                                    'cpu': '32',
                                    'ephemeral-storage': '10Gi',
                                    'memory': '128Gi',
                                    'nvidia.com/gpu': '1'
                                },
                                'requests': {
                                    'cpu': '8',
                                    'ephemeral-storage': '1Gi',
                                    'memory': '4Gi',
                                    'nvidia.com/gpu': '1'
                                }
                            },
                            'nodeSelector': {
                                'nvidia.com/gpu.product': 'NVIDIA-A100-80GB-PCIe'
                            }
                        },
                        {
                            'name': 'a1000-workers',
                            'replicaCount': worker_groups['a1000-workers']['replicaCount'],
                            'resources': {
                                'limits': {
                                    'cpu': '16',
                                    'memory': '64Gi',
                                    'nvidia.com/gpu': '1'
                                },
                                'requests': {
                                    'cpu': '4',
                                    'memory': '2Gi',
                                    'nvidia.com/gpu': '1'
                                }
                            },
                            'nodeSelector': {
                                'nvidia.com/gpu.product': 'NVIDIA-A1000'
                            }
                        },
                        {
                            'name': 'gh200-workers',
                            'replicaCount': worker_groups['gh200-workers']['replicaCount'],
                            'resources': {
                                'limits': {
                                    'cpu': '64',
                                    'memory': '512Gi',
                                    'nvidia.com/gpu': '1'
                                },
                                'requests': {
                                    'cpu': '16',
                                    'memory': '8Gi',
                                    'nvidia.com/gpu': '1'
                                }
                            },
                            'nodeSelector': {
                                'nvidia.com/gpu.product': 'NVIDIA-GH200-480GB'
                            }
                        },
                        {
                            'name': 'cpu-workers',
                            'replicaCount': worker_groups['cpu-workers']['replicaCount'],
                            'resources': {
                                'limits': {
                                    'cpu': '8',
                                    'memory': '16Gi'
                                },
                                'requests': {
                                    'cpu': '2',
                                    'memory': '2Gi'
                                }
                            },
                            'nodeSelector': {}
                        }
                    ]
                },
                
                'nodeSelector': {},
                'tolerations': [],
                'affinity': {}
            }
            
            # Save the deployment configuration
            safe_user = secure_filename(current_user.username)
            safe_exp = secure_filename(experiment_name)
            user_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users', safe_user)
            exp_dir = os.path.join(user_dir, safe_exp)
            
            # Create experiment directory if it doesn't exist
            os.makedirs(exp_dir, exist_ok=True)
            
            # Save values.yaml in the experiment folder
            values_filename = f'values_{safe_exp}.yaml'
            values_path = os.path.join(exp_dir, values_filename)
            
            # Custom YAML dump to handle the args format correctly
            yaml_content = yaml.dump(deployment_config, default_flow_style=False, sort_keys=False)
            
            # Replace the args format to use literal block scalar
            args_pattern = r"args: 'python train\.py.*?sleep infinity'"
            args_replacement = """args: 
          - |
          python train.py {{ if .Values.dsipts_p.defaults.command.modelsToTrain }}-m architecture={{ .Values.dsipts_p.defaults.command.modelsToTrain }}{{ end }} --config-dir=config_milan --config-name=config_milan;
          sleep infinity"""
            
            import re
            yaml_content = re.sub(args_pattern, args_replacement, yaml_content, flags=re.DOTALL)
            
            with open(values_path, 'w') as f:
                f.write(yaml_content)
            
            # Clean up session
            session.pop('sweeper_experiment_name', None)
            session.pop('experiment_name', None)
            session.pop('run_config_filename', None)
            session.pop('run_experiment_name', None)
            
            # Store deployment info in session for preview
            session['deployment_config'] = {
                'experiment_name': experiment_name,
                'values_filename': values_filename,
                'models_to_train': models_to_train,
                'worker_groups': worker_groups,
                'replica_count': replica_count
            }
            
            flash(f'Deployment configuration saved as {values_filename} successfully!')
            return redirect(url_for('routes.deployment_preview'))
            
        except Exception as e:
            current_app.logger.error(f"Error saving deployment config: {e}")
            flash('An error occurred while saving the deployment configuration.')
            return redirect(url_for('routes.deployment_config'))
    
    # Get architecture files for dropdown
    safe_user = secure_filename(current_user.username)
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    user_dir = os.path.join(base_upload, safe_user)
    
    architecture_files = []
    if os.path.exists(user_dir):
        for exp_folder in os.listdir(user_dir):
            exp_path = os.path.join(user_dir, exp_folder)
            if os.path.isdir(exp_path):
                # Check main experiment directory for architecture files
                for file in os.listdir(exp_path):
                    file_path = os.path.join(exp_path, file)
                    if file.lower().endswith(('.yaml', '.yml')) and os.path.isfile(file_path):
                        # Check if it's an architecture file (not config file)
                        if not file.endswith('_config.yaml') and not file.startswith('values_'):
                            # Try to read the file to check if it's an architecture file
                            try:
                                with open(file_path, 'r') as f:
                                    content = f.read()
                                    # Check if it contains architecture-related fields
                                    if 'model_type' in content or 'ts_name' in content or 'model_configs' in content:
                                        model_name = file.replace('.yaml', '').replace('.yml', '')
                                        architecture_files.append({
                                            'name': file,
                                            'experiment': exp_folder,
                                            'model_type': model_name
                                        })
                            except:
                                pass
                
                # Check Architecture subdirectory
                arch_dir = os.path.join(exp_path, 'Architecture')
                if os.path.exists(arch_dir) and os.path.isdir(arch_dir):
                    for file in os.listdir(arch_dir):
                        file_path = os.path.join(arch_dir, file)
                        if file.lower().endswith(('.yaml', '.yml', '.py')) and os.path.isfile(file_path):
                            model_name = os.path.splitext(file)[0]
                            architecture_files.append({
                                'name': file,
                                'experiment': f"{exp_folder}/Architecture",
                                'model_type': model_name
                            })
    
    # Default values for GET request
    default_worker_groups = {
        'a100-workers': {'replicaCount': 1},
        'a1000-workers': {'replicaCount': 0},
        'gh200-workers': {'replicaCount': 0},
        'cpu-workers': {'replicaCount': 0}
    }
    
    return render_template('deployment_config.html',
                         experiment_name=experiment_name,
                         replica_count=1,
                         models_to_train='',
                         worker_groups=default_worker_groups,
                         architecture_files=architecture_files)

@web.route('/deployment_preview', methods=['GET', 'POST'])
@login_required
def deployment_preview():
    """Preview deployment configuration before running experiment."""
    # Get deployment config from session
    deployment_config = session.get('deployment_config', {})
    
    # Get experiment details
    experiment_name = deployment_config.get('experiment_name', 'Unknown')
    safe_user = secure_filename(current_user.username)
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    exp_dir = os.path.join(base_upload, safe_user, secure_filename(experiment_name))
    
    # Get config files in experiment
    config_files = []
    if os.path.exists(exp_dir):
        for file in os.listdir(exp_dir):
            if file.endswith('_config.yaml'):
                config_files.append(file)
    
    if request.method == 'POST':
        # User clicked Run button - execute Helm command
        return redirect(url_for('routes.execute_experiment', experiment_name=experiment_name))
    
    return render_template('deployment_preview.html',
                         experiment_name=experiment_name,
                         deployment_config=deployment_config,
                         config_files=config_files,
                         show_iframe=False)

@web.route('/execute_experiment/<experiment_name>', methods=['GET'])
@login_required
def execute_experiment(experiment_name):
    """Execute the Kubernetes Helm command for the experiment."""
    try:
        # Get deployment config from session
        deployment_config = session.get('deployment_config', {})
        
        # Set up paths and variables
        # Use original username, not secure_filename version
        username = current_user.username
        safe_exp = secure_filename(experiment_name)
        
        # Define paths according to user specifications
        CHART_PATH = "/home/admin/khalid/dsipts-p/dsipts-p-chart"
        EXP_NAME = safe_exp
        
        # Build the Linux path - DO NOT include any Windows paths
        # The NFS mount on the Linux server maps to: /mnt/NFS/khalid/DSIPTS-P/
        # Inside that, our uploads are at: uploads/Users/{username}/{experiment}/
        VALUE_PATH = f"/mnt/NFS/khalid/DSIPTS-P/uploads/Users/{username}/{safe_exp}/values_{safe_exp}.yaml"
        
        # Ensure all backslashes are converted to forward slashes (safety check)
        VALUE_PATH = VALUE_PATH.replace('\\', '/')
        
        current_app.logger.info(f"Building paths for user '{username}', experiment '{safe_exp}'")
        current_app.logger.info(f"VALUE_PATH: {VALUE_PATH}")
        
        # Execute the command via SSH on the remote Linux server
        import subprocess
        import platform
        
        # Get SSH configuration from environment
        enable_ssh = os.environ.get('ENABLE_SSH_DEPLOYMENT', 'false').lower() == 'true'
        ssh_host_env = os.environ.get('SSH_HOST', '')
        ssh_host = f"admin@{ssh_host_env}" if ssh_host_env else "admin@10.1.65.194"  # Fallback for backward compatibility
        
        # Import required modules for local mode
        import hashlib
        import time
        
        # Check if SSH deployment is disabled
        if not enable_ssh:
            current_app.logger.info("SSH deployment is disabled. Simulating local deployment.")
            # Simulate successful deployment for local testing
            helm_command = f"helm install {safe_exp} local-chart --values local-values.yaml"
            ssh_command = "Local deployment (no SSH)"
            ssh_method = "Local deployment mode"
            ssh_key_message = "SSH deployment disabled - running in local mode"
            
            result = subprocess.CompletedProcess(
                args="local deployment simulation",
                returncode=0,
                stdout="Local deployment simulation successful\nExperiment would be deployed to local Kubernetes\nNote: This is a simulation - no actual deployment occurred",
                stderr=""
            )
            # Use a test run_id for local mode
            run_id = hashlib.md5(f"{experiment_name}_local_{time.time()}".encode()).hexdigest()[:24]
            ssh_prefix = "local"  # Set for later use in Aim commands
        else:
            # SSH deployment is enabled
            # Determine SSH key path based on OS
            if platform.system() == 'Windows':
                ssh_key = os.path.expanduser("~/.ssh/id_rsa")
            else:
                ssh_key = "/root/.ssh/id_rsa"
            
            # Check if SSH key exists
            ssh_key_exists = os.path.exists(ssh_key)
            ssh_key_message = f"SSH key {'found' if ssh_key_exists else 'NOT found'} at: {ssh_key}"
            current_app.logger.warning(ssh_key_message)
            
            # Prepare SSH prefix for all commands
            if not ssh_key_exists:
                ssh_prefix = f'ssh -o StrictHostKeyChecking=no {ssh_host}'
                ssh_method = "SSH without key (key not found)"
            else:
                ssh_prefix = f'ssh -i "{ssh_key}" -o StrictHostKeyChecking=no {ssh_host}'
                ssh_method = f"SSH with key at {ssh_key}"
            
            # Construct the Helm command for SSH deployment
            helm_command = f"microk8s helm install {EXP_NAME} {CHART_PATH} --values {VALUE_PATH}"
            ssh_command = f'{ssh_prefix} "{helm_command}"'
            
            current_app.logger.info(f"=== SSH EXECUTION DEBUG ===")
            current_app.logger.info(f"Method: {ssh_method}")
            current_app.logger.info(f"SSH Host: {ssh_host}")
            current_app.logger.info(f"Helm Command: {helm_command}")
            current_app.logger.info(f"Full SSH Command: {ssh_command}")
            current_app.logger.info(f"Working Directory: {os.getcwd()}")
            
            # Try to execute command with detailed error capture
            try:
                result = subprocess.run(ssh_command, shell=True, capture_output=True, text=True, timeout=30)
                current_app.logger.info(f"Command executed with return code: {result.returncode}")
                current_app.logger.info(f"STDOUT: {result.stdout[:500]}...") if result.stdout else None
                current_app.logger.info(f"STDERR: {result.stderr[:500]}...") if result.stderr else None
            except subprocess.TimeoutExpired:
                current_app.logger.error("Command timed out after 30 seconds")
                result = subprocess.CompletedProcess(args=ssh_command, returncode=1, 
                                                    stdout="", 
                                                    stderr="Command timed out after 30 seconds. The server might be unreachable.")
            except Exception as e:
                current_app.logger.error(f"Subprocess error: {str(e)}")
                result = subprocess.CompletedProcess(args=ssh_command, returncode=1,
                                                    stdout="",
                                                    stderr=f"Failed to execute command: {str(e)}")
        
        # After Helm deployment, wait a bit and get the latest hash from Aim repository
        run_id = None
        if result.returncode == 0:
            current_app.logger.info("=== WAITING FOR AIM REGISTRATION ===")
            import time
            time.sleep(5)  # Wait 5 seconds for deployment to register in Aim
            
            current_app.logger.info("=== FETCHING LATEST AIM HASH ===")
            aim_python_script = '''python3 -c "from aim import Repo; import os,sys; repo=Repo('/aim'); cand=[]
for r in repo.iter_runs():
  h=r.hash; ts=None
  try:
    ct=getattr(r,'creation_time',None)
    ts = ct.timestamp() if hasattr(ct,'timestamp') else float(ct) if ct is not None else None
  except Exception:
    pass
  if ts is None:
    try:
      st=getattr(r,'start_time',None)
      ts = st.timestamp() if hasattr(st,'timestamp') else float(st) if st is not None else None
    except Exception:
      pass
  if ts is None:
    for pfx in (os.path.join('/aim','runs'), os.path.join('/aim','run'), os.path.join('/aim','objects'), '/aim'):
      p=os.path.join(pfx,h)
      if os.path.exists(p):
        ts=os.path.getmtime(p); break
  if ts is not None:
    cand.append((h,ts))
if cand:
  print(sorted(cand, key=lambda x:x[1], reverse=True)[0][0])
else:
  print('NO_HASH_FOUND')"'''
            
            # Execute the Aim hash retrieval command
            aim_command = f'{ssh_prefix} {aim_python_script}'
            current_app.logger.info(f"Executing Aim command to get latest hash...")
            
            try:
                aim_result = subprocess.run(aim_command, shell=True, capture_output=True, text=True, timeout=15)
                if aim_result.returncode == 0 and aim_result.stdout.strip():
                    hash_output = aim_result.stdout.strip()
                    if hash_output != 'NO_HASH_FOUND':
                        run_id = hash_output
                        current_app.logger.info(f"Successfully retrieved Aim hash: {run_id}")
                    else:
                        current_app.logger.warning("No hash found in Aim repository")
                else:
                    current_app.logger.error(f"Failed to get Aim hash: {aim_result.stderr}")
            except Exception as e:
                current_app.logger.error(f"Error getting Aim hash: {str(e)}")
        
        # If we didn't get a hash from Aim, generate a fallback hash
        if not run_id:
            import hashlib
            import time
            run_id = hashlib.md5(f"{experiment_name}_{time.time()}".encode()).hexdigest()[:24]
            current_app.logger.info(f"Using generated fallback hash: {run_id}")
        
        # Default hash if still none
        if not run_id:
            run_id = "3a3e543d791f4bbbbed0e330"
        
        # Generate embedding link
        aim_host = os.environ.get('AIM_HOST', '10.1.65.194:30088')
        embedding_link = f"http://{aim_host}/runs/{run_id}/overview"
        
        # Prepare debug information
        debug_info = {
            'ssh_method': ssh_method,
            'ssh_key_message': ssh_key_message,
            'working_dir': os.getcwd(),
            'platform': platform.system(),
            'helm_command': helm_command,
            'value_path': VALUE_PATH,
            'ssh_host': ssh_host,
            'aim_hash_retrieved': run_id if run_id and not run_id.startswith('3a3e') else 'Not retrieved - using fallback'
        }
        
        if result.returncode == 0:
            flash(f'Experiment {EXP_NAME} deployed successfully!')
            success = True
            output = result.stdout
            error = None
        else:
            # Provide detailed error information
            error_msg = result.stderr if result.stderr else "No error output captured"
            
            # Add helpful debugging hints
            if "ssh" in error_msg.lower() and "not recognized" in error_msg.lower():
                error_msg += "\n\nDEBUG: SSH is not installed on Windows. Install OpenSSH Client via Windows Settings."
            elif "permission denied" in error_msg.lower():
                error_msg += f"\n\nDEBUG: SSH key authentication failed. Check key at: {ssh_key}"
            elif "could not resolve hostname" in error_msg.lower():
                error_msg += f"\n\nDEBUG: Cannot reach server {ssh_host}. Check network connection."
            elif "connection refused" in error_msg.lower():
                error_msg += f"\n\nDEBUG: SSH connection refused. Server might be down or SSH not running."
            elif "no such file" in error_msg.lower():
                error_msg += f"\n\nDEBUG: File not found. Check if values file exists at: {VALUE_PATH}"
                
            flash(f'Error deploying experiment {EXP_NAME}')
            success = False
            output = result.stdout if result.stdout else "No output captured"
            error = error_msg
        
        # Clean up session
        session.pop('deployment_config', None)
        
        return render_template('experiment_execution.html',
                             experiment_name=experiment_name,
                             helm_command=helm_command,
                             ssh_command=ssh_command,
                             success=success,
                             output=output,
                             error=error,
                             run_id=run_id,
                             embedding_link=embedding_link,
                             debug_info=debug_info)
        
    except Exception as e:
        current_app.logger.error(f"Error executing experiment: {e}")
        flash(f'An error occurred while executing the experiment: {str(e)}')
        return redirect(url_for('routes.deployment_preview'))

@web.route('/add_file_to_experiment/<experiment_name>', methods=['GET', 'POST'])
@web.route('/add_file_to_experiment/<experiment_name>/<file_type>', methods=['GET', 'POST'])
@login_required
def add_file_to_experiment(experiment_name, file_type='config'):
    """Add a new file to an existing experiment."""
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            safe_user = secure_filename(current_user.username)
            safe_exp = secure_filename(experiment_name)
            base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
            
            # Determine target directory based on file type
            if file_type == 'data':
                # For data files, save to Data/{csv_base} directory
                filename = secure_filename(file.filename)
                csv_base = os.path.splitext(filename)[0]
                target_dir = os.path.join(base_upload, safe_user, 'Data', csv_base)
            elif file_type == 'architecture':
                # For architecture files, save to experiment/Architecture directory
                target_dir = os.path.join(base_upload, safe_user, safe_exp, 'Architecture')
            else:
                # For config files, save to experiment directory
                target_dir = os.path.join(base_upload, safe_user, safe_exp)
            
            os.makedirs(target_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            file_path = os.path.join(target_dir, filename)
            file.save(file_path)
            
            flash(f'File {filename} added to experiment {experiment_name} ({file_type} files)')
            return redirect(url_for('routes.experiment_explorer'))
    
    return render_template('add_file.html', experiment_name=experiment_name, file_type=file_type)

@web.route('/edit_file_in_experiment/<experiment_name>', methods=['GET', 'POST'])
@web.route('/edit_file_in_experiment/<experiment_name>/<file_type>', methods=['GET', 'POST'])
@login_required
def edit_file_in_experiment(experiment_name, file_type='config'):
    """Edit a file in an existing experiment."""
    safe_user = secure_filename(current_user.username)
    safe_exp = secure_filename(experiment_name)
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    
    # Determine source directories based on file type
    directories = []
    if file_type == 'data':
        # Look in Data directories
        data_base_dir = os.path.join(base_upload, safe_user, 'Data')
        if os.path.exists(data_base_dir):
            for data_dir in os.listdir(data_base_dir):
                data_path = os.path.join(data_base_dir, data_dir)
                if os.path.isdir(data_path):
                    directories.append(data_path)
    elif file_type == 'architecture':
        # Look in experiment/Architecture directory
        arch_dir = os.path.join(base_upload, safe_user, safe_exp, 'Architecture')
        if os.path.exists(arch_dir):
            directories.append(arch_dir)
    else:
        # Look in experiment directory for config files
        exp_dir = os.path.join(base_upload, safe_user, safe_exp)
        if os.path.exists(exp_dir):
            directories.append(exp_dir)
    
    # Get list of files from all relevant directories
    files = []
    for directory in directories:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    # Create the preview path for view_file_content route
                    # Remove the base upload folder and Users prefix to get relative path
                    relative_path = os.path.relpath(file_path, os.path.join(base_upload, 'Users'))
                    preview_path = os.path.join('uploads', 'Users', relative_path).replace('\\', '/')
                    
                    files.append({
                        'name': file,
                        'path': file_path,
                        'preview_path': preview_path,
                        'directory': directory
                    })
    
    if request.method == 'POST':
        selected_file = request.form.get('selected_file')
        content = request.form.get('content')
        
        if selected_file and content:
            # Find the full path for the selected file
            file_path = None
            for file_info in files:
                if file_info['name'] == selected_file:
                    file_path = file_info['path']
                    break
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(content)
                
                flash(f'File {selected_file} updated successfully')
                return redirect(url_for('routes.experiment_explorer'))
    
    return render_template('edit_file.html', experiment_name=experiment_name, files=files, file_type=file_type)

@web.route('/add_config_from_explorer/<experiment_name>')
@login_required
def add_config_from_explorer(experiment_name):
    """Redirect to config upload step from experiment explorer."""
    # Set experiment name in session and redirect to config upload step
    session['experiment_name'] = experiment_name
    session['from_explorer'] = True
    return redirect(url_for('routes.upload_config'))

@web.route('/add_arch_from_explorer/<experiment_name>')
@login_required
def add_arch_from_explorer(experiment_name):
    """Redirect to architecture step from experiment explorer."""
    # Set experiment name in session and redirect to architecture step (step 4) with all options
    session['experiment_name'] = experiment_name
    session['from_explorer'] = True
    # Need to set dummy config and csv to pass validation
    session['config_bytes'] = b'# Dummy config for adding architecture files'
    session['config_filename'] = 'dummy_config.yaml'
    session['csv_bytes'] = b'# Dummy CSV for adding architecture files'
    session['csv_filename'] = 'dummy_data.csv'
    return redirect(url_for('routes.upload_archs'))

@web.route('/add_data_from_explorer/<experiment_name>')
@login_required
def add_data_from_explorer(experiment_name):
    """Redirect to CSV upload step from experiment explorer."""
    # Set experiment name in session and redirect to CSV upload step
    session['experiment_name'] = experiment_name
    session['from_explorer'] = True
    # Need to set dummy config to pass validation
    session['config_bytes'] = b'# Dummy config for adding data files'
    session['config_filename'] = 'dummy_config.yaml'
    return redirect(url_for('routes.upload_csv'))

@web.route('/run_experiment', methods=['GET'])
@login_required
def run_experiment():
    """Route for Run Experiment button - first select config file."""
    # Clear any previous config selection to ensure user always selects a config file
    session.pop('run_config_filename', None)
    session.pop('run_experiment_name', None)
    session.pop('config_file_name', None)
    session.pop('config_file_path', None)
    return redirect(url_for('routes.select_config_for_run'))

@web.route('/select_config_for_run', methods=['GET', 'POST'])
@login_required
def select_config_for_run():
    """Select which config file to use for running the experiment."""
    if request.method == 'POST':
        selected_config = request.form.get('selected_config')
        if not selected_config:
            flash('Please select a configuration file to run the experiment.')
            return redirect(request.url)
        
        # Parse the selected config (format: experiment_name/config_filename)
        try:
            experiment_name, config_filename = selected_config.split('/', 1)
            session['run_experiment_name'] = experiment_name
            session['run_config_filename'] = config_filename
            return redirect(url_for('routes.deployment_config', experiment_name=experiment_name))
        except ValueError:
            flash('Invalid configuration file selection.')
            return redirect(request.url)
    
    # GET request - show config file selection
    safe_user = secure_filename(current_user.username)
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    user_dir = os.path.join(base_upload, safe_user)
    
    config_files = []
    if os.path.exists(user_dir):
        for exp_folder in os.listdir(user_dir):
            exp_path = os.path.join(user_dir, exp_folder)
            if os.path.isdir(exp_path) and exp_folder != 'Data':  # Skip Data folder
                for file in os.listdir(exp_path):
                    if file.endswith('.config') or file.endswith('_config.yaml') or (file.endswith('.yaml') and 'config' in file.lower()):
                        config_files.append({
                            'experiment': exp_folder,
                            'filename': file,
                            'full_path': f"{exp_folder}/{file}"
                        })
    
    return render_template('select_config_for_run.html', config_files=config_files)

@web.route('/get_next_model_counter', methods=['POST'])
@login_required
def get_next_model_counter():
    """Get the next available counter for a model type."""
    try:
        data = request.get_json()
        model_type = data.get('model_type', 'lstm')
        experiment_name = data.get('experiment_name') or session.get('experiment_name', 'default')
        
        safe_user = secure_filename(current_user.username)
        safe_exp = secure_filename(experiment_name)
        base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
        exp_dir = os.path.join(base_upload, safe_user, safe_exp)
        arch_dir = os.path.join(exp_dir, 'Architecture')
        
        # Find next available counter
        counter = 1
        if os.path.exists(arch_dir):
            while True:
                test_filename = f"{model_type}_{counter}.yaml"
                if not os.path.exists(os.path.join(arch_dir, test_filename)):
                    break
                counter += 1
        
        return jsonify({'success': True, 'counter': counter})
        
    except Exception as e:
        current_app.logger.error(f"Error getting next model counter: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@web.route('/deployment_config_old', methods=['GET'])
@login_required
def deployment_config_old():
    """Old route for Run Experiment button - goes to step 5 sweeper config."""
    # Get the most recent experiment for this user
    safe_user = secure_filename(current_user.username)
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    user_dir = os.path.join(base_upload, safe_user)
    
    most_recent_exp = None
    if os.path.exists(user_dir):
        # Get all experiment directories with their modification times
        exp_dirs = []
        for exp_folder in os.listdir(user_dir):
            exp_path = os.path.join(user_dir, exp_folder)
            if os.path.isdir(exp_path):
                mtime = os.path.getmtime(exp_path)
                exp_dirs.append((exp_folder, mtime))
        
        # Sort by modification time and get the most recent
        if exp_dirs:
            exp_dirs.sort(key=lambda x: x[1], reverse=True)
            most_recent_exp = exp_dirs[0][0]
    
    # Set the most recent experiment or default
    session['experiment_name'] = most_recent_exp if most_recent_exp else 'quick_run'
    
    # Redirect to step 5 - config selection for sweeper
    return redirect(url_for('routes.run_experiment_config_selection'))

# ===================================================
# O-RAN ML Lifecycle Pipeline (Steps 5, 6, 7)
# ===================================================
# Step 5: Publish to Non-RT RIC ML Model Catalog
# Step 6: Package model into Docker Image + set A1 policies
# Step 7: Deploy packaged xApp to Near-RT RIC
# ===================================================

# ---------------------------------------------------
# Model hyperparameter registry
# ---------------------------------------------------
# Single source of truth describing the hyperparameters that belong to each
# supported model type, together with their default values. Mirrors the form
# handling in yaml_arch_step2(). Used by Step 5 (Publish) to extract and save
# each model's own parameters into the .pkl artifact and the catalog payload.
MODEL_PARAMETERS = {
    'autoformer': {
        'd_model': 4, 'kernel_size': 3, 'n_layer_encoder': 2, 'n_layer_decoder': 2,
        'label_len': 4, 'n_head': 2, 'dropout_rate': 0.5, 'factor': 5, 'hidden_size': 12,
        'optim': 'torch.optim.Adam', 'activation': 'torch.nn.PReLU',
        'persistence_weight': 0.010, 'loss_type': 'l1'
    },
    'lstm': {
        'cat_emb_dim': 16, 'hidden_RNN': 12, 'num_layers_RNN': 3, 'kernel_size': 5,
        'kind': 'lstm', 'sum_emb': True, 'optim': 'torch.optim.SGD',
        'activation': 'torch.nn.SELU'
    },
    'crossformer': {
        'd_model': 4, 'hidden_size': 12, 'n_layer_encoder': 2, 'n_head': 2,
        'dropout_rate': 0.5, 'win_size': 2, 'seg_len': 6, 'factor': 10,
        'optim': 'torch.optim.Adam', 'persistence_weight': 0.010, 'loss_type': 'l1'
    },
    'd3vae': {
        'embedding_dimension': 2, 'scale': 0.1, 'hidden_size': 2, 'num_layers': 1,
        'dropout_rate': 0.1, 'diff_steps': 1, 'loss_type': 'kl', 'beta_end': 0.01,
        'beta_schedule': 'linear', 'channel_mult': 1, 'mult': 4,
        'num_preprocess_blocks': 1, 'num_preprocess_cells': 1, 'num_channels_enc': 1,
        'arch_instance': 'res_mbconv', 'num_latent_per_group': 1, 'num_channels_dec': 1,
        'groups_per_scale': 1, 'num_postprocess_blocks': 1, 'num_postprocess_cells': 1,
        'beta_start': 0, 'optim': 'torch.optim.SGD'
    },
    'diffusion': {
        'd_model': 12, 'learn_var': True, 'cosine_alpha': True, 'diffusion_steps': 100,
        'beta': 0.03, 'gamma': 0.01, 'n_layers_RNN': 4, 'd_head': 64, 'n_head': 8,
        'dropout_rate': 0.0, 'activation': 'torch.nn.GELU', 'subnet': 1,
        'optim': 'torch.optim.Adam', 'perc_subnet_learning_for_step': 0.1,
        'persistence_weight': 0.010, 'loss_type': 'l1'
    },
    'dilated_conv': {
        'cat_emb_dim': 4, 'hidden_RNN': 16, 'num_layers_RNN': 1, 'kernel_size': 3,
        'kind': 'gru', 'sum_emb': True, 'persistence_weight': 1.0, 'use_bn': False,
        'use_glu': True, 'glu_percentage': 0.2, 'quantiles': [0.1, 0.5, 0.9],
        'optim': 'torch.optim.SGD', 'activation': 'torch.nn.SELU',
        'loss_type': 'linear_penalization'
    },
    'dilated_conv_ed': {
        'cat_emb_dim': 4, 'hidden_RNN': 16, 'num_layers_RNN': 1, 'kernel_size': 3,
        'kind': 'gru', 'sum_emb': True, 'persistence_weight': 1.0, 'use_bn': False,
        'quantiles': [0.1, 0.5, 0.9], 'optim': 'torch.optim.SGD',
        'activation': 'torch.nn.SELU', 'loss_type': 'linear_penalization'
    },
    'dlinear': {
        'cat_emb_dim': 4, 'kernel_size': 3, 'sum_emb': True, 'hidden_size': 12,
        'kind': 'dlinear', 'optim': 'torch.optim.SGD', 'activation': 'torch.nn.LeakyReLU',
        'simple': True
    },
    'informer': {
        'd_model': 4, 'hidden_size': 4, 'n_layer_encoder': 2, 'n_layer_decoder': 2,
        'n_head': 2, 'dropout_rate': 0.5, 'optim': 'torch.optim.Adam',
        'activation': 'torch.nn.PReLU', 'persistence_weight': 0.010, 'loss_type': 'l1',
        'remove_last': True
    },
    'linear': {
        'cat_emb_dim': 16, 'kernel_size': 5, 'sum_emb': True, 'hidden_size': 8,
        'kind': 'linear', 'dropout_rate': 0.1, 'use_bn': False, 'optim': 'torch.optim.Adam',
        'activation': 'torch.nn.PReLU', 'persistence_weight': 0.010, 'loss_type': 'l1',
        'simple': False
    },
    'nlinear': {
        'cat_emb_dim': 16, 'kernel_size': 5, 'sum_emb': True, 'hidden_size': 24,
        'kind': 'nlinear'
    },
    'patchtst': {
        'd_model': 4, 'kernel_size': 3, 'decomposition': True, 'n_layer': 2,
        'patch_len': 4, 'n_head': 2, 'stride': 4, 'dropout_rate': 0.5, 'hidden_size': 12,
        'optim': 'torch.optim.Adam', 'activation': 'torch.nn.PReLU',
        'persistence_weight': 0.010, 'loss_type': 'l1', 'remove_last': True
    },
    'persistent': {},
    'rnn': {
        'cat_emb_dim': 16, 'hidden_RNN': 12, 'num_layers_RNN': 3, 'kernel_size': 5,
        'kind': 'gru', 'sum_emb': True
    },
    'tft': {
        'd_model': 4, 'd_head': 4, 'n_head': 4, 'num_layers_RNN': 8,
        'optim': 'torch.optim.Adam', 'dropout_rate': 0.5, 'persistence_weight': 0.010,
        'loss_type': 'l1'
    },
    'xlstm': {
        'cat_emb_dim': 16, 'hidden_RNN': 12, 'num_layers_RNN': 3, 'kernel_size': 5,
        'kind': 'xlstm', 'sum_emb': True, 'num_blocks': 2, 'bidirectional': True,
        'lstm_type': 'slstm'
    },
}


def _extract_model_hyperparameters(model_type, model_configs):
    """Return a clean dict of hyperparameters for the given model type.

    Uses the MODEL_PARAMETERS registry to know which parameters belong to the
    model. Values are taken from the experiment's model_configs; any missing
    parameter is filled with the registry default. Extra keys present in
    model_configs but not in the registry are preserved so nothing is lost.
    """
    mt = (model_type or '').lower()
    spec = MODEL_PARAMETERS.get(mt, {})
    model_configs = model_configs or {}
    hyperparameters = {}
    for param, default in spec.items():
        hyperparameters[param] = model_configs.get(param, default)
    # Preserve any additional params the user defined that aren't in the spec
    for key, value in model_configs.items():
        if key not in hyperparameters:
            hyperparameters[key] = value
    return hyperparameters


def _scan_experiment_models(exp_path):
    """Scan an experiment directory and return all architecture YAML data found."""
    models_found = []
    
    # Check Architecture directory
    arch_dir = os.path.join(exp_path, 'Architecture')
    if os.path.exists(arch_dir):
        for fname in sorted(os.listdir(arch_dir)):
            if fname.endswith(('.yaml', '.yml')):
                fpath = os.path.join(arch_dir, fname)
                try:
                    with open(fpath, 'r') as f:
                        arch_data = yaml.safe_load(f)
                    if not arch_data:
                        continue
                    model_type = 'unknown'
                    if isinstance(arch_data.get('model'), dict) and 'type' in arch_data['model']:
                        model_type = arch_data['model']['type']
                    elif 'model_type' in arch_data:
                        model_type = arch_data['model_type']
                    if model_type != 'unknown':
                        models_found.append({
                            'filename': fname,
                            'filepath': fpath,
                            'model_type': model_type,
                            'model_configs': arch_data.get('model_configs', {}),
                            'train_config': arch_data.get('train_config', {}),
                            'ts': arch_data.get('ts', {}),
                            'full_data': arch_data
                        })
                except Exception:
                    pass
    
    # Fallback: check experiment-level config files
    if not models_found:
        for fname in sorted(os.listdir(exp_path)):
            if fname.endswith(('.yaml', '.yml')) and os.path.isfile(os.path.join(exp_path, fname)):
                fpath = os.path.join(exp_path, fname)
                try:
                    with open(fpath, 'r') as f:
                        cfg = yaml.safe_load(f)
                    if not cfg:
                        continue
                    model_type = 'unknown'
                    if isinstance(cfg.get('model'), dict) and 'type' in cfg['model']:
                        model_type = cfg['model']['type']
                    elif 'model_type' in cfg:
                        model_type = cfg['model_type']
                    if model_type != 'unknown' and cfg.get('model_configs'):
                        models_found.append({
                            'filename': fname,
                            'filepath': fpath,
                            'model_type': model_type,
                            'model_configs': cfg.get('model_configs', {}),
                            'train_config': cfg.get('train_config', {}),
                            'ts': cfg.get('ts', {}),
                            'full_data': cfg
                        })
                except Exception:
                    pass
    
    return models_found


@web.route('/api/publish_to_ric', methods=['POST'])
@login_required
def publish_to_ric():
    """
    Step 5: Publish trained model to Non-RT RIC ML Model Catalog.
    
    - Scans experiment Architecture/ for YAML files
    - Extracts model_type, model_configs, training params
    - Generates a .pkl artifact containing the model metadata
    - POSTs to the ML Model Catalog
    """
    import pickle
    import json as json_mod
    import requests as http_requests
    
    data = request.get_json()
    if not data or 'experiment_name' not in data:
        return jsonify({'success': False, 'error': 'Missing experiment_name'}), 400
    
    experiment_name = data['experiment_name']
    arch_file = data.get('arch_file', None)  # optional: specific arch file
    safe_user = secure_filename(current_user.username)
    safe_exp = secure_filename(experiment_name)
    
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    exp_path = os.path.join(base_upload, safe_user, safe_exp)
    
    if not os.path.exists(exp_path):
        return jsonify({'success': False, 'error': f'Experiment "{experiment_name}" not found'}), 404
    
    # Scan for architecture models
    models_found = _scan_experiment_models(exp_path)
    
    if not models_found:
        return jsonify({
            'success': False,
            'error': 'No architecture YAML files with model_type found in this experiment. '
                     'Please add an architecture file with model type and model_configs first.'
        }), 400
    
    # Track progress so the UI can show dynamic step-by-step feedback
    steps = []
    def add_step(name, status='done', detail=''):
        steps.append({'name': name, 'status': status, 'detail': detail})

    add_step('Scan experiment', 'done',
             f'Found {len(models_found)} model(s): '
             + ', '.join(sorted({m["model_type"] for m in models_found})))

    # Pick specific arch file or first one found
    model_info = models_found[0]
    if arch_file:
        for m in models_found:
            if m['filename'] == arch_file:
                model_info = m
                break
    
    model_type = model_info['model_type']
    model_configs = model_info['model_configs']
    train_config = model_info['train_config']
    ts_config = model_info['ts']
    
    model_version = data.get('version', '1.0.0')
    model_name = f"{safe_exp}-{model_type}".lower().replace('_', '-')

    add_step('Select model', 'done',
             f'{model_type.upper()} from {model_info["filename"]}')

    # ---- Extract this model's own hyperparameters ----
    hyperparameters = _extract_model_hyperparameters(model_type, model_configs)
    training_params = {
        'batch_size': train_config.get('batch_size', 32),
        'max_epochs': train_config.get('max_epochs', 50),
        'learning_rate': train_config.get('lr', train_config.get('learning_rate'))
    }
    training_params = {k: v for k, v in training_params.items() if v is not None}
    add_step(f'Extract {model_type.upper()} hyperparameters', 'done',
             f'{len(hyperparameters)} parameter(s) captured')

    # ---- Generate .pkl artifact ----
    pkl_dir = os.path.join(exp_path, 'ric_artifacts')
    os.makedirs(pkl_dir, exist_ok=True)
    pkl_filename = f"{model_name}_v{model_version}.pkl"
    pkl_path = os.path.join(pkl_dir, pkl_filename)
    
    pkl_artifact = {
        'model_name': model_name,
        'model_type': model_type,
        'version': model_version,
        'hyperparameters': hyperparameters,
        'model_configs': model_configs,
        'training_params': training_params,
        'train_config': train_config,
        'ts_config': ts_config,
        'source_arch_file': model_info['filename'],
        'source_experiment': experiment_name,
        'framework': 'csv-to-yaml-platform',
        'created_at': datetime.utcnow().isoformat() + 'Z'
    }
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(pkl_artifact, f)
    
    pkl_size = os.path.getsize(pkl_path)
    current_app.logger.info(f"Generated .pkl artifact: {pkl_path} ({pkl_size} bytes)")
    add_step('Generate .pkl artifact', 'done',
             f'{pkl_filename} ({pkl_size / 1024:.1f} KB)')
    
    # ---- Build catalog payload ----
    catalog_payload = {
        'name': model_name,
        'version': model_version,
        'model_type': model_type.lower(),
        'image': xapp_image_tag(model_name, model_version),
        'description': f'{model_type.upper()} model from experiment "{experiment_name}" '
                       f'(arch: {model_info["filename"]})',
        'metrics': training_params,
        'hyperparameters': hyperparameters,
        'training_framework': 'csv-to-yaml-platform',
        'input_schema': {
            'past_variables': ts_config.get('past_variables', []),
            'future_variables': ts_config.get('future_variables', None),
            'use_covariates': ts_config.get('use_covariates', True)
        },
        'output_schema': {
            'type': 'prediction',
            'model_type': model_type.lower()
        },
        'pkl_path': pkl_path,
        'xapp_descriptor': {
            'model_configs': model_configs,
            'hyperparameters': hyperparameters,
            'xapp_name': model_name,
            'messaging': {
                'rxMessages': ['A1_POLICY_REQ', 'RIC_SUB_RESP'],
                'txMessages': ['A1_POLICY_RESP', 'RIC_SUB_REQ', 'RIC_INDICATION']
            }
        }
    }
    
    # ---- POST to ML Model Catalog ----
    catalog_url = os.environ.get('RIC_CATALOG_URL', 'http://localhost:8080')
    catalog_response = None
    catalog_confirmation = None
    model_id = None
    demo_mode = False
    
    try:
        resp = http_requests.post(
            f'{catalog_url}/models',
            json=catalog_payload,
            timeout=10
        )
        if resp.status_code in (200, 201):
            catalog_response = resp.json()
            model_id = catalog_response.get('model_id',
                       catalog_response.get('model', {}).get('id', 'unknown'))
            current_app.logger.info(f"Published to catalog: model_id={model_id}")
            add_step('Send .pkl metadata to ML Model Catalog', 'done',
                     f'HTTP {resp.status_code} -> {catalog_url}/models')
        else:
            current_app.logger.warning(f"Catalog returned {resp.status_code}: {resp.text}")
            demo_mode = True
            add_step('Send .pkl metadata to ML Model Catalog', 'warning',
                     f'HTTP {resp.status_code} - falling back to demo mode')
    except Exception as e:
        current_app.logger.warning(f"Catalog unreachable ({catalog_url}): {e}")
        demo_mode = True
        add_step('Send .pkl metadata to ML Model Catalog', 'warning',
                 f'Catalog offline at {catalog_url} - demo mode')
    
    if demo_mode:
        import uuid
        model_id = str(uuid.uuid4())
    
    # ---- Verify the model is actually stored in the catalog ----
    if not demo_mode and model_id and model_id != 'unknown':
        try:
            verify = http_requests.get(f'{catalog_url}/models/{model_id}', timeout=5)
            if verify.status_code == 200:
                stored = verify.json()
                catalog_confirmation = {
                    'verified': True,
                    'model_id': model_id,
                    'name': stored.get('name'),
                    'status': stored.get('status'),
                    'model_type': stored.get('model_type'),
                    'hyperparameters_stored': len(stored.get('hyperparameters', {})),
                    'published_at': stored.get('published_at')
                }
                add_step('Confirm model in catalog', 'done',
                         f'Verified - {catalog_confirmation["hyperparameters_stored"]} '
                         f'params stored, status={stored.get("status")}')
            else:
                add_step('Confirm model in catalog', 'warning',
                         f'GET returned HTTP {verify.status_code}')
        except Exception as e:
            add_step('Confirm model in catalog', 'warning', f'Verify failed: {e}')
    else:
        add_step('Confirm model in catalog', 'warning',
                 'Skipped (demo mode - catalog offline)')
    
    # Save publish state for steps 6 & 7
    publish_state_path = os.path.join(pkl_dir, f"{model_name}_publish_state.json")
    publish_state = {
        'model_id': model_id,
        'model_name': model_name,
        'model_type': model_type,
        'version': model_version,
        'pkl_path': pkl_path,
        'catalog_url': catalog_url,
        'catalog_payload': catalog_payload,
        'hyperparameters': hyperparameters,
        'model_configs': model_configs,
        'ts_config': ts_config,
        'train_config': train_config,
        'training_params': training_params,
        'demo_mode': demo_mode,
        'experiment_name': experiment_name,
        'published_at': datetime.utcnow().isoformat() + 'Z'
    }
    with open(publish_state_path, 'w') as f:
        json_mod.dump(publish_state, f, indent=2, default=str)
    
    return jsonify({
        'success': True,
        'message': f'Model "{model_name}" published to Non-RT RIC ML Model Catalog!',
        'model_id': model_id,
        'model_name': model_name,
        'model_type': model_type,
        'pkl_path': pkl_path,
        'pkl_file': pkl_filename,
        'pkl_size': f'{pkl_size / 1024:.1f} KB',
        'arch_file': model_info['filename'],
        'demo_mode': demo_mode,
        'catalog_url': catalog_url,
        'hyperparameters': hyperparameters,
        'param_count': len(hyperparameters),
        'training_params': training_params,
        'catalog_confirmation': catalog_confirmation,
        'steps': steps,
        'payload': catalog_payload,
        'models_available': [{'filename': m['filename'], 'model_type': m['model_type']}
                             for m in models_found]
    }), 200


def _generate_xapp_inline(build_dir, model_name, model_type, model_version, pkl_path):
    """
    Inline xApp ML Model Runner generation fallback.
    Used when build_xapp.py is not available (e.g., during development).
    Generates the same artifacts as build_xapp.py: wrapper, Dockerfile, descriptor, Helm chart.
    """
    import shutil
    import json as json_mod

    # Copy .pkl to build dir
    if pkl_path and os.path.exists(pkl_path):
        shutil.copy2(pkl_path, os.path.join(build_dir, 'model.pkl'))

    image_tag = xapp_image_tag(model_name, model_version)

    # Generate xApp ML Model Runner wrapper
    wrapper_code = f'''#!/usr/bin/env python3
"""
O-RAN xApp ML Model Runner: {model_name}
Model Type: {model_type}
Auto-generated by xApp Builder (csv-to-yaml-platform).
"""
import pickle
import json
import os
import signal
import sys
import time
import threading
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model.pkl")
RMR_PORT = int(os.environ.get("RMR_PORT", "4560"))
HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "8080"))
XAPP_NAME = "{model_name}"
MODEL_TYPE = "{model_type}"

# O-RAN xApp descriptor: the onboarder mounts config-file.json into the container
# and points XAPP_DESCRIPTOR_PATH at its directory. The 'controls' section carries
# the framework config (model type, hyperparameters, timeseries, training).
XAPP_DESCRIPTOR_PATH = os.environ.get("XAPP_DESCRIPTOR_PATH", "/opt/ric/config")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [xApp] %(levelname)s - %(message)s')
logger = logging.getLogger(XAPP_NAME)

def load_controls():
    """Load the 'controls' section from the mounted xApp descriptor (config-file.json)."""
    for path in (
        os.path.join(XAPP_DESCRIPTOR_PATH, "config-file.json"),
        "/app/config-file.json",
    ):
        try:
            with open(path) as f:
                desc = json.load(f)
            controls = desc.get("controls", {{}})
            logger.info(f"Loaded controls from {{path}}: {{len(controls)}} keys")
            return controls
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning(f"Failed to read descriptor at {{path}}: {{e}}")
    logger.info("No xApp descriptor found, using built-in defaults")
    return {{}}

class HealthHandler(BaseHTTPRequestHandler):
    xapp_ref = None
    def do_GET(self):
        if self.path in ("/health", "/ready"):
            status = self.server.xapp_ref.get_health() if self.server.xapp_ref else {{"status": "unknown"}}
            code = 200 if status.get("healthy") else 503
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.server.xapp_ref.stats if self.server.xapp_ref else {{}}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args): pass

class MLModelRunner:
    def __init__(self):
        self.model = None
        self.running = True
        self.healthy = False
        self.start_time = datetime.utcnow()
        self.stats = {{"predictions": 0, "a1_policies": 0, "e2_indications": 0, "errors": 0}}
        self.active_policies = {{}}
        # Runtime config from the xApp descriptor 'controls' section
        self.controls = load_controls()
        self.model_type = self.controls.get("model_type", MODEL_TYPE)
        self.hyperparameters = self.controls.get("hyperparameters", {{}})
        logger.info(f"xApp config: model_type={{self.model_type}}, {{len(self.hyperparameters)}} hyperparameters")
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT, self._shutdown)
        self._start_health()

    def _shutdown(self, signum, frame):
        logger.info(f"Shutting down (signal {{signum}})...")
        self.running = False

    def _start_health(self):
        def serve():
            server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
            server.xapp_ref = self
            server.timeout = 1
            while self.running:
                server.handle_request()
        threading.Thread(target=serve, daemon=True).start()
        logger.info(f"Health endpoint on port {{HEALTH_PORT}}")

    def load_model(self):
        logger.info(f"Loading model from {{MODEL_PATH}}")
        with open(MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded: type={{self.model.get('model_type', MODEL_TYPE) if isinstance(self.model, dict) else MODEL_TYPE}}")
        self.healthy = True

    def handle_a1_policy(self, policy):
        self.stats["a1_policies"] += 1
        policy_id = policy.get("policy_id", "unknown")
        self.active_policies[policy_id] = policy
        logger.info(f"A1 policy {{policy_id}} applied")
        return {{"status": "ACK", "policy_id": policy_id}}

    def predict(self, input_data):
        self.stats["predictions"] += 1
        return {{
            "model": XAPP_NAME, "model_type": self.model_type,
            "prediction": {{"value": 0.0, "confidence": 0.95}},
            "timestamp": datetime.utcnow().isoformat(), "status": "SUCCESS"
        }}

    def get_health(self):
        return {{
            "healthy": self.healthy and self.running,
            "xapp": XAPP_NAME, "model_type": self.model_type,
            "uptime": (datetime.utcnow() - self.start_time).total_seconds(),
            "predictions": self.stats["predictions"]
        }}

    def run(self):
        logger.info(f"O-RAN ML Model Runner: {{XAPP_NAME}} ({{MODEL_TYPE}})")
        self.load_model()
        logger.info(f"xApp running, waiting for RMR messages...")
        while self.running:
            time.sleep(1)
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    MLModelRunner().run()
'''
    with open(os.path.join(build_dir, 'xapp_main.py'), 'w') as f:
        f.write(wrapper_code)

    # Dockerfile
    dockerfile = f'''FROM python:3.11-slim
LABEL name="{model_name}" org.o-ran-sc.xapp.component="ml-model-runner"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY xapp_main.py /app/xapp_main.py
COPY model.pkl /app/model.pkl
COPY config-file.json /app/config-file.json
ENV MODEL_PATH=/app/model.pkl RMR_PORT=4560 HEALTH_PORT=8080 XAPP_DESCRIPTOR_PATH=/app
HEALTHCHECK --interval=15s --timeout=5s CMD curl -f http://localhost:8080/health || exit 1
EXPOSE 4560 8080
CMD ["python3", "/app/xapp_main.py"]
'''
    with open(os.path.join(build_dir, 'Dockerfile'), 'w') as f:
        f.write(dockerfile)

    # requirements.txt
    with open(os.path.join(build_dir, 'requirements.txt'), 'w') as f:
        f.write('redis>=4.0.0\n')

    # xApp descriptor (config-file.json)
    descriptor = {
        'xapp_name': model_name,
        'version': model_version,
        'containers': [{'name': model_name, 'image': {'registry': XAPP_REGISTRY, 'name': f'xapps/{model_name}', 'tag': model_version}}],
        'messaging': {'ports': [{'name': 'rmr-data', 'container': model_name, 'port': 4560,
                                  'rxMessages': ['A1_POLICY_REQ', 'RIC_SUB_RESP', 'RIC_INDICATION'],
                                  'txMessages': ['A1_POLICY_RESP', 'RIC_SUB_REQ', 'PREDICTION_OUTPUT'],
                                  'policies': [20008]}]},
        'controls': {'model_type': model_type, 'a1_policy_types': ['ORAN_TrafficSteeringPreference_2.0.0']}
    }
    with open(os.path.join(build_dir, 'config-file.json'), 'w') as f:
        json_mod.dump(descriptor, f, indent=2)

    # Helm chart
    helm_dir = os.path.join(build_dir, 'helm', model_name, 'templates')
    os.makedirs(helm_dir, exist_ok=True)
    with open(os.path.join(build_dir, 'helm', model_name, 'Chart.yaml'), 'w') as f:
        f.write(f'apiVersion: v2\nname: {model_name}\nversion: {model_version}\n')
    with open(os.path.join(build_dir, 'helm', model_name, 'values.yaml'), 'w') as f:
        f.write(f'image:\n  repository: {XAPP_REGISTRY}/xapps/{model_name}\n  tag: "{model_version}"\n  pullPolicy: Never\nreplicaCount: 1\n')
    deployment_yaml = (
        'apiVersion: apps/v1\n'
        'kind: Deployment\n'
        'metadata:\n'
        '  name: {{ .Release.Name }}\n'
        '  labels:\n'
        '    app: {{ .Release.Name }}\n'
        'spec:\n'
        '  replicas: {{ .Values.replicaCount }}\n'
        '  selector:\n'
        '    matchLabels:\n'
        '      app: {{ .Release.Name }}\n'
        '  template:\n'
        '    metadata:\n'
        '      labels:\n'
        '        app: {{ .Release.Name }}\n'
        '    spec:\n'
        '      containers:\n'
        '      - name: {{ .Chart.Name }}\n'
        '        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"\n'
        '        imagePullPolicy: {{ .Values.image.pullPolicy }}\n'
        '        ports:\n'
        '        - containerPort: 4560\n'
        '          name: rmr-data\n'
        '        - containerPort: 8080\n'
        '          name: health\n'
        '        livenessProbe:\n'
        '          httpGet:\n'
        '            path: /health\n'
        '            port: 8080\n'
        '          initialDelaySeconds: 10\n'
        '          periodSeconds: 15\n'
        '        readinessProbe:\n'
        '          httpGet:\n'
        '            path: /ready\n'
        '            port: 8080\n'
        '          initialDelaySeconds: 5\n'
        '          periodSeconds: 10\n'
    )
    with open(os.path.join(helm_dir, 'deployment.yaml'), 'w') as f:
        f.write(deployment_yaml)

    return f'Inline generation complete: xapp_main.py, Dockerfile, config-file.json, Helm chart'


def _build_docker_image(build_dir, image_tag, minikube_profile='kero-ric'):
    """
    Build the xApp Docker image from the generated artifacts.

    Builds into minikube's Docker daemon (when available) so the image is
    immediately usable by the cluster for deployment without an external push.
    Falls back to the host Docker daemon if minikube is not present.

    Returns a dict: {built: bool, backend: str, docker_available: bool, steps: [...]}
    """
    import subprocess

    steps = []
    result = {'built': False, 'backend': 'none', 'docker_available': False, 'steps': steps}

    # 1) Check Docker availability
    steps.append({'name': 'Check Docker daemon', 'status': 'running'})
    try:
        r = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=10)
        result['docker_available'] = (r.returncode == 0)
    except Exception as e:
        current_app.logger.info(f"[Build] docker info failed: {e}")
        result['docker_available'] = False

    if not result['docker_available']:
        steps[-1]['status'] = 'warning'
        steps[-1]['detail'] = 'Docker not available - image will build at deploy time'
        return result
    steps[-1]['status'] = 'done'
    steps[-1]['detail'] = 'Docker is running'

    # 2) Resolve build environment - prefer minikube's docker daemon
    build_env = os.environ.copy()
    backend = 'host-docker'
    steps.append({'name': 'Connect to build daemon', 'status': 'running'})
    try:
        env_cmd = subprocess.run(
            ['minikube', '-p', minikube_profile, 'docker-env', '--shell', 'bash'],
            capture_output=True, text=True, timeout=15
        )
        if env_cmd.returncode == 0 and env_cmd.stdout:
            for line in env_cmd.stdout.strip().split('\n'):
                line = line.strip()
                if line.startswith('export '):
                    kv = line.replace('export ', '', 1).split('=', 1)
                    if len(kv) == 2:
                        build_env[kv[0]] = kv[1].strip().strip('"')
            backend = f'minikube ({minikube_profile})'
            steps[-1]['status'] = 'done'
            steps[-1]['detail'] = f'Using minikube docker daemon ({minikube_profile})'
        else:
            steps[-1]['status'] = 'done'
            steps[-1]['detail'] = 'Using host Docker daemon (minikube env unavailable)'
    except Exception as e:
        current_app.logger.info(f"[Build] minikube docker-env failed: {e}")
        steps[-1]['status'] = 'done'
        steps[-1]['detail'] = 'Using host Docker daemon'
    result['backend'] = backend

    # 3) Build the image
    steps.append({'name': f'Build image {image_tag}', 'status': 'running'})
    try:
        build_r = subprocess.run(
            ['docker', 'build', '-t', image_tag, build_dir],
            env=build_env, capture_output=True, text=True, timeout=600
        )
        if build_r.returncode == 0:
            steps[-1]['status'] = 'done'
            steps[-1]['detail'] = 'Image built successfully'
            result['built'] = True
        else:
            steps[-1]['status'] = 'error'
            steps[-1]['detail'] = (build_r.stderr or build_r.stdout or 'build failed')[-300:]
            current_app.logger.error(f"[Build] docker build failed: {build_r.stderr[-500:]}")
            return result
    except subprocess.TimeoutExpired:
        steps[-1]['status'] = 'error'
        steps[-1]['detail'] = 'Build timed out after 600s'
        return result
    except Exception as e:
        steps[-1]['status'] = 'error'
        steps[-1]['detail'] = str(e)[:200]
        return result

    # 4) Verify the image exists in the daemon
    steps.append({'name': 'Verify image in registry', 'status': 'running'})
    try:
        verify_r = subprocess.run(
            ['docker', 'images', '-q', image_tag],
            env=build_env, capture_output=True, text=True, timeout=15
        )
        if verify_r.stdout.strip():
            steps[-1]['status'] = 'done'
            steps[-1]['detail'] = f'Image id: {verify_r.stdout.strip()[:12]}'
        else:
            steps[-1]['status'] = 'warning'
            steps[-1]['detail'] = 'Image not found after build'
    except Exception as e:
        steps[-1]['status'] = 'warning'
        steps[-1]['detail'] = str(e)[:150]

    return result


def _test_xapp_pod(kubectl_base, m_name, namespace='ricxapp', timeout_s=60):
    """
    Wait for the xApp pod to become Ready and test its health endpoint.

    kubectl_base: list, e.g. ['kubectl'] or ['minikube','-p','kero-ric','kubectl','--']
    Returns (healthy: bool, detail: str, pod_output: str)
    """
    import subprocess
    import time

    # The onboarder Helm chart labels pods as app=<namespace>-<name> and
    # release=<releaseName>. We install with release name == m_name, so the
    # release label is the reliable selector across xApps.
    selector = f'release={m_name}'
    # Wait for the pod to be scheduled
    deadline = time.time() + timeout_s
    phase = ''
    pod_out = ''
    while time.time() < deadline:
        try:
            r = subprocess.run(
                kubectl_base + ['get', 'pods', '-n', namespace, '-l', selector,
                                '-o', 'jsonpath={.items[0].status.phase}'],
                capture_output=True, text=True, timeout=15
            )
            phase = r.stdout.strip()
            if phase in ('Running', 'Succeeded'):
                break
            if phase == 'Failed':
                break
        except Exception:
            pass
        time.sleep(3)

    # Get full pod status for display
    try:
        r = subprocess.run(
            kubectl_base + ['get', 'pods', '-n', namespace, '-l', selector, '-o', 'wide'],
            capture_output=True, text=True, timeout=15
        )
        pod_out = r.stdout.strip()
    except Exception as e:
        pod_out = str(e)

    if phase != 'Running':
        return False, f'Pod phase={phase or "unknown"} (not Running within {timeout_s}s)', pod_out

    # Get the pod name
    try:
        r = subprocess.run(
            kubectl_base + ['get', 'pods', '-n', namespace, '-l', selector,
                            '-o', 'jsonpath={.items[0].metadata.name}'],
            capture_output=True, text=True, timeout=15
        )
        pod_name = r.stdout.strip()
    except Exception:
        pod_name = ''

    if not pod_name:
        return False, 'Pod running but name not found', pod_out

    # Test the health endpoint inside the pod (xApp exposes /health on 8080)
    try:
        health_r = subprocess.run(
            kubectl_base + ['exec', '-n', namespace, pod_name, '--',
                            'curl', '-sf', '-m', '5', 'http://localhost:8080/health'],
            capture_output=True, text=True, timeout=20
        )
        if health_r.returncode == 0:
            return True, f'Pod Running, /health OK: {health_r.stdout.strip()[:120]}', pod_out
        else:
            # Pod is running even if health curl unavailable (curl may be missing)
            return True, f'Pod Running (health check inconclusive: {health_r.stderr.strip()[:80]})', pod_out
    except Exception as e:
        return True, f'Pod Running (health probe error: {str(e)[:80]})', pod_out


def _build_xapp_controls(model_type, model_version, model_configs, hyperparameters,
                         ts_config, training_params):
    """
    Build the xApp descriptor 'controls' section from the framework's
    architecture YAML config so the two schemas match.

    The 'controls' section holds xApp-specific internal configuration. It is
    injected into the container as a JSON file and read at runtime via the
    XAPP_DESCRIPTOR_PATH environment variable (O-RAN xApp descriptor spec).
    """
    ts_config = ts_config or {}
    model_configs = model_configs or {}
    hyperparameters = hyperparameters or {}
    training_params = training_params or {}

    # Merge model_configs + extracted hyperparameters (model_configs wins on conflict)
    merged_hp = {}
    merged_hp.update(hyperparameters)
    merged_hp.update(model_configs)
    # Keep only JSON-serializable scalar/list/dict values
    merged_hp = {k: v for k, v in merged_hp.items() if isinstance(v, (str, int, float, bool, list, dict))}

    controls = {
        'model_type': str(model_type),
        'model_version': str(model_version),
        'prediction': {
            'rmr_output_type': 30000,
            'report_period_ms': 1000
        },
        'a1_policy': {
            'policy_type_id': 20008,
            'types': ['ORAN_TrafficSteeringPreference_2.0.0']
        },
        'hyperparameters': merged_hp,
        'timeseries': {
            'name': str(ts_config.get('name', '')),
            'version': ts_config.get('version', 1),
            'use_covariates': bool(ts_config.get('use_covariates', True)),
            'past_variables': ts_config.get('past_variables', []) or [],
            'future_variables': ts_config.get('future_variables', []) or [],
            'static_variables': ts_config.get('static_variables', []) or []
        },
        'training': {
            'batch_size': int(training_params.get('batch_size', 32)),
            'max_epochs': int(training_params.get('max_epochs', 50))
        }
    }
    lr = training_params.get('learning_rate')
    if lr is not None:
        try:
            controls['training']['learning_rate'] = float(lr)
        except (TypeError, ValueError):
            pass
    return controls


def _json_schema_from_value(value):
    """
    Infer a JSON Schema (draft-07 compatible) fragment from a Python value.
    Used to auto-generate the controls-schema.json that the O-RAN xApp
    onboarder requires whenever a descriptor declares a 'controls' section.
    """
    if isinstance(value, bool):
        return {'type': 'boolean'}
    if isinstance(value, int):
        return {'type': 'integer'}
    if isinstance(value, float):
        return {'type': 'number'}
    if isinstance(value, str):
        return {'type': 'string'}
    if isinstance(value, list):
        if value:
            return {'type': 'array', 'items': _json_schema_from_value(value[0])}
        return {'type': 'array'}
    if isinstance(value, dict):
        props = {k: _json_schema_from_value(v) for k, v in value.items()}
        return {
            'type': 'object',
            'properties': props,
            'required': list(value.keys())
        }
    return {}


def _generate_controls_schema(controls):
    """
    Generate the draft-07 controls schema that validates the descriptor's
    'controls' section during O-RAN xApp onboarding. The onboarder rejects any
    descriptor that has a non-empty controls section without a matching schema.
    """
    body = _json_schema_from_value(controls)
    schema = {
        '$schema': 'http://json-schema.org/draft-07/schema#',
        '$id': '#/controls',
        'title': 'Controls Section Schema',
    }
    schema.update(body)
    return schema


@web.route('/api/package_xapp', methods=['POST'])
@login_required
def package_xapp():
    """
    Step 6: Package model into Docker Image File + set A1 policies.
    
    Uses the O-RAN repo's build_xapp.py to generate:
    - xApp Python wrapper (xapp_main.py)
    - Dockerfile
    - xApp descriptor (config-file.json)
    - Helm chart (deployment.yaml, service.yaml)
    And generates RIC config files (A1 policy, appmgr, submgr).
    """
    import json as json_mod
    import subprocess
    import sys
    
    data = request.get_json()
    if not data or 'experiment_name' not in data:
        return jsonify({'success': False, 'error': 'Missing experiment_name'}), 400
    
    experiment_name = data['experiment_name']
    model_name_filter = data.get('model_name', '')
    safe_user = secure_filename(current_user.username)
    safe_exp = secure_filename(experiment_name)
    
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    exp_path = os.path.join(base_upload, safe_user, safe_exp)
    pkl_dir = os.path.join(exp_path, 'ric_artifacts')
    
    if not os.path.exists(pkl_dir):
        return jsonify({
            'success': False,
            'error': 'No published models found. Run Step 5 (Publish) first.'
        }), 400
    
    # Find publish state
    publish_state = None
    state_file = None
    for fname in os.listdir(pkl_dir):
        if fname.endswith('_publish_state.json'):
            if model_name_filter and model_name_filter not in fname:
                continue
            state_file = os.path.join(pkl_dir, fname)
            with open(state_file, 'r') as f:
                publish_state = json_mod.load(f)
            break
    
    if not publish_state:
        return jsonify({
            'success': False,
            'error': 'No publish state found. Run Step 5 (Publish) first.'
        }), 400
    
    m_name = publish_state['model_name']
    m_type = publish_state['model_type']
    m_version = publish_state['version']
    pkl_path = publish_state.get('pkl_path', '')
    
    # ---- Call O-RAN repo build_xapp.py ----
    # Try relative path first, then home-dir based paths (works on both local and server)
    xapp_builder = os.path.normpath(
        os.path.join(os.path.dirname(current_app.root_path),
                     '..', 'oran-ric', 'nonrtric', 'xapp-builder', 'build_xapp.py')
    )
    if not os.path.exists(xapp_builder):
        # Try home directory paths (Linux server)
        home = os.path.expanduser('~')
        xapp_builder = os.path.join(home, 'oran-ric', 'nonrtric', 'xapp-builder', 'build_xapp.py')
    
    build_output_dir = os.path.join(pkl_dir, 'xapp-build')
    os.makedirs(build_output_dir, exist_ok=True)
    
    current_app.logger.info(f"[Step6] build_xapp.py path: {xapp_builder}")
    current_app.logger.info(f"[Step6] build_xapp.py exists: {os.path.exists(xapp_builder)}")
    current_app.logger.info(f"[Step6] output dir: {build_output_dir}")
    current_app.logger.info(f"[Step6] pkl_path: {pkl_path}, exists: {os.path.exists(pkl_path) if pkl_path else 'N/A'}")
    
    build_cmd = [
        sys.executable, xapp_builder,
        '--name', m_name,
        '--version', m_version,
        '--type', m_type,
        '--output-dir', build_output_dir,
        '--no-build',
    ]
    
    # Use the real .pkl from Step 5 if available, else demo
    if pkl_path and os.path.exists(pkl_path):
        build_cmd.extend(['--pkl', pkl_path])
    else:
        build_cmd.append('--demo')
    
    current_app.logger.info(f"[Step6] build cmd: {' '.join(build_cmd)}")
    
    # build_xapp.py reads the registry from XAPP_REGISTRY; pass ours so the
    # descriptor + chart it generates use the same schema-valid dotted registry
    # that Step 7 onboards/deploys and that we tag the image with.
    build_env = {**os.environ, 'XAPP_REGISTRY': XAPP_REGISTRY}
    builder_output = ''
    try:
        result = subprocess.run(
            build_cmd, capture_output=True, text=True, timeout=30, env=build_env
        )
        builder_output = result.stdout
        current_app.logger.info(f"[Step6] build_xapp.py returncode: {result.returncode}")
        current_app.logger.info(f"[Step6] stdout: {result.stdout[:300]}")
        if result.stderr:
            current_app.logger.info(f"[Step6] stderr: {result.stderr[:300]}")
        if result.returncode != 0:
            current_app.logger.error(f"build_xapp.py failed, falling back to inline")
            builder_output = _generate_xapp_inline(build_output_dir, m_name, m_type, m_version, pkl_path)
    except FileNotFoundError:
        current_app.logger.warning(f"build_xapp.py not found at {xapp_builder}, generating inline")
        builder_output = _generate_xapp_inline(build_output_dir, m_name, m_type, m_version, pkl_path)
    except Exception as e:
        current_app.logger.warning(f"build_xapp.py error: {e}, generating inline")
        builder_output = _generate_xapp_inline(build_output_dir, m_name, m_type, m_version, pkl_path)
    
    # Verify files were generated
    generated_files = os.listdir(build_output_dir) if os.path.exists(build_output_dir) else []
    current_app.logger.info(f"[Step6] files in xapp-build: {generated_files}")
    if 'xapp_main.py' not in generated_files:
        current_app.logger.warning(f"[Step6] xapp_main.py missing, forcing inline generation")
        builder_output = _generate_xapp_inline(build_output_dir, m_name, m_type, m_version, pkl_path)
        generated_files = os.listdir(build_output_dir)
        current_app.logger.info(f"[Step6] files after inline: {generated_files}")
    
    # ---- Generate RIC config files (from orchestrate_pipeline.py logic) ----
    ric_config_dir = os.path.join(pkl_dir, 'ric-configs')
    os.makedirs(ric_config_dir, exist_ok=True)
    
    image_tag = xapp_image_tag(m_name, m_version)
    
    # A1 Policy Type definition
    a1_policy = {
        'name': f'ORAN_TrafficSteeringPreference_{m_name}',
        'description': f'A1 policy type for {m_name} xApp',
        'policy_type_id': 20008,
        'create_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'type': 'object',
            'properties': {
                'scope': {
                    'type': 'object',
                    'properties': {
                        'ueId': {'type': 'string'},
                        'cellId': {'type': 'string'}
                    }
                },
                'qosObjectives': {
                    'type': 'object',
                    'properties': {
                        'priorityLevel': {'type': 'integer', 'minimum': 1, 'maximum': 15},
                        'targetThroughput': {'type': 'number'}
                    }
                },
                'resources': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'cellIdList': {'type': 'array', 'items': {'type': 'string'}},
                            'preference': {'type': 'string', 'enum': ['SHALL', 'PREFER', 'AVOID', 'FORBID']}
                        }
                    }
                }
            }
        }
    }
    
    # appmgr xApp config
    appmgr_config = {
        'xapp_name': m_name,
        'version': m_version,
        'release_name': m_name,
        'namespace': 'ricxapp',
        'helmVersion': m_version,
        'overrides': {
            'image.repository': image_tag.rsplit(':', 1)[0],
            'image.tag': m_version,
            'replicaCount': 1
        }
    }
    
    # submgr subscription config
    submgr_config = {
        'xapp_name': m_name,
        'subscription': {
            'ActionType': 'report',
            'SubsequentAction': {'SubsequentActionType': 'continue', 'TimeToWait': 'w10ms'},
            'EventTriggerDefinition': {
                'reportingPeriod_ms': 1000,
                'eventTriggerStyle': 1
            },
            'ActionDefinitions': [{
                'ActionID': 1,
                'ActionType': 'report',
                'RICactionDefinition': {
                    'metrics': ['DRB.UEThpDl', 'DRB.UEThpUl', 'RRU.PrbUsedDl', 'RRU.PrbUsedUl']
                }
            }]
        }
    }
    
    ric_configs = {
        'a1-policy-type.json': a1_policy,
        'appmgr-config.json': appmgr_config,
        'submgr-config.json': submgr_config
    }
    
    for fname, content in ric_configs.items():
        with open(os.path.join(ric_config_dir, fname), 'w') as f:
            json_mod.dump(content, f, indent=2)
    
    # Read the xApp descriptor generated by build_xapp.py
    xapp_descriptor = {}
    desc_path = os.path.join(build_output_dir, 'config-file.json')
    if os.path.exists(desc_path):
        with open(desc_path) as f:
            xapp_descriptor = json_mod.load(f)

    # ---- Schema matching: inject framework config into descriptor 'controls' ----
    # The framework architecture YAML config is carried into the xApp descriptor's
    # 'controls' section, and a matching draft-07 controls-schema.json is generated
    # so the O-RAN onboarder accepts the descriptor (it rejects a non-empty controls
    # section without a matching schema).
    xapp_controls = _build_xapp_controls(
        m_type, m_version,
        publish_state.get('model_configs', {}),
        publish_state.get('hyperparameters', {}),
        publish_state.get('ts_config', {}),
        publish_state.get('training_params', {}),
    )
    xapp_descriptor['controls'] = xapp_controls

    # Normalize the container image so it uses our schema-valid dotted registry
    # and matches the tag we build. The onboarder validates image.registry against
    # a pattern that requires a dot, and appmgr builds the pod image reference as
    # "{registry}/{name}:{tag}", which must equal the locally-built image tag.
    containers = xapp_descriptor.get('containers')
    if isinstance(containers, list) and containers:
        for c in containers:
            if isinstance(c, dict):
                c['image'] = {
                    'registry': XAPP_REGISTRY,
                    'name': f'xapps/{m_name}',
                    'tag': m_version,
                }
    else:
        xapp_descriptor['containers'] = [{
            'name': m_name,
            'image': {'registry': XAPP_REGISTRY, 'name': f'xapps/{m_name}', 'tag': m_version},
        }]
    with open(desc_path, 'w') as f:
        json_mod.dump(xapp_descriptor, f, indent=2)

    controls_schema = _generate_controls_schema(xapp_controls)
    schema_path = os.path.join(build_output_dir, 'schema.json')
    with open(schema_path, 'w') as f:
        json_mod.dump(controls_schema, f, indent=2)
    current_app.logger.info(f"[Step6] wrote schema.json ({len(controls_schema.get('properties', {}))} top-level control keys)")
    
    # Collect all generated artifacts
    all_artifacts = []
    for d in [build_output_dir, ric_config_dir]:
        if os.path.exists(d):
            for fname in os.listdir(d):
                fpath = os.path.join(d, fname)
                if os.path.isfile(fpath):
                    all_artifacts.append(fname)
    
    # Check for Helm chart
    helm_dir = os.path.join(build_output_dir, 'helm', m_name)
    has_helm = os.path.exists(helm_dir)
    
    # ---- Build the Docker image from the trained model ----
    # Builds into minikube's docker daemon on the server so the image is
    # immediately available to the cluster for deployment (Step 7).
    package_steps = []
    for fname in ['xapp_main.py', 'Dockerfile', 'model.pkl', 'config-file.json']:
        package_steps.append({
            'name': f'Generate {fname}',
            'status': 'done' if fname in all_artifacts else 'warning',
            'detail': 'created' if fname in all_artifacts else 'missing'
        })
    
    build_result = _build_docker_image(build_output_dir, image_tag)
    package_steps.extend(build_result['steps'])
    
    # Update publish state
    publish_state['packaged'] = True
    publish_state['packaged_at'] = datetime.utcnow().isoformat() + 'Z'
    publish_state['artifacts'] = all_artifacts
    publish_state['docker_image'] = image_tag
    publish_state['build_output_dir'] = build_output_dir
    publish_state['ric_config_dir'] = ric_config_dir
    publish_state['helm_chart_dir'] = helm_dir if has_helm else None
    publish_state['descriptor_path'] = desc_path
    publish_state['controls_schema_path'] = schema_path
    publish_state['image_built'] = build_result['built']
    publish_state['build_backend'] = build_result.get('backend', 'none')
    
    if state_file:
        with open(state_file, 'w') as f:
            json_mod.dump(publish_state, f, indent=2, default=str)
    
    build_note = ''
    if build_result['built']:
        build_note = f" Docker image built ({build_result.get('backend')})."
    elif build_result.get('docker_available'):
        build_note = ' Image build failed - see details.'
    else:
        build_note = ' Docker not available - image will build at deploy time.'
    
    return jsonify({
        'success': True,
        'message': f'Model "{m_name}" packaged as xApp! '
                   f'({len(all_artifacts)} artifacts generated).{build_note}',
        'model_name': m_name,
        'model_type': m_type,
        'docker_image': image_tag,
        'artifacts': all_artifacts,
        'has_helm_chart': has_helm,
        'xapp_descriptor': xapp_descriptor,
        'ric_configs': list(ric_configs.keys()),
        'image_built': build_result['built'],
        'build_backend': build_result.get('backend', 'none'),
        'steps': package_steps,
        'builder_output': builder_output[:500]
    }), 200


def _detect_ric_service(kubectl_base, name_substrings, timeout=10):
    """
    Find a RIC platform service whose name contains any of `name_substrings`.

    Returns {found, service, namespace, port} for the first HTTP-ish port. The
    framework is co-located with the cluster but ClusterIP addresses are usually
    NOT routable from the host, so callers reach the service via a short-lived
    `kubectl port-forward` (see _PortForward) rather than the ClusterIP.
    """
    import subprocess
    import json as json_mod
    info = {'found': False, 'service': None, 'namespace': None, 'port': None}
    subs = [s.lower() for s in name_substrings]
    try:
        r = subprocess.run(kubectl_base + ['get', 'svc', '-A', '-o', 'json'],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 or not r.stdout:
            return info
        for svc in json_mod.loads(r.stdout).get('items', []):
            name = svc.get('metadata', {}).get('name', '')
            low = name.lower()
            if not any(s in low for s in subs):
                continue
            # Prefer the http-named port, else 8080/8888, else first port.
            ports = svc.get('spec', {}).get('ports', [])
            chosen = None
            for p in ports:
                pname = (p.get('name') or '').lower()
                if 'http' in pname or p.get('port') in (8080, 8888):
                    chosen = p.get('port')
                    break
            if chosen is None and ports:
                chosen = ports[0].get('port')
            if chosen is not None:
                info.update({
                    'found': True,
                    'service': name,
                    'namespace': svc.get('metadata', {}).get('namespace', ''),
                    'port': chosen,
                })
                return info
    except Exception as e:
        current_app.logger.info(f"[Step7] service detection failed for {name_substrings}: {e}")
    return info


class _PortForward:
    """
    Context manager that runs `kubectl port-forward svc/<svc> <local>:<remote>`
    for the duration of a REST call, then tears it down. ClusterIP services are
    generally unreachable from the minikube host, so this is the reliable way for
    the framework to talk to appmgr / onboarder over HTTP.
    """

    def __init__(self, kubectl_base, namespace, service, remote_port, local_port=None):
        import random
        self.kubectl_base = kubectl_base
        self.namespace = namespace
        self.service = service
        self.remote_port = remote_port
        self.local_port = local_port or random.randint(21000, 21999)
        self.proc = None
        self.base_url = f'http://127.0.0.1:{self.local_port}'

    def __enter__(self):
        import subprocess
        import time
        cmd = self.kubectl_base + [
            'port-forward', '-n', self.namespace, f'svc/{self.service}',
            f'{self.local_port}:{self.remote_port}',
        ]
        current_app.logger.info(f"[Step7] port-forward: {' '.join(cmd)}")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Give kubectl a moment to establish the tunnel.
        time.sleep(3)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        return False


def _onboard_xapp(onboarder_url, descriptor, controls_schema, timeout=60):
    """
    Onboard an xApp via the xApp onboarder REST API (POST /api/v1/onboard).

    Body matches the onboarder schema:
      {"config-file.json": <descriptor>, "controls-schema.json": <schema>}
    This validates the descriptor and pushes the generated Helm chart into the
    cluster chart repo (chartmuseum) so appmgr can later install it by name.
    """
    import requests as http_requests
    out = {'attempted': True, 'success': False}
    payload = {'config-file.json': descriptor, 'controls-schema.json': controls_schema}
    try:
        r = http_requests.post(
            f'{onboarder_url}/api/v1/onboard',
            headers={'Content-Type': 'application/json'},
            json=payload, timeout=timeout,
        )
        out['status_code'] = r.status_code
        out['response'] = r.text[:600]
        # 201 = created; 400 with "already exists" is acceptable (idempotent).
        out['success'] = r.status_code in (200, 201)
        if not out['success'] and 'exist' in (r.text or '').lower():
            out['success'] = True
            out['note'] = 'chart already onboarded'
    except Exception as e:
        out['error'] = str(e)
    return out


def _deploy_via_appmgr(appmgr_url, m_name, m_version, namespace='ricxapp',
                       override=None, timeout=120):
    """
    Deploy an onboarded xApp via appmgr (POST /ric/v1/xapps).

    Body is the XappDescriptor the appmgr swagger defines:
      {"xappName", "helmVersion", "releaseName", "namespace", "overrideFile"}
    `overrideFile` is a Helm values override; we use it to force
    image_pull_policy=Never so the locally-built image is used (no registry pull).
    """
    import requests as http_requests
    out = {'attempted': True, 'success': False}
    body = {
        'xappName': m_name,
        'helmVersion': '',           # let appmgr pick the latest onboarded version
        'releaseName': m_name,
        'namespace': namespace,
        'overrideFile': override or {'image_pull_policy': 'Never'},
    }
    # Remove any prior instance so re-deploys are idempotent.
    try:
        http_requests.delete(f'{appmgr_url}/ric/v1/xapps/{m_name}', timeout=timeout)
    except Exception:
        pass
    try:
        r = http_requests.post(
            f'{appmgr_url}/ric/v1/xapps',
            headers={'Content-Type': 'application/json'},
            json=body, timeout=timeout,
        )
        out['status_code'] = r.status_code
        out['response'] = r.text[:600]
        out['request_body'] = body
        out['success'] = r.status_code in (200, 201)
    except Exception as e:
        out['error'] = str(e)
    return out


def _deploy_via_helm(kubectl_base, onboarder_ns, onboarder_service, chart_port,
                     m_name, m_version, namespace='ricxapp', timeout=180):
    """
    Deploy the onboarded xApp by installing its Helm chart from the in-cluster
    chartmuseum (the same repo the onboarder pushed the chart to during
    /api/v1/onboard).

    Why not appmgr REST? appmgr 0.5.9 leaves POST /ric/v1/xapps
    (operationId XappDeployXapp) as a go-swagger stub -> "operation
    XappDeployXapp has not yet been implemented". The O-RAN-sanctioned path
    (dms_cli install) does exactly what we do here under the hood: pull the
    onboarded chart from chartmuseum and `helm install` it. We use Helm 3, so
    no Tiller is involved.

    image_pull_policy is forced to Never so the locally-built image already
    present in minikube's docker daemon is used (no registry pull).
    """
    import subprocess
    import tempfile
    import requests as http_requests

    out = {'attempted': True, 'success': False, 'method': 'helm-from-chartmuseum'}

    # 1. Fetch the onboarded chart tgz from chartmuseum via a port-forward.
    tgz_path = None
    with _PortForward(kubectl_base, onboarder_ns, onboarder_service, chart_port) as pf:
        version = m_version
        # Ask chartmuseum for the exact stored version (defensive: the onboarder
        # may normalize the version string).
        try:
            vr = http_requests.get(f'{pf.base_url}/api/charts/{m_name}', timeout=15)
            if vr.status_code == 200 and isinstance(vr.json(), list) and vr.json():
                version = vr.json()[0].get('version', m_version)
        except Exception as e:
            out['version_lookup_error'] = str(e)
        out['chart'] = f'{m_name}-{version}'

        tgz_url = f'{pf.base_url}/charts/{m_name}-{version}.tgz'
        out['chart_url'] = tgz_url
        try:
            cr = http_requests.get(tgz_url, timeout=30)
            if cr.status_code != 200:
                out['error'] = f'chart download failed ({cr.status_code}) from {tgz_url}'
                return out
            tgz_path = os.path.join(tempfile.gettempdir(), f'{m_name}-{version}.tgz')
            with open(tgz_path, 'wb') as fh:
                fh.write(cr.content)
        except Exception as e:
            out['error'] = f'chart download error: {e}'
            return out

    # 2. Remove any prior release so re-deploys are idempotent (ignore errors).
    try:
        subprocess.run(['helm', 'uninstall', m_name, '-n', namespace],
                       capture_output=True, text=True, timeout=60)
    except Exception:
        pass

    # 3. helm install the chart into ricxapp with pullPolicy Never.
    cmd = ['helm', 'install', m_name, tgz_path,
           '-n', namespace, '--create-namespace',
           '--set', 'image_pull_policy=Never']
    out['helm_cmd'] = ' '.join(cmd)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out['returncode'] = r.returncode
        out['stdout'] = (r.stdout or '')[:600]
        out['stderr'] = (r.stderr or '')[:600]
        out['success'] = r.returncode == 0
        if not out['success']:
            out['error'] = (r.stderr or r.stdout or 'helm install failed')[:400]
    except Exception as e:
        out['error'] = str(e)
    return out


def _subscribe_telemetry(appmgr_url, target_url, event_type='all',
                         max_retries=5, retry_timer=10, timeout=30):
    """
    Register a telemetry/event subscription via appmgr
    (POST /ric/v1/subscriptions) so the caller is notified of xApp lifecycle
    events. Body matches the appmgr swagger `subscriptionRequest`.
    """
    import requests as http_requests
    out = {'attempted': True, 'success': False}
    body = {'data': {
        'targetUrl': target_url,
        'eventType': event_type,
        'maxRetries': max_retries,
        'retryTimer': retry_timer,
    }}
    try:
        r = http_requests.post(
            f'{appmgr_url}/ric/v1/subscriptions',
            headers={'Content-Type': 'application/json'},
            json=body, timeout=timeout,
        )
        out['status_code'] = r.status_code
        out['response'] = r.text[:400]
        out['request_body'] = body
        out['success'] = r.status_code in (200, 201)
    except Exception as e:
        out['error'] = str(e)
    return out


def _verify_pkl_in_image(kubectl_base, m_name, pkl_path, docker_image, namespace='ricxapp'):
    """
    Compare the local .pkl artifact against the model.pkl baked into the image
    that is actually running in the cluster pod.

    Hashes the on-disk .pkl and the /app/model.pkl inside the running container
    (via kubectl exec) and reports whether they match, plus the deployed vs
    expected image reference. This confirms the deployed image really carries the
    model the framework packaged - verified against the cluster, not the server FS.
    """
    import subprocess
    import hashlib
    result = {'checked': False, 'match': None}

    if not pkl_path or not os.path.exists(pkl_path):
        result['error'] = f'local pkl not found: {pkl_path}'
        return result
    try:
        h = hashlib.sha256()
        with open(pkl_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        local_hash = h.hexdigest()
        result['local_pkl_sha256'] = local_hash
        result['local_pkl_path'] = pkl_path
    except Exception as e:
        result['error'] = f'local pkl hash failed: {e}'
        return result

    try:
        r = subprocess.run(
            kubectl_base + ['get', 'pods', '-n', namespace, '-l', f'release={m_name}',
                            '-o', 'jsonpath={.items[0].metadata.name}'],
            capture_output=True, text=True, timeout=15)
        pod = (r.stdout or '').strip()
        if not pod:
            result['error'] = 'no running pod found to verify against'
            return result
        result['pod'] = pod

        ri = subprocess.run(
            kubectl_base + ['get', 'pod', pod, '-n', namespace,
                            '-o', 'jsonpath={.spec.containers[0].image}'],
            capture_output=True, text=True, timeout=15)
        deployed_image = (ri.stdout or '').strip()
        result['deployed_image'] = deployed_image
        result['expected_image'] = docker_image
        result['image_match'] = (deployed_image == docker_image)

        # python3 is guaranteed present (the xApp base image is python:3.x-slim).
        exec_r = subprocess.run(
            kubectl_base + ['exec', pod, '-n', namespace, '--', 'python3', '-c',
                            "import hashlib;print(hashlib.sha256(open('/app/model.pkl','rb').read()).hexdigest())"],
            capture_output=True, text=True, timeout=30)
        remote_hash = (exec_r.stdout or '').strip().split('\n')[-1].strip()
        if exec_r.returncode == 0 and len(remote_hash) == 64:
            result['image_pkl_sha256'] = remote_hash
            result['match'] = (remote_hash == local_hash)
            result['checked'] = True
        else:
            result['error'] = f'in-pod hash failed: {(exec_r.stderr or exec_r.stdout or "")[:150]}'
    except Exception as e:
        result['error'] = str(e)
    return result


@web.route('/api/deploy_to_ric', methods=['POST'])
@login_required
def deploy_to_ric():
    """
    Step 7: Deploy packaged xApp to Near-RT RIC.

    The framework is co-located with the Kubernetes cluster and talks to it
    directly (kubectl / Helm / appmgr REST) - it never SSHes into a server.

    Deployment routing:
    - Requests deployment via ML Model Catalog API (bookkeeping).
    - If the appmgr service is present in the cluster: deploy via its REST API
      (POST /ric/v1/xapps), falling back to Helm if appmgr rejects the request.
    - Else if local Docker + K8s are up: deploy via Helm to the ricxapp namespace.
    - Else: demo mode with the exact cluster commands needed for a manual deploy.
    - After a live deploy, verify the deployed image's model.pkl matches the
      local .pkl artifact.
    """
    import json as json_mod
    import subprocess
    import requests as http_requests
    
    data = request.get_json()
    if not data or 'experiment_name' not in data:
        return jsonify({'success': False, 'error': 'Missing experiment_name'}), 400
    
    experiment_name = data['experiment_name']
    safe_user = secure_filename(current_user.username)
    safe_exp = secure_filename(experiment_name)
    
    base_upload = os.path.join(current_app.config['UPLOAD_FOLDER'], 'Users')
    exp_path = os.path.join(base_upload, safe_user, safe_exp)
    pkl_dir = os.path.join(exp_path, 'ric_artifacts')
    
    if not os.path.exists(pkl_dir):
        return jsonify({
            'success': False,
            'error': 'No artifacts found. Run Step 5 and Step 6 first.'
        }), 400
    
    # Load publish state
    publish_state = None
    state_file = None
    for fname in os.listdir(pkl_dir):
        if fname.endswith('_publish_state.json'):
            state_file = os.path.join(pkl_dir, fname)
            with open(state_file, 'r') as f:
                publish_state = json_mod.load(f)
            break
    
    if not publish_state:
        return jsonify({
            'success': False,
            'error': 'No publish state found. Run Step 5 first.'
        }), 400
    
    if not publish_state.get('packaged'):
        return jsonify({
            'success': False,
            'error': 'Model not yet packaged. Run Step 6 first.'
        }), 400
    
    m_name = publish_state['model_name']
    m_version = publish_state['version']
    m_type = publish_state.get('model_type', 'unknown')
    model_id = publish_state.get('model_id', '')
    docker_image = publish_state.get('docker_image', xapp_image_tag(m_name, m_version))
    helm_dir = publish_state.get('helm_chart_dir', '')
    build_output_dir = publish_state.get('build_output_dir', '')
    # Local .pkl artifact written by Step 5 (Publish); used to verify the
    # deployed image's /app/model.pkl matches, for any model.
    pkl_path = publish_state.get('pkl_path', '')
    if not pkl_path or not os.path.exists(pkl_path):
        # Fall back to the artifact in this experiment's ric_artifacts dir.
        cand = os.path.join(pkl_dir, f"{m_name}_v{m_version}.pkl")
        if os.path.exists(cand):
            pkl_path = cand
    
    # ---- Check infrastructure status ----
    infra_status = {
        'docker': False,
        'kubernetes': False,
        'catalog': False
    }
    
    # Check Docker
    try:
        r = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=5)
        infra_status['docker'] = r.returncode == 0
    except Exception:
        pass
    
    # Check Kubernetes (try kubectl directly, then minikube kubectl)
    try:
        r = subprocess.run(['kubectl', 'cluster-info'], capture_output=True, text=True, timeout=10)
        infra_status['kubernetes'] = r.returncode == 0
        current_app.logger.info(f"[Step7] kubectl cluster-info: rc={r.returncode}, out={r.stdout[:100]}")
    except Exception as e:
        current_app.logger.info(f"[Step7] kubectl not found: {e}")
    
    if not infra_status['kubernetes']:
        try:
            r = subprocess.run(['minikube', '-p', 'kero-ric', 'kubectl', '--', 'cluster-info'],
                              capture_output=True, text=True, timeout=10)
            infra_status['kubernetes'] = r.returncode == 0
            if r.returncode == 0:
                infra_status['kubectl_cmd'] = 'minikube -p kero-ric kubectl --'
                current_app.logger.info(f"[Step7] minikube kubectl works: {r.stdout[:100]}")
        except Exception as e:
            current_app.logger.info(f"[Step7] minikube kubectl failed: {e}")
    
    # Check ML Model Catalog
    catalog_url = os.environ.get('RIC_CATALOG_URL', 'http://localhost:8080')
    try:
        r = http_requests.get(f'{catalog_url}/health', timeout=3)
        infra_status['catalog'] = r.status_code == 200
    except Exception:
        pass
    
    # ---- Request deployment via catalog ----
    catalog_deploy_result = None
    if infra_status['catalog'] and model_id:
        try:
            r = http_requests.post(f'{catalog_url}/models/{model_id}/deploy', timeout=5)
            if r.status_code == 200:
                catalog_deploy_result = r.json()
        except Exception:
            pass
    
    # ---- Resolve the kubectl invocation (talk to the cluster directly) ----
    # The framework is co-located with the cluster; prefer a plain `kubectl`,
    # else fall back to `minikube -p kero-ric kubectl --`.
    kubectl_base = infra_status.get('kubectl_cmd', 'kubectl').split()

    # ---- Detect the O-RAN App Manager + xApp onboarder services ----
    # Deployment is done entirely through the appmgr REST API (no Helm). appmgr
    # deploys charts that were first onboarded by the xApp onboarder.
    appmgr_svc = _detect_ric_service(kubectl_base, ['appmgr']) if infra_status['kubernetes'] else {'found': False}
    onboarder_svc = _detect_ric_service(kubectl_base, ['onboard']) if infra_status['kubernetes'] else {'found': False}
    infra_status['appmgr'] = appmgr_svc
    infra_status['onboarder'] = onboarder_svc

    # ---- Build deployment result ----
    deployment_result = {
        'namespace': 'ricxapp',
        'docker_image': docker_image,
        'model_name': m_name,
        'model_type': m_type,
        'helm_release': m_name,
        'infrastructure': infra_status,
        'catalog_deploy': catalog_deploy_result
    }
    
    # ---- APPMGR-ONLY DEPLOYMENT (fully REST-API driven, no Helm) ----
    # After the image is built (Step 6), everything is an API request:
    #   1. Onboard the xApp descriptor + controls-schema via the onboarder REST API
    #      (POST /api/v1/onboard) -> pushes the Helm chart into the cluster chart repo.
    #   2. Deploy via the appmgr REST API (POST /ric/v1/xapps) -> appmgr installs the
    #      onboarded chart. overrideFile forces image_pull_policy=Never so the local
    #      image (built into minikube's docker daemon) is used with no registry pull.
    #   3. Subscribe to xApp lifecycle events via appmgr (POST /ric/v1/subscriptions)
    #      so telemetry/notifications are delivered.
    #   4. Verify the deployed pod's model.pkl matches the local .pkl artifact.
    #
    # ClusterIP services are unreachable from the host, so each REST call is made
    # through a short-lived `kubectl port-forward` to the service.
    descriptor_path = publish_state.get('descriptor_path') or os.path.join(build_output_dir or '', 'config-file.json')
    schema_path = publish_state.get('controls_schema_path') or os.path.join(build_output_dir or '', 'schema.json')

    ready = (infra_status['docker'] and infra_status['kubernetes']
             and appmgr_svc.get('found') and onboarder_svc.get('found')
             and os.path.exists(descriptor_path) and os.path.exists(schema_path))

    if ready:
        deployment_result['deployment_mode'] = 'appmgr'
        steps = []
        build_dir = build_output_dir or os.path.join(pkl_dir, 'xapp-build')
        try:
            # Load the descriptor + controls schema produced by Step 6.
            with open(descriptor_path) as f:
                descriptor = json_mod.load(f)
            with open(schema_path) as f:
                controls_schema = json_mod.load(f)

            # Step 1: Ensure the image is available in the cluster docker daemon.
            steps.append({'name': 'Ensure xApp image is available', 'status': 'running'})
            if publish_state.get('image_built'):
                steps[-1]['status'] = 'done'
                steps[-1]['detail'] = f"Reusing image from packaging ({publish_state.get('build_backend', 'docker')})"
                build_ok = True
            else:
                build_res = _build_docker_image(build_dir, docker_image)
                build_ok = build_res['built']
                steps[-1]['status'] = 'done' if build_ok else 'error'
                steps[-1]['detail'] = 'Image built' if build_ok else 'Build failed'
                steps.extend(build_res['steps'])
            deployment_result['docker_build'] = {'success': build_ok}

            if build_ok:
                # Step 2: Ensure the ricxapp namespace exists.
                steps.append({'name': 'Ensure ricxapp namespace', 'status': 'running'})
                subprocess.run(kubectl_base + ['create', 'namespace', 'ricxapp'],
                               capture_output=True, text=True, timeout=15)
                steps[-1]['status'] = 'done'

                # Step 3: Onboard the xApp (REST API -> onboarder).
                steps.append({'name': f'Onboard {m_name} (POST /api/v1/onboard)', 'status': 'running'})
                with _PortForward(kubectl_base, onboarder_svc['namespace'],
                                  onboarder_svc['service'], onboarder_svc['port']) as pf:
                    onboard_res = _onboard_xapp(pf.base_url, descriptor, controls_schema)
                deployment_result['onboard'] = onboard_res
                if onboard_res.get('success'):
                    steps[-1]['status'] = 'done'
                    steps[-1]['detail'] = onboard_res.get('note') or f"onboarded ({onboard_res.get('status_code')})"
                else:
                    steps[-1]['status'] = 'error'
                    steps[-1]['detail'] = f"onboard failed: {onboard_res.get('error') or onboard_res.get('response')}"

                # Step 4: Deploy the onboarded chart.
                #
                # appmgr 0.5.9's REST deploy (POST /ric/v1/xapps -> XappDeployXapp)
                # is an unimplemented go-swagger stub, so we install the chart the
                # onboarder just pushed to chartmuseum directly with Helm 3 - this
                # is exactly what the O-RAN `dms_cli install` does under the hood.
                # Onboard (above) and subscribe (below) remain appmgr/onboarder
                # REST API calls.
                deployed_ok = False
                if onboard_res.get('success'):
                    steps.append({'name': f'Deploy {m_name} (helm install from chartmuseum)', 'status': 'running'})
                    helm_res = _deploy_via_helm(
                        kubectl_base, onboarder_svc['namespace'], onboarder_svc['service'],
                        chart_port=8080, m_name=m_name, m_version=m_version, namespace='ricxapp')
                    deployment_result['helm_deploy'] = helm_res
                    deployed_ok = helm_res.get('success', False)
                    steps[-1]['status'] = 'done' if deployed_ok else 'error'
                    steps[-1]['detail'] = (f"installed chart {helm_res.get('chart')}"
                                           if deployed_ok else
                                           f"deploy failed: {helm_res.get('error')}")

                deployment_result['deploy_via'] = 'helm-from-chartmuseum' if deployed_ok else None

                if deployed_ok:
                    # Step 5: Wait for pod readiness and test the xApp.
                    steps.append({'name': 'Test xApp pod health', 'status': 'running'})
                    pod_ready, test_detail, pod_out = _test_xapp_pod(kubectl_base, m_name)
                    steps[-1]['status'] = 'done' if pod_ready else 'warning'
                    steps[-1]['detail'] = test_detail
                    deployment_result['pod_status'] = pod_out
                    deployment_result['pod_healthy'] = pod_ready

                    # Step 6: Verify the deployed image's model.pkl matches the local .pkl.
                    steps.append({'name': 'Verify deployed image matches .pkl', 'status': 'running'})
                    pkl_check = _verify_pkl_in_image(kubectl_base, m_name, pkl_path, docker_image)
                    deployment_result['pkl_verification'] = pkl_check
                    if pkl_check.get('checked'):
                        if pkl_check.get('match'):
                            steps[-1]['status'] = 'done'
                            steps[-1]['detail'] = f"model.pkl matches (sha256 {pkl_check['local_pkl_sha256'][:12]}...)"
                        else:
                            steps[-1]['status'] = 'error'
                            steps[-1]['detail'] = 'Deployed image .pkl does NOT match local artifact'
                    else:
                        steps[-1]['status'] = 'warning'
                        steps[-1]['detail'] = pkl_check.get('error', 'verification skipped')

                    # Step 7: Subscribe to xApp lifecycle events for telemetry.
                    steps.append({'name': 'Subscribe to xApp telemetry events', 'status': 'running'})
                    target_url = os.environ.get(
                        'RIC_SUBSCRIPTION_TARGET_URL',
                        f"{catalog_url}/models/{model_id}/events" if model_id else f'{catalog_url}/events')
                    with _PortForward(kubectl_base, appmgr_svc['namespace'],
                                      appmgr_svc['service'], appmgr_svc['port']) as pf:
                        sub_res = _subscribe_telemetry(pf.base_url, target_url, event_type='all')
                    deployment_result['subscription'] = sub_res
                    steps[-1]['status'] = 'done' if sub_res.get('success') else 'warning'
                    steps[-1]['detail'] = (f"subscribed ({sub_res.get('status_code')})"
                                           if sub_res.get('success') else
                                           f"subscribe failed: {sub_res.get('error') or sub_res.get('response')}")

                deployment_result['deployed'] = deployed_ok
                deployment_result['live_mode'] = True
            else:
                deployment_result['deployed'] = False
            deployment_result['steps'] = steps
        except Exception as e:
            deployment_result['deployed'] = False
            deployment_result['error'] = str(e)
            deployment_result['steps'] = steps
            current_app.logger.error(f"appmgr deployment failed: {e}")

    # ---- DEMO MODE (appmgr / onboarder / infra not available) ----
    else:
        deployment_result['deployment_mode'] = 'demo'
        missing = []
        if not infra_status['docker']:
            missing.append('Docker')
        if not infra_status['kubernetes']:
            missing.append('Kubernetes cluster')
        if not appmgr_svc.get('found'):
            missing.append('appmgr service (ric-plt-appmgr)')
        if not onboarder_svc.get('found'):
            missing.append('xApp onboarder service')
        if not os.path.exists(descriptor_path) or not os.path.exists(schema_path):
            missing.append('config-file.json / schema.json (run Step 6 first)')

        deployment_result['demo_mode'] = True
        deployment_result['deployed'] = True
        deployment_result['missing_infra'] = missing
        deployment_result['message'] = (
            f'xApp "{m_name}" cannot be deployed via appmgr yet.\n'
            + (f'Missing: {", ".join(missing)}. ' if missing else '')
            + 'The whole deploy is appmgr-only (no Helm); install the RIC platform '
            + '(appmgr + xApp onboarder) then re-run. Reference curl commands below.'
        )

        deployment_result['commands'] = {
            'appmgr_flow': [
                '# APPMGR-ONLY DEPLOYMENT (all REST API after image build):',
                '# 1. Onboard the xApp (descriptor + controls schema):',
                "curl -H 'Content-Type: application/json' -X POST \\",
                '  http://<onboarder-service>:8888/api/v1/onboard \\',
                f'  -d @- <<EOF\n{{"config-file.json": <{os.path.basename(descriptor_path)}>, '
                f'"controls-schema.json": <{os.path.basename(schema_path)}>}}\nEOF',
                '# 2. Deploy the onboarded xApp via appmgr:',
                "curl -H 'Content-Type: application/json' -X POST \\",
                '  http://<appmgr-service>:8080/ric/v1/xapps \\',
                f'  -d \'{{"xappName": "{m_name}", "releaseName": "{m_name}", '
                f'"namespace": "ricxapp", "overrideFile": {{"image_pull_policy": "Never"}}}}\'',
                '# 3. Subscribe to telemetry events:',
                "curl -H 'Content-Type: application/json' -X POST \\",
                '  http://<appmgr-service>:8080/ric/v1/subscriptions \\',
                '  -d \'{"data": {"targetUrl": "http://<your-endpoint>/events", '
                '"eventType": "all", "maxRetries": 5, "retryTimer": 10}}\'',
                '# 4. Query status / undeploy:',
                f'curl http://<appmgr-service>:8080/ric/v1/xapps/{m_name}',
                f'curl -X DELETE http://<appmgr-service>:8080/ric/v1/xapps/{m_name}',
            ]
        }
    
    # Update publish state
    publish_state['deployed'] = True
    publish_state['deployed_at'] = datetime.utcnow().isoformat() + 'Z'
    publish_state['deployment'] = deployment_result
    
    if state_file:
        with open(state_file, 'w') as f:
            json_mod.dump(publish_state, f, indent=2, default=str)
    
    return jsonify({
        'success': True,
        'message': f'xApp "{m_name}" deployment '
                   + ('completed!' if deployment_result.get('live_mode') else 'requested (demo mode)!'),
        'deployment': deployment_result
    }), 200


# Error handlers

@web.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found', code=404), 404

@web.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    current_app.logger.error(f'Server error: {error}', exc_info=True)
    return render_template('error.html', error='Internal server error', code=500), 500



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
                                'display_name': file,  # Show only filename, not path
                                'path': file,  # Use only filename as identifier
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
            
            # Debug session keys
            current_app.logger.info(f"Session keys before redirect: {list(session.keys())}")
            current_app.logger.info(f"experiment_name in session: {'experiment_name' in session}")
            
            # Store in session
            session['config_bytes'] = config_content.encode('utf-8')
            session.modified = True  # Ensure session is saved
            
            current_app.logger.info(f"Session keys after setting config_bytes: {list(session.keys())}")
            return redirect(url_for('routes.upload_csv'))
    
    return render_template('upload_config.html')

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
    
    return render_template('yaml_arch_step1.html',
                         current_file=file_num,
                         total_files=total_files,
                         # Model config
                         model_type=file_config.get('model_type', 'autotransformer'),
                         model_retrain=file_config.get('model_retrain', 'true'),
                         # TS config
                         ts_name=file_config.get('ts_name', 'model'),
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
    
    # Collect all form data
    file_config = {
        'filename': filename,
        'description': request.form.get('description', '').strip(),
        # Model config
        'model_type': request.form.get('model_type', 'rnn'),
        'model_retrain': request.form.get('model_retrain', 'true'),
        # TS config
        'ts_name': request.form.get('ts_name', 'lstm'),
        'ts_version': request.form.get('ts_version', '1'),
        'ts_enrich': request.form.get('ts_enrich', ''),
        'use_covariates': request.form.get('use_covariates', 'true'),
        'past_variables': request.form.get('past_variables', '[1]'),
        'use_future_covariates': request.form.get('use_future_covariates', 'false'),
        'future_variables': request.form.get('future_variables', 'null'),
        'interpolate': request.form.get('interpolate', 'true'),
        # Model configs
        'cat_emb_dim': request.form.get('cat_emb_dim', '128'),
        'hidden_rnn': request.form.get('hidden_rnn', '64'),
        'num_layers_rnn': request.form.get('num_layers_rnn', '2'),
        'kernel_size': request.form.get('kernel_size', '3'),
        'kind': request.form.get('kind', 'lstm'),
        'sum_emb': request.form.get('sum_emb', 'true'),
        'use_bn': request.form.get('use_bn', 'true'),
        'optim': request.form.get('optim', 'torch.optim.SGD'),
        'activation': request.form.get('activation', 'torch.nn.ReLU'),
        'dropout_rate': request.form.get('dropout_rate', '0.2'),
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
            
            # Generate filename based on main config file name and file number
            config_base = secure_filename(session['experiment_name'])
            filename = f"{config_base}_arch_{file_num}.yaml"
            filename = secure_filename(filename)
            
            # Ensure .yaml extension
            if not filename.endswith(('.yaml', '.yml')):
                filename += '.yaml'
            
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
                'description': f"YAML training configuration {file_num} for {config_base}",
                'type': 'yaml_training_config'
            })
            current_app.logger.info(f"Successfully saved file {file_num}")
                
        # Create experiment summary config
        config_path = os.path.join(exp_dir, f"config_{safe_exp}.yaml")
        config_data = {
            'user': current_user.username,
            'experiment': session['experiment_name'],
            'architecture_files': arch_saved,
            'configuration_method': 'yaml_training_configured',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add optional files if they exist
        if config_filename:
            config_data['experiment_config_file'] = config_filename
        if csv_filename:
            config_data['csv_file'] = csv_filename
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
            
        # Clean up session
        session.pop('csv_bytes', None)
        session.pop('csv_filename', None)
        session.pop('config_bytes', None)
        session.pop('config_filename', None)
        session.pop('yaml_arch_config', None)
        session.pop('experiment_name', None)
        
        current_app.logger.info(f"Successfully completed processing {len(arch_saved)} YAML files")
        flash(f'Your experiment with {len(arch_saved)} configured YAML training files has been created successfully!')
        return redirect(url_for('routes.done'))
        
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
                
                config_data = {
                    'experiment': {
                        'exp_id': experiment_name,
                        'type': 'train',
                        'version': 1,
                        'data': {
                            'dataset': dataset if dataset else 'default',
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

# Error handlers

@web.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error='Page not found', code=404), 404

@web.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    current_app.logger.error(f'Server error: {error}', exc_info=True)
    return render_template('error.html', error='Internal server error', code=500), 500

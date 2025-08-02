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

"""
API v1 routes for the CSV to YAML Converter application.

This module defines the API endpoints for version 1 of the API.
"""

import os
from flask import request, jsonify, current_app, send_from_directory, url_for
from werkzeug.utils import secure_filename
from flask_restx import Resource, Namespace, fields, reqparse
from .validators import (
    validate_json, validate_file_extension, sanitize_filename,
    validate_experiment_name, validate_username, validate_upload_folder,
    validate_api_key
)

# Create a namespace for our API
ns = Namespace('converter', description='CSV to YAML conversion operations')

# Request models
csv_upload_parser = reqparse.RequestParser()
csv_upload_parser.add_argument('file', 
                             type=request.files.get('file'), 
                             location='files',
                             required=True,
                             help='CSV file to convert')
csv_upload_parser.add_argument('username',
                             type=str,
                             required=True,
                             help='Username for multi-tenancy')
csv_upload_parser.add_argument('experiment',
                             type=str,
                             required=True,
                             help='Experiment name')

# Response models
conversion_result = ns.model('ConversionResult', {
    'status': fields.String(required=True, description='Conversion status'),
    'message': fields.String(description='Result message'),
    'yaml': fields.String(description='Converted YAML content'),
    'download_url': fields.Url(description='URL to download the YAML file')
})

error_model = ns.model('Error', {
    'error': fields.String(required=True, description='Error message'),
    'status': fields.Integer(description='HTTP status code')
})

@ns.route('/convert')
class CsvToYamlConverter(Resource):
    """Convert CSV to YAML"""
    
    @ns.doc('convert_csv_to_yaml')
    @ns.expect(csv_upload_parser)
    @ns.response(200, 'Success', conversion_result)
    @ns.response(400, 'Bad Request', error_model)
    @ns.response(401, 'Unauthorized', error_model)
    @ns.response(500, 'Internal Server Error', error_model)
    def post(self):
        """
        Convert a CSV file to YAML format.
        
        This endpoint accepts a CSV file and returns the converted YAML.
        The file is also saved on the server for future reference.
        """
        # Check API key
        api_key = request.headers.get('X-API-Key')
        if not validate_api_key(api_key):
            return {'error': 'Invalid or missing API key', 'status': 401}, 401
        
        # Get form data
        if 'file' not in request.files:
            return {'error': 'No file part', 'status': 400}, 400
            
        file = request.files['file']
        username = request.form.get('username')
        experiment = request.form.get('experiment')
        
        # Validate inputs
        if not file or file.filename == '':
            return {'error': 'No selected file', 'status': 400}, 400
            
        is_valid, error_msg = validate_username(username)
        if not is_valid:
            return {'error': f'Invalid username: {error_msg}', 'status': 400}, 400
            
        is_valid, error_msg = validate_experiment_name(experiment)
        if not is_valid:
            return {'error': f'Invalid experiment name: {error_msg}', 'status': 400}, 400
            
        # Validate file extension
        if not validate_file_extension(file.filename, {'csv'}):
            return {'error': 'Invalid file type. Only CSV files are allowed.', 'status': 400}, 400
        
        # Sanitize filename
        filename = sanitize_filename(file.filename)
        if not filename:
            return {'error': 'Invalid filename', 'status': 400}, 400
        
        try:
            # Ensure upload directories exist
            base_dir = os.path.join(current_app.config['BASE_UPLOAD_FOLDER'], username, 'Experiment', experiment)
            os.makedirs(base_dir, exist_ok=True)
            
            # Save the uploaded file
            filepath = os.path.join(base_dir, f"{experiment}.csv")
            file.save(filepath)
            
            # Convert CSV to YAML (this would be your existing conversion logic)
            from app import csv_to_yaml
            yaml_data, error = csv_to_yaml(filepath)
            
            if error:
                return {'error': f'Conversion failed: {error}', 'status': 400}, 400
            
            # Save the YAML file
            yaml_filename = f"config_{experiment}.yaml"
            yaml_path = os.path.join(base_dir, yaml_filename)
            with open(yaml_path, 'w', encoding='utf-8') as f:
                f.write(yaml_data)
            
            # Create a download URL
            download_url = url_for('api.download_file', 
                                 username=username, 
                                 experiment=experiment, 
                                 filename=yaml_filename,
                                 _external=True)
            
            return {
                'status': 'success',
                'message': 'File successfully converted',
                'yaml': yaml_data,
                'download_url': download_url
            }
            
        except Exception as e:
            current_app.logger.error(f'Error processing file: {str(e)}', exc_info=True)
            return {'error': 'An error occurred during conversion', 'status': 500}, 500

# Register the namespace with the API
# This is done in the v1/__init__.py file

def init_app(app, api):
    """Initialize the API with the Flask app."""
    api.add_namespace(ns, path='')
    
    # Add other namespaces here if needed
    # from .other_module import ns as other_ns
    # api.add_namespace(other_ns, path='/other')

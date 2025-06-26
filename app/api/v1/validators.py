"""
Request validation utilities for the API v1.

This module provides functions for validating and sanitizing API request data.
"""
import re
import os
from functools import wraps
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename

def validate_json(f):
    ""
    Decorator to ensure the request contains valid JSON.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({
                'error': 'Invalid content type. Expected application/json',
                'status': 400
            }), 400
        return f(*args, **kwargs)
    return decorated_function

def validate_file_extension(filename, allowed_extensions=None):
    """
    Validate that a filename has an allowed extension.
    
    Args:
        filename (str): The name of the file to validate
        allowed_extensions (set, optional): Set of allowed file extensions.
                                          Defaults to {'csv', 'yaml', 'yml'}.
    
    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    if allowed_extensions is None:
        allowed_extensions = {'csv', 'yaml', 'yml'}
    
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def sanitize_filename(filename):
    """
    Sanitize a filename to prevent directory traversal and other security issues.
    
    Args:
        filename (str): The filename to sanitize.
    
    Returns:
        str: The sanitized filename.
    """
    # Secure the filename
    filename = secure_filename(filename)
    
    # Additional validation to prevent directory traversal
    if '..' in filename or filename.startswith('/') or '~' in filename:
        return None
        
    return filename

def validate_experiment_name(name):
    """
    Validate an experiment name.
    
    Args:
        name (str): The experiment name to validate.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not name or not isinstance(name, str):
        return False, 'Experiment name must be a non-empty string'
    
    if len(name) > 100:
        return False, 'Experiment name must be 100 characters or less'
    
    # Only allow alphanumeric, hyphen, and underscore
    if not re.match(r'^[a-zA-Z0-9_-]+$', name):
        return False, 'Experiment name can only contain letters, numbers, hyphens, and underscores'
    
    return True, ''

def validate_username(username):
    """
    Validate a username.
    
    Args:
        username (str): The username to validate.
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if not username or not isinstance(username, str):
        return False, 'Username must be a non-empty string'
    
    if len(username) > 50:
        return False, 'Username must be 50 characters or less'
    
    # Only allow alphanumeric, dot, hyphen, and underscore
    if not re.match(r'^[a-zA-Z0-9._-]+$', username):
        return False = 'Username can only contain letters, numbers, dots, hyphens, and underscores'
    
    return True, ''

def validate_upload_folder(folder):
    """
    Validate that the upload folder path is safe.
    
    Args:
        folder (str): The upload folder path to validate.
    
    Returns:
        tuple: (is_valid, error_message, sanitized_path)
    """
    if not folder or not isinstance(folder, str):
        return False, 'Upload folder must be a non-empty string', None
    
    # Normalize the path to prevent directory traversal
    base_dir = os.path.abspath(current_app.config.get('BASE_UPLOAD_FOLDER', 'Users'))
    abs_path = os.path.abspath(os.path.join(base_dir, folder))
    
    # Ensure the path is within the base directory
    if not abs_path.startswith(base_dir):
        return False, 'Invalid upload folder path', None
    
    return True, '', abs_path

def validate_api_key(api_key):
    """
    Validate an API key.
    
    In a production environment, this would validate against a database or key store.
    For now, we'll just check if it's a non-empty string.
    
    Args:
        api_key (str): The API key to validate.
    
    Returns:
        bool: True if the API key is valid, False otherwise.
    """
    if not api_key or not isinstance(api_key, str):
        return False
    
    # In a real app, you would validate against a database or key store
    # For now, we'll just require a non-empty string
    return len(api_key.strip()) > 0

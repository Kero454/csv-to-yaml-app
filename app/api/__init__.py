"""
API package for the CSV to YAML Converter application.

This package contains all API-related modules and blueprints.
"""

from flask import Blueprint

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api/v1')

# Import API routes after creating the blueprint to avoid circular imports
from . import v1  # noqa: E402, F401

"""
API v1 package for the CSV to YAML Converter application.

This module initializes the v1 API endpoints.
"""

from flask import Blueprint, jsonify
from flask_restx import Api, Resource, fields, reqparse

# Create v1 blueprint
v1_bp = Blueprint('api_v1', __name__)

# Initialize Flask-RESTx API
authorizations = {
    'apikey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-Key'
    }
}

api = Api(
    v1_bp,
    version='1.0',
    title='CSV to YAML Converter API',
    description='A REST API for converting CSV files to YAML format',
    authorizations=authorizations,
    security='apikey',
    doc='/docs'  # Enable Swagger UI at /api/v1/docs/
)

# Import namespaces after creating api to avoid circular imports
from . import routes  # noqa: E402, F401

# Health check endpoint
@api.route('/health')
class HealthCheck(Resource):
    """Health check endpoint for API v1."""
    
    @api.doc('health_check')
    @api.response(200, 'API is healthy')
    def get(self):
        """Check if the API is running."""
        return {
            'status': 'healthy',
            'version': '1.0.0',
            'service': 'csv-to-yaml-converter',
            'timestamp': '2023-01-01T00:00:00Z'  # This would be dynamic in a real app
        }

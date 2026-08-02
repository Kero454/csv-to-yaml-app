"""
ML Model Catalog Service for Non-RT RIC
========================================
This service acts as the ML Model Catalog shown in the O-RAN ML lifecycle diagram.
It stores metadata about trained ML models that have been published to the Non-RT RIC.

Endpoints:
  POST   /models              - Publish (register) a new model
  GET    /models              - List all registered models
  GET    /models/<model_id>   - Get details of a specific model
  DELETE /models/<model_id>   - Remove a model from the catalog
  GET    /health              - Health check
"""

from flask import Flask, request, jsonify
import uuid
import datetime
import json
import os

app = Flask(__name__)

# In-memory model catalog (in production, this would be a database)
model_catalog = {}


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "ml-model-catalog", "version": "1.0.0"})


@app.route('/models', methods=['POST'])
def publish_model():
    """
    Publish a trained ML model to the Non-RT RIC Model Catalog.
    This is Step 5 in the O-RAN ML lifecycle diagram.
    
    Expected JSON body:
    {
        "name": "traffic-prediction-lstm",
        "version": "1.0.0",
        "description": "LSTM model for traffic prediction",
        "model_type": "lstm",
        "image": "localhost:30500/models/traffic-prediction:v1.0.0",
        "metrics": {
            "accuracy": 0.95,
            "loss": 0.05
        },
        "training_framework": "csv-to-yaml-platform",
        "input_schema": {"type": "timeseries", "features": ["traffic_load", "time_of_day"]},
        "output_schema": {"type": "prediction", "horizon": 10}
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    required_fields = ['name', 'version', 'image']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    model_id = str(uuid.uuid4())
    model_entry = {
        "id": model_id,
        "name": data['name'],
        "version": data['version'],
        "description": data.get('description', ''),
        "model_type": data.get('model_type', 'unknown'),
        "image": data['image'],
        "metrics": data.get('metrics', {}),
        "hyperparameters": data.get('hyperparameters', {}),
        "training_framework": data.get('training_framework', ''),
        "input_schema": data.get('input_schema', {}),
        "output_schema": data.get('output_schema', {}),
        "pkl_path": data.get('pkl_path', ''),
        "xapp_descriptor": data.get('xapp_descriptor', {}),
        "status": "published",
        "published_at": datetime.datetime.utcnow().isoformat() + "Z",
        "published_by": data.get('published_by', 'update-manager')
    }
    
    model_catalog[model_id] = model_entry
    
    return jsonify({
        "message": "Model published successfully to Non-RT RIC catalog",
        "model_id": model_id,
        "model": model_entry
    }), 201


@app.route('/models', methods=['GET'])
def list_models():
    """List all models in the catalog."""
    models = list(model_catalog.values())
    return jsonify({
        "count": len(models),
        "models": models
    })


@app.route('/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get details of a specific model."""
    if model_id not in model_catalog:
        return jsonify({"error": "Model not found"}), 404
    return jsonify(model_catalog[model_id])


@app.route('/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Remove a model from the catalog."""
    if model_id not in model_catalog:
        return jsonify({"error": "Model not found"}), 404
    deleted = model_catalog.pop(model_id)
    return jsonify({"message": "Model removed from catalog", "model": deleted})


@app.route('/models/<model_id>/deploy', methods=['POST'])
def request_deployment(model_id):
    """
    Request deployment of a model to the Near-RT RIC.
    This triggers Step 6-7 in the O-RAN ML lifecycle:
    - Package model, set policies
    - Download to Near-RT RIC
    """
    if model_id not in model_catalog:
        return jsonify({"error": "Model not found"}), 404
    
    model = model_catalog[model_id]
    model['status'] = 'deployment_requested'
    model['deployment_requested_at'] = datetime.datetime.utcnow().isoformat() + "Z"
    
    return jsonify({
        "message": "Deployment to Near-RT RIC requested",
        "model_id": model_id,
        "target": "near-rt-ric",
        "status": "deployment_requested"
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)

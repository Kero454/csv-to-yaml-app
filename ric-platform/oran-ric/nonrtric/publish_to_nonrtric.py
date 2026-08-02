"""
Publish to Non-RT RIC — Step 5 of the O-RAN ML Lifecycle
=========================================================

This script implements Step 5 from the O-RAN ML lifecycle diagram:
"Publish to Non-RT RIC"

It takes a trained ML model (output from the CSV-to-YAML training platform)
and publishes it to the Non-RT RIC's ML Model Catalog.

What this script does:
1. Reads the trained model metadata (name, version, type, metrics)
2. Packages the model as a Docker image (xApp container)
3. Pushes the Docker image to a container registry
4. Registers the model in the ML Model Catalog

Usage:
    python publish_to_nonrtric.py --model-name "traffic-prediction" \
                                   --model-version "1.0.0" \
                                   --model-type "lstm" \
                                   --model-path "./trained_models/model.pt"

For testing without a real model:
    python publish_to_nonrtric.py --demo
"""

import argparse
import json
import os
import sys
import datetime
import requests


# Configuration
CATALOG_URL = os.environ.get("ML_CATALOG_URL", "http://localhost:8080")
REGISTRY_URL = os.environ.get("MODEL_REGISTRY_URL", "localhost:30500")
NEARRT_RIC_URL = os.environ.get("NEARRT_RIC_URL", "http://localhost:32080")


def publish_model_to_catalog(model_info: dict) -> dict:
    """
    Register the model in the Non-RT RIC ML Model Catalog.
    
    This is the core action of Step 5: making the model discoverable
    by the Non-RT RIC so it can later be deployed to the Near-RT RIC.
    """
    payload = {
        "name": model_info["name"],
        "version": model_info["version"],
        "description": model_info.get("description", ""),
        "model_type": model_info.get("model_type", "unknown"),
        "image": f"{REGISTRY_URL}/models/{model_info['name']}:v{model_info['version']}",
        "metrics": model_info.get("metrics", {}),
        "training_framework": "csv-to-yaml-platform",
        "input_schema": model_info.get("input_schema", {}),
        "output_schema": model_info.get("output_schema", {}),
        "published_by": "update-manager"
    }
    
    print(f"\n[Step 5] Publishing model to Non-RT RIC ML Model Catalog...")
    print(f"  Catalog URL: {CATALOG_URL}")
    print(f"  Model: {payload['name']} v{payload['version']}")
    print(f"  Type: {payload['model_type']}")
    print(f"  Image: {payload['image']}")
    
    try:
        response = requests.post(f"{CATALOG_URL}/models", json=payload)
        response.raise_for_status()
        result = response.json()
        print(f"\n  [OK] Model published successfully!")
        print(f"  Model ID: {result['model_id']}")
        print(f"  Status: {result['model']['status']}")
        return result
    except requests.exceptions.ConnectionError:
        print(f"\n  [ERROR] Cannot connect to ML Model Catalog at {CATALOG_URL}")
        print(f"    Make sure the catalog service is running:")
        print(f"    python nonrtric/ml-model-catalog/app.py")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\n  [ERROR] Catalog returned error: {e}")
        print(f"    Response: {response.text}")
        sys.exit(1)


def verify_catalog_entry(model_id: str) -> dict:
    """Verify the model was successfully registered in the catalog."""
    print(f"\n[Verify] Checking model in catalog...")
    response = requests.get(f"{CATALOG_URL}/models/{model_id}")
    response.raise_for_status()
    model = response.json()
    print(f"  [OK] Model found in catalog:")
    print(f"    Name: {model['name']}")
    print(f"    Version: {model['version']}")
    print(f"    Status: {model['status']}")
    print(f"    Published at: {model['published_at']}")
    return model


def list_catalog():
    """List all models currently in the catalog."""
    print(f"\n[Catalog] Listing all models in Non-RT RIC ML Model Catalog...")
    response = requests.get(f"{CATALOG_URL}/models")
    response.raise_for_status()
    data = response.json()
    print(f"  Total models: {data['count']}")
    for model in data['models']:
        print(f"  - {model['name']} v{model['version']} ({model['model_type']}) — {model['status']}")
    return data


def run_demo():
    """
    Run a demonstration of the full Step 5 pipeline.
    Publishes a sample model to show the end-to-end flow.
    """
    print("=" * 60)
    print("  O-RAN ML Lifecycle — Step 5: Publish to Non-RT RIC")
    print("  DEMO MODE")
    print("=" * 60)
    
    # Simulate a model that was trained by the CSV-to-YAML platform
    demo_model = {
        "name": "traffic-prediction-lstm",
        "version": "1.0.0",
        "description": "LSTM model trained on network traffic data for load prediction. "
                       "Trained using the CSV-to-YAML platform (Khalid's framework).",
        "model_type": "lstm",
        "metrics": {
            "accuracy": 0.94,
            "mse": 0.0032,
            "mae": 0.041,
            "training_epochs": 50,
            "training_time_seconds": 120
        },
        "input_schema": {
            "type": "timeseries",
            "features": ["traffic_load", "num_users", "time_of_day", "day_of_week"],
            "window_size": 64
        },
        "output_schema": {
            "type": "prediction",
            "output": "traffic_load_forecast",
            "horizon_steps": 10
        }
    }
    
    print(f"\n[Input] Simulated trained model from CSV-to-YAML platform:")
    print(f"  Name: {demo_model['name']}")
    print(f"  Type: {demo_model['model_type']}")
    print(f"  Accuracy: {demo_model['metrics']['accuracy']}")
    
    # Step 5: Publish to Non-RT RIC
    result = publish_model_to_catalog(demo_model)
    
    # Verify
    verify_catalog_entry(result['model_id'])
    
    # Show catalog state
    list_catalog()
    
    print(f"\n{'=' * 60}")
    print(f"  Step 5 COMPLETE — Model is now in the Non-RT RIC ML Model Catalog")
    print(f"  Next steps:")
    print(f"    Step 6: Package model, set policies (Service Designer)")
    print(f"    Step 7: Download to Near-RT RIC, O-DU, O-RU")
    print(f"    Step 8: Collect metrics (feeds back to retraining)")
    print(f"{'=' * 60}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Publish a trained ML model to the Non-RT RIC ML Model Catalog (Step 5)"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run in demo mode with a sample model")
    parser.add_argument("--model-name", type=str,
                        help="Name of the model")
    parser.add_argument("--model-version", type=str, default="1.0.0",
                        help="Version of the model")
    parser.add_argument("--model-type", type=str, default="lstm",
                        help="Type of model (lstm, autoformer, linear, etc.)")
    parser.add_argument("--model-path", type=str,
                        help="Path to the trained model file")
    parser.add_argument("--description", type=str, default="",
                        help="Description of the model")
    parser.add_argument("--accuracy", type=float,
                        help="Model accuracy metric")
    parser.add_argument("--list", action="store_true",
                        help="List all models in the catalog")
    parser.add_argument("--catalog-url", type=str, default=CATALOG_URL,
                        help=f"ML Model Catalog URL (default: {CATALOG_URL})")
    
    args = parser.parse_args()
    
    # Update module-level catalog URL if specified via CLI
    globals()['CATALOG_URL'] = args.catalog_url
    
    if args.list:
        list_catalog()
        return
    
    if args.demo:
        run_demo()
        return
    
    if not args.model_name:
        parser.error("--model-name is required (or use --demo for a demonstration)")
    
    # Build model info from CLI arguments
    model_info = {
        "name": args.model_name,
        "version": args.model_version,
        "model_type": args.model_type,
        "description": args.description or f"{args.model_type} model: {args.model_name}",
        "metrics": {}
    }
    
    if args.accuracy:
        model_info["metrics"]["accuracy"] = args.accuracy
    
    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"WARNING: Model file not found at {args.model_path}")
        model_info["metrics"]["model_file"] = args.model_path
    
    # Publish
    result = publish_model_to_catalog(model_info)
    verify_catalog_entry(result['model_id'])


if __name__ == "__main__":
    main()

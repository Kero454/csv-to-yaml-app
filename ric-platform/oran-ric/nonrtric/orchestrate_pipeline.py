#!/usr/bin/env python3
"""
End-to-End Orchestration: Framework -> PKL -> Catalog -> xApp Image -> Deploy to Near-RT RIC

This script automates the full pipeline:
  1. Extract .pkl from the training framework (via API or local file)
  2. Register model metadata in ML Model Catalog
  3. Build xApp Docker image wrapping the .pkl
  4. Generate appmgr/submgr/A1 config files
  5. Deploy the xApp to ricxapp namespace via Helm

Usage:
  python3 orchestrate_pipeline.py --demo
  python3 orchestrate_pipeline.py --pkl /path/to/model.pkl --name my-model --type lstm
  python3 orchestrate_pipeline.py --framework-url http://localhost:5000/api/models/latest
"""

import argparse
import json
import os
import subprocess
import sys
import requests
import tempfile
import time

CATALOG_URL = os.environ.get("CATALOG_URL", "http://localhost:8080")
XAPP_BUILDER = os.path.join(os.path.dirname(__file__), "xapp-builder", "build_xapp.py")
RICXAPP_NS = "ricxapp"
KUBECTL = ["minikube", "-p", "kero-ric", "kubectl", "--"]


def banner(text):
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def step(num, total, text):
    print(f"\n  [{num}/{total}] {text}")


def fetch_pkl_from_framework(framework_url, output_dir):
    """Fetch the latest trained model .pkl from the framework API."""
    print(f"    Fetching from: {framework_url}")
    try:
        resp = requests.get(framework_url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        pkl_url = data.get("pkl_url") or data.get("model_url") or data.get("download_url")
        model_name = data.get("name", data.get("model_name", "framework-model"))
        model_type = data.get("type", data.get("model_type", "generic"))
        accuracy = data.get("accuracy", data.get("metrics", {}).get("accuracy", 0.0))

        if pkl_url:
            print(f"    Downloading .pkl from: {pkl_url}")
            pkl_resp = requests.get(pkl_url, timeout=60)
            pkl_resp.raise_for_status()
            pkl_path = os.path.join(output_dir, "model.pkl")
            with open(pkl_path, 'wb') as f:
                f.write(pkl_resp.content)
            print(f"    [OK] Downloaded: {pkl_path}")
        else:
            print("    [WARN] No pkl_url in response, using metadata only")
            pkl_path = None

        return pkl_path, model_name, model_type, accuracy

    except requests.exceptions.ConnectionError:
        print(f"    [WARN] Cannot connect to framework at {framework_url}")
        print(f"    [INFO] Falling back to demo mode")
        return None, None, None, None
    except Exception as e:
        print(f"    [WARN] Framework API error: {e}")
        return None, None, None, None


def register_in_catalog(model_name, model_version, model_type, image_tag, accuracy, pkl_path, descriptor):
    """Register the model in the ML Model Catalog."""
    payload = {
        "name": model_name,
        "version": model_version,
        "model_type": model_type,
        "image": image_tag,
        "description": f"{model_type.upper()} model built by xApp Builder",
        "metrics": {"accuracy": accuracy},
        "training_framework": "csv-to-yaml-platform",
        "pkl_path": pkl_path or "",
        "xapp_descriptor": descriptor
    }
    try:
        resp = requests.post(f"{CATALOG_URL}/models", json=payload, timeout=10)
        if resp.status_code == 201:
            data = resp.json()
            model_id = data.get("model_id", data.get("model", {}).get("id", "unknown"))
            print(f"    [OK] Registered in catalog: model_id={model_id}")
            return model_id
        else:
            print(f"    [WARN] Catalog returned {resp.status_code}: {resp.text}")
            return None
    except Exception as e:
        print(f"    [WARN] Cannot reach catalog: {e}")
        return None


def build_xapp(pkl_path, model_name, model_version, model_type, output_dir):
    """Run the xApp Builder to create Docker image + descriptor."""
    cmd = [
        sys.executable, XAPP_BUILDER,
        "--name", model_name,
        "--version", model_version,
        "--type", model_type,
        "--output-dir", output_dir,
    ]
    if pkl_path:
        cmd.extend(["--pkl", pkl_path])
    else:
        cmd.append("--demo")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"    [FAIL] xApp Builder error: {result.stderr}")
        return False
    return True


def generate_ric_configs(output_dir, model_name, model_version, image_tag):
    """Generate A1, appmgr, and submgr config files for the xApp."""
    config_dir = os.path.join(output_dir, "ric-configs")
    os.makedirs(config_dir, exist_ok=True)

    # A1 Policy Type definition
    a1_policy = {
        "name": f"ORAN_TrafficSteeringPreference_{model_name}",
        "description": f"A1 policy type for {model_name} xApp",
        "policy_type_id": 20008,
        "create_schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "scope": {
                    "type": "object",
                    "properties": {
                        "ueId": {"type": "string"},
                        "cellId": {"type": "string"}
                    }
                },
                "qosObjectives": {
                    "type": "object",
                    "properties": {
                        "priorityLevel": {"type": "integer", "minimum": 1, "maximum": 15},
                        "targetThroughput": {"type": "number"}
                    }
                },
                "resources": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "cellIdList": {"type": "array", "items": {"type": "string"}},
                            "preference": {"type": "string", "enum": ["SHALL", "PREFER", "AVOID", "FORBID"]}
                        }
                    }
                }
            }
        }
    }
    a1_path = os.path.join(config_dir, "a1-policy-type.json")
    with open(a1_path, 'w') as f:
        json.dump(a1_policy, f, indent=2)

    # appmgr xApp config (what appmgr uses to onboard/deploy)
    appmgr_config = {
        "xapp_name": model_name,
        "version": model_version,
        "release_name": model_name,
        "namespace": "ricxapp",
        "helmVersion": model_version,
        "overrides": {
            "image.repository": image_tag.rsplit(':', 1)[0],
            "image.tag": model_version,
            "replicaCount": 1
        }
    }
    appmgr_path = os.path.join(config_dir, "appmgr-config.json")
    with open(appmgr_path, 'w') as f:
        json.dump(appmgr_config, f, indent=2)

    # submgr subscription config (what subscriptions the xApp needs)
    submgr_config = {
        "xapp_name": model_name,
        "subscription": {
            "ActionType": "report",
            "SubsequentAction": {"SubsequentActionType": "continue", "TimeToWait": "w10ms"},
            "EventTriggerDefinition": {
                "reportingPeriod_ms": 1000,
                "eventTriggerStyle": 1
            },
            "ActionDefinitions": [
                {
                    "ActionID": 1,
                    "ActionType": "report",
                    "RICactionDefinition": {
                        "metrics": ["DRB.UEThpDl", "DRB.UEThpUl", "RRU.PrbUsedDl", "RRU.PrbUsedUl"]
                    }
                }
            ]
        }
    }
    submgr_path = os.path.join(config_dir, "submgr-config.json")
    with open(submgr_path, 'w') as f:
        json.dump(submgr_config, f, indent=2)

    print(f"    [OK] A1 policy type:    {a1_path}")
    print(f"    [OK] appmgr config:     {appmgr_path}")
    print(f"    [OK] submgr config:     {submgr_path}")

    return config_dir


def deploy_xapp(output_dir, model_name):
    """Deploy the xApp to ricxapp namespace via Helm."""
    chart_dir = os.path.join(output_dir, "helm", model_name)
    if not os.path.exists(chart_dir):
        print(f"    [FAIL] Helm chart not found at: {chart_dir}")
        return False

    # Check if already deployed
    result = subprocess.run(
        ["helm", "list", "-n", RICXAPP_NS, "-q"],
        capture_output=True, text=True
    )
    if model_name in result.stdout:
        print(f"    [INFO] {model_name} already deployed, upgrading...")
        cmd = ["helm", "upgrade", model_name, chart_dir, "--namespace", RICXAPP_NS]
    else:
        cmd = ["helm", "install", model_name, chart_dir, "--namespace", RICXAPP_NS]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [FAIL] Helm deploy failed: {result.stderr}")
        return False
    print(f"    [OK] xApp deployed to {RICXAPP_NS}")
    return True


def verify_deployment(model_name):
    """Verify the xApp pod is running."""
    for attempt in range(6):
        result = subprocess.run(
            KUBECTL + ["get", "pods", "-n", RICXAPP_NS, "-l", f"app={model_name}",
             "-o", "jsonpath={.items[0].status.phase}"],
            capture_output=True, text=True
        )
        # Also try without label selector
        if not result.stdout:
            result = subprocess.run(
                KUBECTL + ["get", "pods", "-n", RICXAPP_NS],
                capture_output=True, text=True
            )
            if model_name in result.stdout and "Running" in result.stdout:
                print(f"    [OK] xApp pod is Running")
                print(f"    {result.stdout.strip()}")
                return True

        if result.stdout.strip() == "Running":
            print(f"    [OK] xApp pod is Running")
            return True

        print(f"    Waiting for pod... (attempt {attempt + 1}/6)")
        time.sleep(5)

    print(f"    [WARN] Pod not Running after 30s. Check with: kubectl get pods -n {RICXAPP_NS}")
    return False


def main():
    parser = argparse.ArgumentParser(description="End-to-end xApp pipeline orchestrator")
    parser.add_argument("--pkl", help="Path to .pkl model file")
    parser.add_argument("--name", default="ml-xapp", help="Model/xApp name")
    parser.add_argument("--version", default="1.0.0", help="Model version")
    parser.add_argument("--type", default="generic", help="Model type (lstm, autoformer, etc.)")
    parser.add_argument("--demo", action="store_true", help="Run full pipeline with sample data")
    parser.add_argument("--framework-url", help="URL to fetch .pkl from training framework API")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--skip-deploy", action="store_true", help="Build only, don't deploy")
    args = parser.parse_args()

    banner("O-RAN ML xApp Pipeline - End-to-End Orchestration")

    total_steps = 6
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "xapp-build-output"
    )
    os.makedirs(output_dir, exist_ok=True)

    pkl_path = None
    model_name = args.name
    model_type = args.type
    model_version = args.version
    accuracy = 0.0

    # --- Step 1: Get the .pkl file ---
    step(1, total_steps, "Acquiring trained model (.pkl)...")

    if args.framework_url:
        pkl_path, fname, ftype, facc = fetch_pkl_from_framework(args.framework_url, output_dir)
        if fname:
            model_name = fname
        if ftype:
            model_type = ftype
        if facc:
            accuracy = facc

    if args.pkl:
        pkl_path = os.path.abspath(args.pkl)
        print(f"    Using local .pkl: {pkl_path}")

    if args.demo:
        model_name = "traffic-prediction-lstm"
        model_type = "lstm"
        accuracy = 0.94
        print("    Demo mode: will create sample .pkl")

    if not pkl_path and not args.demo:
        print("    [FAIL] No .pkl source. Use --pkl, --framework-url, or --demo")
        sys.exit(1)

    print(f"    Model: {model_name} v{model_version} ({model_type})")

    # --- Step 2: Build xApp image ---
    step(2, total_steps, "Building xApp Docker image from .pkl...")
    success = build_xapp(pkl_path, model_name, model_version, model_type, output_dir)
    if not success:
        print("    [FAIL] Build failed")
        sys.exit(1)

    # Load descriptor
    desc_path = os.path.join(output_dir, "config-file.json")
    with open(desc_path) as f:
        descriptor = json.load(f)

    registry = os.environ.get("XAPP_REGISTRY", "localhost:5000")
    image_tag = f"{registry}/xapps/{model_name}:{model_version}"

    # --- Step 3: Register in ML Model Catalog ---
    step(3, total_steps, "Registering model in ML Model Catalog...")
    model_id = register_in_catalog(
        model_name, model_version, model_type, image_tag, accuracy, pkl_path, descriptor
    )

    # --- Step 4: Generate RIC config files ---
    step(4, total_steps, "Generating A1 / appmgr / submgr config files...")
    config_dir = generate_ric_configs(output_dir, model_name, model_version, image_tag)

    # --- Step 5: Deploy to Near-RT RIC ---
    if not args.skip_deploy:
        step(5, total_steps, "Deploying xApp to Near-RT RIC (ricxapp)...")
        deploy_xapp(output_dir, model_name)

        # --- Step 6: Verify ---
        step(6, total_steps, "Verifying deployment...")
        verify_deployment(model_name)
    else:
        step(5, total_steps, "Skipping deployment (--skip-deploy)")
        step(6, total_steps, "Skipping verification")

    # --- Summary ---
    banner("PIPELINE COMPLETE")
    print(f"  Model:        {model_name} v{model_version} ({model_type})")
    print(f"  Accuracy:     {accuracy}")
    print(f"  Image:        {image_tag}")
    print(f"  Catalog ID:   {model_id or 'not registered'}")
    print(f"  Descriptor:   {desc_path}")
    print(f"  Helm chart:   {os.path.join(output_dir, 'helm', model_name)}")
    print(f"  RIC configs:  {config_dir}")
    print(f"    - a1-policy-type.json   (A1 mediator)")
    print(f"    - appmgr-config.json    (App Manager)")
    print(f"    - submgr-config.json    (Subscription Manager)")
    print()
    print("  Deployed to:  ricxapp namespace")
    print("=" * 60)


if __name__ == "__main__":
    main()

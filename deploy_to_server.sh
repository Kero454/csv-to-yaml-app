#!/bin/bash
# ============================================================
# Deploy CSV-to-YAML Framework to Remote Server
# ============================================================
# This script deploys the csv-to-yaml-app to the smartgridwks6 server
# where Docker and Kubernetes are available for the full O-RAN pipeline.
#
# Usage:
#   ./deploy_to_server.sh
#
# Prerequisites:
#   - SSH access to the server (kero@smartgridwks6 or kero@10.1.65.251)
#   - Python 3.10+ on the server
#   - The oran-ric repo already cloned on the server
# ============================================================

set -e

# ---- Configuration ----
SERVER_HOST="${RIC_SERVER_HOST:-10.1.65.251}"
SERVER_USER="${RIC_SERVER_USER:-kero}"
REMOTE_APP_DIR="/home/${SERVER_USER}/csv-to-yaml-app"
REMOTE_ORAN_DIR="/home/${SERVER_USER}/oran-ric"
LOCAL_APP_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  CSV-to-YAML Framework → Server Deployment"
echo "============================================================"
echo "  Server: ${SERVER_USER}@${SERVER_HOST}"
echo "  Remote path: ${REMOTE_APP_DIR}"
echo "  Local path: ${LOCAL_APP_DIR}"
echo "============================================================"
echo ""

# ---- Step 1: Sync app code to server ----
echo "[1/5] Syncing csv-to-yaml-app to server..."
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude 'env' \
    --exclude '.env' \
    --exclude 'node_modules' \
    --exclude 'Thesis/' \
    --exclude 'Helm/' \
    --exclude 'Users/' \
    --exclude 'temp_*' \
    --exclude '_e2e_*' \
    --exclude '*.xlsx' \
    --exclude 'TimeWorksheet*' \
    --exclude 'temp_rendered.html' \
    --exclude 'temp_page.html' \
    --exclude 'deploy_to_server.*' \
    --exclude 'build-docker.*' \
    --exclude 'deploy-helm.*' \
    --exclude 'test-docker.*' \
    --exclude 'docker-compose.yaml' \
    --exclude 'k8s-resources.yaml' \
    --exclude 'Dockerfile' \
    "${LOCAL_APP_DIR}/" "${SERVER_USER}@${SERVER_HOST}:${REMOTE_APP_DIR}/"

echo "  [OK] App code synced"
echo ""

# ---- Step 2: Sync oran-ric repo (xapp-builder + catalog) ----
echo "[2/5] Syncing oran-ric (xapp-builder + ml-model-catalog)..."
ORAN_LOCAL="$(dirname "${LOCAL_APP_DIR}")/oran-ric"
if [ -d "${ORAN_LOCAL}" ]; then
    rsync -avz --progress \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'venv' \
        "${ORAN_LOCAL}/" "${SERVER_USER}@${SERVER_HOST}:${REMOTE_ORAN_DIR}/"
    echo "  [OK] oran-ric synced"
else
    echo "  [SKIP] oran-ric not found locally at ${ORAN_LOCAL}"
    echo "         Make sure it exists on the server at ${REMOTE_ORAN_DIR}"
fi
echo ""

# ---- Step 3: Install dependencies on server ----
echo "[3/5] Installing Python dependencies on server..."
ssh "${SERVER_USER}@${SERVER_HOST}" << 'REMOTE_SETUP'
set -e
cd ~/csv-to-yaml-app

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  [OK] Virtual environment created"
fi

# Activate and install
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install paramiko flask-cors

echo "  [OK] Dependencies installed"
REMOTE_SETUP
echo ""

# ---- Step 4: Initialize database on server ----
echo "[4/5] Initializing database on server..."
ssh "${SERVER_USER}@${SERVER_HOST}" << 'REMOTE_DB'
set -e
cd ~/csv-to-yaml-app
source venv/bin/activate

# Initialize Flask-Migrate if needed
if [ ! -d "migrations" ]; then
    flask db init
fi
flask db migrate -m "initial" 2>/dev/null || true
flask db upgrade

echo "  [OK] Database ready"
REMOTE_DB
echo ""

# ---- Step 5: Create systemd service (optional) ----
echo "[5/5] Creating startup script on server..."
ssh "${SERVER_USER}@${SERVER_HOST}" << 'REMOTE_START'
set -e
cat > ~/csv-to-yaml-app/start_app.sh << 'EOF'
#!/bin/bash
# Start the CSV-to-YAML platform + ML Model Catalog
cd ~/csv-to-yaml-app
source venv/bin/activate

# Set environment
export FLASK_APP=run.py
export FLASK_ENV=production
export RIC_CATALOG_URL=http://localhost:8080
export RIC_SERVER_HOST=localhost
export RIC_SERVER_USER=kero

# Start ML Model Catalog in background
echo "Starting ML Model Catalog on port 8080..."
cd ~/oran-ric/nonrtric/ml-model-catalog
python3 app.py &
CATALOG_PID=$!
echo "  Catalog PID: $CATALOG_PID"

# Start CSV-to-YAML app
echo "Starting CSV-to-YAML platform on port 5000..."
cd ~/csv-to-yaml-app
source venv/bin/activate
python3 run.py &
APP_PID=$!
echo "  App PID: $APP_PID"

echo ""
echo "============================================================"
echo "  Both services running!"
echo "  CSV-to-YAML: http://$(hostname -I | awk '{print $1}'):5000"
echo "  ML Catalog:  http://$(hostname -I | awk '{print $1}'):8080"
echo "============================================================"
echo "  To stop: kill $CATALOG_PID $APP_PID"
echo ""

# Wait for either to exit
wait
EOF
chmod +x ~/csv-to-yaml-app/start_app.sh

cat > ~/csv-to-yaml-app/stop_app.sh << 'EOF'
#!/bin/bash
# Stop both services
pkill -f "python3 run.py" 2>/dev/null || true
pkill -f "python3 app.py" 2>/dev/null || true
echo "Services stopped."
EOF
chmod +x ~/csv-to-yaml-app/stop_app.sh

echo "  [OK] Startup scripts created:"
echo "       ~/csv-to-yaml-app/start_app.sh"
echo "       ~/csv-to-yaml-app/stop_app.sh"
REMOTE_START

echo ""
echo "============================================================"
echo "  DEPLOYMENT COMPLETE!"
echo "============================================================"
echo ""
echo "  To start on the server:"
echo "    ssh ${SERVER_USER}@${SERVER_HOST}"
echo "    cd ~/csv-to-yaml-app && ./start_app.sh"
echo ""
echo "  Then access the app at:"
echo "    http://${SERVER_HOST}:5000"
echo ""
echo "  The full O-RAN pipeline will run on the server with"
echo "  Docker + Kubernetes available for Steps 6 & 7."
echo "============================================================"

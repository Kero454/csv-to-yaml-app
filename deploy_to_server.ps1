# ============================================================
# Deploy CSV-to-YAML Framework to Remote Server (PowerShell)
# ============================================================
# Usage: .\deploy_to_server.ps1
# Prerequisites: SSH access to kero@10.1.65.251
# ============================================================

$SERVER_HOST = if ($env:RIC_SERVER_HOST) { $env:RIC_SERVER_HOST } else { "10.1.65.251" }
$SERVER_USER = if ($env:RIC_SERVER_USER) { $env:RIC_SERVER_USER } else { "kero" }
$REMOTE_APP_DIR = "/home/$SERVER_USER/csv-to-yaml-app"
$REMOTE_ORAN_DIR = "/home/$SERVER_USER/oran-ric"
$LOCAL_APP_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
$LOCAL_ORAN_DIR = Join-Path (Split-Path -Parent $LOCAL_APP_DIR) "oran-ric"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  CSV-to-YAML Framework -> Server Deployment" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Server: $SERVER_USER@$SERVER_HOST"
Write-Host "  Remote app: $REMOTE_APP_DIR"
Write-Host "  Remote oran: $REMOTE_ORAN_DIR"
Write-Host "============================================================"
Write-Host ""

# ---- Step 1: Sync csv-to-yaml-app ----
Write-Host "[1/4] Syncing csv-to-yaml-app to server..." -ForegroundColor Yellow
scp -r `
    "$LOCAL_APP_DIR\app" `
    "$LOCAL_APP_DIR\config.py" `
    "$LOCAL_APP_DIR\run.py" `
    "$LOCAL_APP_DIR\requirements.txt" `
    "${SERVER_USER}@${SERVER_HOST}:${REMOTE_APP_DIR}/"

# Also sync instance/uploads if they exist (user data)
if (Test-Path "$LOCAL_APP_DIR\instance") {
    scp -r "$LOCAL_APP_DIR\instance" "${SERVER_USER}@${SERVER_HOST}:${REMOTE_APP_DIR}/"
}
# Sync migrations
if (Test-Path "$LOCAL_APP_DIR\migrations") {
    scp -r "$LOCAL_APP_DIR\migrations" "${SERVER_USER}@${SERVER_HOST}:${REMOTE_APP_DIR}/"
}
Write-Host "  [OK] App code synced" -ForegroundColor Green

# ---- Step 2: Sync oran-ric ----
Write-Host "[2/4] Syncing oran-ric repo to server..." -ForegroundColor Yellow
if (Test-Path $LOCAL_ORAN_DIR) {
    scp -r `
        "$LOCAL_ORAN_DIR\nonrtric" `
        "${SERVER_USER}@${SERVER_HOST}:${REMOTE_ORAN_DIR}/"
    Write-Host "  [OK] oran-ric synced" -ForegroundColor Green
} else {
    Write-Host "  [SKIP] oran-ric not found at $LOCAL_ORAN_DIR" -ForegroundColor DarkYellow
}

# ---- Step 3: Setup on server ----
Write-Host "[3/4] Setting up on server (venv + deps)..." -ForegroundColor Yellow
ssh "${SERVER_USER}@${SERVER_HOST}" @"
set -e
mkdir -p ~/csv-to-yaml-app
cd ~/csv-to-yaml-app

# Create venv
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install paramiko flask-cors -q

# Init DB
export FLASK_APP=run.py
flask db upgrade 2>/dev/null || flask db init && flask db migrate -m init && flask db upgrade

echo "  [OK] Server setup complete"
"@
Write-Host "  [OK] Dependencies installed" -ForegroundColor Green

# ---- Step 4: Create start/stop scripts ----
Write-Host "[4/4] Creating startup scripts..." -ForegroundColor Yellow
ssh "${SERVER_USER}@${SERVER_HOST}" @"
cat > ~/csv-to-yaml-app/start_app.sh << 'STARTEOF'
#!/bin/bash
cd ~/csv-to-yaml-app
source venv/bin/activate
export FLASK_APP=run.py
export RIC_CATALOG_URL=http://localhost:8080

# Kill old instances
pkill -f 'python3.*app.py' 2>/dev/null || true
pkill -f 'python3.*run.py' 2>/dev/null || true
sleep 1

# Start catalog
cd ~/oran-ric/nonrtric/ml-model-catalog
python3 app.py > /tmp/catalog.log 2>&1 &
echo "Catalog started (PID: \$!)"

# Start csv-to-yaml app
cd ~/csv-to-yaml-app
source venv/bin/activate
python3 run.py > /tmp/csvtoyaml.log 2>&1 &
echo "App started (PID: \$!)"

IP=\$(hostname -I | awk '{print \$1}')
echo ""
echo "Services running:"
echo "  App:     http://\$IP:5000"
echo "  Catalog: http://\$IP:8080"
STARTEOF
chmod +x ~/csv-to-yaml-app/start_app.sh

cat > ~/csv-to-yaml-app/stop_app.sh << 'STOPEOF'
#!/bin/bash
pkill -f 'python3.*run.py' 2>/dev/null || true
pkill -f 'python3.*app.py' 2>/dev/null || true
echo "Services stopped."
STOPEOF
chmod +x ~/csv-to-yaml-app/stop_app.sh
echo "  [OK] Scripts created"
"@
Write-Host "  [OK] Startup scripts created" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  DEPLOYMENT COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  To start on the server:" -ForegroundColor White
Write-Host "    ssh $SERVER_USER@$SERVER_HOST" -ForegroundColor White
Write-Host "    cd ~/csv-to-yaml-app && ./start_app.sh" -ForegroundColor White
Write-Host ""
Write-Host "  Then access:" -ForegroundColor White
Write-Host "    http://${SERVER_HOST}:5000" -ForegroundColor White
Write-Host ""

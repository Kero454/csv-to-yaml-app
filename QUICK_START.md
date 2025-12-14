# Quick Start Guide: Docker & Helm Deployment

## Step 1: Ensure Docker is Running

**Docker Desktop is not currently running. Please:**
1. Start Docker Desktop application
2. Wait for Docker to fully start (icon shows "Docker Desktop is running")
3. Verify with: `docker --version`

## Step 2: Build the Docker Image

Once Docker is running, build the image:

```bash
# Use the provided batch file
build-docker.bat

# OR manually run:
docker build -t kerollosadel/csv-to-yaml-app:latest .
```

## Step 3: Test Locally (Optional)

```bash
# Quick test with Docker
docker run -p 5000:5000 kerollosadel/csv-to-yaml-app:latest

# OR use docker-compose for full local environment
docker-compose up
```

Visit: http://localhost:5000

## Step 4: Deploy to Kubernetes with Helm

### Prerequisites Check:
```bash
# Check Kubernetes cluster
kubectl cluster-info

# Check Helm
helm version
```

### Deploy:
```bash
# Windows
deploy-helm.bat

# OR manually:
helm install csv-to-yaml-app ./Helm --namespace default --create-namespace
```

## Step 5: Access the Application

### Get Service Information:
```bash
kubectl get svc csv-to-yaml-app-csv-to-yaml-app -n default
```

### Access Methods:

**Method 1: NodePort (if using minikube/local cluster)**
```bash
# Get minikube IP
minikube ip

# Access at:
http://<MINIKUBE_IP>:30888
```

**Method 2: Port Forwarding**
```bash
kubectl port-forward svc/csv-to-yaml-app-csv-to-yaml-app 5000:80 -n default

# Access at:
http://localhost:5000
```

## Files Created for You:

| File | Purpose |
|------|---------|
| `Dockerfile` | Containerizes the Flask application |
| `docker-compose.yaml` | Local development with Docker |
| `.dockerignore` | Optimizes Docker build |
| `build-docker.bat` | Automates Docker image building |
| `deploy-helm.bat` | Automates Helm deployment |
| `Helm/` | Complete Helm chart for Kubernetes deployment |
| `DOCKER_DEPLOYMENT.md` | Comprehensive deployment guide |

## Troubleshooting

### Docker Issues:
- **"Cannot connect to Docker daemon"**: Start Docker Desktop
- **"Image build fails"**: Check `requirements.txt` for all dependencies

### Kubernetes Issues:
- **"No cluster found"**: Set up minikube or connect to a cluster
- **"Helm not found"**: Install Helm 3.x

### Quick Commands:
```bash
# Check pod status
kubectl get pods -n default

# View logs
kubectl logs -l app.kubernetes.io/name=csv-to-yaml-app -n default

# Delete and redeploy
helm uninstall csv-to-yaml-app -n default
helm install csv-to-yaml-app ./Helm -n default
```

## Next Steps:
1. Start Docker Desktop
2. Build the image using `build-docker.bat`
3. Deploy to Kubernetes using `deploy-helm.bat`
4. Access your application!

For detailed instructions, see `DOCKER_DEPLOYMENT.md`

# Docker & Helm Deployment Guide for CSV-to-YAML App

## Prerequisites
- Docker installed and running
- Kubernetes cluster (minikube, microk8s, or cloud provider)
- Helm 3.x installed
- kubectl configured to access your cluster

## 🐳 Building the Docker Image

### 1. Build the Image Locally
```bash
# Windows
build-docker.bat

# Or manually:
docker build -t kerollosadel/csv-to-yaml-app:latest .
```

### 2. Test Locally with Docker
```bash
# Run the container
docker run -p 5000:5000 kerollosadel/csv-to-yaml-app:latest

# Or use docker-compose for easier local testing:
docker-compose up
```

Access the app at: http://localhost:5000

### 3. Push to Docker Registry (Optional)
```bash
# Login to Docker Hub
docker login

# Push the image
docker push kerollosadel/csv-to-yaml-app:latest
```

## ⚓ Deploying with Helm

### 1. Prepare the Kubernetes Cluster

#### Create SSH Secret (if needed)
```bash
# Create secret for SSH key
kubectl create secret generic ssh-key-secret \
  --from-file=id_rsa=$HOME/.ssh/id_rsa \
  --namespace default
```

#### Create NFS PVC (if using shared storage)
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nfs-ts-framework-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  # Adjust storageClassName as per your cluster
  storageClassName: nfs-client
EOF
```

### 2. Deploy with Helm

#### Using the deployment script:
```bash
# Windows
deploy-helm.bat

# Linux/Mac
helm install csv-to-yaml-app ./Helm --namespace default --create-namespace
```

#### Or deploy manually:
```bash
# Install
helm install csv-to-yaml-app ./Helm \
  --namespace default \
  --create-namespace \
  --wait

# Upgrade (if already installed)
helm upgrade csv-to-yaml-app ./Helm \
  --namespace default \
  --wait
```

### 3. Custom Values

Create a custom values file `my-values.yaml`:
```yaml
image:
  repository: kerollosadel/csv-to-yaml-app
  tag: latest
  pullPolicy: Always

service:
  type: NodePort
  nodePort: 30888

resources:
  requests:
    cpu: 200m
    memory: 256Mi
  limits:
    cpu: 1000m
    memory: 1Gi

env:
  uploadFolder: /uploads
```

Deploy with custom values:
```bash
helm install csv-to-yaml-app ./Helm -f my-values.yaml
```

## 📊 Monitoring the Deployment

### Check Deployment Status
```bash
# View Helm release
helm status csv-to-yaml-app -n default

# View pods
kubectl get pods -n default -l app.kubernetes.io/name=Helm

# View services
kubectl get svc -n default

# View logs
kubectl logs -l app.kubernetes.io/name=Helm -n default --tail=100 -f
```

### Access the Application

#### NodePort Service
```bash
# Get the NodePort
kubectl get svc -n default

# Access at:
http://<NODE_IP>:30888
```

#### Port Forwarding (for testing)
```bash
kubectl port-forward svc/csv-to-yaml-app-helm 5000:80 -n default

# Access at:
http://localhost:5000
```

## 🔧 Troubleshooting

### Check Pod Status
```bash
kubectl describe pod -l app.kubernetes.io/name=Helm -n default
```

### Check Events
```bash
kubectl get events -n default --sort-by='.lastTimestamp'
```

### Debug Container
```bash
# Execute into the container
kubectl exec -it <POD_NAME> -n default -- /bin/bash

# Check logs
kubectl logs <POD_NAME> -n default --previous
```

### Common Issues

1. **ImagePullBackOff**: Image not found
   - Ensure image is pushed to registry
   - Check image name and tag in values.yaml

2. **CrashLoopBackOff**: Container keeps crashing
   - Check logs for Python errors
   - Verify all dependencies in requirements.txt

3. **PVC Issues**: Storage not mounting
   - Verify PVC exists and is bound
   - Check storage class availability

4. **SSH Key Issues**: SSH operations failing
   - Ensure ssh-key-secret exists
   - Verify key permissions (should be 0600)

## 🗑️ Cleanup

### Uninstall the application
```bash
helm uninstall csv-to-yaml-app -n default
```

### Delete PVC (if needed)
```bash
kubectl delete pvc nfs-ts-framework-pvc -n default
```

### Delete secrets
```bash
kubectl delete secret ssh-key-secret -n default
```

## 📦 Docker Image Details

### Image Structure
- **Base**: Python 3.10-slim
- **Dependencies**: Flask, PyYAML, gunicorn, gevent
- **Exposed Port**: 5000
- **Working Directory**: /app
- **Upload Directory**: /uploads (mounted to PVC in K8s)

### Health Endpoints
- `/health` - Liveness probe
- `/ready` - Readiness probe

### Environment Variables
- `UPLOAD_FOLDER` - Directory for uploaded files (default: /uploads)
- `PYTHONUNBUFFERED` - Ensures real-time logging

## 🚀 CI/CD Pipeline (Optional)

### GitHub Actions Example
```yaml
name: Build and Deploy

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t kerollosadel/csv-to-yaml-app:${{ github.sha }} .
    
    - name: Push to Registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push kerollosadel/csv-to-yaml-app:${{ github.sha }}
        docker tag kerollosadel/csv-to-yaml-app:${{ github.sha }} kerollosadel/csv-to-yaml-app:latest
        docker push kerollosadel/csv-to-yaml-app:latest
    
    - name: Deploy with Helm
      run: |
        helm upgrade csv-to-yaml-app ./Helm \
          --set image.tag=${{ github.sha }} \
          --namespace production \
          --wait
```

## 📝 Notes

- The application uses gunicorn with gevent workers for production
- SSH key is mounted for remote operations (Kubernetes deployments)
- NFS volume is used for shared storage across pods
- Health checks ensure proper pod lifecycle management
- The Helm chart supports autoscaling (disabled by default)

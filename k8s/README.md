# Kubernetes Configuration for CSV to YAML Converter

This directory contains Kubernetes configuration files for deploying the CSV to YAML Converter application.

## Directory Structure

```
k8s/
├── configmap.yaml          # Application configuration
├── deployment.yaml         # Deployment configuration
├── hpa.yaml                # Horizontal Pod Autoscaler
├── ingress.yaml            # Ingress configuration
├── kustomization.yaml      # Kustomize configuration
├── namespace.yaml          # Namespace configuration
├── network-policy.yaml     # Network policies
├── pdb.yaml               # Pod Disruption Budget
├── pvc.yaml               # Persistent Volume Claim
├── README.md              # This file
├── secret.yaml            # Sensitive configuration
└── service-account.yaml   # Service account and RBAC
```

## Prerequisites

- Kubernetes cluster (v1.19+)
- `kubectl` command-line tool
- `kustomize` (recommended) or `kubectl` with server-side apply support

## Deployment

### Using Kustomize (Recommended)

1. Update the `kustomization.yaml` file with your image registry and tag.
2. Apply the configuration:

   ```bash
   kubectl apply -k .
   ```

### Using kubectl

1. Apply each configuration file individually:

   ```bash
   kubectl apply -f namespace.yaml
   kubectl apply -f service-account.yaml
   kubectl apply -f configmap.yaml
   kubectl apply -f secret.yaml
   kubectl apply -f pvc.yaml
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f ingress.yaml
   kubectl apply -f hpa.yaml
   kubectl apply -f network-policy.yaml
   kubectl apply -f pdb.yaml
   ```

## Configuration

### Environment Variables

Update the `configmap.yaml` and `secret.yaml` files with your application's configuration.

### Storage

The application uses a PersistentVolumeClaim for storing user uploads. Update the `pvc.yaml` file with your storage requirements.

### Ingress

Update the `ingress.yaml` file with your domain name and TLS configuration.

## Monitoring and Scaling

- **Horizontal Pod Autoscaler (HPA)**: Automatically scales the number of pods based on CPU and memory usage.
- **Resource Requests/Limits**: Configured in the deployment for both CPU and memory.

## Security

- **Network Policies**: Restrict network traffic to/from the application.
- **Pod Security Context**: Runs as a non-root user with read-only root filesystem.
- **RBAC**: Minimal permissions are granted to the service account.

## Maintenance

### Updating the Application

1. Build and push a new container image.
2. Update the image tag in the deployment or kustomization file.
3. Apply the changes:

   ```bash
   kubectl apply -k .
   ```

### Accessing Logs

```bash
# Get pod name
kubectl get pods -l app=csv-to-yaml

# View logs
kubectl logs -f <pod-name>
```

### Troubleshooting

```bash
# Describe pod
kubectl describe pod -l app=csv-to-yaml

# Check events
kubectl get events --sort-by='.metadata.creationTimestamp'

# Check ingress
kubectl get ingress
kubectl describe ingress csv-to-yaml-ingress
```

## Cleanup

To delete all resources:

```bash
kubectl delete -k .
# Or
kubectl delete -f .
```

## Production Considerations

1. **TLS**: Configure TLS in the ingress for secure communication.
2. **Backup**: Implement a backup solution for the persistent volume.
3. **Monitoring**: Set up monitoring and alerting for the application.
4. **CI/CD**: Set up a CI/CD pipeline for automated deployments.
5. **Secrets Management**: Use a dedicated secrets management solution in production.

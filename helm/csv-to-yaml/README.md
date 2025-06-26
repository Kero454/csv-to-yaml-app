# CSV to YAML Converter Helm Chart

This chart deploys the CSV to YAML Converter application on a Kubernetes cluster using the Helm package manager.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure (if persistence is needed)

## Installing the Chart

To install the chart with the release name `csv-to-yaml`:

```bash
helm install csv-to-yaml ./csv-to-yaml
```

The command deploys the CSV to YAML Converter on the Kubernetes cluster with the default configuration. The [Parameters](#parameters) section lists the parameters that can be configured during installation.

## Uninstalling the Chart

To uninstall/delete the `csv-to-yaml` deployment:

```bash
helm delete csv-to-yaml
```

The command removes all the Kubernetes components associated with the chart and deletes the release.

## Parameters

### Common parameters

| Name                | Description                                                                 | Value           |
| ------------------- | --------------------------------------------------------------------------- | --------------- |
| `replicaCount`      | Number of replicas to deploy                                                | `1`             |
| `image.repository`  | Image repository                                                            | `csv-to-yaml-app` |
| `image.tag`         | Image tag (defaults to app version)                                         | `""`            |
| `image.pullPolicy`  | Image pull policy                                                          | `IfNotPresent`  |
| `service.type`      | Kubernetes service type                                                     | `ClusterIP`     |
| `service.port`      | Service port                                                               | `80`            |
| `service.targetPort`| Target port of the container                                               | `5000`          |
| `ingress.enabled`   | Enable ingress                                                             | `false`         |
| `persistence.enabled` | Enable persistence using PVC                                              | `true`          |
| `persistence.size`  | Size of the PVC to request                                                 | `1Gi`           |


### Configuration parameters

| Name                              | Description                                                                 | Value          |
| --------------------------------- | --------------------------------------------------------------------------- | -------------- |
| `config.uploadFolder`            | Directory to store uploaded files                                           | `/app/uploads` |
| `config.allowedExtensions`       | Comma-separated list of allowed file extensions                            | `csv`          |
| `config.maxContentLength`        | Maximum file size in bytes (default: 16MB)                                 | `16777216`     |

## Persistence

The chart mounts a [Persistent Volume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) volume at `/app/uploads`. The volume is created using dynamic volume provisioning. If a PersistentVolumeClaim (PVC) already exists, specify it during installation.

## Ingress

This chart provides support for Ingress resource. To enable Ingress, set `ingress.enabled` to `true` and configure at least one host.

## Upgrading

To upgrade the chart with the release name `csv-to-yaml`:

```bash
helm upgrade csv-to-yaml ./csv-to-yaml
```

## Troubleshooting

If the application is not accessible, check the following:

1. Check the status of the pods:
   ```bash
   kubectl get pods -l app.kubernetes.io/name=csv-to-yaml
   ```

2. Check the logs of the pod:
   ```bash
   kubectl logs <pod-name>
   ```

3. Check if the service is running:
   ```bash
   kubectl get svc -l app.kubernetes.io/name=csv-to-yaml
   ```

4. If using ingress, check the ingress resource:
   ```bash
   kubectl get ingress -l app.kubernetes.io/name=csv-to-yaml
   ```

## License

This project is licensed under the MIT License.

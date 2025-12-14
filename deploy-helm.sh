#!/bin/bash

echo "================================================"
echo "Deploying CSV-to-YAML App with Helm"
echo "================================================"

# Set deployment variables
RELEASE_NAME="csv-to-yaml-app"
NAMESPACE="default"
CHART_PATH="./Helm"

echo ""
echo "Deployment Configuration:"
echo "  Release Name: ${RELEASE_NAME}"
echo "  Namespace: ${NAMESPACE}"
echo "  Chart Path: ${CHART_PATH}"
echo ""

# Check if release already exists
echo "Checking if release exists..."
if helm list -n ${NAMESPACE} | grep -q ${RELEASE_NAME}; then
    echo "Release ${RELEASE_NAME} already exists. Upgrading..."
    helm upgrade ${RELEASE_NAME} ${CHART_PATH} -n ${NAMESPACE} --wait
else
    echo "Installing new release ${RELEASE_NAME}..."
    helm install ${RELEASE_NAME} ${CHART_PATH} -n ${NAMESPACE} --create-namespace --wait
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Deployment successful!"
    echo "================================================"
    echo ""
    echo "To check the deployment status:"
    echo "  helm status ${RELEASE_NAME} -n ${NAMESPACE}"
    echo ""
    echo "To get the application URL:"
    echo "  kubectl get svc -n ${NAMESPACE}"
    echo ""
    echo "To view logs:"
    echo "  kubectl logs -l app.kubernetes.io/name=csv-to-yaml-app -n ${NAMESPACE}"
    echo ""
    echo "To uninstall:"
    echo "  helm uninstall ${RELEASE_NAME} -n ${NAMESPACE}"
    echo ""
else
    echo ""
    echo "================================================"
    echo "ERROR: Deployment failed!"
    echo "================================================"
    exit 1
fi

#!/bin/bash

echo "================================================"
echo "Building Docker Image for CSV-to-YAML App"
echo "================================================"

# Set the image name and tag
IMAGE_NAME="kerollosadel/csv-to-yaml-app"
TAG="latest"

echo ""
echo "Building Docker image: ${IMAGE_NAME}:${TAG}"
echo ""

# Build the Docker image
docker build -t ${IMAGE_NAME}:${TAG} .

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "Docker image built successfully!"
    echo "Image: ${IMAGE_NAME}:${TAG}"
    echo "================================================"
    echo ""
    echo "To push to Docker Hub, run:"
    echo "  docker push ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To run locally for testing:"
    echo "  docker run -p 5000:5000 ${IMAGE_NAME}:${TAG}"
    echo ""
else
    echo ""
    echo "================================================"
    echo "ERROR: Failed to build Docker image!"
    echo "================================================"
    exit 1
fi

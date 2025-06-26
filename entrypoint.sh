#!/bin/bash
set -e

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Start the application
exec gunicorn --bind 0.0.0.0:5000 --workers=4 --threads=2 --worker-class=gthread --worker-tmp-dir /dev/shm app:app

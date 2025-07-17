# Base Python image
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UPLOAD_FOLDER=/app/uploads

# Create app directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure uploads directory exists (will be mounted to PVC in k8s)
RUN mkdir -p ${UPLOAD_FOLDER}/Users

EXPOSE 5000

# Start the Flask app via gunicorn (factory pattern)
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:5000", "-k", "gevent", "--log-level", "info", "run:app"]

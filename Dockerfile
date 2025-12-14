# Base Python image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y openssh-client

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

# Copy user creation script
COPY create_user.py /app/

# Ensure uploads directory exists (will be mounted to PVC in k8s)
RUN mkdir -p ${UPLOAD_FOLDER}/Users

# Create instance directory for database
RUN mkdir -p /app/instance

EXPOSE 5000

# Initialize database and create default users if not exists, then start the Flask app
CMD ["sh", "-c", "python init_db.py && python create_user.py && gunicorn --workers=4 --bind=0.0.0.0:5000 -k gevent --log-level info run:app"]
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    UPLOAD_FOLDER=/app/Users \
    MAX_CONTENT_LENGTH=16777216

# Create a non-root user and set up the working directory
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /sbin/nologin appuser && \
    mkdir -p /app/Users && \
    chown -R appuser:appuser /app

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY --chown=appuser:appuser . .

# Set proper permissions
RUN chmod 755 /app && \
    chmod -R 755 /app/static && \
    chmod -R 755 /app/templates && \
    chmod 644 /app/*.py && \
    chmod 644 /app/requirements.txt

# Switch to non-root user
USER appuser

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script as the default command
ENTRYPOINT ["/app/entrypoint.sh"]

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--worker-class", "gthread", "--threads", "2", "app:app"]

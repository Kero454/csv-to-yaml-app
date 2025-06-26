# CSV to YAML Converter

A web application for converting CSV files to YAML format with multi-tenant support.

## Features

- Upload and convert CSV files to YAML format
- Multi-tenant architecture
- RESTful API for programmatic access
- Containerized with Docker
- Kubernetes deployment ready

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git
- (Optional) Docker and Kubernetes for container deployment

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/csv-to-yaml-app.git
   cd csv-to-yaml-app
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the root directory with the following variables:

```env
FLASK_APP=wsgi.py
FLASK_ENV=development
SECRET_KEY=your-secret-key
UPLOAD_FOLDER=./Users
MAX_CONTENT_LENGTH=16777216  # 16MB
```

## Running the Application

### Development Server

```bash
flask run
```

### Production with Gunicorn

```bash
gunicorn -w 4 -b :5000 wsgi:app
```

### With Docker

```bash
docker build -t csv-to-yaml-app .
docker run -p 5000:5000 csv-to-yaml-app
```

## API Documentation

Once the application is running, you can access the API documentation at:
- Swagger UI: `http://localhost:5000/api/v1/docs`
- OpenAPI Spec: `http://localhost:5000/api/v1/swagger.json`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

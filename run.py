# run.py
import logging
from app import create_app

app = create_app()

# Configure logging to ensure INFO-level messages are captured
if __name__ != '__main__':
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(logging.INFO)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

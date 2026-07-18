FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir setuptools==69.5.1
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --force-reinstall opencv-python-headless --no-deps

COPY server.py .

CMD ["sh", "-c", "pip install gunicorn && gunicorn server:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8080} --limit-request-line 0 --limit-request-field_size 0"]
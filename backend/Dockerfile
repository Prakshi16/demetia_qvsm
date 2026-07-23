# Backend image for the Cortex Health Portal API.
# Python pinned to 3.10 to match the environment the Phase 1 pickle was built in.
FROM python:3.10-slim

# Faster, quieter, unbuffered Python; no .pyc clutter in the bind mount.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: build-essential + libpq for psycopg2/scientific wheels that may
# need compilation on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code. In dev this is shadowed by the docker-compose bind mount; in a real
# build (Render) it's the actual source.
COPY . .

EXPOSE 8000

# `app.main:app` resolves because WORKDIR is /app and the package is /app/app.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

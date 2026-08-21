# Backend image for the Cortex Health Portal API.
# Python pinned to 3.10 to match the environment the Phase 1 pickle was built in.
#
# BUILD CONTEXT IS THE REPO ROOT, not ./backend — the image has to contain the
# Phase 1 pickles in ./results, and a build context cannot reach above itself.
# Locally docker-compose sets this for you; on Render it is `dockerContext: .`
# in render.yaml. Building with `docker build backend/` will fail on the COPY.
FROM python:3.10-slim

# Faster, quieter, unbuffered Python; no .pyc clutter in the bind mount.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps: build-essential + libpq for psycopg2/scientific wheels that may
# need compilation on slim. ffmpeg is for decoding browser recordings — the
# bundled libsndfile reads wav/mp3 but not the webm/opus and m4a that
# MediaRecorder hands us from the browser's record button.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential libpq-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so the layer caches across code changes.
COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# The Phase 1 pickles, at the default MODEL_DIR. ~40 KB, so baking them in costs
# nothing and means a deployed instance has no external model dependency —
# without this /health reports model:"error" in production, because the
# docker-compose bind mount that supplies them in dev does not exist on Render.
COPY results/qsvm_model.pkl results/svm_model.pkl /model/

# App code. In dev this is shadowed by the docker-compose bind mount; in a real
# build (Render) it's the actual source.
COPY backend/ .

EXPOSE 8000

# Render injects $PORT and it is not always 8000, so bind to it when present.
# Shell form (not exec form) is required for the variable to expand.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}

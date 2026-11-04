"""FastAPI application entry point for the Cortex Health Portal backend.

Phase 0 scaffold: app object, CORS, an /api/v1 router, and a health check that
proves the Supabase connection works. Auth/patient/visit routers are added in
later passes and mounted onto `api_router` here.
"""
import logging
import math
import threading
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.routers import (
    auth,
    dashboard,
    hospital,
    models_info,
    mri_upload,
    patients,
    speech,
    staff,
    visits,
)
from app.services.mri_features import calibration_info
from app.services.prediction import USE_REAL_MODEL, qsvm_available, warm_models

# Without this the app's own loggers propagate to an unconfigured root logger and
# are dropped — uvicorn only configures its own. The per-prediction log line in
# services/prediction.py is the main thing we want visible during the demo.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)

# Set on startup so /health can report whether the Phase 1 pickles actually
# loaded. Prediction takes ~445 ms warm; preloading keeps that cost off the
# first upload request.
_models_ready = False


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the Phase 1 pickles once, before serving traffic.

    Never fatal: warm_models() swallows its own failures so a missing or
    unloadable pickle degrades predictions rather than stopping the API.
    """
    global _models_ready
    _models_ready = warm_models()
    # Speech extraction pays a one-off librosa/numba JIT cost on first use, which
    # is slow on a small host. Pay it in the background now so the first real
    # upload doesn't, and so startup (and the platform's health check) never waits.
    threading.Thread(target=_warm_speech_extractor, daemon=True).start()
    yield


def _warm_speech_extractor() -> None:
    try:
        import io
        import wave

        import numpy as np

        from app.services.speech_features import extract_speech_features

        rate = 16000
        t = np.linspace(0, 1.0, rate, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 140 * t) * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(tone.tobytes())
        buffer.seek(0)
        extract_speech_features(buffer)
        logging.getLogger(__name__).info("Speech extractor warmed up")
    except Exception:  # noqa: BLE001 - warm-up is best effort
        logging.getLogger(__name__).exception("Speech extractor warm-up failed (non-fatal)")


app = FastAPI(title="Cortex Health Portal API", version="0.1.0", lifespan=lifespan)


def _json_safe(value):
    """Replace NaN/Infinity (valid in Python's JSON parser, invalid in JSON)."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """FastAPI's default 422 echoes the offending input back. When that input is a
    bare NaN or Infinity literal, encoding the response raises and the client gets
    a 500 instead of the 422. This handler sanitises the echo."""
    errors = jsonable_encoder(exc.errors(), custom_encoder={float: lambda f: f if math.isfinite(f) else str(f)})
    return JSONResponse(status_code=422, content={"detail": _json_safe(errors)})

# CORS: driven by CORS_ORIGINS (comma-separated), defaulting to the Vite dev
# server. This used to include "*" for the demo, which cannot survive deploy:
# every request carries an Authorization header, and a credentialed request
# answered with a wildcard origin is rejected by the browser, not the server.
# Add the deployed frontend origin as a Render env var rather than widening this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Every endpoint in the API contract (§5) is prefixed /api/v1.
api_router = APIRouter(prefix="/api/v1")


@api_router.get("/health")
def health(db: Session = Depends(get_db)) -> dict:
    """Liveness + DB connectivity. Returns db:'ok' only if a SELECT 1 succeeds."""
    try:
        db.execute(text("SELECT 1"))
        db_state = "ok"
    except Exception:  # noqa: BLE001 - report any DB failure as a degraded state
        db_state = "error"
    if not USE_REAL_MODEL:
        model_state = "stub"
    else:
        model_state = "ok" if _models_ready else "error"
    return {
        "status": "ok",
        "db": db_state,
        "model": model_state,
        # Research view + MRI extractor: informational, never fail the check.
        "qsvm": "ok" if (USE_REAL_MODEL and qsvm_available()) else "unavailable",
        "mri_calibration": "ok" if calibration_info() else "uncalibrated",
    }


# Feature routers (§5). Upload endpoints (Bishal/Sheetal) mount onto visits later.
api_router.include_router(auth.router)
api_router.include_router(staff.router)
api_router.include_router(hospital.router)
api_router.include_router(patients.router)
api_router.include_router(visits.router)
api_router.include_router(speech.router)
api_router.include_router(mri_upload.router)
api_router.include_router(dashboard.router)
api_router.include_router(models_info.router)

app.include_router(api_router)

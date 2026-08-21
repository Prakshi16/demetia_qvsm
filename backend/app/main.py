"""FastAPI application entry point for the Cortex Health Portal backend.

Phase 0 scaffold: app object, CORS, an /api/v1 router, and a health check that
proves the Supabase connection works. Auth/patient/visit routers are added in
later passes and mounted onto `api_router` here.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.routers import auth, dashboard, patients, visits, speech, mri_upload
from app.services.prediction import USE_REAL_MODEL, warm_models

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
    yield


app = FastAPI(title="Cortex Health Portal API", version="0.1.0", lifespan=lifespan)

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
    return {"status": "ok", "db": db_state, "model": model_state}


# Feature routers (§5). Upload endpoints (Bishal/Sheetal) mount onto visits later.
api_router.include_router(auth.router)
api_router.include_router(patients.router)
api_router.include_router(visits.router)
api_router.include_router(speech.router)
api_router.include_router(mri_upload.router)
api_router.include_router(dashboard.router)

app.include_router(api_router)

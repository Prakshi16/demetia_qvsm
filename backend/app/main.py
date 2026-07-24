"""FastAPI application entry point for the Cortex Health Portal backend.

Phase 0 scaffold: app object, CORS, an /api/v1 router, and a health check that
proves the Supabase connection works. Auth/patient/visit routers are added in
later passes and mounted onto `api_router` here.
"""
from fastapi import APIRouter, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db import get_db
from app.routers import auth, dashboard, patients, visits

app = FastAPI(title="Cortex Health Portal API", version="0.1.0")

# CORS: the Vite dev server runs on :5173; "*" is kept for the demo so teammates
# can point their frontends at this backend without per-origin config. Tighten
# before any non-demo deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
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
    return {"status": "ok", "db": db_state}


# Feature routers (§5). Upload endpoints (Bishal/Sheetal) mount onto visits later.
api_router.include_router(auth.router)
api_router.include_router(patients.router)
api_router.include_router(visits.router)
api_router.include_router(dashboard.router)

app.include_router(api_router)

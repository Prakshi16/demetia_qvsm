"""GET /models/info — the cross-validated comparison behind the model toggle.

Read-only and the same for every hospital, but still behind sign-in like every
other non-auth endpoint. Served from ``results/model_info.json`` (written by
``backend/scripts/build_model_info.py``), loaded once.
"""
from __future__ import annotations

import json
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from app.deps import CurrentUser, get_current_user
from app.schemas import ModelInfoOut
from app.services.prediction import _model_dir, qsvm_available

router = APIRouter(prefix="/models", tags=["models"])

MODEL_INFO_FILE = "model_info.json"


@lru_cache(maxsize=1)
def _model_info() -> dict | None:
    path = _model_dir() / MODEL_INFO_FILE
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/info", response_model=ModelInfoOut)
def get_model_info(_: CurrentUser = Depends(get_current_user)) -> dict:
    info = _model_info()
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model comparison data is not available on this server.",
        )
    models = [m for m in info["models"] if m["key"] != "qsvm" or qsvm_available()]
    return {"evaluation": info["evaluation"], "models": models}

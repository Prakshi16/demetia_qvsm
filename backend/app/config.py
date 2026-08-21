"""Application settings, loaded from environment (backend/.env in local dev).

Every required variable is listed in backend/.env.example. `pydantic-settings`
reads them from the process environment; docker-compose injects them via
`env_file: backend/.env`, and Render injects them as dashboard env vars.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/.env sits one level above this package (backend/app/config.py -> backend/.env).
# Resolving it absolutely means it loads regardless of the current working directory.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # Postgres (Supabase). Required — the app cannot start without a database.
    DATABASE_URL: str

    # Supabase project + storage (used by the upload endpoints in a later pass).
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_BUCKET: str = "patient-uploads"

    # JWT signing. 24h expiry, no refresh token (Phase 2 §2 simplification note).
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_HOURS: int = 24

    # Phase 1 model pickles (qsvm_model.pkl + svm_model.pkl). docker-compose
    # mounts the repo's ./results here read-only; when running uvicorn directly
    # this path won't exist and the loader falls back to <repo>/results.
    MODEL_DIR: str = "/model"

    # Browser origins allowed to call this API, comma-separated. The Vite dev
    # server is the default; a deployed frontend must add its own origin here
    # (Render dashboard env var), because the app sends an Authorization header
    # and a credentialed request cannot be answered with a wildcard origin.
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # `.env` lives next to the backend package. In Docker the file is provided via
    # env_file so this path simply won't exist there — that's fine.
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

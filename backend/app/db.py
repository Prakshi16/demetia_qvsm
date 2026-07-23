"""Database engine, session factory, and the FastAPI `get_db` dependency.

SQLAlchemy 2.0 style. `pool_pre_ping` guards against Supabase closing idle
connections (common on the free tier) by checking liveness before each checkout.
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,  # recycle connections every 30 min; Supabase drops idle ones
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """Yield a request-scoped session, always closed afterward."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

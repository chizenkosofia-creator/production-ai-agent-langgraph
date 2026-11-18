"""Async SQLAlchemy engine / session factory.

A process-wide ``async_sessionmaker`` is created in the app lifespan. This
engine is separate from both the LangGraph checkpointer pool and the pgvector
engine, so the three never contend for connections or share transactions.
"""

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from core.config import Settings

logger = logging.getLogger(__name__)

_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None


def init_db(settings: Settings) -> async_sessionmaker[AsyncSession]:
    """Create the engine and session factory (call once at startup)."""
    global _engine, _sessionmaker
    _engine = create_async_engine(
        settings.sqlalchemy_url,
        pool_size=settings.db_pool_max_size,
        max_overflow=0,
        pool_pre_ping=True,  # survive DB restarts / idle disconnects
    )
    _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    logger.info("SQLAlchemy engine ready")
    return _sessionmaker


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the session factory or fail loudly."""
    if _sessionmaker is None:
        raise RuntimeError("Database not initialised; call init_db()")
    return _sessionmaker


async def close_db() -> None:
    """Dispose the engine at shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None

"""
Database engine and session management with async SQLAlchemy.
"""

import ssl
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings

settings = get_settings()


def create_engine_with_ssl(database_url: str):
    """Create engine with proper SSL support for Tiger Cloud."""
    # Tiger Cloud requires SSL - asyncpg uses ssl parameter, not sslmode
    if "sslmode=require" in database_url:
        clean_url = database_url.replace("?sslmode=require", "").replace("&sslmode=require", "")
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE
        return create_async_engine(
            clean_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            connect_args={"ssl": ssl_ctx},
        )
    return create_async_engine(database_url, echo=False, pool_size=10, max_overflow=20)


engine = create_engine_with_ssl(settings.database_url)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Create all tables and enable pgvector extension."""
    async with engine.begin() as conn:
        # Enable pgvector extension (needed for vector columns)
        await conn.execute(__import__('sqlalchemy').text(
            "CREATE EXTENSION IF NOT EXISTS vector"
        ))
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    await engine.dispose()

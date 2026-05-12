import asyncpg
import logging
from config import (
    PG_HOST, PG_PORT, PG_DATABASE,
    PG_USER, PG_PASSWORD, PG_MIN_POOL, PG_MAX_POOL
)

logger  = logging.getLogger(__name__)
_pool: asyncpg.Pool | None = None


async def startup() -> None:
    global _pool
    try:
        _pool = await asyncpg.create_pool(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD,
            min_size=PG_MIN_POOL,
            max_size=PG_MAX_POOL,
            command_timeout=10,       # query-level hard ceiling
            ssl="require",            # always encrypted to cloud PostgreSQL
        )
        logger.info("PostgreSQL pool ready (%d–%d connections)", PG_MIN_POOL, PG_MAX_POOL)
    except Exception as e:
        logger.error(f"Failed to start PostgreSQL pool: {e}")
        # We don't raise here to allow the app to start even if DB is down
        # but queries will fail with RuntimeError from get_pool()


async def shutdown() -> None:
    global _pool
    if _pool:
        await _pool.close()
        logger.info("PostgreSQL pool closed")


def get_pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised — call startup() first")
    return _pool

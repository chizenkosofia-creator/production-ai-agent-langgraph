"""Per-thread mutual exclusion using Postgres advisory locks.

Two simultaneous requests on the same ``thread_id`` would both read the same
checkpoint and then overwrite each other's writes. We serialise turns per
thread with a *session-level* advisory lock on a dedicated connection pool.

* ``pg_try_advisory_lock`` never blocks: if the thread is busy the caller gets
  ``acquired=False`` immediately and can answer 409 / emit an SSE error.
* The lock lives on its own small pool so lock-holding connections can never
  starve the checkpointer's pool (which would deadlock in-flight turns).
* If the API process dies, Postgres releases the lock when the session ends.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from psycopg_pool import AsyncConnectionPool


@asynccontextmanager
async def thread_lock(pool: AsyncConnectionPool, thread_id: str) -> AsyncIterator[bool]:
    """Try to take the lock for ``thread_id``; yields whether it was acquired."""
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (thread_id,)
        )
        row = await cur.fetchone()
        acquired = bool(row and row[0])
        try:
            yield acquired
        finally:
            if acquired:
                await conn.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (thread_id,)
                )

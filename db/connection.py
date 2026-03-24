"""
Thread-safe DuckDB connection manager.

DuckDB uses MVCC (multi-version concurrency control) natively — no explicit WAL pragma needed.
One persistent connection per process is the recommended pattern.
"""
from __future__ import annotations

import atexit
import logging
import threading
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_conn: Optional[duckdb.DuckDBPyConnection] = None
_db_path: Optional[Path] = None


def get_connection(db_path: Optional[Path] = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Return the singleton DuckDB connection. First call must supply db_path.
    Thread-safe. Subsequent calls reuse the existing connection.

    If the DB is locked by another process, falls back to read_only mode
    automatically to allow concurrent access (dashboard + pipeline).
    """
    global _conn, _db_path
    with _lock:
        if _conn is None:
            if db_path is None:
                raise RuntimeError(
                    "db_path must be provided on the first call to get_connection()"
                )
            _db_path = Path(db_path)
            _db_path.parent.mkdir(parents=True, exist_ok=True)

            # Try read-write first; if locked, fallback to read-only; last resort: in-memory
            try:
                _conn = duckdb.connect(str(_db_path), read_only=read_only)
                _conn.execute("PRAGMA threads=4")
                _conn.execute("PRAGMA memory_limit='512MB'")
                mode = "read-only" if read_only else "read-write"
                logger.info("DuckDB connection opened (%s): %s", mode, _db_path)
            except Exception as e:
                if "being used by another process" in str(e) or "locked" in str(e).lower():
                    # Fallback 1: read-only access (preserves data visibility)
                    try:
                        _conn = duckdb.connect(str(_db_path), read_only=True)
                        _conn.execute("PRAGMA threads=4")
                        logger.warning(
                            "DuckDB locked — opened read-only (data visible, writes disabled): %s",
                            _db_path,
                        )
                    except Exception:
                        # Fallback 2: in-memory (last resort)
                        logger.warning("DuckDB read-only also failed — using in-memory fallback")
                        _conn = duckdb.connect(":memory:")
                        _conn.execute("PRAGMA threads=4")
                else:
                    raise
        return _conn


def close_connection() -> None:
    """Close the singleton connection (call at application shutdown)."""
    global _conn
    with _lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            finally:
                _conn = None
            logger.info("DuckDB connection closed")


def _shutdown_hook() -> None:
    """
    Python 3.13 changed GC ordering at interpreter shutdown, causing DuckDB's
    C-extension destructor to segfault if the connection object is still alive.
    Explicitly close + delete the singleton here — registered via atexit so it
    runs before the interpreter tears down extension modules.
    """
    global _conn
    # Acquire without blocking — if the lock is held we skip rather than deadlock
    acquired = _lock.acquire(blocking=False)
    try:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
    finally:
        if acquired:
            _lock.release()


atexit.register(_shutdown_hook)

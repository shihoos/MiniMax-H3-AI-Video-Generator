from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable


class ProductionJobQueue:
    """Small persistent SQLite queue; designed to keep UI lifetime separate from job state."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, production_id TEXT NOT NULL, plan_path TEXT NOT NULL,
                status TEXT NOT NULL, payload_json TEXT NOT NULL, result_json TEXT, error TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL)""")

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def submit(self, production_id: str, plan_path: Path, payload: dict[str, Any] | None = None) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("INSERT INTO jobs VALUES (?, ?, ?, 'queued', ?, NULL, NULL, ?, ?)", (job_id, str(production_id), str(Path(plan_path).resolve()), json.dumps(payload or {}, ensure_ascii=False), now, now))
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def claim_next(self) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            conn.execute("UPDATE jobs SET status='running', updated_at=? WHERE job_id=? AND status='queued'", (now, row['job_id']))
            return dict(conn.execute("SELECT * FROM jobs WHERE job_id=?", (row['job_id'],)).fetchone())

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status='completed', result_json=?, updated_at=? WHERE job_id=?", (json.dumps(result, ensure_ascii=False), time.time(), job_id))

    def fail(self, job_id: str, error: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status='failed', error=?, updated_at=? WHERE job_id=?", (str(error), time.time(), job_id))

    def recover_stale(self, max_age_seconds: float = 3600.0) -> int:
        cutoff = time.time() - float(max_age_seconds)
        with self._connect() as conn:
            cur = conn.execute("UPDATE jobs SET status='queued', updated_at=? WHERE status='running' AND updated_at < ?", (time.time(), cutoff))
            return int(cur.rowcount)

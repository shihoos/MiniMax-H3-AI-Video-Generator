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
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                worker_token TEXT, lease_expires_at REAL, heartbeat_at REAL)""")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            for name, ddl in ((
                "worker_token", "ALTER TABLE jobs ADD COLUMN worker_token TEXT"),
                ("lease_expires_at", "ALTER TABLE jobs ADD COLUMN lease_expires_at REAL"),
                ("heartbeat_at", "ALTER TABLE jobs ADD COLUMN heartbeat_at REAL"),
            ):
                if name not in columns:
                    conn.execute(ddl)

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
            conn.execute("INSERT INTO jobs (job_id, production_id, plan_path, status, payload_json, result_json, error, created_at, updated_at, worker_token, lease_expires_at, heartbeat_at) VALUES (?, ?, ?, 'queued', ?, NULL, NULL, ?, ?, NULL, NULL, NULL)", (job_id, str(production_id), str(Path(plan_path).resolve()), json.dumps(payload or {}, ensure_ascii=False), now, now))
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def claim_next(self, lease_seconds: float = 21600.0) -> dict[str, Any] | None:
        worker_token = str(uuid.uuid4())
        now = time.time()
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
                ).fetchone()
                if row is None:
                    conn.execute("COMMIT")
                    return None
                cur = conn.execute(
                    "UPDATE jobs SET status='running', updated_at=?, heartbeat_at=?, lease_expires_at=?, worker_token=? WHERE job_id=? AND status='queued'",
                    (now, now, now + float(lease_seconds), worker_token, row['job_id']),
                )
                if cur.rowcount != 1:
                    conn.execute("ROLLBACK")
                    return None
                claimed = conn.execute("SELECT * FROM jobs WHERE job_id=?", (row['job_id'],)).fetchone()
                conn.execute("COMMIT")
                return dict(claimed) if claimed else None
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def heartbeat(self, job_id: str, worker_token: str, lease_seconds: float = 21600.0) -> bool:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE jobs SET updated_at=?, heartbeat_at=?, lease_expires_at=? WHERE job_id=? AND status='running' AND worker_token=?",
                (now, now, now + float(lease_seconds), job_id, worker_token),
            )
            return cur.rowcount == 1

    def complete(self, job_id: str, result: dict[str, Any], worker_token: str | None = None) -> None:
        with self._connect() as conn:
            if worker_token:
                cur = conn.execute("UPDATE jobs SET status='completed', result_json=?, updated_at=?, lease_expires_at=NULL WHERE job_id=? AND status='running' AND worker_token=?", (json.dumps(result, ensure_ascii=False), time.time(), job_id, worker_token))
            else:
                cur = conn.execute("UPDATE jobs SET status='completed', result_json=?, updated_at=?, lease_expires_at=NULL WHERE job_id=?", (json.dumps(result, ensure_ascii=False), time.time(), job_id))
            if cur.rowcount != 1:
                raise RuntimeError(f"Could not complete job {job_id}; lease ownership was lost.")

    def fail(self, job_id: str, error: str, worker_token: str | None = None) -> None:
        with self._connect() as conn:
            if worker_token:
                cur = conn.execute("UPDATE jobs SET status='failed', error=?, updated_at=?, lease_expires_at=NULL WHERE job_id=? AND status='running' AND worker_token=?", (str(error), time.time(), job_id, worker_token))
            else:
                cur = conn.execute("UPDATE jobs SET status='failed', error=?, updated_at=?, lease_expires_at=NULL WHERE job_id=?", (str(error), time.time(), job_id))
            if cur.rowcount != 1:
                raise RuntimeError(f"Could not fail job {job_id}; lease ownership was lost.")

    def recover_stale(self, max_age_seconds: float = 21600.0) -> int:
        now = time.time()
        cutoff = now - float(max_age_seconds)
        with self._connect() as conn:
            cur = conn.execute("UPDATE jobs SET status='queued', updated_at=?, worker_token=NULL, lease_expires_at=NULL, heartbeat_at=NULL WHERE status='running' AND (lease_expires_at IS NULL AND updated_at < ? OR lease_expires_at IS NOT NULL AND lease_expires_at < ?)", (now, cutoff, now))
            return int(cur.rowcount)

from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


class ProductionPlanStore:
    """Atomic, per-plan persistence for managed production plan JSON files."""

    _locks: dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    @classmethod
    def _thread_lock_for(cls, path: Path) -> threading.RLock:
        key = str(path.resolve())
        with cls._locks_guard:
            lock = cls._locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._locks[key] = lock
            return lock

    @staticmethod
    def _lock_path(path: Path) -> Path:
        return path.with_name(f".{path.name}.lock")

    @classmethod
    @contextmanager
    def lock(cls, path: Path | str) -> Iterator[Path]:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with cls._thread_lock_for(target):
            lock_path = cls._lock_path(target)
            handle = lock_path.open("a+b")
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield target
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                handle.close()

    @staticmethod
    def load_unlocked(path: Path | str) -> dict[str, Any]:
        target = Path(path)
        try:
            plan = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid production plan JSON: {target}") from exc
        if not isinstance(plan, dict):
            raise RuntimeError(f"Production plan must be a JSON object: {target}")
        return plan

    @classmethod
    def load(cls, path: Path | str) -> dict[str, Any]:
        target = Path(path).resolve()
        with cls.lock(target):
            return cls.load_unlocked(target)

    @staticmethod
    def atomic_save_unlocked(path: Path | str, plan: dict[str, Any]) -> Path:
        target = Path(path).resolve()
        if not isinstance(plan, dict):
            raise TypeError("Production plan must be a dictionary.")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(plan, indent=2, ensure_ascii=False).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
            try:
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            try:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            except OSError:
                pass
        return target

    @classmethod
    def atomic_save(cls, path: Path | str, plan: dict[str, Any]) -> Path:
        target = Path(path).resolve()
        with cls.lock(target):
            return cls.atomic_save_unlocked(target, plan)

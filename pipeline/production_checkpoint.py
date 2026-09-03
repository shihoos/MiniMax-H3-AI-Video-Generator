from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager

try:
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


class ProductionCheckpoint:
    """Durable, atomic, cross-process-safe production state.

    The checkpoint is the authority for render progress. Thread locks protect
    one Python process; an adjacent lock file + flock protects independent
    processes on Linux/Kaggle as well.
    """

    VERSION = 3
    FILENAME = "director_checkpoint.json"
    LOCK_FILENAME = ".director_checkpoint.lock"

    _update_locks: dict[str, threading.RLock] = {}
    _update_locks_guard = threading.Lock()

    def __init__(self, project_root: Path | str):
        self.project_root = Path(project_root).resolve()
        self.root = self.project_root / "data" / "production" / "sessions"
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _lock_for(cls, path: Path) -> threading.RLock:
        key = str(path.resolve())
        with cls._update_locks_guard:
            lock = cls._update_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._update_locks[key] = lock
            return lock

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def digest_text(value: str) -> str:
        normalized = (
            str(value or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def digest_object(value: Any) -> str:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def plan_digest(cls, plan: dict[str, Any]) -> str:
        """Hash semantic plan content while excluding volatile metadata.

        This prevents a changed/foreign plan from reusing old completed shot
        outputs while allowing harmless metadata such as timestamps to differ.
        """
        if not isinstance(plan, dict):
            raise TypeError("Production plan must be a dictionary.")

        normalized = deepcopy(plan)
        for key in (
            "created_at",
            "updated_at",
            "preview_ready",
            "production_id",
            "plan_sha256",
            "runtime_diagnostics",
            "runtime_diagnostics_warning",
            "production_manifest",
            "production_manifest_path",
            "approval",
        ):
            normalized.pop(key, None)
        return cls.digest_object(normalized)

    @staticmethod
    def digest_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        sid = str(session_id or "").strip()
        if not sid or sid in {".", ".."}:
            raise ValueError("Invalid production session id.")
        if "/" in sid or "\\" in sid:
            raise ValueError("Production session id cannot contain path separators.")
        sid = re.sub(r"[^A-Za-z0-9._-]+", "_", sid).strip("._-")
        if not sid:
            raise ValueError("Production session id contains no valid characters.")
        return sid[:128]

    def session_dir(self, session_id: str) -> Path:
        path = self.root / self._safe_session_id(session_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / self.FILENAME

    def _lock_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / self.LOCK_FILENAME

    @contextmanager
    def _process_lock(self, session_id: str) -> Iterator[None]:
        lock_path = self._lock_path(session_id)
        thread_lock = self._lock_for(self.path(session_id))
        with thread_lock:
            handle = lock_path.open("a+b")
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except OSError:
                        pass
                handle.close()

    def _save_unlocked(self, session_id: str, state: dict[str, Any]) -> Path:
        if not isinstance(state, dict):
            raise TypeError("Checkpoint state must be a dictionary.")

        path = self.path(session_id)
        payload = deepcopy(state)
        payload["checkpoint_version"] = self.VERSION
        payload["session_id"] = self._safe_session_id(session_id)
        payload["updated_at"] = self._now()

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")

        fd, tmp_name = tempfile.mkstemp(
            prefix=".checkpoint_",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
            try:
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:
                pass
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass
        return path

    def save(self, session_id: str, state: dict[str, Any]) -> Path:
        with self._process_lock(session_id):
            return self._save_unlocked(session_id, state)

    def _load_unlocked(self, session_id: str) -> dict[str, Any]:
        path = self.path(session_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid production checkpoint: {path}") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Production checkpoint must be a JSON object.")

        try:
            version = int(data.get("checkpoint_version", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Production checkpoint version is invalid.") from exc
        if version not in {1, 2, self.VERSION}:
            raise RuntimeError(f"Unsupported production checkpoint version: {version}.")

        # Migrate legacy checkpoints in memory. Rendering resume is rejected by
        # ProductionRunner if the semantic plan fingerprint is unavailable.
        if version in {1, 2}:
            if not str(data.get("user_input", "") or "").strip():
                for candidate in (data.get("director_plan"), data.get("base_plan")):
                    if isinstance(candidate, dict):
                        story = str(candidate.get("story", "") or "").strip()
                        if story:
                            data["user_input"] = story
                            break
            user_input = str(data.get("user_input", "") or "").strip()
            if user_input:
                data.setdefault("user_input_sha256", self.digest_text(user_input))
            data["checkpoint_version"] = self.VERSION

        return data

    def load(self, session_id: str) -> dict[str, Any]:
        # Readers are locked too, so load never observes a partial replace.
        with self._process_lock(session_id):
            return self._load_unlocked(session_id)

    def delete(self, session_id: str) -> None:
        with self._process_lock(session_id):
            path = self.path(session_id)
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def update(self, session_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(updates, dict):
            raise TypeError("Checkpoint updates must be a dictionary.")

        with self._process_lock(session_id):
            state = self._load_unlocked(session_id)
            merged = deepcopy(updates)

            for key in ("completed_shot_ids", "completed_scene_ids"):
                if key in merged:
                    existing = {
                        str(v).strip()
                        for v in (state.get(key, []) or [])
                        if str(v).strip()
                    }
                    incoming = {
                        str(v).strip()
                        for v in (merged.get(key, []) or [])
                        if str(v).strip()
                    }
                    merged[key] = sorted(existing | incoming)

            if "completed_shots" in merged:
                existing_shots = state.get("completed_shots", {})
                if not isinstance(existing_shots, dict):
                    existing_shots = {}
                incoming_shots = merged.get("completed_shots", {})
                if not isinstance(incoming_shots, dict):
                    incoming_shots = {}
                combined = deepcopy(existing_shots)
                combined.update(deepcopy(incoming_shots))
                merged["completed_shots"] = combined

            durable_shots = merged.get("completed_shots", state.get("completed_shots", {}))
            if isinstance(durable_shots, dict):
                merged["completed_shot_ids"] = sorted(
                    {
                        str(k).strip()
                        for k in durable_shots
                        if str(k).strip()
                    }
                    | {
                        str(v).strip()
                        for v in (merged.get("completed_shot_ids", []) or [])
                        if str(v).strip()
                    }
                )

            state.update(merged)
            return_state = deepcopy(state)
            self._save_unlocked(session_id, return_state)
            return return_state

    def mark_rendering(
        self,
        session_id: str,
        current_scene_id: str = "",
        completed_shot_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.update(
            session_id,
            {
                "status": "rendering",
                "stage": "rendering",
                "current_scene_id": current_scene_id,
                "completed_shot_ids": list(completed_shot_ids or []),
                "error": "",
            },
        )

    def mark_failed(
        self,
        session_id: str,
        *,
        error: str,
        stage: str = "rendering",
        current_scene_id: str = "",
        completed_shot_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.update(
            session_id,
            {
                "status": "failed",
                "stage": stage,
                "current_scene_id": current_scene_id,
                "completed_shot_ids": list(completed_shot_ids or []),
                "error": str(error or ""),
            },
        )

    def mark_completed(
        self,
        session_id: str,
        *,
        final_video: str,
        shot_outputs: list[str],
        completed_shot_ids: list[str],
    ) -> dict[str, Any]:
        return self.update(
            session_id,
            {
                "status": "completed",
                "stage": "render_complete",
                "current_scene_id": "",
                "completed_shot_ids": list(completed_shot_ids),
                "final_video": str(final_video),
                "shot_outputs": [str(p) for p in shot_outputs],
                "error": "",
            },
        )

    def list_sessions(self) -> list[str]:
        if not self.root.exists():
            return []
        result = []
        for path in self.root.iterdir():
            if path.is_dir() and (path / self.FILENAME).is_file():
                result.append(path.name)
        result.sort(
            key=lambda sid: (self.path(sid).stat().st_mtime, sid),
            reverse=True,
        )
        return result

    def latest_resumable(self, mode: str, user_input: str) -> dict[str, Any] | None:
        wanted_hash = self.digest_text(user_input)
        for sid in self.list_sessions():
            try:
                state = self.load(sid)
            except (OSError, RuntimeError, ValueError, TypeError):
                continue
            if state.get("status") not in {"ready", "running", "rendering", "interrupted", "failed"}:
                continue
            if state.get("mode") != mode:
                continue
            if state.get("user_input_sha256") != wanted_hash:
                continue
            if not state.get("plan_sha256"):
                # Never surface a legacy checkpoint as a safe resume candidate.
                continue
            state["session_id"] = sid
            return state
        return None

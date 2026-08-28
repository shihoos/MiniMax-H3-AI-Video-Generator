from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any


class ProductionCheckpoint:
    VERSION = 2
    _update_locks: dict[str, threading.RLock] = {}
    _update_locks_guard = threading.Lock()
    FILENAME = "director_checkpoint.json"

    def __init__(
        self,
        project_root: Path | str,
    ):
        self.project_root = Path(
            project_root
        ).resolve()

        self.root = (
            self.project_root
            / "data"
            / "production"
            / "sessions"
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @classmethod
    def _lock_for(
        cls,
        path: Path,
    ) -> threading.RLock:
        key = str(path.resolve())

        with cls._update_locks_guard:
            lock = cls._update_locks.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._update_locks[key] = lock
            return lock

    @staticmethod
    def digest_text(
        value: str,
    ) -> str:
        normalized = (
            str(value or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def digest_file(
        path: Path,
    ) -> str:
        digest = hashlib.sha256()

        with Path(path).open(
            "rb"
        ) as handle:
            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _safe_session_id(
        session_id: str,
    ) -> str:
        sid = str(
            session_id or ""
        ).strip()

        if not sid:
            raise ValueError(
                "Production session id cannot be empty."
            )

        if sid in {".", ".."}:
            raise ValueError(
                "Invalid production session id."
            )

        if "/" in sid or "\\" in sid:
            raise ValueError(
                "Production session id cannot contain path separators."
            )

        sid = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            sid,
        ).strip("._-")

        if not sid:
            raise ValueError(
                "Production session id contains no valid characters."
            )

        return sid[:128]

    def session_dir(
        self,
        session_id: str,
    ) -> Path:
        sid = self._safe_session_id(
            session_id
        )

        path = self.root / sid

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def path(
        self,
        session_id: str,
    ) -> Path:
        return (
            self.session_dir(session_id)
            / self.FILENAME
        )

    def save(
        self,
        session_id: str,
        state: dict[str, Any],
    ) -> Path:
        if not isinstance(state, dict):
            raise TypeError(
                "Checkpoint state must be a dictionary."
            )

        path = self.path(
            session_id
        )

        payload = deepcopy(
            state
        )

        payload["checkpoint_version"] = self.VERSION
        payload["session_id"] = (
            self._safe_session_id(
                session_id
            )
        )

        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")

        fd, tmp_name = tempfile.mkstemp(
            prefix=".checkpoint_",
            suffix=".tmp",
            dir=str(path.parent),
        )

        try:
            with os.fdopen(
                fd,
                "wb",
            ) as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(
                    handle.fileno()
                )

            os.replace(
                tmp_name,
                path,
            )

        finally:
            try:
                if os.path.exists(
                    tmp_name
                ):
                    os.unlink(
                        tmp_name
                    )
            except OSError:
                pass

        return path

    def load(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        path = self.path(
            session_id
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid production checkpoint: {path}"
            ) from exc

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                "Production checkpoint must be a JSON object."
            )

        try:
            version = int(
                data.get(
                    "checkpoint_version",
                    0,
                )
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                "Production checkpoint version is invalid."
            ) from exc

        if version not in {1, self.VERSION}:
            raise RuntimeError(
                "Unsupported production checkpoint version: "
                f"{version}."
            )

        # Version-1 checkpoints predate the explicit user_input field.
        # Migrate them in memory from the persisted planning payload so
        # resume_latest/resume_production_plan remains usable. The next
        # atomic update/save persists the migrated state as VERSION 2.
        if version == 1:
            if not str(
                data.get(
                    "user_input",
                    "",
                )
                or ""
            ).strip():
                candidates = (
                    data.get("director_plan"),
                    data.get("base_plan"),
                )

                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    story = str(
                        candidate.get(
                            "story",
                            "",
                        )
                        or ""
                    ).strip()
                    if story:
                        data["user_input"] = story
                        break

            data["checkpoint_version"] = self.VERSION

            user_input = str(
                data.get(
                    "user_input",
                    "",
                )
                or ""
            ).strip()

            if user_input:
                data.setdefault(
                    "user_input_sha256",
                    self.digest_text(user_input),
                )

        return data

    def delete(
        self,
        session_id: str,
    ) -> None:
        path = self.path(
            session_id
        )

        try:
            path.unlink()
        except FileNotFoundError:
            pass


    def update(
        self,
        session_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically read, merge, and persist one checkpoint state.

        The production runner can update one checkpoint from multiple GPU
        worker threads. A plain load -> modify -> save sequence loses updates
        under concurrency. This method serializes read/modify/write per
        checkpoint path and merges monotonic progress fields.
        """
        if not isinstance(
            updates,
            dict,
        ):
            raise TypeError(
                "Checkpoint updates must be a dictionary."
            )

        path = self.path(
            session_id
        )
        lock = self._lock_for(path)

        with lock:
            state = self.load(
                session_id
            )

            merged = deepcopy(
                updates
            )

            # Completed shots/scenes are monotonic during a production run.
            # Union them with the durable state instead of replacing the
            # current set with a stale worker snapshot.
            for key in (
                "completed_shot_ids",
                "completed_scene_ids",
            ):
                if key in merged:
                    existing = {
                        str(value).strip()
                        for value in (
                            state.get(key, [])
                            or []
                        )
                        if str(value).strip()
                    }
                    incoming = {
                        str(value).strip()
                        for value in (
                            merged.get(key, [])
                            or []
                        )
                        if str(value).strip()
                    }
                    merged[key] = sorted(
                        existing | incoming
                    )

            # Render workers complete independently on different GPUs.
            # Merge their per-shot records atomically so one worker cannot
            # erase another worker's completed shot or GPU ownership.
            if "completed_shots" in merged:
                existing_shots = state.get(
                    "completed_shots",
                    {},
                )
                if not isinstance(
                    existing_shots,
                    dict,
                ):
                    existing_shots = {}

                incoming_shots = merged.get(
                    "completed_shots",
                    {},
                )
                if not isinstance(
                    incoming_shots,
                    dict,
                ):
                    incoming_shots = {}

                combined_shots = deepcopy(
                    existing_shots
                )
                combined_shots.update(
                    incoming_shots
                )

                merged[
                    "completed_shots"
                ] = combined_shots

            # Keep completed_shot_ids synchronized with the durable
            # completed_shots map.
            durable_shots = merged.get(
                "completed_shots",
                {},
            )
            if isinstance(
                durable_shots,
                dict,
            ):
                merged[
                    "completed_shot_ids"
                ] = sorted(
                    {
                        str(shot_id).strip()
                        for shot_id
                        in durable_shots.keys()
                        if str(shot_id).strip()
                    }
                    | {
                        str(value).strip()
                        for value
                        in (
                            merged.get(
                                "completed_shot_ids",
                                [],
                            )
                            or []
                        )
                        if str(value).strip()
                    }
                )

            state.update(
                merged
            )

            state["updated_at"] = (
                datetime.now().isoformat()
            )

            self.save(
                session_id,
                state,
            )

            return state

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
                "completed_shot_ids": list(
                    completed_shot_ids or []
                ),
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
                "completed_shot_ids": list(
                    completed_shot_ids or []
                ),
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
        state = self.load(session_id)

        return self.update(
            session_id,
            {
                "status": "completed",
                "stage": "render_complete",
                "current_scene_id": "",
                "completed_scene_ids": list(
                    state.get(
                        "completed_scene_ids",
                        [],
                    )
                ),
                "completed_shot_ids": list(
                    completed_shot_ids
                ),
                "final_video": str(final_video),
                "shot_outputs": [
                    str(path)
                    for path in shot_outputs
                ],
                "error": "",
            },
        )

    def list_sessions(
        self,
    ) -> list[str]:
        if not self.root.exists():
            return []

        result = []

        for path in self.root.iterdir():
            if (
                path.is_dir()
                and (
                    path / self.FILENAME
                ).is_file()
            ):
                result.append(
                    path.name
                )

        result.sort(
            key=lambda sid: (
                self.path(sid).stat().st_mtime,
                sid,
            ),
            reverse=True,
        )

        return result

    def latest_resumable(
        self,
        mode: str,
        user_input: str,
    ) -> dict[str, Any] | None:
        wanted_hash = self.digest_text(
            user_input
        )

        for sid in self.list_sessions():
            try:
                state = self.load(
                    sid
                )
            except (
                OSError,
                RuntimeError,
                ValueError,
                TypeError,
            ):
                continue

            if state.get(
                "status"
            ) not in {
                "ready",
                "running",
                "interrupted",
                "failed",
            }:
                continue

            if state.get(
                "mode"
            ) != mode:
                continue

            if state.get(
                "user_input_sha256"
            ) != wanted_hash:
                continue

            state["session_id"] = sid
            return state

        return None

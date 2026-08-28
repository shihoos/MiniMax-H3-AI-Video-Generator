from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any


class ProductionCheckpoint:
    VERSION = 2
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

        version = int(
            data.get(
                "checkpoint_version",
                0,
            )
        )

        if version not in {
            1,
            self.VERSION,
        }:
            raise RuntimeError(
                "Unsupported production checkpoint version: "
                f"{version}."
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

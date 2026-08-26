from __future__ import annotations

import json
import shutil
from pathlib import Path

from planner.config import IDENTITY_DIR


class IdentityAnchorStore:

    def __init__(
        self,
        project_root: Path,
    ):
        self.project_root = Path(
            project_root
        )

        self.root = (
            self.project_root
            / IDENTITY_DIR.relative_to(
                self.project_root
            )
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _safe(
        value: str,
    ) -> str:
        return "".join(
            char
            if (
                char.isalnum()
                or char in "._-"
            )
            else "_"
            for char in str(value)
        )

    def _character_dir(
        self,
        character_id: str,
    ) -> Path:

        path = (
            self.root
            / self._safe(
                character_id
            )
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    def latest_anchor(
        self,
        character_id: str,
    ) -> Path | None:

        directory = (
            self._character_dir(
                character_id
            )
        )

        anchors = sorted(
            directory.glob(
                "*_anchor.png"
            ),
            key=lambda path:
                path.stat().st_mtime,
            reverse=True,
        )

        return (
            anchors[0]
            if anchors
            else None
        )

    def save_first_anchor(
        self,
        *,
        character_id: str,
        shot_id: str,
        source_frame: Path,
    ) -> Path:

        existing = self.latest_anchor(
            character_id
        )

        if existing is not None:
            return existing

        source_frame = Path(
            source_frame
        )

        if not source_frame.is_file():
            raise FileNotFoundError(
                source_frame
            )

        directory = (
            self._character_dir(
                character_id
            )
        )

        destination = (
            directory
            / (
                f"{self._safe(shot_id)}"
                "_anchor.png"
            )
        )

        shutil.copy2(
            source_frame,
            destination,
        )

        metadata = {
            "character_id": character_id,
            "shot_id": shot_id,
            "anchor_path": str(
                destination
            ),
            "type": "first_successful_visual_anchor",
        }

        destination.with_suffix(
            ".json"
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return destination

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
    def _safe(value: str) -> str:
        return "".join(
            char
            if (
                char.isalnum()
                or char in "._-"
            )
            else "_"
            for char in str(value)
        )

    def character_dir(
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

    def save_anchor(
        self,
        *,
        character_id: str,
        shot_id: str,
        source_frame: Path,
    ) -> Path:

        source_frame = Path(
            source_frame
        )

        if not source_frame.is_file():
            raise FileNotFoundError(
                source_frame
            )

        destination = (
            self.character_dir(
                character_id
            )
            / f"{self._safe(shot_id)}_anchor.png"
        )

        shutil.copy2(
            source_frame,
            destination,
        )

        metadata = {
            "character_id": character_id,
            "shot_id": shot_id,
            "anchor": str(
                destination
            ),
        }

        (
            destination.with_suffix(
                ".json"
            )
        ).write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        return destination

    def latest_anchor(
        self,
        character_id: str,
    ):

        directory = self.character_dir(
            character_id
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

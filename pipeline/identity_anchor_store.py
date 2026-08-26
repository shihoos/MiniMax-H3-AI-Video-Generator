from __future__ import annotations

import json
import shutil
from pathlib import Path


class IdentityAnchorStore:

    def __init__(
        self,
        project_root: Path,
        production_id: str | None = None,
    ):

        self.project_root = (
            Path(project_root)
            .resolve()
        )

        # Derive storage from the supplied project root.
        #
        # This is important for:
        # - normal repository execution
        # - temporary isolated tests
        # - production-scoped storage
        #
        # Do not use planner.config.IDENTITY_DIR here because
        # that constant points at the real repository root and
        # cannot be made relative to an arbitrary temporary root.

        base = (
            self.project_root
            / "data"
            / "production"
            / "identity"
        )

        self.production_id = (
            self._safe(
                production_id
            )
            if production_id
            else None
        )

        self.root = (
            base / self.production_id
            if self.production_id
            else base
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _safe(
        value: str | None,
    ) -> str:

        text = str(
            value or ""
        ).strip()

        return "".join(
            char
            if (
                char.isalnum()
                or char in "._-"
            )
            else "_"
            for char in text
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

        existing = (
            self.latest_anchor(
                character_id
            )
        )

        if existing is not None:
            return existing

        source_frame = (
            Path(source_frame)
            .resolve()
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

        if (
            not destination.is_file()
            or destination.stat().st_size <= 0
        ):

            raise RuntimeError(
                "Identity anchor was not "
                "created correctly:\n"
                f"{destination}"
            )

        metadata = {
            "character_id":
                character_id,

            "shot_id":
                shot_id,

            "production_id":
                self.production_id,

            "anchor_path":
                str(destination),

            "type":
                "first_successful_visual_anchor",
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

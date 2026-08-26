from __future__ import annotations

import subprocess
from pathlib import Path

from pipeline.identity_anchor_store import (
    IdentityAnchorStore,
)


class H3SceneContinuity:

    def __init__(
        self,
        project_root: Path,
    ):
        self.project_root = Path(
            project_root
        )

        self.root = (
            self.project_root
            / "data"
            / "production"
            / "continuity"
        )

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.identity_store = (
            IdentityAnchorStore(
                self.project_root
            )
        )

    def extract_last_frame(
        self,
        video_path: Path,
        scene_id: str,
        shot_id: str,
    ) -> Path:

        video_path = Path(
            video_path
        )

        if not video_path.is_file():
            raise FileNotFoundError(
                video_path
            )

        destination = (
            self.root
            / str(scene_id)
            / f"{shot_id}_last_frame.png"
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        command = [
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.05",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-update",
            "1",
            str(destination),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "Unable to extract H3 last frame:\n"
                + result.stderr[-5000:]
            )

        if not destination.is_file():
            raise RuntimeError(
                f"Last frame was not created: "
                f"{destination}"
            )

        return destination

    def prepare_next_shot(
        self,
        video_path: Path,
        scene_id: str,
        shot_id: str,
    ) -> Path:

        return self.extract_last_frame(
            video_path=video_path,
            scene_id=scene_id,
            shot_id=shot_id,
        )

    def persist_character_anchors(
        self,
        *,
        shot: dict,
        frame_path: Path,
    ) -> list[str]:

        character_ids = (
            shot.get(
                "character_ids",
                [],
            )
            or []
        )

        saved = []

        for character_id in character_ids:

            anchor = (
                self.identity_store.save_first_anchor(
                    character_id=character_id,
                    shot_id=shot["shot_id"],
                    source_frame=frame_path,
                )
            )

            saved.append(
                str(anchor)
            )

        return saved

    def character_anchor(
        self,
        character_id: str,
    ):

        anchor = (
            self.identity_store.latest_anchor(
                character_id
            )
        )

        return (
            str(anchor)
            if anchor
            else None
        )

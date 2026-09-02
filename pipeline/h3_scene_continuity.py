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
        production_id: str | None = None,
    ):

        self.project_root = Path(
            project_root
        ).resolve()

        base = (
            self.project_root
            / "data"
            / "production"
            / "continuity"
        )

        self.production_id = (
            self._safe(production_id)
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

        self.identity_store = (
            IdentityAnchorStore(
                self.project_root,
                production_id=self.production_id,
            )
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

    def extract_last_frame(
        self,
        video_path: Path,
        scene_id: str,
        shot_id: str,
    ) -> Path:

        video_path = Path(
            video_path
        ).resolve()

        if not video_path.is_file():
            raise FileNotFoundError(
                video_path
            )

        safe_scene = self._safe(
            scene_id
        )

        safe_shot = self._safe(
            shot_id
        )

        destination = (
            self.root
            / safe_scene
            / f"{safe_shot}_last_frame.png"
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

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=30.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out while extracting the last frame from {video_path}."
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                "Unable to extract H3 last frame:\n"
                + result.stderr[-5000:]
            )

        if (
            not destination.is_file()
            or destination.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Last frame was not created correctly: "
                f"{destination}"
            )

        return destination

    def extract_frame_at(
        self,
        video_path: Path,
        seconds: float,
        *,
        scene_id: str,
        shot_id: str,
        label: str,
    ) -> Path:
        video_path = Path(video_path).resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        seconds = max(0.0, float(seconds))
        safe_scene = self._safe(scene_id)
        safe_shot = self._safe(shot_id)
        safe_label = self._safe(label)
        destination = self.root / safe_scene / f"{safe_shot}_{safe_label}_frame.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Output seeking: decode from the input and seek after -i so the
        # requested timestamp is selected from decoded frames instead of
        # snapping to the nearest preceding keyframe. This is important for
        # exact boundary continuity references.
        command = [
            "ffmpeg", "-y", "-i", str(video_path), "-ss", f"{seconds:.6f}",
            "-frames:v", "1", "-update", "1", str(destination),
        ]
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=30.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Timed out while extracting frame at {seconds:.3f}s from {video_path}.") from exc
        if result.returncode != 0 or not destination.is_file() or destination.stat().st_size <= 0:
            raise RuntimeError("Unable to extract requested frame:\n" + result.stderr[-5000:])
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
        character_map: dict[str, str] | None = None,
    ) -> list[str]:

        # character_ids remains the canonical identity-store key.
        # When a caller only has production-level character names, it may
        # provide an explicit name->character_id map. Never derive IDs
        # heuristically from display names.
        character_ids = list(
            shot.get(
                "character_ids",
                [],
            )
            or []
        )

        if (
            not character_ids
            and character_map
        ):
            for character_name in (
                shot.get(
                    "characters",
                    [],
                )
                or []
            ):
                key = str(
                    character_name or ""
                ).strip().lower()

                character_id = character_map.get(
                    key
                )

                if character_id:
                    character_ids.append(
                        character_id
                    )

        saved = []
        seen = set()

        for character_id in character_ids:

            character_id = str(
                character_id or ""
            ).strip()

            if not character_id:
                continue

            key = character_id.lower()

            if key in seen:
                continue

            seen.add(
                key
            )

            anchor = (
                self.identity_store
                .save_first_anchor(
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

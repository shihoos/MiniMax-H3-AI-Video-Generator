from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class ContinuityState:

    previous_shot_id: str | None = None
    previous_video: str | None = None
    last_frame: str | None = None

    location: str = ""
    characters: list[str] | None = None
    camera_shot: str = ""
    camera_movement: str = ""
    lighting: str = ""
    mood: str = ""
    continuity_notes: str = ""

    def __post_init__(self):
        if self.characters is None:
            self.characters = []


class ContinuityManager:

    def __init__(
        self,
        project_root=None,
    ):
        self.project_root = (
            Path(project_root)
            if project_root is not None
            else Path.cwd()
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

        self.states = {}

    def build_context(
        self,
        previous_shot=None,
    ) -> str:

        if previous_shot is None:
            return ""

        parts = []

        if previous_shot.location:
            parts.append(
                f"Location continuity: "
                f"{previous_shot.location}"
            )

        if previous_shot.characters:
            parts.append(
                "Character continuity: "
                + ", ".join(
                    previous_shot.characters
                )
            )

        if previous_shot.camera_shot:
            parts.append(
                f"Camera continuity: "
                f"{previous_shot.camera_shot}"
            )

        if previous_shot.camera_movement:
            parts.append(
                f"Camera movement continuity: "
                f"{previous_shot.camera_movement}"
            )

        if previous_shot.lighting:
            parts.append(
                f"Lighting continuity: "
                f"{previous_shot.lighting}"
            )

        if previous_shot.mood:
            parts.append(
                f"Mood continuity: "
                f"{previous_shot.mood}"
            )

        if previous_shot.continuity_notes:
            parts.append(
                "Continuity requirements:\n"
                + previous_shot.continuity_notes
            )

        return "\n".join(parts)

    def apply_scene_continuity(
        self,
        shots,
        previous_shot=None,
    ):

        if not shots:
            return []

        for index, shot in enumerate(shots):

            if index == 0:
                shot.previous_shot = (
                    previous_shot.shot_id
                    if previous_shot
                    else None
                )
            else:
                shot.previous_shot = (
                    shots[index - 1].shot_id
                )

            if index + 1 < len(shots):
                shot.next_shot = (
                    shots[index + 1].shot_id
                )
            else:
                shot.next_shot = None

        if previous_shot is not None:

            context = self.build_context(
                previous_shot
            )

            shots[0].continuity_notes = (
                (
                    context
                    + "\n"
                    + shots[0].continuity_notes
                ).strip()
            )

            # Last frame becomes a real reference.
            last_frame = getattr(
                previous_shot,
                "last_frame_path",
                None,
            )

            if last_frame:
                shots[0].reference_images = [
                    last_frame,
                    *shots[0].reference_images,
                ][:9]

        return shots

    def save_state(
        self,
        scene_id: str,
        state: ContinuityState,
    ) -> None:

        path = (
            self.root
            / f"{scene_id}.json"
        )

        path.write_text(
            json.dumps(
                asdict(state),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self.states[
            scene_id
        ] = state

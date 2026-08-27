from __future__ import annotations

from copy import deepcopy
from typing import Any


class CinematicCompiler:
    """
    Deterministic production compiler for Qwen cinematic shot blueprints.

    Qwen is responsible for creative decisions:
      - action
      - camera shot
      - camera movement
      - lens / depth of field
      - composition
      - lighting
      - color temperature
      - mood
      - visual prompt

    This compiler is responsible for:
      - preserving those creative decisions
      - filling mechanical production fields
      - enforcing scene / character contracts
      - preserving continuity
      - producing the complete downstream Shot dictionary

    No model or GPU work happens here.
    """

    DEFAULT_NEGATIVE_PROMPT = (
        "identity drift, duplicate character, face deformation, "
        "inconsistent clothing, inconsistent appearance, malformed hands"
    )

    DEFAULT_SOUND = (
        "Natural cinematic environmental sound appropriate to the scene."
    )

    DEFAULT_MUSIC = (
        "Subtle cinematic score supporting the scene without "
        "overpowering dialogue or sound effects."
    )

    REQUIRED_CREATIVE_FIELDS = (
        "shot_id",
        "scene_id",
        "action",
        "camera_shot",
        "camera_movement",
        "lens_and_depth_of_field",
        "composition_notes",
        "lighting",
        "color_temperature",
        "mood",
        "visual_prompt",
    )

    def __init__(
        self,
        character_names: set[str] | None = None,
    ) -> None:

        self.character_names = {
            str(name).strip().lower()
            for name in (
                character_names or set()
            )
            if str(name).strip()
        }

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    @staticmethod
    def _string(
        value: Any,
        default: str = "",
    ) -> str:

        text = str(
            value
            or ""
        ).strip()

        return (
            text
            if text
            else default
        )

    @staticmethod
    def _list(
        value: Any,
    ) -> list[str]:

        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):

            return [
                part.strip()
                for part
                in value.split(",")
                if part.strip()
            ]

        if isinstance(
            value,
            (list, tuple, set),
        ):

            return [
                str(item).strip()
                for item
                in value
                if str(item).strip()
            ]

        return [
            str(value).strip()
        ] if str(value).strip() else []

    @staticmethod
    def _float(
        value: Any,
        default: float,
    ) -> float:

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return default

        if number <= 0:
            return default

        return number

    # ============================================================
    # CHARACTER CONTRACT
    # ============================================================

    def _canonical_characters(
        self,
        values: Any,
        scene_characters: list[str],
    ) -> list[str]:

        allowed = dict(
            (
                name.lower(),
                name,
            )
            for name
            in (
                scene_characters
                or []
            )
        )

        if self.character_names:

            allowed.update(
                (
                    name,
                    name,
                )
                for name
                in self.character_names
            )

        result: list[str] = []
        seen: set[str] = set()

        for value in self._list(
            values
        ):

            key = value.lower()

            canonical = (
                allowed.get(
                    key
                )
            )

            if canonical is None:
                continue

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                canonical
            )

        if not result:

            for value in (
                scene_characters
                or []
            ):

                key = str(
                    value
                ).strip().lower()

                if not key:
                    continue

                if key in seen:
                    continue

                seen.add(
                    key
                )

                result.append(
                    str(
                        value
                    ).strip()
                )

        return result

    # ============================================================
    # DERIVED PRODUCTION FIELDS
    # ============================================================

    @classmethod
    def _derive_detailed_description(
        cls,
        scene: dict,
        shot: dict,
    ) -> str:

        description = cls._string(
            shot.get(
                "detailed_description"
            )
        )

        if description:
            return description

        action = cls._string(
            shot.get(
                "action"
            )
        )

        camera = cls._string(
            shot.get(
                "camera_shot"
            )
        )

        movement = cls._string(
            shot.get(
                "camera_movement"
            )
        )

        location = cls._string(
            shot.get(
                "location"
            )
            or scene.get(
                "location"
            )
        )

        parts = [
            part
            for part
            in (
                action,
                (
                    f"{camera} framing"
                    if camera
                    else ""
                ),
                (
                    f"{movement}"
                    if movement
                    else ""
                ),
                (
                    f"in {location}"
                    if location
                    else ""
                ),
            )
            if part
        ]

        return (
            ". ".join(
                parts
            )
            + "."
            if parts
            else cls._string(
                scene.get(
                    "description"
                )
            )
        )

    @classmethod
    def _derive_soundscape(
        cls,
        scene: dict,
        shot: dict,
    ) -> str:

        existing = cls._string(
            shot.get(
                "overall_soundscape"
            )
        )

        if existing:
            return existing

        atmosphere = cls._string(
            scene.get(
                "atmosphere"
            )
        )

        if atmosphere:
            return (
                f"{atmosphere}. "
                + cls.DEFAULT_SOUND
            )

        return cls.DEFAULT_SOUND

    @classmethod
    def _derive_music(
        cls,
        scene: dict,
        shot: dict,
    ) -> str:

        existing = cls._string(
            shot.get(
                "non_diegetic_music"
            )
        )

        if existing:
            return existing

        mood = cls._string(
            shot.get(
                "mood"
            )
            or scene.get(
                "mood"
            )
        )

        if mood:
            return (
                f"Score shaped around the "
                f"{mood.lower()} mood."
            )

        return cls.DEFAULT_MUSIC

    @classmethod
    def _derive_continuity(
        cls,
        scene: dict,
        shot: dict,
    ) -> str:

        existing = cls._string(
            shot.get(
                "continuity_notes"
            )
        )

        if existing:
            return existing

        scene_notes = cls._string(
            scene.get(
                "continuity_notes"
            )
        )

        if scene_notes:
            return scene_notes

        return (
            "Maintain character identity, clothing, "
            "location, props, lighting direction, "
            "and visual style from the preceding shot."
        )

    @classmethod
    def _derive_negative_prompt(
        cls,
        shot: dict,
    ) -> str:

        existing = cls._string(
            shot.get(
                "negative_prompt"
            )
        )

        if existing:
            return existing

        return cls.DEFAULT_NEGATIVE_PROMPT

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_creative_fields(
        self,
        shot: dict,
    ) -> None:

        missing = [
            field
            for field
            in self.REQUIRED_CREATIVE_FIELDS
            if not self._string(
                shot.get(
                    field
                )
            )
        ]

        if missing:

            raise ValueError(
                "Cinematic shot is missing required "
                "creative fields: "
                + ", ".join(
                    missing
                )
            )

    def _validate_characters(
        self,
        characters: list[str],
    ) -> None:

        if not self.character_names:
            return

        unknown = [
            name
            for name
            in characters
            if name.lower()
            not in self.character_names
        ]

        if unknown:

            raise ValueError(
                "Cinematic shot contains unknown "
                "characters: "
                + ", ".join(
                    unknown
                )
            )

    # ============================================================
    # SHOT COMPILATION
    # ============================================================

    def compile_shot(
        self,
        scene: dict,
        qwen_shot: dict,
        ordinal: int,
    ) -> dict:

        if not isinstance(
            scene,
            dict,
        ):

            raise TypeError(
                "scene must be a dictionary."
            )

        if not isinstance(
            qwen_shot,
            dict,
        ):

            raise TypeError(
                "qwen_shot must be a dictionary."
            )

        scene_id = self._string(
            scene.get(
                "scene_id"
            )
        )

        if not scene_id:

            raise ValueError(
                "Scene has no scene_id."
            )

        shot = deepcopy(
            qwen_shot
        )

        self._validate_creative_fields(
            shot
        )

        scene_characters = self._list(
            scene.get(
                "characters",
                [],
            )
        )

        characters = (
            self._canonical_characters(
                shot.get(
                    "characters",
                    [],
                ),
                scene_characters,
            )
        )

        self._validate_characters(
            characters
        )

        shot_id = self._string(
            shot.get(
                "shot_id"
            )
        )

        if not shot_id:

            shot_id = (
                f"{scene_id}"
                f"_shot_{ordinal:03d}"
            )

        location = self._string(
            shot.get(
                "location"
            )
            or scene.get(
                "location"
            )
        )

        duration = self._float(
            shot.get(
                "duration_seconds"
            ),
            5.0,
        )

        compiled = {
            "shot_id":
                shot_id,

            "scene_id":
                scene_id,

            "order":
                ordinal,

            "duration_seconds":
                duration,

            "characters":
                characters,

            "location":
                location,

            "action":
                self._string(
                    shot.get(
                        "action"
                    ),
                    self._string(
                        scene.get(
                            "scene_objective"
                        )
                        or scene.get(
                            "description"
                        )
                    ),
                ),

            "camera_shot":
                self._string(
                    shot.get(
                        "camera_shot"
                    )
                ),

            "camera_movement":
                self._string(
                    shot.get(
                        "camera_movement"
                    )
                ),

            "lens_and_depth_of_field":
                self._string(
                    shot.get(
                        "lens_and_depth_of_field"
                    )
                ),

            "composition_notes":
                self._string(
                    shot.get(
                        "composition_notes"
                    )
                ),

            "lighting":
                self._string(
                    shot.get(
                        "lighting"
                    )
                    or scene.get(
                        "lighting"
                    )
                ),

            "color_temperature":
                self._string(
                    shot.get(
                        "color_temperature"
                    )
                    or scene.get(
                        "color_temperature"
                    )
                ),

            "mood":
                self._string(
                    shot.get(
                        "mood"
                    )
                    or scene.get(
                        "mood"
                    )
                ),

            "visual_prompt":
                self._string(
                    shot.get(
                        "visual_prompt"
                    ),
                    self._string(
                        scene.get(
                            "description"
                        )
                    ),
                ),

            "retention_analysis":
                self._string(
                    shot.get(
                        "retention_analysis"
                    ),
                    "Advance the scene while preserving "
                    "visual and narrative continuity.",
                ),

            "detailed_description":
                self._derive_detailed_description(
                    scene,
                    shot,
                ),

            "overall_soundscape":
                self._derive_soundscape(
                    scene,
                    shot,
                ),

            "non_diegetic_music":
                self._derive_music(
                    scene,
                    shot,
                ),

            "negative_prompt":
                self._derive_negative_prompt(
                    shot
                ),

            "continuity_notes":
                self._derive_continuity(
                    scene,
                    shot,
                ),

            "speaking_characters":
                self._canonical_characters(
                    shot.get(
                        "speaking_characters",
                        [],
                    ),
                    characters,
                ),

            "speech_text":
                self._string(
                    shot.get(
                        "speech_text"
                    )
                ),
        }

        return compiled

    def compile_scene(
        self,
        scene: dict,
        qwen_shots: list[dict],
    ) -> list[dict]:

        if not qwen_shots:

            raise ValueError(
                "Cannot compile a scene with no Qwen shots."
            )

        result: list[dict] = []

        for index, shot in enumerate(
            qwen_shots,
            start=1,
        ):

            result.append(
                self.compile_shot(
                    scene,
                    shot,
                    index,
                )
            )

        return result

    def compile_all(
        self,
        scenes: list[dict],
        qwen_shots: list[dict],
    ) -> list[dict]:

        if not isinstance(
            scenes,
            list,
        ):

            raise TypeError(
                "scenes must be a list."
            )

        if not isinstance(
            qwen_shots,
            list,
        ):

            raise TypeError(
                "qwen_shots must be a list."
            )

        scene_map = {
            str(
                scene.get(
                    "scene_id",
                    "",
                )
            ).strip():
                scene
            for scene
            in scenes
            if isinstance(
                scene,
                dict,
            )
        }

        grouped: dict[
            str,
            list[dict],
        ] = {}

        for shot in qwen_shots:

            if not isinstance(
                shot,
                dict,
            ):
                continue

            scene_id = self._string(
                shot.get(
                    "scene_id"
                )
            )

            if scene_id:

                grouped.setdefault(
                    scene_id,
                    [],
                ).append(
                    shot
                )

        compiled: list[dict] = []

        for scene in scenes:

            scene_id = self._string(
                scene.get(
                    "scene_id"
                )
            )

            scene_shots = grouped.get(
                scene_id,
                [],
            )

            if not scene_shots:

                continue

            compiled.extend(
                self.compile_scene(
                    scene,
                    scene_shots,
                )
            )

        return compiled

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


VALID_REFERENCE_MODES = {
    "missing",
    "provided",
    "story_generated",
}


def _require_nonempty(value: str, field_name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(
            f"{field_name} cannot be empty."
        )
    return value


def _as_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, (list, tuple, set)):
        raise TypeError(
            f"{field_name} must be a list-like collection of strings."
        )

    result = []
    for item in value:
        text = str(item).strip()
        if not text:
            raise ValueError(
                f"{field_name} cannot contain empty values."
            )
        result.append(text)

    return result


def _as_dict(value: Any, field_name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            f"{field_name} must be a dictionary."
        )
    return dict(value)


@dataclass
class Character:
    character_id: str
    name: str
    role: str
    description: str
    personality: str

    appearance: dict = field(default_factory=dict)
    clothing: dict = field(default_factory=dict)
    distinctive_features: list[str] = field(
        default_factory=list
    )
    character_state: dict = field(default_factory=dict)
    continuity_rules: list[str] = field(
        default_factory=list
    )

    reference_mode: str = "missing"

    reference_paths: list[str] = field(
        default_factory=list
    )
    reference_video_paths: list[str] = field(
        default_factory=list
    )
    reference_audio_paths: list[str] = field(
        default_factory=list
    )

    reference_path: Optional[str] = None
    reference_video_path: Optional[str] = None
    reference_audio_path: Optional[str] = None
    reference_mask_path: Optional[str] = None

    identity_profile: dict = field(default_factory=dict)
    story_state_profile: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.character_id = _require_nonempty(
            self.character_id,
            "character_id",
        )
        self.name = _require_nonempty(
            self.name,
            "name",
        )
        self.role = _require_nonempty(
            self.role,
            "role",
        )
        self.description = str(
            self.description or ""
        ).strip()
        self.personality = str(
            self.personality or ""
        ).strip()

        self.appearance = _as_dict(
            self.appearance,
            "appearance",
        )
        self.clothing = _as_dict(
            self.clothing,
            "clothing",
        )
        self.character_state = _as_dict(
            self.character_state,
            "character_state",
        )

        self.distinctive_features = _as_string_list(
            self.distinctive_features,
            "distinctive_features",
        )
        self.continuity_rules = _as_string_list(
            self.continuity_rules,
            "continuity_rules",
        )

        self.reference_mode = str(
            self.reference_mode or "missing"
        ).strip()

        if self.reference_mode not in VALID_REFERENCE_MODES:
            raise ValueError(
                "reference_mode must be one of: "
                + ", ".join(sorted(VALID_REFERENCE_MODES))
            )

        self.reference_paths = _as_string_list(
            self.reference_paths,
            "reference_paths",
        )
        self.reference_video_paths = _as_string_list(
            self.reference_video_paths,
            "reference_video_paths",
        )
        self.reference_audio_paths = _as_string_list(
            self.reference_audio_paths,
            "reference_audio_paths",
        )

        for field_name in (
            "reference_path",
            "reference_video_path",
            "reference_audio_path",
            "reference_mask_path",
        ):
            value = getattr(self, field_name)
            if value is not None:
                value = str(value).strip()
                setattr(
                    self,
                    field_name,
                    value or None,
                )

        self.identity_profile = _as_dict(
            self.identity_profile,
            "identity_profile",
        )
        self.story_state_profile = _as_dict(
            self.story_state_profile,
            "story_state_profile",
        )

    def normalized_reference_paths(self) -> list[str]:
        values = list(self.reference_paths or [])

        if (
            self.reference_path
            and self.reference_path not in values
        ):
            values.insert(0, self.reference_path)

        return [
            str(path)
            for path in values
            if str(path).strip()
        ]

    def normalized_video_paths(self) -> list[str]:
        values = list(
            self.reference_video_paths or []
        )

        if (
            self.reference_video_path
            and self.reference_video_path not in values
        ):
            values.insert(
                0,
                self.reference_video_path,
            )

        return [
            str(path)
            for path in values
            if str(path).strip()
        ]

    def normalized_audio_paths(self) -> list[str]:
        values = list(
            self.reference_audio_paths or []
        )

        if (
            self.reference_audio_path
            and self.reference_audio_path not in values
        ):
            values.insert(
                0,
                self.reference_audio_path,
            )

        return [
            str(path)
            for path in values
            if str(path).strip()
        ]

    def build_identity_profile(self) -> dict:
        appearance = self.appearance or {}

        self.identity_profile = {
            "name": self.name,
            "facial_features": appearance.get(
                "facial_features",
                "",
            ),
            "hair": appearance.get(
                "hair",
                "",
            ),
            "body_build": appearance.get(
                "body_build",
                "",
            ),
            "body_proportions": appearance.get(
                "body_proportions",
                "",
            ),
            "skin_tone": appearance.get(
                "skin_tone",
                "",
            ),
            "age_range": appearance.get(
                "age_range",
                "",
            ),
            "stable_identity_marks": appearance.get(
                "stable_identity_marks",
                [],
            ),
        }

        return self.identity_profile

    def build_story_state_profile(self) -> dict:
        self.story_state_profile = {
            "clothing": self.clothing,
            "distinctive_features": (
                self.distinctive_features
            ),
            "character_state": (
                self.character_state
            ),
        }

        return self.story_state_profile

    def identity_lock_text(self) -> str:
        identity = self.build_identity_profile()

        def stringify(value: Any) -> str:
            if isinstance(value, list):
                return ", ".join(
                    str(item)
                    for item in value
                )

            if isinstance(value, dict):
                return ", ".join(
                    f"{key}: {item}"
                    for key, item in value.items()
                )

            return str(
                value or ""
            ).strip()

        return " ".join(
            (
                f"IMMUTABLE CHARACTER IDENTITY: {self.name}.",
                "Preserve face geometry and facial proportions.",
                "Preserve hairstyle and hairline.",
                "Preserve body structure and body proportions.",
                "Preserve skin tone.",
                "Preserve stable identity-bearing features.",
                (
                    "Facial features: "
                    f"{stringify(identity['facial_features'])}."
                ),
                f"Hair: {stringify(identity['hair'])}.",
                f"Body: {stringify(identity['body_build'])}.",
                (
                    "Body proportions: "
                    f"{stringify(identity['body_proportions'])}."
                ),
                f"Skin tone: {stringify(identity['skin_tone'])}.",
                f"Age range: {stringify(identity['age_range'])}.",
                (
                    "Stable identity marks: "
                    f"{stringify(identity['stable_identity_marks'])}."
                ),
            )
        )

    def story_state_text(self) -> str:
        state = self.build_story_state_profile()

        return (
            f"CURRENT STORY STATE FOR {self.name}: "
            f"clothing={state['clothing']}; "
            f"distinctive_features={state['distinctive_features']}; "
            f"character_state={state['character_state']}."
        )

    def to_dict(self) -> dict:
        images = self.normalized_reference_paths()
        videos = self.normalized_video_paths()
        audios = self.normalized_audio_paths()

        self.build_identity_profile()
        self.build_story_state_profile()

        return {
            "character_id": self.character_id,
            "name": self.name,
            "role": self.role,
            "description": self.description,
            "personality": self.personality,
            "appearance": self.appearance,
            "clothing": self.clothing,
            "distinctive_features": self.distinctive_features,
            "character_state": self.character_state,
            "continuity_rules": self.continuity_rules,
            "reference_mode": self.reference_mode,
            "reference_paths": images,
            "reference_video_paths": videos,
            "reference_audio_paths": audios,
            "reference_path": (
                images[0] if images else self.reference_path
            ),
            "reference_video_path": (
                videos[0]
                if videos
                else self.reference_video_path
            ),
            "reference_audio_path": (
                audios[0]
                if audios
                else self.reference_audio_path
            ),
            "reference_mask_path": self.reference_mask_path,
            "identity_profile": self.identity_profile,
            "story_state_profile": self.story_state_profile,
            "identity_lock": self.identity_lock_text(),
            "story_state_lock": self.story_state_text(),
        }

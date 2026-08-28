from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, (list, tuple, set)):
        raise TypeError(
            f"{field_name} must be a list-like collection."
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


@dataclass
class Scene:
    scene_id: str
    order: int

    location: str = ""
    time_of_day: str = ""
    weather: str = ""
    atmosphere: str = ""
    description: str = ""
    mood: str = ""
    lighting: str = ""
    color_temperature: str = ""

    environment_details: list[str] = field(
        default_factory=list
    )
    key_props: list[str] = field(
        default_factory=list
    )

    scene_objective: str = ""

    characters: list[str] = field(
        default_factory=list
    )

    story_summary: str = ""
    continuity_notes: str = ""

    shot_ids: list[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.scene_id = str(
            self.scene_id or ""
        ).strip()

        if not self.scene_id:
            raise ValueError(
                "scene_id cannot be empty."
            )

        try:
            self.order = int(self.order)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Scene order must be an integer."
            ) from exc

        if self.order <= 0:
            raise ValueError(
                "Scene order must be greater than zero."
            )

        for field_name in (
            "location",
            "time_of_day",
            "weather",
            "atmosphere",
            "description",
            "mood",
            "lighting",
            "color_temperature",
            "scene_objective",
            "story_summary",
            "continuity_notes",
        ):
            setattr(
                self,
                field_name,
                str(
                    getattr(self, field_name) or ""
                ).strip(),
            )

        self.environment_details = _string_list(
            self.environment_details,
            "environment_details",
        )
        self.key_props = _string_list(
            self.key_props,
            "key_props",
        )
        self.characters = _string_list(
            self.characters,
            "characters",
        )
        self.shot_ids = _string_list(
            self.shot_ids,
            "shot_ids",
        )

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "order": self.order,
            "location": self.location,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "atmosphere": self.atmosphere,
            "description": self.description,
            "mood": self.mood,
            "lighting": self.lighting,
            "color_temperature": self.color_temperature,
            "environment_details": self.environment_details,
            "key_props": self.key_props,
            "scene_objective": self.scene_objective,
            "characters": self.characters,
            "story_summary": self.story_summary,
            "continuity_notes": self.continuity_notes,
            "shot_ids": self.shot_ids,
        }

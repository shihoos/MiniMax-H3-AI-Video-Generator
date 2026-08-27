from dataclasses import (
    dataclass,
    field,
)


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

    environment_details: list = field(
        default_factory=list
    )

    key_props: list = field(
        default_factory=list
    )

    scene_objective: str = ""

    characters: list = field(
        default_factory=list
    )

    story_summary: str = ""

    continuity_notes: str = ""

    shot_ids: list = field(
        default_factory=list
    )

    def to_dict(self):

        return {
            "scene_id":
                self.scene_id,

            "order":
                self.order,

            "location":
                self.location,

            "time_of_day":
                self.time_of_day,

            "weather":
                self.weather,

            "atmosphere":
                self.atmosphere,

            "description":
                self.description,

            "mood":
                self.mood,

            "lighting":
                self.lighting,

            "color_temperature":
                self.color_temperature,

            "environment_details":
                self.environment_details,

            "key_props":
                self.key_props,

            "scene_objective":
                self.scene_objective,

            "characters":
                self.characters,

            "story_summary":
                self.story_summary,

            "continuity_notes":
                self.continuity_notes,

            "shot_ids":
                self.shot_ids,
        }

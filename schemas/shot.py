from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Shot:

    shot_id: str
    scene_id: str
    order: int

    duration_seconds: float

    characters: list = field(
        default_factory=list
    )

    location: str = ""
    action: str = ""

    camera_shot: str = ""
    camera_movement: str = ""

    lighting: str = ""
    mood: str = ""

    visual_prompt: str = ""

    retention_analysis: str = ""
    detailed_description: str = ""

    overall_soundscape: str = ""
    non_diegetic_music: str = ""

    negative_prompt: str = ""

    continuity_notes: str = ""

    seed: Optional[int] = None

    reference_images: list = field(
        default_factory=list
    )

    reference_videos: list = field(
        default_factory=list
    )

    reference_audio: Optional[str] = None

    reference_audio_paths: list = field(
        default_factory=list
    )

    reference_audio_by_character: dict = field(
        default_factory=dict
    )

    reference_video_by_character: dict = field(
        default_factory=dict
    )

    speaking_characters: list = field(
        default_factory=list
    )

    speech_text: str = ""

    reference_bindings: list = field(
        default_factory=list
    )

    identity_locks: list = field(
        default_factory=list
    )

    # Production workflow controls.
    workflow_mode: str = "auto"

    keyframe_images: list = field(
        default_factory=list
    )

    keyframe_positions: list = field(
        default_factory=list
    )

    extend_take_source_video: Optional[str] = None

    width: int = 1344
    height: int = 768
    fps: int = 24
    frames_per_shot: int = 124
    steps: int = 20

    previous_shot: Optional[str] = None
    next_shot: Optional[str] = None

    def h3_prompt(self) -> str:

        subjects = "\n".join(
            self.identity_locks
        )

        if not subjects:
            subjects = (
                "No special immutable identity "
                "constraints were provided."
            )

        references = "\n".join(
            self.reference_bindings
        )

        if not references:
            references = (
                "No external visual references."
            )

        dialogue = (
            self.speech_text.strip()
            if self.speech_text
            else "N/A"
        )

        soundscape = (
            self.overall_soundscape.strip()
            if self.overall_soundscape
            else "Natural scene ambience."
        )

        music = (
            self.non_diegetic_music.strip()
            if self.non_diegetic_music
            else "N/A"
        )

        description = (
            self.detailed_description.strip()
            or self.visual_prompt.strip()
            or self.action.strip()
        )

        return (
            "subject_definitions:\n"
            f"{subjects}\n\n"

            "reference_bindings:\n"
            f"{references}\n\n"

            "summary:\n"
            f"{self.action.strip()}\n\n"

            "retention_analysis:\n"
            f"{self.retention_analysis.strip()}\n\n"

            "detailed_description:\n"
            f"{description}\n"
            f"Location: {self.location}\n"
            f"Camera: {self.camera_shot}; "
            f"{self.camera_movement}\n"
            f"Lighting: {self.lighting}\n"
            f"Mood: {self.mood}\n"
            f"Continuity: {self.continuity_notes}\n\n"

            "overall_soundscape:\n"
            f"{soundscape}\n"
            f"Dialogue: {dialogue}\n\n"

            "non_diegetic_music:\n"
            f"{music}"
        )

    def to_dict(self):

        return {
            "shot_id": self.shot_id,
            "scene_id": self.scene_id,
            "order": self.order,
            "duration_seconds": self.duration_seconds,

            "characters": self.characters,

            "location": self.location,
            "action": self.action,

            "camera_shot": self.camera_shot,
            "camera_movement": self.camera_movement,

            "lighting": self.lighting,
            "mood": self.mood,

            "visual_prompt": self.visual_prompt,

            "retention_analysis": (
                self.retention_analysis
            ),

            "detailed_description": (
                self.detailed_description
            ),

            "overall_soundscape": (
                self.overall_soundscape
            ),

            "non_diegetic_music": (
                self.non_diegetic_music
            ),

            "negative_prompt": (
                self.negative_prompt
            ),

            "continuity_notes": (
                self.continuity_notes
            ),

            "seed": self.seed,

            "reference_images": (
                self.reference_images
            ),

            "reference_videos": (
                self.reference_videos
            ),

            "reference_audio": (
                self.reference_audio
            ),

            "reference_audio_paths": (
                self.reference_audio_paths
            ),

            "reference_audio_by_character": (
                self.reference_audio_by_character
            ),

            "reference_video_by_character": (
                self.reference_video_by_character
            ),

            "speaking_characters": (
                self.speaking_characters
            ),

            "speech_text": self.speech_text,

            "reference_bindings": (
                self.reference_bindings
            ),

            "identity_locks": (
                self.identity_locks
            ),

            "workflow_mode": (
                self.workflow_mode
            ),

            "keyframe_images": (
                self.keyframe_images
            ),

            "keyframe_positions": (
                self.keyframe_positions
            ),

            "extend_take_source_video": (
                self.extend_take_source_video
            ),

            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames_per_shot": (
                self.frames_per_shot
            ),
            "steps": self.steps,

            "previous_shot": (
                self.previous_shot
            ),

            "next_shot": (
                self.next_shot
            ),

            "h3_prompt": self.h3_prompt(),
        }

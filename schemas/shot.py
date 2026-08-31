from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


def _continuity_state(value: Any, field_name: str) -> dict:
    """Normalize canonical or legacy continuity state into a dictionary."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"state_description": text}
        if isinstance(parsed, dict):
            return parsed
        return {"state_description": str(parsed)}
    raise TypeError(f"{field_name} must be a dictionary or legacy JSON/text string.")


VALID_WORKFLOW_MODES = {
    "auto",
    "ref2v",
    "turbo_ref2v",
    "upscale",
}


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


def _mapping(value: Any, field_name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            f"{field_name} must be a dictionary."
        )
    return dict(value)


@dataclass
class Shot:
    shot_id: str
    scene_id: str
    order: int
    duration_seconds: float

    characters: list[str] = field(default_factory=list)

    location: str = ""
    action: str = ""

    camera_shot: str = ""
    camera_movement: str = ""

    lens_and_depth_of_field: str = ""
    composition_notes: str = ""

    lighting: str = ""
    color_temperature: str = ""
    mood: str = ""

    visual_prompt: str = ""

    retention_analysis: str = ""
    detailed_description: str = ""

    overall_soundscape: str = ""
    non_diegetic_music: str = ""

    negative_prompt: str = ""
    continuity_notes: str = ""

    seed: Optional[int] = None

    reference_images: list[str] = field(default_factory=list)
    reference_videos: list[str] = field(default_factory=list)

    reference_audio: Optional[str] = None
    reference_audio_paths: list[str] = field(
        default_factory=list
    )

    reference_audio_by_character: dict = field(
        default_factory=dict
    )
    reference_video_by_character: dict = field(
        default_factory=dict
    )

    speaking_characters: list[str] = field(
        default_factory=list
    )
    speech_text: str = ""
    dialogue_events: list[dict] = field(default_factory=list)
    audio_duration_seconds: Optional[float] = None
    audio_duration_source: str = "unvalidated"

    continuity_start_state: dict = field(default_factory=dict)
    continuity_end_state: dict = field(default_factory=dict)
    continuity_repair_applied: bool = False
    identity_fingerprints: dict = field(default_factory=dict)
    is_scene_boundary: bool = False
    character_spatial_bboxes: dict = field(default_factory=dict)
    character_spatial_regions: dict = field(default_factory=dict)
    character_spatial_bboxes_start: dict = field(default_factory=dict)
    character_spatial_bboxes_end: dict = field(default_factory=dict)
    character_spatial_regions_start: dict = field(default_factory=dict)
    character_spatial_regions_end: dict = field(default_factory=dict)

    reference_roles: list[dict] = field(default_factory=list)
    storyboard_reference: Optional[str] = None
    reference_role_manifest: Optional[str] = None

    reference_bindings: list[str] = field(
        default_factory=list
    )
    identity_locks: list[str] = field(
        default_factory=list
    )

    workflow_mode: str = "auto"

    keyframe_images: list[str] = field(
        default_factory=list
    )
    keyframe_positions: list[Any] = field(
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

    def __post_init__(self) -> None:
        self.shot_id = str(
            self.shot_id or ""
        ).strip()

        self.scene_id = str(
            self.scene_id or ""
        ).strip()

        if not self.shot_id:
            raise ValueError(
                "shot_id cannot be empty."
            )

        if not self.scene_id:
            raise ValueError(
                "scene_id cannot be empty."
            )

        try:
            self.order = int(self.order)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Shot order must be an integer."
            ) from exc

        if self.order <= 0:
            raise ValueError(
                "Shot order must be greater than zero."
            )

        try:
            self.duration_seconds = float(
                self.duration_seconds
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "duration_seconds must be numeric."
            ) from exc

        if self.duration_seconds <= 0:
            raise ValueError(
                "duration_seconds must be greater than zero."
            )

        for field_name in (
            "location",
            "action",
            "camera_shot",
            "camera_movement",
            "lens_and_depth_of_field",
            "composition_notes",
            "lighting",
            "color_temperature",
            "mood",
            "visual_prompt",
            "retention_analysis",
            "detailed_description",
            "overall_soundscape",
            "non_diegetic_music",
            "negative_prompt",
            "continuity_notes",
            "speech_text",
        ):
            setattr(
                self,
                field_name,
                str(
                    getattr(self, field_name) or ""
                ).strip(),
            )

        self.characters = _string_list(
            self.characters,
            "characters",
        )
        self.reference_images = _string_list(
            self.reference_images,
            "reference_images",
        )
        self.reference_videos = _string_list(
            self.reference_videos,
            "reference_videos",
        )
        self.reference_audio_paths = _string_list(
            self.reference_audio_paths,
            "reference_audio_paths",
        )
        self.speaking_characters = _string_list(
            self.speaking_characters,
            "speaking_characters",
        )
        self.reference_bindings = _string_list(
            self.reference_bindings,
            "reference_bindings",
        )
        self.identity_locks = _string_list(
            self.identity_locks,
            "identity_locks",
        )
        self.keyframe_images = _string_list(
            self.keyframe_images,
            "keyframe_images",
        )

        if self.dialogue_events is None:
            self.dialogue_events = []
        if not isinstance(self.dialogue_events, list):
            raise TypeError("dialogue_events must be a list.")
        normalized_dialogue = []
        for event in self.dialogue_events:
            if not isinstance(event, dict):
                raise TypeError("dialogue_events entries must be dictionaries.")
            normalized_dialogue.append(dict(event))
        self.dialogue_events = normalized_dialogue

        self.continuity_start_state = _continuity_state(
            self.continuity_start_state,
            "continuity_start_state",
        )
        self.continuity_end_state = _continuity_state(
            self.continuity_end_state,
            "continuity_end_state",
        )
        self.identity_fingerprints = _mapping(
            self.identity_fingerprints,
            "identity_fingerprints",
        )

        if self.reference_roles is None:
            self.reference_roles = []
        if not isinstance(self.reference_roles, list):
            raise TypeError("reference_roles must be a list.")
        self.reference_roles = [
            dict(item) for item in self.reference_roles if isinstance(item, dict)
        ]

        def normalize_bboxes(value: Any, field_name: str) -> dict:
            source = _mapping(value, field_name)
            normalized = {}
            for name, bbox in source.items():
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError(
                        f"{field_name} values must be [x1, y1, x2, y2]."
                    )
                values = [float(item) for item in bbox]
                if any(item < 0.0 or item > 1.0 for item in values):
                    raise ValueError(
                        f"{field_name} coordinates must be normalized to [0, 1]."
                    )
                if values[2] < values[0] or values[3] < values[1]:
                    raise ValueError(
                        f"{field_name} must satisfy x2>=x1 and y2>=y1."
                    )
                normalized[str(name)] = values
            return normalized

        def normalize_regions(value: Any, field_name: str) -> dict:
            source = _mapping(value, field_name)
            return {
                str(key): str(item).strip()
                for key, item in source.items()
                if str(item).strip()
            }

        self.character_spatial_bboxes = normalize_bboxes(
            self.character_spatial_bboxes, "character_spatial_bboxes"
        )
        self.character_spatial_regions = normalize_regions(
            self.character_spatial_regions, "character_spatial_regions"
        )
        self.character_spatial_bboxes_start = normalize_bboxes(
            self.character_spatial_bboxes_start, "character_spatial_bboxes_start"
        )
        self.character_spatial_bboxes_end = normalize_bboxes(
            self.character_spatial_bboxes_end, "character_spatial_bboxes_end"
        )
        self.character_spatial_regions_start = normalize_regions(
            self.character_spatial_regions_start, "character_spatial_regions_start"
        )
        self.character_spatial_regions_end = normalize_regions(
            self.character_spatial_regions_end, "character_spatial_regions_end"
        )
        if not self.character_spatial_bboxes_end and self.character_spatial_bboxes:
            self.character_spatial_bboxes_end = dict(self.character_spatial_bboxes)
        if not self.character_spatial_regions_end and self.character_spatial_regions:
            self.character_spatial_regions_end = dict(self.character_spatial_regions)
        if not self.character_spatial_bboxes and self.character_spatial_bboxes_end:
            self.character_spatial_bboxes = dict(self.character_spatial_bboxes_end)
        if not self.character_spatial_regions and self.character_spatial_regions_end:
            self.character_spatial_regions = dict(self.character_spatial_regions_end)

        if self.audio_duration_seconds is not None:
            self.audio_duration_seconds = float(self.audio_duration_seconds)
            if self.audio_duration_seconds <= 0:
                raise ValueError("audio_duration_seconds must be greater than zero when supplied.")
        self.audio_duration_source = str(self.audio_duration_source or "unvalidated").strip()

        self.reference_audio_by_character = _mapping(
            self.reference_audio_by_character,
            "reference_audio_by_character",
        )
        self.reference_video_by_character = _mapping(
            self.reference_video_by_character,
            "reference_video_by_character",
        )

        self.workflow_mode = str(
            self.workflow_mode or "auto"
        ).strip()

        if self.workflow_mode not in VALID_WORKFLOW_MODES:
            raise ValueError(
                "workflow_mode must be one of: "
                + ", ".join(sorted(VALID_WORKFLOW_MODES))
            )

        if self.seed is not None:
            try:
                self.seed = int(self.seed)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "seed must be an integer or None."
                ) from exc

        for field_name in (
            "width",
            "height",
            "fps",
            "frames_per_shot",
            "steps",
        ):
            try:
                value = int(getattr(self, field_name))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{field_name} must be an integer."
                ) from exc

            if value <= 0:
                raise ValueError(
                    f"{field_name} must be greater than zero."
                )

            setattr(
                self,
                field_name,
                value,
            )

        for field_name in (
            "reference_audio",
            "extend_take_source_video",
            "previous_shot",
            "next_shot",
            "storyboard_reference",
            "reference_role_manifest",
        ):
            value = getattr(self, field_name)
            if value is not None:
                value = str(value).strip()
                setattr(
                    self,
                    field_name,
                    value or None,
                )

        if not isinstance(
            self.keyframe_positions,
            list,
        ):
            self.keyframe_positions = list(
                self.keyframe_positions or []
            )

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

        timeline_lines = []
        for event in self.dialogue_events:
            if not isinstance(event, dict):
                continue
            speaker_id = str(event.get("speaker_id", "")).strip()
            speaker_name = str(event.get("speaker_name", "")).strip()
            text = str(event.get("text", ""))
            start = float(event.get("start_seconds", 0.0) or 0.0)
            end = float(event.get("end_seconds", 0.0) or 0.0)
            timeline_lines.append(
                f"{speaker_name} [{speaker_id}] {start:.3f}-{end:.3f}s: {text}"
            )
        dialogue_timeline = (
            "\n".join(timeline_lines)
            if timeline_lines
            else "No dialogue events."
        )

        spatial_lines = []
        for name in self.characters:
            start_bbox = self.character_spatial_bboxes_start.get(name)
            end_bbox = self.character_spatial_bboxes_end.get(name) or self.character_spatial_bboxes.get(name)
            start_region = self.character_spatial_regions_start.get(name)
            end_region = self.character_spatial_regions_end.get(name) or self.character_spatial_regions.get(name)
            parts = []
            if start_bbox:
                parts.append(f"start_bbox={[round(v, 4) for v in start_bbox]}")
            if end_bbox:
                parts.append(f"end_bbox={[round(v, 4) for v in end_bbox]}")
            if start_region:
                parts.append(f"start_region={start_region}")
            if end_region:
                parts.append(f"end_region={end_region}")
            if parts:
                spatial_lines.append(f"{name}: " + ", ".join(parts))
        spatial_contract = "\n".join(spatial_lines) if spatial_lines else "No explicit normalized spatial constraints."
        boundary_note = (
            "This is a scene boundary; do not carry environment, lighting, props, or previous-shot image state across the cut."
            if self.is_scene_boundary
            else "Continue the locked continuity state from the previous shot unless an explicit story event changes it."
        )

        continuity_start = (
            json.dumps(self.continuity_start_state, ensure_ascii=False, sort_keys=True)
            if self.continuity_start_state else "{}"
        )
        continuity_end = (
            json.dumps(self.continuity_end_state, ensure_ascii=False, sort_keys=True)
            if self.continuity_end_state else "{}"
        )

        description = (
            self.detailed_description.strip()
            or self.visual_prompt.strip()
            or self.action.strip()
        )

        cinematography = (
            f"Shot: {self.camera_shot}; "
            f"Movement: {self.camera_movement}; "
            f"Lens/DOF: {self.lens_and_depth_of_field}; "
            f"Composition: {self.composition_notes}; "
            f"Lighting: {self.lighting}; "
            f"Color temperature: {self.color_temperature}; "
            f"Mood: {self.mood}"
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
            f"{cinematography}\n"
            f"Continuity: {self.continuity_notes}\n"
            f"Continuity boundary policy: {boundary_note}\n"
            "continuity_start_state:\n"
            f"{continuity_start}\n\n"
            "continuity_end_state:\n"
            f"{continuity_end}\n\n"
            "Spatial constraints:\n"
            f"{spatial_contract}\n\n"
            "overall_soundscape:\n"
            f"{soundscape}\n"
            f"Dialogue: {dialogue}\n\n"
            "dialogue_timeline:\n"
            f"{dialogue_timeline}\n\n"
            "non_diegetic_music:\n"
            f"{music}"
        )

    def to_dict(self) -> dict:
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
            "lens_and_depth_of_field": self.lens_and_depth_of_field,
            "composition_notes": self.composition_notes,
            "lighting": self.lighting,
            "color_temperature": self.color_temperature,
            "mood": self.mood,
            "visual_prompt": self.visual_prompt,
            "retention_analysis": self.retention_analysis,
            "detailed_description": self.detailed_description,
            "overall_soundscape": self.overall_soundscape,
            "non_diegetic_music": self.non_diegetic_music,
            "negative_prompt": self.negative_prompt,
            "continuity_notes": self.continuity_notes,
            "seed": self.seed,
            "reference_images": self.reference_images,
            "reference_videos": self.reference_videos,
            "reference_audio": self.reference_audio,
            "reference_audio_paths": self.reference_audio_paths,
            "reference_audio_by_character": self.reference_audio_by_character,
            "speaking_characters": self.speaking_characters,
            "speech_text": self.speech_text,
            "dialogue_events": self.dialogue_events,
            "audio_duration_seconds": self.audio_duration_seconds,
            "audio_duration_source": self.audio_duration_source,
            "continuity_start_state": self.continuity_start_state,
            "continuity_end_state": self.continuity_end_state,
            "continuity_repair_applied": self.continuity_repair_applied,
            "identity_fingerprints": self.identity_fingerprints,
            "is_scene_boundary": self.is_scene_boundary,
            "character_spatial_bboxes": self.character_spatial_bboxes,
            "character_spatial_regions": self.character_spatial_regions,
            "character_spatial_bboxes_start": self.character_spatial_bboxes_start,
            "character_spatial_bboxes_end": self.character_spatial_bboxes_end,
            "character_spatial_regions_start": self.character_spatial_regions_start,
            "character_spatial_regions_end": self.character_spatial_regions_end,
            "reference_roles": self.reference_roles,
            "storyboard_reference": self.storyboard_reference,
            "reference_role_manifest": self.reference_role_manifest,
            "reference_bindings": self.reference_bindings,
            "identity_locks": self.identity_locks,
            "workflow_mode": self.workflow_mode,
            "keyframe_images": self.keyframe_images,
            "keyframe_positions": self.keyframe_positions,
            "extend_take_source_video": self.extend_take_source_video,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames_per_shot": self.frames_per_shot,
            "steps": self.steps,
            "previous_shot": self.previous_shot,
            "next_shot": self.next_shot,
            "h3_prompt": self.h3_prompt(),
        }

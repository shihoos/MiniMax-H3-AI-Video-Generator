from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.identity_continuity import (
    IdentityContinuity,
)

from pipeline.reference_manager import (
    ReferenceManager,
)

from planner.config import (
    H3_FPS,
    H3_FRAMES_PER_SHOT,
    H3_HEIGHT,
    H3_STEPS,
    H3_WIDTH,
    PRESERVE_USER_STORY_MODE,
    EXPAND_USER_STORY_MODE,
    AI_STORY_MODE,
    WORKFLOW_AUTO,
    WORKFLOW_REF2V,
    WORKFLOW_TURBO_REF2V,
)

from schemas.character import (
    Character,
)

from schemas.scene import (
    Scene,
)

from schemas.shot import (
    Shot,
)


@dataclass
class ParsedSegment:
    text: str
    order: int


class ProductionPlanner:
    """
    Local, dependency-free production planner.

    IMPORTANT:
    - No external API.
    - No second language model.
    - No additional Qwen model.
    - The locked Qwen3-VL checkpoint remains the MiniMax H3
      conditioning encoder inside ComfyUI.
    - This class converts the user's story and local reference
      assets into Character / Scene / Shot objects consumed by
      the existing production pipeline.
    """

    MAX_CHARACTER_REFERENCES = 9
    MAX_VIDEO_REFERENCES = 3
    MAX_AUDIO_REFERENCES = 3

    DEFAULT_SHOT_DURATION = 5.0

    def __init__(
        self,
        project_root: Path | str,
    ):
        self.project_root = Path(
            project_root
        )

        self.references = (
            ReferenceManager(
                self.project_root
            )
        )

    # ============================================================
    # STORY NORMALIZATION
    # ============================================================

    @staticmethod
    def _clean_text(
        text: str,
    ) -> str:

        text = str(
            text or ""
        )

        text = text.replace(
            "\r\n",
            "\n",
        )

        text = text.replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()

    def normalize_story(
        self,
        mode: str,
        user_input: str,
    ) -> str:

        story = self._clean_text(
            user_input
        )

        if not story:
            raise ValueError(
                "Story input cannot be empty."
            )

        if mode not in {
            AI_STORY_MODE,
            PRESERVE_USER_STORY_MODE,
            EXPAND_USER_STORY_MODE,
        }:
            raise ValueError(
                f"Unsupported story mode: {mode}"
            )

        # We deliberately do not call another LLM.
        #
        # preserve_user_story:
        #   exact user story after whitespace normalization.
        #
        # ai_story:
        #   user input is treated as the production story seed.
        #
        # expand_user_story:
        #   keep the supplied story as the canonical source.
        #
        # This prevents the planner from silently introducing
        # another model dependency.

        return story

    # ============================================================
    # LOCAL REFERENCE DISCOVERY
    # ============================================================

    @staticmethod
    def _normalize_name(
        value: str,
    ) -> str:

        return re.sub(
            r"[^a-z0-9]+",
            " ",
            str(value).lower(),
        ).strip()

    def _known_asset_names(
        self,
    ) -> list[str]:

        return self.references.character_asset_names()

    def _explicit_asset_names(
        self,
        text: str,
    ) -> list[str]:

        text = str(
            text or ""
        )

        result = []

        for asset_name in (
            self._known_asset_names()
        ):

            if not asset_name.strip():
                continue

            pattern = (
                r"(?<!\w)"
                + re.escape(asset_name)
                + r"(?!\w)"
            )

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):
                result.append(
                    asset_name
                )

        return result

    # ============================================================
    # CHARACTER EXTRACTION
    # ============================================================

    @staticmethod
    def _looks_like_name(
        value: str,
    ) -> bool:

        value = str(
            value or ""
        ).strip()

        if not value:
            return False

        words = value.split()

        if not 1 <= len(words) <= 4:
            return False

        return all(
            bool(
                re.match(
                    r"^[A-Z][A-Za-z0-9'_-]*$",
                    word,
                )
            )
            for word in words
        )

    def detect_character_names(
        self,
        story: str,
    ) -> list[str]:

        result = []
        seen = set()

        # 1. Prefer explicit local asset names.
        for name in self._explicit_asset_names(
            story
        ):
            key = self._normalize_name(
                name
            )

            if key and key not in seen:
                seen.add(key)
                result.append(name)

        # 2. Extract likely proper names.
        #
        # This is intentionally conservative. We do not
        # manufacture names.
        for match in re.findall(
            r"\b[A-Z][A-Za-z0-9'_-]*(?:\s+[A-Z][A-Za-z0-9'_-]*){0,2}\b",
            story,
        ):

            candidate = (
                str(match)
                .strip()
            )

            # Common sentence-start words are not character
            # names when they are only one word.
            if candidate in {
                "The",
                "A",
                "An",
                "Then",
                "When",
                "While",
                "After",
                "Before",
                "Suddenly",
                "Meanwhile",
                "Finally",
                "But",
                "And",
                "In",
                "On",
                "At",
            }:
                continue

            if not self._looks_like_name(
                candidate
            ):
                continue

            key = self._normalize_name(
                candidate
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(
                candidate
            )

        return result

    def create_characters(
        self,
        story: str,
    ) -> list[Character]:

        names = self.detect_character_names(
            story
        )

        characters = []

        for index, name in enumerate(
            names,
            start=1,
        ):

            source = (
                self.references
                .get_character_source(
                    name
                )
            )

            character = Character(
                character_id=(
                    f"character_{index:03d}"
                ),
                name=name,
                role="story character",
                description=(
                    f"{name} appears in the supplied story."
                ),
                personality="",
                appearance={},
                clothing={},
                distinctive_features=[],
                character_state={},
                continuity_rules=[],
                reference_mode=source.get(
                    "mode",
                    "missing",
                ),
                reference_paths=list(
                    source.get(
                        "reference_paths",
                        [],
                    )
                ),
                reference_video_paths=list(
                    source.get(
                        "reference_video_paths",
                        [],
                    )
                ),
                reference_audio_paths=list(
                    source.get(
                        "reference_audio_paths",
                        [],
                    )
                ),
                reference_path=source.get(
                    "path"
                ),
                reference_video_path=source.get(
                    "reference_video_path"
                ),
                reference_audio_path=source.get(
                    "reference_audio_path"
                ),
            )

            character.build_identity_profile()
            character.build_story_state_profile()

            characters.append(
                character
            )

        self.references.resolve_characters(
            characters
        )

        return characters

    # ============================================================
    # SCENE SEGMENTATION
    # ============================================================

    def _split_segments(
        self,
        story: str,
    ) -> list[ParsedSegment]:

        text = self._clean_text(
            story
        )

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split(
                "\n\n"
            )
            if paragraph.strip()
        ]

        segments = []

        if paragraphs:
            source_segments = paragraphs
        else:
            source_segments = re.split(
                r"(?<=[.!?])\s+",
                text,
            )

        for index, segment in enumerate(
            source_segments,
            start=1,
        ):

            cleaned = self._clean_text(
                segment
            )

            if not cleaned:
                continue

            segments.append(
                ParsedSegment(
                    text=cleaned,
                    order=index,
                )
            )

        if not segments:
            segments.append(
                ParsedSegment(
                    text=text,
                    order=1,
                )
            )

        return segments

    @staticmethod
    def _extract_location(
        text: str,
    ) -> str:

        patterns = [
            r"\bin (?:the )?([A-Za-z][A-Za-z0-9' -]{2,60})",
            r"\bat (?:the )?([A-Za-z][A-Za-z0-9' -]{2,60})",
            r"\binside (?:the )?([A-Za-z][A-Za-z0-9' -]{2,60})",
            r"\bon (?:the )?([A-Za-z][A-Za-z0-9' -]{2,60})",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                value = (
                    match.group(1)
                    .strip(" ,.;:")
                )

                if value:
                    return value

        return "unspecified location"

    @staticmethod
    def _extract_time_of_day(
        text: str,
    ) -> str:

        lower = text.lower()

        if any(
            token in lower
            for token in [
                "sunrise",
                "dawn",
                "morning",
            ]
        ):
            return "morning"

        if any(
            token in lower
            for token in [
                "noon",
                "midday",
                "afternoon",
            ]
        ):
            return "afternoon"

        if any(
            token in lower
            for token in [
                "sunset",
                "evening",
                "dusk",
            ]
        ):
            return "evening"

        if any(
            token in lower
            for token in [
                "night",
                "midnight",
            ]
        ):
            return "night"

        return "unspecified"

    @staticmethod
    def _extract_mood(
        text: str,
    ) -> str:

        lower = text.lower()

        mood_words = [
            "tense",
            "dangerous",
            "joyful",
            "sad",
            "melancholic",
            "romantic",
            "hopeful",
            "mysterious",
            "dark",
            "peaceful",
            "urgent",
            "violent",
            "dramatic",
            "calm",
            "fearful",
            "exciting",
        ]

        found = [
            word
            for word in mood_words
            if word in lower
        ]

        return (
            ", ".join(found)
            if found
            else "cinematic"
        )

    def _scene_characters(
        self,
        segment_text: str,
        characters: list[Character],
    ) -> list[str]:

        selected = []

        lower_text = (
            segment_text.lower()
        )

        for character in characters:

            if (
                character.name.lower()
                in lower_text
            ):
                selected.append(
                    character.name
                )

        return selected

    def create_scenes(
        self,
        story: str,
        characters: list[Character],
    ) -> list[Scene]:

        segments = self._split_segments(
            story
        )

        scenes = []

        for index, segment in enumerate(
            segments,
            start=1,
        ):

            selected_characters = (
                self._scene_characters(
                    segment.text,
                    characters,
                )
            )

            scenes.append(
                Scene(
                    scene_id=(
                        f"scene_{index:03d}"
                    ),
                    order=index,
                    location=self._extract_location(
                        segment.text
                    ),
                    time_of_day=self._extract_time_of_day(
                        segment.text
                    ),
                    weather="",
                    atmosphere=(
                        self._extract_mood(
                            segment.text
                        )
                    ),
                    description=segment.text,
                    mood=self._extract_mood(
                        segment.text
                    ),
                    lighting="cinematic",
                    environment_details=[],
                    key_props=[],
                    scene_objective=segment.text,
                    characters=selected_characters,
                    story_summary=segment.text,
                    continuity_notes="",
                    shot_ids=[],
                )
            )

        return scenes

    # ============================================================
    # SHOTS
    # ============================================================

    @staticmethod
    def _shot_action(
        text: str,
    ) -> str:

        return text.strip()

    @staticmethod
    def _camera_for_scene(
        order: int,
    ) -> tuple[str, str]:

        choices = [
            (
                "wide cinematic shot",
                "slow controlled push-in",
            ),
            (
                "medium cinematic shot",
                "subtle lateral movement",
            ),
            (
                "close cinematic shot",
                "slow forward movement",
            ),
        ]

        return choices[
            (order - 1)
            % len(choices)
        ]

    def _character_references(
        self,
        characters: list[Character],
        selected_names: list[str],
    ) -> tuple[
        list[str],
        list[str],
        list[str],
        dict[str, list[str]],
    ]:

        by_name = {
            character.name.lower(): character
            for character in characters
        }

        image_paths = []
        video_paths = []
        audio_paths = []

        image_bindings: dict[
            str,
            list[str],
        ] = {}

        for name in selected_names:

            character = by_name.get(
                name.lower()
            )

            if character is None:
                continue

            images = (
                character.normalized_reference_paths()
            )

            videos = (
                character.normalized_video_paths()
            )

            audios = (
                character.normalized_audio_paths()
            )

            image_bindings[
                character.name
            ] = images

            for path in images:

                if (
                    path not in image_paths
                    and len(image_paths)
                    < self.MAX_CHARACTER_REFERENCES
                ):
                    image_paths.append(
                        path
                    )

            for path in videos:

                if (
                    path not in video_paths
                    and len(video_paths)
                    < self.MAX_VIDEO_REFERENCES
                ):
                    video_paths.append(
                        path
                    )

            for path in audios:

                if (
                    path not in audio_paths
                    and len(audio_paths)
                    < self.MAX_AUDIO_REFERENCES
                ):
                    audio_paths.append(
                        path
                    )

        return (
            image_paths,
            video_paths,
            audio_paths,
            image_bindings,
        )

    def create_shots(
        self,
        story: str,
        characters: list[Character],
        scenes: list[Scene],
        workflow_mode: str = WORKFLOW_AUTO,
        profile: str = "base",
    ) -> list[Shot]:

        if not scenes:
            raise RuntimeError(
                "Cannot create shots without scenes."
            )

        shots = []

        by_character = {
            character.name.lower(): character
            for character in characters
        }

        for scene in scenes:

            selected_names = list(
                scene.characters
            )

            # If the scene parser didn't explicitly find
            # a character but only one named character exists,
            # use that character conservatively.
            if (
                not selected_names
                and len(characters) == 1
            ):
                selected_names = [
                    characters[0].name
                ]

            (
                image_paths,
                video_paths,
                audio_paths,
                image_bindings,
            ) = self._character_references(
                characters,
                selected_names,
            )

            locks = (
                IdentityContinuity.build_locks(
                    characters,
                    selected_names,
                )
            )

            bindings = (
                IdentityContinuity
                .build_reference_bindings(
                    image_paths,
                    image_bindings,
                )
            )

            camera_shot, camera_movement = (
                self._camera_for_scene(
                    scene.order
                )
            )

            description = (
                f"{scene.description} "
                f"Location: {scene.location}. "
                f"Time: {scene.time_of_day}. "
                f"Mood: {scene.mood}. "
                f"Lighting: {scene.lighting}."
            )

            visual_prompt = (
                description
                + " "
                + self._shot_action(
                    scene.scene_objective
                )
            )

            negative_prompt = (
                "identity drift, altered face geometry, "
                "different hairstyle, altered body proportions, "
                "facial deformation, extra limbs, duplicate person"
            )

            (
                visual_prompt,
                negative_prompt,
            ) = IdentityContinuity.merge(
                visual_prompt=visual_prompt,
                locks=locks,
                bindings=bindings,
                negative_prompt=negative_prompt,
            )

            actual_workflow = (
                WORKFLOW_TURBO_REF2V
                if profile == "turbo"
                else WORKFLOW_REF2V
            )

            if (
                workflow_mode
                not in {
                    WORKFLOW_AUTO,
                    WORKFLOW_REF2V,
                    WORKFLOW_TURBO_REF2V,
                }
            ):
                actual_workflow = (
                    WORKFLOW_REF2V
                )

            shot = Shot(
                shot_id=(
                    f"shot_{len(shots) + 1:03d}"
                ),
                scene_id=scene.scene_id,
                order=len(shots) + 1,
                duration_seconds=(
                    self.DEFAULT_SHOT_DURATION
                ),
                characters=selected_names,
                location=scene.location,
                action=scene.scene_objective,
                camera_shot=camera_shot,
                camera_movement=camera_movement,
                lighting=scene.lighting,
                mood=scene.mood,
                visual_prompt=visual_prompt,
                retention_analysis="",
                detailed_description=description,
                overall_soundscape=(
                    scene.atmosphere
                    or "natural cinematic ambience"
                ),
                non_diegetic_music="",
                negative_prompt=negative_prompt,
                continuity_notes=(
                    scene.continuity_notes
                ),
                seed=(
                    100000
                    + len(shots)
                ),
                reference_images=image_paths,
                reference_videos=video_paths,
                reference_audio=(
                    audio_paths[0]
                    if audio_paths
                    else None
                ),
                reference_audio_paths=audio_paths,
                reference_audio_by_character={
                    name: (
                        by_character[name.lower()]
                        .normalized_audio_paths()
                    )
                    for name in selected_names
                    if name.lower()
                    in by_character
                },
                reference_video_by_character={
                    name: (
                        by_character[name.lower()]
                        .normalized_video_paths()
                    )
                    for name in selected_names
                    if name.lower()
                    in by_character
                },
                speaking_characters=[],
                speech_text="",
                reference_bindings=bindings,
                identity_locks=locks,
                workflow_mode=actual_workflow,
                keyframe_images=[],
                keyframe_positions=[],
                extend_take_source_video=None,
                width=H3_WIDTH,
                height=H3_HEIGHT,
                fps=H3_FPS,
                frames_per_shot=H3_FRAMES_PER_SHOT,
                steps=H3_STEPS,
            )

            shots.append(
                shot
            )

        return shots

    # ============================================================
    # COMPLETE PLAN
    # ============================================================

    def build(
        self,
        *,
        mode: str,
        user_input: str,
        workflow_mode: str = WORKFLOW_AUTO,
        profile: str = "base",
    ) -> dict:

        story = self.normalize_story(
            mode,
            user_input,
        )

        characters = self.create_characters(
            story
        )

        # Character identity references are mandatory for
        # character-driven H3 production.
        if characters:

            self.references.validate(
                characters,
                require_images=True,
            )

        scenes = self.create_scenes(
            story,
            characters,
        )

        shots = self.create_shots(
            story,
            characters,
            scenes,
            workflow_mode=workflow_mode,
            profile=profile,
        )

        # Scene -> shot relationship.
        by_scene = {}

        for shot in shots:
            by_scene.setdefault(
                shot.scene_id,
                []
            ).append(
                shot.shot_id
            )

        for scene in scenes:
            scene.shot_ids = list(
                by_scene.get(
                    scene.scene_id,
                    [],
                )
            )

        # Validate reference limits before execution.
        for shot in shots:

            image_count = len(
                shot.reference_images
            )

            video_count = len(
                shot.reference_videos
            )

            audio_count = len(
                shot.reference_audio_paths
            )

            total = (
                image_count
                + video_count
                + audio_count
            )

            if image_count > 9:
                raise RuntimeError(
                    f"{shot.shot_id}: more than 9 "
                    "image references."
                )

            if video_count > 3:
                raise RuntimeError(
                    f"{shot.shot_id}: more than 3 "
                    "video references."
                )

            if audio_count > 3:
                raise RuntimeError(
                    f"{shot.shot_id}: more than 3 "
                    "audio references."
                )

            if total > 12:
                raise RuntimeError(
                    f"{shot.shot_id}: more than 12 "
                    "total reference files."
                )

            if shot.characters:

                if not shot.reference_images:
                    raise RuntimeError(
                        f"{shot.shot_id}: character shot "
                        "has no reference image."
                    )

                if not shot.identity_locks:
                    raise RuntimeError(
                        f"{shot.shot_id}: missing identity locks."
                    )

                if not shot.reference_bindings:
                    raise RuntimeError(
                        f"{shot.shot_id}: missing "
                        "reference bindings."
                    )

        return {
            "story": story,
            "character_names": [
                character.name
                for character in characters
            ],
            "characters": [
                character.to_dict()
                for character in characters
            ],
            "scenes": [
                scene.to_dict()
                for scene in scenes
            ],
            "shots": [
                shot.to_dict()
                for shot in shots
            ],
            "profile": profile,
            "workflow_mode": workflow_mode,
            "width": H3_WIDTH,
            "height": H3_HEIGHT,
            "fps": H3_FPS,
            "frames_per_shot": (
                H3_FRAMES_PER_SHOT
            ),
            "steps": H3_STEPS,
        }

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
    AI_STORY_MODE,
    EXPAND_USER_STORY_MODE,
    H3_FPS,
    H3_FRAMES_PER_SHOT,
    H3_HEIGHT,
    H3_MAX_REFERENCE_AUDIO,
    H3_MAX_REFERENCE_FILES,
    H3_MAX_REFERENCE_IMAGES,
    H3_MAX_REFERENCE_VIDEOS,
    H3_STEPS,
    H3_WIDTH,
    PRESERVE_USER_STORY_MODE,
    TURBO_STEPS,
    VALID_STORY_MODES,
    WORKFLOW_AUTO,
    WORKFLOW_REF2V,
    WORKFLOW_TURBO_REF2V,
)
from schemas.character import Character
from schemas.scene import Scene
from schemas.shot import Shot


@dataclass
class StoryUnit:
    order: int
    text: str


class ProductionPlanner:
    """
    Dependency-free production planner.

    Important architectural rule:
    the repository uses one locked Qwen3-VL model inside the
    MiniMax H3 ComfyUI workflow. This planner does not call an
    external LLM, API, OpenAI service, second Qwen model, or
    remote planner.

    Character creation therefore means:
      story -> deterministic canonical character profile
      -> identity locks -> H3 prompt

    When reference media exists, it is attached.
    When it does not exist, the generated character profile
    becomes the canonical identity source for the production.
    """

    ROLE_PATTERNS = (
        (
            r"\b(?:a|an|the)\s+(young\s+)?woman\b",
            "woman",
        ),
        (
            r"\b(?:a|an|the)\s+(young\s+)?man\b",
            "man",
        ),
        (
            r"\b(?:a|an|the)\s+girl\b",
            "girl",
        ),
        (
            r"\b(?:a|an|the)\s+boy\b",
            "boy",
        ),
        (
            r"\b(?:a|an|the)\s+child\b",
            "child",
        ),
        (
            r"\b(?:a|an|the)\s+person\b",
            "person",
        ),
        (
            r"\b(?:a|an|the)\s+hero\b",
            "hero",
        ),
        (
            r"\b(?:a|an|the)\s+heroine\b",
            "heroine",
        ),
        (
            r"\b(?:a|an|the)\s+explorer\b",
            "explorer",
        ),
        (
            r"\b(?:a|an|the)\s+detective\b",
            "detective",
        ),
        (
            r"\b(?:a|an|the)\s+scientist\b",
            "scientist",
        ),
        (
            r"\b(?:a|an|the)\s+soldier\b",
            "soldier",
        ),
        (
            r"\b(?:a|an|the)\s+warrior\b",
            "warrior",
        ),
        (
            r"\b(?:a|an|the)\s+king\b",
            "king",
        ),
        (
            r"\b(?:a|an|the)\s+queen\b",
            "queen",
        ),
        (
            r"\b(?:a|an|the)\s+child\b",
            "child",
        ),
        (
            r"\b(?:a|an|the)\s+robot\b",
            "robot",
        ),
        (
            r"\b(?:a|an|the)\s+android\b",
            "android",
        ),
        (
            r"\b(?:a|an|the)\s+pilot\b",
            "pilot",
        ),
    )

    COMMON_PROPER_WORDS = {
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
        "As",
        "During",
    }

    ACTION_WORDS = (
        "walk",
        "runs",
        "run",
        "moves",
        "move",
        "looks",
        "look",
        "turns",
        "turn",
        "enters",
        "enter",
        "leaves",
        "leave",
        "fights",
        "fight",
        "talks",
        "talk",
        "speaks",
        "speak",
        "stands",
        "stand",
        "sits",
        "sit",
        "drives",
        "drive",
        "flies",
        "fly",
        "jumps",
        "jump",
        "opens",
        "open",
        "closes",
        "close",
        "reaches",
        "reach",
        "holds",
        "hold",
        "runs",
    )

    LOCATION_PATTERNS = (
        r"\bin\s+(?:the\s+)?([^,.!?]+)",
        r"\bat\s+(?:the\s+)?([^,.!?]+)",
        r"\bon\s+(?:the\s+)?([^,.!?]+)",
        r"\binside\s+(?:the\s+)?([^,.!?]+)",
        r"\bnear\s+(?:the\s+)?([^,.!?]+)",
        r"\bthrough\s+(?:the\s+)?([^,.!?]+)",
    )

    TIME_WORDS = {
        "sunrise": "sunrise",
        "dawn": "dawn",
        "morning": "morning",
        "noon": "midday",
        "midday": "midday",
        "afternoon": "afternoon",
        "sunset": "sunset",
        "evening": "evening",
        "dusk": "dusk",
        "night": "night",
        "midnight": "midnight",
    }

    MOOD_WORDS = (
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
        "warm",
        "lonely",
        "eerie",
        "epic",
    )

    WEATHER_WORDS = (
        "rain",
        "rainy",
        "storm",
        "stormy",
        "fog",
        "foggy",
        "snow",
        "snowy",
        "wind",
        "windy",
        "clear",
        "cloudy",
    )

    def __init__(
        self,
        project_root: Path | str,
    ):
        self.project_root = Path(
            project_root
        )

        self.references = ReferenceManager(
            self.project_root
        )

    # ============================================================
    # TEXT NORMALIZATION
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
        ).replace(
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

    def _split_story(
        self,
        story: str,
    ) -> list[StoryUnit]:

        paragraphs = [
            self._clean_text(
                paragraph
            )
            for paragraph in story.split(
                "\n\n"
            )
            if self._clean_text(
                paragraph
            )
        ]

        if paragraphs:
            parts = paragraphs
        else:
            parts = [
                self._clean_text(
                    item
                )
                for item in re.split(
                    r"(?<=[.!?])\s+",
                    story,
                )
                if self._clean_text(
                    item
                )
            ]

        if not parts:
            return [
                StoryUnit(
                    order=1,
                    text=story,
                )
            ]

        return [
            StoryUnit(
                order=index,
                text=value,
            )
            for index, value in enumerate(
                parts,
                start=1,
            )
        ]

    # ============================================================
    # STORY MODES
    # ============================================================

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
                "Story cannot be empty."
            )
    
        if mode not in VALID_STORY_MODES:
            raise ValueError(
                f"Unsupported story mode: {mode}"
            )
    
        # IMPORTANT:
        # The story itself is never rewritten here.
        #
        # Qwen is the story/director model and is responsible for:
        #   - developing AI stories
        #   - expanding supplied stories
        #   - cinematic interpretation
        #
        # The deterministic planner must never inject words such
        # as "Treat", "Develop", "Clarify", etc. into the story.
        return story

    # ============================================================
    # CHARACTER DISCOVERY
    # ============================================================

    @staticmethod
    def _proper_names(
        story: str,
    ) -> list[str]:

        values = re.findall(
            r"\b[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2}\b",
            story,
        )

        result = []
        seen = set()

        for value in values:
            value = value.strip()

            if value in ProductionPlanner.COMMON_PROPER_WORDS:
                continue

            words = value.split()

            if not words:
                continue

            if len(words) == 1 and value in {
                "City",
                "Street",
                "Road",
                "House",
                "Tower",
                "Castle",
                "Forest",
                "Mountain",
                "River",
                "Ocean",
            }:
                continue

            key = value.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(
                value
            )

        return result
    def detect_character_descriptors(
        self,
        story: str,
    ) -> list[str]:
    
        candidates = []
    
        # Deterministic fallback detection is intentionally limited
        # to concrete role descriptors. Qwen is responsible for
        # creative character creation and naming.
        for pattern, label in self.ROLE_PATTERNS:
    
            count = len(
                re.findall(
                    pattern,
                    story,
                    flags=re.IGNORECASE,
                )
            )
    
            if count:
                candidates.append(
                    label
                )
    
        # Explicitly named characters can still be detected when
        # the user actually says "named X" or "called X".
        explicit_names = re.findall(
            r"\b(?:named|called)\s+"
            r"([A-Z][A-Za-z0-9'_-]+"
            r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})\b",
            story,
        )
    
        for name in explicit_names:
    
            name = name.strip()
    
            if not name:
                continue
    
            if name in (
                self.COMMON_PROPER_WORDS
            ):
                continue
    
            candidates.append(
                name
            )
    
        result = []
        seen = set()
    
        for value in candidates:
    
            key = value.lower()
    
            if key in seen:
                continue
    
            seen.add(
                key
            )
    
            result.append(
                value
            )
    
        return result

    
    @staticmethod
    def _appearance_from_story(
        name: str,
        story: str,
    ) -> dict:

        lower = story.lower()

        hair = "stable hairstyle inferred from the story"
        body_build = "natural consistent body structure"
        skin_tone = "preserve consistent skin tone"
        age_range = "consistent age appropriate to the story"

        if "long hair" in lower:
            hair = "long hair"
        elif "short hair" in lower:
            hair = "short hair"
        elif "curly hair" in lower:
            hair = "curly hair"
        elif "straight hair" in lower:
            hair = "straight hair"

        if "beard" in lower:
            stable_marks = [
                "stable beard/facial-hair appearance"
            ]
        else:
            stable_marks = []

        clothing = {}

        for token in (
            "black",
            "white",
            "red",
            "blue",
            "green",
            "leather",
            "armor",
            "jacket",
            "coat",
            "dress",
            "shirt",
            "uniform",
        ):
            if token in lower:
                clothing[token] = (
                    f"{token} clothing detail"
                )

        return {
            "facial_features": (
                f"stable facial identity for {name}"
            ),
            "hair": hair,
            "body_build": body_build,
            "body_proportions": (
                "consistent proportions across every shot"
            ),
            "skin_tone": skin_tone,
            "age_range": age_range,
            "stable_identity_marks": stable_marks,
            "story-derived": True,
            "clothing": clothing,
        }

    def _make_character(
        self,
        name: str,
        index: int,
        story: str,
    ) -> Character:

        appearance = (
            self._appearance_from_story(
                name,
                story,
            )
        )

        role = (
            name
            if name
            and name.lower()
            not in {
                "man",
                "woman",
                "girl",
                "boy",
                "child",
                "person",
            }
            else "story character"
        )

        character = Character(
            character_id=(
                f"character_{index:03d}"
            ),
            name=name,
            role=role,
            description=(
                f"Canonical character identity derived "
                f"from the supplied story for {name}."
            ),
            personality=(
                "Preserve personality and behavior "
                "consistently across the story."
            ),
            appearance=appearance,
            clothing=appearance[
                "clothing"
            ],
            distinctive_features=(
                appearance[
                    "stable_identity_marks"
                ]
            ),
            character_state={
                "origin": (
                    "story-derived canonical profile"
                ),
                "continuity": (
                    "preserve identity and state "
                    "across all relevant shots"
                ),
            },
            continuity_rules=[
                "preserve face geometry",
                "preserve hair and hairline",
                "preserve body proportions",
                "preserve stable identity features",
                "preserve story-state continuity",
            ],
        )

        character.build_identity_profile()
        character.build_story_state_profile()

        source = (
            self.references.get_character_source(
                name
            )
        )

        character.reference_mode = source[
            "mode"
        ]

        character.reference_paths = source[
            "reference_paths"
        ]

        character.reference_video_paths = source[
            "reference_video_paths"
        ]

        character.reference_audio_paths = source[
            "reference_audio_paths"
        ]

        character.reference_path = source[
            "path"
        ]

        character.reference_video_path = source[
            "reference_video_path"
        ]

        character.reference_audio_path = source[
            "reference_audio_path"
        ]

        # No supplied reference is not an error.
        # The story-derived identity profile becomes the
        # canonical identity source.
        if not character.reference_paths:
            character.reference_mode = (
                "story_generated"
            )

        return character

    def create_characters(
        self,
        story: str,
    ) -> list[Character]:

        descriptors = (
            self.detect_character_descriptors(
                story
            )
        )

        if not descriptors:
            # A story with no explicit human/character descriptor
            # is allowed to remain character-free.
            return []

        characters = []

        for index, descriptor in enumerate(
            descriptors,
            start=1,
        ):

            characters.append(
                self._make_character(
                    descriptor,
                    index,
                    story,
                )
            )

        self.references.resolve_characters(
            characters
        )

        # References are optional now.
        self.references.validate(
            characters,
            require_images=False,
        )

        return characters

    # ============================================================
    # SCENE EXTRACTION
    # ============================================================

    @staticmethod
    def _location(
        text: str,
    ) -> str:

        for pattern in ProductionPlanner.LOCATION_PATTERNS:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                value = (
                    match.group(1)
                    .strip(
                        " ,.;:"
                    )
                )

                if value:
                    return value

        return "cinematic environment"

    @staticmethod
    def _time_of_day(
        text: str,
    ) -> str:

        lower = text.lower()

        for key, value in ProductionPlanner.TIME_WORDS.items():
            if key in lower:
                return value

        return "unspecified time"

    @staticmethod
    def _mood(
        text: str,
    ) -> str:

        lower = text.lower()

        values = [
            word
            for word in ProductionPlanner.MOOD_WORDS
            if word in lower
        ]

        return (
            ", ".join(values)
            if values
            else "cinematic"
        )

    @staticmethod
    def _weather(
        text: str,
    ) -> str:

        lower = text.lower()

        values = [
            word
            for word in ProductionPlanner.WEATHER_WORDS
            if word in lower
        ]

        return (
            values[0]
            if values
            else "natural"
        )

    @staticmethod
    def _lighting(
        text: str,
    ) -> str:

        lower = text.lower()

        if "sunset" in lower:
            return "warm directional sunset light"

        if "sunrise" in lower:
            return "soft sunrise light"

        if "night" in lower:
            return "cinematic night illumination"

        if "neon" in lower:
            return "neon cinematic lighting"

        if "dark" in lower:
            return "low-key dramatic lighting"

        return "cinematic naturalistic lighting"

    def _characters_in_scene(
        self,
        text: str,
        characters: list[Character],
    ) -> list[str]:

        lower = text.lower()

        result = []

        for character in characters:
            if (
                character.name.lower()
                in lower
            ):
                result.append(
                    character.name
                )

        # Role descriptors may not be referenced by full
        # names after normalization. Match canonical role text.
        if not result:

            for character in characters:
                role = (
                    character.role
                    .lower()
                )

                if (
                    role
                    and role != "story character"
                    and role in lower
                ):
                    result.append(
                        character.name
                    )

        # If one character exists and the segment has no
        # explicit name, it belongs to that character.
        if (
            not result
            and len(characters) == 1
        ):
            result.append(
                characters[0].name
            )

        return result

    def create_scenes(
        self,
        story: str,
        characters: list[Character],
    ) -> list[Scene]:

        units = self._split_story(
            story
        )

        scenes = []

        for index, unit in enumerate(
            units,
            start=1,
        ):

            mood = self._mood(
                unit.text
            )

            location = self._location(
                unit.text
            )

            scenes.append(
                Scene(
                    scene_id=(
                        f"scene_{index:03d}"
                    ),
                    order=index,
                    location=location,
                    time_of_day=self._time_of_day(
                        unit.text
                    ),
                    weather=self._weather(
                        unit.text
                    ),
                    atmosphere=(
                        f"{mood}; natural cinematic "
                        "environmental ambience"
                    ),
                    description=unit.text,
                    mood=mood,
                    lighting=self._lighting(
                        unit.text
                    ),
                    environment_details=[
                        "cinematic depth",
                        "stable environmental continuity",
                    ],
                    key_props=[],
                    scene_objective=unit.text,
                    characters=(
                        self._characters_in_scene(
                            unit.text,
                            characters,
                        )
                    ),
                    story_summary=unit.text,
                    continuity_notes="",
                    shot_ids=[],
                )
            )

        return scenes

    # ============================================================
    # SHOT PLANNING
    # ============================================================

    @staticmethod
    def _camera(
        order: int,
    ) -> tuple[str, str]:

        choices = [
            (
                "wide establishing shot",
                "slow controlled push-in",
            ),
            (
                "medium cinematic shot",
                "subtle lateral tracking",
            ),
            (
                "close cinematic shot",
                "slow controlled forward movement",
            ),
            (
                "over-the-shoulder shot",
                "gentle cinematic follow",
            ),
        ]

        return choices[
            (order - 1)
            % len(choices)
        ]

    def _refs_for_characters(
        self,
        characters: list[Character],
        names: list[str],
    ):
        by_name = {
            character.name.lower():
                character
            for character in characters
        }

        images = []
        videos = []
        audio = []

        character_image_bindings = {}

        for name in names:

            character = by_name.get(
                name.lower()
            )

            if character is None:
                continue

            character_images = (
                character.normalized_reference_paths()
            )

            character_videos = (
                character.normalized_video_paths()
            )

            character_audio = (
                character.normalized_audio_paths()
            )

            character_image_bindings[
                character.name
            ] = character_images

            for path in character_images:
                if (
                    path not in images
                    and len(images)
                    < H3_MAX_REFERENCE_IMAGES
                ):
                    images.append(
                        path
                    )

            for path in character_videos:
                if (
                    path not in videos
                    and len(videos)
                    < H3_MAX_REFERENCE_VIDEOS
                ):
                    videos.append(
                        path
                    )

            for path in character_audio:
                if (
                    path not in audio
                    and len(audio)
                    < H3_MAX_REFERENCE_AUDIO
                ):
                    audio.append(
                        path
                    )

        return (
            images,
            videos,
            audio,
            character_image_bindings,
        )

    @staticmethod
    def _native_audio_instruction(
        scene: Scene,
    ) -> str:

        return (
            "Native H3 audio generation policy: "
            "generate suitable scene ambience natively. "
            f"Soundscape: {scene.atmosphere}. "
            "Do not require an external audio reference unless "
            "one is explicitly supplied."
        )

    def create_shots(
        self,
        story: str,
        characters: list[Character],
        scenes: list[Scene],
        workflow_mode: str = WORKFLOW_AUTO,
        profile: str = "base",
    ) -> list[Shot]:

        shots = []

        by_name = {
            character.name.lower():
                character
            for character in characters
        }

        for scene in scenes:

            selected_characters = list(
                scene.characters
            )

            (
                images,
                videos,
                audio,
                bindings,
            ) = self._refs_for_characters(
                characters,
                selected_characters,
            )

            locks = (
                IdentityContinuity.build_locks(
                    characters,
                    selected_characters,
                )
            )

            reference_bindings = (
                IdentityContinuity
                .build_reference_bindings(
                    images,
                    bindings,
                )
            )

            camera_shot, camera_movement = (
                self._camera(
                    len(shots) + 1
                )
            )

            detailed = (
                f"{scene.description} "
                f"Location: {scene.location}. "
                f"Time: {scene.time_of_day}. "
                f"Weather: {scene.weather}. "
                f"Lighting: {scene.lighting}. "
                f"Mood: {scene.mood}. "
                f"Camera: {camera_shot}; "
                f"{camera_movement}. "
                f"Environment: "
                f"{', '.join(scene.environment_details)}."
            )

            positive_prompt = (
                f"{detailed} "
                f"{self._native_audio_instruction(scene)}"
            )

            negative_prompt = (
                "identity drift, altered face geometry, "
                "different hairstyle, altered hairline, "
                "different body proportions, "
                "altered skin tone, facial deformation, "
                "extra limbs, duplicate person, "
                "inconsistent clothing, inconsistent props"
            )

            (
                positive_prompt,
                negative_prompt,
            ) = IdentityContinuity.merge(
                visual_prompt=positive_prompt,
                locks=locks,
                bindings=reference_bindings,
                negative_prompt=negative_prompt,
            )

            if (
                workflow_mode
                == WORKFLOW_TURBO_REF2V
                or profile == "turbo"
            ):
                selected_workflow = (
                    WORKFLOW_TURBO_REF2V
                )
                steps = TURBO_STEPS
            else:
                selected_workflow = (
                    WORKFLOW_REF2V
                )
                steps = H3_STEPS

            shot = Shot(
                shot_id=(
                    f"shot_{len(shots) + 1:03d}"
                ),
                scene_id=scene.scene_id,
                order=len(shots) + 1,
                duration_seconds=5.2,
                characters=selected_characters,
                location=scene.location,
                action=scene.scene_objective,
                camera_shot=camera_shot,
                camera_movement=camera_movement,
                lighting=scene.lighting,
                mood=scene.mood,
                visual_prompt=positive_prompt,
                retention_analysis=(
                    "Maintain visual continuity, "
                    "clear subject readability and "
                    "cinematic progression."
                ),
                detailed_description=detailed,
                overall_soundscape=(
                    self._native_audio_instruction(
                        scene
                    )
                ),
                non_diegetic_music=(
                    "Subtle cinematic score when "
                    "appropriate to the story."
                ),
                negative_prompt=negative_prompt,
                continuity_notes=(
                    scene.continuity_notes
                ),
                seed=(
                    100000
                    + len(shots)
                ),
                reference_images=images,
                reference_videos=videos,
                reference_audio=(
                    audio[0]
                    if audio
                    else None
                ),
                reference_audio_paths=audio,
                reference_audio_by_character={
                    name: (
                        by_name[
                            name.lower()
                        ]
                        .normalized_audio_paths()
                    )
                    for name in selected_characters
                    if name.lower()
                    in by_name
                },
                reference_video_by_character={
                    name: (
                        by_name[
                            name.lower()
                        ]
                        .normalized_video_paths()
                    )
                    for name in selected_characters
                    if name.lower()
                    in by_name
                },
                speaking_characters=selected_characters,
                speech_text="",
                reference_bindings=reference_bindings,
                identity_locks=locks,
                workflow_mode=selected_workflow,
                keyframe_images=[],
                keyframe_positions=[],
                extend_take_source_video=None,
                width=H3_WIDTH,
                height=H3_HEIGHT,
                fps=H3_FPS,
                frames_per_shot=H3_FRAMES_PER_SHOT,
                steps=steps,
            )

            shots.append(
                shot
            )

        return shots

    # ============================================================
    # PLAN VALIDATION
    # ============================================================

    def _validate_plan(
        self,
        characters,
        shots,
    ):

        for shot in shots:

            if (
                shot.width,
                shot.height,
            ) != (
                H3_WIDTH,
                H3_HEIGHT,
            ):
                raise RuntimeError(
                    f"{shot.shot_id}: invalid H3 resolution."
                )

            if shot.fps != H3_FPS:
                raise RuntimeError(
                    f"{shot.shot_id}: invalid FPS."
                )

            images = len(
                shot.reference_images
            )

            videos = len(
                shot.reference_videos
            )

            audio = len(
                shot.reference_audio_paths
            )

            if images > H3_MAX_REFERENCE_IMAGES:
                raise RuntimeError(
                    f"{shot.shot_id}: too many image refs."
                )

            if videos > H3_MAX_REFERENCE_VIDEOS:
                raise RuntimeError(
                    f"{shot.shot_id}: too many video refs."
                )

            if audio > H3_MAX_REFERENCE_AUDIO:
                raise RuntimeError(
                    f"{shot.shot_id}: too many audio refs."
                )

            if (
                images
                + videos
                + audio
                > H3_MAX_REFERENCE_FILES
            ):
                raise RuntimeError(
                    f"{shot.shot_id}: too many total refs."
                )

            # Audio reference is optional.
            # H3 may generate native audio from the prompt.
            if not audio:
                if (
                    "Native H3 audio generation policy"
                    not in shot.overall_soundscape
                ):
                    raise RuntimeError(
                        f"{shot.shot_id}: missing native audio policy."
                    )

            if shot.characters:
                if not shot.identity_locks:
                    raise RuntimeError(
                        f"{shot.shot_id}: "
                        "character identity locks missing."
                    )

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

        scene_shot_ids = {}

        for shot in shots:
            scene_shot_ids.setdefault(
                shot.scene_id,
                [],
            ).append(
                shot.shot_id
            )

        scene_dicts = []

        for scene in scenes:
            scene.shot_ids = list(
                scene_shot_ids.get(
                    scene.scene_id,
                    [],
                )
            )

            scene_dicts.append(
                scene.to_dict()
            )

        self._validate_plan(
            characters,
            shots,
        )

        character_dicts = [
            character.to_dict()
            for character in characters
        ]

        shot_dicts = [
            shot.to_dict()
            for shot in shots
        ]

        return {
            "story": story,
            "story_mode": mode,
            "profile": profile,
            "workflow_mode": workflow_mode,
            "preview_ready": True,

            "character_count": len(
                character_dicts
            ),
            "scene_count": len(
                scene_dicts
            ),
            "shot_count": len(
                shot_dicts
            ),

            "characters": character_dicts,
            "scenes": scene_dicts,
            "shots": shot_dicts,

            "width": H3_WIDTH,
            "height": H3_HEIGHT,
            "fps": H3_FPS,
            "frames_per_shot": (
                H3_FRAMES_PER_SHOT
            ),
            "normal_steps": H3_STEPS,
            "turbo_steps": TURBO_STEPS,

            "audio_policy": (
                "Use supplied reference audio when present; "
                "otherwise request native H3 audio generation "
                "from the shot soundscape/dialogue prompt."
            ),
        }

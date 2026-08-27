from __future__ import annotations

import ctypes
import gc
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

from planner.config import (
    AI_STORY_MODE,
    DIRECTOR_KAGGLE_INPUT_ROOT,
    DIRECTOR_MAX_TOKENS,
    DIRECTOR_MODEL_ENV,
    DIRECTOR_MODEL_FILENAME,
    DIRECTOR_N_BATCH,
    DIRECTOR_N_CTX,
    DIRECTOR_N_GPU_LAYERS,
    DIRECTOR_TEMPERATURE,
    DIRECTOR_THREADS,
    DIRECTOR_TOP_P,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    director_enabled,
)


class QwenDirector:

    FORBIDDEN_CHARACTER_NAMES = {
        "treat",
        "develop",
        "clarify",
        "every",
        "above",
        "far",
        "tone",
        "visual",
        "story",
        "scene",
        "scenes",
        "shot",
        "shots",
        "camera",
        "lighting",
        "sound",
        "soundscape",
        "environment",
        "action",
        "continuity",
        "mood",
        "location",
        "weather",
        "dialogue",
        "music",
        "character",
        "characters",
        "the",
        "a",
        "an",
        "he",
        "she",
        "his",
        "her",
        "it",
        "they",
        "them",
        "this",
        "that",
        "these",
        "those",
        "when",
        "while",
        "after",
        "before",
        "finally",
        "suddenly",
        "meanwhile",
        "developing",
        "preserve",
        "expand",
        "generate",
        "generation",
        "priority",
        "description",
        "details",
        "detail",
        "camera_shot",
        "camera_movement",
        "negative_prompt",
        "visual_prompt",
    }

    VALID_GENERIC_ROLES = {
        "man",
        "woman",
        "girl",
        "boy",
        "child",
        "person",
        "hero",
        "heroine",
        "explorer",
        "detective",
        "scientist",
        "soldier",
        "warrior",
        "king",
        "queen",
        "robot",
        "android",
        "pilot",
    }

    _MODE_LABELS = {
        AI_STORY_MODE: "AI STORY MODE",
        EXPAND_USER_STORY_MODE: "EXPAND STORY MODE",
        PRESERVE_USER_STORY_MODE: "PRESERVE STORY MODE",
    }

    def __init__(
        self,
        project_root: Path,
    ):

        self.project_root = (
            Path(
                project_root
            )
            .resolve()
        )

        self._llama = None

        self._model_path = (
            self._find_model()
            if director_enabled()
            else None
        )

        self._fallback_planner = None
        self._current_visual_language: dict = {}

    # ========================================================
    # MODEL DISCOVERY
    # ========================================================

    def _find_model(
        self,
    ) -> Path:

        explicit = os.getenv(
            DIRECTOR_MODEL_ENV,
            "",
        ).strip()

        candidates: list[Path] = []

        if explicit:

            candidates.append(
                Path(
                    explicit
                )
            )

        candidates.extend(
            [
                (
                    self.project_root
                    / "data"
                    / "models"
                    / DIRECTOR_MODEL_FILENAME
                ),
                (
                    self.project_root
                    / DIRECTOR_MODEL_FILENAME
                ),
                (
                    DIRECTOR_KAGGLE_INPUT_ROOT
                    / DIRECTOR_MODEL_FILENAME
                ),
            ]
        )

        for root in (
            self.project_root,
            DIRECTOR_KAGGLE_INPUT_ROOT,
        ):

            if not root.exists():
                continue

            try:

                candidates.extend(
                    root.rglob(
                        DIRECTOR_MODEL_FILENAME
                    )
                )

            except OSError:

                continue

        unique: list[Path] = []
        seen: set[str] = set()

        for candidate in candidates:

            candidate = Path(
                candidate
            )

            try:

                key = str(
                    candidate.resolve()
                )

            except OSError:

                key = str(
                    candidate
                )

            if key in seen:
                continue

            seen.add(
                key
            )

            unique.append(
                candidate
            )

        existing = [
            path
            for path in unique
            if path.is_file()
        ]

        if not existing:

            raise FileNotFoundError(
                "Qwen director model was not found.\n"
                f"Expected filename: "
                f"{DIRECTOR_MODEL_FILENAME}\n"
                "Attach the Kaggle director-model dataset "
                "or set H3_DIRECTOR_MODEL_PATH."
            )

        if len(existing) > 1:

            raise RuntimeError(
                "Multiple Qwen director models were found:\n"
                + "\n".join(
                    str(path)
                    for path in existing
                )
            )

        return existing[0]

    @property
    def model_path(
        self,
    ) -> Path | None:

        return self._model_path

    @property
    def available(
        self,
    ) -> bool:

        return bool(
            director_enabled()
            and self._model_path is not None
            and self._model_path.is_file()
        )

    # ========================================================
    # DETERMINISTIC FALLBACK
    # ========================================================

    def _planner(
        self,
    ):

        if self._fallback_planner is None:

            from planner.production_planner import (
                ProductionPlanner,
            )

            self._fallback_planner = (
                ProductionPlanner(
                    self.project_root
                )
            )

        return self._fallback_planner

    def _build_deterministic_fallback(
        self,
        story: str,
    ) -> tuple[
        list[dict],
        list[dict],
    ]:

        story = str(
            story or ""
        ).strip()

        if not story:

            return (
                [],
                [],
            )

        planner = self._planner()

        characters = []
        scenes = []

        try:

            characters = (
                planner.create_characters(
                    story
                )
                or []
            )

        except Exception:

            characters = []

        try:

            scenes = (
                planner.create_scenes(
                    story,
                    characters,
                )
                or []
            )

        except Exception:

            scenes = []

        return (
            [
                character.to_dict()
                for character
                in characters
                if character is not None
            ],
            [
                scene.to_dict()
                for scene
                in scenes
                if scene is not None
            ],
        )

    # ========================================================
    # CUDA / MODEL LIFECYCLE
    # ========================================================

    @staticmethod
    def _load_nvidia_cuda_libraries() -> None:

        import site

        site_roots: list[Path] = []

        try:

            site_roots.extend(
                Path(path)
                for path
                in site.getsitepackages()
                if path
            )

        except Exception:
            pass

        try:

            user_site = (
                site.getusersitepackages()
            )

            if user_site:

                site_roots.append(
                    Path(
                        user_site
                    )
                )

        except Exception:
            pass

        cudart: list[Path] = []
        cublas: list[Path] = []

        for site_root in site_roots:

            nvidia_root = (
                site_root
                / "nvidia"
            )

            if not nvidia_root.is_dir():
                continue

            try:

                cudart.extend(
                    path
                    for path
                    in nvidia_root.rglob(
                        "libcudart.so.13*"
                    )
                    if path.is_file()
                )

                cublas.extend(
                    path
                    for path
                    in nvidia_root.rglob(
                        "libcublas.so.13*"
                    )
                    if path.is_file()
                )

            except OSError:

                continue

        if not cudart:

            raise RuntimeError(
                "libcudart.so.13 was not found."
            )

        if not cublas:

            raise RuntimeError(
                "libcublas.so.13 was not found."
            )

        cudart_lib = cudart[0]

        matching_cublas = [
            path
            for path
            in cublas
            if path.parent
            == cudart_lib.parent
        ]

        cublas_lib = (
            matching_cublas[0]
            if matching_cublas
            else cublas[0]
        )

        directories = [
            str(
                cudart_lib.parent
            ),
            str(
                cublas_lib.parent
            ),
        ]

        old_ld = os.environ.get(
            "LD_LIBRARY_PATH",
            "",
        )

        if old_ld:

            directories.append(
                old_ld
            )

        os.environ[
            "LD_LIBRARY_PATH"
        ] = ":".join(
            directories
        )

        try:

            ctypes.CDLL(
                str(
                    cudart_lib
                ),
                mode=ctypes.RTLD_GLOBAL,
            )

            ctypes.CDLL(
                str(
                    cublas_lib
                ),
                mode=ctypes.RTLD_GLOBAL,
            )

        except OSError as exc:

            raise RuntimeError(
                "Unable to load NVIDIA CUDA libraries:\n"
                f"CUDA runtime: {cudart_lib}\n"
                f"cuBLAS: {cublas_lib}\n"
                f"{exc}"
            ) from exc

    def load(
        self,
    ) -> None:

        if not self.available:
            return

        if self._llama is not None:
            return

        self._load_nvidia_cuda_libraries()

        try:

            from llama_cpp import Llama

        except ImportError as exc:

            raise RuntimeError(
                "llama-cpp-python is not installed."
            ) from exc

        try:

            self._llama = Llama(
                model_path=str(
                    self._model_path
                ),
                n_ctx=DIRECTOR_N_CTX,
                n_gpu_layers=DIRECTOR_N_GPU_LAYERS,
                n_batch=DIRECTOR_N_BATCH,
                n_threads=DIRECTOR_THREADS,
                flash_attn=True,
                verbose=False,
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to initialize Qwen3-14B director.\n"
                f"Model: {self._model_path}\n"
                f"Error: {exc}"
            ) from exc

    def unload(
        self,
    ) -> None:

        model = self._llama

        self._llama = None

        if model is not None:

            del model

        gc.collect()

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

                try:

                    torch.cuda.ipc_collect()

                except Exception:
                    pass

        except Exception:
            pass

    # ========================================================
    # TOKEN BUDGET
    # ========================================================

    def _count_tokens(
        self,
        text: str,
    ) -> int:

        if self._llama is None:

            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        return len(
            self._llama.tokenize(
                text.encode(
                    "utf-8"
                ),
                add_bos=True,
                special=True,
            )
        )

    def _available_output_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 512,
    ) -> tuple[int, int]:

        context = int(
            DIRECTOR_N_CTX
        )

        safety = 128

        prompt_tokens = (
            self._count_tokens(
                system_prompt
                + "\n\n"
                + user_prompt
            )
        )

        available = (
            context
            - prompt_tokens
            - safety
        )

        if available < minimum_completion:

            raise RuntimeError(
                "Qwen director prompt is too large "
                f"for the {context}-token context window.\n"
                f"Prompt tokens: {prompt_tokens}.\n"
                f"Available completion tokens: {available}."
            )

        return (
            prompt_tokens,
            min(
                int(
                    DIRECTOR_MAX_TOKENS
                ),
                available,
            ),
        )

    # ========================================================
    # MODE PROMPTS
    # ========================================================

    def _mode_instruction(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:

            return """
AI STORY MODE.

Treat the user input as a PREMISE, not a finished story.

Create a genuinely developed cinematic narrative.

Build:
- a memorable protagonist;
- meaningful supporting characters;
- a clear desire or objective;
- conflict;
- escalating complications;
- emotional or thematic progression;
- a strong climax;
- a satisfying resolution.

The resulting story should feel like a real short-film
story, not a paraphrase of the premise.

You are allowed to invent details that improve the story.
""".strip()

        if mode == EXPAND_USER_STORY_MODE:

            return """
EXPAND STORY MODE.

The user supplied an existing story.

The supplied story is the source of truth for its core events,
characters, chronology, setting and outcome.

Your task is to make that story substantially more compelling
without replacing it.

Preserve:
- important characters;
- important events;
- chronology;
- setting;
- outcome;
- explicit constraints.

Enrich it with:
- motivation;
- emotional depth;
- cause and effect;
- transitions;
- intermediate events;
- stakes;
- tension;
- sensory detail;
- character reactions;
- meaningful dialogue where appropriate;
- stronger escalation;
- a clearer dramatic progression;
- richer ending consequences.

Do NOT merely add adjectives.

Do NOT simply convert prose into camera directions.

The result must feel like a professionally expanded short-film
story while remaining recognizably the same story.
""".strip()

        if mode == PRESERVE_USER_STORY_MODE:

            return """
PRESERVE STORY MODE.

The supplied story text is immutable.

Do not rewrite it.

Do not add narrative events.

Use the supplied story as-is and create only the production
structure needed to visualize it.
""".strip()

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    def _story_text_system(
        self,
        mode: str,
    ) -> str:

        if mode == PRESERVE_USER_STORY_MODE:
            raise ValueError(
                "Preserve Story does not regenerate story text."
            )

        if mode == AI_STORY_MODE:

            instruction = """
You are the narrative writer for MiniMax H3.

The user provides a PREMISE.

Write a genuinely developed cinematic short-film story.

Create:
- a clear protagonist objective;
- meaningful conflict;
- escalating complications;
- character reactions;
- a strong climax;
- a satisfying resolution.

Add substantive events and cause-and-effect.
Do not merely paraphrase the premise.
Do not discuss the task.
Do not output notes, camera directions, metadata, or JSON wrappers around the story.
The story itself must be materially different from the premise.

Return JSON only:
{
  "story": "complete cinematic narrative"
}
""".strip()

        else:

            instruction = """
You are the narrative expansion writer for MiniMax H3.

The user provides an existing story.

Expand that story substantially while preserving:
- important characters;
- important events;
- chronology;
- setting;
- outcome;
- explicit constraints.

Add:
- motivation;
- emotional depth;
- cause and effect;
- transitions;
- intermediate events;
- stakes;
- tension;
- sensory detail;
- character reactions;
- stronger escalation;
- richer consequences.

Do not merely add adjectives.
Do not convert the story into camera directions.
Do not replace the original plot.
The result must be recognizably the same story but materially more developed.

Return JSON only:
{
  "story": "complete expanded narrative"
}
""".strip()

        return instruction

    def _story_text_user(
        self,
        mode: str,
        story: str,
    ) -> str:

        return json.dumps(
            {
                "mode": mode,
                "source_story": self._limit_text(
                    story,
                    6500,
                ),
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    def _story_text_system(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:
            return """
You are the narrative writer for MiniMax H3.

The user provides a premise.

Write a complete cinematic short-film story.
Add a protagonist objective, meaningful conflict,
escalation, character reactions, a climax, and a resolution.
Add substantive events; do not merely paraphrase the premise.

Output ONLY the story prose.
Do not output JSON.
Do not output labels, analysis, notes, camera directions,
or explanations.
""".strip()

        if mode == EXPAND_USER_STORY_MODE:
            return """
You are the narrative expansion writer for MiniMax H3.

Expand the supplied story substantially while preserving
its important characters, events, chronology, setting,
outcome, and explicit constraints.

Add motivation, emotional depth, cause and effect,
transitions, intermediate events, stakes, tension,
sensory detail, character reactions, stronger escalation,
and richer consequences.

Do not merely add adjectives.
Do not replace the original plot.
Do not convert the story into camera directions.

Output ONLY the expanded story prose.
Do not output JSON, labels, analysis, notes, or explanations.
""".strip()

        raise ValueError(
            "Preserve Story does not use a story-text pass."
        )

    def _story_text_user(
        self,
        mode: str,
        story: str,
    ) -> str:

        return (
            "MODE: "
            + str(mode)
            + "\n\nSOURCE STORY / PREMISE:\n"
            + self._limit_text(
                story,
                7000,
            )
        )

    def _metadata_director_system(
        self,
        mode: str,
    ) -> str:

        return f"""
You are the STORYBOARD PRODUCTION DIRECTOR for MiniMax H3.

{self._mode_instruction(mode)}

The story text supplied by the user is authoritative.
Do not rewrite narrative prose in this pass.

Create only:
- director_notes
- visual_language
- characters
- scenes

Do NOT create shots in this pass.

Create 4–6 meaningful narrative scenes.
Every scene must represent a real dramatic beat, not a camera instruction.

Avoid scene titles such as:
"distance"
"wind"
"atmosphere"
"cinematic environment"
"a wide cinematic shot"
"camera"

Use concise narrative-beat titles instead.

Every scene must include:
scene_id
title
order
location
time_of_day
weather
atmosphere
description
mood
lighting
color_temperature
environment_details
key_props
characters
scene_objective
continuity_notes

Create one coherent visual_language bible:
genre_tone
color_palette
lighting_philosophy
camera_philosophy
pacing

Return JSON only:
{{
  "director_notes": "string",
  "visual_language": {{
    "genre_tone": "string",
    "color_palette": "string",
    "lighting_philosophy": "string",
    "camera_philosophy": "string",
    "pacing": "string"
  }},
  "characters": [],
  "scenes": []
}}
""".strip()

    def _metadata_director_user(
        self,
        mode: str,
        story: str,
    ) -> str:

        return json.dumps(
            {
                "mode": mode,
                "story": self._limit_text(
                    story,
                    6000,
                ),
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    def _story_director_system(
        self,
        mode: str,
    ) -> str:

        return f"""
You are the STORY DIRECTOR for MiniMax H3.

{self._mode_instruction(mode)}

Return JSON only.

Top-level fields:

story
director_notes
characters
scenes
visual_language

VISUAL LANGUAGE BIBLE:

Create one coherent visual language for the whole production.
The visual_language object contains genre_tone, color_palette,
lighting_philosophy, camera_philosophy, and pacing.

Do NOT create shots in this pass.

STORY QUALITY:

The story must read as a coherent cinematic narrative,
not as notes or instructions.

CHARACTERS:

Only create actual story characters.

SCENES:

Create 4–6 meaningful story beats.

Do not use scene titles such as:
"distance"
"wind"
"camera"
"cinematic environment"

Instead use narrative beat titles such as:
"The First Warning"
"The Broken City"
"The Descent"
"The Last Beacon"

Every scene must include:

scene_id
title
order
location
time_of_day
weather
atmosphere
description
mood
lighting
color_temperature
environment_details
key_props
characters
scene_objective
continuity_notes

Use concrete, filmable environment details and a concrete lighting/color-temperature description.

A scene description must describe an actual narrative event.

Return:

{{
  "story": "string",
  "director_notes": "string",
  "visual_language": {{
    "genre_tone": "string",
    "color_palette": "string",
    "lighting_philosophy": "string",
    "camera_philosophy": "string",
    "pacing": "string"
  }},
  "characters": [],
  "scenes": []
}}
""".strip()

    def _sampling_for_mode(
        self,
        mode: str,
    ) -> tuple[float, float]:
    
        if mode == AI_STORY_MODE:
            return (
                0.78,
                0.90,
            )
    
        if mode == EXPAND_USER_STORY_MODE:
            return (
                0.62,
                0.88,
            )
    
        if mode == PRESERVE_USER_STORY_MODE:
            return (
                0.10,
                0.80,
            )
    
        return (
            DIRECTOR_TEMPERATURE,
            DIRECTOR_TOP_P,
        )
    
    def _shot_sampling(
        self,
    ) -> tuple[float, float]:

        return (
            0.68,
            0.92,
        )

    def _story_director_user(
        self,
        mode: str,
        story: str,
    ) -> str:

        return json.dumps(
            {
                "mode": mode,
                "user_story": story,
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    # ========================================================
    # CHARACTER RECOVERY
    # ========================================================

    def _character_recovery_system(
        self,
        mode: str,
    ) -> str:

        return f"""
You are the CHARACTER DIRECTOR for MiniMax H3.

{self._mode_instruction(mode)}

Extract or create ONLY the characters genuinely required
by the story.

Return JSON only:

{{
  "characters": [
    {{
      "name": "string",
      "role": "string",
      "description": "string",
      "personality": "string",
      "appearance": {{}},
      "clothing": {{}},
      "distinctive_features": [],
      "character_state": {{}},
      "continuity_rules": []
    }}
  ]
}}
""".strip()

    def _character_recovery_user(
        self,
        story: str,
        visual_language: dict | None = None,
    ) -> str:

        return json.dumps(
            {
                "story": self._limit_text(
                    story,
                    5500,
                ),
                "visual_language": dict(
                    visual_language
                    if isinstance(
                        visual_language,
                        dict,
                    )
                    else {}
                ),
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    def _recover_characters(
        self,
        mode: str,
        story: str,
    ) -> list[dict]:

        try:

            temperature, top_p = (
                self._sampling_for_mode(
                    mode
                )
            )

            response = self._chat_json(
                self._character_recovery_system(
                    mode
                ),
                self._character_recovery_user(
                    story,
                    getattr(
                        self,
                        "_current_visual_language",
                        {},
                    ),
                ),
                minimum_completion=400,
                temperature=temperature,
                top_p=top_p,
                call_name="character_recovery",
                max_completion=1000,
            )

        except Exception:

            return []

        return self._sanitize_characters(
            response.get(
                "characters",
                [],
            )
        )

    # ========================================================
    # SCENE RECOVERY
    # ========================================================

    def _scene_recovery_system(
        self,
        mode: str,
    ) -> str:

        return f"""
You are the SCENE DIRECTOR for MiniMax H3.

{self._mode_instruction(mode)}

Create 4–6 meaningful narrative scenes from the supplied story.

Each scene must be a real dramatic beat.

Return JSON only:

{{
  "scenes": [
    {{
      "scene_id": "scene_001",
      "title": "narrative beat title",
      "order": 1,
      "location": "string",
      "time_of_day": "string",
      "weather": "string",
      "atmosphere": "string",
      "description": "real narrative event",
      "mood": "string",
      "lighting": "string",
      "color_temperature": "string",
      "environment_details": [],
      "key_props": [],
      "characters": [],
      "scene_objective": "string",
      "continuity_notes": "string"
    }}
  ]
}}
""".strip()

    def _scene_recovery_user(
        self,
        story: str,
        characters: list[dict],
        visual_language: dict | None = None,
    ) -> str:

        return json.dumps(
            {
                "story": self._limit_text(
                    story,
                    5000,
                ),
                "characters": [
                    {
                        "name": item.get(
                            "name",
                            "",
                        ),
                        "role": item.get(
                            "role",
                            "",
                        ),
                    }
                    for item
                    in characters
                ],
                "visual_language": dict(
                    visual_language
                    if isinstance(
                        visual_language,
                        dict,
                    )
                    else {}
                ),
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    def _recover_scenes(
        self,
        mode: str,
        story: str,
        characters: list[dict],
        character_names: set[str],
    ) -> list[dict]:

        try:

            temperature, top_p = (
                self._sampling_for_mode(
                    mode
                )
            )

            response = self._chat_json(
                self._scene_recovery_system(
                    mode
                ),
                self._scene_recovery_user(
                    story,
                    characters,
                    getattr(
                        self,
                        "_current_visual_language",
                        {},
                    ),
                ),
                minimum_completion=450,
                temperature=temperature,
                top_p=top_p,
                call_name="scene_recovery",
                max_completion=1600,
            )

        except Exception:

            return []

        return self._sanitize_scenes(
            response.get(
                "scenes",
                [],
            ),
            character_names,
        )

    # ========================================================
    # SHOT PROMPTS
    # ========================================================

    def _shot_director_system(
        self,
    ) -> str:

        return """
You are the CINEMATIC SHOT DIRECTOR for MiniMax H3.

Create exactly 2 or 3 distinct shots for ONE scene.

Use ONLY characters supplied in the scene.

Every shot must materially advance or visualize the scene.
Use clear cinematic progression:
1. orientation / establishing
2. action / subject
3. detail, reaction, escalation, or reveal when justified

Vary framing, movement, lens/DOF, composition, and lighting
when the scene supports it. Do not repeat the same camera
combination by default.

FRAMING:
extreme wide, wide, full shot, medium wide, medium,
medium close-up, close-up, extreme close-up,
over-the-shoulder, two-shot, POV, insert,
low angle, high angle, dutch angle.

MOVEMENT:
static, slow pan, tilt, dolly in, dolly out, truck,
pedestal, crane, handheld, steadicam glide,
whip pan, push-in, pull-out, orbit, tracking shot.

LENS / DOF:
wide-angle, normal perspective, telephoto compression,
shallow depth of field, deep focus, selective focus.

COMPOSITION:
rule of thirds, centered symmetry, leading lines,
foreground framing, negative space, silhouette,
depth layering, diagonal composition, subject isolation.

LIGHTING:
warm tungsten, cool daylight, golden-hour backlight,
blue-hour ambient, moonlight, practical neon,
hard chiaroscuro, soft diffused overcast,
firelight flicker, mixed practical and ambient.

Return JSON only with this compact structure:

{
  "shots": [
    {
      "shot_id": "scene_001_shot_001",
      "scene_id": "scene_001",
      "duration_seconds": 5.2,
      "characters": [],
      "location": "string",
      "action": "string",
      "camera_shot": "string",
      "camera_movement": "string",
      "lens_and_depth_of_field": "string",
      "composition_notes": "string",
      "lighting": "string",
      "color_temperature": "string",
      "mood": "string",
      "visual_prompt": "string",
      "detailed_description": "string",
      "continuity_notes": "string",
      "speaking_characters": [],
      "speech_text": ""
    }
  ]
}
""".strip()

    def _shot_director_user(
        self,
        story: str,
        characters: list[dict],
        scene: dict,
        visual_language: dict | None = None,
    ) -> str:

        return json.dumps(
            {
                "story": self._limit_text(
                    story,
                    3500,
                ),
                "characters": [
                    {
                        "name": item.get(
                            "name",
                            "",
                        ),
                        "role": item.get(
                            "role",
                            "",
                        ),
                        "appearance": item.get(
                            "appearance",
                            {},
                        ),
                        "clothing": item.get(
                            "clothing",
                            {},
                        ),
                    }
                    for item
                    in characters
                ],
                "visual_language": dict(
                    visual_language
                    if isinstance(
                        visual_language,
                        dict,
                    )
                    else {}
                ),
                "scene": {
                    "scene_id": scene.get(
                        "scene_id",
                        "",
                    ),
                    "title": scene.get(
                        "title",
                        "",
                    ),
                    "location": scene.get(
                        "location",
                        "",
                    ),
                    "description": self._limit_text(
                        scene.get(
                            "description",
                            "",
                        ),
                        1800,
                    ),
                    "mood": scene.get(
                        "mood",
                        "",
                    ),
                    "lighting": scene.get(
                        "lighting",
                        "",
                    ),
                    "color_temperature": scene.get(
                        "color_temperature",
                        "",
                    ),
                    "time_of_day": scene.get(
                        "time_of_day",
                        "",
                    ),
                    "weather": scene.get(
                        "weather",
                        "",
                    ),
                    "atmosphere": scene.get(
                        "atmosphere",
                        "",
                    ),
                    "environment_details": scene.get(
                        "environment_details",
                        [],
                    ),
                    "key_props": scene.get(
                        "key_props",
                        [],
                    ),
                    "characters": scene.get(
                        "characters",
                        [],
                    ),
                    "scene_objective": scene.get(
                        "scene_objective",
                        "",
                    ),
                    "continuity_notes": scene.get(
                        "continuity_notes",
                        "",
                    ),
                },
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    # ========================================================
    # JSON / TEXT HELPERS
    # ========================================================

    @staticmethod
    def _limit_text(
        text: str,
        max_chars: int,
    ) -> str:

        value = str(
            text or ""
        ).strip()

        if len(value) <= max_chars:
            return value

        return (
            value[
                :max_chars
            ]
            .rstrip()
            + "…"
        )

    @staticmethod
    def _extract_json(
        text: str,
    ) -> dict:

        value = str(
            text or ""
        ).strip()

        value = re.sub(
            r"^```(?:json)?",
            "",
            value,
            flags=re.IGNORECASE,
        ).strip()

        value = re.sub(
            r"```$",
            "",
            value,
        ).strip()

        start = value.find(
            "{"
        )

        end = value.rfind(
            "}"
        )

        if start < 0 or end <= start:

            raise RuntimeError(
                "Qwen director did not return JSON."
            )

        try:

            result = json.loads(
                value[
                    start:end + 1
                ]
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Qwen director returned invalid JSON: "
                f"{exc}"
            ) from exc

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Qwen director output must be a JSON object."
            )

        return result

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 512,
        temperature: float | None = None,
        top_p: float | None = None,
        call_name: str = "unknown",
        max_completion: int | None = None,
        json_mode: bool = True,
    ) -> dict:

        if self._llama is None:
            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        if temperature is None:
            temperature = DIRECTOR_TEMPERATURE

        if top_p is None:
            top_p = DIRECTOR_TOP_P

        _, max_tokens = (
            self._available_output_tokens(
                system_prompt,
                user_prompt,
                minimum_completion=minimum_completion,
            )
        )

        if max_completion is not None:
            max_tokens = min(
                max_tokens,
                int(max_completion),
            )

        kwargs = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {
                "type": "json_object",
            }

        started = time.perf_counter()
        response = None

        try:
            response = (
                self._llama.create_chat_completion(
                    **kwargs
                )
            )
        finally:
            elapsed = (
                time.perf_counter()
                - started
            )

            usage = (
                response.get(
                    "usage",
                    {},
                )
                if isinstance(
                    response,
                    dict,
                )
                else {}
            )

            print(
                "[QWEN]",
                call_name,
                f"elapsed={elapsed:.2f}s",
                f"prompt_tokens={usage.get('prompt_tokens', 0)}",
                f"completion_tokens={usage.get('completion_tokens', 0)}",
                f"max_tokens={max_tokens}",
                flush=True,
            )

        try:
            content = (
                response[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]
            )
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Qwen director returned an "
                "unexpected completion structure."
            ) from exc

        if not content:
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        return self._extract_json(
            content
        )

    def _chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 256,
        temperature: float | None = None,
        top_p: float | None = None,
        call_name: str = "unknown",
        max_completion: int | None = None,
    ) -> str:

        if self._llama is None:
            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        if temperature is None:
            temperature = DIRECTOR_TEMPERATURE

        if top_p is None:
            top_p = DIRECTOR_TOP_P

        _, max_tokens = (
            self._available_output_tokens(
                system_prompt,
                user_prompt,
                minimum_completion=minimum_completion,
            )
        )

        if max_completion is not None:
            max_tokens = min(
                max_tokens,
                int(max_completion),
            )

        started = time.perf_counter()
        response = None

        try:
            response = (
                self._llama.create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
            )
        finally:
            elapsed = (
                time.perf_counter()
                - started
            )

            usage = (
                response.get(
                    "usage",
                    {},
                )
                if isinstance(
                    response,
                    dict,
                )
                else {}
            )

            print(
                "[QWEN]",
                call_name,
                f"elapsed={elapsed:.2f}s",
                f"prompt_tokens={usage.get('prompt_tokens', 0)}",
                f"completion_tokens={usage.get('completion_tokens', 0)}",
                f"max_tokens={max_tokens}",
                flush=True,
            )

        try:
            content = (
                response[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]
            )
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Qwen director returned an "
                "unexpected completion structure."
            ) from exc

        content = str(
            content or ""
        ).strip()

        if not content:
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        return content

    # ========================================================
    # CHARACTER SANITIZATION
    # ========================================================

    def _valid_character_name(
        self,
        name: str,
    ) -> bool:

        value = str(
            name or ""
        ).strip()

        if not value:
            return False

        lowered = value.lower()

        if lowered in (
            self.VALID_GENERIC_ROLES
        ):
            return True

        if lowered in (
            self.FORBIDDEN_CHARACTER_NAMES
        ):
            return False

        if len(
            value.split()
        ) > 5:
            return False

        if any(
            token in value
            for token in (
                ":",
                ";",
                "|",
                "{",
                "}",
                "[",
                "]",
            )
        ):
            return False

        return True

    @staticmethod
    def _slug(
        value: str,
    ) -> str:

        cleaned = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(
                value or ""
            ).lower(),
        ).strip("_")

        return (
            cleaned
            or "character"
        )

    def _sanitize_characters(
        self,
        characters,
    ) -> list[dict]:

        result: list[dict] = []
        seen: set[str] = set()

        for value in (
            characters or []
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            name = str(
                value.get(
                    "name",
                    "",
                )
                or ""
            ).strip()

            if not self._valid_character_name(
                name
            ):
                continue

            key = name.lower()

            if key in seen:
                continue

            seen.add(
                key
            )

            character_id = str(
                value.get(
                    "character_id",
                    "",
                )
                or ""
            ).strip()

            if not character_id:

                character_id = (
                    f"char_{self._slug(name)}"
                )

            result.append(
                {
                    "character_id":
                        character_id,

                    "name":
                        name,

                    "role":
                        str(
                            value.get(
                                "role",
                                "story character",
                            )
                            or "story character"
                        ),

                    "description":
                        str(
                            value.get(
                                "description",
                                "",
                            )
                            or ""
                        ),

                    "personality":
                        str(
                            value.get(
                                "personality",
                                "",
                            )
                            or ""
                        ),

                    "appearance":
                        dict(
                            value.get(
                                "appearance",
                                {},
                            )
                            or {}
                        ),

                    "clothing":
                        dict(
                            value.get(
                                "clothing",
                                {},
                            )
                            or {}
                        ),

                    "distinctive_features":
                        list(
                            value.get(
                                "distinctive_features",
                                [],
                            )
                            or []
                        ),

                    "character_state":
                        dict(
                            value.get(
                                "character_state",
                                {},
                            )
                            or {}
                        ),

                    "continuity_rules":
                        list(
                            value.get(
                                "continuity_rules",
                                [],
                            )
                            or []
                        ),
                }
            )

        return result

    @staticmethod
    def _clean_list(
        value,
        limit: int | None = None,
    ) -> list[str]:
        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):
            items = [
                item.strip()
                for item in re.split(
                    r"[\n,;]+",
                    value,
                )
                if item.strip()
            ]
        elif isinstance(
            value,
            (list, tuple, set),
        ):
            items = [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]
        else:
            items = [
                str(value).strip()
            ] if str(value).strip() else []

        result: list[str] = []
        seen: set[str] = set()

        for item in items:
            key = item.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

            if (
                limit is not None
                and len(result) >= limit
            ):
                break

        return result

    # ========================================================
    # SCENE SANITIZATION
    # ========================================================

    def _sanitize_scenes(
        self,
        scenes,
        character_names: set[str],
    ) -> list[dict]:

        result: list[dict] = []

        for index, value in enumerate(
            scenes or [],
            start=1,
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            description = str(
                value.get(
                    "description",
                    "",
                )
                or ""
            ).strip()

            if not description:

                description = str(
                    value.get(
                        "scene_objective",
                        "",
                    )
                    or ""
                ).strip()

            if not description:
                continue

            lower = description.lower()

            if lower.startswith(
                (
                    "tone:",
                    "visual priority:",
                    "visual priorities:",
                    "camera:",
                    "lighting:",
                    "mood:",
                    "sound:",
                )
            ):
                continue

            selected: list[str] = []

            for name in (
                value.get(
                    "characters",
                    [],
                )
                or []
            ):

                canonical = str(
                    name
                ).strip()

                if (
                    canonical.lower()
                    in character_names
                ):
                    selected.append(
                        canonical
                    )

            scene_id = str(
                value.get(
                    "scene_id",
                    f"scene_{index:03d}",
                )
                or f"scene_{index:03d}"
            ).strip()

            title = str(
                value.get(
                    "title",
                    "",
                )
                or ""
            ).strip()

            if not title:

                title = (
                    str(
                        value.get(
                            "location",
                            "",
                        )
                        or ""
                    ).strip()
                    or f"Scene {index}"
                )

            result.append(
                {
                    "scene_id":
                        scene_id,

                    "title":
                        title,

                    "order":
                        len(result) + 1,

                    "location":
                        str(
                            value.get(
                                "location",
                                "",
                            )
                            or ""
                        ),

                    "time_of_day":
                        str(
                            value.get(
                                "time_of_day",
                                "",
                            )
                            or ""
                        ),

                    "weather":
                        str(
                            value.get(
                                "weather",
                                "",
                            )
                            or ""
                        ),

                    "atmosphere":
                        str(
                            value.get(
                                "atmosphere",
                                "",
                            )
                            or ""
                        ),

                    "description":
                        description,

                    "mood":
                        str(
                            value.get(
                                "mood",
                                "",
                            )
                            or ""
                        ),

                    "lighting":
                        str(
                            value.get(
                                "lighting",
                                "",
                            )
                            or ""
                        ),

                    "color_temperature":
                        str(
                            value.get(
                                "color_temperature",
                                "",
                            )
                            or ""
                        ),

                    "environment_details":
                        self._clean_list(
                            value.get(
                                "environment_details",
                                [],
                            ),
                            limit=12,
                        ),

                    "key_props":
                        self._clean_list(
                            value.get(
                                "key_props",
                                [],
                            ),
                            limit=8,
                        ),

                    "scene_objective":
                        str(
                            value.get(
                                "scene_objective",
                                "",
                            )
                            or ""
                        ),

                    "characters":
                        self._clean_list(
                        selected,
                    ),

                    "story_summary":
                        str(
                            value.get(
                                "story_summary",
                                description,
                            )
                            or description
                        ),

                    "continuity_notes":
                        str(
                            value.get(
                                "continuity_notes",
                                "",
                            )
                            or ""
                        ),

                    "shot_ids":
                        [],
                }
            )

        return result

    # ========================================================
    # SHOT SANITIZATION
    # ========================================================

    def _sanitize_shots(
        self,
        shots,
        scene: dict,
        character_names: set[str],
    ) -> list[dict]:

        result: list[dict] = []

        scene_id = str(
            scene.get(
                "scene_id",
                "",
            )
        )

        allowed = {
            name.lower(): name
            for name
            in character_names
        }

        for index, value in enumerate(
            shots or [],
            start=1,
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            candidate_scene_id = str(
                value.get(
                    "scene_id",
                    scene_id,
                )
                or scene_id
            ).strip()

            if candidate_scene_id != scene_id:
                continue

            selected: list[str] = []

            for name in (
                value.get(
                    "characters",
                    scene.get(
                        "characters",
                        [],
                    ),
                )
                or []
            ):

                canonical = allowed.get(
                    str(
                        name
                    ).strip().lower()
                )

                if canonical is not None:
                    selected.append(
                        canonical
                    )

            speaking: list[str] = []

            for name in (
                value.get(
                    "speaking_characters",
                    [],
                )
                or []
            ):

                canonical = allowed.get(
                    str(
                        name
                    ).strip().lower()
                )

                if canonical is not None:
                    speaking.append(
                        canonical
                    )

            try:

                duration = float(
                    value.get(
                        "duration_seconds",
                        5.2,
                    )
                    or 5.2
                )

            except (
                TypeError,
                ValueError,
            ):

                duration = 5.2

            result.append(
                {
                    "shot_id": str(
                        value.get(
                            "shot_id",
                            "",
                        )
                        or ""
                    ).strip(),

                    "scene_id":
                        scene_id,

                    "order":
                        len(result) + 1,

                    "duration_seconds":
                        duration,

                    "characters":
                        selected,

                    "location":
                        str(
                            value.get(
                                "location",
                                scene.get(
                                    "location",
                                    "",
                                ),
                            )
                            or scene.get(
                                "location",
                                "",
                            )
                        ),

                    "action":
                        str(
                            value.get(
                                "action",
                                scene.get(
                                    "scene_objective",
                                    scene.get(
                                        "description",
                                        "",
                                    ),
                                ),
                            )
                            or ""
                        ),

                    "camera_shot":
                        str(
                            value.get(
                                "camera_shot",
                                "cinematic medium shot",
                            )
                            or "cinematic medium shot"
                        ),

                    "camera_movement":
                        str(
                            value.get(
                                "camera_movement",
                                "controlled cinematic movement",
                            )
                            or "controlled cinematic movement"
                        ),

                    "lens_and_depth_of_field":
                        str(
                            value.get(
                                "lens_and_depth_of_field",
                                "normal perspective with moderate depth of field",
                            )
                            or "normal perspective with moderate depth of field"
                        ),

                    "composition_notes":
                        str(
                            value.get(
                                "composition_notes",
                                "clear subject separation and readable depth",
                            )
                            or "clear subject separation and readable depth"
                        ),

                    "color_temperature":
                        str(
                            value.get(
                                "color_temperature",
                                scene.get(
                                    "color_temperature",
                                    "",
                                ),
                            )
                            or scene.get(
                                "color_temperature",
                                "",
                            )
                        ),

                    "lighting":
                        str(
                            value.get(
                                "lighting",
                                scene.get(
                                    "lighting",
                                    "",
                                ),
                            )
                            or scene.get(
                                "lighting",
                                "",
                            )
                        ),

                    "mood":
                        str(
                            value.get(
                                "mood",
                                scene.get(
                                    "mood",
                                    "",
                                ),
                            )
                            or scene.get(
                                "mood",
                                "",
                            )
                        ),

                    "visual_prompt":
                        str(
                            value.get(
                                "visual_prompt",
                                scene.get(
                                    "description",
                                    "",
                                ),
                            )
                            or scene.get(
                                "description",
                                "",
                            )
                        ),

                    "retention_analysis":
                        str(
                            value.get(
                                "retention_analysis",
                                "Maintain narrative and visual continuity.",
                            )
                            or "Maintain narrative and visual continuity."
                        ),

                    "detailed_description":
                        str(
                            value.get(
                                "detailed_description",
                                scene.get(
                                    "description",
                                    "",
                                ),
                            )
                            or scene.get(
                                "description",
                                "",
                            )
                        ),

                    "overall_soundscape":
                        str(
                            value.get(
                                "overall_soundscape",
                                "Natural cinematic environmental sound.",
                            )
                            or "Natural cinematic environmental sound."
                        ),

                    "non_diegetic_music":
                        str(
                            value.get(
                                "non_diegetic_music",
                                "Subtle cinematic score when appropriate.",
                            )
                            or "Subtle cinematic score when appropriate."
                        ),

                    "negative_prompt":
                        str(
                            value.get(
                                "negative_prompt",
                                "identity drift, face deformation, inconsistent clothing",
                            )
                            or "identity drift, face deformation, inconsistent clothing"
                        ),

                    "continuity_notes":
                        str(
                            value.get(
                                "continuity_notes",
                                scene.get(
                                    "continuity_notes",
                                    "",
                                ),
                            )
                            or scene.get(
                                "continuity_notes",
                                "",
                            )
                        ),

                    "speaking_characters":
                        speaking,

                    "speech_text":
                        str(
                            value.get(
                                "speech_text",
                                "",
                            )
                            or ""
                        ),
                }
            )

        return result

    @staticmethod
    def _fallback_shots_for_scene(
        scene: dict,
        start_index: int,
    ) -> list[dict]:

        description = str(
            scene.get(
                "description",
                "",
            )
            or ""
        )

        characters = list(
            scene.get(
                "characters",
                [],
            )
            or []
        )

        base = {
            "scene_id": scene.get(
                "scene_id",
                "",
            ),
            "characters": characters,
            "location": scene.get(
                "location",
                "",
            ),
            "lighting": scene.get(
                "lighting",
                "",
            ),
            "mood": scene.get(
                "mood",
                "",
            ),
            "continuity_notes": scene.get(
                "continuity_notes",
                "",
            ),
        }

        shot_a = {
            **base,
            "shot_id":
                "",
            "order":
                1,
            "duration_seconds":
                5.2,
            "action":
                description,
            "camera_shot":
                "wide cinematic establishing shot",
            "camera_movement":
                "slow controlled tracking movement",
            "lens_and_depth_of_field":
                "wide-angle immersive perspective with deep focus",
            "composition_notes":
                "rule of thirds with strong leading lines and environmental depth",
            "color_temperature":
                str(scene.get(
                    "color_temperature",
                    "",
                ) or ""),
            "visual_prompt":
                description,
            "retention_analysis":
                "Establish the environment and narrative situation.",
            "detailed_description":
                description,
            "overall_soundscape":
                "Natural cinematic environmental sound.",
            "non_diegetic_music":
                "Subtle cinematic score.",
            "negative_prompt":
                "identity drift, duplicate character, face deformation, inconsistent clothing",
            "speaking_characters":
                [],
            "speech_text":
                "",
        }

        shot_b = {
            **base,
            "shot_id":
                "",
            "order":
                2,
            "duration_seconds":
                5.2,
            "action":
                (
                    scene.get(
                        "scene_objective",
                        "",
                    )
                    or description
                ),
            "camera_shot":
                "medium close-up",
            "camera_movement":
                "slow push-in",
            "lens_and_depth_of_field":
                "telephoto compression with shallow depth of field and soft background bokeh",
            "composition_notes":
                "subject isolation with negative space and layered foreground framing",
            "color_temperature":
                str(scene.get(
                    "color_temperature",
                    "",
                ) or ""),
            "visual_prompt":
                description,
            "retention_analysis":
                "Move from environment into the primary narrative beat.",
            "detailed_description":
                description,
            "overall_soundscape":
                "Natural cinematic environmental sound.",
            "non_diegetic_music":
                "Cinematic score building with the scene.",
            "negative_prompt":
                "identity drift, duplicate character, face deformation, inconsistent clothing",
            "speaking_characters":
                [],
            "speech_text":
                "",
        }

        return [
            shot_a,
            shot_b,
        ]

    # ========================================================
    # FINAL ID NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_ids(
        scenes: list[dict],
        shots: list[dict],
    ) -> None:

        seen_scene_ids: set[str] = set()

        for index, scene in enumerate(
            scenes,
            start=1,
        ):

            scene_id = str(
                scene.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()

            if (
                not scene_id
                or scene_id in seen_scene_ids
            ):

                scene_id = (
                    f"scene_{index:03d}"
                )

            seen_scene_ids.add(
                scene_id
            )

            scene[
                "scene_id"
            ] = scene_id

            scene[
                "order"
            ] = index

        used_shot_ids: set[str] = set()

        scene_counters: dict[str, int] = {}

        for global_index, shot in enumerate(
            shots,
            start=1,
        ):

            scene_id = str(
                shot.get(
                    "scene_id",
                    "",
                )
            ).strip()

            scene_counters[
                scene_id
            ] = (
                scene_counters.get(
                    scene_id,
                    0,
                )
                + 1
            )

            shot_number = (
                scene_counters[
                    scene_id
                ]
            )

            preferred = str(
                shot.get(
                    "shot_id",
                    "",
                )
                or ""
            ).strip()

            candidate = preferred

            if (
                not candidate
                or candidate in used_shot_ids
            ):

                candidate = (
                    f"{scene_id}"
                    f"_shot_{shot_number:03d}"
                )

            while candidate in used_shot_ids:

                candidate = (
                    f"{scene_id}"
                    f"_shot_{global_index:03d}"
                )

                global_index += 1

            used_shot_ids.add(
                candidate
            )

            shot[
                "shot_id"
            ] = candidate

    # ========================================================
    # MODE VALIDATION
    # ========================================================

    @staticmethod
    def _normalize_story(
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            str(
                text or ""
            ).strip(),
        )

    @staticmethod
    def _meaningful_tokens(
        text: str,
    ) -> set[str]:

        words = re.findall(
            r"[A-Za-z][A-Za-z'-]{3,}",
            str(
                text or ""
            ).lower(),
        )

        stop = {
            "this",
            "that",
            "with",
            "from",
            "into",
            "about",
            "have",
            "will",
            "they",
            "their",
            "there",
            "which",
            "while",
            "where",
            "would",
            "could",
            "should",
            "story",
            "then",
            "than",
            "when",
        }

        return {
            word
            for word in words
            if word not in stop
        }

    def _validate_mode_output(
        self,
        mode: str,
        user_input: str,
        story: str,
    ) -> None:

        source = self._normalize_story(
            user_input
        )

        result = self._normalize_story(
            story
        )

        if not result:

            raise RuntimeError(
                "Qwen director returned an empty story."
            )

        if mode == PRESERVE_USER_STORY_MODE:

            if source != result:

                raise RuntimeError(
                    "Preserve Story mode changed "
                    "the supplied story."
                )

            return

        if mode == AI_STORY_MODE:

            if source == result:

                raise RuntimeError(
                    "AI Story mode returned "
                    "the premise unchanged."
                )

            # No arbitrary length multiplier.
            # A concise but excellent story is valid.

            sentences = [
                value
                for value
                in re.split(
                    r"[.!?]+",
                    result,
                )
                if value.strip()
            ]

            if len(sentences) < 2:

                raise RuntimeError(
                    "AI Story mode did not produce "
                    "a complete narrative."
                )

            return

        if mode == EXPAND_USER_STORY_MODE:

            if source == result:

                raise RuntimeError(
                    "Expand Story mode returned "
                    "the supplied story unchanged."
                )

            source_words = (
                self._meaningful_tokens(
                    source
                )
            )

            if source_words:

                result_words = (
                    self._meaningful_tokens(
                        result
                    )
                )

                overlap = (
                    len(
                        source_words
                        & result_words
                    )
                    / max(
                        1,
                        len(source_words),
                    )
                )

                if overlap < 0.20:

                    raise RuntimeError(
                        "Expand Story mode changed "
                        "too much of the supplied story."
                    )

            sentences = [
                value
                for value
                in re.split(
                    r"[.!?]+",
                    result,
                )
                if value.strip()
            ]

            if len(sentences) < 3:

                raise RuntimeError(
                    "Expand Story mode did not "
                    "provide enough narrative development."
                )

            return

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    def _validate_shot_character_contract(
        self,
        shots: list[dict],
        characters: list[dict],
    ) -> None:

        if not characters:
            return

        allowed = {
            str(
                value.get(
                    "name",
                    "",
                )
            ).strip().lower()
            for value in characters
        }

        for shot in shots:

            for field in (
                "characters",
                "speaking_characters",
            ):

                for name in (
                    shot.get(
                        field,
                        [],
                    )
                    or []
                ):

                    if (
                        str(
                            name
                        )
                        .strip()
                        .lower()
                        not in allowed
                    ):

                        raise RuntimeError(
                            f"Shot {shot.get('shot_id', '')} "
                            f"contains unknown character "
                            f"'{name}'."
                        )

    @staticmethod
    def _sanitize_visual_language(
        value,
    ) -> dict:

        if not isinstance(value, dict):
            return {}

        fields = (
            "genre_tone",
            "color_palette",
            "lighting_philosophy",
            "camera_philosophy",
            "pacing",
        )

        return {
            field: str(
                value.get(
                    field,
                    "",
                )
                or ""
            ).strip()
            for field in fields
        }

    # ========================================================
    # GENERATE
    # ========================================================

    def generate(
        self,
        *,
        mode: str,
        user_input: str,
        base_plan: dict,
    ) -> dict:

        if not director_enabled():

            return {
                "enabled": False,
                "plan": deepcopy(
                    base_plan
                ),
                "director_notes": "",
            }

        if mode not in (
            self._MODE_LABELS
        ):

            raise ValueError(
                f"Unsupported story mode: {mode}"
            )

        self.load()

        if self._llama is None:

            raise RuntimeError(
                "Qwen director model failed to load."
            )

        temperature, top_p = (
            self._sampling_for_mode(
                mode
            )
        )

        # ----------------------------------------------------
        # PASS 1A: narrative
        # ----------------------------------------------------

        if mode == PRESERVE_USER_STORY_MODE:

            story = (
                self._normalize_story(
                    user_input
                )
            )

            story_plan = {
                "story":
                    story,
            }

            self._current_visual_language = {}

        else:

            story = ""

            story_system = (
                self._story_text_system(
                    mode
                )
            )

            story_user = (
                self._story_text_user(
                    mode,
                    user_input,
                )
            )

            try:

                story = (
                    self._chat_text(
                        story_system,
                        story_user,
                        minimum_completion=350,
                        temperature=temperature,
                        top_p=top_p,
                        call_name=(
                            "ai_story_text_pass"
                            if mode == AI_STORY_MODE
                            else "expand_story_text_pass"
                        ),
                        max_completion=1600,
                    )
                )

                self._validate_mode_output(
                    mode,
                    user_input,
                    story,
                )

            except RuntimeError as first_error:

                retry_user = (
                    story_user
                    + "\n\n"
                    "REPAIR REQUIRED.\n"
                    f"Previous validation failure: {first_error}\n"
                    "Write the complete narrative again. "
                    "Return ONLY the story prose. "
                    "Do not output JSON or commentary."
                )

                story = (
                    self._chat_text(
                        story_system,
                        retry_user,
                        minimum_completion=350,
                        temperature=min(
                            0.85,
                            max(
                                0.70,
                                temperature + 0.10,
                            ),
                        ),
                        top_p=0.92,
                        call_name=(
                            "ai_story_text_retry"
                            if mode == AI_STORY_MODE
                            else "expand_story_text_retry"
                        ),
                        max_completion=1600,
                    )
                )

                self._validate_mode_output(
                    mode,
                    user_input,
                    story,
                )

            story_plan = {
                "story":
                    story,
            }

        # ----------------------------------------------------
        # PASS 1B: production metadata
        # ----------------------------------------------------

        metadata_response = self._chat_json(
            self._metadata_director_system(
                mode
            ),
            self._metadata_director_user(
                mode,
                story,
            ),
            minimum_completion=600,
            temperature=(
                0.20
                if mode == PRESERVE_USER_STORY_MODE
                else temperature
            ),
            top_p=(
                0.82
                if mode == PRESERVE_USER_STORY_MODE
                else top_p
            ),
            call_name=(
                "preserve_metadata_pass"
                if mode == PRESERVE_USER_STORY_MODE
                else (
                    "ai_story_metadata_pass"
                    if mode == AI_STORY_MODE
                    else "expand_story_metadata_pass"
                )
            ),
            max_completion=2600,
            json_mode=True,
        )

        story_plan.update(
            metadata_response
        )

        story = (
            self._normalize_story(
                story_plan.get(
                    "story",
                    story,
                )
                or story
            )
        )

        director_notes = str(
            story_plan.get(
                "director_notes",
                "",
            )
            or ""
        ).strip()

        visual_language = (
            self._sanitize_visual_language(
                story_plan.get(
                    "visual_language",
                    {},
                )
            )
        )

        self._current_visual_language = (
            dict(
                visual_language
            )
        )

        # ----------------------------------------------------
        # CHARACTERS
        # ----------------------------------------------------

        characters = (
            self._sanitize_characters(
                story_plan.get(
                    "characters",
                    [],
                )
            )
        )

        if not characters:

            characters = (
                self._recover_characters(
                    mode,
                    story,
                )
            )

        # ----------------------------------------------------
        # SCENES
        # ----------------------------------------------------

        character_names = {
            str(
                character.get(
                    "name",
                    "",
                )
            ).strip().lower()
            for character
            in characters
            if str(
                character.get(
                    "name",
                    "",
                )
            ).strip()
        }

        scenes = (
            self._sanitize_scenes(
                story_plan.get(
                    "scenes",
                    [],
                ),
                character_names,
            )
        )

        if not scenes:

            scenes = (
                self._recover_scenes(
                    mode,
                    story,
                    characters,
                    character_names,
                )
            )

        if not scenes:

            fallback_characters, fallback_scenes = (
                self._build_deterministic_fallback(
                    story
                )
            )

            if not characters:

                characters = (
                    self._sanitize_characters(
                        fallback_characters
                    )
                )

                character_names = {
                    str(
                        character.get(
                            "name",
                            "",
                        )
                    ).strip().lower()
                    for character
                    in characters
                    if str(
                        character.get(
                            "name",
                            "",
                        )
                    ).strip()
                }

            scenes = (
                self._sanitize_scenes(
                    fallback_scenes,
                    character_names,
                )
            )

        if not scenes:

            raise RuntimeError(
                "Qwen director produced no usable scenes."
            )

        # ----------------------------------------------------
        # PASS 2: cinematography
        # ----------------------------------------------------

        all_shots: list[dict] = []

        for scene in scenes:

            shot_temperature, shot_top_p = (
                self._shot_sampling()
            )

            scene_shots: list[dict] = []

            shot_user = (
                self._shot_director_user(
                    story,
                    characters,
                    scene,
                    visual_language,
                )
            )

            try:

                shot_plan = self._chat_json(
                    self._shot_director_system(),
                    shot_user,
                    minimum_completion=250,
                    temperature=shot_temperature,
                    top_p=shot_top_p,
                    call_name=(
                        "shot_pass:"
                        + str(
                            scene.get(
                                "scene_id",
                                "unknown",
                            )
                        )
                    ),
                    max_completion=1000,
                    json_mode=False,
                )

                scene_shots = (
                    self._sanitize_shots(
                        shot_plan.get(
                            "shots",
                            [],
                        ),
                        scene,
                        character_names,
                    )
                )

            except Exception as first_shot_error:

                print(
                    "[QWEN]",
                    "shot_retry",
                    scene.get(
                        "scene_id",
                        "unknown",
                    ),
                    str(
                        first_shot_error
                    ),
                    flush=True,
                )

                retry_user = (
                    shot_user
                    + "\n\n"
                    "RETRY. Return exactly 2 shots. "
                    "Keep each field concise. "
                    "Return JSON only with the required keys."
                )

                try:

                    shot_plan = self._chat_json(
                        self._shot_director_system(),
                        retry_user,
                        minimum_completion=250,
                        temperature=0.72,
                        top_p=0.94,
                        call_name=(
                            "shot_retry:"
                            + str(
                                scene.get(
                                    "scene_id",
                                    "unknown",
                                )
                            )
                        ),
                        max_completion=900,
                        json_mode=False,
                    )

                    scene_shots = (
                        self._sanitize_shots(
                            shot_plan.get(
                                "shots",
                                [],
                            ),
                            scene,
                            character_names,
                        )
                    )

                except Exception:
                    scene_shots = []

            if len(scene_shots) < 2:

                print(
                    "[QWEN]",
                    "shot_fallback",
                    scene.get(
                        "scene_id",
                        "unknown",
                    ),
                    flush=True,
                )

                scene_shots = (
                    self._fallback_shots_for_scene(
                        scene,
                        len(
                            all_shots
                        ) + 1,
                    )
                )

            all_shots.extend(
                scene_shots
            )

        if not all_shots:

            raise RuntimeError(
                "No usable shots could be produced."
            )

        self._normalize_ids(
            scenes,
            all_shots,
        )

        for scene in scenes:

            scene[
                "shot_ids"
            ] = [
                shot[
                    "shot_id"
                ]
                for shot
                in all_shots
                if shot.get(
                    "scene_id"
                )
                == scene.get(
                    "scene_id"
                )
            ]

        self._validate_shot_character_contract(
            all_shots,
            characters,
        )

        return {
            "enabled": True,
            "plan": {
                "story":
                    story,

                "story_mode":
                    mode,

                "director_notes":
                    director_notes,

                "visual_language":
                    visual_language,

                "characters":
                    characters,

                "scenes":
                    scenes,

                "shots":
                    all_shots,
            },
            "director_notes":
                director_notes,
        }

    # ========================================================
    # MERGE
    # ========================================================

    def enrich_plan(
        self,
        *,
        mode: str,
        user_input: str,
        base_plan: dict,
    ) -> dict:

        result = self.generate(
            mode=mode,
            user_input=user_input,
            base_plan=base_plan,
        )

        if not result.get(
            "enabled",
            False,
        ):

            return deepcopy(
                base_plan
            )

        creative = (
            result.get(
                "plan",
                {},
            )
            or {}
        )

        merged = deepcopy(
            base_plan
        )

        if mode == PRESERVE_USER_STORY_MODE:

            merged[
                "story"
            ] = user_input.strip()

        else:

            merged[
                "story"
            ] = str(
                creative.get(
                    "story",
                    merged.get(
                        "story",
                        user_input,
                    ),
                )
                or merged.get(
                    "story",
                    user_input,
                )
            ).strip()

        merged[
            "story_mode"
        ] = mode

        merged[
            "director_notes"
        ] = str(
            creative.get(
                "director_notes",
                "",
            )
            or ""
        )

        creative_visual_language = (
            creative.get(
                "visual_language",
                {},
            )
            or {}
        )

        if creative_visual_language:

            merged[
                "visual_language"
            ] = deepcopy(
                creative_visual_language
            )

        creative_characters = (
            creative.get(
                "characters",
                [],
            )
            or []
        )

        if creative_characters:

            merged[
                "characters"
            ] = deepcopy(
                creative_characters
            )

        creative_scenes = (
            creative.get(
                "scenes",
                [],
            )
            or []
        )

        if creative_scenes:

            merged[
                "scenes"
            ] = deepcopy(
                creative_scenes
            )

        creative_shots = (
            creative.get(
                "shots",
                [],
            )
            or []
        )

        if creative_shots:

            merged[
                "shots"
            ] = deepcopy(
                creative_shots
            )

        return merged

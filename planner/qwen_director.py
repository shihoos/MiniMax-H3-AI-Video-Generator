from __future__ import annotations

import ctypes
import gc
import json
import os
import re
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
    """
    Local Qwen3-14B creative director.

    Pass 1:
        user story
            -> story
            -> characters
            -> scenes

    Pass 2:
        story + characters + scenes
            -> cinematic shots

    The deterministic ProductionPlanner remains the safe
    fallback when H3_DIRECTOR_ENABLED=0.

    This class is responsible for creative planning only.
    The ProductionOrchestrator is responsible for converting
    the creative plan into H3-ready references, identity locks,
    workflow settings and delivery configuration.
    """

    # These are prose/instruction/metadata tokens that should
    # NEVER be promoted into character names.
    #
    # Generic role descriptors such as "man" or "woman" are
    # intentionally NOT forbidden. They are legitimate story
    # characters when the story actually contains them.
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

    # Generic roles that are valid characters and must not be
    # removed by sanitization.
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

    def __init__(
        self,
        project_root: Path,
    ):
        self.project_root = Path(
            project_root
        )

        self._llama = None

        if director_enabled():
            self._model_path = (
                self._find_model()
            )
        else:
            self._model_path = None

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

        candidates = []

        if explicit:
            candidates.append(
                Path(
                    explicit
                )
            )

        candidates.extend(
            [
                self.project_root
                / "data"
                / "models"
                / DIRECTOR_MODEL_FILENAME,

                self.project_root
                / DIRECTOR_MODEL_FILENAME,

                DIRECTOR_KAGGLE_INPUT_ROOT
                / DIRECTOR_MODEL_FILENAME,
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

        seen = set()
        unique = []

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
    def model_path(self):
        return self._model_path

    @property
    def available(
        self,
    ) -> bool:

        return (
            director_enabled()
            and self._model_path is not None
            and self._model_path.is_file()
        )

    # ========================================================
    # CUDA / MODEL LIFECYCLE
    # ========================================================

    @staticmethod
    def _load_nvidia_cuda_libraries():

        import site

        site_roots = []

        try:
            site_roots.extend(
                Path(path)
                for path in site.getsitepackages()
                if path
            )
        except Exception:
            pass

        user_site = (
            site.getusersitepackages()
        )

        if user_site:
            site_roots.append(
                Path(
                    user_site
                )
            )

        cudart = []
        cublas = []

        for site_root in site_roots:

            root = (
                site_root
                / "nvidia"
            )

            if not root.is_dir():
                continue

            try:
                cudart.extend(
                    root.rglob(
                        "libcudart.so.13*"
                    )
                )

                cublas.extend(
                    root.rglob(
                        "libcublas.so.13*"
                    )
                )
            except OSError:
                continue

        cudart = [
            path
            for path in cudart
            if path.is_file()
        ]

        cublas = [
            path
            for path in cublas
            if path.is_file()
        ]

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
            for path in cublas
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
                str(cudart_lib),
                mode=ctypes.RTLD_GLOBAL,
            )

            ctypes.CDLL(
                str(cublas_lib),
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
            from llama_cpp import (
                Llama
            )
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
    ) -> tuple[int, int]:

        context = int(
            DIRECTOR_N_CTX
        )

        safety = 128

        prompt = (
            system_prompt
            + "\n\n"
            + user_prompt
        )

        prompt_tokens = (
            self._count_tokens(
                prompt
            )
        )

        available = (
            context
            - prompt_tokens
            - safety
        )

        if available < 512:
            raise RuntimeError(
                "Qwen director prompt is too large.\n"
                f"Prompt tokens: {prompt_tokens}\n"
                f"Context window: {context}\n"
                f"Available output tokens: {available}"
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
    # MODE CONTRACT
    # ========================================================

    def _mode_instruction(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:

            return """
AI STORY MODE.

The user's text is a premise or idea.

You are allowed and expected to:
- invent a coherent protagonist when needed;
- invent additional meaningful characters when needed;
- create motivations and stakes;
- develop the narrative;
- create intermediate events;
- create escalation;
- create a climax;
- create an ending;
- expand the premise substantially.

Do not merely paraphrase the user's text.

The result should feel like an actual developed cinematic
story rather than a conversion of the user's paragraphs into
scenes.
""".strip()

        if mode == EXPAND_USER_STORY_MODE:

            return """
EXPAND STORY MODE.

The user's text is an existing story and is authoritative
about its core facts.

You MAY enrich:
- character motivation;
- emotional beats;
- transitions;
- intermediate events;
- environment progression;
- cinematic pacing;
- tension;
- sensory detail;
- dialogue when appropriate;
- cause-and-effect connections.

You MUST preserve:
- the user's named characters;
- important entities;
- chronology;
- major events;
- core setting;
- intended outcome;
- explicit story facts.

Do NOT simply copy the user's paragraphs.
Do NOT replace the supplied story with an unrelated story.

Before generating, mentally separate:
1. immutable story facts;
2. details that can be enriched.

The output must visibly be a richer version of the supplied
story, not merely a shot annotation.
""".strip()

        if mode == PRESERVE_USER_STORY_MODE:

            return """
PRESERVE STORY MODE.

The user's story is immutable source material.

You MUST preserve the supplied story itself.

Do not:
- change events;
- change chronology;
- invent a different ending;
- replace named characters;
- remove important facts;
- add contradictory events;
- rewrite the meaning of the story.

You MAY only convert the supplied story into a cinematic
production structure by improving:
- scene boundaries;
- camera direction;
- lighting;
- visual detail;
- sound direction;
- pacing;
- continuity metadata.

The story returned in the JSON MUST equal the supplied story
after whitespace normalization.
""".strip()

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    # ========================================================
    # PASS 1 — STORY / CHARACTERS / SCENES
    # ========================================================

    def _story_director_system(
        self,
        mode: str,
    ) -> str:

        return f"""
You are the STORY DIRECTOR for a MiniMax H3 cinematic
video-production system.

{self._mode_instruction(mode)}

Your job in this pass is ONLY:

1. Develop/preserve the story.
2. Create the character bible.
3. Create meaningful cinematic scenes.

Do NOT create shots in this pass.

CHARACTER RULES:

- A character must be a real story entity.
- Never convert an ordinary word from prose into a character.
- Never use verbs as character names.
- Never use pronouns as character names.
- Never use metadata as character names.
- Never use camera terms as character names.
- Never use words such as:
  Treat, Develop, Clarify, Every, Above, Far, Tone, Visual,
  Camera, Scene, Shot, Lighting, Soundscape, Continuity,
  Story, Dialogue, Music.
- Generic roles such as man, woman, girl, boy, soldier,
  detective, scientist and robot are VALID characters when
  the story actually refers to them.
- Invented characters are allowed only in AI STORY and EXPAND
  modes and must have a meaningful narrative purpose.
- Do not create a character from sentence capitalization.

SCENE RULES:

- Do not mechanically map one paragraph to one scene.
- Merge related material into coherent cinematic beats.
- Do not create scenes for "Tone", "Visual priority", or similar
  metadata.
- Prefer roughly 4–8 meaningful scenes.
- Every scene must advance the story, establish a distinct
  environment, introduce a meaningful action/beat, or build
  the escalation.
- Preserve continuity between scenes.

OUTPUT:

Return JSON only.

{{
  "story": "string",
  "director_notes": "string",
  "characters": [],
  "scenes": []
}}

Each character MUST contain:

name
role
description
personality
appearance
clothing
distinctive_features
character_state
continuity_rules

Each scene MUST contain:

scene_id
order
location
time_of_day
weather
atmosphere
description
mood
lighting
environment_details
key_props
scene_objective
characters
story_summary
continuity_notes
""".strip()

    def _story_director_user(
        self,
        mode: str,
        story: str,
    ) -> str:

        return json.dumps(
            {
                "mode": mode,
                "user_story": story,
                "input_policy": (
                    "The user_story is complete creative source material. "
                    "Read and use all of it. Do not filter, delete, or "
                    "ignore tone statements, visual priorities, camera "
                    "suggestions, sound ideas, or other creative guidance. "
                    "You may interpret, improve, expand, combine, or "
                    "reorganize those ideas according to the selected "
                    "story mode. Character-name validation happens only "
                    "after generation and must never be applied to the "
                    "input text."
                ),
                "task": (
                    "Create the story/character/scene plan "
                    "according to the selected mode."
                ),
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    # ========================================================
    # PASS 2 — CINEMATIC SHOTS
    # ========================================================

    def _shot_director_system(
        self,
    ) -> str:

        return """
You are the CINEMATIC SHOT DIRECTOR for MiniMax H3.

You receive:
- the final story;
- the authoritative character bible;
- the authoritative scene plan.

Create the shot plan.

CHARACTER RULES:

- Do NOT create new characters.
- Use ONLY character names supplied in the character bible.
- Never turn prose words into characters.
- Never create shots whose characters are not in the bible.

SHOT RULES:

- Do not make one shot per scene by default.
- A meaningful scene normally requires 2–4 shots.
- A complex action scene may require more.
- A simple transition scene may require one.
- Camera choices must serve the narrative.
- Build visual escalation.
- Preserve identity.
- Preserve clothing.
- Preserve story-state.
- Preserve environmental continuity.
- Include establishing, wide, tracking, medium, close,
  reveal/detail and low/high-angle shots when appropriate.
- Use camera variety intentionally.
- Dialogue should only exist when justified by the story.
- Soundscape should correspond to the environment and action.
- Music should support the scene rather than replace sound effects.

OUTPUT JSON ONLY:

{
  "shots": []
}

Each shot MUST contain:

shot_id
scene_id
order
duration_seconds
characters
location
action
camera_shot
camera_movement
lighting
mood
visual_prompt
retention_analysis
detailed_description
overall_soundscape
non_diegetic_music
negative_prompt
continuity_notes
speaking_characters
speech_text
""".strip()

    def _shot_director_user(
        self,
        story: str,
        characters: list[dict],
        scenes: list[dict],
    ) -> str:

        return json.dumps(
            {
                "story": story,
                "characters": characters,
                "scenes": scenes,
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    # ========================================================
    # JSON
    # ========================================================

    @staticmethod
    def _extract_json(
        text: str,
    ) -> dict:

        text = str(
            text or ""
        ).strip()

        text = re.sub(
            r"^```(?:json)?",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        text = re.sub(
            r"```$",
            "",
            text,
        ).strip()

        start = text.find(
            "{"
        )

        end = text.rfind(
            "}"
        )

        if start < 0 or end <= start:
            raise RuntimeError(
                "Qwen director did not return JSON."
            )

        try:

            value = json.loads(
                text[
                    start:end + 1
                ]
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Qwen director returned invalid JSON:\n"
                f"{exc}"
            ) from exc

        if not isinstance(
            value,
            dict,
        ):
            raise RuntimeError(
                "Qwen director output must be a JSON object."
            )

        return value

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:

        prompt_tokens, max_tokens = (
            self._available_output_tokens(
                system_prompt,
                user_prompt,
            )
        )

        response = (
            self._llama
            .create_chat_completion(
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
                temperature=(
                    DIRECTOR_TEMPERATURE
                ),
                top_p=(
                    DIRECTOR_TOP_P
                ),
                max_tokens=max_tokens,
                response_format={
                    "type": "json_object"
                },
            )
        )

        content = (
            response[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]
        )

        if not content:
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        return self._extract_json(
            content
        )

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

        # Generic role descriptors are explicitly valid.
        if lowered in self.VALID_GENERIC_ROLES:
            return True

        if lowered in self.FORBIDDEN_CHARACTER_NAMES:
            return False

        # Reject long instruction-like strings.
        if len(value.split()) > 4:
            return False

        # Reject obvious metadata punctuation.
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

    def _sanitize_characters(
        self,
        characters,
    ) -> list[dict]:

        result = []
        seen = set()

        for value in (
            characters
            or []
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

            result.append(
                {
                    "name": name,
                    "role": str(
                        value.get(
                            "role",
                            "story character",
                        )
                        or "story character"
                    ),
                    "description": str(
                        value.get(
                            "description",
                            "",
                        )
                        or ""
                    ),
                    "personality": str(
                        value.get(
                            "personality",
                            "",
                        )
                        or ""
                    ),
                    "appearance": dict(
                        value.get(
                            "appearance",
                            {},
                        )
                        or {}
                    ),
                    "clothing": dict(
                        value.get(
                            "clothing",
                            {},
                        )
                        or {}
                    ),
                    "distinctive_features": list(
                        value.get(
                            "distinctive_features",
                            [],
                        )
                        or []
                    ),
                    "character_state": dict(
                        value.get(
                            "character_state",
                            {},
                        )
                        or {}
                    ),
                    "continuity_rules": list(
                        value.get(
                            "continuity_rules",
                            [],
                        )
                        or []
                    ),
                }
            )

        return result

    # ========================================================
    # SCENE SANITIZATION
    # ========================================================

    def _sanitize_scenes(
        self,
        scenes,
        character_names: set[str],
    ) -> list[dict]:

        result = []

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

            lower = description.lower()

            if (
                lower.startswith(
                    "tone:"
                )
                or lower.startswith(
                    "visual priority:"
                )
                or lower.startswith(
                    "visual priorities:"
                )
            ):
                continue

            scene_id = str(
                value.get(
                    "scene_id",
                    f"scene_{index:03d}",
                )
                or f"scene_{index:03d}"
            ).strip()

            selected_characters = []

            for name in (
                value.get(
                    "characters",
                    [],
                )
                or []
            ):

                name = str(
                    name
                ).strip()

                if (
                    name.lower()
                    in character_names
                ):
                    selected_characters.append(
                        name
                    )

            result.append(
                {
                    "scene_id": scene_id,
                    "order": len(result) + 1,
                    "location": str(
                        value.get(
                            "location",
                            "",
                        )
                        or ""
                    ),
                    "time_of_day": str(
                        value.get(
                            "time_of_day",
                            "",
                        )
                        or ""
                    ),
                    "weather": str(
                        value.get(
                            "weather",
                            "",
                        )
                        or ""
                    ),
                    "atmosphere": str(
                        value.get(
                            "atmosphere",
                            "",
                        )
                        or ""
                    ),
                    "description": description,
                    "mood": str(
                        value.get(
                            "mood",
                            "",
                        )
                        or ""
                    ),
                    "lighting": str(
                        value.get(
                            "lighting",
                            "",
                        )
                        or ""
                    ),
                    "environment_details": list(
                        value.get(
                            "environment_details",
                            [],
                        )
                        or []
                    ),
                    "key_props": list(
                        value.get(
                            "key_props",
                            [],
                        )
                        or []
                    ),
                    "scene_objective": str(
                        value.get(
                            "scene_objective",
                            "",
                        )
                        or ""
                    ),
                    "characters": selected_characters,
                    "story_summary": str(
                        value.get(
                            "story_summary",
                            description,
                        )
                        or description
                    ),
                    "continuity_notes": str(
                        value.get(
                            "continuity_notes",
                            "",
                        )
                        or ""
                    ),
                    "shot_ids": [],
                }
            )

        return result

    # ========================================================
    # SHOT SANITIZATION
    # ========================================================

    def _sanitize_shots(
        self,
        shots,
        scenes: list[dict],
        character_names: set[str],
    ) -> list[dict]:

        scene_ids = {
            scene[
                "scene_id"
            ]
            for scene in scenes
        }

        result = []

        for index, value in enumerate(
            shots or [],
            start=1,
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            scene_id = str(
                value.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()

            if scene_id not in scene_ids:
                continue

            selected_characters = []

            for name in (
                value.get(
                    "characters",
                    [],
                )
                or []
            ):

                name = str(
                    name
                ).strip()

                if (
                    name.lower()
                    in character_names
                ):
                    selected_characters.append(
                        name
                    )

            speaking = []

            for name in (
                value.get(
                    "speaking_characters",
                    [],
                )
                or []
            ):

                name = str(
                    name
                ).strip()

                if (
                    name.lower()
                    in character_names
                ):
                    speaking.append(
                        name
                    )

            try:
                duration = float(
                    value.get(
                        "duration_seconds",
                        5.2,
                    )
                    or 5.2
                )
            except (TypeError, ValueError):
                duration = 5.2

            result.append(
                {
                    "shot_id": str(
                        value.get(
                            "shot_id",
                            f"shot_{index:03d}",
                        )
                        or f"shot_{index:03d}"
                    ).strip(),
                    "scene_id": scene_id,
                    "order": len(result) + 1,
                    "duration_seconds": duration,
                    "characters": selected_characters,
                    "location": str(
                        value.get(
                            "location",
                            "",
                        )
                        or ""
                    ),
                    "action": str(
                        value.get(
                            "action",
                            "",
                        )
                        or ""
                    ),
                    "camera_shot": str(
                        value.get(
                            "camera_shot",
                            "",
                        )
                        or ""
                    ),
                    "camera_movement": str(
                        value.get(
                            "camera_movement",
                            "",
                        )
                        or ""
                    ),
                    "lighting": str(
                        value.get(
                            "lighting",
                            "",
                        )
                        or ""
                    ),
                    "mood": str(
                        value.get(
                            "mood",
                            "",
                        )
                        or ""
                    ),
                    "visual_prompt": str(
                        value.get(
                            "visual_prompt",
                            "",
                        )
                        or ""
                    ),
                    "retention_analysis": str(
                        value.get(
                            "retention_analysis",
                            "",
                        )
                        or ""
                    ),
                    "detailed_description": str(
                        value.get(
                            "detailed_description",
                            "",
                        )
                        or ""
                    ),
                    "overall_soundscape": str(
                        value.get(
                            "overall_soundscape",
                            "",
                        )
                        or ""
                    ),
                    "non_diegetic_music": str(
                        value.get(
                            "non_diegetic_music",
                            "",
                        )
                        or ""
                    ),
                    "negative_prompt": str(
                        value.get(
                            "negative_prompt",
                            "",
                        )
                        or ""
                    ),
                    "continuity_notes": str(
                        value.get(
                            "continuity_notes",
                            "",
                        )
                        or ""
                    ),
                    "speaking_characters": speaking,
                    "speech_text": str(
                        value.get(
                            "speech_text",
                            "",
                        )
                        or ""
                    ),
                }
            )

        return result

    # ========================================================
    # SHOT COVERAGE
    # ========================================================

    def _ensure_scene_shot_coverage(
        self,
        scenes: list[dict],
        shots: list[dict],
    ) -> list[dict]:

        by_scene = {}

        for shot in shots:
            by_scene.setdefault(
                shot[
                    "scene_id"
                ],
                0,
            )

            by_scene[
                shot[
                    "scene_id"
                ]
            ] += 1

        for scene in scenes:

            scene_id = scene[
                "scene_id"
            ]

            if by_scene.get(
                scene_id,
                0,
            ) > 0:
                continue

            shots.append(
                {
                    "shot_id": (
                        f"shot_{len(shots) + 1:03d}"
                    ),
                    "scene_id": scene_id,
                    "order": len(shots) + 1,
                    "duration_seconds": 5.2,
                    "characters": list(
                        scene.get(
                            "characters",
                            [],
                        )
                        or []
                    ),
                    "location": scene.get(
                        "location",
                        "",
                    ),
                    "action": (
                        scene.get(
                            "scene_objective",
                            "",
                        )
                        or scene.get(
                            "description",
                            "",
                        )
                    ),
                    "camera_shot": (
                        "wide establishing shot"
                    ),
                    "camera_movement": (
                        "slow controlled cinematic movement"
                    ),
                    "lighting": scene.get(
                        "lighting",
                        "",
                    ),
                    "mood": scene.get(
                        "mood",
                        "",
                    ),
                    "visual_prompt": scene.get(
                        "description",
                        "",
                    ),
                    "retention_analysis": (
                        "Maintain story and visual continuity."
                    ),
                    "detailed_description": scene.get(
                        "description",
                        "",
                    ),
                    "overall_soundscape": (
                        "Natural cinematic environmental "
                        "sound appropriate to the scene."
                    ),
                    "non_diegetic_music": (
                        "Subtle cinematic score when appropriate."
                    ),
                    "negative_prompt": (
                        "identity drift, duplicate character, "
                        "face deformation, inconsistent clothing"
                    ),
                    "continuity_notes": scene.get(
                        "continuity_notes",
                        "",
                    ),
                    "speaking_characters": [],
                    "speech_text": "",
                }
            )

        return shots

    # ========================================================
    # MODE OUTPUT VALIDATION
    # ========================================================

    @staticmethod
    def _normalize_story_for_comparison(
        text: str,
    ) -> str:

        return re.sub(
            r"\s+",
            " ",
            str(
                text or ""
            ).strip(),
        )

    def _validate_mode_output(
        self,
        mode: str,
        user_input: str,
        story: str,
    ) -> None:

        if mode == PRESERVE_USER_STORY_MODE:

            source = (
                self._normalize_story_for_comparison(
                    user_input
                )
            )

            result = (
                self._normalize_story_for_comparison(
                    story
                )
            )

            if source != result:
                raise RuntimeError(
                    "Preserve Story mode changed the supplied story."
                )

        elif mode == EXPAND_USER_STORY_MODE:

            source = (
                self._normalize_story_for_comparison(
                    user_input
                )
            )

            result = (
                self._normalize_story_for_comparison(
                    story
                )
            )

            if not result:
                raise RuntimeError(
                    "Expand Story mode returned an empty story."
                )

            if len(result) < max(
                80,
                int(
                    len(source) * 0.75
                ),
            ):
                raise RuntimeError(
                    "Expand Story mode produced a story that "
                    "is unexpectedly shorter than the supplied "
                    "story."
                )

    # ========================================================
    # GENERATION
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

        self.load()

        if self._llama is None:
            raise RuntimeError(
                "Qwen director model failed to load."
            )

        # ----------------------------------------------------
        # PASS 1
        # ----------------------------------------------------

        story_plan = self._chat_json(
            self._story_director_system(
                mode
            ),
            self._story_director_user(
                mode,
                user_input,
            ),
        )

        characters = (
            self._sanitize_characters(
                story_plan.get(
                    "characters",
                    [],
                )
            )
        )

        character_names = {
            character[
                "name"
            ].lower()
            for character
            in characters
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

        story = str(
            story_plan.get(
                "story",
                user_input,
            )
            or user_input
        ).strip()

        self._validate_mode_output(
            mode,
            user_input,
            story,
        )

        director_notes = str(
            story_plan.get(
                "director_notes",
                "",
            )
            or ""
        ).strip()

        # ----------------------------------------------------
        # Fallback only if Qwen failed to create a character.
        # This never allows prose tokens to become characters.
        # ----------------------------------------------------

        if not characters:

            fallback = []

            for value in (
                base_plan.get(
                    "characters",
                    [],
                )
                or []
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

                if not name:
                    continue

                if not self._valid_character_name(
                    name
                ):
                    continue

                fallback.append(
                    deepcopy(
                        value
                    )
                )

            characters = fallback

            character_names = {
                character[
                    "name"
                ].lower()
                for character
                in characters
            }

        # ----------------------------------------------------
        # PASS 2
        # ----------------------------------------------------

        shot_plan = self._chat_json(
            self._shot_director_system(),
            self._shot_director_user(
                story,
                characters,
                scenes,
            ),
        )

        shots = (
            self._sanitize_shots(
                shot_plan.get(
                    "shots",
                    [],
                ),
                scenes,
                character_names,
            )
        )

        shots = (
            self._ensure_scene_shot_coverage(
                scenes,
                shots,
            )
        )

        # Re-number after coverage correction.
        for index, shot in enumerate(
            shots,
            start=1,
        ):
            shot[
                "order"
            ] = index

        for index, scene in enumerate(
            scenes,
            start=1,
        ):
            scene[
                "order"
            ] = index

        return {
            "enabled": True,
            "plan": {
                "story": story,
                "story_mode": mode,
                "director_notes": director_notes,
                "characters": characters,
                "scenes": scenes,
                "shots": shots,
            },
            "director_notes": director_notes,
        }

    # ========================================================
    # MERGE
    # ========================================================

    @staticmethod
    def _merge_dict(
        original: dict,
        updated: dict,
        allowed: set[str],
    ) -> dict:

        result = dict(
            original
        )

        for key in allowed:

            if key not in updated:
                continue

            value = updated[
                key
            ]

            if value is None:
                continue

            result[
                key
            ] = value

        return result

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

        if not result[
            "enabled"
        ]:
            return base_plan

        creative = result[
            "plan"
        ]

        merged = deepcopy(
            base_plan
        )

        # ----------------------------------------------------
        # Story
        # ----------------------------------------------------

        if mode == PRESERVE_USER_STORY_MODE:

            merged[
                "story"
            ] = str(
                user_input
            ).strip()

        else:

            merged[
                "story"
            ] = str(
                creative.get(
                    "story",
                    user_input,
                )
                or user_input
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
        ).strip()

        # ----------------------------------------------------
        # Characters
        # ----------------------------------------------------

        base_characters = list(
            merged.get(
                "characters",
                [],
            )
            or []
        )

        existing_by_name = {
            str(
                character.get(
                    "name",
                    "",
                )
            ).strip().lower():
                character
            for character
            in base_characters
            if str(
                character.get(
                    "name",
                    "",
                )
            ).strip()
        }

        characters = []

        for index, spec in enumerate(
            creative.get(
                "characters",
                [],
            ),
            start=1,
        ):

            if not isinstance(
                spec,
                dict,
            ):
                continue

            name = str(
                spec.get(
                    "name",
                    "",
                )
                or ""
            ).strip()

            if not self._valid_character_name(
                name
            ):
                continue

            existing = (
                existing_by_name.get(
                    name.lower()
                )
            )

            if existing is None:

                existing = {
                    "character_id": (
                        f"character_{index:03d}"
                    ),
                    "name": name,
                    "role": "story character",
                    "description": "",
                    "personality": "",
                    "appearance": {},
                    "clothing": {},
                    "distinctive_features": [],
                    "character_state": {},
                    "continuity_rules": [],
                    "reference_mode": (
                        "story_generated"
                    ),
                    "reference_paths": [],
                    "reference_video_paths": [],
                    "reference_audio_paths": [],
                    "reference_path": None,
                    "reference_video_path": None,
                    "reference_audio_path": None,
                }

            characters.append(
                self._merge_dict(
                    existing,
                    spec,
                    {
                        "name",
                        "role",
                        "description",
                        "personality",
                        "appearance",
                        "clothing",
                        "distinctive_features",
                        "character_state",
                        "continuity_rules",
                    },
                )
            )

        if characters:
            merged[
                "characters"
            ] = characters

        # ----------------------------------------------------
        # Scenes
        # ----------------------------------------------------

        base_scenes = list(
            merged.get(
                "scenes",
                [],
            )
            or []
        )

        base_scene_map = {
            str(
                scene.get(
                    "scene_id",
                    "",
                )
            ): scene
            for scene
            in base_scenes
        }

        scenes = []

        for index, spec in enumerate(
            creative.get(
                "scenes",
                [],
            ),
            start=1,
        ):

            if not isinstance(
                spec,
                dict,
            ):
                continue

            scene_id = str(
                spec.get(
                    "scene_id",
                    f"scene_{index:03d}",
                )
                or f"scene_{index:03d}"
            ).strip()

            existing = base_scene_map.get(
                scene_id,
                {
                    "scene_id": scene_id,
                    "order": index,
                    "shot_ids": [],
                },
            )

            scenes.append(
                self._merge_dict(
                    existing,
                    spec,
                    {
                        "scene_id",
                        "order",
                        "location",
                        "time_of_day",
                        "weather",
                        "atmosphere",
                        "description",
                        "mood",
                        "lighting",
                        "environment_details",
                        "key_props",
                        "scene_objective",
                        "characters",
                        "story_summary",
                        "continuity_notes",
                    },
                )
            )

        if scenes:
            merged[
                "scenes"
            ] = scenes

        # ----------------------------------------------------
        # Shots
        # ----------------------------------------------------

        base_shots = list(
            merged.get(
                "shots",
                [],
            )
            or []
        )

        base_shot_map = {
            str(
                shot.get(
                    "shot_id",
                    "",
                )
            ): shot
            for shot
            in base_shots
        }

        shots = []

        for index, spec in enumerate(
            creative.get(
                "shots",
                [],
            ),
            start=1,
        ):

            if not isinstance(
                spec,
                dict,
            ):
                continue

            shot_id = str(
                spec.get(
                    "shot_id",
                    "",
                )
                or f"shot_{index:03d}"
            ).strip()

            existing = base_shot_map.get(
                shot_id,
                {
                    "shot_id": shot_id,
                    "scene_id": spec.get(
                        "scene_id",
                        "",
                    ),
                    "order": index,
                    "duration_seconds": 5.2,
                },
            )

            shots.append(
                self._merge_dict(
                    existing,
                    spec,
                    {
                        "shot_id",
                        "scene_id",
                        "order",
                        "duration_seconds",
                        "characters",
                        "location",
                        "action",
                        "camera_shot",
                        "camera_movement",
                        "lighting",
                        "mood",
                        "visual_prompt",
                        "retention_analysis",
                        "detailed_description",
                        "overall_soundscape",
                        "non_diegetic_music",
                        "negative_prompt",
                        "continuity_notes",
                        "speaking_characters",
                        "speech_text",
                    },
                )
            )

        if shots:
            merged[
                "shots"
            ] = shots

        return merged

    def close(
        self,
    ):
        self.unload()

    def __enter__(
        self,
    ):
        self.load()
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        self.unload()

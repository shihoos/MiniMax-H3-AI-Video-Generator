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
        "man",
        "woman",
        "girl",
        "boy",
        "person",
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

            if root.exists():

                try:

                    candidates.extend(
                        root.rglob(
                            DIRECTOR_MODEL_FILENAME
                        )
                    )

                except OSError:
                    pass

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

        matching = [
            path
            for path in cublas
            if path.parent
            == cudart_lib.parent
        ]

        cublas_lib = (
            matching[0]
            if matching
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

        if available < 256:

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
    # MODE PROMPTS
    # ========================================================

    def _mode_instruction(
        self,
        mode: str,
    ) -> str:

        if mode == PRESERVE_USER_STORY_MODE:

            return (
                "PRESERVE MODE. The supplied story is authoritative. "
                "Do not change its facts, chronology, named entities, "
                "events, or requested ending. Improve only the cinematic "
                "development and production planning."
            )

        if mode == EXPAND_USER_STORY_MODE:

            return (
                "EXPAND MODE. Preserve every important fact and intent "
                "from the supplied story, but meaningfully enrich it. "
                "Add character motivation, transitions, intermediate "
                "events, environmental progression, emotional beats, "
                "cinematic staging, and useful visual details. "
                "Do not merely repeat or annotate the supplied prose."
            )

        return (
            "AI STORY MODE. Treat the user's input as a premise, not "
            "as a finished screenplay. Develop an actual coherent "
            "cinematic story with a beginning, progression, escalation, "
            "climax and ending. Create characters only when they are "
            "meaningful to the story."
        )

    # ========================================================
    # STAGE 1 — STORY / CHARACTER / SCENE DIRECTOR
    # ========================================================

    def _story_director_system(
        self,
        mode: str,
    ) -> str:

        return f"""
You are the STORY DIRECTOR for a MiniMax H3 cinematic
video generation system.

{self._mode_instruction(mode)}

Your job in this pass is ONLY:

1. Develop or preserve the story.
2. Create the real character bible.
3. Divide the story into meaningful cinematic scenes.

Do NOT generate shots in this pass.

Do NOT create a scene merely because the user wrote a
"Tone:" line, "Visual priority:" line, or camera instruction.
Treat those as global creative guidance.

IMPORTANT CHARACTER RULES:

- A character must be a real entity in the story.
- Never turn ordinary prose words into character names.
- Never use verbs, pronouns, adjectives, instructions,
  camera terms, scene terms, or metadata as character names.
- Never use words such as:
  Treat, Develop, Clarify, Every, Above, Far, Tone, Visual,
  Camera, Scene, Shot, Lighting, Soundscape, Continuity,
  Story, He, She, His, Her, It.
- Generic role descriptors such as "man", "woman", "soldier",
  "detective" are valid characters when the story refers to
  them as actual entities.
- Invented names are allowed in AI STORY and EXPAND modes only
  when Qwen deliberately creates a meaningful character.
- Every invented character must have a narrative purpose.
- Do not create characters from sentence-initial capitalization.

SCENE RULES:

- Do not create one scene per paragraph mechanically.
- Merge related prose into coherent cinematic scenes.
- Do not create metadata scenes for tone or visual priority.
- Target roughly 4–8 meaningful scenes unless the story truly
  requires another count.
- Each scene must represent a distinct narrative, environmental,
  emotional, or action beat.
- Maintain continuity between scenes.

Return JSON only.

The JSON MUST have exactly:

{{
  "story": "string",
  "director_notes": "string",
  "characters": [],
  "scenes": []
}}

Character objects MUST contain:

name,
role,
description,
personality,
appearance,
clothing,
distinctive_features,
character_state,
continuity_rules

Scene objects MUST contain:

scene_id,
order,
location,
time_of_day,
weather,
atmosphere,
description,
mood,
lighting,
environment_details,
key_props,
scene_objective,
characters,
story_summary,
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
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    # ========================================================
    # STAGE 2 — SHOT DIRECTOR
    # ========================================================

    def _shot_director_system(
        self,
    ) -> str:

        return """
You are the CINEMATIC SHOT DIRECTOR for a MiniMax H3
production.

You are given an approved story, character bible and scene plan.

Create the shot plan.

IMPORTANT:

- Do not create characters.
- Only use character names supplied in the character bible.
- Never invent character names in this pass.
- Never use prose words as character names.
- Do not create shots for metadata such as "Tone" or
  "Visual priority".
- Every meaningful scene should normally receive 2–4 shots.
- Short transition scenes may receive 1 shot only when justified.
- Vary camera language intentionally.
- Use establishing, tracking, close, medium, over-the-shoulder,
  low-angle, high-angle, reveal and detail shots where appropriate.
- Camera movement must serve the action.
- Preserve character identity and story-state continuity.
- Preserve environmental continuity.
- Build escalation toward the climax.
- Write detailed visual prompts suitable for H3.
- Write soundscape, dialogue and music intent.
- Do not output explanations.

Return JSON only:

{
  "shots": []
}

Each shot MUST contain:

shot_id,
scene_id,
order,
duration_seconds,
characters,
location,
action,
camera_shot,
camera_movement,
lighting,
mood,
visual_prompt,
retention_analysis,
detailed_description,
overall_soundscape,
non_diegetic_music,
negative_prompt,
continuity_notes,
speaking_characters,
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
                "Qwen director output must be an object."
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

        if lowered in (
            self.FORBIDDEN_CHARACTER_NAMES
        ):
            return False

        words = lowered.split()

        if len(words) <= 3:

            if all(
                word
                in self.FORBIDDEN_CHARACTER_NAMES
                for word in words
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

            clean = {
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

            result.append(
                clean
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
            scenes
            or [],
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

            lower_description = (
                description.lower()
            )

            # Metadata-only scenes must never become scenes.
            if (
                lower_description.startswith(
                    "tone:"
                )
                or lower_description.startswith(
                    "visual priority:"
                )
                or lower_description.startswith(
                    "visual priorities:"
                )
            ):
                continue

            scene_id = str(
                value.get(
                    "scene_id",
                    f"scene_{index:03d}",
                )
            ).strip()

            valid_characters = []

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
                    valid_characters.append(
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
                    "characters": valid_characters,
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
            for scene
            in scenes
        }

        result = []

        for index, value in enumerate(
            shots
            or [],
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
            ).strip()

            if scene_id not in scene_ids:
                continue

            selected = []

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
                    selected.append(
                        name
                    )

            result.append(
                {
                    "shot_id": (
                        str(
                            value.get(
                                "shot_id",
                                f"shot_{index:03d}",
                            )
                        )
                        .strip()
                    ),
                    "scene_id": scene_id,
                    "order": len(result) + 1,
                    "duration_seconds": float(
                        value.get(
                            "duration_seconds",
                            5.2,
                        )
                        or 5.2
                    ),
                    "characters": selected,
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
                    "speaking_characters": [
                        str(name)
                        for name
                        in (
                            value.get(
                                "speaking_characters",
                                [],
                            )
                            or []
                        )
                        if str(
                            name
                        ).strip().lower()
                        in character_names
                    ],
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
    # DIRECTOR GENERATION
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

        story_prompt = (
            self._story_director_user(
                mode,
                user_input,
            )
        )

        story_plan = self._chat_json(
            self._story_director_system(
                mode
            ),
            story_prompt,
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

        director_notes = str(
            story_plan.get(
                "director_notes",
                "",
            )
            or ""
        ).strip()

        # Preserve mode must retain the supplied story exactly.
        if mode == PRESERVE_USER_STORY_MODE:

            story = str(
                user_input
            ).strip()

        # If Qwen dropped every character, use the deterministic
        # base planner's character set only as a safe fallback.
        if not characters:

            fallback_characters = []

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

                if (
                    name.lower()
                    in self.FORBIDDEN_CHARACTER_NAMES
                ):
                    continue

                fallback_characters.append(
                    deepcopy(
                        value
                    )
                )

            characters = (
                fallback_characters
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
                    scenes,
                    character_names,
                )
            )

        # ----------------------------------------------------
        # PASS 2 — SHOT DIRECTOR
        # ----------------------------------------------------

        shot_prompt = (
            self._shot_director_user(
                story,
                characters,
                scenes,
            )
        )

        shot_plan = self._chat_json(
            self._shot_director_system(),
            shot_prompt,
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

        # ----------------------------------------------------
        # Continuity defaults
        # ----------------------------------------------------

        for index, scene in enumerate(
            scenes,
            start=1,
        ):

            scene[
                "order"
            ] = index

        for index, shot in enumerate(
            shots,
            start=1,
        ):

            shot[
                "order"
            ] = index

        # Keep at least one sensible shot for a scene if the
        # shot director returned none for it.
        existing_scene_ids = {
            shot[
                "scene_id"
            ]
            for shot
            in shots
        }

        for scene in scenes:

            if (
                scene[
                    "scene_id"
                ]
                in existing_scene_ids
            ):
                continue

            shots.append(
                {
                    "shot_id": (
                        f"shot_{len(shots) + 1:03d}"
                    ),
                    "scene_id": (
                        scene[
                            "scene_id"
                        ]
                    ),
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
                    "action": scene.get(
                        "scene_objective",
                        scene.get(
                            "description",
                            "",
                        ),
                    ),
                    "camera_shot": (
                        "wide establishing shot"
                    ),
                    "camera_movement": (
                        "slow controlled movement"
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
                        "Maintain cinematic continuity."
                    ),
                    "detailed_description": (
                        scene.get(
                            "description",
                            "",
                        )
                    ),
                    "overall_soundscape": (
                        "Generate appropriate native H3 "
                        "environmental audio."
                    ),
                    "non_diegetic_music": (
                        "Subtle cinematic score when "
                        "appropriate."
                    ),
                    "negative_prompt": (
                        "identity drift, face deformation, "
                        "duplicate person, inconsistent clothing"
                    ),
                    "continuity_notes": scene.get(
                        "continuity_notes",
                        "",
                    ),
                    "speaking_characters": [],
                    "speech_text": "",
                }
            )

        return {
            "enabled": True,
            "plan": {
                "story": story,
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

            generated_story = str(
                creative.get(
                    "story",
                    user_input,
                )
                or user_input
            ).strip()

            if generated_story:

                merged[
                    "story"
                ] = generated_story

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

        by_name = {
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

        for spec in creative.get(
            "characters",
            [],
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

            if not name:
                continue

            if (
                name.lower()
                in self.FORBIDDEN_CHARACTER_NAMES
            ):
                continue

            existing = by_name.get(
                name.lower()
            )

            if existing is None:

                existing = {
                    "character_id": (
                        f"character_"
                        f"{len(base_characters) + len(characters) + 1:03d}"
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

            updated = self._merge_dict(
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

            characters.append(
                updated
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
            ).strip()

            existing = base_scene_map.get(
                scene_id,
                {
                    "scene_id":
                        scene_id,
                    "order":
                        index,
                    "shot_ids":
                        [],
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
                or ""
            ).strip()

            existing = (
                base_shot_map.get(
                    shot_id
                )
                if shot_id
                else None
            )

            if existing is None:

                existing = {
                    "shot_id":
                        shot_id
                        or
                        f"shot_{index:03d}",
                    "scene_id":
                        spec.get(
                            "scene_id",
                            "",
                        ),
                    "order":
                        index,
                    "duration_seconds":
                        5.2,
                }

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

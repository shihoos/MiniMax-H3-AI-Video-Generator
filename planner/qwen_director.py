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

    Qwen provides creative enrichment.

    ProductionPlanner remains the deterministic safety layer.
    If Qwen omits or corrupts a structural section, the base
    planner data is retained instead of failing the production.
    """

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
            Path(project_root)
            .resolve()
        )

        self._llama = None

        self._model_path = (
            self._find_model()
            if director_enabled()
            else None
        )

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
    # CUDA / MODEL LIFECYCLE
    # ========================================================

    @staticmethod
    def _load_nvidia_cuda_libraries() -> None:

        import site

        site_roots: list[Path] = []

        try:
            site_roots.extend(
                Path(path)
                for path in site.getsitepackages()
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
            for path in cublas
            if path.parent == cudart_lib.parent
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
                "Qwen director prompt is too large for "
                f"the {context}-token context window.\n"
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
    # MODE CONTRACT
    # ========================================================

    def _mode_instruction(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:

            return """
AI STORY MODE.

The user supplied a premise or idea.

Develop it into a coherent cinematic story.

You may invent:
- protagonist;
- supporting characters;
- motivations;
- stakes;
- intermediate events;
- escalation;
- climax;
- ending.

Do not merely paraphrase the premise.
""".strip()

        if mode == EXPAND_USER_STORY_MODE:

            return """
EXPAND STORY MODE.

The user's text is authoritative about its core facts.

Preserve important:
- characters;
- chronology;
- events;
- settings;
- outcomes;
- explicit constraints.

You may enrich:
- motivations;
- transitions;
- emotional beats;
- pacing;
- sensory detail;
- tension;
- dialogue;
- cause and effect.

The result must remain recognizably the same story.
""".strip()

        if mode == PRESERVE_USER_STORY_MODE:

            return """
PRESERVE STORY MODE.

The supplied story is immutable.

Do not rewrite the story.

You may generate production metadata around it,
but the story itself must remain unchanged after
whitespace normalization.
""".strip()

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    # ========================================================
    # PROMPTS
    # ========================================================

    def _story_director_system(
        self,
        mode: str,
    ) -> str:

        return f"""
You are the STORY DIRECTOR for a MiniMax H3
cinematic production system.

{self._mode_instruction(mode)}

Produce three things:

1. story
2. characters
3. scenes

Do NOT create shots.

The context window is limited.
Keep every field concise.

Create approximately 3–6 meaningful scenes.

A scene is a REAL narrative beat.

A scene description must contain an actual event,
action, environment or dramatic moment.

Do not create metadata-only scenes.

Character names must be real story entities.
Never convert ordinary prose words into character names.

Return JSON only:

{{
  "story": "string",
  "director_notes": "string",
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
  ],
  "scenes": [
    {{
      "scene_id": "scene_001",
      "order": 1,
      "location": "string",
      "description": "actual narrative event",
      "mood": "string",
      "lighting": "string",
      "characters": [],
      "scene_objective": "string",
      "continuity_notes": "string"
    }}
  ]
}}
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

    def _scene_recovery_system(
        self,
        mode: str,
    ) -> str:

        return f"""
You are a SCENE DIRECTOR for MiniMax H3.

{self._mode_instruction(mode)}

The primary director response did not provide
usable scene data.

Create 3–6 meaningful scenes.

Do not create shots.

Do not create metadata-only scenes.

Return JSON only:

{{
  "scenes": [
    {{
      "scene_id": "scene_001",
      "order": 1,
      "location": "string",
      "description": "real narrative event",
      "mood": "string",
      "lighting": "string",
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
    ) -> str:

        compact_characters = []

        for character in characters:

            compact_characters.append(
                {
                    "name": character.get(
                        "name",
                        "",
                    ),
                    "role": character.get(
                        "role",
                        "",
                    ),
                    "description": self._limit_text(
                        character.get(
                            "description",
                            "",
                        ),
                        350,
                    ),
                }
            )

        return json.dumps(
            {
                "story": self._limit_text(
                    story,
                    4500,
                ),
                "characters": compact_characters,
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    def _shot_director_system(
        self,
    ) -> str:

        return """
You are the CINEMATIC SHOT DIRECTOR for MiniMax H3.

Create cinematic shots for exactly ONE supplied scene.

Do not create characters.

Only use supplied character names.

Use 2–4 shots when useful.

Return JSON only:

{
  "shots": []
}

Every shot must contain:

shot_id
scene_id
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
        scene: dict,
    ) -> str:

        compact_characters = []

        for character in characters:

            compact_characters.append(
                {
                    "name": character.get(
                        "name",
                        "",
                    ),
                    "role": character.get(
                        "role",
                        "",
                    ),
                    "description": self._limit_text(
                        character.get(
                            "description",
                            "",
                        ),
                        350,
                    ),
                    "appearance": character.get(
                        "appearance",
                        {},
                    ),
                    "clothing": character.get(
                        "clothing",
                        {},
                    ),
                }
            )

        compact_scene = {
            "scene_id": scene.get(
                "scene_id",
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
            "characters": scene.get(
                "characters",
                [],
            ),
            "scene_objective": self._limit_text(
                scene.get(
                    "scene_objective",
                    "",
                ),
                500,
            ),
            "continuity_notes": self._limit_text(
                scene.get(
                    "continuity_notes",
                    "",
                ),
                600,
            ),
        }

        return json.dumps(
            {
                "story": self._limit_text(
                    story,
                    3500,
                ),
                "characters": compact_characters,
                "scene": compact_scene,
            },
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    # ========================================================
    # HELPERS
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
            ].rstrip()
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

            parsed = json.loads(
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
            parsed,
            dict,
        ):

            raise RuntimeError(
                "Qwen director output must be a JSON object."
            )

        return parsed

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 512,
    ) -> dict:

        if self._llama is None:

            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        _, max_tokens = (
            self._available_output_tokens(
                system_prompt,
                user_prompt,
                minimum_completion=minimum_completion,
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
                    "type": "json_object",
                },
            )
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

    # ========================================================
    # CHARACTER CLEANING
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

    def _character_fallback(
        self,
        base_plan: dict,
    ) -> list[dict]:

        base_characters = (
            base_plan.get(
                "characters",
                [],
            )
            or []
        )

        sanitized = (
            self._sanitize_characters(
                base_characters
            )
        )

        if sanitized:
            return sanitized

        # Preserve valid deterministic-planner objects even
        # if their names are generic roles that the Qwen filter
        # would normally reject.
        result: list[dict] = []
        seen: set[str] = set()

        for value in base_characters:

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

            key = name.lower()

            if key in seen:
                continue

            seen.add(
                key
            )

            result.append(
                deepcopy(
                    value
                )
            )

        return result

    # ========================================================
    # SCENE CLEANING
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
                description = str(
                    value.get(
                        "story_summary",
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

                alternative = str(
                    value.get(
                        "scene_objective",
                        "",
                    )
                    or ""
                ).strip()

                if alternative:
                    description = alternative
                else:
                    continue

            selected: list[str] = []

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

            scene_id = str(
                value.get(
                    "scene_id",
                    f"scene_{index:03d}",
                )
                or f"scene_{index:03d}"
            ).strip()

            result.append(
                {
                    "scene_id": scene_id,
                    "order": (
                        len(result) + 1
                    ),
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
                    )[:12],
                    "key_props": list(
                        value.get(
                            "key_props",
                            [],
                        )
                        or []
                    )[:8],
                    "scene_objective": str(
                        value.get(
                            "scene_objective",
                            "",
                        )
                        or ""
                    ),
                    "characters": selected,
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

    def _scene_fallback(
        self,
        base_plan: dict,
        character_names: set[str],
    ) -> list[dict]:

        return self._sanitize_scenes(
            base_plan.get(
                "scenes",
                [],
            )
            or [],
            character_names,
        )

    # ========================================================
    # SHOT CLEANING
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

            if (
                candidate_scene_id
                != scene_id
            ):
                continue

            selected = []

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

                raw = str(
                    name
                ).strip()

                canonical = (
                    allowed.get(
                        raw.lower()
                    )
                )

                if canonical is not None:
                    selected.append(
                        canonical
                    )

            speaking = []

            for name in (
                value.get(
                    "speaking_characters",
                    [],
                )
                or []
            ):

                raw = str(
                    name
                ).strip()

                canonical = (
                    allowed.get(
                        raw.lower()
                    )
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
                            f"shot_{index:03d}",
                        )
                        or f"shot_{index:03d}"
                    ).strip(),
                    "scene_id": scene_id,
                    "order": index,
                    "duration_seconds": duration,
                    "characters": selected,
                    "location": str(
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
                    "action": str(
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
                    "camera_shot": str(
                        value.get(
                            "camera_shot",
                            "cinematic medium shot",
                        )
                        or "cinematic medium shot"
                    ),
                    "camera_movement": str(
                        value.get(
                            "camera_movement",
                            "controlled cinematic movement",
                        )
                        or "controlled cinematic movement"
                    ),
                    "lighting": str(
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
                    "mood": str(
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
                    "visual_prompt": str(
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
                    "retention_analysis": str(
                        value.get(
                            "retention_analysis",
                            "Maintain narrative and visual continuity.",
                        )
                        or "Maintain narrative and visual continuity."
                    ),
                    "detailed_description": str(
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
                    "overall_soundscape": str(
                        value.get(
                            "overall_soundscape",
                            "Natural cinematic environmental sound.",
                        )
                        or "Natural cinematic environmental sound."
                    ),
                    "non_diegetic_music": str(
                        value.get(
                            "non_diegetic_music",
                            "Subtle cinematic score.",
                        )
                        or "Subtle cinematic score."
                    ),
                    "negative_prompt": str(
                        value.get(
                            "negative_prompt",
                            "identity drift, face deformation, inconsistent clothing",
                        )
                        or "identity drift, face deformation, inconsistent clothing"
                    ),
                    "continuity_notes": str(
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

    @staticmethod
    def _fallback_shot(
        scene: dict,
        index: int,
    ) -> dict:

        description = str(
            scene.get(
                "description",
                "",
            )
            or ""
        )

        return {
            "shot_id": (
                f"shot_{index:03d}"
            ),
            "scene_id": scene.get(
                "scene_id",
                f"scene_{index:03d}",
            ),
            "order": index,
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
                or description
            ),
            "camera_shot": (
                "wide cinematic establishing shot"
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
            "visual_prompt": description,
            "retention_analysis": (
                "Maintain narrative continuity "
                "and character identity."
            ),
            "detailed_description": description,
            "overall_soundscape": (
                "Natural cinematic environmental sound."
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

    # ========================================================
    # MODE VALIDATION
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

    @staticmethod
    def _significant_words(
        text: str,
    ) -> set[str]:

        words = re.findall(
            r"[A-Za-z][A-Za-z'-]{4,}",
            str(
                text or ""
            ).lower(),
        )

        stop = {
            "about",
            "after",
            "again",
            "their",
            "there",
            "which",
            "while",
            "where",
            "would",
            "could",
            "should",
            "every",
            "through",
            "story",
            "world",
            "person",
            "scene",
            "camera",
            "visual",
            "tone",
            "with",
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
                "Qwen director returned an empty story."
            )

        if mode == PRESERVE_USER_STORY_MODE:

            if source != result:
                raise RuntimeError(
                    "Preserve Story mode changed "
                    "the supplied story."
                )

            return

        if mode == EXPAND_USER_STORY_MODE:

            if source == result:

                raise RuntimeError(
                    "Expand Story mode returned "
                    "the supplied story unchanged."
                )

            if len(result) < max(
                120,
                int(
                    len(source)
                    * 1.15
                ),
            ):

                raise RuntimeError(
                    "Expand Story mode did not produce "
                    "a meaningfully richer story."
                )

            source_words = (
                self._significant_words(
                    source
                )
            )

            if source_words:

                overlap = len(
                    source_words
                    & self._significant_words(
                        result
                    )
                )

                if (
                    overlap
                    / max(
                        1,
                        len(source_words),
                    )
                    < 0.25
                ):

                    raise RuntimeError(
                        "Expand Story mode changed too much "
                        "of the supplied story."
                    )

            return

        if mode == AI_STORY_MODE:

            if source == result:

                raise RuntimeError(
                    "AI Story mode returned "
                    "the premise unchanged."
                )

            if len(result) < max(
                160,
                int(
                    len(source)
                    * 1.20
                ),
            ):

                raise RuntimeError(
                    "AI Story mode did not substantially "
                    "develop the supplied premise."
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

    # ========================================================
    # SCENE RECOVERY
    # ========================================================

    def _recover_scenes(
        self,
        mode: str,
        story: str,
        characters: list[dict],
        character_names: set[str],
    ) -> list[dict]:

        try:

            recovery = self._chat_json(
                self._scene_recovery_system(
                    mode
                ),
                self._scene_recovery_user(
                    story,
                    characters,
                ),
                minimum_completion=500,
            )

        except Exception:

            return []

        return self._sanitize_scenes(
            recovery.get(
                "scenes",
                [],
            ),
            character_names,
        )

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
            minimum_completion=650,
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
                self._character_fallback(
                    base_plan
                )
            )

        if not characters:

            # This is the only situation where character
            # generation is genuinely impossible: neither
            # Qwen nor the deterministic planner supplied
            # a valid character structure.
            raise RuntimeError(
                "No usable characters are available from "
                "the director or deterministic planner."
            )

        character_names = {
            character[
                "name"
            ].lower()
            for character
            in characters
        }

        # ----------------------------------------------------
        # SCENES
        # ----------------------------------------------------

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

            recovery_story = str(
                story_plan.get(
                    "story",
                    user_input,
                )
                or user_input
            ).strip()

            scenes = self._recover_scenes(
                mode=mode,
                story=recovery_story,
                characters=characters,
                character_names=character_names,
            )

        if not scenes:

            scenes = (
                self._scene_fallback(
                    base_plan,
                    character_names,
                )
            )

        if not scenes:

            raise RuntimeError(
                "No usable scenes are available from "
                "the director or deterministic planner."
            )

        # ----------------------------------------------------
        # STORY
        # ----------------------------------------------------

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
        # PASS 2 — ONE SCENE AT A TIME
        # ----------------------------------------------------

        all_shots: list[dict] = []

        for scene in scenes:

            try:

                shot_plan = self._chat_json(
                    self._shot_director_system(),
                    self._shot_director_user(
                        story,
                        characters,
                        scene,
                    ),
                    minimum_completion=450,
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

            if not scene_shots:

                scene_shots = [
                    self._fallback_shot(
                        scene,
                        len(all_shots) + 1,
                    )
                ]

            all_shots.extend(
                scene_shots
            )

        if not all_shots:

            raise RuntimeError(
                "No usable shots could be produced."
            )

        for index, shot in enumerate(
            all_shots,
            start=1,
        ):

            shot[
                "order"
            ] = index

            shot[
                "shot_id"
            ] = str(
                shot.get(
                    "shot_id",
                    f"shot_{index:03d}",
                )
                or f"shot_{index:03d}"
            )

        for index, scene in enumerate(
            scenes,
            start=1,
        ):

            scene[
                "order"
            ] = index

        self._validate_shot_character_contract(
            all_shots,
            characters,
        )

        return {
            "enabled": True,
            "plan": {
                "story": story,
                "story_mode": mode,
                "director_notes": director_notes,
                "characters": characters,
                "scenes": scenes,
                "shots": all_shots,
            },
            "director_notes": director_notes,
        }

    # ========================================================
    # ENRICHMENT / MERGE
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

        # ----------------------------------------------------
        # STORY
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CHARACTERS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SCENES
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SHOTS
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # FINAL SAFETY
        # ----------------------------------------------------

        if not merged.get(
            "characters"
        ):

            merged[
                "characters"
            ] = deepcopy(
                base_plan.get(
                    "characters",
                    [],
                )
                or []
            )

        if not merged.get(
            "scenes"
        ):

            merged[
                "scenes"
            ] = deepcopy(
                base_plan.get(
                    "scenes",
                    [],
                )
                or []
            )

        if not merged.get(
            "shots"
        ):

            merged[
                "shots"
            ] = deepcopy(
                base_plan.get(
                    "shots",
                    [],
                )
                or []
            )

        return merged

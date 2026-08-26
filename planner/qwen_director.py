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
    DIRECTOR_MAX_PLAN_CHARS,
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
    Local Qwen3-14B planning/director layer.

    Responsibilities:
      - story development
      - character bible
      - scene structure
      - shot design
      - cinematography
      - lighting
      - dialogue
      - soundscape
      - continuity planning

    It does NOT run during H3 generation.

    Architecture:

        Qwen3-14B Q4_K_M
              ↓
        production plan
              ↓
        model unload
              ↓
        MiniMax H3
    """

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

    def _find_model(self) -> Path:
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
    def available(self) -> bool:
        return (
            director_enabled()
            and self._model_path is not None
            and self._model_path.is_file()
        )

    # ========================================================
    # MODEL LIFECYCLE
    # ========================================================

    @staticmethod
    def _load_nvidia_cuda_libraries() -> None:
        """
        Load the CUDA 13 runtime libraries before llama.cpp
        loads libllama.so.

        Kaggle installs the NVIDIA CUDA packages under a
        shared nvidia/cu13/lib directory, so do not assume
        package-specific paths such as:

            nvidia/cuda_runtime/lib
            nvidia/cublas/lib
        """

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

        cudart_candidates = []
        cublas_candidates = []

        for site_root in site_roots:

            nvidia_root = (
                site_root
                / "nvidia"
            )

            if not nvidia_root.is_dir():
                continue

            try:

                cudart_candidates.extend(
                    nvidia_root.rglob(
                        "libcudart.so.13*"
                    )
                )

                cublas_candidates.extend(
                    nvidia_root.rglob(
                        "libcublas.so.13*"
                    )
                )

            except OSError:
                continue

        cudart_candidates = [
            path
            for path in cudart_candidates
            if path.is_file()
        ]

        cublas_candidates = [
            path
            for path in cublas_candidates
            if path.is_file()
        ]

        if not cudart_candidates:
            raise RuntimeError(
                "libcudart.so.13 was not found in the "
                "installed NVIDIA Python packages."
            )

        if not cublas_candidates:
            raise RuntimeError(
                "libcublas.so.13 was not found in the "
                "installed NVIDIA Python packages."
            )

        # Prefer the CUDA 13 shared runtime directory that
        # contains both libraries, exactly as installed by
        # the Kaggle bootstrap.
        cudart_lib = cudart_candidates[0]

        matching_cublas = [
            path
            for path in cublas_candidates
            if path.parent
            == cudart_lib.parent
        ]

        cublas_lib = (
            matching_cublas[0]
            if matching_cublas
            else cublas_candidates[0]
        )

        library_dirs = [
            str(
                cudart_lib.parent
            ),
            str(
                cublas_lib.parent
            ),
        ]

        existing_ld = os.environ.get(
            "LD_LIBRARY_PATH",
            "",
        )

        if existing_ld:
            library_dirs.append(
                existing_ld
            )

        os.environ[
            "LD_LIBRARY_PATH"
        ] = ":".join(
            library_dirs
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
                f"Error: {exc}"
            ) from exc
            
    def load(self) -> None:
        if not self.available:
            return

        if self._llama is not None:
            return

        # IMPORTANT:
        # The CUDA 13 llama.cpp wheel expects libcudart.so.13
        # and related NVIDIA libraries to be available before
        # libllama.so is loaded.
        self._load_nvidia_cuda_libraries()

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Run the Kaggle bootstrap before using "
                "the Qwen director."
            ) from exc

        except OSError as exc:
            raise RuntimeError(
                "llama-cpp-python native CUDA library "
                "could not be loaded:\n"
                f"{exc}"
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
                "Failed to initialize the Qwen3-14B "
                "director model.\n"
                f"Model: {self._model_path}\n"
                f"Error: {exc}"
            ) from exc

    def unload(self) -> None:
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
    # INPUT COMPACTION
    # ========================================================

    @staticmethod
    def _creative_character(
        character: dict,
    ) -> dict:

        return {
            "name": character.get(
                "name",
                "",
            ),
            "role": character.get(
                "role",
                "",
            ),
            "description": character.get(
                "description",
                "",
            ),
            "personality": character.get(
                "personality",
                "",
            ),
            "appearance": character.get(
                "appearance",
                {},
            ),
            "clothing": character.get(
                "clothing",
                {},
            ),
            "distinctive_features": character.get(
                "distinctive_features",
                [],
            ),
            "character_state": character.get(
                "character_state",
                {},
            ),
            "continuity_rules": character.get(
                "continuity_rules",
                [],
            ),
        }

    @staticmethod
    def _creative_scene(
        scene: dict,
    ) -> dict:

        return {
            "scene_id": scene.get(
                "scene_id",
                "",
            ),
            "order": scene.get(
                "order",
                0,
            ),
            "location": scene.get(
                "location",
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
            "description": scene.get(
                "description",
                "",
            ),
            "mood": scene.get(
                "mood",
                "",
            ),
            "lighting": scene.get(
                "lighting",
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
            "scene_objective": scene.get(
                "scene_objective",
                "",
            ),
            "characters": scene.get(
                "characters",
                [],
            ),
            "story_summary": scene.get(
                "story_summary",
                "",
            ),
            "continuity_notes": scene.get(
                "continuity_notes",
                "",
            ),
        }

    @staticmethod
    def _creative_shot(
        shot: dict,
    ) -> dict:

        return {
            "shot_id": shot.get(
                "shot_id",
                "",
            ),
            "scene_id": shot.get(
                "scene_id",
                "",
            ),
            "order": shot.get(
                "order",
                0,
            ),
            "duration_seconds": shot.get(
                "duration_seconds",
                5.2,
            ),
            "characters": shot.get(
                "characters",
                [],
            ),
            "location": shot.get(
                "location",
                "",
            ),
            "action": shot.get(
                "action",
                "",
            ),
            "camera_shot": shot.get(
                "camera_shot",
                "",
            ),
            "camera_movement": shot.get(
                "camera_movement",
                "",
            ),
            "lighting": shot.get(
                "lighting",
                "",
            ),
            "mood": shot.get(
                "mood",
                "",
            ),
            "visual_prompt": shot.get(
                "visual_prompt",
                "",
            ),
            "retention_analysis": shot.get(
                "retention_analysis",
                "",
            ),
            "detailed_description": shot.get(
                "detailed_description",
                "",
            ),
            "overall_soundscape": shot.get(
                "overall_soundscape",
                "",
            ),
            "non_diegetic_music": shot.get(
                "non_diegetic_music",
                "",
            ),
            "negative_prompt": shot.get(
                "negative_prompt",
                "",
            ),
            "continuity_notes": shot.get(
                "continuity_notes",
                "",
            ),
            "speaking_characters": shot.get(
                "speaking_characters",
                [],
            ),
            "speech_text": shot.get(
                "speech_text",
                "",
            ),
        }

    def _compact_plan(
        self,
        plan: dict,
    ) -> dict:

        compact = {
            "story": plan.get(
                "story",
                "",
            ),
            "story_mode": plan.get(
                "story_mode",
                "",
            ),
            "characters": [
                self._creative_character(
                    value
                )
                for value
                in plan.get(
                    "characters",
                    [],
                )
            ],
            "scenes": [
                self._creative_scene(
                    value
                )
                for value
                in plan.get(
                    "scenes",
                    [],
                )
            ],
            "shots": [
                self._creative_shot(
                    value
                )
                for value
                in plan.get(
                    "shots",
                    [],
                )
            ],
        }

        encoded = json.dumps(
            compact,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

        if len(encoded) > DIRECTOR_MAX_PLAN_CHARS:

            encoded = encoded[
                :DIRECTOR_MAX_PLAN_CHARS
            ]

            return {
                "truncated_context": True,
                "context": encoded,
            }

        return compact

    # ========================================================
    # DIRECTOR PROMPT
    # ========================================================

    def _system_prompt(
        self,
        mode: str,
    ) -> str:

        if mode == PRESERVE_USER_STORY_MODE:

            mode_rule = (
                "PRESERVE MODE: never change the user's "
                "story facts, chronology, named entities, "
                "or requested outcome. Improve only the "
                "cinematic interpretation."
            )

        elif mode == EXPAND_USER_STORY_MODE:

            mode_rule = (
                "EXPAND MODE: preserve the user's core "
                "facts and intent, but enrich the story "
                "with character depth, scene development, "
                "cinematography, dialogue, atmosphere, "
                "and visual detail."
            )

        else:

            mode_rule = (
                "AI STORY MODE: treat the user input as "
                "the creative premise. Develop a complete "
                "coherent cinematic story around that premise."
            )

        return f"""
You are the AI story director for a MiniMax H3 cinematic
video production system.

{mode_rule}

Your output controls:
- story development
- character creation
- character personality and visual identity
- character continuity rules
- scene progression
- shot progression
- cinematic camera choices
- lens/shot language
- camera movement
- lighting
- environment detail
- action staging
- dialogue
- soundscape
- music intent
- continuity between shots

Do not discuss your reasoning.
Do not output markdown.
Output JSON only.

Create internally consistent characters.
Never create contradictory identity descriptions.
A character's appearance must remain stable unless the
story explicitly changes it.
Keep clothing and story state consistent.
Use cinematic camera variety intentionally rather than
cycling mechanically through shot types.

The JSON must have exactly these top-level keys:

{{
  "story": "string",
  "director_notes": "string",
  "characters": [],
  "scenes": [],
  "shots": []
}}

Character objects must contain:
name, role, description, personality, appearance,
clothing, distinctive_features, character_state,
continuity_rules.

Scene objects must contain:
scene_id, order, location, time_of_day, weather,
atmosphere, description, mood, lighting,
environment_details, key_props, scene_objective,
characters, story_summary, continuity_notes.

Shot objects must contain:
shot_id, scene_id, order, duration_seconds,
characters, location, action, camera_shot,
camera_movement, lighting, mood, visual_prompt,
retention_analysis, detailed_description,
overall_soundscape, non_diegetic_music,
negative_prompt, continuity_notes,
speaking_characters, speech_text.

Return a practical production plan, not an essay.
""".strip()

    def _user_prompt(
        self,
        mode: str,
        user_input: str,
        plan: dict,
    ) -> str:

        payload = {
            "mode": mode,
            "user_input": user_input,
            "existing_plan": self._compact_plan(
                plan
            ),
        }

        return (
            "Develop the following production request.\n\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )

    # ========================================================
    # JSON GENERATION
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
                "Qwen director did not return a JSON object."
            )

        candidate = text[
            start:end + 1
        ]

        try:

            value = json.loads(
                candidate
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Qwen director returned invalid JSON:\n"
                + str(exc)
            ) from exc

        if not isinstance(
            value,
            dict,
        ):

            raise RuntimeError(
                "Qwen director output is not a JSON object."
            )

        return value

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

        messages = [
            {
                "role": "system",
                "content": self._system_prompt(
                    mode
                ),
            },
            {
                "role": "user",
                "content": self._user_prompt(
                    mode,
                    user_input,
                    base_plan,
                ),
            },
        ]

        try:

            response = (
                self._llama
                .create_chat_completion(
                    messages=messages,
                    temperature=(
                        DIRECTOR_TEMPERATURE
                    ),
                    top_p=(
                        DIRECTOR_TOP_P
                    ),
                    max_tokens=(
                        DIRECTOR_MAX_TOKENS
                    ),
                    response_format={
                        "type": "json_object"
                    },
                    chat_template_kwargs={
                        "enable_thinking": False,
                    },
                )
            )

        except TypeError:

            response = (
                self._llama
                .create_chat_completion(
                    messages=messages,
                    temperature=(
                        DIRECTOR_TEMPERATURE
                    ),
                    top_p=(
                        DIRECTOR_TOP_P
                    ),
                    max_tokens=(
                        DIRECTOR_MAX_TOKENS
                    ),
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

        return {
            "enabled": True,
            "plan": self._extract_json(
                content
            ),
            "director_notes": "",
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

        if mode != (
            PRESERVE_USER_STORY_MODE
        ):

            generated_story = str(
                creative.get(
                    "story",
                    "",
                )
                or ""
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
            ).strip()

            if not name:
                continue

            existing = by_name.get(
                name.lower()
            )

            if existing is None:

                character_id = (
                    f"character_"
                    f"{len(base_characters) + len(characters) + 1:03d}"
                )

                existing = {
                    "character_id":
                        character_id,
                    "name":
                        name,
                    "role":
                        "story character",
                    "description":
                        "",
                    "personality":
                        "",
                    "appearance":
                        {},
                    "clothing":
                        {},
                    "distinctive_features":
                        [],
                    "character_state":
                        {},
                    "continuity_rules":
                        [],
                    "reference_mode":
                        "story_generated",
                    "reference_paths":
                        [],
                    "reference_video_paths":
                        [],
                    "reference_audio_paths":
                        [],
                    "reference_path":
                        None,
                    "reference_video_path":
                        None,
                    "reference_audio_path":
                        None,
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
            for scene in base_scenes
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
            )

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
            for shot in base_shots
        }

        base_shots_by_order = {
            int(
                shot.get(
                    "order",
                    index,
                )
            ): shot
            for index, shot
            in enumerate(
                base_shots,
                start=1,
            )
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
            ).strip()

            existing = (
                base_shot_map.get(
                    shot_id
                )
                if shot_id
                else None
            )

            if existing is None:

                existing = (
                    base_shots_by_order.get(
                        int(
                            spec.get(
                                "order",
                                index,
                            )
                        )
                    )
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
                            f"scene_{index:03d}",
                        ),
                    "order":
                        int(
                            spec.get(
                                "order",
                                index,
                            )
                        ),
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
        self
    ):
        self.unload()

    def __enter__(
        self
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

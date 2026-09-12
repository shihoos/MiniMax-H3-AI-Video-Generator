from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import os
import json
import re

from planner.cinematic_compiler import CinematicCompiler
from planner.entity_resolver import EntityResolver
from pipeline.production_checkpoint import ProductionCheckpoint

from planner.config import (
    AI_STORY_MODE,
    DIRECTOR_CRITIC_STORY_CONTEXT_CHARS,
    DIRECTOR_MAX_TOKENS,
    DIRECTOR_SHOTS_PER_SCENE,
    DIRECTOR_N_CTX,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    director_enabled,
)

from planner.qwen_director_runtime import (
    _with_faulthandler_watchdog,
    QwenDirectorRuntimeMixin,
)
from planner.qwen_director_prompts import (
    QwenDirectorPromptMixin,
)
from planner.qwen_director_scene import QwenDirectorSceneMixin
from planner.qwen_director_sanitize import QwenDirectorSanitizeMixin


class QwenDirector(
    QwenDirectorRuntimeMixin,
    QwenDirectorPromptMixin,
    QwenDirectorSceneMixin,
    QwenDirectorSanitizeMixin,
):

    # Production planning limits. Keep the narrative rich, but bound the
    # production graph so an LLM cannot accidentally explode a short film
    # into dozens of scenes and therefore dozens of expensive Qwen calls.
    MAX_SCENES = 6
    SHOTS_PER_SCENE = DIRECTOR_SHOTS_PER_SCENE
    # Creative shot batching may cover up to two fresh adjacent scenes.
    # The runtime selects the largest batch that still fits the 8K context
    # budget with a bounded completion reserve. Missing shots are deterministic
    # fallbacks; no per-scene Qwen recovery is used.
    MAX_SHOT_BATCH_SCENES = 2

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

        self._vllm_session = None

        self._model_path = (
            self._find_model()
            if director_enabled()
            else None
        )

        self._fallback_planner = None
        self._entity_resolver = EntityResolver()
        self._current_visual_language: dict = {}
        self._reference_visual_context: dict[str, dict] = {}
        self._character_semantic_calls = 0

        # Optional development diagnostics. Both are disabled unless the
        # corresponding environment variable is explicitly configured.
        self._trace_dir = self._optional_directory_env(
            "H3_DIRECTOR_TRACE_DIR"
        )
        self._cache_dir = self._optional_directory_env(
            "H3_DIRECTOR_CACHE_DIR"
        )
        self._cache_namespace = "minimax-h3-qwen-vllm-json-v1"

        # Runtime Qwen telemetry is intentionally lightweight: keep only
        # aggregate/per-call metrics needed to diagnose latency, token usage,
        # retries, cache behavior, and deterministic recovery decisions.
        self._qwen_telemetry = {
            "calls": [],
            "total_elapsed_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retries": 0,
            "cache_hits": 0,
            "deterministic_recoveries": 0,
        }

    def set_reference_visual_context(self, context: dict[str, dict] | None) -> None:
        self._reference_visual_context = {
            str(key): dict(value)
            for key, value in (context or {}).items()
            if isinstance(value, dict)
        }

    @_with_faulthandler_watchdog
    def generate(
        self,
        *,
        mode: str,
        user_input: str,
        base_plan: dict,
        checkpoint_session_id: str | None = None,
        resume_state: dict | None = None,
    ) -> dict:

        self._character_semantic_calls = 0
        if not director_enabled():

            plan = deepcopy(base_plan)
            scenes = list(plan.get("scenes", []) or [])
            shots = list(plan.get("shots", []) or [])
            characters = {
                str(character.get("name", "")).strip()
                for character in plan.get("characters", []) or []
                if isinstance(character, dict)
                and str(character.get("name", "")).strip()
            }
            if scenes and shots:
                plan["shots"] = CinematicCompiler(
                    character_names=characters
                ).compile_all(scenes, shots)

            return {
                "enabled": False,
                "plan": plan,
                "director_notes": "",
            }

        if mode not in (
            self._MODE_LABELS
        ):

            raise ValueError(
                f"Unsupported story mode: {mode}"
            )

        if (resume_state or {}).get("stage") in {
            "director_complete",
            "production_plan",
            "rendering",
            "render_complete",
        }:
            prior = deepcopy(
                (resume_state or {}).get("director_plan", {}) or {}
            )
            if (
                prior.get("story")
                and prior.get("scenes")
                and prior.get("shots")
            ):
                return {
                    "enabled": True,
                    "plan": prior,
                    "director_notes": str(
                        prior.get("director_notes", "") or ""
                    ),
                }

        self.load()

        if self._vllm_session is None:

            raise RuntimeError(
                "Qwen director model failed to load."
            )

        checkpoint_store = (
            ProductionCheckpoint(
                self.project_root
            )
            if checkpoint_session_id
            else None
        )

        prior_director_plan = (
            deepcopy(
                (resume_state or {}).get(
                    "director_plan",
                    {},
                )
                or {}
            )
        )

        prior_shots = list(
            prior_director_plan.get(
                "shots",
                [],
            )
            or []
        )

        resume_stage = str(
            (resume_state or {}).get(
                "stage",
                "",
            )
            or ""
        ).strip()

        resuming = bool(
            resume_state
            and prior_director_plan
            and resume_stage in {
                "initialized",
                "narrative",
                "metadata",
                "shots",
                "director_complete",
            }
        )

        # Each top-level plan gets an isolated telemetry session. On resume,
        # carry forward the semantic-call budget recorded in the checkpoint
        # instead of resetting the character extractor/adjudicator allowance.
        self._qwen_telemetry = {
            "calls": [],
            "total_elapsed_seconds": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "retries": 0,
            "cache_hits": 0,
            "deterministic_recoveries": 0,
        }
        if resuming:
            try:
                self._character_semantic_calls = max(
                    0,
                    min(2, int(prior_director_plan.get("_character_semantic_calls_used", 0) or 0)),
                )
            except (TypeError, ValueError):
                self._character_semantic_calls = 0

        temperature, top_p = (
            self._sampling_for_mode(
                mode
            )
        )

        # ----------------------------------------------------
        # PASS 1A: narrative
        # ----------------------------------------------------

        if resuming and prior_director_plan.get(
            "story"
        ):

            story = self._normalize_story(
                prior_director_plan.get(
                    "story",
                    "",
                )
            )

            story_plan = deepcopy(
                prior_director_plan
            )

            self._current_visual_language = (
                self._sanitize_visual_language(
                    story_plan.get(
                        "visual_language",
                        {},
                    )
                )
            )

        elif mode == PRESERVE_USER_STORY_MODE:

            story = self._normalize_story(
                user_input
            )

            story_plan = {
                "story": story,
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

                story = self._chat_text(
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
                    disable_thinking=True,
                )

                self._validate_mode_output(
                    mode,
                    user_input,
                    story,
                )

            except RuntimeError as first_error:

                if mode == EXPAND_USER_STORY_MODE:
                    # Expansion failure must not trigger another expensive
                    # Qwen call. The original user story is the deterministic
                    # correctness fallback; downstream planning can continue.
                    self._record_recovery(
                        "expand_story_source_fallback",
                        str(first_error),
                    )
                    story = self._normalize_story(
                        user_input
                    )
                    self._validate_mode_output(
                        PRESERVE_USER_STORY_MODE,
                        user_input,
                        story,
                    )
                else:
                    failure_text = str(first_error)
                    retry_user = (
                        story_user
                        + "\n\n"
                        "REPAIR REQUIRED.\n"
                        + f"Previous validation failure: {failure_text}\n"
                        + "Write a complete, shorter story of about 350-450 words. "
                        "Prioritize reaching the resolution and ending on a complete sentence over extra detail. "
                        "Return ONLY the story prose. "
                        "Do not output JSON or commentary."
                    )

                    story = self._chat_text(
                        story_system,
                        retry_user,
                        minimum_completion=350,
                        temperature=min(
                            0.85,
                            max(0.70, temperature + 0.10),
                        ),
                        top_p=0.92,
                        call_name="ai_story_text_retry",
                        max_completion=1600,
                        disable_thinking=True,
                    )

                    self._validate_mode_output(
                        mode,
                        user_input,
                        story,
                    )

            story_plan = {
                "story": story,
            }

        # ----------------------------------------------------
        # PASS 1B: deterministic production foundation
        # ----------------------------------------------------
        #
        # The ProductionPlanner has already created the canonical
        # characters and scene topology. Do not spend a Qwen call
        # regenerating deterministic metadata. Qwen is only the creative
        # enrichment layer from this point onward.
        #
        # This also guarantees that the Director always has canonical
        # entities/scene IDs even when the Director is enabled.
        base_characters = deepcopy(
            base_plan.get("characters", [])
            or []
        )

        base_scenes = deepcopy(
            base_plan.get("scenes", [])
            or []
        )

        story_plan = {
            "story": story,
            "characters": base_characters,
            "scenes": base_scenes,
        }

        story = self._normalize_story(
            story_plan.get(
                "story",
                story,
            )
            or story
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
        baseline_visual_language = self._baseline_visual_language()
        for key, value in baseline_visual_language.items():
            if not visual_language.get(key):
                visual_language[key] = value

        self._current_visual_language = (
            dict(
                visual_language
            )
        )

        print("[DIRECTOR] resolving canonical roster", flush=True)

        # ----------------------------------------------------
        # CANONICAL CHARACTERS / SCENES
        # ----------------------------------------------------
        #
        # AI Story / Expand Story:
        # Qwen first creates the final narrative. The canonical production
        # roster and scene topology must then be derived from THAT final story.
        #
        # Preserve Story:
        # the supplied user story remains the source of truth.
        #
        # Qwen owns semantic character identity. ProductionPlanner performs
        # only bounded validation/canonicalization before production binding.

        planner = self._planner()

        if mode in (
            AI_STORY_MODE,
            EXPAND_USER_STORY_MODE,
        ):
            canonical_source_story = story
        else:
            canonical_source_story = user_input

        resume_roster = []
        if resuming and prior_director_plan.get("_canonical_character_roster_verified") is True:
            prior_roster = prior_director_plan.get("characters", []) or []
            if isinstance(prior_roster, list):
                resume_roster = [dict(item) for item in prior_roster if isinstance(item, dict)]

        if resume_roster:
            from schemas.character import Character
            canonical_characters = []
            for item in resume_roster:
                allowed = {
                    "character_id", "name", "role", "description", "personality",
                    "appearance", "clothing", "distinctive_features", "character_state",
                    "continuity_rules", "reference_mode", "reference_paths",
                    "reference_video_paths", "reference_audio_paths", "reference_path",
                    "reference_video_path", "reference_audio_path", "reference_mask_path",
                    "identity_profile", "story_state_profile",
                }
                payload = {key: deepcopy(value) for key, value in item.items() if key in allowed}
                canonical_characters.append(Character(**payload))
        else:
            canonical_characters = planner.create_characters(
                canonical_source_story,
                qwen_character_extractor=self.extract_character_entities,
                qwen_character_adjudicator=self.adjudicate_character_entities,
            )

        characters = self._sanitize_characters(
            [
                character.to_dict()
                for character in canonical_characters
                if character is not None
            ]
        )

        if not characters:
            raise RuntimeError(
                "No canonical characters could be derived from the final story."
            )

        character_names = {
            str(character.get("name", "")).strip().lower()
            for character in characters
            if isinstance(character, dict)
            and str(character.get("name", "")).strip()
        }

        if not character_names:
            raise RuntimeError(
                "Canonical character extraction produced no usable names."
            )

        canonical_scenes = planner.create_scenes(
            canonical_source_story,
            canonical_characters,
        )

        scenes = self._sanitize_scenes(
            [
                scene.to_dict()
                for scene in canonical_scenes
                if scene is not None
            ],
            character_names,
        )
    
        if not scenes:
            raise RuntimeError(
                "Deterministic base plan contains no canonical scenes."
            )

        if len(scenes) < 4 or len(scenes) > self.MAX_SCENES:
            raise RuntimeError(
                "Deterministic base plan scene count is outside "
                f"the required 4-{self.MAX_SCENES} range: {len(scenes)}"
            )

        # Preserve deterministic scene topology and only add the
        # Director's creative scene annotations if they already exist.
        scenes = self._annotate_scene_functions(
            scenes
        )

        director_plan = {
            "story": story,
            "story_mode": mode,
            "director_notes": director_notes,
            "visual_language": visual_language,
            "characters": deepcopy(characters),
            # This marker is set only after ProductionPlanner has performed
            # deterministic extraction plus the bounded Qwen semantic pass.
            # enrich_plan may use this verified roster, but never raw Qwen
            # character metadata from the creative response.
            "_canonical_character_roster_verified": True,
            "_character_semantic_calls_used": int(self._character_semantic_calls),
            "scenes": deepcopy(scenes),
            "shots": prior_shots,
        }

        completed_scene_ids = []
        prior_by_scene = {}

        for shot in prior_shots:
            if not isinstance(shot, dict):
                continue

            sid = str(
                shot.get("scene_id", "") or ""
            ).strip()

            if sid:
                prior_by_scene.setdefault(
                    sid,
                    [],
                ).append(shot)

        for scene in scenes:
            sid = str(
                scene.get("scene_id", "") or ""
            ).strip()

            if len(prior_by_scene.get(sid, [])) >= self.SHOTS_PER_SCENE:
                completed_scene_ids.append(sid)

        if checkpoint_store and checkpoint_session_id:
            self._save_checkpoint(
                checkpoint_store,
                checkpoint_session_id,
                self._checkpoint_state(
                    checkpoint_session_id,
                    mode,
                    user_input,
                    base_plan,
                    director_plan,
                    "running",
                    "shots",
                    completed_scene_ids,
                    completed_scene_ids[-1]
                    if completed_scene_ids
                    else "",
                ),
            )

        print(f"[DIRECTOR] roster={len(characters)} scenes={len(scenes)}; scene count locked before shot batches", flush=True)

        # ----------------------------------------------------
        # PASS 2: cinematography
        #
        # Fresh scenes are planned in bounded creative batches of up to two.
        # Qwen supplies only
        # creative shot direction; deterministic compilation/rebinding
        # supplies all production identity and technical fields.
        # ----------------------------------------------------

        all_shots: list[dict] = []
        shot_temperature, shot_top_p = self._shot_sampling()

        try:

            scene_index = 0

            while scene_index < len(scenes):

                scene = scenes[
                    scene_index
                ]

                scene_id = str(
                    scene.get(
                        "scene_id",
                        "",
                    )
                    or ""
                ).strip()

                existing_scene_shots = [
                    deepcopy(item)
                    for item
                    in prior_by_scene.get(
                        scene_id,
                        [],
                    )[: self.SHOTS_PER_SCENE]
                    if isinstance(
                        item,
                        dict,
                    )
                ]

                # Resumed scene: never regenerate completed work.
                if len(existing_scene_shots) >= self.SHOTS_PER_SCENE:

                    existing_scene_shots = (
                        existing_scene_shots[
                            : self.SHOTS_PER_SCENE
                        ]
                    )

                    all_shots.extend(
                        existing_scene_shots
                    )

                    scene_index += 1
                    continue

                # Build the largest safe batch of fresh adjacent scenes.
                # A partially completed/resumed scene is deterministic-only:
                # its existing shots are preserved and any shortage is filled
                # later from the canonical base plan. Fresh scenes receive
                # exactly one creative Qwen batch call.
                if existing_scene_shots:
                    batch_scenes = [scene]
                else:
                    max_count = min(
                        self.MAX_SHOT_BATCH_SCENES,
                        len(scenes) - scene_index,
                    )

                    batch_scenes = []
                    for offset in range(max_count):
                        candidate_scene = scenes[scene_index + offset]
                        candidate_id = str(
                            candidate_scene.get("scene_id", "") or ""
                        ).strip()

                        # Do not cross a checkpoint/resume boundary.
                        if prior_by_scene.get(candidate_id):
                            break

                        batch_scenes.append(candidate_scene)

                    if not batch_scenes:
                        batch_scenes = [scene]

                generated_by_scene: dict[str, list[dict]] = {}

                # ------------------------------------------------
                # CREATIVE SHOT PASS
                # ------------------------------------------------
                # One fresh batch call for 1–2 scenes. No per-scene retry
                # or missing-shot Qwen recovery is performed.
                if existing_scene_shots:
                    generated_by_scene[scene_id] = list(
                        existing_scene_shots
                    )[: self.SHOTS_PER_SCENE]

                else:
                    # Start with the largest candidate and shrink only if the
                    # prompt would leave less than a useful completion reserve
                    # inside the fixed 8192-token context.
                    while True:
                        batch_user = self._shot_director_batch_user(
                            story,
                            characters,
                            batch_scenes,
                            visual_language,
                        )

                        desired_completion = self._shot_batch_completion_budget(
                            len(batch_scenes)
                        )

                        prompt_tokens = self._count_tokens(
                            self._shot_director_batch_system()
                            + "\n\n"
                            + batch_user
                        )

                        if (
                            prompt_tokens
                            <= int(DIRECTOR_N_CTX)
                            - 128
                            - desired_completion
                        ):
                            break

                        if len(batch_scenes) == 1:
                            break

                        batch_scenes = batch_scenes[:-1]

                    batch_ids = [
                        str(
                            item.get("scene_id", "") or ""
                        ).strip()
                        for item in batch_scenes
                    ]

                    try:
                        batch_response = self._chat_json(
                            self._shot_director_batch_system(),
                            batch_user,
                            minimum_completion=320,
                            temperature=shot_temperature,
                            top_p=shot_top_p,
                            call_name=(
                                "shot_batch:"
                                + "_".join(batch_ids)
                            ),
                            max_completion=self._shot_batch_completion_budget(
                                len(batch_scenes)
                            ),
                            json_mode=True,
                            disable_thinking=True,
                            response_schema=self._shot_batch_json_schema(
                                scene_count=len(batch_scenes),
                            ),
                        )

                        batch_map = self._normalize_batch_shot_response(
                            batch_response
                        )

                        for target_scene in batch_scenes:
                            target_id = str(
                                target_scene.get("scene_id", "") or ""
                            ).strip()

                            candidate = self._sanitize_shots(
                                batch_map.get(
                                    target_id,
                                    [],
                                ),
                                target_scene,
                                character_names,
                            )

                            if candidate:
                                generated_by_scene[target_id] = candidate[
                                    : self.SHOTS_PER_SCENE
                                ]

                    except Exception as batch_error:
                        self._record_recovery(
                            "shot_batch_deterministic_fallback",
                            str(batch_error),
                        )

                # The deterministic repair pass after the loop owns missing
                # shots. Never launch another Qwen request here.
                # PERSIST THIS BATCH
                # ------------------------------------------------
                batch_added = []

                for target_scene in batch_scenes:

                    target_id = str(
                        target_scene.get(
                            "scene_id",
                            "",
                        )
                        or ""
                    ).strip()

                    scene_shots = list(
                        generated_by_scene.get(
                            target_id,
                            [],
                        )
                    )

                    # Missing shots are intentionally not regenerated with Qwen.
                    # The deterministic repair pass below fills them from base_plan.
                    if (
                        target_id == scene_id
                        and existing_scene_shots
                    ):

                        if not scene_shots:
                            scene_shots = existing_scene_shots

                        elif len(existing_scene_shots) == 1:
                            scene_shots = [
                                existing_scene_shots[0],
                                scene_shots[0],
                            ]

                    scene_shots = scene_shots[
                        : self.SHOTS_PER_SCENE
                    ]

                    batch_added.extend(
                        scene_shots
                    )

                    if len(scene_shots) >= self.SHOTS_PER_SCENE:
                        if target_id not in completed_scene_ids:
                            completed_scene_ids.append(
                                target_id
                            )

                all_shots.extend(
                    batch_added
                )

                if batch_added:

                    director_plan["shots"] = deepcopy(
                        all_shots
                    )

                    last_scene_id = str(
                        batch_scenes[-1].get(
                            "scene_id",
                            "",
                        )
                        or ""
                    ).strip()

                    self._save_checkpoint(
                        checkpoint_store,
                        checkpoint_session_id,
                        self._checkpoint_state(
                            checkpoint_session_id or "",
                            mode,
                            user_input,
                            base_plan,
                            director_plan,
                            "running",
                            "shots",
                            completed_scene_ids,
                            last_scene_id,
                        ),
                    )

                scene_index += len(
                    batch_scenes
                )

        except Exception as exc:

            director_plan["shots"] = deepcopy(
                all_shots
            )

            self._save_checkpoint(
                checkpoint_store,
                checkpoint_session_id,
                self._checkpoint_state(
                    checkpoint_session_id or "",
                    mode,
                    user_input,
                    base_plan,
                    director_plan,
                    "failed",
                    "shots",
                    completed_scene_ids,
                    scene_id if 'scene_id' in locals() else "",
                    str(exc),
                ),
            )

            raise

        # Deterministic shot fallback: if Qwen failed to provide enough
        # creative shots for a scene, reuse only the corresponding canonical
        # base-plan shots. This keeps the production structurally complete
        # without inventing entities or making another model call.
        base_shots_by_scene: dict[str, list[dict]] = {}

        for base_shot in (
            base_plan.get("shots", [])
            or []
        ):
            if not isinstance(base_shot, dict):
                continue

            sid = str(
                base_shot.get("scene_id", "") or ""
            ).strip()

            if sid:
                base_shots_by_scene.setdefault(
                    sid,
                    [],
                ).append(
                    deepcopy(base_shot)
                )

        shots_by_scene: dict[str, list[dict]] = {}

        for shot in all_shots:
            if not isinstance(shot, dict):
                continue

            sid = str(
                shot.get("scene_id", "") or ""
            ).strip()

            if sid:
                shots_by_scene.setdefault(
                    sid,
                    [],
                ).append(shot)

        repaired_shots: list[dict] = []

        for scene in scenes:
            sid = str(
                scene.get("scene_id", "") or ""
            ).strip()

            current = list(
                shots_by_scene.get(sid, [])
            )[: self.SHOTS_PER_SCENE]

            if len(current) < self.SHOTS_PER_SCENE:
                for fallback in base_shots_by_scene.get(sid, []):
                    if len(current) >= self.SHOTS_PER_SCENE:
                        break

                    candidate = deepcopy(fallback)

                    # Never let fallback structural identity conflict with
                    # the canonical scene.
                    candidate["scene_id"] = sid

                    existing_ids = {
                        str(
                            item.get("shot_id", "")
                        ).strip()
                        for item in current
                        if isinstance(item, dict)
                    }

                    candidate_id = str(
                        candidate.get("shot_id", "")
                    ).strip()

                    if candidate_id in existing_ids:
                        continue

                    current.append(candidate)

            # Final safety gate: deterministic fallback shots must traverse
            # the same sanitizer as Qwen-generated shots before compilation.
            # This prevents missing cinematography fields from reaching the
            # strict CinematicCompiler validator.
            current = self._sanitize_shots(
                current,
                scene,
                character_names,
            )[: self.SHOTS_PER_SCENE]

            if len(current) < self.SHOTS_PER_SCENE:
                self._record_recovery(
                    "shot_field_sanitization_incomplete",
                    f"scene={sid} count={len(current)} expected={self.SHOTS_PER_SCENE}",
                )

            repaired_shots.extend(current)

        all_shots = repaired_shots

        # No Qwen-shot failure is fatal here: the deterministic compiler
        # completes production fields while preserving every valid creative
        # shot Qwen produced.
        # Canonicalize scene/shot IDs before the final dialogue-boundary pass.
        # This ordering is deliberate: the boundary pass groups shots by the
        # final canonical scene IDs, so temporary or legacy scene IDs cannot
        # cause dialogue continuation to leak across scene boundaries.
        self._normalize_ids(
            scenes,
            all_shots,
        )

        # Final whole-plan dialogue boundary normalization. Individual
        # sanitizer passes can operate on partial scene batches; this pass
        # canonicalizes continuation flags after all Qwen and fallback shots
        # have been assembled. Dialogue continuation is never allowed to
        # cross a scene boundary or start on the first shot of a scene.
        shots_by_scene_order: dict[str, list[dict]] = {}
        for shot in all_shots:
            sid = str(shot.get("scene_id", "") or "").strip()
            if sid:
                shots_by_scene_order.setdefault(sid, []).append(shot)

        normalized_all_shots: list[dict] = []
        for scene in scenes:
            sid = str(scene.get("scene_id", "") or "").strip()
            scene_shots = shots_by_scene_order.get(sid, [])
            previous_events: list[dict] = []
            for position, shot in enumerate(scene_shots):
                events = shot.get("dialogue_events", [])
                if not isinstance(events, list) or not events:
                    if previous_events:
                        previous_events[-1]["continues_to_next_shot"] = False
                    previous_events = []
                    normalized_all_shots.append(shot)
                    continue

                for event in events:
                    if isinstance(event, dict):
                        event["continues_from_previous_shot"] = bool(
                            event.get("continues_from_previous_shot", False)
                        )
                        event["continues_to_next_shot"] = bool(
                            event.get("continues_to_next_shot", False)
                        )

                if position == 0:
                    events[0]["continues_from_previous_shot"] = False
                    if len(events) > 1:
                        for event in events[1:]:
                            event["continues_from_previous_shot"] = False
                else:
                    previous_flag = bool(
                        previous_events[-1].get("continues_to_next_shot", False)
                    ) if previous_events else False
                    current_flag = bool(
                        events[0].get("continues_from_previous_shot", False)
                    )
                    continuation = previous_flag or current_flag
                    if not previous_events:
                        continuation = False
                    if previous_events:
                        previous_events[-1]["continues_to_next_shot"] = continuation
                    events[0]["continues_from_previous_shot"] = continuation
                    if len(events) > 1:
                        for event in events[1:]:
                            event["continues_from_previous_shot"] = False

                previous_events = [
                    event for event in events if isinstance(event, dict)
                ]
                normalized_all_shots.append(shot)

        all_shots = normalized_all_shots

        # Canonicalize and semantically filter dialogue BEFORE compilation so
        # CinematicCompiler can never embed invalid speech into h3_prompt.
        self._normalize_dialogue_speakers(
            story,
            all_shots,
            characters,
        )
        self._validate_dialogue_speaker_contract(
            all_shots,
            characters,
        )

        all_shots = CinematicCompiler(
            character_names=character_names,
        ).compile_all(
            scenes,
            all_shots,
        )

        fallback_shot_count = sum(
            1
            for shot in all_shots
            if "_shot_fallback_" in str(shot.get("shot_id", ""))
        )
        print(
            "[DIRECTOR] creative shot coverage: "
            f"{len(all_shots) - fallback_shot_count}/{len(all_shots)} Qwen-generated",
            flush=True,
        )
        if fallback_shot_count:
            self._record_recovery(
                "shot_fallback_templates_used",
                f"count={fallback_shot_count}",
            )

        for scene in scenes:

            scene["shot_ids"] = [
                shot["shot_id"]
                for shot in all_shots
                if shot.get("scene_id") == scene.get("scene_id")
            ]

        self._validate_shot_character_contract(
            all_shots,
            characters,
        )
        self._validate_dialogue_speaker_contract(
            all_shots,
            characters,
        )

        self._validate_production_quality(
            mode=mode,
            story=story,
            scenes=scenes,
            shots=all_shots,
            characters=characters,
        )

        final_director_plan = {
            "story": story,
            "story_mode": mode,
            "director_notes": director_notes,
            "visual_language": visual_language,
            "characters": characters,
            # Preserve the verified canonical roster marker produced by the
            # planner. Without this marker, the orchestrator correctly falls
            # back to its pre-director roster, which is empty for AI_STORY
            # before Qwen has generated the final narrative.
            "_canonical_character_roster_verified": True,
            "scenes": scenes,
            "shots": all_shots,
        }

        self._print_qwen_summary("POST-GENERATION")

        if not os.getenv("H3_DIRECTOR_CRITIC", "1").strip().lower() in {"1", "true", "yes", "on"}:
            self._print_qwen_summary("FINAL")

        self._save_checkpoint(
            checkpoint_store,
            checkpoint_session_id,
            self._checkpoint_state(
                checkpoint_session_id or "",
                mode,
                user_input,
                base_plan,
                final_director_plan,
                "director_completed",
                "director_complete",
                [
                    str(scene.get("scene_id", "")).strip()
                    for scene in scenes
                    if str(scene.get("scene_id", "")).strip()
                ],
                "",
                "",
            ),
        )

        return {
            "enabled": True,
            "plan": final_director_plan,
            "director_notes": director_notes,
        }

    @staticmethod
    def _normalize_dialogue_text(value: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(value or "").strip().strip('"“”‘’'),
        ).lower()

    @classmethod
    def _extract_story_spoken_texts(cls, story: str) -> dict[str, set[str]]:
        """Extract spoken-text anchors and preserve explicit source speaker labels.

        Quoted speech contributes an anchor with no explicit source speaker. Simple
        script labels such as ``Eli: Hello`` or ``Eli — Hello`` contribute the same
        normalized anchor plus the normalized source speaker label. Multiple source
        labels for the same text are preserved so repeated dialogue never gets
        silently assigned to one speaker. Screen/UI text inside quotes is excluded.
        """
        text = str(story or "")
        anchors: dict[str, set[str]] = {}

        def _add_anchor(spoken: str, source_speaker: str | None = None) -> None:
            normalized = cls._normalize_dialogue_text(spoken)
            if not normalized:
                return
            speakers = anchors.setdefault(normalized, set())
            if source_speaker:
                normalized_speaker = EntityResolver.normalize(source_speaker)
                if normalized_speaker:
                    speakers.add(normalized_speaker)

        quote_pattern = re.compile(
            r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)',
            flags=re.UNICODE,
        )
        screen_context = re.compile(
            r"\b(?:on|from|across|over|inside)\s+(?:the\s+)?(?:screen|monitor|display|terminal)\b|"
            r"\b(?:screen|monitor|display|terminal)\b.{0,80}\b(?:read|reads|show|shows|display|displayed|displays|flash|flashed|flashes|appear|appeared|appears|message|text)\b|"
            r"\b(?:read|reads|show|shows|display|displayed|displays|flash|flashed|flashes|appear|appeared|appears)\b.{0,80}\b(?:screen|monitor|display|terminal)\b|"
            r"\b(?:message|text|label|caption)\b.{0,80}\b(?:on|over|across|inside|appeared|displayed|flashed|read|shows|shown)\b.{0,40}\b(?:screen|monitor|display|terminal)\b",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in quote_pattern.finditer(text):
            value = next((part for part in match.groups() if part), "")
            if not value.strip():
                continue
            prefix_window = text[max(0, match.start() - 180):match.start()]
            prefix = re.split(r"[.!?][\"”’]?\s+", prefix_window)[-1]
            suffix_window = text[match.end():match.end() + 100]
            suffix = re.split(r"[.!?]\s+", suffix_window, maxsplit=1)[0]
            suffix_context = re.sub(r"^[,;:\s]+", "", suffix)
            if screen_context.search(prefix) or re.search(
                r"^(?:is|was|were|appears|appeared|appearing|shows|showed|display|displayed|displays|displaying|reads|read|flashed|flashes|shown|showing)\b.{0,80}\b(?:on|in|across|inside)\s+(?:the\s+)?(?:screen|monitor|display|terminal)\b",
                suffix_context,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                continue
            _add_anchor(value)

        label_pattern = re.compile(
            r"(?m)^\s*([A-Z][A-Za-z0-9.'’\-]*(?:\s+[A-Z][A-Za-z0-9.'’\-]*){0,4})\s*(?::|—|–)\s*([^\n]+?)\s*$"
        )
        for match in label_pattern.finditer(text):
            speaker = match.group(1).strip()
            spoken = match.group(2).strip()
            if speaker and spoken:
                _add_anchor(spoken, speaker)

        return anchors

    def _normalize_dialogue_speakers(
        self,
        story: str,
        shots: list[dict],
        characters: list[dict],
    ) -> None:
        """Canonicalize speakers and remove dialogue not anchored in explicit speech."""
        allowed_names = [
            str(value.get("name", "")).strip()
            for value in characters
            if isinstance(value, dict)
            and str(value.get("name", "")).strip()
        ]
        if not allowed_names:
            return

        canonical_by_norm = {name.lower(): name for name in allowed_names}
        aliases = EntityResolver.build_alias_map(allowed_names)
        spoken_anchors = self._extract_story_spoken_texts(story)

        def _resolve(value: str) -> str | None:
            normalized = EntityResolver.normalize(value)
            if normalized in canonical_by_norm:
                return canonical_by_norm[normalized]
            resolved = aliases.get(normalized)
            if resolved and resolved in canonical_by_norm:
                return canonical_by_norm[resolved]
            stripped = EntityResolver.strip_honorific(normalized)
            resolved = aliases.get(stripped)
            if resolved and resolved in canonical_by_norm:
                return canonical_by_norm[resolved]
            return None

        for shot in shots:
            if not isinstance(shot, dict):
                continue

            shot_id = str(shot.get("shot_id", "")).strip()
            bound = {
                canonical.lower()
                for canonical in (
                    _resolve(str(name))
                    for name in (shot.get("characters", []) or [])
                )
                if canonical
            }

            raw_events = shot.get("dialogue_events", [])
            events = raw_events if isinstance(raw_events, list) else []
            repaired_events = []

            for event in events:
                if not isinstance(event, dict):
                    continue
                speaker = str(event.get("speaker", "") or "").strip()
                text = str(event.get("text", "") or "").strip()
                if not speaker or not text:
                    continue

                # When the source story contains explicit speech anchors, only
                # anchored speech can become audio. Substring matching supports
                # a quoted line split into multiple valid events while still
                # rejecting whole narrative/action sentences.
                normalized_text = self._normalize_dialogue_text(text)
                if not spoken_anchors:
                    continue

                matched_source_speakers: set[str] = set()
                matched_any = False
                for anchor, source_speakers in spoken_anchors.items():
                    if (
                        normalized_text == anchor
                        or normalized_text in anchor
                        or anchor in normalized_text
                    ):
                        matched_any = True
                        matched_source_speakers.update(source_speakers)
                if not matched_any:
                    continue

                canonical = _resolve(speaker)
                if canonical is None:
                    # Qwen may attribute grounded speech to a role label that
                    # never became a canonical character. Do not invent an
                    # identity or abort the whole production; discard only the
                    # unresolved dialogue event and record the recovery.
                    self._record_recovery(
                        "dialogue_speaker_unresolved",
                        f"shot={shot_id} speaker={speaker!r}",
                    )
                    continue

                explicit_source_canonicals = {
                    resolved.lower()
                    for source_speaker in matched_source_speakers
                    if (resolved := _resolve(source_speaker)) is not None
                }
                if matched_source_speakers and not explicit_source_canonicals:
                    # The source explicitly labels this line, but that label does
                    # not resolve to a canonical character. Never guess which
                    # canonical character Qwen intended.
                    self._record_recovery(
                        "dialogue_source_speaker_unresolved",
                        f"shot={shot_id} speaker={speaker!r} source={sorted(matched_source_speakers)!r}",
                    )
                    continue

                normalized_speaker = canonical.lower()
                if explicit_source_canonicals and normalized_speaker not in explicit_source_canonicals:
                    # The source gives explicit speaker provenance that conflicts
                    # with Qwen's attribution. Preserve the source contract rather
                    # than silently remapping the line to another character.
                    self._record_recovery(
                        "dialogue_speaker_source_mismatch",
                        f"shot={shot_id} speaker={speaker!r} source={sorted(explicit_source_canonicals)!r}",
                    )
                    continue

                if bound and normalized_speaker not in bound:
                    # The line is real speech and the identity is canonical, but
                    # Qwen bound it to a character that is not present in this
                    # shot. Do not invent a new binding or abort the production;
                    # discard only the inconsistent dialogue event and preserve
                    # the strict post-normalization validator as a safety net.
                    self._record_recovery(
                        "dialogue_speaker_unbound",
                        f"shot={shot_id} speaker={speaker!r}",
                    )
                    continue

                repaired = dict(event)
                repaired["speaker"] = canonical
                repaired_events.append(repaired)

            shot["dialogue_events"] = repaired_events
            shot["speaking_characters"] = list(dict.fromkeys(
                str(event["speaker"]).strip()
                for event in repaired_events
                if str(event.get("speaker", "")).strip()
            ))
            shot["speech_text"] = " ".join(
                str(event.get("text", "")).strip()
                for event in repaired_events
                if str(event.get("text", "")).strip()
            )

    def _validate_dialogue_speaker_contract(
        self,
        shots: list[dict],
        characters: list[dict],
    ) -> None:
        """Pure post-normalization dialogue contract validation."""
        allowed_names = {
            str(value.get("name", "")).strip().lower()
            for value in characters
            if isinstance(value, dict)
            and str(value.get("name", "")).strip()
        }
        if not allowed_names:
            return

        for shot in shots:
            if not isinstance(shot, dict):
                continue
            shot_id = str(shot.get("shot_id", "")).strip()
            bound = {
                str(name).strip().lower()
                for name in (shot.get("characters", []) or [])
                if str(name).strip()
            }
            events = shot.get("dialogue_events", [])
            if not isinstance(events, list):
                raise RuntimeError(
                    f"Shot {shot_id} dialogue_events must be a list."
                )
            expected_speakers = []
            for event in events:
                if not isinstance(event, dict):
                    raise RuntimeError(
                        f"Shot {shot_id} contains a non-object dialogue event."
                    )
                speaker = str(event.get("speaker", "") or "").strip().lower()
                text = str(event.get("text", "") or "").strip()
                if not speaker or not text:
                    raise RuntimeError(
                        f"Shot {shot_id} contains an empty dialogue speaker/text."
                    )
                if speaker not in allowed_names:
                    raise RuntimeError(
                        f"Shot {shot_id} contains unknown dialogue speaker '{event.get('speaker', '')}'."
                    )
                if bound and speaker not in bound:
                    raise RuntimeError(
                        f"Shot {shot_id} has dialogue speaker '{event.get('speaker', '')}' not present in its character bindings."
                    )
                expected_speakers.append(event["speaker"])

            actual_speakers = [
                str(name).strip()
                for name in (shot.get("speaking_characters", []) or [])
                if str(name).strip()
            ]
            if actual_speakers != list(dict.fromkeys(expected_speakers)):
                raise RuntimeError(
                    f"Shot {shot_id} speaking_characters is inconsistent with dialogue_events."
                )

            expected_speech_text = " ".join(
                str(event.get("text", "")).strip()
                for event in events
                if str(event.get("text", "")).strip()
            )
            if str(shot.get("speech_text", "") or "").strip() != expected_speech_text:
                raise RuntimeError(
                    f"Shot {shot_id} speech_text is inconsistent with dialogue_events."
                )

    @staticmethod
    def _shot_batch_completion_budget(scene_count: int) -> int:
        """Return the completion-token cap for one batched shot request.

        SHOTS_PER_SCENE is a topology constraint, not a token budget.
        Never use it as a completion-token cap.
        """
        count = max(1, int(scene_count))
        return min(
            int(DIRECTOR_MAX_TOKENS),
            max(320, 1400 * count),
        )

    @_with_faulthandler_watchdog
    def critique_plan(self, *, mode: str, user_input: str, plan: dict) -> dict:
        """Run an optional read-only cinematic critique.

        The critic may identify problems but never mutates the canonical plan.
        """
        system_prompt = """
    You are a conservative cinematic production critic.
    Review the supplied production plan for narrative, shot-design, continuity,
    reference-binding, and dialogue/action risks. Return a conservative critique.
    When a fix is warranted, provide a minimal patch for an existing shot using
    only the explicitly allowed creative fields in the schema. Never change scene
    identity, shot identity, characters, reference bindings, timing, or continuity
    state. Do not invent facts that are not present in the plan.
    """.strip()
        def _slim_shot(shot: dict) -> dict:
            return {
                "shot_id": str(shot.get("shot_id", "") or ""),
                "scene_id": str(shot.get("scene_id", "") or ""),
                "camera_shot": str(shot.get("camera_shot", "") or ""),
                "camera_movement": str(shot.get("camera_movement", "") or ""),
                "lens_and_depth_of_field": str(shot.get("lens_and_depth_of_field", "") or ""),
                "lighting": str(shot.get("lighting", "") or ""),
                "mood": str(shot.get("mood", "") or ""),
                "visual_prompt": str(shot.get("visual_prompt", "") or "")[:400],
                "action": str(shot.get("action", "") or "")[:300],
                "dialogue_events": [
                    {
                        "speaker": str(event.get("speaker", "") or ""),
                        "text": str(event.get("text", "") or "")[:240],
                    }
                    for event in (shot.get("dialogue_events", []) or [])
                    if isinstance(event, dict)
                ][:4],
            }

        def _slim_scene(scene: dict) -> dict:
            return {
                "scene_id": str(scene.get("scene_id", "") or ""),
                "title": str(scene.get("title", "") or ""),
                "description": str(scene.get("description", "") or "")[:500],
                "scene_objective": str(scene.get("scene_objective", "") or "")[:300],
                "location": str(scene.get("location", "") or ""),
                "characters": scene.get("characters", []) or [],
            }

        compact = {
            "mode": mode,
            "story": self._compact_story_context(
                str(plan.get("story", user_input) or ""),
                DIRECTOR_CRITIC_STORY_CONTEXT_CHARS,
            ),
            "visual_language": plan.get("visual_language", {}) or {},
            "scenes": [
                _slim_scene(scene)
                for scene in (plan.get("scenes", []) or [])
                if isinstance(scene, dict)
            ],
            "shots": [
                _slim_shot(shot)
                for shot in (plan.get("shots", []) or [])
                if isinstance(shot, dict)
            ],
        }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overall_score": {"type": "number"},
                "status": {"type": "string", "enum": ["pass", "review"]},
                "findings": {"type": "array", "items": {"type": "string"}},
                "shot_findings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "shot_id": {"type": "string"},
                            "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                            "finding": {"type": "string"},
                        },
                        "required": ["shot_id", "severity", "finding"],
                    },
                },
                "recommended_focus": {"type": "array", "items": {"type": "string"}},
                "shot_patches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "shot_id": {"type": "string"},
                            "patch": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "action": {"type": "string"},
                                    "camera_shot": {"type": "string"},
                                    "camera_movement": {"type": "string"},
                                    "lens_and_depth_of_field": {"type": "string"},
                                    "composition_notes": {"type": "string"},
                                    "lighting": {"type": "string"},
                                    "color_temperature": {"type": "string"},
                                    "mood": {"type": "string"},
                                    "visual_prompt": {"type": "string"},
                                    "overall_soundscape": {"type": "string"},
                                    "non_diegetic_music": {"type": "string"},
                                },
                            },
                        },
                        "required": ["shot_id", "patch"],
                    },
                },
            },
            "required": ["overall_score", "status", "findings", "shot_findings", "recommended_focus", "shot_patches"],
        }
        try:
            result = self._chat_json(
                system_prompt,
                json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
                minimum_completion=300,
                temperature=0.10,
                top_p=0.75,
                call_name="director_critique",
                max_completion=1000,
                json_mode=True,
                disable_thinking=True,
                response_schema=schema,
            )
            return result
        finally:
            self._print_qwen_summary("FINAL")

    def enrich_plan(
        self,
        *,
        mode: str,
        user_input: str,
        base_plan: dict,
        checkpoint_session_id: str | None = None,
        resume_state: dict | None = None,
    ) -> dict:

        result = self.generate(
            mode=mode,
            user_input=user_input,
            base_plan=base_plan,
            checkpoint_session_id=checkpoint_session_id,
            resume_state=resume_state,
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

        # Visual language is creative metadata, not production identity.
        # Merge field-by-field so a partial Qwen response cannot erase
        # deterministic/default visual-language fields already present in
        # the base plan. Qwen never gets ownership of unrelated keys.
        base_visual_language = merged.get(
            "visual_language",
            {},
        )
        if not isinstance(base_visual_language, dict):
            base_visual_language = {}

        if isinstance(creative_visual_language, dict):
            for key in (
                "genre_tone",
                "color_palette",
                "lighting_philosophy",
                "camera_philosophy",
                "pacing",
            ):
                value = creative_visual_language.get(key)
                if value not in (None, "", [], {}):
                    base_visual_language[key] = deepcopy(value)

        merged["visual_language"] = base_visual_language

        # Canonical structure is never taken directly from raw creative Qwen
        # metadata. The only exception is the roster that generate() itself
        # marked as verified after deterministic + semantic reconciliation.
        base_characters = deepcopy(
            base_plan.get("characters", [])
            or []
        )

        creative_characters = deepcopy(
            creative.get("characters", [])
            or []
        )

        if creative.get("_canonical_character_roster_verified") is True:
            merged["characters"] = creative_characters or base_characters
        else:
            merged["characters"] = base_characters

        # Propagate the verification flag itself. Without this, the
        # orchestrator's boundary check (which relies on this exact key
        # to decide whether it may trust the roster just computed above)
        # always sees it missing and silently discards a correctly
        # verified, story-derived roster in favor of its own premise-
        # derived one -- which is empty for AI Story / Expand Story mode,
        # since the premise rarely names the characters Qwen goes on to
        # invent in the final story.
        merged["_canonical_character_roster_verified"] = (
            creative.get("_canonical_character_roster_verified") is True
        )

        # Canonical scene topology defaults to the premise-derived base
        # plan, but a verified director pass (generate() succeeded and
        # derived its roster/topology from the FINAL story, not the
        # premise) produces its own scene topology that must take
        # priority. Without this, any scene beyond what the short
        # premise alone produces gets silently dropped later by the
        # valid_scene_ids filter -- discarding real, already-paid-for
        # Qwen shot-batch work for those scenes.
        verified_pass = (
            creative.get("_canonical_character_roster_verified") is True
        )

        creative_scenes = (
            creative.get("scenes", [])
            or []
        )

        premise_scenes = deepcopy(
            base_plan.get("scenes", [])
            or []
        )

        story_derived_scenes = [
            deepcopy(scene)
            for scene in creative_scenes
            if isinstance(scene, dict)
            and str(scene.get("scene_id", "") or "").strip()
        ]

        if verified_pass and story_derived_scenes:
            canonical_scenes = story_derived_scenes
        else:
            canonical_scenes = premise_scenes

        creative_by_id = {
            str(scene.get("scene_id", "") or "").strip(): scene
            for scene in creative_scenes
            if isinstance(scene, dict)
            and str(scene.get("scene_id", "") or "").strip()
        }

        # Creative scene fields may enrich an existing scene, but structural
        # identity/topology remains deterministic.
        protected_scene_fields = {
            "scene_id",
            "order",
            "characters",
            "shot_ids",
        }

        for scene in canonical_scenes:
            sid = str(
                scene.get("scene_id", "") or ""
            ).strip()

            creative_scene = creative_by_id.get(sid)

            if not isinstance(creative_scene, dict):
                continue

            for key, value in creative_scene.items():
                if key in protected_scene_fields:
                    continue
                if value in (None, "", [], {}):
                    continue
                scene[key] = deepcopy(value)

        merged["scenes"] = canonical_scenes

        creative_shots = (
            creative.get("shots", [])
            or []
        )

        if creative_shots:
            valid_scene_ids = {
                str(scene.get("scene_id", "") or "").strip()
                for scene in canonical_scenes
            }

            merged["shots"] = [
                deepcopy(shot)
                for shot in creative_shots
                if isinstance(shot, dict)
                and str(
                    shot.get("scene_id", "") or ""
                ).strip() in valid_scene_ids
            ]
        else:
            merged["shots"] = deepcopy(
                base_plan.get("shots", [])
                or []
            )

        return merged

from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import os
import json
import random
import re

from planner.cinematic_compiler import CinematicCompiler
from planner.entity_resolver import EntityResolver
from pipeline.production_checkpoint import ProductionCheckpoint
from pipeline.dialogue_timeline import DialogueTimeline

from planner.config import (
    AI_STORY_MODE,
    DIRECTOR_CRITIC_STORY_CONTEXT_CHARS,
    DIRECTOR_MAX_TOKENS,
    DIRECTOR_SHOTS_PER_SCENE,
    DIRECTOR_STORY_SCENES,
    STORY_WORDS_PER_SCENE,
    STORY_MIN_WORDS,
    STORY_ACCEPT_MIN_WORDS,
    DIRECTOR_N_CTX,
    DIRECTOR_SHOT_STORY_CONTEXT_CHARS,
    DIRECTOR_STORY_MAX_TOKENS,
    DIRECTOR_STORY_PLAN_MAX_TOKENS,
    DIRECTOR_STORY_PLAN_THINKING_BUDGET,
    DIRECTOR_STORY_WRITER_MAX_TOKENS,
    DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
    DIRECTOR_STORY_THINKING_TEMPERATURE,
    DIRECTOR_STORY_THINKING_TOP_P,
    DIRECTOR_STORY_THINKING_TOP_K,
    DIRECTOR_STORY_THINKING_MIN_P,
    DIRECTOR_STORY_WRITER_TOP_K,
    DIRECTOR_STORY_WRITER_MIN_P,
    DIRECTOR_STORY_PRESENCE_PENALTY,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    director_enabled,
    director_critic_enabled,
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
    MAX_SCENES = DIRECTOR_STORY_SCENES
    SHOTS_PER_SCENE = DIRECTOR_SHOTS_PER_SCENE
    # Creative shot batching may cover up to two fresh adjacent scenes.
    # The runtime selects the largest batch that still fits the 8K context
    # budget with a bounded completion reserve. Missing shots are deterministic
    # fallbacks; no per-scene Qwen recovery is used.
    MAX_SHOT_BATCH_SCENES = 2

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
        self._reference_visual_context: dict[str, dict] = {}
        self._character_semantic_calls = 0
        self._semantic_spoken_dialogue: list[dict] = []
        self._last_story_draft: str = ""
        self._story_min_words: int = STORY_ACCEPT_MIN_WORDS

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
        self._semantic_spoken_dialogue = []
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
                normalized_shots = []
                shots_by_scene = {}
                for raw_shot in shots:
                    if isinstance(raw_shot, dict):
                        shots_by_scene.setdefault(str(raw_shot.get("scene_id", "")).strip(), []).append(raw_shot)
                for scene in scenes:
                    if not isinstance(scene, dict):
                        continue
                    scene_id = str(scene.get("scene_id", "")).strip()
                    normalized_shots.extend(
                        self._sanitize_shots(
                            shots_by_scene.get(scene_id, []),
                            scene,
                            characters,
                        )
                    )
                plan["shots"] = CinematicCompiler(
                    character_names=characters
                ).compile_all(scenes, normalized_shots)

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
            # For AI Story / Expand, the single semantic extraction call is required before resume.
            # All modes use the same single semantic extraction stage; resume restores its exact state.
            try:
                self._character_semantic_calls = max(
                    0,
                    min(1, int(prior_director_plan.get("_character_semantic_calls_used", 0) or 0)),
                )
            except (TypeError, ValueError):
                self._character_semantic_calls = 0

            restored_semantic_dialogue = prior_director_plan.get("_semantic_spoken_dialogue", [])
            if isinstance(restored_semantic_dialogue, list):
                self._semantic_spoken_dialogue = [
                    {
                        "text": str(item.get("text", "") or "").strip(),
                        "speaker": str(item.get("speaker", "") or "").strip(),
                        **({"quote_id": int(item["quote_id"])} if str(item.get("quote_id", "")).isdigit() else {}),
                    }
                    for item in restored_semantic_dialogue
                    if isinstance(item, dict)
                    and str(item.get("text", "") or "").strip()
                    and str(item.get("speaker", "") or "").strip()
                ]

        # Two distinct, bounded stages: Architect (thinking ON), then Author (thinking OFF).
        # Each stage is invoked exactly once. No retries, repairs, best-of-N or rescue calls.
        temperature, top_p = self._sampling_for_mode(mode)

        # ----------------------------------------------------
        # PASS 1A: bounded-thinking Architect + single non-thinking Author
        # ----------------------------------------------------

        generated_story = False
        # Only Expand Story has source-canon character anchors. AI Story must not
        # inherit a cast from a deterministic/template base plan: Qwen authors it.
        source_character_names: list[str] = (
            [
                str(character.get("name", "")).strip()
                for character in (base_plan.get("characters", []) or [])
                if isinstance(character, dict) and str(character.get("name", "")).strip()
            ]
            if mode == EXPAND_USER_STORY_MODE
            else []
        )

        if resuming and prior_director_plan.get(
            "story"
        ):

            story = self._normalize_story(
                prior_director_plan.get(
                    "story",
                    "",
                )
            )


        elif mode == PRESERVE_USER_STORY_MODE:

            story = self._normalize_story(
                user_input
            )


        else:

            story = ""

            # Architect request (thinking ON): one normalized 12-scene causal plan + CAST line.
            story_blueprint = self._generate_story_blueprint(
                mode,
                user_input,
                source_character_names=source_character_names,
            )
            # Author request (thinking OFF): one complete prose draft grounded in the plan.
            story_system = self._story_text_system(mode)
            story_user = self._story_text_user(
                mode,
                user_input,
                source_character_names=source_character_names,
                story_blueprint=story_blueprint,
            )
            story, generated_story = self._generate_story_once(
                mode,
                user_input,
                story_system,
                story_user,
                temperature=temperature,
                top_p=top_p,
                source_character_names=source_character_names,
                story_blueprint=story_blueprint,
            )

        # ----------------------------------------------------
        # PASS 1B: deterministic production foundation
        # ----------------------------------------------------
        #
        # Qwen supplies the final creative narrative AND semantic character roster.
        # ProductionPlanner performs only bounded safety validation/canonicalization
        # before binding those Qwen identities into scenes and shots.
        story = self._normalize_story(story)
        if mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE) and not generated_story:
            # Resume/checkpoint safety only: generated stories have already passed the
            # authoritative raw-story validation above. Older checkpoints may predate
            # the configured scene-paragraph contract; repair without changing prose.
            paragraphs = [
                part.strip()
                for part in re.split(r"\n\s*\n+", story)
                if part.strip()
            ]
            if len(paragraphs) != DIRECTOR_STORY_SCENES:
                story = self._coerce_story_to_scene_paragraphs(story)
            if mode == EXPAND_USER_STORY_MODE:
                self._validate_expand_story_cast(
                    user_input,
                    story,
                    source_character_names=source_character_names,
                )

        metadata_source = (
            prior_director_plan
            if resuming
            else base_plan
        )
        director_notes = str(
            metadata_source.get("director_notes", "") or ""
        ).strip()

        visual_language = self._sanitize_visual_language(
            metadata_source.get("visual_language", {})
        )
        for key, value in self._baseline_visual_language().items():
            if not visual_language.get(key):
                visual_language[key] = value

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
        # the supplied user story remains the character/story source of truth;
        # the single semantic Qwen pass is used only for dialogue attribution.
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
                canonical_characters.append(Character.from_dict(deepcopy(item)))

            # The semantic dialogue contract is checkpoint state. Never re-run the semantic
            # Qwen extraction during resume: doing so would make checkpoint continuation
            # nondeterministic and could disagree with the roster already used by earlier shots.
            if "_semantic_spoken_dialogue" not in prior_director_plan:
                raise RuntimeError(
                    "Director checkpoint is incompatible with the semantic dialogue contract: "
                    "_semantic_spoken_dialogue was not persisted. Restart from the narrative stage."
                )
        else:
            if mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
                # Every creative story gets exactly one Qwen semantic character pass.
                # Never bypass it because a deterministic scan appears "named-only".
                canonical_characters = planner.create_characters(
                    canonical_source_story,
                    qwen_character_extractor=self.extract_character_entities,
                )
            else:
                # Preserve Story uses one semantic extraction pass for BOTH the complete
                # cast and dialogue, even when the source has no quote marks. Deterministic
                # name discovery is not authoritative and can miss aliases or late entrants.
                canonical_characters = planner.create_characters(
                    canonical_source_story,
                    qwen_character_extractor=self.extract_character_entities,
                )
                if self._story_has_explicit_dialogue_surface(story) and not self._semantic_spoken_dialogue:
                    self._record_recovery(
                        "preserve_semantic_dialogue_empty",
                        "Preserve Story contains an explicit dialogue surface, but the semantic extraction returned no anchored speech items.",
                    )

        character_payloads = []
        for character in canonical_characters:
            if character is None:
                continue
            payload = character.to_dict()
            metadata = getattr(character, "_semantic_identity_metadata", None)
            if isinstance(metadata, dict) and metadata:
                profile = dict(payload.get("identity_profile", {}) or {})
                profile.update(metadata)
                payload["identity_profile"] = profile
            character_payloads.append(payload)

        characters = self._sanitize_characters(
            character_payloads
        )

        if not characters:
            raise RuntimeError(
                "No canonical characters could be derived from the final story."
            )
        self._validate_generated_story_cast_size(mode, characters)

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
            minimum_scenes=(1 if mode == PRESERVE_USER_STORY_MODE else 4),
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

        minimum_scene_count = 1 if mode == PRESERVE_USER_STORY_MODE else 4
        if len(scenes) < minimum_scene_count or len(scenes) > self.MAX_SCENES:
            raise RuntimeError(
                "Deterministic base plan scene count is outside "
                f"the required {minimum_scene_count}-{self.MAX_SCENES} range for {mode}: {len(scenes)}"
            )

        # Preserve deterministic scene topology. Scene-specific dramatic purpose
        # is inferred by Qwen in the existing shot-batch call, not guessed by index.
        scenes = self._annotate_scene_context(scenes)

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
            "_semantic_spoken_dialogue": deepcopy(self._semantic_spoken_dialogue),
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
                    checkpoint_store=checkpoint_store,
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

        scene_id = ""

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
                    # Fit the largest candidate batch first. When two scenes do
                    # not fit, compact only the global story recap in stages; the
                    # per-scene descriptions, dialogue whitelist, canonical cast,
                    # and continuity evidence remain intact. Shrink to one scene
                    # only if the smaller recap still cannot fit.
                    batch_context_fits = False
                    batch_user = ""
                    prompt_tokens = 0
                    desired_completion = 0
                    chosen_context_chars = None
                    while True:
                        base_user = self._shot_director_batch_user(
                            story,
                            characters,
                            batch_scenes,
                            visual_language,
                        )
                        desired_completion = self._shot_batch_completion_budget(
                            len(batch_scenes)
                        )
                        prompt_limit = int(DIRECTOR_N_CTX) - 128 - desired_completion
                        try:
                            base_payload = json.loads(base_user)
                        except (TypeError, json.JSONDecodeError):
                            base_payload = None

                        context_caps = (
                            self._shot_batch_story_context_capacities(
                                DIRECTOR_SHOT_STORY_CONTEXT_CHARS
                            )
                            if base_payload is not None
                            else [DIRECTOR_SHOT_STORY_CONTEXT_CHARS]
                        )
                        candidate_users = []
                        seen_candidate_users = set()
                        for context_chars in context_caps:
                            if base_payload is not None:
                                candidate_payload = dict(base_payload)
                                if context_chars != DIRECTOR_SHOT_STORY_CONTEXT_CHARS:
                                    candidate_payload["story_context"] = self._compact_story_context(
                                        story, context_chars
                                    )
                                candidate_user = json.dumps(
                                    candidate_payload,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                            else:
                                candidate_user = base_user
                            if candidate_user not in seen_candidate_users:
                                candidate_users.append((candidate_user, context_chars))
                                seen_candidate_users.add(candidate_user)

                        fitted_user = None
                        fitted_tokens = None
                        fitted_chars = None
                        for candidate_user, candidate_chars in candidate_users:
                            candidate_tokens = self._count_tokens(
                                self._shot_director_batch_system()
                                + "\n\n"
                                + candidate_user
                            )
                            if candidate_tokens <= prompt_limit:
                                fitted_user = candidate_user
                                fitted_tokens = candidate_tokens
                                fitted_chars = candidate_chars
                                break

                        if fitted_user is not None:
                            batch_user = fitted_user
                            prompt_tokens = int(fitted_tokens)
                            chosen_context_chars = fitted_chars
                            batch_context_fits = True
                            if (
                                len(batch_scenes) > 1
                                and chosen_context_chars is not None
                                and chosen_context_chars < DIRECTOR_SHOT_STORY_CONTEXT_CHARS
                            ):
                                print(
                                    "[DIRECTOR] compacted global story recap for shot batch "
                                    f"scenes={len(batch_scenes)} chars={chosen_context_chars}/"
                                    f"{DIRECTOR_SHOT_STORY_CONTEXT_CHARS} prompt_tokens={prompt_tokens} "
                                    f"limit={prompt_limit}; scene evidence/dialogue preserved",
                                    flush=True,
                                )
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

                    batch_response = None
                    batch_dialogue_index: dict[str, dict[str, dict]] = {}
                    batch_dialogue_ids: list[str] = []
                    if not batch_context_fits:
                        self._record_recovery(
                            "shot_batch_context_budget_fallback",
                            "Even a single-scene prompt exceeded the context budget after "
                            "global-story recap compaction; used deterministic shot fallback.",
                        )
                    else:
                        batch_dialogue_index, _batch_dialogue_ids = self._shot_batch_dialogue_index(
                            batch_user
                        )
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
                                    dialogue_ids_by_scene={
                                        scene_id: list(source_index.keys())
                                        for scene_id, source_index in batch_dialogue_index.items()
                                    },
                                ),
                            )
                        except RuntimeError as batch_error:
                            self._record_recovery(
                                "shot_batch_deterministic_fallback",
                                str(batch_error),
                            )

                    # Model/runtime failure is the only recovery boundary here. Parsing,
                    # sanitization and invariant errors are intentionally outside the catch so
                    # programmer defects cannot be misclassified as an ordinary Qwen fallback.
                    if batch_response is not None:
                        scene_metadata_map = self._normalize_batch_scene_metadata_response(
                            batch_response
                        )
                        for target_scene in batch_scenes:
                            target_id = str(target_scene.get("scene_id", "") or "").strip()
                            self._apply_scene_metadata(
                                target_scene, scene_metadata_map.get(target_id, {})
                            )

                        batch_map = self._normalize_batch_shot_response(
                            batch_response
                        )

                        for target_scene in batch_scenes:
                            target_id = str(
                                target_scene.get("scene_id", "") or ""
                            ).strip()

                            referenced_shots = self._expand_shot_batch_dialogue_references(
                                batch_map.get(target_id, []),
                                batch_dialogue_index.get(target_id, {}),
                            )
                            candidate = self._sanitize_shots(
                                referenced_shots,
                                target_scene,
                                character_names,
                            )

                            if candidate:
                                generated_by_scene[target_id] = candidate[
                                    : self.SHOTS_PER_SCENE
                                ]

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
                            checkpoint_store=checkpoint_store,
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
                    scene_id,
                    str(exc),
                    checkpoint_store=checkpoint_store,
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

        # Scene-level coherence: one paragraph == one scene in one place under one
        # lighting setup. Qwen is free to vary framing, lens and movement between the
        # shots of a scene, but not to teleport the location or flip the light source.
        self._enforce_scene_coherence(
            scenes,
            all_shots,
            characters,
        )

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

        # Canonicalize continuation flags before semantic speaker filtering.
        # The pass is alias-aware, so a valid continuation may survive when Qwen
        # uses a safe alias (for example, a first-name form) for the same canonical
        # character. Speaker changes are never allowed to inherit continuation.
        # A second final pass is performed after filtering below because dialogue
        # normalization may remove boundary events.
        self._normalize_dialogue_continuations(
            scenes,
            all_shots,
            characters,
        )

        # Canonicalize and semantically filter dialogue BEFORE compilation so
        # CinematicCompiler can never embed invalid speech into h3_prompt.
        self._normalize_dialogue_speakers(
            story,
            scenes,
            all_shots,
            characters,
        )

        final_dialogue_count = sum(
            len(shot.get("dialogue_events", []) or [])
            for shot in all_shots
            if isinstance(shot, dict)
        )
        if self._semantic_spoken_dialogue and final_dialogue_count == 0:
            if mode == EXPAND_USER_STORY_MODE:
                recovery_code = "expand_semantic_dialogue_empty"
            elif mode == PRESERVE_USER_STORY_MODE:
                recovery_code = "preserve_semantic_dialogue_empty"
            else:
                recovery_code = "ai_story_semantic_dialogue_empty"
            self._record_recovery(
                recovery_code,
                "Semantic extraction returned spoken dialogue, but no dialogue events survived deterministic reconciliation.",
            )
        elif (
            mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE)
            and not self._semantic_spoken_dialogue
            and self._story_has_explicit_dialogue_surface(story)
        ):
            if mode == EXPAND_USER_STORY_MODE:
                recovery_code = "expand_semantic_dialogue_empty"
            else:
                recovery_code = "ai_story_semantic_dialogue_empty"
            self._record_recovery(
                recovery_code,
                "Story contains explicit dialogue surface, but the single semantic extraction returned no spoken dialogue items.",
            )

        # Dialogue speaker normalization may remove or replace events. That can
        # change which event is actually at a shot boundary, so continuation
        # flags must be canonicalized again against the FINAL dialogue event
        # lists before the timeline scheduler sees the plan. Without this second
        # pass, a removed boundary event can leave stale continuation metadata
        # on either side of a shot boundary.
        self._normalize_dialogue_continuations(
            scenes,
            all_shots,
            characters,
        )

        # Enforce the same H3 dialogue feasibility contract used downstream by
        # DialogueTimeline before the creative plan leaves the Director. This
        # pass never rewrites spoken text or invents dialogue; it only performs
        # deterministic, scene-local redistribution when two adjacent shots can
        # legally carry the existing dialogue in a different partition. A final
        # continuation-aware split is allowed only for an explicitly continuing
        # final event, preserving the original spoken text exactly in order.
        self._normalize_dialogue_h3_feasibility(
            scenes,
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
            "_character_semantic_calls_used": int(self._character_semantic_calls),
            "_semantic_spoken_dialogue": deepcopy(self._semantic_spoken_dialogue),
            "scenes": scenes,
            "shots": all_shots,
        }

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
                checkpoint_store=checkpoint_store,
            ),
        )

        return {
            "enabled": True,
            "plan": final_director_plan,
            "director_notes": director_notes,
        }

    def _validate_story_output_contracts(
        self,
        mode: str,
        user_input: str,
        story: str,
        source_character_names: list[str] | None = None,
        *,
        strict: bool = True,
    ) -> str:
        """Validate the current story without rewriting its topology.

        Raw model output is authoritative. The single-pass production writer uses strict=True:
        detected premise-coverage, sentence-depth, completion, or ending-contract defects fail
        before entity extraction and shot batches. No retry, repair call, or paragraph reshaping is allowed.
        Paragraph topology is validated after explicit scene markers are normalized.
        """
        working = self._normalize_story(story)
        errors: list[str] = []
        # Prevent a warning from one draft leaking into a later resume/checkpoint.
        self._last_story_contract_warnings = []

        try:
            self._validate_mode_output(mode, user_input, working)
        except RuntimeError as exc:
            errors.append(str(exc))

        if errors:
            self._last_story_contract_warnings = list(errors)
            detail = " | ".join(errors)
            if strict:
                raise RuntimeError(detail)
            # Single-pass generation must not make another Qwen request just to
            # satisfy a soft length/source-coverage heuristic. Keep the draft, but
            # make the miss visible so PASS never means semantic quality was proven.
            print("[QWEN] story_contract=WARNING " + detail[:900], flush=True)
        return working

    @staticmethod
    def _validate_generated_story_cast_size(mode: str, characters: list[dict]) -> None:
        """Validate that a cast exists without imposing a numeric cast-size quota."""
        if mode not in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
            return
        names = []
        seen = set()
        for item in characters or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "") or "").strip()
            key = EntityResolver.normalize(name)
            if name and key and key not in seen:
                names.append(name)
                seen.add(key)
        if not names:
            raise RuntimeError(
                "Generated-story cast contract failed: no named character could be canonicalized. "
                "A non-empty cast is required for production planning; there is no numeric cast-size quota. "
                "No shot batches were launched and no story retry was attempted."
            )

    def _validate_expand_story_cast(
        self,
        source_story: str,
        generated_story: str,
        source_character_names: list[str] | None = None,
    ) -> None:
        """Compatibility check: established source anchors must survive Expand.

        New character identity is intentionally NOT restricted here. Qwen semantic
        extraction owns the final cast; this helper only checks source-anchor continuity.
        """
        required = [
            str(name).strip()
            for name in (source_character_names or [])
            if str(name).strip()
        ]
        if not required:
            return
        planner = self._planner()
        missing = [
            name
            for name in required
            if planner.identity_presence(generated_story, name) == "absent"
        ]
        if missing:
            raise RuntimeError(
                "Expand Story omitted established source character(s): "
                + ", ".join(missing)
            )

    # ------------------------------------------------------------------
    # Story generation: one thinking-enabled call, deterministic checks, no retries
    # ------------------------------------------------------------------
    def _salvage_runaway_story(self, text: str) -> str:
        """Recover a clean story from a draft that kept generating after it ended."""
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n+", str(text or "").replace("\r\n", "\n").strip())
            if part.strip()
        ]
        seen: set[str] = set()
        kept: list[str] = []
        for part in paragraphs:
            key = re.sub(r"\W+", " ", part.lower()).strip()[:160]
            if key in seen:
                break  # the loop starts here
            seen.add(key)
            kept.append(part)
        kept = kept[:DIRECTOR_STORY_SCENES]
        while kept and not self._ends_cleanly(kept[-1]):
            kept.pop()
        return "\n\n".join(kept) if len(kept) >= max(4, int(DIRECTOR_STORY_SCENES * 0.8)) else ""

    def _trim_to_last_sentence(self, text: str) -> str:
        """Cut a draft back to its last complete sentence; '' if too little remains."""
        value = str(text or "").replace("\r\n", "\n").strip()
        matches = list(re.finditer(r"[.!?][\"'\u2019\u201d\u00bb)\]]*(?=\s|$)", value))
        if not matches:
            return ""
        value = value[: matches[-1].end()].strip()
        minimum_words = int(getattr(self, "_story_min_words", STORY_MIN_WORDS) or STORY_MIN_WORDS)
        return value if len(re.findall(r"\b[\w'’-]+\b", value)) >= minimum_words else ""

    @staticmethod
    def _trim_repeated_terminal_sentence(story: str) -> tuple[str, bool]:
        """Remove only an exact duplicate of the final sentence in its final paragraph.

        This is a narrow deterministic safeguard for the observed duplicated ending. It
        performs no semantic rewrite, does not remove repeated phrases elsewhere, and
        never triggers a second model call. Ambiguous or quoted boundaries fail closed.
        """
        value = str(story or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", value) if part.strip()]
        if not paragraphs:
            return value, False
        paragraph = paragraphs[-1]

        # Split only at sentence punctuation outside balanced straight/curly quotes.
        closing_for = {"“": "”", "‘": "’"}
        quote_stack: list[str] = []
        spans: list[str] = []
        start = 0
        for index, char in enumerate(paragraph):
            if char in closing_for:
                quote_stack.append(closing_for[char])
                continue
            if char in {"\"", "'", "”", "’"}:
                if char == "'" and index > 0 and index + 1 < len(paragraph):
                    if paragraph[index - 1].isalnum() and paragraph[index + 1].isalnum():
                        continue
                if quote_stack and char == quote_stack[-1]:
                    quote_stack.pop()
                elif not quote_stack and char in {"\"", "'"}:
                    previous = paragraph[index - 1] if index else " "
                    following = paragraph[index + 1] if index + 1 < len(paragraph) else " "
                    if (previous.isspace() or previous in "([{—–:;") and not following.isspace():
                        quote_stack.append(char)
                continue
            if char not in ".!?" or quote_stack:
                continue
            if index + 1 < len(paragraph) and paragraph[index + 1].isspace():
                span = paragraph[start:index + 1].strip()
                if span:
                    spans.append(span)
                cursor = index + 1
                while cursor < len(paragraph) and paragraph[cursor].isspace():
                    cursor += 1
                start = cursor
        tail = paragraph[start:].strip()
        if tail:
            spans.append(tail)
        if len(spans) < 2:
            return value, False

        def normalized(sentence: str) -> str:
            # Ignore only whitespace and capitalization; punctuation/content must match.
            return re.sub(r"\s+", " ", sentence.strip()).casefold()

        final_key = normalized(spans[-1])
        if len(final_key.split()) < 5 or not any(normalized(item) == final_key for item in spans[:-1]):
            return value, False

        # Remove the final sentence from the final paragraph only; preserve every
        # other sentence and all other paragraphs verbatim.
        cut = paragraph.rfind(spans[-1])
        if cut < 0:
            return value, False
        updated_paragraph = paragraph[:cut].rstrip()
        if not updated_paragraph:
            return value, False
        paragraphs[-1] = updated_paragraph
        return "\n\n".join(paragraphs), True

    def _story_quality_issues(self, story: str) -> list[str]:
        """Log-only structural craft diagnostics; never fail or rewrite the story."""
        text = str(story or "")
        words = re.findall(r"[\w'’-]+", text.lower())
        if not words:
            return ["empty story"]

        issues: list[str] = []

        # Repetition metrics are intentionally structural: no stock-phrase or
        # English vocabulary blacklist is used here.
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if trigrams and (1 - len(set(trigrams)) / len(trigrams)) > 0.08:
            issues.append("repeated phrasing")

        vocabulary_ratio = len(set(words)) / len(words)
        if len(words) >= 420 and vocabulary_ratio < 0.46:
            issues.append("low vocabulary variety")

        sentences = [
            sentence
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        starts = [
            re.findall(r"[\w'’-]+", sentence.lower())[:1]
            for sentence in sentences
        ]
        run = 1
        for index in range(1, len(starts)):
            run = run + 1 if starts[index] and starts[index] == starts[index - 1] else 1
            if run >= 3:
                issues.append("three sentences in a row open with the same word")
                break

        lengths = [len(re.findall(r"[\w'’-]+", sentence)) for sentence in sentences]
        if len(lengths) >= 8 and (max(lengths) - min(lengths)) < 9:
            issues.append("monotone sentence rhythm")

        # Log repeated substantial quoted dialogue across different scene paragraphs.
        # This is deliberately advisory: repeated lines may be a purposeful callback,
        # so never delete dialogue or trigger an extra model call from this diagnostic.
        paragraphs = [part for part in re.split(r"\n\s*\n+", text) if part.strip()]
        quote_pattern = re.compile(
            r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)',
            flags=re.UNICODE,
        )
        quote_scenes: dict[str, set[int]] = {}
        for scene_index, paragraph in enumerate(paragraphs):
            for match in quote_pattern.finditer(paragraph):
                quoted = next((value for value in match.groups() if value is not None), "")
                normalized = re.sub(r"[^a-z0-9]+", " ", quoted.casefold()).strip()
                if len(normalized.split()) < 5:
                    continue
                quote_scenes.setdefault(normalized, set()).add(scene_index)
        if any(len(scene_indexes) > 1 for scene_indexes in quote_scenes.values()):
            issues.append("substantial quoted dialogue repeats across scenes")

        return issues

    def _generate_story_blueprint(
        self,
        mode: str,
        user_input: str,
        *,
        source_character_names: list[str] | None = None,
    ) -> str:
        """Create one compact 12-scene plan with a hard Qwen reasoning-token budget."""
        from planner.qwen_director_runtime import DIRECTOR_VLLM_SEED, StoryTruncated

        call_name = (
            "ai_story_architecture_pass"
            if mode == AI_STORY_MODE
            else "expand_story_architecture_pass"
        )
        system_prompt = self._story_blueprint_system(mode)
        user_prompt = self._story_blueprint_user(
            mode,
            user_input,
            source_character_names=source_character_names,
        )
        print(
            f"[QWEN] story_architecture=bounded thinking_budget={DIRECTOR_STORY_PLAN_THINKING_BUDGET} "
            f"max_tokens={DIRECTOR_STORY_PLAN_MAX_TOKENS} "
            f"scenes={DIRECTOR_STORY_SCENES} retry=false",
            flush=True,
        )
        try:
            raw = self._chat_text(
                system_prompt,
                user_prompt,
                minimum_completion=400,
                temperature=float(DIRECTOR_STORY_THINKING_TEMPERATURE),
                top_p=float(DIRECTOR_STORY_THINKING_TOP_P),
                call_name=call_name,
                max_completion=min(
                    int(DIRECTOR_STORY_PLAN_MAX_TOKENS),
                    int(DIRECTOR_STORY_MAX_TOKENS),
                ),
                disable_thinking=False,
                seed=(
                    DIRECTOR_VLLM_SEED
                    if mode == EXPAND_USER_STORY_MODE
                    else random.randint(1, 2**31 - 1)
                ),
                creative=False,
                thinking_token_budget=int(DIRECTOR_STORY_PLAN_THINKING_BUDGET),
            )
        except StoryTruncated as exc:
            # Accept a complete 12-beat plan salvaged from a capped response; never
            # make another planning request. Incomplete plans fail explicitly.
            raw = str(exc.partial or "")
            try:
                return self._normalize_story_blueprint(raw, DIRECTOR_STORY_SCENES)
            except RuntimeError as plan_error:
                raise RuntimeError(
                    f"{call_name} hit its token ceiling and the partial output was not a complete "
                    f"{DIRECTOR_STORY_SCENES}-scene blueprint. No retry was attempted."
                ) from plan_error
        return self._normalize_story_blueprint(raw, DIRECTOR_STORY_SCENES)

    def _generate_story_once(
        self,
        mode: str,
        user_input: str,
        story_system: str,
        story_user: str,
        *,
        temperature: float,
        top_p: float,
        source_character_names: list[str],
        story_blueprint: str | None = None,
    ) -> tuple[str, bool]:
        """Write the story once in non-thinking mode from the bounded causal blueprint.

        Paragraph topology is normalized deterministically. Length and completion are
        hard contracts; source-coverage heuristics stay advisory. Nothing is retried.
        """
        from planner.qwen_director_runtime import (
            DIRECTOR_VLLM_SEED,
            StoryTruncated,
        )

        call_name = (
            "ai_story_text_pass"
            if mode == AI_STORY_MODE
            else "expand_story_text_pass"
        )
        failure_prefix = (
            "AI Story generation failed validation: "
            if mode == AI_STORY_MODE
            else "Expand Story generation failed validation: "
        )
        # The YAML global ceiling remains authoritative; the dedicated non-thinking
        # author cap keeps output bounded without spending its allowance on reasoning.
        # The runtime further clamps each request to available 8K-context headroom.
        story_min_output_tokens = DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS
        story_budget = min(
            int(DIRECTOR_STORY_MAX_TOKENS),
            int(DIRECTOR_STORY_WRITER_MAX_TOKENS),
        )
        temperature = float(temperature)
        top_p = float(top_p)
        print(
            f"[QWEN] story_writer=nonthinking max_tokens={story_budget} temperature={temperature:.2f} "
            f"top_p={top_p:.2f} min_tokens={story_min_output_tokens} "
            "single_pass=true retries=0 repairs=0",
            flush=True,
        )

        try:
            raw = self._chat_text(
                story_system,
                story_user,
                minimum_completion=350,
                minimum_output_tokens=story_min_output_tokens,
                temperature=temperature,
                top_p=top_p,
                call_name=call_name,
                max_completion=story_budget,
                disable_thinking=True,
                seed=(DIRECTOR_VLLM_SEED if mode == EXPAND_USER_STORY_MODE else random.randint(1, 2**31 - 1)),
                creative=True,
            )
        except StoryTruncated as exc:
            raw = self._salvage_runaway_story(exc.partial) or self._trim_to_last_sentence(exc.partial)
            if not raw:
                raise RuntimeError(failure_prefix + str(exc)) from exc

        raw = self._strip_story_markdown_emphasis(raw)
        raw, cleanup_reason = self._strip_story_nonprose_tail(raw)
        if cleanup_reason:
            print(f"[QWEN] story_cleanup={cleanup_reason}", flush=True)
        raw = self._reflow_story_paragraphs(raw)
        try:
            raw = self._coerce_story_to_scene_paragraphs(raw)
        except ValueError as exc:
            raise RuntimeError(failure_prefix + str(exc)) from exc
        if not str(raw or "").strip():
            raise RuntimeError(failure_prefix + "Qwen returned an empty story after non-prose cleanup.")
        initial_scene_count = len([part for part in re.split(r"\n\s*\n+", raw) if part.strip()])
        if initial_scene_count != DIRECTOR_STORY_SCENES:
            raise RuntimeError(
                failure_prefix
                + f"Author returned {initial_scene_count} scene paragraphs; exactly {DIRECTOR_STORY_SCENES} are required. "
                "Scene boundaries were preserved; no split, merge, retry, or repair call was attempted."
            )
        self._last_story_draft = raw

        # Length is measured only over cleaned narrative, so checklists, fake word counts,
        # and duplicated blocks cannot satisfy the contract. The acceptance floor equals
        # STORY_MIN_WORDS: an under-length single-pass draft is rejected before any
        # character-extraction or shot-batch calls are spent. No retry or repair call.
        usable_words = len(re.findall(r"\b[\w'’-]+\b", raw))
        if usable_words < self._story_min_words:
            raise RuntimeError(
                failure_prefix + f"Qwen returned only {usable_words} usable story words "
                f"after removing non-story output (minimum production contract {self._story_min_words}+). "
                "The single writer call is not retried and no downstream Qwen calls are launched."
            )
        # One paragraph per configured scene is enforced before quality checks. The
        # scene marker parser above never manufactures missing beats from prose.
        candidate = str(raw or "").strip()

        self._last_story_contract_warnings = []
        candidate = self._validate_story_output_contracts(
            mode,
            user_input,
            candidate,
            source_character_names=source_character_names,
            strict=True,
        )

        # Never accept a dangling generation tail. Word-count is rechecked after any safe trim.
        candidate, removed_duplicate_ending = self._trim_repeated_terminal_sentence(candidate)
        if removed_duplicate_ending:
            print("[QWEN] story_cleanup=removed_exact_duplicate_terminal_sentence", flush=True)

        word_count = len(re.findall(r"\b[\w'’-]+\b", candidate))
        if word_count < self._story_min_words:
            raise RuntimeError(
                failure_prefix + f"Qwen returned only {word_count} usable words "
                f"(need {self._story_min_words}+ to satisfy the configured story-length contract) "
                "check the Qwen server/trace. No retry was attempted."
            )
        if candidate and not self._ends_cleanly(candidate):
            trimmed = self._trim_to_last_sentence(candidate)
            if not trimmed:
                raise RuntimeError(
                    failure_prefix + "The cleaned story ends mid-sentence and cannot be safely trimmed "
                    "while preserving the minimum word count. No retry was attempted."
                )
            candidate = trimmed
            word_count = len(re.findall(r"\b[\w'’-]+\b", candidate))
            if word_count < self._story_min_words:
                raise RuntimeError(
                    failure_prefix + f"Safe sentence-boundary cleanup leaves only {word_count} words "
                    f"(need {self._story_min_words}+). No retry was attempted."
                )
        paragraph_count = len([
            part for part in re.split(r"\n\s*\n+", candidate) if part.strip()
        ])
        if paragraph_count != DIRECTOR_STORY_SCENES:
            raise RuntimeError(
                failure_prefix
                + f"Final story topology must contain exactly {DIRECTOR_STORY_SCENES} scene paragraphs "
                f"(got {paragraph_count}). No retry or paragraph reshaping was attempted."
            )
        issues = self._story_quality_issues(candidate)
        issues.extend(self._story_cast_contract_issues(candidate, story_blueprint))
        issues.extend(self._story_sentence_contract_issues(candidate))
        issues.extend(self._story_craft_issues(candidate))
        issues.extend(
            "contract: " + str(item)
            for item in (getattr(self, "_last_story_contract_warnings", []) or [])
        )
        if issues:
            print(
                "[QWEN] story_static_checks=WARN",
                f"words={word_count}",
                f"paragraphs={paragraph_count}",
                "issues=" + "; ".join(issues[:6]),
                flush=True,
            )
        else:
            print(
                "[QWEN] story_static_checks=PASS",
                f"words={word_count}",
                f"paragraphs={paragraph_count}",
                "semantic_quality=NOT_EVALUATED",
                flush=True,
            )
        return candidate, True

    @staticmethod
    def _blueprint_cast_names(story_blueprint: str | None) -> list[str]:
        """Read plausible person entries from the sanitized Architect CAST line."""
        non_person_role = re.compile(
            r"\b(?:non[- ]human|relentless force|environmental force|weather system|weather phenomenon|sealed chamber|chamber|location|place|building|object|device|machine|artifact|organization|organisation|company|corporation|abstract concept)\b",
            re.IGNORECASE,
        )
        for line in str(story_blueprint or "").splitlines():
            if line.upper().startswith("CAST:"):
                names = []
                for entry in line[5:].split("|"):
                    role = entry[entry.find("(") + 1:entry.rfind(")")] if "(" in entry and ")" in entry else ""
                    if non_person_role.search(role.split(";", 1)[0]):
                        continue
                    name = re.split(r"[(\[:,\u2013\u2014]", entry, maxsplit=1)[0].strip(" *_`")
                    name = re.sub(r"\s+", " ", name)
                    key = EntityResolver.normalize(name)
                    if 1 < len(name) <= 96 and key and key not in {EntityResolver.normalize(x) for x in names}:
                        names.append(name)
                return names
        return []

    @staticmethod
    def _name_surfaces(name: str) -> list[str]:
        """Safe full/titleless/given-name surfaces; surname-only matches are avoided."""
        value = re.sub(r"\s+", " ", str(name or "").strip())
        if not value:
            return []
        titleless = EntityResolver.strip_honorific(value)
        surfaces = [value, titleless]
        tokens = titleless.split()
        if tokens and len(tokens[0]) >= 3:
            surfaces.append(tokens[0])
        return list(dict.fromkeys(x for x in surfaces if x))

    def _story_cast_contract_issues(self, story: str, story_blueprint: str | None) -> list[str]:
        """Log-only cast diagnostics with title/first-name awareness; no numeric cast quota."""
        paragraphs = [p for p in re.split(r"\n\s*\n+", str(story or "")) if p.strip()]
        names = self._blueprint_cast_names(story_blueprint)
        if not names:
            return []
        issues: list[str] = []
        presence: dict[str, list[int]] = {}
        for name in names:
            surfaces = self._name_surfaces(name)
            matched = [
                i for i, paragraph in enumerate(paragraphs)
                if any(re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", paragraph, re.IGNORECASE) for surface in surfaces)
            ]
            presence[name] = matched
            if not matched:
                issues.append(f"cast: planned core character '{name}' has no clear name-form match in the finished prose")
        dialogue_scenes = 0
        for i, paragraph in enumerate(paragraphs):
            present = [name for name in names if i in presence[name]]
            if len(present) >= 2 and re.search(r"[\"\u201c][^\"\u201d]{3,}[\"\u201d]", paragraph):
                dialogue_scenes += 1
        if len(names) >= 2 and dialogue_scenes < 2:
            issues.append(f"cast: only {dialogue_scenes} scene(s) pair 2+ planned core characters with quoted dialogue")
        final_text = paragraphs[-1] if paragraphs else ""
        if re.search(r"\bno survivors?\b", final_text, re.IGNORECASE) and re.search(
            r"\b(?:alive|survived|survivor|hospital|woke up|recovered)\b", " ".join(paragraphs), re.IGNORECASE
        ):
            issues.append("continuity: ending says 'no survivors' while the story also depicts a survivor/alive character")
        return issues

    def _story_sentence_contract_issues(self, story: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", str(story or "")) if p.strip()]
        issues = []
        if len(paragraphs) != DIRECTOR_STORY_SCENES:
            issues.append(f"topology: expected {DIRECTOR_STORY_SCENES} paragraphs, found {len(paragraphs)}")
        for index, paragraph in enumerate(paragraphs, start=1):
            protected = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|Capt|Cmdr|Lt|Sgt|St|Rev|Hon|Sr|Jr)\.(?=\s)", r"\1<dot>", paragraph)
            count = len([x for x in re.split(r"(?<=[.!?])(?:[\"\u201d\u2019']?\s+|$)", protected) if x.strip()])
            if not 6 <= count <= 8:
                issues.append(f"scene {index:02d}: sentence count {count}, expected 6–8")
        return issues

    def _story_craft_issues(self, story: str) -> list[str]:
        """Log-only craft diagnostics; hard contract defects remain separate."""
        issues: list[str] = []
        try:
            planner = self._planner()
            named = planner.detect_character_descriptors(story)
            absent = [name for name in named if planner._is_mention_only_identity(story, name)]
        except Exception:
            absent = []
        if absent:
            issues.append("named people who are never on screen: " + ", ".join(absent[:3]))
        return issues

    @staticmethod
    def _strip_story_markdown_emphasis(raw: str) -> str:
        """Remove leaked Markdown emphasis without rewriting narrative wording."""
        value = str(raw or "").replace("\r\n", "\n").replace("\r", "\n")
        value = re.sub(r"(?<!\w)(\*{1,2})([^*\n]+?)\1(?!\w)", r"\2", value)
        value = re.sub(r"(?<!\w)(_{1,2})([^_\n]+?)\1(?!\w)", r"\2", value)
        return value

    @staticmethod
    def _strip_story_nonprose_tail(raw: str) -> tuple[str, str]:
        """Remove recognizable author commentary or an exact repeated story block.

        The writer is contractually prose-only. This sanitizer is intentionally narrow:
        it removes clear postscript markers after a substantial story, and an exact
        repeated run of three or more whole paragraphs. It never rewrites story prose.
        """
        value = str(raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", value) if part.strip()]
        if not paragraphs:
            return "", ""

        kept: list[str] = []
        word_count = 0
        meta_start = re.compile(
            r"(?ix)^\s*(?:"
            r"\(?\s*(?:word|paragraph)\s+count\b|"
            r"final\s+word\s+count\b|"
            r"note\s*:\s*this\s+story\b|"
            r"story\s+complete\b|"
            r"this\s+concludes\s+the\s+ai[- ]generated\s+story\b|"
            r"\[\s*(?:end\s+of\s+story|word\s+count\s+check)\s*\]|"
            r"total\s*:\s*[*_`]*\d+\s+words\b|"
            r"all\s+requirements\s+met\b|"
            r"here\s+is\s+the\s+(?:completed\s+)?story\b"
            r")"
        )
        separator = re.compile(r"^(?:-{3,}|_{3,}|\*{3,})$")
        for part in paragraphs:
            # Only treat a separator/checklist as a postscript after a meaningful story body.
            if len(kept) >= 6 and word_count >= 350 and separator.fullmatch(part.strip()):
                return "\n\n".join(kept), "removed_nonprose_postscript"
            if len(kept) >= 6 and word_count >= 350 and meta_start.match(part):
                return "\n\n".join(kept), "removed_nonprose_postscript"
            kept.append(part)
            word_count += len(re.findall(r"\b[\w'’-]+\b", part))

        # Catch runaway exact duplication, e.g. all 11 scenes emitted a second time
        # to satisfy an incorrect min_tokens floor. Require a repeated block of >=3
        # exact whole paragraphs after at least 350 narrative words.
        normalized = [re.sub(r"\s+", " ", part).strip().casefold() for part in kept]
        prefix_words = [0]
        for part in kept:
            prefix_words.append(prefix_words[-1] + len(re.findall(r"\b[\w'’-]+\b", part)))
        for second in range(6, len(kept) - 2):
            if prefix_words[second] < 350:
                continue
            for first in range(0, second):
                run = 0
                max_run = min(second - first, len(kept) - second)
                while run < max_run and normalized[first + run] == normalized[second + run]:
                    run += 1
                if run >= 3:
                    return "\n\n".join(kept[:second]), "removed_exact_repeated_story_block"

        return "\n\n".join(kept), ""

    def _trim_runaway_tail(self, raw: str) -> str:
        """Return the configured scene paragraphs when complete and in range; otherwise unchanged."""
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n+", self._normalize_story(raw))
            if part.strip()
        ]
        if len(paragraphs) <= DIRECTOR_STORY_SCENES or len(paragraphs) > DIRECTOR_STORY_SCENES + 3:
            return raw
        head = paragraphs[:DIRECTOR_STORY_SCENES]
        words = len(re.findall(r"\b[\w'’-]+\b", " ".join(head)))
        minimum_words = int(getattr(self, "_story_min_words", STORY_MIN_WORDS) or STORY_MIN_WORDS)
        if words < minimum_words:
            return raw
        return "\n\n".join(head)

    def _reflow_story_paragraphs(self, text: str) -> str:
        """Deterministic, prose-preserving paragraph repair (no model call, no word changes).

        Qwen sometimes writes screenplay-style output with every spoken line on its own
        line, which yields 10 to 15 "paragraphs". A scene paragraph is about 80 words, so
        a short paragraph, or a short one that opens with a quotation mark, is a dialogue
        fragment that belongs to the scene paragraph before it.
        """
        value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if re.search(r"(?im)^\s*\[\[SCENE_\d{2}\]\]", value):
            # Numbered markers are scene boundaries, not prose fragments. Leave them
            # untouched so the strict parser below can validate order and coverage.
            return value
        paragraphs = [
            re.sub(r"[ \t\n]+", " ", part).strip()
            for part in re.split(r"\n\s*\n+", value)
            if part.strip()
        ]
        if len(paragraphs) < 2 or len(paragraphs) == DIRECTOR_STORY_SCENES:
            return text
        if len(paragraphs) > DIRECTOR_STORY_SCENES and self._trim_runaway_tail(text) != text:
            return text  # a finished story followed by surplus text: handled by the tail trim

        def words(part: str) -> int:
            return len(re.findall(r"\b[\w'’-]+\b", part))

        def opens_with_quote(part: str) -> bool:
            return part[:1] in "\"“‘'"

        merged: list[str] = []
        for part in paragraphs:
            fragment = words(part) < 28 or (opens_with_quote(part) and words(part) < 45)
            if fragment and merged:
                merged[-1] = merged[-1] + " " + part
            else:
                merged.append(part)
        if len(merged) > 1 and words(merged[0]) < 28:
            merged[1] = merged[0] + " " + merged[1]
            del merged[0]

        if len(merged) != len(paragraphs):
            print(
                f"[DIRECTOR] story reflow: {len(paragraphs)} -> {len(merged)} paragraphs "
                "(dialogue fragments merged into their scene; prose unchanged)",
                flush=True,
            )
        return "\n\n".join(merged)

    @staticmethod
    def _coerce_story_to_scene_paragraphs(text: str) -> str:
        """Remove explicit scene markers without inventing, merging, or splitting beats.

        New story drafts use ``[[SCENE_01]]`` ... ``[[SCENE_12]]`` at the start of
        each scene block. The marker is compiler scaffolding and is removed from the
        delivered story. Legacy/checkpoint prose without markers is accepted only when
        it already has exactly the configured number of paragraphs. A malformed marker
        sequence raises instead of silently manufacturing scene boundaries.
        """
        value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not value:
            return ""

        marker_pattern = re.compile(r"(?im)^[ \t]*\[\[SCENE_(\d{2})\]\][ \t]*(?:\n|[ \t]+|$)")
        markers = list(marker_pattern.finditer(value))
        if markers:
            expected = list(range(1, DIRECTOR_STORY_SCENES + 1))
            actual = [int(match.group(1)) for match in markers]
            if actual != expected:
                raise ValueError(
                    "Story scene markers must appear exactly once and in order as "
                    f"[[SCENE_01]] through [[SCENE_{DIRECTOR_STORY_SCENES:02d}]]; "
                    f"found {actual}. No retry or paragraph reshaping was attempted."
                )
            scenes: list[str] = []
            for index, marker in enumerate(markers):
                content_start = marker.end()
                content_end = markers[index + 1].start() if index + 1 < len(markers) else len(value)
                scene = re.sub(r"[ \t\n]+", " ", value[content_start:content_end]).strip()
                if not scene:
                    raise ValueError(
                        f"Story scene marker [[SCENE_{index + 1:02d}]] has no prose. "
                        "No retry or paragraph reshaping was attempted."
                    )
                scenes.append(scene)
            if len(scenes) != DIRECTOR_STORY_SCENES:
                raise ValueError(
                    f"Story must contain exactly {DIRECTOR_STORY_SCENES} marked scenes; "
                    f"found {len(scenes)}. No retry or paragraph reshaping was attempted."
                )
            return "\n\n".join(scenes)

        paragraphs = [
            re.sub(r"[ \t\n]+", " ", part).strip()
            for part in re.split(r"\n\s*\n+", value)
            if part.strip()
        ]
        # A legacy story can be normalized for whitespace only. Scene count is a hard
        # topology invariant; never split a prose paragraph at arbitrary sentence boundaries
        # and never merge independent beats just to reach the target count.
        if len(paragraphs) != DIRECTOR_STORY_SCENES:
            print(
                f"[DIRECTOR] story topology mismatch: got {len(paragraphs)} paragraphs, "
                f"expected {DIRECTOR_STORY_SCENES}; boundaries preserved (no split/merge)",
                flush=True,
            )
        return "\n\n".join(paragraphs)

    _STOCK_PHRASES = re.compile(
        r"\b(?:heart (?:pounded|pounding|thudded|thudding|hammered|hammering|raced|racing)|"
        r"(?:her|his|their) pulse (?:quickened|roared|pounded|raced)|breath hitch(?:ed|ing)|"
        r"(?:the )?weight of (?:what|his|her|the|their)|knuckles (?:white|whitened|turned white)|"
        r"jaw (?:tightened|clenched|set)|voice (?:steady|steadier|sharp over)|jagged shadows|"
        r"trembl(?:ed|ing) hands|(?:reeked|reeking) of|eerie glow|skeletal (?:hands|fingers|remains)|"
        r"(?:shadows|shapes) (?:clawing|danced|dancing|crawled)|mask of (?:fury|rage|calm)|"
        r"sent a shiver|chill ran down|ice in (?:his|her) veins|time bomb|"
        r"choices? that could never be undone|only the weight of|carried no answers)\b",
        flags=re.IGNORECASE,
    )

    @staticmethod
    def _apply_scene_metadata(scene: dict, metadata: dict) -> None:
        """Apply Qwen-authored scene semantics while keeping structural IDs deterministic."""
        if not isinstance(scene, dict) or not isinstance(metadata, dict):
            return
        string_fields = (
            "location", "time_of_day", "weather", "atmosphere", "mood",
            "lighting", "color_temperature", "dramatic_purpose",
        )
        unset_values = {"", "n/a", "unknown", "not specified", "unspecified"}
        for field in string_fields:
            raw = metadata.get(field)
            if not isinstance(raw, str):
                continue
            value = re.sub(r"\s+", " ", raw).strip()
            if field in {"time_of_day", "weather"} and value.casefold() in unset_values:
                value = "unspecified"
            elif value.casefold() in unset_values:
                value = ""
            if value:
                scene[field] = value
        for field, limit in (("environment_details", 4), ("key_props", 6)):
            raw_values = metadata.get(field)
            if isinstance(raw_values, list):
                scene[field] = list(dict.fromkeys(
                    re.sub(r"\s+", " ", str(value or "")).strip()
                    for value in raw_values
                    if re.sub(r"\s+", " ", str(value or "")).strip()
                ))[:limit]

    @staticmethod
    def _enforce_scene_coherence(
        scenes: list[dict],
        shots: list[dict],
        characters: list[dict],
    ) -> None:
        """Enforce within-scene consistency without guessing semantics from vocabulary.

        Qwen-authored scene metadata wins when supplied. If legacy/fallback data has
        no scene metadata, the first usable Qwen shot value becomes the scene value.
        IDs, character canonicalization, and equality across shots are deterministic;
        no list of English place/weather/lighting words decides what a scene means.
        """
        canonical_by_norm = {
            str(c.get("name", "")).strip().casefold(): str(c.get("name", "")).strip()
            for c in (characters or [])
            if isinstance(c, dict) and str(c.get("name", "")).strip()
        }
        scene_by_id = {
            str(sc.get("scene_id", "") or "").strip(): sc
            for sc in (scenes or []) if isinstance(sc, dict)
        }
        by_scene: dict[str, list[dict]] = {}
        for shot in shots or []:
            if isinstance(shot, dict):
                by_scene.setdefault(str(shot.get("scene_id", "") or "").strip(), []).append(shot)

        scene_string_fields = (
            "location", "time_of_day", "weather", "atmosphere", "mood",
            "lighting", "color_temperature",
        )
        shot_shared_fields = {
            "location", "time_of_day", "weather", "mood", "lighting",
            "color_temperature",
        }
        for scene_id, group in by_scene.items():
            scene = scene_by_id.get(scene_id)
            if scene is None:
                continue
            # Scene-level Qwen direction owns shared fields. For older checkpoints or
            # deterministic fallback shots, inherit the first populated shot value.
            for field in scene_string_fields:
                value = str(scene.get(field, "") or "").strip()
                if field in {"time_of_day", "weather"} and value.casefold() in {"", "unknown", "not specified", "n/a"}:
                    value = "unspecified"
                if not value and field in shot_shared_fields:
                    value = next((str(shot.get(field, "") or "").strip() for shot in group
                                  if str(shot.get(field, "") or "").strip()), "")
                if field in {"time_of_day", "weather"} and not value:
                    value = "unspecified"
                if value:
                    scene[field] = value
                    if field in shot_shared_fields:
                        for shot in group:
                            shot[field] = value

            for shot in group:
                for state_key in ("continuity_start_state", "continuity_end_state"):
                    state = shot.get(state_key)
                    if not isinstance(state, dict):
                        continue
                    for field in ("location", "lighting", "environment"):
                        scene_field = "environment_details" if field == "environment" else field
                        value = scene.get(scene_field, "")
                        if field == "environment" and isinstance(value, list):
                            value = ", ".join(str(item) for item in value if str(item).strip())
                        if value:
                            state[field] = value
                names = shot.get("characters")
                if isinstance(names, list):
                    shot["characters"] = list(dict.fromkeys(
                        canonical_by_norm.get(str(name).strip().casefold(), str(name).strip())
                        for name in names if str(name).strip()
                    ))

    @staticmethod
    def _refresh_dialogue_summary(shot: dict) -> None:
        events = [
            event
            for event in (shot.get("dialogue_events", []) or [])
            if isinstance(event, dict)
        ]
        shot["dialogue_events"] = events
        shot["speaking_characters"] = list(dict.fromkeys(
            str(event.get("speaker", "") or "").strip()
            for event in events
            if str(event.get("speaker", "") or "").strip()
        ))
        shot["speech_text"] = " ".join(
            str(event.get("text", "") or "").strip()
            for event in events
            if str(event.get("text", "") or "").strip()
        )

    @staticmethod
    def _dialogue_scene_fits_h3(
        scene_shots: list[dict],
        characters: list[dict],
    ) -> tuple[bool, str]:
        """Check dialogue feasibility using the exact downstream scheduler."""
        trial_plan = {"shots": deepcopy(scene_shots)}
        try:
            with redirect_stdout(io.StringIO()):
                DialogueTimeline(
                    [dict(character) for character in characters if isinstance(character, dict)]
                ).apply_to_plan(trial_plan)
        except ValueError as exc:
            detail = str(exc)
            # Treat only scheduler errors that specifically mean the dialogue
            # cannot fit the H3 timing contract as a failed feasibility trial.
            # Other ValueErrors (empty dialogue, invalid speaker binding,
            # continuation mismatch, overlap, etc.) are real source/contract
            # defects and must propagate instead of being misclassified as a
            # candidate partition that simply does not fit.
            feasibility_markers = (
                "maximum H3 runtime",
                "maximum schedulable H3 runtime",
                "maximum legal duration",
                "H3-effective shot boundary",
                "dialogue does not fit shot duration",
                "dialogue timing exceeds",
                "dialogue exceeds H3-effective shot duration",
            )
            if any(marker in detail for marker in feasibility_markers):
                return False, detail
            raise
        return True, ""

    def _normalize_dialogue_h3_feasibility(
        self,
        scenes: list[dict],
        shots: list[dict],
        characters: list[dict],
    ) -> None:
        """Make generated dialogue satisfy the exact H3 timing contract.

        Dialogue is a semantic contract, while H3 duration is a hard production
        contract. This pass therefore searches only over existing dialogue events
        and existing shot boundaries. It never invents, truncates, paraphrases, or
        reorders spoken content, and it never creates or removes creative shots.

        The search is scene-global rather than greedy pairwise: every candidate
        repartition is validated against the exact downstream ``DialogueTimeline``
        for the complete scene. Candidates are ordered by boundary movement so the
        first successful solution is the smallest deterministic change to Qwen's
        original editorial partition. Only when no intact-event repartition works
        is an explicitly continuing final event eligible for a whitespace split.
        """
        if not shots:
            return

        shots_by_scene: dict[str, list[dict]] = {}
        for shot in shots:
            if not isinstance(shot, dict):
                continue
            scene_id = str(shot.get("scene_id", "") or "").strip()
            if scene_id:
                shots_by_scene.setdefault(scene_id, []).append(shot)

        scene_lookup = {
            str(scene.get("scene_id", "") or "").strip(): scene
            for scene in scenes
            if isinstance(scene, dict) and str(scene.get("scene_id", "") or "").strip()
        }

        for scene_id, scene_shots in shots_by_scene.items():
            if len(scene_shots) < 2:
                continue

            for item in scene_shots:
                self._refresh_dialogue_summary(item)

            fits, detail = self._dialogue_scene_fits_h3(scene_shots, characters)
            if fits:
                continue

            original_events: list[dict] = []
            original_boundaries: list[int] = []
            running = 0
            for shot in scene_shots[:-1]:
                events = [
                    dict(event)
                    for event in (shot.get("dialogue_events", []) or [])
                    if isinstance(event, dict)
                ]
                original_events.extend(events)
                running += len(events)
                original_boundaries.append(running)
            original_events.extend(
                dict(event)
                for event in (scene_shots[-1].get("dialogue_events", []) or [])
                if isinstance(event, dict)
            )

            if not original_events:
                # The scheduler rejected the scene despite there being no dialogue
                # events. That is not a repartition problem and must propagate.
                raise RuntimeError(
                    "Director dialogue feasibility failed for a dialogue-free scene: "
                    f"scene={scene_id} detail={detail or 'unknown scheduler failure'}"
                )

            shot_count = len(scene_shots)
            event_count = len(original_events)

            # Keep the exact scene-global partition search bounded. The current
            # production contract uses two shots per scene; this guard prevents a
            # future shot-count/configuration change from turning the recursive
            # enumeration into an unbounded combinatorial search.
            if shot_count > 4 or event_count > 12:
                raise RuntimeError(
                    "Dialogue H3 feasibility search exceeds the bounded partition "
                    f"contract: scene={scene_id} shots={shot_count} events={event_count}."
                )

            # Enumerate all monotonic cuts through the original ordered event list.
            # For the project's normal 2-shot scenes this is simply every possible
            # cut. More generally this is an exact scene-global search over all
            # event-to-shot partitions, with no duplicated timing model.
            partition_candidates: list[tuple[int, tuple[int, ...]]] = []

            def _enumerate_cuts(
                boundary_index: int,
                previous_cut: int,
                cuts: list[int],
            ) -> None:
                if boundary_index == shot_count - 1:
                    cost = sum(
                        abs(cut - original)
                        for cut, original in zip(cuts, original_boundaries)
                    )
                    partition_candidates.append((cost, tuple(cuts)))
                    return
                for cut in range(previous_cut, event_count + 1):
                    cuts.append(cut)
                    _enumerate_cuts(boundary_index + 1, cut, cuts)
                    cuts.pop()

            _enumerate_cuts(0, 0, [])
            partition_candidates.sort(key=lambda item: (item[0], item[1]))

            repaired = False
            failure_detail = detail
            original_counts = [
                len(shot.get("dialogue_events", []) or [])
                for shot in scene_shots
            ]

            for _, cuts in partition_candidates:
                candidate_shots = deepcopy(scene_shots)
                starts = [0, *cuts]
                ends = [*cuts, event_count]

                for index, (start, end) in enumerate(zip(starts, ends)):
                    candidate_shots[index]["dialogue_events"] = deepcopy(
                        original_events[start:end]
                    )

                for item in candidate_shots:
                    self._refresh_dialogue_summary(item)

                temp_scene = deepcopy(
                    scene_lookup.get(scene_id, {"scene_id": scene_id})
                )
                self._normalize_dialogue_continuations(
                    [temp_scene],
                    candidate_shots,
                    characters,
                )
                for item in candidate_shots:
                    self._refresh_dialogue_summary(item)

                fits, candidate_detail = self._dialogue_scene_fits_h3(
                    candidate_shots,
                    characters,
                )
                if not fits:
                    failure_detail = candidate_detail or failure_detail
                    continue

                for index, shot in enumerate(scene_shots):
                    shot["dialogue_events"] = deepcopy(
                        candidate_shots[index]["dialogue_events"]
                    )
                    self._refresh_dialogue_summary(shot)

                self._normalize_dialogue_continuations(
                    [scene_lookup.get(scene_id, {"scene_id": scene_id})],
                    scene_shots,
                    characters,
                )
                for item in scene_shots:
                    self._refresh_dialogue_summary(item)

                final_counts = [
                    len(shot.get("dialogue_events", []) or [])
                    for shot in scene_shots
                ]
                self._record_recovery(
                    "dialogue_h3_repartition",
                    f"scene={scene_id} original_counts={original_counts} "
                    f"final_counts={final_counts} cuts={cuts}",
                )
                repaired = True
                break

            if repaired:
                # Revalidate the final mutated scene using the exact downstream
                # scheduler. This is an explicit invariant, not merely a property
                # inferred from the candidate that happened to succeed.
                final_fits, final_detail = self._dialogue_scene_fits_h3(
                    scene_shots,
                    characters,
                )
                if not final_fits:
                    raise RuntimeError(
                        "Director dialogue repair produced a scene that no longer "
                        "satisfies the H3 timing contract: "
                        f"scene={scene_id} detail={final_detail or failure_detail}"
                    )
                continue

            # Last resort: split only an explicitly continuing event. Search all
            # eligible continuation events in scene order and all whitespace cuts,
            # while still requiring the COMPLETE scene to pass the real scheduler.
            flattened_positions: list[tuple[int, int, dict]] = []
            for shot_index, shot in enumerate(scene_shots):
                for event_index, event in enumerate(
                    shot.get("dialogue_events", []) or []
                ):
                    if isinstance(event, dict):
                        flattened_positions.append((shot_index, event_index, event))

            for source_shot_index, source_event_index, source_event in flattened_positions:
                if not bool(source_event.get("continues_to_next_shot", False)):
                    continue
                source_text = str(source_event.get("text", "") or "")
                if not source_text.strip():
                    continue
                if any(
                    source_event.get(key) not in (None, "")
                    for key in (
                        "expected_duration_seconds",
                        "duration_seconds",
                        "expected_duration_ms",
                    )
                ):
                    continue

                split_positions = [
                    index
                    for index, char in enumerate(source_text)
                    if char == " " and 0 < index < len(source_text) - 1
                ]

                for split_index in reversed(split_positions):
                    first_text = source_text[:split_index]
                    second_text = source_text[split_index + 1:]
                    if not first_text.strip() or not second_text.strip():
                        continue

                    first_part = dict(source_event)
                    second_part = dict(source_event)
                    first_part["text"] = first_text
                    second_part["text"] = second_text
                    second_part["continues_from_previous_shot"] = True
                    second_part["continues_to_next_shot"] = False

                    candidate_shots = deepcopy(scene_shots)
                    source_events = candidate_shots[source_shot_index].get(
                        "dialogue_events", []
                    ) or []
                    candidate_shots[source_shot_index]["dialogue_events"] = (
                        deepcopy(source_events[:source_event_index])
                        + [first_part]
                        + deepcopy(source_events[source_event_index + 1:])
                    )

                    # The split must cross an existing adjacent shot boundary. A
                    # continuation event is therefore only eligible when its source
                    # shot has a real next shot in the same scene.
                    if source_shot_index + 1 >= shot_count:
                        continue
                    target_events = candidate_shots[source_shot_index + 1].get(
                        "dialogue_events", []
                    ) or []
                    candidate_shots[source_shot_index + 1]["dialogue_events"] = (
                        [second_part] + deepcopy(target_events)
                    )

                    for item in candidate_shots:
                        self._refresh_dialogue_summary(item)
                    temp_scene = deepcopy(
                        scene_lookup.get(scene_id, {"scene_id": scene_id})
                    )
                    self._normalize_dialogue_continuations(
                        [temp_scene],
                        candidate_shots,
                        characters,
                    )
                    for item in candidate_shots:
                        self._refresh_dialogue_summary(item)

                    fits, candidate_detail = self._dialogue_scene_fits_h3(
                        candidate_shots,
                        characters,
                    )
                    if not fits:
                        failure_detail = candidate_detail or failure_detail
                        continue

                    for index, shot in enumerate(scene_shots):
                        shot["dialogue_events"] = deepcopy(
                            candidate_shots[index]["dialogue_events"]
                        )
                        self._refresh_dialogue_summary(shot)
                    self._normalize_dialogue_continuations(
                        [scene_lookup.get(scene_id, {"scene_id": scene_id})],
                        scene_shots,
                        characters,
                    )
                    for item in scene_shots:
                        self._refresh_dialogue_summary(item)

                    final_fits, final_detail = self._dialogue_scene_fits_h3(
                        scene_shots,
                        characters,
                    )
                    if not final_fits:
                        raise RuntimeError(
                            "Director dialogue continuation split produced a scene "
                            "that no longer satisfies the H3 timing contract: "
                            f"scene={scene_id} detail={final_detail or failure_detail}"
                        )

                    self._record_recovery(
                        "dialogue_h3_continuation_split",
                        f"scene={scene_id} shot={scene_shots[source_shot_index].get('shot_id','')} "
                        f"source_chars={len(source_text)} split_at={split_index}",
                    )
                    repaired = True
                    break

                if repaired:
                    break

            if not repaired:
                raise RuntimeError(
                    "Director dialogue cannot satisfy the H3 timing contract without "
                    "dropping, truncating, or reordering spoken content: "
                    f"scene={scene_id} detail={failure_detail or 'no legal scene-global repartition found'}"
                )

    def _normalize_dialogue_continuations(
        self,
        scenes: list[dict],
        shots: list[dict],
        characters: list[dict] | None = None,
    ) -> None:
        """Canonicalize dialogue continuation flags against canonical speakers.

        Speaker/entity normalization can remove or canonicalize dialogue events.
        This pass therefore runs both before and after semantic speaker filtering.
        A continuation edge is valid only when the final speaker of the previous
        shot and the first speaker of the current shot resolve to the same
        canonical character. A speaker change always starts a new dialogue turn.
        """
        alias_map = EntityResolver.build_character_alias_map(characters or [])

        def _speaker_key(event: dict) -> str:
            raw = str(
                event.get("speaker", event.get("speaker_name", "")) or ""
            ).strip()
            normalized = EntityResolver.normalize(raw)
            return alias_map.get(normalized, normalized)
        shots_by_scene_order: dict[str, list[dict]] = {}
        for shot in shots:
            if not isinstance(shot, dict):
                continue
            scene_id = str(shot.get("scene_id", "") or "").strip()
            if scene_id:
                shots_by_scene_order.setdefault(scene_id, []).append(shot)

        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id", "") or "").strip()
            scene_shots = shots_by_scene_order.get(scene_id, [])
            previous_events: list[dict] | None = None

            for position, shot in enumerate(scene_shots):
                events = shot.get("dialogue_events", [])
                if not isinstance(events, list):
                    events = []
                    shot["dialogue_events"] = events

                events[:] = [event for event in events if isinstance(event, dict)]

                if not events:
                    if previous_events:
                        previous_events[-1]["continues_to_next_shot"] = False
                    previous_events = None
                    continue

                for event in events:
                    event["continues_from_previous_shot"] = bool(
                        event.get("continues_from_previous_shot", False)
                    )
                    event["continues_to_next_shot"] = bool(
                        event.get("continues_to_next_shot", False)
                    )

                if position == 0 or previous_events is None:
                    events[0]["continues_from_previous_shot"] = False
                else:
                    previous_flag = bool(
                        previous_events[-1].get("continues_to_next_shot", False)
                    )
                    current_flag = bool(
                        events[0].get("continues_from_previous_shot", False)
                    )
                    continuation_requested = previous_flag or current_flag
                    same_speaker = (
                        bool(_speaker_key(previous_events[-1]))
                        and bool(_speaker_key(events[0]))
                        and _speaker_key(previous_events[-1]) == _speaker_key(events[0])
                    )
                    continuation = continuation_requested and same_speaker
                    if continuation_requested and not same_speaker:
                        self._record_recovery(
                            "dialogue_continuation_speaker_boundary_reset",
                            (
                                f"previous={_speaker_key(previous_events[-1])!r} "
                                f"current={_speaker_key(events[0])!r}"
                            ),
                        )
                    previous_events[-1]["continues_to_next_shot"] = continuation
                    events[0]["continues_from_previous_shot"] = continuation

                for event in events[1:]:
                    event["continues_from_previous_shot"] = False

                # Only the final surviving event in a shot may carry the
                # continuation-to-next-shot flag. Clear any stale flags on
                # earlier events before storing the shot as the boundary state.
                for event in events[:-1]:
                    event["continues_to_next_shot"] = False

                previous_events = events

            # A scene boundary is a hard semantic boundary for dialogue. The
            # final event of the final shot in this scene can never continue
            # into another scene, even if Qwen emitted a stale continuation
            # flag or an earlier reconciliation left one behind.
            if previous_events:
                for event in previous_events:
                    event["continues_to_next_shot"] = False

    @staticmethod
    def _normalize_dialogue_text(value: str, keep_case: bool = False) -> str:
        cleaned = re.sub(r"\s+", " ", str(value or "").strip().strip('"“”‘’'))
        if keep_case:
            # Display/source text must preserve the user's exact apostrophe, dash and
            # punctuation choices; only the comparison key is normalized elsewhere.
            return cleaned
        return cleaned.replace("’", "'").replace("‘", "'").replace("—", "-").replace("–", "-").lower()

    @staticmethod
    def _dialogue_anchor_key(value: str) -> str:
        """Canonical comparison key; source text is preserved separately."""
        text = str(value or "").replace("’", "'").replace("‘", "'").replace("…", "...").casefold()
        return " ".join(re.findall(r"[\w]+(?:'[\w]+)*", text, flags=re.UNICODE))

    @staticmethod
    def _story_has_explicit_dialogue_surface(story: str) -> bool:
        """Return whether prose has evidence of an actually attributed speech line.

        Quoted signs, labels, notes, and document text are not a spoken-dialogue surface.
        This remains an advisory predicate, not a dialogue extractor.
        """
        text = str(story or "")
        if not text.strip():
            return False
        quote_pattern = re.compile(
            r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)',
            flags=re.UNICODE,
        )
        speech_attribution = re.compile(
            r"\b(?:said|asks?|asked|repl(?:y|ied)|whisper(?:s|ed)|shout(?:s|ed)|"
            r"mutter(?:s|ed)|answer(?:s|ed)|warn(?:s|ed)|tell(?:s|t)|"
            r"cr(?:y|ies|ied)|yell(?:s|ed)|bark(?:s|ed))\b",
            flags=re.IGNORECASE,
        )
        for match in quote_pattern.finditer(text):
            if not any(str(value or "").strip() for value in match.groups()):
                continue
            before = text[max(0, match.start() - 110):match.start()]
            after = text[match.end():min(len(text), match.end() + 110)]
            local = before + " " + after
            if speech_attribution.search(local):
                return True
        label_pattern = re.compile(
            r"(?m)^\s*([A-Z][A-Za-z0-9.'’\-]*(?:\s+[A-Z][A-Za-z0-9.'’\-]*){0,4})\s*(?::|—|–)\s*([^\n]+?)\s*$"
        )
        return any(str(match.group(2) or "").strip() for match in label_pattern.finditer(text))

    @classmethod
    def _extract_story_spoken_segments(
        cls,
        story: str,
        semantic_spoken_dialogue: list[dict] | None = None,
        allowed_speakers: set[str] | list[str] | None = None,
        speaker_aliases: dict[str, str] | None = None,
    ) -> list[dict]:
        """Extract direct spoken source segments from one semantic dialogue contract.

        Creative modes pass the `spoken_dialogue` field returned by the existing single Qwen
        semantic extraction call. Deterministic parsing only reconciles those semantic spans to
        exact source occurrences; it does not maintain a speech-verb, machine-word, or domain-word list.
        When no semantic result is available (for example, a legacy caller or checkpoint), only
        explicit screenplay-style `Name: line` labels are accepted, which is deliberately conservative.
        """
        text = str(story or "")
        segments: list[dict] = []

        alias_map = {
            EntityResolver.normalize(str(key)): str(value).strip()
            for key, value in (speaker_aliases or {}).items()
            if str(key or "").strip() and str(value or "").strip()
        }
        allowed_surface_map = {
            EntityResolver.normalize(str(value)): str(value).strip()
            for value in (allowed_speakers or [])
            if str(value or "").strip()
        }
        allowed_titleless_map = {
            EntityResolver.normalize(EntityResolver.strip_honorific(normalized)): display
            for normalized, display in allowed_surface_map.items()
        }

        def canonicalize_speaker(surface: str) -> str:
            normalized = EntityResolver.normalize(surface)
            # Prefer exact approved roster spelling; only strip an honorific when
            # that title-free surface is itself in the approved roster or alias map.
            # This avoids dropping a valid "Dr. Elias" speaker by comparing "Elias"
            # against an allowed-speaker set that contains only "Dr. Elias".
            if normalized in allowed_surface_map:
                return EntityResolver.normalize(allowed_surface_map[normalized])
            titleless = EntityResolver.normalize(EntityResolver.strip_honorific(normalized))
            if titleless in allowed_titleless_map:
                return EntityResolver.normalize(allowed_titleless_map[titleless])
            resolved = alias_map.get(normalized) or alias_map.get(titleless)
            # Preserve the source spelling for an unrostered speaker so a valid
            # explicitly-remote audio source can remain named without becoming a
            # visual character. Such surfaces are admitted below only for remote turns.
            return EntityResolver.normalize(resolved) if resolved else str(surface or "").strip()

        # A device noun by itself is not evidence that the quoted character is
        # off-screen: a radio or phone may simply be a prop in the room. Require
        # an audio-route phrase or an explicit audio action on the device.
        remote_device = (
            r"radio|walkie-talkie|intercom|comms|communications? relay|"
            r"phone|telephone|transmission|communicator|speaker|"
            r"terminal|loudspeaker|console"
        )
        remote_context_re = re.compile(
            rf"\b(?:{remote_device})\b[^.!?]{{0,80}}"
            r"\b(?:crackled|hissed|buzzed|played|transmitted|relayed|repeated|"
            r"announced|said|spoke|came through|was heard|sounded)\b|"
            rf"\b(?:voice|words|message|signal|someone|he|she|they)\b[^.!?]{{0,80}}"
            r"\b(?:came|crackled|hissed|buzzed|played|transmitted|relayed|repeated|"
            r"announced|said|spoke|was heard|sounded)\b[^.!?]{{0,80}}"
            rf"\b(?:through|over|from|on)\s+(?:the|a|an)?\s*(?:{remote_device})\b|"
            rf"\b(?:through|over|from|on)\s+(?:the|a|an)?\s*(?:{remote_device})\b|"
            r"\b(?:in|through|amid|from)\s+(?:the\s+)?static\b|"
            r"\bover the line\b",
            flags=re.IGNORECASE,
        )
        remote_speaker_phrase_re = re.compile(
            r"\b(?:through|from|over)\s+(?:(?:a|the)\s+)?(?:long-dead\s+)?speaker\b",
            flags=re.IGNORECASE,
        )
        anonymous_voice_re = re.compile(
            r"\b(?:a|an|the)\s+(?:unseen|unidentified|disembodied|unknown\s+)?voice\s+"
            r"(?:boomed|echoed|rang\s+out|called\s+out|announced|spoke|said)\b",
            flags=re.IGNORECASE,
        )
        remembered_context_re = re.compile(
            r"\b(?:memory surfaced|a memory (?:of|came|returned)|remembered|recalled|"
            r"replaying|replayed|mind replayed|mind replaying|words echoed\s*:|"
            r"(?:last|final|previous|earlier|remembered|recorded)\s+(?:words|messages?|voice|line|advice|warning)|"
            r"(?:voice|words|advice|warning) echoed in (?:his|her|their) (?:mind|head|skull))",
            flags=re.IGNORECASE,
        )
        written_before_re = re.compile(
            r"\b(?:screen|monitor|terminal)\s+(?:showed|displayed|flashed|read|was frozen on)\b[^.!?]{0,80}$|"
            r"\b(?:written|scrawled|inscribed|printed|displayed)\s+(?:in|on|beneath)\b[^.!?]{0,60}$|"
            r"\b(?:journal|diary|note|letter|dossier|file|document|log|margin|entry|handwriting|label|sign|recording|message)"
            r"\s+(?:read|reads|wrote|had written|scrawled|stated|said|showed|displayed|revealed|listed|contained|recorded|filled)"
            r"\b[^.!?]{0,90}$|"
            r"\b(?:journal|diary|note|letter|dossier|file|document|log|margin|entry|handwriting|label|sign|recording|message)"
            r"\s*[:—-]\s*$",
            flags=re.IGNORECASE,
        )
        named_reader = r"[A-Z][A-Za-z0-9.'’\-]*(?:\s+[A-Z][A-Za-z0-9.'’\-]*){0,3}"
        written_after_re = re.compile(
            r"^\s*[,;:—-]?\s*(?:(?:she|he|they)\s+(?:read|reads|had read|wrote|had written)\b|"
            + named_reader + r"\s+(?:read|reads|had read|wrote|had written)\b|"
            r"(?:a|the|his|her|their)\s+(?:note|journal|diary|letter|dossier|file|document|log|label|screen|monitor)"
            r"\b.{0,70}\b(?:read|reads|said|stated|showed|displayed|revealed)\b)",
            flags=re.IGNORECASE,
        )
        written_reader_before_re = re.compile(
            r"(?:\b(?:she|he|they)\s+(?:read|reads|had read|wrote|had written)|"
            + named_reader + r"\s+(?:read|reads|had read|wrote|had written))\s*[,;:]?\s*$",
            flags=re.IGNORECASE,
        )
        physical_presence_re = re.compile(
            r"\b(?:appeared|arrived|entered|stood|stepped|walked|emerged|joined|"
            r"came in|came closer|was standing|standing beside|standing at|stood beside|stood at|in person|"
            r"gripped|grabbed|took her arm|took his arm)\b|"
            r"\bfound\s+(?:a|an|the)\s+(?:man|woman|person|figure|scientist|stranger|doctor)\b",
            flags=re.IGNORECASE,
        )
        # Carry an explicitly established off-screen audio source across dialogue
        # turns in the same paragraph, but stop carrying it as soon as narration
        # places a character physically in the scene. Arbitrary earlier device nouns
        # do not establish this state.
        remote_channel_end_by_paragraph: dict[int, int] = {}

        def quote_delivery(position: int, end_position: int, semantic_delivery: str = "") -> str:
            """Classify a quote as visible in-person audio, live remote audio, or excluded.

            Remote speech stays available to the audio/compiler path while its source is
            not automatically made a visible character. Remembered and written quotations
            take precedence over a nearby device noun or remote-channel carry state.
            """
            paragraph_break = text.rfind("\n\n", 0, position)
            paragraph_start = paragraph_break + 2 if paragraph_break >= 0 else 0
            paragraph_end = text.find("\n\n", end_position)
            if paragraph_end < 0:
                paragraph_end = len(text)
            before = text[paragraph_start:position]
            after = text[end_position:paragraph_end]

            prior_boundary = max(before.rfind("."), before.rfind("?"), before.rfind("!"))
            local_before = before[prior_boundary + 1:][-260:]
            next_boundaries = [index for index in (after.find("."), after.find("?"), after.find("!")) if index >= 0]
            local_after = after[:min(next_boundaries) + 1] if next_boundaries else after[:180]

            # These exclusions win over remote context; a device nearby does not turn
            # a journal, memory, archive, or anonymous voice into present-time dialogue.
            if remembered_context_re.search(local_before) or remembered_context_re.search(local_after):
                return "excluded"
            if (written_before_re.search(local_before) or written_reader_before_re.search(local_before)
                    or written_after_re.search(local_after)):
                return "excluded"
            if anonymous_voice_re.search(local_before):
                return "excluded"

            active_remote_end = remote_channel_end_by_paragraph.get(paragraph_start)
            if active_remote_end is not None and active_remote_end <= position:
                intervening_text = text[active_remote_end:position]
                if physical_presence_re.search(intervening_text):
                    remote_channel_end_by_paragraph.pop(paragraph_start, None)
                else:
                    return "remote"

            remote_positions = [match.start() for match in remote_context_re.finditer(local_before)]
            physical_positions = [match.start() for match in physical_presence_re.finditer(local_before)]
            local_speaker_positions = [match.start() for match in remote_speaker_phrase_re.finditer(local_before)]
            local_physical_positions = [match.start() for match in physical_presence_re.finditer(local_before)]
            remote_evidence = bool(
                (local_speaker_positions and (
                    not local_physical_positions or max(local_speaker_positions) > max(local_physical_positions)
                ))
                or (remote_positions and (
                    not physical_positions or max(remote_positions) > max(physical_positions)
                ))
            )

            # With no local lead-in, check only the immediately preceding sentence.
            if not remote_evidence and not local_before.strip() and before.strip():
                prior_sentence_match = re.search(r"([^.!?]+[.!?])\s*$", before.strip())
                if prior_sentence_match:
                    prior_sentence = prior_sentence_match.group(1)
                    prior_remote = list(remote_context_re.finditer(prior_sentence))
                    prior_speaker = list(remote_speaker_phrase_re.finditer(prior_sentence))
                    prior_physical = list(physical_presence_re.finditer(prior_sentence))
                    remote_evidence = bool((prior_remote or prior_speaker) and (
                        not prior_physical
                        or max((m.start() for m in prior_remote + prior_speaker), default=-1) > prior_physical[-1].start()
                    ))

            if remote_evidence or str(semantic_delivery or "").strip().lower() == "remote":
                remote_channel_end_by_paragraph[paragraph_start] = end_position
                return "remote"
            return "in_person"

        remote_excluded = 0
        semantic_items: list[dict] = []
        semantic_by_quote_id: dict[int, dict] = {}
        for item in (semantic_spoken_dialogue or []):
            if not isinstance(item, dict):
                continue
            spoken = str(item.get("text", "") or "").strip()
            speaker = str(item.get("speaker", "") or "").strip()
            key = cls._dialogue_anchor_key(spoken)
            if not speaker:
                continue
            quote_id = item.get("quote_id")
            canonical_speaker = canonicalize_speaker(speaker)
            if isinstance(quote_id, int) and quote_id > 0:
                semantic_by_quote_id[quote_id] = {"key": key, "speaker": canonical_speaker, "delivery": "remote" if str(item.get("delivery", "in_person")).strip().lower() == "remote" else "in_person"}
            # ID-addressed items are consumed only by their exact source occurrence.
            # Text-only legacy items use the fuzzy compatibility path below.
            if key and not (isinstance(quote_id, int) and quote_id > 0):
                semantic_items.append({"key": key, "speaker": canonical_speaker, "quote_id": None, "delivery": "remote" if str(item.get("delivery", "in_person")).strip().lower() == "remote" else "in_person"})

        allowed = set(allowed_surface_map) | {
            EntityResolver.normalize(EntityResolver.strip_honorific(value))
            for value in allowed_surface_map.values()
            if str(value or "").strip()
        }
        remote_only_speaker_keys = set()
        for item in (semantic_spoken_dialogue or []):
            if not isinstance(item, dict) or str(item.get("delivery", "in_person")).strip().lower() != "remote":
                continue
            surface = str(item.get("speaker", "") or "").strip()
            if not surface:
                continue
            resolved_surface = canonicalize_speaker(surface)
            resolved_key = EntityResolver.normalize(resolved_surface)
            raw_key = EntityResolver.normalize(surface)
            titleless_key = EntityResolver.normalize(EntityResolver.strip_honorific(surface))
            if resolved_key not in allowed and raw_key not in allowed and titleless_key not in allowed:
                remote_only_speaker_keys.add(resolved_key or raw_key)

        # Deterministic fallback only for an explicit local attribution that names an
        # already-approved canonical person. It does not classify arbitrary quotations.
        speech_cue_re = re.compile(
            r"\b(?:said|says|asked|asks|replied|replies|whispered|whispers|shouted|shouts|"
            r"muttered|mutters|answered|answers|warned|warns|told|tells|cried|cries|"
            r"yelled|yells|barked|barks|called|calls|snapped|snaps|added|adds|"
            r"insisted|insists|admitted|admits|demanded|demands|repeated|repeats)\b",
            re.IGNORECASE,
        )
        speaker_surfaces: list[tuple[str, str]] = []
        for canonical in dict.fromkeys(allowed_surface_map.values()):
            for surface in cls._name_surfaces(canonical):
                speaker_surfaces.append((surface, canonical))
        for alias, canonical in alias_map.items():
            resolved = str(canonical or "").strip()
            if resolved:
                speaker_surfaces.append((str(alias), resolved))

        def explicit_local_speaker(position: int, end_position: int) -> str | None:
            before = text[max(0, position - 130):position]
            after = text[end_position:min(len(text), end_position + 130)]
            before_clause = re.split(r"(?<=[.!?])[\"\u201d\u2019']?\s+", before)[-1]
            after_clause = re.split(r"[.!?](?:[\"\u201d\u2019']?\s+|$)", after, maxsplit=1)[0]
            matches = set()
            for surface, canonical in speaker_surfaces:
                if not surface or len(surface) < 2:
                    continue
                surface_re = r"(?<!\w)" + re.escape(surface) + r"(?!\w)"
                if re.search(surface_re, before_clause, re.IGNORECASE) and speech_cue_re.search(before_clause):
                    matches.add(canonical)
                if re.search(surface_re, after_clause, re.IGNORECASE) and speech_cue_re.search(after_clause):
                    matches.add(canonical)
            return next(iter(matches)) if len(matches) == 1 else None

        semantic_consumed: set[int] = set()

        def consume_semantic_speaker(key: str) -> tuple[set[str], str]:
            for index, item in enumerate(semantic_items):
                if index in semantic_consumed or item["key"] != key:
                    continue
                semantic_consumed.add(index)
                return {item["speaker"]}, str(item.get("delivery", "in_person"))
            return set(), "in_person"

        quote_pattern = re.compile(
            r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)',
            flags=re.UNICODE,
        )
        quote_spans: list[tuple[int, int]] = []
        quote_matches = list(quote_pattern.finditer(text))
        previous_quote_match = None
        previous_quote_speakers: set[str] = set()
        previous_quote_delivery = "in_person"
        pronoun_attribution_only_re = re.compile(
            r"\s*[\"\u201d\u2019\u00bb]?\s*(?:and\s+)?(?:he|she|they)\s+"
            r"(?:said|says|asked|asks|replied|replies|whispered|whispers|shouted|shouts|screamed|screams|"
            r"muttered|mutters|answered|answers|warned|warns|cried|cries|yelled|yells|"
            r"barked|barks|called|calls|snapped|snaps|insisted|insists|demanded|demands)"
            r"\s*[,;.!?…\u2014\u2013-]*\s*[\"\u201c\u2018\u00ab]?\s*\Z",
            re.IGNORECASE,
        )

        for quote_id, match in enumerate(quote_matches, start=1):
            value = next((part for part in match.groups() if part is not None), "")
            key = cls._dialogue_anchor_key(value)
            if not key:
                continue

            semantic_quote = semantic_by_quote_id.get(quote_id)
            if semantic_quote and semantic_quote.get("speaker"):
                speakers = {semantic_quote["speaker"]}
                semantic_delivery = str(semantic_quote.get("delivery", "in_person"))
            else:
                speakers, semantic_delivery = consume_semantic_speaker(key)

            # A short pronoun attribution belongs to the preceding quotation. If the next
            # quotation starts immediately afterward and no new named attribution appears,
            # preserve that speaker across the continuing turn instead of trusting a
            # contradictory per-quote extraction label. This is intentionally narrow:
            # it does not infer a speaker from a reaction or from a bare character mention.
            if previous_quote_match is not None and previous_quote_speakers:
                between_quotes = text[previous_quote_match.end():match.start()]
                if pronoun_attribution_only_re.fullmatch(between_quotes):
                    speakers = set(previous_quote_speakers)
                    semantic_delivery = previous_quote_delivery

            if not speakers:
                previous_quote_match = match
                previous_quote_speakers = set()
                previous_quote_delivery = "in_person"
                explicit_hint = explicit_local_speaker(match.start(), match.end())
                if explicit_hint:
                    speakers = {canonicalize_speaker(explicit_hint)}
            if not speakers:
                continue

            if allowed:
                is_remote_turn = str(semantic_delivery or "in_person").strip().lower() == "remote"
                speakers = {
                    speaker for speaker in speakers
                    if speaker in allowed
                    or (is_remote_turn and EntityResolver.normalize(speaker) in remote_only_speaker_keys)
                }
                if not speakers:
                    quote_spans.append((match.start(), match.end()))
                    continue

            display = cls._normalize_dialogue_text(value, keep_case=True).strip()
            quote_spans.append((match.start(), match.end()))
            if not display or len(re.findall(r"[A-Za-z]", display)) < 2:
                continue
            delivery = quote_delivery(match.start(), match.end(), semantic_delivery)
            if delivery == "excluded":
                remote_excluded += 1
                previous_quote_match = match
                previous_quote_speakers = set()
                previous_quote_delivery = "in_person"
                continue

            segments.append({
                "text": key,
                "display": display,
                "source_speakers": speakers,
                "start": match.start(),
                "end": match.end(),
                "semantic": True,
                "delivery": delivery,
            })
            previous_quote_match = match
            previous_quote_speakers = set(speakers)
            previous_quote_delivery = delivery

        # Explicit screenplay/script labels can be recovered deterministically even when a
        # semantic payload is unavailable. This path uses only the speaker surface, not a
        # hand-maintained speech vocabulary.
        label_pattern = re.compile(
            r"(?m)^\s*([A-Z][A-Za-z0-9.'’\-]*(?:\s+[A-Z][A-Za-z0-9.'’\-]*){0,4})\s*(?::|—|–)\s*([^\n]+?)\s*$"
        )
        for match in label_pattern.finditer(text):
            if any(
                start <= match.start() < end
                or start < match.end() <= end
                for start, end in quote_spans
            ):
                continue
            speaker = canonicalize_speaker(match.group(1).strip())
            if allowed and speaker not in allowed:
                continue
            display = cls._normalize_dialogue_text(match.group(2), keep_case=True).rstrip(",;: ").strip()
            key = cls._dialogue_anchor_key(display)
            if not key or len(re.findall(r"[A-Za-z]", display)) < 2:
                continue
            semantic_speakers, semantic_delivery = consume_semantic_speaker(key)
            if semantic_speakers:
                # The explicit screenplay label and Qwen's semantic speaker must agree.
                # This prevents a malformed model response from reassigning a source line.
                if speaker not in semantic_speakers:
                    continue
                speaker_set = {speaker}
            elif semantic_items:
                # A semantic payload exists and did not bless this line, so do not override Qwen.
                continue
            else:
                speaker_set = {speaker}
                semantic_delivery = "in_person"
            delivery = quote_delivery(match.start(), match.end(), semantic_delivery)
            if delivery == "excluded":
                remote_excluded += 1
                continue
            segments.append({
                "text": key,
                "display": display,
                "source_speakers": speaker_set,
                "start": match.start(),
                "end": match.end(),
                "semantic": bool(semantic_speakers),
                "delivery": delivery,
            })

        segments.sort(key=lambda item: (item.get("start", 0), item.get("end", 0)))
        # Keep source offsets for exact source-to-scene recovery without another model call.
        if remote_excluded:
            print(
                f"[DIRECTOR] dialogue source filter excluded {remote_excluded} remembered/written/anonymous utterance(s); "
                "explicit live remote dialogue is retained as off-screen audio",
                flush=True,
            )
        return segments

    @classmethod
    def _extract_story_spoken_texts(
        cls,
        story: str,
        semantic_spoken_dialogue: list[dict] | None = None,
    ) -> dict[str, set[str]]:
        """Return source dialogue anchors for compatibility with existing callers."""
        anchors: dict[str, set[str]] = {}
        for segment in cls._extract_story_spoken_segments(story, semantic_spoken_dialogue):
            anchor = str(segment.get("text", "") or "").strip()
            if not anchor:
                continue
            anchors.setdefault(anchor, set()).update(
                str(value).strip()
                for value in (segment.get("source_speakers", set()) or set())
                if str(value).strip()
            )
        return anchors

    def _remote_only_dialogue_speakers(self, characters: list[dict]) -> list[str]:
        """Return explicit remote speakers not represented in the visual/canonical cast.

        This is a derived audio whitelist, not a character-discovery path: it never
        mutates or supplements ``characters``. Only the one semantic extraction's
        explicitly remote, source-anchored speaker labels can enter this list.
        """
        canonical = {
            EntityResolver.normalize(str(item.get("name", "")))
            for item in (characters or [])
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }
        canonical |= {
            EntityResolver.normalize(EntityResolver.strip_honorific(str(item.get("name", ""))))
            for item in (characters or [])
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }
        aliases = EntityResolver.build_character_alias_map(characters or [])
        known = canonical | set(aliases) | {
            EntityResolver.normalize(value) for value in aliases.values() if str(value).strip()
        }
        remote = {}
        for item in (getattr(self, "_semantic_spoken_dialogue", []) or []):
            if not isinstance(item, dict) or str(item.get("delivery", "in_person")).strip().lower() != "remote":
                continue
            speaker = str(item.get("speaker", "") or "").strip()
            if not speaker:
                continue
            key = EntityResolver.normalize(speaker)
            titleless = EntityResolver.normalize(EntityResolver.strip_honorific(speaker))
            if key not in known and titleless not in known:
                remote.setdefault(key, speaker)
        return list(remote.values())

    def _normalize_dialogue_speakers(
        self,
        story: str,
        scenes: list[dict],
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

        scene_by_id = {
            str(scene.get("scene_id", "") or "").strip(): scene
            for scene in (scenes or [])
            if isinstance(scene, dict) and str(scene.get("scene_id", "") or "").strip()
        }

        canonical_by_norm = {name.lower(): name for name in allowed_names}
        for remote_name in self._remote_only_dialogue_speakers(characters):
            canonical_by_norm.setdefault(remote_name.lower(), remote_name)
            canonical_by_norm.setdefault(EntityResolver.normalize(remote_name), remote_name)
        aliases = EntityResolver.build_character_alias_map(characters)
        spoken_segments = self._extract_story_spoken_segments(
            story,
            getattr(self, "_semantic_spoken_dialogue", None),
            allowed_speakers=allowed_names,
            speaker_aliases=aliases,
        )
        # Progress is measured in normalized word tokens, not characters, so curly
        # quotes, ellipses, and punctuation changes cannot corrupt the source cursor.
        story_paragraphs = [part for part in re.split(r"\n\s*\n+", str(story or "")) if part.strip()]
        # Scene units may join source paragraphs with spaces during deterministic rebalancing.
        # Map original source offsets into a whitespace-collapsed coordinate space so long
        # Preserve Story inputs still bind each quote to the exact source-derived scene.
        collapsed_chars = []
        original_to_collapsed = [0] * (len(str(story or "")) + 1)
        in_whitespace = False
        for original_index, character in enumerate(str(story or "")):
            if character.isspace():
                if collapsed_chars and not in_whitespace:
                    collapsed_chars.append(" ")
                original_to_collapsed[original_index] = max(0, len(collapsed_chars) - 1)
                in_whitespace = True
            else:
                collapsed_chars.append(character)
                original_to_collapsed[original_index] = len(collapsed_chars) - 1
                in_whitespace = False
        collapsed_story = "".join(collapsed_chars).strip()
        scene_ranges = []
        cursor = 0
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id", "") or "").strip()
            description = str(scene.get("description", "") or scene.get("story_summary", "") or "").strip()
            normalized_description = re.sub(r"\s+", " ", description).strip()
            start_at = collapsed_story.find(normalized_description, cursor) if normalized_description else -1
            if start_at < 0 and normalized_description:
                start_at = collapsed_story.find(normalized_description)
            if start_at >= 0:
                scene_ranges.append((start_at, start_at + len(normalized_description), scene_id))
                cursor = max(cursor, start_at + len(normalized_description))
        segment_scene_ids = []
        for segment in spoken_segments:
            source_position = int(segment.get("start", -1))
            collapsed_position = original_to_collapsed[source_position] if 0 <= source_position < len(original_to_collapsed) else -1
            scene_id = next((sid for lo, hi, sid in scene_ranges if lo <= collapsed_position < hi), None)
            if not scene_id and len(story_paragraphs) == len(scenes) and source_position >= 0:
                paragraph_index = len(re.findall(r"\n\s*\n+", story[:source_position]))
                if 0 <= paragraph_index < len(scenes):
                    scene_id = str(scenes[paragraph_index].get("scene_id", "") or "").strip()
            segment_scene_ids.append(scene_id)

        segment_progress = [0 for _ in spoken_segments]
        segment_first_shot: dict[int, str] = {}
        segment_last_shot: dict[int, str] = {}
        last_source_index: list[int | None] = [None]
        last_display: list[str] = [""]
        dialogue_counts = {
            "input": 0,
            "kept": 0,
            "empty": 0,
            "no_source_anchor": 0,
            "unmatched_text": 0,
        }

        def _consume_source_dialogue(normalized_text: str, shot_scene_id: str):
            last_display[0] = ""
            last_source_index[0] = None
            candidate_key = self._dialogue_anchor_key(normalized_text)
            candidate_tokens = candidate_key.split()
            if not candidate_tokens:
                return None

            for index, segment in enumerate(spoken_segments):
                if segment_scene_ids[index] and segment_scene_ids[index] != shot_scene_id:
                    continue
                if not segment_scene_ids[index]:
                    # Do not permit an unassigned utterance to drift into an arbitrary scene.
                    continue
                source_tokens = str(segment.get("text", "") or "").split()
                progress = segment_progress[index]
                remaining_tokens = source_tokens[progress:]
                if not remaining_tokens or len(candidate_tokens) > len(remaining_tokens):
                    continue
                if remaining_tokens[:len(candidate_tokens)] != candidate_tokens:
                    continue

                new_progress = progress + len(candidate_tokens)
                segment_progress[index] = new_progress
                display_text = str(segment.get("display", "") or "")
                display_spans = list(re.finditer(r"[\w]+(?:['’][\w]+)*", display_text, flags=re.UNICODE))
                if display_spans and len(display_spans) >= new_progress:
                    start_char = display_spans[progress].start() if progress < len(display_spans) else 0
                    end_char = len(display_text) if new_progress == len(source_tokens) else display_spans[new_progress - 1].end()
                    last_display[0] = display_text[start_char:end_char].strip()
                elif new_progress == len(source_tokens):
                    last_display[0] = display_text
                last_source_index[0] = index
                return set(segment.get("source_speakers", set()) or set())
            return None

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
            contextual = EntityResolver.contextual_generic_alias(
                value,
                characters,
                bound_names=None,
                story=story,
            )
            if contextual and contextual.lower() in canonical_by_norm:
                return canonical_by_norm[contextual.lower()]
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
                    dialogue_counts["empty"] += 1
                    continue
                dialogue_counts["input"] += 1
                speaker = str(event.get("speaker", "") or "").strip()
                text = str(event.get("text", "") or "").strip()
                if not speaker or not text:
                    dialogue_counts["empty"] += 1
                    continue

                # When the source story contains explicit speech anchors, only
                # anchored speech can become audio. Substring matching supports
                # a quoted line split into multiple valid events while still
                # rejecting whole narrative/action sentences.
                normalized_text = self._normalize_dialogue_text(text)
                if not spoken_segments:
                    dialogue_counts["no_source_anchor"] += 1
                    continue

                matched_source_speakers = _consume_source_dialogue(
                    normalized_text, str(shot.get("scene_id", "") or "").strip()
                )
                if matched_source_speakers is not None and last_source_index[0] is not None:
                    segment_first_shot.setdefault(last_source_index[0], shot_id)
                    segment_last_shot[last_source_index[0]] = shot_id
                if matched_source_speakers is None:
                    # Qwen sometimes emits a narrative sentence or repeats a source
                    # line in multiple shots. Neither is valid audio. Drop it before
                    # any timing/compilation stage can treat it as speech, and count
                    # the reason so dialogue loss is no longer silent.
                    dialogue_counts["unmatched_text"] += 1
                    continue

                explicit_source_canonicals = {
                    resolved.lower()
                    for source_speaker in matched_source_speakers
                    if (resolved := _resolve(source_speaker)) is not None
                }
                canonical = _resolve(speaker)
                if canonical is None:
                    contextual = EntityResolver.contextual_generic_alias(
                        speaker,
                        characters,
                        bound_names=bound,
                        story=story,
                    )
                    if contextual and contextual.lower() in canonical_by_norm:
                        canonical = canonical_by_norm[contextual.lower()]
                if canonical is None and len(explicit_source_canonicals) == 1:
                    # The exact source utterance and its canonical speaker outrank an
                    # unresolvable generic/alias label emitted by the shot model.
                    source_canonical = next(iter(explicit_source_canonicals))
                    canonical = canonical_by_norm.get(source_canonical, source_canonical)
                    self._record_recovery(
                        "dialogue_speaker_restored_from_source",
                        f"shot={shot_id} speaker={speaker!r} -> {canonical!r}",
                    )
                if canonical is None:
                    self._record_recovery(
                        "dialogue_speaker_unresolved",
                        f"shot={shot_id} speaker={speaker!r}",
                    )
                    raise RuntimeError(
                        "Explicit dialogue speaker could not be canonically resolved from the source: "
                        f"shot={shot_id} speaker={speaker!r}"
                    )

                if matched_source_speakers and not explicit_source_canonicals:
                    # Explicit source attribution is authoritative enough to reject
                    # an unsafe remap, but not to justify deleting the spoken line.
                    # Stop production so the caller can surface the exact semantic
                    # conflict instead of silently losing dialogue.
                    self._record_recovery(
                        "dialogue_source_speaker_unresolved",
                        f"shot={shot_id} speaker={speaker!r} source={sorted(matched_source_speakers)!r}",
                    )
                    raise RuntimeError(
                        "Explicit dialogue source speaker could not be resolved: "
                        f"shot={shot_id} speaker={speaker!r} source={sorted(matched_source_speakers)!r}"
                    )

                normalized_speaker = canonical.lower()
                if explicit_source_canonicals and normalized_speaker not in explicit_source_canonicals:
                    # The source gives explicit speaker provenance that conflicts
                    # with Qwen's attribution. Never silently delete or remap the
                    # line; fail closed so the semantic conflict is visible.
                    # The shot director sees one scene; the source attribution comes from the whole
                    # story. The line and its text are kept; only the speaker label is corrected to
                    # the attributed speaker (preferring one already bound to this shot).
                    corrected = sorted(
                        explicit_source_canonicals,
                        key=lambda name: (name not in bound, name),
                    )[0]
                    self._record_recovery(
                        "dialogue_speaker_source_mismatch",
                        f"shot={shot_id} speaker={speaker!r} -> {canonical_by_norm.get(corrected, corrected)!r} "
                        f"(source={sorted(explicit_source_canonicals)!r})",
                    )
                    canonical = canonical_by_norm.get(corrected, canonical)
                    normalized_speaker = canonical.lower()

                source_delivery = (
                    str(spoken_segments[last_source_index[0]].get("delivery", "in_person"))
                    if last_source_index[0] is not None else str(event.get("delivery", "in_person"))
                ).strip().lower()
                is_remote = source_delivery == "remote"
                if normalized_speaker not in bound and not is_remote:
                    # Only in-person source speech may justify restoring a visible
                    # shot/scene binding. Remote speakers remain canonical audio sources.
                    shot_characters = shot.get("characters")
                    if not isinstance(shot_characters, list):
                        shot_characters = list(shot_characters or [])
                        shot["characters"] = shot_characters
                    existing_norm = {
                        EntityResolver.normalize(str(value or ""))
                        for value in shot_characters
                        if str(value or "").strip()
                    }
                    if EntityResolver.normalize(canonical) not in existing_norm:
                        shot_characters.append(canonical)
                    bound.add(normalized_speaker)

                    scene_id = str(shot.get("scene_id", "") or "").strip()
                    scene = scene_by_id.get(scene_id)
                    if scene is not None:
                        scene_characters = scene.get("characters", [])
                        if not isinstance(scene_characters, list):
                            scene_characters = list(scene_characters or [])
                            scene["characters"] = scene_characters
                        scene_norms = {
                            EntityResolver.normalize(str(value or ""))
                            for value in scene_characters
                            if str(value or "").strip()
                        }
                        if EntityResolver.normalize(canonical) not in scene_norms:
                            scene_characters.append(canonical)

                repaired = dict(event)
                repaired["speaker"] = canonical
                if is_remote:
                    repaired["delivery"] = "remote"
                else:
                    repaired.pop("delivery", None)
                if last_source_index[0] is not None:
                    repaired["_source_segment_index"] = last_source_index[0]
                # Dialogue text must be the exact source span (original casing and
                # punctuation), never a lower-cased matching key.
                if last_display[0]:
                    repaired["text"] = last_display[0]
                repaired_events.append(repaired)
                dialogue_counts["kept"] += 1

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

        # Conservation pass: any source utterance that Qwen's shot output omitted (or
        # only partially emitted) is restored from the exact source span. This is a
        # deterministic assignment, not a model retry, and never invents dialogue.
        shot_by_id = {str(shot.get("shot_id", "") or "").strip(): shot for shot in shots if isinstance(shot, dict)}
        shots_by_scene: dict[str, list[dict]] = {}
        for shot in shots:
            if isinstance(shot, dict):
                shots_by_scene.setdefault(str(shot.get("scene_id", "") or "").strip(), []).append(shot)
        restored_count = 0
        restored_partial_count = 0
        for index, segment in enumerate(spoken_segments):
            source_tokens = str(segment.get("text", "") or "").split()
            progress = segment_progress[index]
            if not source_tokens or progress >= len(source_tokens):
                continue
            display = str(segment.get("display", "") or "").strip()
            spans = list(re.finditer(r"[\w]+(?:['’][\w]+)*", display, flags=re.UNICODE))
            if progress <= 0:
                missing_text = display
            elif spans and progress < len(spans):
                missing_text = display[spans[progress].start():].strip()
                restored_partial_count += 1
            else:
                # A punctuation-only tail is not a separate spoken event.
                continue
            source_speakers = list(segment.get("source_speakers", set()) or set())
            resolved_speakers = [_resolve(value) for value in source_speakers]
            canonical_speaker = next((value for value in resolved_speakers if value), None)
            if not canonical_speaker:
                raise RuntimeError(
                    "Source dialogue conservation failed: cannot resolve source speaker "
                    + repr(source_speakers)
                )

            source_position = int(segment.get("start", -1))
            scene_id = segment_scene_ids[index]
            if not scene_id:
                # Exact source quote in a scene description is a safe fallback. If the
                # same line occurs multiple times, source offsets/ranges above take precedence.
                key = self._dialogue_anchor_key(display)
                candidates = []
                for scene in scenes:
                    if not isinstance(scene, dict):
                        continue
                    text_value = str(scene.get("description", "") or scene.get("story_summary", "") or "")
                    if key and key in self._dialogue_anchor_key(text_value):
                        candidates.append(str(scene.get("scene_id", "") or "").strip())
                if len(candidates) == 1:
                    scene_id = candidates[0]
                elif len(story_paragraphs) == len(scenes) and source_position >= 0:
                    paragraph_index = len(re.findall(r"\n\s*\n+", story[:source_position]))
                    if 0 <= paragraph_index < len(scenes):
                        scene_id = str(scenes[paragraph_index].get("scene_id", "") or "").strip()
            scene_shots = shots_by_scene.get(scene_id or "", [])
            if not scene_shots:
                raise RuntimeError(
                    "Source dialogue conservation failed: no production shot could be mapped to "
                    f"utterance {display!r} (scene={scene_id!r}). No retry was attempted."
                )

            previous_shot_id = segment_last_shot.get(index)
            target = None
            scene_shot_index = {str(item.get("shot_id", "") or "").strip(): pos for pos, item in enumerate(scene_shots)}
            if progress > 0 and previous_shot_id:
                previous_pos = scene_shot_index.get(previous_shot_id, -1)
                target = scene_shots[previous_pos + 1] if 0 <= previous_pos + 1 < len(scene_shots) else scene_shots[max(0, previous_pos)]
            else:
                next_index = next((j for j in range(index + 1, len(spoken_segments))
                                   if segment_scene_ids[j] == scene_id and j in segment_first_shot), None)
                previous_index = next((j for j in range(index - 1, -1, -1)
                                       if segment_scene_ids[j] == scene_id and j in segment_last_shot), None)
                if next_index is not None:
                    next_pos = scene_shot_index.get(segment_first_shot[next_index], 0)
                    target = scene_shots[max(0, next_pos - 1)]
                elif previous_index is not None:
                    previous_pos = scene_shot_index.get(segment_last_shot[previous_index], len(scene_shots) - 1)
                    target = scene_shots[min(len(scene_shots) - 1, previous_pos + 1)]
            is_remote = str(segment.get("delivery", "in_person")).strip().lower() == "remote"
            if target is None:
                # Remote audio belongs to a shot grounded in the established receiver,
                # channel, or listener reaction—not a fabricated visible speaker.
                speech_action_re = re.compile(r"\b(?:say|says|said|speak|speaks|speaking|ask|asks|asked|reply|replies|replied|shout|shouts|shouted|whisper|whispers|whispered|call|calls|called|tell|tells|told|listen|listens|turned|startled|reacted|flinched|heard|crackled|intercom|radio|terminal|speaker)\b", re.IGNORECASE)
                target = min(
                    scene_shots,
                    key=lambda item: (
                        not bool(speech_action_re.search(str(item.get("action", "") or "") + " " + str(item.get("visual_prompt", "") or ""))),
                        len(item.get("dialogue_events", []) or []),
                    ) if is_remote else (
                        EntityResolver.normalize(canonical_speaker) not in {
                            EntityResolver.normalize(str(name)) for name in (item.get("characters", []) or [])
                        },
                        not bool(speech_action_re.search(str(item.get("action", "") or ""))),
                        len(item.get("dialogue_events", []) or []),
                    ),
                )

            target_events = target.get("dialogue_events", [])
            if not isinstance(target_events, list):
                target_events = []
            event = {"speaker": canonical_speaker, "text": missing_text, "_source_segment_index": index}
            if is_remote:
                event["delivery"] = "remote"
            if progress > 0 and previous_shot_id:
                event["continues_from_previous_shot"] = True
                previous_events = shot_by_id[previous_shot_id].get("dialogue_events", [])
                if previous_events and isinstance(previous_events[-1], dict):
                    previous_events[-1]["continues_to_next_shot"] = True
            target_events.append(event)
            target["dialogue_events"] = target_events

            speech_action_re = re.compile(r"\b(?:say|says|said|speak|speaks|speaking|ask|asks|asked|reply|replies|replied|shout|shouts|shouted|whisper|whispers|whispered|call|calls|called|tell|tells|told|listen|listens|heard|hears|crackled|transmitted)\b", re.IGNORECASE)
            action_text = str(target.get("action", "") or "").strip()
            if is_remote:
                remote_clause = f"The established in-world channel carries {canonical_speaker}'s voice; keep the source off-screen."
                channel_re = r"\b(?:radio|intercom|terminal|speaker|receiver|transmission|communicator|phone|channel|voice|heard|listening)\b"
                visual_text = str(target.get("visual_prompt", "") or "").strip()
                if not re.search(channel_re, action_text + " " + visual_text, re.IGNORECASE):
                    target["action"] = ((action_text.rstrip(" .;,:—–-") + "; ") if action_text else "") + remote_clause
                if not re.search(channel_re, visual_text, re.IGNORECASE):
                    target["visual_prompt"] = " ".join((visual_text + " A receiver or established channel carries the off-screen voice; the source is not visible.").split())
            else:
                speaker_surface_present = any(
                    re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", action_text, re.IGNORECASE)
                    for surface in self._name_surfaces(canonical_speaker)
                )
                if not speech_action_re.search(action_text) or not speaker_surface_present:
                    speaking_clause = f"{canonical_speaker} visibly speaks the source line to the listener."
                    if action_text:
                        action_base = " ".join(action_text.split()[:22]).rstrip(" .;,:—–-")
                        target["action"] = f"{action_base}; {speaking_clause}" if action_base else speaking_clause
                    else:
                        target["action"] = f"{canonical_speaker} speaks the source line, with their face and mouth clearly visible to the listener."
                visual_text = str(target.get("visual_prompt", "") or "").strip()
                visual_speaker_present = any(
                    re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", visual_text, re.IGNORECASE)
                    for surface in self._name_surfaces(canonical_speaker)
                )
                if not speech_action_re.search(visual_text) or not visual_speaker_present:
                    visual_base = " ".join(visual_text.split()[:30])
                    target["visual_prompt"] = (visual_base + " " if visual_base else "") + f"{canonical_speaker} visibly speaking, mouth and expression clear in frame."
                    target["visual_prompt"] = " ".join(target["visual_prompt"].split())

                target_chars = target.get("characters", [])
                if not isinstance(target_chars, list):
                    target_chars = list(target_chars or [])
                    target["characters"] = target_chars
                if not any(EntityResolver.normalize(str(name)) == EntityResolver.normalize(canonical_speaker) for name in target_chars):
                    target_chars.append(canonical_speaker)
            target["speaking_characters"] = list(dict.fromkeys(
                str(item.get("speaker", "") or "").strip()
                for item in target_events if isinstance(item, dict) and str(item.get("speaker", "") or "").strip()
            ))
            target["speech_text"] = " ".join(str(item.get("text", "") or "").strip() for item in target_events if isinstance(item, dict) and str(item.get("text", "") or "").strip())
            scene = scene_by_id.get(scene_id or "")
            if scene is not None and not is_remote:
                scene_chars = scene.get("characters", [])
                if not isinstance(scene_chars, list):
                    scene_chars = list(scene_chars or [])
                    scene["characters"] = scene_chars
                if not any(EntityResolver.normalize(str(name)) == EntityResolver.normalize(canonical_speaker) for name in scene_chars):
                    scene_chars.append(canonical_speaker)
            segment_progress[index] = len(source_tokens)
            restored_count += 1

        for shot in shots:
            if not isinstance(shot, dict):
                continue
            events = shot.get("dialogue_events", [])
            if not isinstance(events, list):
                continue
            if any(isinstance(event, dict) and "_source_segment_index" in event for event in events):
                events.sort(key=lambda event: int(event.get("_source_segment_index", 10**9)) if isinstance(event, dict) else 10**9)
            for event in events:
                if isinstance(event, dict):
                    event.pop("_source_segment_index", None)
            shot["dialogue_events"] = events
            shot["speaking_characters"] = list(dict.fromkeys(
                str(event.get("speaker", "") or "").strip()
                for event in events if isinstance(event, dict) and str(event.get("speaker", "") or "").strip()
            ))
            shot["speech_text"] = " ".join(
                str(event.get("text", "") or "").strip()
                for event in events if isinstance(event, dict) and str(event.get("text", "") or "").strip()
            )

        if restored_count:
            print(
                f"[DIRECTOR] dialogue source conservation restored={restored_count} "
                f"(partial_tails={restored_partial_count}); no extra Qwen call",
                flush=True,
            )

        dropped = dialogue_counts["input"] - dialogue_counts["kept"]
        if dialogue_counts["input"] and dropped:
            reasons = ", ".join(
                f"{key}={value}"
                for key, value in dialogue_counts.items()
                if key not in {"input", "kept"} and value
            )
            print(
                f"[DIRECTOR] dialogue reconciliation kept={dialogue_counts['kept']}/{dialogue_counts['input']} "
                f"dropped={dropped} ({reasons or 'unclassified'})",
                flush=True,
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
        remote_only_names = {
            str(value).strip().lower()
            for value in self._remote_only_dialogue_speakers(characters)
            if str(value).strip()
        }
        allowed_names |= remote_only_names
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
                remote_delivery = str(event.get("delivery", "in_person")).strip().lower() == "remote"
                if speaker not in allowed_names:
                    raise RuntimeError(
                        f"Shot {shot_id} contains unknown dialogue speaker '{event.get('speaker', '')}'."
                    )
                if speaker in remote_only_names and not remote_delivery:
                    raise RuntimeError(
                        f"Shot {shot_id} uses audio-only remote speaker '{event.get('speaker', '')}' for in-person dialogue."
                    )
                if speaker not in bound and not remote_delivery:
                    raise RuntimeError(
                        f"Shot {shot_id} has in-person dialogue speaker '{event.get('speaker', '')}' not present in its character bindings."
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
    def _shot_batch_dialogue_index(
        user_payload: str,
    ) -> tuple[dict[str, dict[str, dict]], list[str]]:
        """Index the exact, scene-local dialogue whitelist used by a shot batch."""
        try:
            payload = json.loads(str(user_payload or ""))
        except (TypeError, json.JSONDecodeError):
            return {}, []
        scenes = payload.get("scenes", []) if isinstance(payload, dict) else []
        if not isinstance(scenes, list):
            return {}, []
        result: dict[str, dict[str, dict]] = {}
        all_ids: list[str] = []
        for scene in scenes:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id", "") or "").strip()
            if not scene_id:
                continue
            local: dict[str, dict] = {}
            entries = scene.get("source_dialogue", [])
            if isinstance(entries, list):
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    dialogue_id = str(entry.get("dialogue_id", "") or "").strip()
                    speaker = str(entry.get("speaker", "") or "").strip()
                    text = str(entry.get("text", "") or "").strip()
                    if not dialogue_id or not speaker or not text:
                        continue
                    if dialogue_id in local:
                        # IDs are compiler-owned and must be unique within their scene.
                        raise ValueError(
                            f"Duplicate source dialogue ID {dialogue_id!r} in scene {scene_id!r}."
                        )
                    local[dialogue_id] = {"speaker": speaker, "text": text}
                    all_ids.append(dialogue_id)
            result[scene_id] = local
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("Source dialogue IDs must be unique across a shot batch.")
        return result, sorted(all_ids)

    def _expand_shot_batch_dialogue_references(
        self,
        shots: list[dict],
        source_dialogue_index: dict[str, dict],
    ) -> list[dict]:
        """Restore compact dialogue-ID references to exact source text and speakers.

        One source ID may be assigned to at most one shot in its scene. Legacy
        speaker/text events are retained only for backwards compatibility and are
        still subjected to the normal source-anchor reconciliation pipeline.
        """
        expanded_shots: list[dict] = []
        used_ids: set[str] = set()
        for raw_shot in shots or []:
            if not isinstance(raw_shot, dict):
                continue
            shot = dict(raw_shot)
            raw_events = shot.get("dialogue_events", [])
            if not isinstance(raw_events, list):
                raw_events = []
            expanded_events = []
            for raw_event in raw_events:
                if not isinstance(raw_event, dict):
                    continue
                dialogue_id = str(raw_event.get("source_dialogue_id", "") or "").strip()
                if not dialogue_id:
                    # Backwards compatibility with older cached/nonconforming responses;
                    # the later source-anchor reconciler remains authoritative.
                    if str(raw_event.get("speaker", "") or "").strip() and str(raw_event.get("text", "") or "").strip():
                        expanded_events.append(dict(raw_event))
                    continue
                source = source_dialogue_index.get(dialogue_id)
                if source is None:
                    self._record_recovery(
                        "unknown_shot_dialogue_reference",
                        f"shot={shot.get('shot_id', '')!r} id={dialogue_id!r}; source-scene whitelist enforced",
                    )
                    continue
                if dialogue_id in used_ids:
                    self._record_recovery(
                        "duplicate_shot_dialogue_reference",
                        f"shot={shot.get('shot_id', '')!r} id={dialogue_id!r}; repeated reference dropped",
                    )
                    continue
                used_ids.add(dialogue_id)
                expanded_events.append({
                    "speaker": str(source["speaker"]),
                    "text": str(source["text"]),
                    "continues_from_previous_shot": False,
                    "continues_to_next_shot": False,
                })
            shot["dialogue_events"] = expanded_events
            expanded_shots.append(shot)
        return expanded_shots

    @staticmethod
    def _shot_batch_story_context_capacities(max_chars: int) -> list[int]:
        """Largest-first global-story context caps for fitting two-scene batches.

        Per-scene descriptions, canonical cast, source-dialogue spans and local
        continuity notes are never changed by this strategy. Only the redundant
        global narrative recap is compacted, and only when the context budget
        requires it. The original configured cap is always tried first.
        """
        maximum = max(1, int(max_chars))
        candidates = [maximum, min(maximum, 2400), min(maximum, 1800), min(maximum, 1200)]
        result = []
        for value in candidates:
            if value not in result:
                result.append(value)
        return result

    @staticmethod
    def _shot_batch_completion_budget(scene_count: int) -> int:
        """Return the token cap for a batched shot request.

        The 600-token per-shot reserve is multiplied by the configured
        shots-per-scene and the number of scenes in this batch, then capped
        by the global Director token ceiling. This estimate is calibrated
        against the October 20 traces after compacting derived response fields;
        it preserves headroom above observed compact-output estimates and frees
        context capacity for effective two-scene batching. Live vLLM traces must
        confirm the budget remains sufficient on the deployed model/runtime.
        """
        count = max(1, int(scene_count))
        return min(
            int(DIRECTOR_MAX_TOKENS),
            max(320, 600 * int(DIRECTOR_SHOTS_PER_SCENE) * count),
        )

    @_with_faulthandler_watchdog
    def _reconcile_critique(self, critique: dict, plan: dict) -> dict:
        """Make the critic's verdict self-consistent and add deterministic checks the 14B critic misses.

        Trace evidence: the critic returned score 10 for a plan with a sign spoken as dialogue and
        contradictory lighting, and score 1 / status "review" with ZERO findings elsewhere. An
        unexplained score is not actionable, so the verdict is derived from findings that can be pointed to.
        """
        if not isinstance(critique, dict):
            return critique
        findings = [str(f).strip() for f in (critique.get("findings") or []) if str(f).strip()]
        shot_findings = [
            f for f in (critique.get("shot_findings") or [])
            if isinstance(f, dict) and str(f.get("finding", "")).strip()
        ]

        shots = [s for s in (plan.get("shots") or []) if isinstance(s, dict)]
        by_scene: dict[str, list[dict]] = {}
        for shot in shots:
            by_scene.setdefault(str(shot.get("scene_id", "") or ""), []).append(shot)
        seen_dialogue: dict[str, str] = {}
        for shot in shots:
            sid = str(shot.get("shot_id", "") or "")
            for event in shot.get("dialogue_events") or []:
                if not isinstance(event, dict):
                    continue
                key = self._normalize_dialogue_text(event.get("text", ""))
                if key and key in seen_dialogue and not event.get("continues_from_previous_shot"):
                    shot_findings.append({"shot_id": sid, "severity": "warning",
                                          "finding": f"Dialogue line repeats {seen_dialogue[key]}."})
                elif key:
                    seen_dialogue[key] = sid
        for scene_id, group in by_scene.items():
            lights = {str(s.get("lighting", "") or "").strip().lower() for s in group}
            lights.discard("")
            if len(lights) > 1:
                shot_findings.append({"shot_id": str(group[0].get("shot_id", "") or scene_id), "severity": "warning",
                                      "finding": f"Scene {scene_id} changes lighting between shots."})
            places = {str(s.get("location", "") or "").strip().lower() for s in group}
            places.discard("")
            if len(places) > 1:
                shot_findings.append({"shot_id": str(group[0].get("shot_id", "") or scene_id), "severity": "warning",
                                      "finding": f"Scene {scene_id} changes location between shots."})
            pairs = [(str(s.get("camera_shot", "")).strip().lower(), str(s.get("camera_movement", "")).strip().lower()) for s in group]
            if len(pairs) > 1 and len(set(pairs)) == 1:
                shot_findings.append({"shot_id": str(group[0].get("shot_id", "") or scene_id), "severity": "warning",
                                      "finding": f"Scene {scene_id} repeats the same framing and movement."})

        def _is_case_only_roster_false_positive(finding: str) -> bool:
            lowered = str(finding or "").casefold()
            if "roster" not in lowered or not re.search(r"\b(mismatch|different|missing|not in)\b", lowered):
                return False
            quoted = re.findall(r"['\"]([^'\"]{1,80})['\"]", str(finding or ""))
            if len(quoted) < 2:
                return False
            return EntityResolver.normalize(quoted[0]) == EntityResolver.normalize(quoted[1])

        def _is_cross_scene_coverage_false_positive(finding: str) -> bool:
            lowered = str(finding or "").casefold()
            if not re.search(r"\b(framing|action|distinct|similar|repeat|repeated|coverage)\b", lowered):
                return False
            scene_ids = re.findall(r"\b(scene_\d+)_shot_\d+\b", lowered)
            return len(scene_ids) >= 2 and len(set(scene_ids)) > 1

        def _is_unsupported_finding(finding: str) -> bool:
            return _is_case_only_roster_false_positive(finding) or _is_cross_scene_coverage_false_positive(finding)

        original_finding_count = len(findings) + len(shot_findings)
        findings = [item for item in findings if not _is_unsupported_finding(item)]
        shot_findings = [
            item for item in shot_findings
            if not _is_unsupported_finding(str(item.get("finding", "") or ""))
        ]
        filtered_finding_count = original_finding_count - (len(findings) + len(shot_findings))
        if filtered_finding_count:
            print(
                f"[DIRECTOR] critic post-filter removed {filtered_finding_count} unsupported case/cross-scene finding(s)",
                flush=True,
            )

        # Reject no-op and out-of-vocabulary camera patches before the orchestrator
        # sees them. Valid aliases are canonicalized; unknown labels are not applied.
        from planner.qwen_director_prompts import (
            CAMERA_MOVEMENT_OPTIONS, CAMERA_SHOT_OPTIONS,
            normalize_camera_movement, normalize_camera_shot,
        )
        shot_by_id = {str(item.get("shot_id", "") or "").strip(): item for item in shots
                       if str(item.get("shot_id", "") or "").strip()}
        patchable_fields = {
            "action", "camera_shot", "camera_movement", "lens_and_depth_of_field",
            "composition_notes", "mood", "visual_prompt", "overall_soundscape", "non_diegetic_music",
        }
        filtered_patches = []
        for patch_item in critique.get("shot_patches", []) or []:
            if not isinstance(patch_item, dict):
                continue
            shot_id = str(patch_item.get("shot_id", "") or "").strip()
            current_shot = shot_by_id.get(shot_id)
            patch = patch_item.get("patch", {})
            if current_shot is None or not isinstance(patch, dict):
                continue
            changed = {}
            for field, raw_value in patch.items():
                if field not in patchable_fields or not isinstance(raw_value, str) or not raw_value.strip():
                    continue
                value = raw_value.strip()
                if field == "camera_shot":
                    value, status = normalize_camera_shot(value)
                    if status == "fallback" or value not in CAMERA_SHOT_OPTIONS:
                        continue
                elif field == "camera_movement":
                    value, status = normalize_camera_movement(value)
                    if status == "fallback" or value not in CAMERA_MOVEMENT_OPTIONS:
                        continue
                if value != str(current_shot.get(field, "") or "").strip():
                    changed[field] = value
            if changed:
                filtered_patches.append({"shot_id": shot_id, "patch": changed})
        critique["shot_patches"] = filtered_patches

        critique["shot_findings"] = shot_findings
        has_findings = bool(findings or shot_findings)
        try:
            score = float(critique.get("overall_score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        if not has_findings:
            # Never manufacture a score. An unexplained low critic score is itself an actionable
            # defect in the critic output, so surface that inconsistency without rewriting the score.
            if 0.0 < score < 8.0:
                findings.append(
                    f"Critic returned a low overall score ({score:g}) without an actionable finding."
                )
                critique["findings"] = findings
                critique["status"] = "review"
            else:
                critique["status"] = "pass"
        else:
            critique["status"] = "review"
            ceiling = max(1.0, 10.0 - 1.5 * len(shot_findings) - len(findings))
            critique["overall_score"] = round(min(score or ceiling, ceiling), 1)
        critique["findings"] = findings
        return critique

    def critique_plan(self, *, mode: str, user_input: str, plan: dict) -> dict | None:
        """Run the read-only cinematic critique when the shared feature setting enables it.

        H3_DIRECTOR_CRITIQUE is the preferred explicit override; H3_DIRECTOR_CRITIC
        remains a legacy override, followed by the runtime manifest default.
        """
        if not director_critic_enabled():
            print(
                "[QWEN] director_critique skipped (set H3_DIRECTOR_CRITIQUE=1 to enable)",
                flush=True,
            )
            self._print_qwen_summary("FINAL")
            # None is intentional: ProductionOrchestrator will not attach an empty
            # critique or needlessly recompile the same shots after a skipped review.
            return None
        system_prompt = """
    You audit a film production plan against its story. Check each item and report only defects you
    can point to.

    EVIDENCE RULES
    - Every finding must include a short verbatim quote from the story or the affected shot field.
      For a mismatch, quote the relevant phrase from both sides. If exact support is unavailable, omit the finding.
    - Compare character names case-insensitively for identity after trimming whitespace. Capitalization alone
      is not a roster defect; preserve canonical roster spelling in any patch.
    - A person mentioned only as a radio/intercom/recording voice is not necessarily visible in the shot.
      Do not report the remote speaker as missing from the visible-character list unless the story places them there.
    - Do not claim an event or object transition is absent when the story states it explicitly. Read the whole
      relevant sentence before reporting a continuity defect.

    CHECKS
    1. ROSTER: enforce the roster in both directions. Every name in a scene or shot `characters` list
       must appear in `roster`, and every roster entry must correspond to an actual active person/character
       established by the story (not merely a remembered, missing, recorded, document-only, machine, or object entity).
    2. FIDELITY: each shot `action` and `visual_prompt` must depict events that happen in the story
       for that scene. Report invented events, objects, or people.
    3. MATCH: `visual_prompt` must describe the same moment as `action`. Report contradictions.
    4. CONTINUITY: check lighting, location, and who is present between consecutive shots, AND track
       story-established key objects across the full story. Once a character pockets, hands over, drops,
       breaks, seals, or destroys an object, later shots must respect that state unless an intervening
       event explains the change. Do not show an object back at an earlier location after it was moved.
    5. FRAMING: shots of one scene must not repeat the same camera_shot and camera_movement.
    6. ACTION DISTINCTNESS: paired shots in one scene must use complementary actions, not two
       paraphrases of the same observable action. If they repeat the same beat, identify both shot_ids
       and quote both action fields. A patch may refine the later shot only by selecting another
       reaction, evidence detail, or state change already supported by the story/scene; never invent an event.

    OUTPUT RULES
    - overall_score is 1 (unusable) to 10 (no defects).
    - status is "review" if there is at least one finding, otherwise "pass".
    - Report at most 3 findings, 3 shot_findings, and 1 shot_patch. Merge repeats of the same
      defect into ONE finding. Keep each finding concise. Never return a patch identical to the current field.
    - Every finding names a shot_id or scene_id and states one actionable defect concisely; include only the short evidence quote needed to prove it.
    - Add a shot_patch only for a defect in checks 2, 3, 5, or 6. Patch only `action`, `camera_shot`,
      `camera_movement`, `lens_and_depth_of_field`, `composition_notes`, `mood`, `visual_prompt`,
      `overall_soundscape`, or `non_diegetic_music`, using the same style and length. Never patch
      ids, characters, timing, lighting, color_temperature, or continuity. If no concrete defect exists,
      return empty arrays.
    """.strip()
        def _slim_shot(shot: dict) -> dict:
            return {
                "shot_id": str(shot.get("shot_id", "") or ""),
                "scene_id": str(shot.get("scene_id", "") or ""),
                "camera_shot": str(shot.get("camera_shot", "") or ""),
                "camera_movement": str(shot.get("camera_movement", "") or ""),
                "lens_and_depth_of_field": str(shot.get("lens_and_depth_of_field", "") or ""),
                "lighting": str(shot.get("lighting", "") or ""),
                "color_temperature": str(shot.get("color_temperature", "") or ""),
                "location": str(shot.get("location", "") or ""),
                "mood": str(shot.get("mood", "") or ""),
                "characters": shot.get("characters", []) or [],
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
            "roster": [
                str(c.get("name", "") or "").strip()
                for c in (plan.get("characters", []) or [])
                if isinstance(c, dict) and str(c.get("name", "") or "").strip()
            ],
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
        from planner.qwen_director_prompts import CAMERA_MOVEMENT_OPTIONS, CAMERA_SHOT_OPTIONS
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "overall_score": {"type": "number"},
                "status": {"type": "string", "enum": ["pass", "review"]},
                "findings": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 280}},
                "shot_findings": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "shot_id": {"type": "string"},
                            "severity": {"type": "string", "enum": ["info", "warning", "critical"]},
                            "finding": {"type": "string", "maxLength": 280},
                        },
                        "required": ["shot_id", "severity", "finding"],
                    },
                },
                "recommended_focus": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 120}},
                "shot_patches": {
                    "type": "array",
                    "maxItems": 1,
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
                                    "camera_shot": {"type": "string", "enum": list(CAMERA_SHOT_OPTIONS), "maxLength": 24},
                                    "camera_movement": {"type": "string", "enum": list(CAMERA_MOVEMENT_OPTIONS), "maxLength": 12},
                                    "lens_and_depth_of_field": {"type": "string"},
                                    "composition_notes": {"type": "string"},
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
                max_completion=1400,
                json_mode=True,
                disable_thinking=True,
                response_schema=schema,
            )
            result = self._reconcile_critique(result, plan)
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

        if "_character_semantic_calls_used" in creative:
            merged["_character_semantic_calls_used"] = int(
                creative.get("_character_semantic_calls_used", 0) or 0
            )
        if "_semantic_spoken_dialogue" in creative:
            merged["_semantic_spoken_dialogue"] = deepcopy(
                creative.get("_semantic_spoken_dialogue", []) or []
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


# BEGIN H3_KAGGLE_RUNTIME_FIX_20261010
def _h3_apply_kaggle_runtime_fix_20261010():
    if getattr(QwenDirector, "_H3_KAGGLE_RUNTIME_FIX_20261010", False):
        return

    from copy import deepcopy
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE

    # 1) Compact extraction schema while preserving the planner contract.
    # is_character remains required because the existing planner reads it.
    original_schema = QwenDirector._character_extraction_json_schema

    def compact_character_schema():
        schema = deepcopy(original_schema())

        candidates = schema["properties"]["candidates"]
        candidates.pop("maxItems", None)  # no artificial cast-size limit

        item = candidates["items"]
        props = item["properties"]

        props["name"] = {
            "type": "string",
            "maxLength": 96,
        }
        props["entity_type"] = {
            "type": "string",
            "enum": ["PERSON", "CHARACTER", "SENTIENT"],
        }
        props["is_character"] = {"type": "boolean"}
        props["identity_type"] = {
            "type": "string",
            "enum": [
                "named_character",
                "relational_character",
                "descriptive_character",
            ],
        }
        props["aliases"] = {
            "type": "array",
            "maxItems": 2,
            "items": {"type": "string", "maxLength": 96},
        }
        props["relationship_to"] = {
            "type": "string",
            "maxLength": 96,
        }
        props["relationship"] = {
            "type": "string",
            "maxLength": 64,
        }

        # Keep is_character=True explicit for planner compatibility.
        # Relationship/alias metadata is optional, not mandatory boilerplate.
        item["required"] = [
            "name",
            "entity_type",
            "is_character",
            "identity_type",
        ]
        item["additionalProperties"] = False

        dialogue = schema["properties"]["quote_classifications"]
        dialogue["maxItems"] = 256  # every source quote receives a bounded classification
        dialogue["items"]["properties"]["speaker"] = {"type": "string", "maxLength": 96}
        return schema

    QwenDirector._character_extraction_json_schema = staticmethod(
        compact_character_schema
    )

    # 2) Keep the extraction prompt single-source. Qwen returns its cast directly
    # from the completed story; no deterministic hint list or planned cast is sent
    # for per-name classification, and no post-hoc roster add-back is performed.
    original_chat_json = QwenDirector._chat_json

    def compact_chat_json(self, system_prompt, user_prompt, *args, **kwargs):
        if kwargs.get("call_name") == "character_entity_extraction" and not kwargs.get("response_schema"):
            kwargs["response_schema"] = QwenDirector._character_extraction_json_schema()
        return original_chat_json(
            self, system_prompt, user_prompt, *args, **kwargs
        )

    QwenDirector._chat_json = compact_chat_json

    # 3) Distinguish extraction truncation from an empty validated roster.
    class KaggleCharacterExtractionTruncated(RuntimeError):
        def __init__(self, message, partial=""):
            super().__init__(message)
            self.partial = partial

    QwenDirector._H3_KaggleCharacterExtractionTruncated = (
        KaggleCharacterExtractionTruncated
    )
    original_post_chat = QwenDirector._post_chat

    def guarded_post_chat(self, **kwargs):
        response = original_post_chat(self, **kwargs)
        response_format = kwargs.get("response_format") or {}
        schema_name = (
            (response_format.get("json_schema") or {}).get("name") or ""
        )

        if schema_name == "character_entity_extraction":
            try:
                choice = response["choices"][0]
                finish_reason = str(choice.get("finish_reason", "") or "")
                partial = str(
                    (choice.get("message") or {}).get("content", "") or ""
                )
            except (KeyError, IndexError, TypeError, AttributeError):
                finish_reason, partial = "", ""

            if finish_reason == "length":
                print(
                    "[QWEN] character_entity_extraction finish_reason=length; "
                    "raising explicit truncation error",
                    flush=True,
                )
                raise KaggleCharacterExtractionTruncated(
                    "QwenJsonTruncated: character_entity_extraction hit the "
                    "completion limit before complete JSON.",
                    partial=partial[:4000],
                )

        return response

    QwenDirector._post_chat = guarded_post_chat

    # Story-writing guidance is centralized in qwen_director_prompts.py.
    # Do not append duplicate story guards or word-count subtargets here.

    QwenDirector._H3_KAGGLE_RUNTIME_FIX_20261010 = True

_h3_apply_kaggle_runtime_fix_20261010()


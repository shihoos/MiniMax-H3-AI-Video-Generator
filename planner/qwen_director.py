from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
from pathlib import Path
import os
import json
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
    DIRECTOR_N_CTX,
    DIRECTOR_STORY_MAX_TOKENS,
    DIRECTOR_STORY_THINKING_TOP_K,
    DIRECTOR_STORY_THINKING_MIN_P,
    DIRECTOR_STORY_PRESENCE_PENALTY,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    director_critic_enabled,
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
        self._planned_cast: list[str] = []
        self._last_story_draft: str = ""
        self._story_min_words: int = 420

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
        self._semantic_spoken_dialogue = []
        self._planned_cast = []
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
            # Preserve Story may legitimately have used zero calls when its text contained no
            # quote-delimited prose dialogue; the checkpoint still records its exact call count.
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
                    }
                    for item in restored_semantic_dialogue
                    if isinstance(item, dict)
                    and str(item.get("text", "") or "").strip()
                    and str(item.get("speaker", "") or "").strip()
                ]

        # Preserve the September 25 per-mode sampler; thinking remains enabled
        # only for the single story-writing call in this single-pass profile.
        temperature, top_p = self._sampling_for_mode(mode)

        # ----------------------------------------------------
        # PASS 1A: narrative
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

            story_system = (
                self._story_text_system(
                    mode
                )
            )

            story_user = self._story_text_user(
                mode,
                user_input,
                source_character_names=source_character_names,
            )
            # One deliberate plan-before-write call is enabled by default because
            # the October 15 run showed that unplanned prose and the final cast can
            # diverge. The plan is provisional; it does not constrain cast size or
            # override Expand Story source canon. Disable with H3_STORY_PLAN=0.
            story_bible = self._maybe_plan_story_bible(
                mode,
                user_input,
                source_character_names,
            )
            bible_block = self._format_story_bible(story_bible)
            if bible_block:
                story_user += "\n\n" + bible_block
            # Exactly one prose-writing request. No automatic story repair/retry is
            # run in the default path; hard/soft validation behavior stays unchanged.
            story, generated_story = self._generate_story_once(
                mode,
                user_input,
                story_system,
                story_user,
                temperature=temperature,
                top_p=top_p,
                source_character_names=source_character_names,
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
            # the six-paragraph contract, so repair topology here without changing prose.
            paragraphs = [
                part.strip()
                for part in re.split(r"\n\s*\n+", story)
                if part.strip()
            ]
            if len(paragraphs) != 6:
                story = self._coerce_story_to_six_paragraphs(story)
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
                # Preserve mode keeps the deterministic user-story roster as source of truth.
                # Use the single semantic Qwen pass only when quote-delimited prose dialogue exists;
                # explicit screenplay-style speaker labels remain fully deterministic. This preserves
                # the zero-inference fast path for stories that contain no quote-style dialogue.
                canonical_characters = planner.create_characters(
                    canonical_source_story,
                    qwen_character_extractor=None,
                )
                if self._preserve_story_requires_semantic_dialogue(story):
                    preserve_hints = [
                        str(character.name).strip()
                        for character in canonical_characters
                        if character is not None and str(getattr(character, "name", "")).strip()
                    ]
                    self.extract_character_entities(
                        story,
                        deterministic_candidates=preserve_hints,
                    )
                    if not self._semantic_spoken_dialogue:
                        self._record_recovery(
                            "preserve_semantic_dialogue_empty",
                            "Preserve Story contains quote-delimited prose but the semantic extraction returned no spoken dialogue items.",
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

                    batch_response = None
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

    def _maybe_plan_story_bible(
        self,
        mode: str,
        user_input: str,
        source_character_names: list[str] | None,
    ) -> dict | None:
        """Run one Qwen planning call before prose; optional by explicit opt-out."""
        self._planned_cast = []
        if mode not in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
            return None
        enabled = os.getenv("H3_STORY_PLAN", "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if not enabled:
            print("[QWEN] story_plan skipped (H3_STORY_PLAN=0)", flush=True)
            return None
        return self._plan_story_bible(mode, user_input, source_character_names)

    def _validate_story_output_contracts(
        self,
        mode: str,
        user_input: str,
        story: str,
        source_character_names: list[str] | None = None,
        enforce_planned_cast: bool = False,
        *,
        strict: bool = True,
    ) -> str:
        """Validate the current story without rewriting its topology.

        Raw model output is authoritative. In strict mode every detected defect raises;
        the single-pass writer uses strict=False so soft word/source-coverage misses are
        logged while the already-generated draft continues without a second Qwen call.
        Paragraph topology is normalized deterministically before this validation.
        """
        working = self._normalize_story(story)
        errors: list[str] = []
        # Prevent a warning from one draft leaking into a later resume/checkpoint.
        self._last_story_contract_warnings = []

        try:
            self._validate_mode_output(mode, user_input, working)
        except RuntimeError as exc:
            errors.append(str(exc))

        if enforce_planned_cast and mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE) and self._planned_cast:
            try:
                planner = self._planner()
                missing_planned_cast = []
                mention_only_planned_cast = []
                for name in dict.fromkeys(str(value).strip() for value in self._planned_cast if str(value).strip()):
                    if planner.identity_presence(working, name) == "absent":
                        missing_planned_cast.append(name)
                    elif planner._is_mention_only_identity(working, name):
                        mention_only_planned_cast.append(name)
                if missing_planned_cast:
                    errors.append(
                        "Story omitted planned on-screen cast member(s): "
                        + ", ".join(missing_planned_cast[:8])
                    )
                if mention_only_planned_cast:
                    errors.append(
                        "Story only mentions planned cast member(s) without present-scene action: "
                        + ", ".join(mention_only_planned_cast[:8])
                    )
            except Exception as exc:
                # Fail closed for a plan/story identity contract error; do not silently
                # allow a plan whose cast was promised to be present to be ignored.
                errors.append(
                    "Could not validate Story Bible on-screen cast contract: "
                    + type(exc).__name__
                )

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
    # Story generation: one thinking-mode call, deterministic checks, no retries
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
        kept = kept[:6]
        while kept and not self._ends_cleanly(kept[-1]):
            kept.pop()
        return "\n\n".join(kept) if len(kept) >= 5 else ""

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

        return issues

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
    ) -> tuple[str, bool]:
        """Generate the story in one creative Qwen call, with no retry or challenger.

        Paragraph topology is repaired without changing prose. Word-count and source
        coverage misses are observable warnings; empty or unfinished output fails closed.
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
        # Do not set min_tokens: forcing continuation after a complete story has
        # previously caused repeated filler. One pass uses a 3,200-token completion
        # cap with thinking enabled; no retry, alternative candidate, critic, or editor
        # call is made on this path.
        story_min_output_tokens = 0
        story_budget = DIRECTOR_STORY_MAX_TOKENS
        print(
            f"[QWEN] story_thinking=on story_max_tokens={story_budget} temperature={temperature:.2f} top_p={top_p:.2f} top_k={DIRECTOR_STORY_THINKING_TOP_K} min_p={DIRECTOR_STORY_THINKING_MIN_P:.1f} presence_penalty={DIRECTOR_STORY_PRESENCE_PENALTY:.1f} single_pass=true",
            flush=True,
        )

        try:
            raw = self._chat_text(
                story_system,
                story_user,
                minimum_completion=600,
                minimum_output_tokens=story_min_output_tokens,
                temperature=temperature,
                top_p=top_p,
                call_name=call_name,
                max_completion=story_budget,
                disable_thinking=False,
                seed=DIRECTOR_VLLM_SEED,
                creative=True,
            )
        except StoryTruncated as exc:
            raw = self._salvage_runaway_story(exc.partial)
            if not raw:
                raise RuntimeError(failure_prefix + str(exc)) from exc

        raw = self._reflow_story_paragraphs(raw)
        self._last_story_draft = raw
        if not str(raw or "").strip():
            raise RuntimeError(failure_prefix + "Qwen returned an empty story.")

        # Six scene paragraphs are a production-topology requirement. Repair only
        # their boundaries deterministically; never ask the model to rewrite prose.
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n+", str(raw or ""))
            if part.strip()
        ]
        candidate = str(raw or "").strip()
        if len(paragraphs) != 6:
            candidate = self._coerce_story_to_six_paragraphs(candidate)
            repaired_count = len([
                part for part in re.split(r"\n\s*\n+", candidate) if part.strip()
            ])
            print(
                f"[DIRECTOR] story topology normalized without regeneration: {len(paragraphs)} -> {repaired_count} paragraphs",
                flush=True,
            )

        self._last_story_contract_warnings = []
        candidate = self._validate_story_output_contracts(
            mode,
            user_input,
            candidate,
            source_character_names=source_character_names,
            enforce_planned_cast=False,
            strict=False,
        )

        # A truncated/incomplete final sentence cannot safely become a story, but
        # ordinary word-count and source-coverage misses are warnings in this
        # no-retry profile rather than a trigger for another generation call.
        if not candidate or not self._ends_cleanly(candidate):
            raise RuntimeError(
                failure_prefix + "The single-pass draft is empty or ends mid-sentence. "
                "No retry was attempted; inspect the draft/trace before rerunning manually."
            )

        word_count = len(re.findall(r"\b[\w'’-]+\b", candidate))
        paragraph_count = len([
            part for part in re.split(r"\n\s*\n+", candidate) if part.strip()
        ])
        issues = self._story_quality_issues(candidate)
        issues.extend(self._story_craft_issues(candidate))
        issues.extend(
            "contract: " + str(item)
            for item in (getattr(self, "_last_story_contract_warnings", []) or [])
        )
        # 420-560 is preferred, not a hard rejection boundary. A complete 562-word
        # story is production-safe; only flag a meaningful departure from the safe band.
        if not 420 <= word_count <= 600:
            issues.append(f"word count {word_count} outside safe range 420-600 (preferred 420-560)")
        if paragraph_count != 6:
            issues.append(f"paragraph count {paragraph_count} outside target 6")
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

    def _trim_runaway_tail(self, raw: str) -> str:
        """Return the first six paragraphs when they are a complete in-range story and the
        model kept writing afterwards; otherwise return `raw` unchanged."""
        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n+", self._normalize_story(raw))
            if part.strip()
        ]
        if len(paragraphs) <= 6:
            return raw
        head = paragraphs[:6]
        words = len(re.findall(r"\b[\w'’-]+\b", " ".join(head)))
        if not 420 <= words <= 600:
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
        paragraphs = [
            re.sub(r"[ \t\n]+", " ", part).strip()
            for part in re.split(r"\n\s*\n+", value)
            if part.strip()
        ]
        if len(paragraphs) < 2 or len(paragraphs) == 6:
            return text
        if len(paragraphs) > 6 and self._trim_runaway_tail(text) != text:
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
    def _coerce_story_to_six_paragraphs(text: str) -> str:
        """Last-resort topology adapter; never changes story prose tokens."""
        value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not value:
            return ""

        paragraphs = [
            re.sub(r"[ \t\n]+", " ", part).strip()
            for part in re.split(r"\n\s*\n+", value)
            if part.strip()
        ]
        if not paragraphs or len(paragraphs) == 6:
            return "\n\n".join(paragraphs)

        original_count = len(paragraphs)

        def word_count(part: str) -> int:
            return len(re.findall(r"\b[\w'’-]+\b", part))

        def sentence_spans(part: str) -> list[str]:
            pieces = re.split(r"(?<=[.!?])\s+", part.strip())
            return [piece.strip() for piece in pieces if piece.strip()]

        # More than six paragraphs: merge the smallest adjacent interior pair.
        # Keep opening and aftermath paragraphs distinct whenever possible.
        while len(paragraphs) > 6:
            n = len(paragraphs)
            candidate_starts = list(range(1, n - 2)) or list(range(0, n - 1))
            start = min(
                candidate_starts,
                key=lambda i: (
                    word_count(paragraphs[i]) + word_count(paragraphs[i + 1]),
                    abs(word_count(paragraphs[i]) - word_count(paragraphs[i + 1])),
                    i,
                ),
            )
            paragraphs[start] = paragraphs[start] + " " + paragraphs[start + 1]
            del paragraphs[start + 1]

        # Fewer than six paragraphs: split only at an existing sentence boundary.
        # Prefer the longest splittable paragraph and a split that keeps both halves
        # reasonably close to the six-scene target without rewriting any prose.
        while len(paragraphs) < 6:
            best = None
            for index, paragraph in enumerate(paragraphs):
                sentences = sentence_spans(paragraph)
                if len(sentences) < 2:
                    continue
                total = word_count(paragraph)
                for split_index in range(1, len(sentences)):
                    left = " ".join(sentences[:split_index]).strip()
                    right = " ".join(sentences[split_index:]).strip()
                    left_words = word_count(left)
                    right_words = word_count(right)
                    if not left or not right:
                        continue
                    # Avoid creating a tiny final aftermath fragment.
                    final_penalty = 60 if index == len(paragraphs) - 1 and right_words < 25 else 0
                    target_penalty = abs(left_words - 80) + abs(right_words - 80)
                    score = (final_penalty + target_penalty, -total, index, split_index)
                    candidate = (score, index, split_index, left, right)
                    if best is None or score < best[0]:
                        best = candidate
            if best is None:
                break
            _, index, split_index, left, right = best
            paragraphs[index:index + 1] = [left, right]

        result = "\n\n".join(paragraphs)
        if len(paragraphs) != original_count:
            print(
                f"[DIRECTOR] story topology fallback: {original_count} -> {len(paragraphs)} paragraphs; prose preserved",
                flush=True,
            )
        return result


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

    def _story_style_findings(self, story: str) -> list[dict]:
        """Lightweight deterministic style warnings; these are heuristics, not a quality score."""
        text = self._normalize_story(story)
        findings: list[dict] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        hits = []
        for sentence in sentences:
            match = self._STOCK_PHRASES.search(sentence)
            if match:
                hits.append((match.group(0), sentence.strip()))
        # One familiar phrase may be deliberate; only repeated use is a finding.
        if len(hits) >= 2:
            phrases = "; ".join(dict.fromkeys(hit[0].lower() for hit in hits))[:200]
            findings.append({
                "code": "repetition_or_generic_beat",
                "severity": "major",
                "evidence": hits[0][1][:220],
                "problem": f"Repeated stock phrasing ({len(hits)} hits): {phrases}.",
                "repair": "Replace repeated generic phrasing with premise-specific action, evidence, or objects.",
            })
        narration = re.sub(r"[\"\u201c][^\"\u201d]*[\"\u201d]", " ", text.lower())
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", narration.replace("\u2019", "'"))
        stop = {"the", "a", "an", "and", "of", "to", "in", "on", "at", "his", "her", "was", "were", "it", "he", "she", "with", "as", "that", "from", "had"}
        grams: dict[str, int] = {}
        for i in range(len(words) - 2):
            tri = words[i:i + 3]
            if sum(1 for word in tri if word not in stop) >= 2:
                key = " ".join(tri)
                grams[key] = grams.get(key, 0) + 1
        repeated = [gram for gram, count in grams.items() if count >= 2]
        if repeated:
            gram = repeated[0]
            sentence = next((item.strip() for item in sentences if gram in item.lower()), "")
            findings.append({
                "code": "repetition_or_generic_beat",
                "severity": "major",
                "evidence": sentence[:220] or gram,
                "problem": "Repeated phrasing: " + "; ".join(repeated[:3]),
                "repair": "Vary or remove repeated phrasing so each beat has its own concrete image.",
            })
        return findings[:2]

    @staticmethod
    def _audit_score(findings: list[dict]) -> int:
        """Retained for diagnostics only; this score never chooses a story."""
        weights = {"blocking": 3, "major": 2, "minor": 1, "advisory": 0}
        return sum(weights.get(str(item.get("severity", "")).lower(), 0) for item in findings)

    def _audit_story(self, mode: str, story: str) -> list[dict]:
        """Return narrow deterministic style warnings; never treat them as narrative quality."""
        return self._story_style_findings(story)

    @staticmethod
    def _candidate_judgment_json_schema() -> dict:
        """Pairwise preference with evidence spans grounded in both candidate drafts."""
        return {
            "type": "object",
            "properties": {
                "winner": {"type": "string", "enum": ["A", "B", "tie"]},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                "reason": {"type": "string", "minLength": 1, "maxLength": 320},
                "evidence_a": {"type": "string", "minLength": 8, "maxLength": 180},
                "evidence_b": {"type": "string", "minLength": 8, "maxLength": 180},
            },
            "required": ["winner", "confidence", "reason", "evidence_a", "evidence_b"],
            "additionalProperties": False,
        }

    _STORY_CANDIDATE_JUDGE_SYSTEM = (
        "You are a strict pairwise short-film story evaluator. Compare two completed drafts of the same task; do not rewrite either and do not award independent numerical scores. "
        "Treat source facts as hard constraints. In Expand Story, the user-provided story is canon; in AI Story, preserve the premise's defining facts and constraints while allowing creative development. "
        "The Story Bible is provisional, so an absent planned character is not a defect unless the actual story establishes that the character's causal role is necessary. "
        "Reject candidates outside the six-paragraph/420-600 safe contract, those that drop a source-established fact, introduce a material contradiction, or leave the physical outcome of the central choice unresolved; prefer 420-560 words when quality is equal. "
        "Then compare, in priority order: causal coherence and event order; premise/source fidelity; protagonist agency and understandable opposition; setup/payoff and a visible price for the climax; concrete resolution; premise-specific originality; natural purposeful dialogue; prose style. "
        "Do not reward extra characters, dialogue quantity, stock genre beats, twists, length, or melodrama by themselves. Treat supplied critic findings as hypotheses and verify them against the actual text. "
        "Choose A, B, or tie. Give one exact, verbatim evidence span copied from each draft. The evidence should show a meaningful strength or defect relevant to your choice, not merely quote an arbitrary sentence. Never prefer the first or second draft because of its position. Return JSON only."
    )

    @staticmethod
    def _quote_is_grounded(quote: str, story: str) -> bool:
        quote_norm = re.sub(r"\s+", " ", str(quote or "")).strip().casefold()
        story_norm = re.sub(r"\s+", " ", str(story or "")).strip().casefold()
        return len(quote_norm) >= 12 and quote_norm in story_norm

    def _pairwise_story_judgment(
        self,
        mode: str,
        user_input: str,
        story_a: str,
        story_b: str,
        findings_a: list[dict],
        findings_b: list[dict],
        story_bible: dict | None,
        *,
        call_name: str,
    ) -> dict | None:
        """Return a grounded comparison, or None so selection safely keeps the incumbent."""
        try:
            payload = {
                "mode": str(mode),
                "source_premise_or_story": self._compact_story_context(user_input, max_chars=5000),
                "provisional_story_bible": self._format_story_bible(story_bible),
                "candidate_a": str(story_a or "").strip(),
                "candidate_a_review_findings": (findings_a or [])[:5],
                "candidate_b": str(story_b or "").strip(),
                "candidate_b_review_findings": (findings_b or [])[:5],
                "instruction": "Source facts and explicit output contracts outrank style. Compare the actual drafts and ground the reason in both.",
            }
            result = self._chat_json(
                self._STORY_CANDIDATE_JUDGE_SYSTEM,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                minimum_completion=100,
                temperature=0.0,
                top_p=0.9,
                call_name=call_name,
                max_completion=1000,
                json_mode=True,
                disable_thinking=False,
                response_schema=self._candidate_judgment_json_schema(),
            )
            if not isinstance(result, dict):
                return None
            winner = str(result.get("winner", "")).strip().lower()
            confidence = str(result.get("confidence", "")).strip().lower()
            evidence_a = re.sub(r"\s+", " ", str(result.get("evidence_a", ""))).strip()
            evidence_b = re.sub(r"\s+", " ", str(result.get("evidence_b", ""))).strip()
            if winner not in {"a", "b", "tie"} or confidence not in {"high", "medium", "low"}:
                return None
            if not self._quote_is_grounded(evidence_a, story_a) or not self._quote_is_grounded(evidence_b, story_b):
                print(f"[QWEN] {call_name} rejected (ungrounded evidence)", flush=True)
                return None
            return {
                "winner": winner.upper(),
                "confidence": confidence,
                "reason": str(result.get("reason", "")).strip()[:320],
                "evidence_a": evidence_a,
                "evidence_b": evidence_b,
            }
        except Exception as exc:
            print(f"[QWEN] {call_name} skipped ({type(exc).__name__}: {str(exc)[:140]}); preserving incumbent", flush=True)
            return None

    @staticmethod
    def _has_new_hard_candidate_failure(
        incumbent_findings: list[dict],
        challenger_findings: list[dict],
    ) -> bool:
        """Do not replace a draft with a candidate carrying a new source/blocking failure."""
        incumbent = incumbent_findings or []
        challenger = challenger_findings or []
        incumbent_source_failure = any(str(item.get("code", "")) == "source_fidelity" for item in incumbent)
        challenger_source_failure = any(str(item.get("code", "")) == "source_fidelity" for item in challenger)
        if challenger_source_failure and not incumbent_source_failure:
            return True
        incumbent_blocking = sum(1 for item in incumbent if str(item.get("severity", "")).lower() == "blocking")
        challenger_blocking = sum(1 for item in challenger if str(item.get("severity", "")).lower() == "blocking")
        return challenger_blocking > incumbent_blocking

    def _resample_for_audit(
        self,
        mode: str,
        user_input: str,
        story: str,
        story_system: str,
        story_user: str,
        *,
        temperature: float,
        top_p: float,
        source_character_names: list[str] | None,
        story_bible: dict | None = None,
        initial_findings: list[dict] | None = None,
    ) -> tuple[str, list[dict]]:
        """Create at most one alternative when needed; use evidence-grounded pairwise selection.

        Candidate A is the incumbent. A second candidate is generated only when the
        first draft has a material critic finding or deterministic style warning. It
        replaces A only when grounded pairwise comparison prefers B and a reversed-
        order confirmation agrees. On judge failure or disagreement, retain A.
        """
        try:
            attempts = max(1, min(2, int(os.getenv("H3_STORY_CANDIDATES", "1"))))
        except (TypeError, ValueError):
            attempts = 2
        incumbent = str(story or "").strip()
        incumbent_findings = list(initial_findings or [])
        style_findings = self._audit_story(mode, incumbent)
        trigger = bool(incumbent_findings or style_findings)
        if attempts < 2 or not trigger:
            print(
                f"[QWEN] story_candidate selection=incumbent attempts={attempts} "
                f"material_findings={len(incumbent_findings)} style_findings={len(style_findings)}",
                flush=True,
            )
            return incumbent, incumbent_findings

        self._story_seed_offset = 1
        try:
            challenger, _ = self._generate_story_once(
                mode, user_input, story_system, story_user,
                temperature=temperature, top_p=top_p,
                source_character_names=list(source_character_names or []),
            )
        except Exception as exc:
            print(f"[QWEN] story_candidate skipped ({type(exc).__name__}: {str(exc)[:140]})", flush=True)
            return incumbent, incumbent_findings
        finally:
            self._story_seed_offset = 0

        challenger = str(challenger or "").strip()
        if not challenger or challenger == incumbent:
            print("[QWEN] story_candidate skipped (empty or identical candidate)", flush=True)
            return incumbent, incumbent_findings

        challenger_findings = self._collect_story_findings(
            mode, user_input, challenger, story_bible, source_character_names
        )
        if self._has_new_hard_candidate_failure(incumbent_findings, challenger_findings):
            print("[QWEN] story_candidate rejected (new blocking/source-fidelity finding)", flush=True)
            return incumbent, incumbent_findings

        first = self._pairwise_story_judgment(
            mode, user_input, incumbent, challenger,
            incumbent_findings, challenger_findings, story_bible,
            call_name="story_candidate_pairwise",
        )
        if not first or first.get("winner") != "B" or first.get("confidence") == "low":
            print("[QWEN] story_candidate selection=incumbent (pairwise did not prefer challenger)", flush=True)
            return incumbent, incumbent_findings

        # Position-bias check: only accept the challenger if it also wins when the
        # two drafts swap positions. Disagreement is resolved conservatively in favor
        # of the already-valid incumbent rather than trusting a biased single vote.
        reversed_judgment = self._pairwise_story_judgment(
            mode, user_input, challenger, incumbent,
            challenger_findings, incumbent_findings, story_bible,
            call_name="story_candidate_pairwise_swapped",
        )
        if not reversed_judgment or reversed_judgment.get("winner") != "A" or reversed_judgment.get("confidence") == "low":
            print("[QWEN] story_candidate selection=incumbent (position-swap confirmation failed)", flush=True)
            return incumbent, incumbent_findings

        print(
            "[QWEN] story_candidate selection=challenger confirmed_by_position_swap "
            f"findings={len(incumbent_findings)}->{len(challenger_findings)}",
            flush=True,
        )
        return challenger, challenger_findings

    def _planned_cast_findings(self, mode: str, story: str) -> list[dict]:
        """Deterministic, zero-cost audit of the Story Bible cast against the draft.

        This is the "reliable external feedback" that prompted self-correction needs:
        it uses the production planner's own presence rules, so it cannot hallucinate.
        """
        if mode not in (AI_STORY_MODE, EXPAND_USER_STORY_MODE) or not self._planned_cast:
            return []
        findings: list[dict] = []
        try:
            planner = self._planner()
            text = self._normalize_story(story)
            for name in dict.fromkeys(str(v).strip() for v in self._planned_cast if str(v).strip()):
                presence = planner.identity_presence(text, name)
                if presence == "absent":
                    findings.append({
                        "code": "planned_cast_presence",
                        "severity": "advisory",
                        "evidence": "(name does not appear in the draft)",
                        "problem": f"Outline-only person {name} is not present in the draft; this is not a defect unless the person's causal role is essential.",
                        "repair": f"Keep {name} only if their independent action changes the protagonist's options; otherwise leave them out.",
                    })
                elif presence == "mention_only":
                    sentence = next(
                        (part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if name.casefold() in part.casefold()),
                        "",
                    )
                    findings.append({
                        "code": "planned_cast_presence",
                        "severity": "advisory",
                        "evidence": sentence[:220] or "(name only appears as a reference)",
                        "problem": f"Outline-only person {name} may be reference-only; check whether this absence weakens the plot before changing the draft.",
                        "repair": f"Bring {name} on screen only if their independent action changes the protagonist's options; otherwise remove the unnecessary reference.",
                    })
        except Exception as exc:
            print(f"[QWEN] planned_cast_audit skipped ({type(exc).__name__})", flush=True)
        return findings[:5]

    def _collect_story_findings(
        self,
        mode: str,
        user_input: str,
        story: str,
        story_bible: dict | None,
        source_character_names: list[str] | None,
    ) -> list[dict]:
        """Run a semantic story review by default; keep outline drift advisory, not a cast quota."""
        outline_notes = self._planned_cast_findings(mode, story)
        if outline_notes:
            print(
                "[QWEN] story_outline_drift advisory=" + str(len(outline_notes))
                + " codes=" + ",".join(item["code"] for item in outline_notes)
                + " (not automatically sent to editor)",
                flush=True,
            )
        critic = self._review_story_draft(mode, user_input, story, story_bible, source_character_names)
        if not critic and not outline_notes:
            print("[QWEN] story_audit no material evidence-backed defects", flush=True)
        return critic[:5]

    _STORY_REVIEW_SYSTEM = (
        "You are a skeptical short-film story auditor, not a co-writer or a style judge. Return JSON findings only; do not rewrite the story or award it a self-reported score. "
        "Audit the actual draft against the user's source and the provisional Story Bible, with source canon taking priority in Expand Story. The Story Bible is a hypothesis, not a compliance contract. "
        "A planned cast omission is NOT a defect by itself: report it only when the story text makes that person causally necessary and the omission leaves a material gap. Do not force a cast count or retain a weak outline. "
        "Report only material defects with an exact copied evidence span from the draft and a concrete minimal repair; never invent or paraphrase the evidence quote. "
        "Prioritize: (1) cause-and-effect gaps, contradictory facts or character motives; (2) a turn that does not arise from earlier evidence; (3) an object/skill with no established function or payoff; (4) a choice whose consequences are not shown; (5) a central conflict or protagonist goal left unresolved; (6) people/objects changing location or state without an event; (7) dialogue that only explains the premise; (8) generic plot machinery that could fit almost any premise; and (9) Expand Story source-fidelity violations. "
        "Check the ending against what the protagonist actually did: do not accept a lyrical last line as a substitute for showing whether the objective succeeded, what the choice cost, and what happened to the immediate danger. "
        "Do not demand a twist, a happy ending, extra mysteries, a standard genre structure, or a subjective prose preference. Preserve ambiguity only when the physical/causal outcome is still clear. "
        "Return no more than five high-impact findings. If there is no material, evidence-backed defect, return an empty findings array."
    )

    @staticmethod
    def _story_review_json_schema() -> dict:
        """Structured, evidence-grounded feedback; this does not define a cast quota."""
        finding = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "enum": [
                        "planned_cast_presence",
                        "causality_gap",
                        "unpaid_setup",
                        "stakes_or_choice",
                        "unresolved_conflict",
                        "object_or_location_continuity",
                        "dialogue_not_live_or_functional",
                        "source_fidelity",
                        "repetition_or_generic_beat",
                        "premise_grounding",
                    ],
                },
                "severity": {"type": "string", "enum": ["blocking", "major", "minor"]},
                "evidence": {"type": "string", "minLength": 8, "maxLength": 220},
                "problem": {"type": "string", "minLength": 1, "maxLength": 240},
                "repair": {"type": "string", "minLength": 1, "maxLength": 240},
            },
            "required": ["code", "severity", "evidence", "problem", "repair"],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "maxItems": 5,
                    "items": finding,
                }
            },
            "required": ["findings"],
            "additionalProperties": False,
        }

    def _review_story_draft(
        self,
        mode: str,
        user_input: str,
        story: str,
        story_bible: dict | None,
        source_character_names: list[str] | None,
    ) -> list[dict]:
        """Run one bounded Qwen3-thinking story review; return only evidence-backed major findings.

        Critic findings are guidance for the editor, not hard rejection decisions. A
        finding is accepted only if its quoted evidence occurs in the exact draft.
        This prevents unsupported generic critique from driving a rewrite.
        """
        if os.getenv("H3_STORY_CRITIC", "0").strip().lower() in {"0", "false", "no", "off"}:
            return []
        try:
            plan = self._format_story_bible(story_bible)
            payload = {
                "mode": str(mode),
                "source_premise_or_story": self._compact_story_context(user_input, max_chars=DIRECTOR_CRITIC_STORY_CONTEXT_CHARS),
                "source_character_anchors": list(dict.fromkeys(
                    str(value).strip()
                    for value in (source_character_names or [])
                    if str(value).strip()
                ))[:16],
                "story_bible": plan,
                "draft": str(story or "").strip(),
            }
            result = self._chat_json(
                self._STORY_REVIEW_SYSTEM,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                minimum_completion=250,
                temperature=0.25,
                top_p=0.90,
                call_name="story_quality_review",
                max_completion=1800,
                json_mode=True,
                disable_thinking=False,
                response_schema=self._story_review_json_schema(),
            )
            raw_findings = result.get("findings", []) if isinstance(result, dict) else []
            normalized_story = re.sub(r"\s+", " ", str(story or "")).strip().casefold()
            planned_names: list[str] = []
            for value in (self._planned_cast or []):
                full = str(value or "").strip()
                if not full:
                    continue
                planned_names.append(full.casefold())
                titleless = re.sub(
                    r"^(?:dr|doctor|mr|mrs|ms|miss|prof|professor|captain|commander|detective|agent)\.?\s+",
                    "", full, count=1, flags=re.IGNORECASE,
                ).strip()
                if titleless:
                    planned_names.append(titleless.casefold())
            accepted: list[dict] = []
            seen: set[tuple[str, str]] = set()
            if isinstance(raw_findings, list):
                for item in raw_findings:
                    if not isinstance(item, dict):
                        continue
                    code = str(item.get("code", "")).strip()
                    severity = str(item.get("severity", "")).strip().lower()
                    evidence = re.sub(r"\s+", " ", str(item.get("evidence", ""))).strip()
                    evidence_norm = evidence.casefold()
                    # Minor/style-only findings are deliberately not allowed to
                    # trigger substantive rewriting. The source span must be exact.
                    if severity not in {"blocking", "major"}:
                        continue
                    if len(evidence) < 18 or evidence_norm not in normalized_story:
                        continue
                    if code == "planned_cast_presence":
                        finding_context = (str(item.get("problem", "")) + " " + str(item.get("repair", ""))).casefold()
                        if not any(name in finding_context for name in planned_names):
                            continue
                    key = (code, evidence_norm)
                    if key in seen:
                        continue
                    seen.add(key)
                    accepted.append({
                        "code": code,
                        "severity": severity,
                        "evidence": evidence,
                        "problem": str(item.get("problem", "")).strip()[:240],
                        "repair": str(item.get("repair", "")).strip()[:240],
                    })
                    if len(accepted) >= 5:
                        break
            if accepted:
                print(
                    "[QWEN] story_quality_review findings="
                    + str(len(accepted))
                    + " codes="
                    + ",".join(item["code"] for item in accepted),
                    flush=True,
                )
            else:
                print("[QWEN] story_quality_review no evidence-backed major findings", flush=True)
            return accepted
        except Exception as exc:
            print(
                f"[QWEN] story_quality_review skipped ({type(exc).__name__}: {str(exc)[:180]}); preserving draft",
                flush=True,
            )
            return []

    _STORY_EDITOR_SYSTEM = (
        "You are the final short-film story editor. Improve the draft without replacing the user's premise or its strongest effective material. "
        "Return only the finished story: exactly six paragraphs, 420-600 words (420-560 preferred), plain prose. "
        "In Expand Story, preserve user-established characters, chronology, setting, events, outcome, and defining meaning. The Story Bible is provisional: do not preserve an invented plan detail or character if it makes the actual story less causal or less coherent. "
        "Do not add decorative people, generic revelations, or a subplot to satisfy a quota. Add or retain a consequential present-day counterpart only when their independent action changes the protagonist's options and the premise supports it. A single-character story is valid when that is the strongest source-grounded choice. "
        "Repair only high-impact defects: unclear goal/resistance; an event without a causal setup; a turn unrelated to earlier evidence; an object or skill with no concrete function/payoff; contradictory motives or physical states; dialogue that only explains; a choice without a visible cost; an unresolved central conflict; or an ending that obscures what the protagonist's action actually caused. "
        "Preserve one consistent mechanism/rule, track important objects and locations, give the protagonist specific interiority, keep dialogue natural and in-scene, and show the concrete consequence of the choice in the aftermath. Prefer details tied to this profession and setting; remove generic genre machinery and repeated phrasing. "
        "Do not change source canon in Expand Story. Do not force every outline beat or planned character when doing so would weaken the story. Maintain the word and paragraph contract, and output only the edited prose."
    )

    def _edit_story_draft(
        self,
        mode: str,
        user_input: str,
        story: str,
        story_bible: dict | None,
        source_character_names: list[str] | None,
        *,
        temperature: float,
        top_p: float,
        review_findings: list[dict] | None = None,
        call_name: str = "story_editor_pass",
    ) -> str:
        """One thinking-mode editing pass guided by the Story Bible and evidence-backed review."""
        editor_mode = os.getenv("H3_STORY_EDITOR", "0").strip().lower()
        if editor_mode in {"0", "false", "no", "off"}:
            return story
        if editor_mode not in {"1", "true", "yes", "on", "always"} and not review_findings:
            # auto (default): the editor costs a full thinking call, so it runs only
            # when the audit found a concrete defect to repair.
            print("[QWEN] story_editor skipped (no audit findings)", flush=True)
            return story
        from planner.qwen_director_runtime import DIRECTOR_VLLM_SEED

        def count(text: str) -> int:
            return len(re.findall(r"\b[\w'’-]+\b", text))

        try:
            plan = self._format_story_bible(story_bible)
            review_block = ""
            if review_findings:
                review_block = (
                    "\n\nEVIDENCE-BACKED REVIEW FINDINGS (repair these material issues first; preserve all effective material):\n"
                    + json.dumps(review_findings[:5], ensure_ascii=False, separators=(",", ":"))
                    + "\nEach evidence span was copied from the draft. Make the smallest change that actually resolves each finding. "
                    "Do not satisfy a finding by adding a disconnected subplot or forcing a character count."
                )
            user = (
                "MODE: " + str(mode)
                + "\n\nSOURCE PREMISE / STORY:\n" + self._compact_story_context(user_input, max_chars=5000)
                + ("\n\n" + plan if plan else "")
                + review_block
                + "\n\nDRAFT TO EDIT:\n" + story
            )
            edited = self._chat_text(
                self._STORY_EDITOR_SYSTEM,
                user,
                minimum_completion=600,
                temperature=min(float(temperature), 0.55),
                top_p=float(top_p),
                call_name=call_name,
                max_completion=3200,
                disable_thinking=False,
                seed=DIRECTOR_VLLM_SEED,
                creative=True,
            )
            edited = self._reflow_story_paragraphs(edited)
            self._story_min_words = 420
            candidate = self._validate_story_output_contracts(
                mode,
                user_input,
                edited,
                source_character_names=source_character_names,
                enforce_planned_cast=False,
            )
            # The Story Bible is advisory; do not veto a coherent edit simply
            # because it drops a nonessential plan character. Structural output
            # contracts and Expand source-canon checks remain authoritative.
            if len(re.split(r"\n\s*\n+", candidate.strip())) != 6:
                print("[QWEN] story_editor rejected (paragraph count); keeping draft", flush=True)
                return story
            print(f"[QWEN] story_editor accepted words {count(story)} -> {count(candidate)}", flush=True)
            return candidate
        except Exception as exc:
            print(f"[QWEN] story_editor skipped ({type(exc).__name__}: {str(exc)[:200]}); keeping draft", flush=True)
            return story

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
        cleaned = (
            re.sub(
                r"\s+",
                " ",
                str(value or "").strip().strip('"“”‘’'),
            )
            .replace("’", "'")
            .replace("‘", "'")
            .replace("—", "-")
            .replace("–", "-")
        )
        return cleaned if keep_case else cleaned.lower()

    @staticmethod
    def _dialogue_anchor_key(value: str) -> str:
        text = QwenDirector._normalize_dialogue_text(value)
        text = re.sub(r"^[,;:!?\.\-\s]+|[,;:!?\.\-\s]+$", "", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _story_has_explicit_dialogue_surface(story: str) -> bool:
        """Return whether the story contains an explicit surface that can carry spoken dialogue.

        This is intentionally syntax-only: it recognizes quotation-delimited text and explicit
        screenplay-style speaker labels, but it never infers dialogue from English vocabulary.
        """
        text = str(story or "")
        if not text.strip():
            return False
        quote_pattern = re.compile(
            r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)',
            flags=re.UNICODE,
        )
        if any(
            str(value or "").strip()
            for match in quote_pattern.finditer(text)
            for value in match.groups()
        ):
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
            EntityResolver.normalize(str(key)): EntityResolver.normalize(str(value))
            for key, value in (speaker_aliases or {}).items()
            if str(key or "").strip() and str(value or "").strip()
        }

        def canonicalize_speaker(surface: str) -> str:
            normalized = EntityResolver.normalize(surface)
            # The extraction model commonly returns a title-bearing surface
            # ("Dr. Renn") while the production roster deliberately stores the
            # canonical name ("Renn"). Resolve the title-free form BEFORE the
            # allowed-speaker gate, not later after the source line was discarded.
            titleless = EntityResolver.strip_honorific(normalized)
            resolved = alias_map.get(normalized) or alias_map.get(titleless)
            return EntityResolver.normalize(resolved or titleless or normalized)

        remote_context_re = re.compile(
            r"\b(?:radio|walkie-talkie|intercom|loudspeaker|speaker|phone|telephone|"
            r"voicemail|recording|transmission|communicator|through the static|over the line)\b",
            flags=re.IGNORECASE,
        )
        physical_presence_re = re.compile(
            r"\b(?:appeared|arrived|entered|stood|stepped|walked|emerged|joined|"
            r"came in|came closer|was standing|stood beside|stood at|in person|"
            r"gripped|grabbed|took her arm|took his arm)\b",
            flags=re.IGNORECASE,
        )

        def is_remote_quote(position: int) -> bool:
            paragraph_start = text.rfind("\n\n", 0, position) + 2
            context = text[paragraph_start:position]
            remote_positions = [match.start() for match in remote_context_re.finditer(context)]
            physical_positions = [match.start() for match in physical_presence_re.finditer(context)]
            return bool(remote_positions) and (
                not physical_positions or max(remote_positions) > max(physical_positions)
            )

        remote_excluded = 0
        semantic_items: list[dict] = []
        for item in (semantic_spoken_dialogue or []):
            if not isinstance(item, dict):
                continue
            spoken = str(item.get("text", "") or "").strip()
            speaker = str(item.get("speaker", "") or "").strip()
            key = cls._dialogue_anchor_key(spoken)
            if not key or not speaker:
                continue
            semantic_items.append({
                "key": key,
                "speaker": canonicalize_speaker(speaker),
            })

        allowed = {
            EntityResolver.normalize(value)
            for value in (allowed_speakers or [])
            if str(value or "").strip()
        }

        semantic_consumed: set[int] = set()

        def consume_semantic_speaker(key: str) -> set[str]:
            for index, item in enumerate(semantic_items):
                if index in semantic_consumed or item["key"] != key:
                    continue
                semantic_consumed.add(index)
                return {item["speaker"]}
            return set()

        quote_pattern = re.compile(
            r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)',
            flags=re.UNICODE,
        )
        quote_spans: list[tuple[int, int]] = []

        for match in quote_pattern.finditer(text):
            value = next((part for part in match.groups() if part), "")
            key = cls._dialogue_anchor_key(value)
            if not key:
                continue

            speakers = consume_semantic_speaker(key)
            # Without Qwen semantic dialogue, do not guess from generic English prose.
            if not speakers:
                continue

            if allowed:
                speakers = {speaker for speaker in speakers if speaker in allowed}
                if not speakers:
                    quote_spans.append((match.start(), match.end()))
                    continue

            display = cls._normalize_dialogue_text(value, keep_case=True)
            display = display.rstrip(",;: ").strip()
            quote_spans.append((match.start(), match.end()))
            if not display or len(re.findall(r"[A-Za-z]", display)) < 2:
                continue
            if is_remote_quote(match.start()):
                # The shot contract requires visible speakers. Do not bind a voice
                # from a radio/speaker/recording to a physical shot; live, in-person
                # lines in later paragraphs remain eligible.
                remote_excluded += 1
                continue

            segments.append({
                "text": key,
                "display": display,
                "source_speakers": speakers,
                "start": match.start(),
                "end": match.end(),
                "semantic": True,
            })

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
            semantic_speakers = consume_semantic_speaker(key)
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
            segments.append({
                "text": key,
                "display": display,
                "source_speakers": speaker_set,
                "start": match.start(),
                "end": match.end(),
                "semantic": bool(semantic_speakers),
            })

        segments.sort(key=lambda item: (item.get("start", 0), item.get("end", 0)))
        for segment in segments:
            segment.pop("start", None)
            segment.pop("end", None)
        if remote_excluded:
            print(
                f"[DIRECTOR] dialogue source filter excluded {remote_excluded} remote/off-screen utterance(s); "
                "scene-local dialogue keeps only visible speaking characters",
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
        aliases = EntityResolver.build_character_alias_map(characters)
        spoken_segments = self._extract_story_spoken_segments(
            story,
            getattr(self, "_semantic_spoken_dialogue", None),
            allowed_speakers=allowed_names,
            speaker_aliases=aliases,
        )
        segment_progress = [0 for _ in spoken_segments]

        last_display: list[str] = [""]
        dialogue_counts = {
            "input": 0,
            "kept": 0,
            "empty": 0,
            "no_source_anchor": 0,
            "unmatched_text": 0,
        }

        def _consume_source_dialogue(normalized_text: str):
            last_display[0] = ""
            if not normalized_text:
                return None

            def _match_key(value: str) -> str:
                return re.sub(r"[.,!?;:]+$", "", str(value or "").strip())

            candidate_key = _match_key(normalized_text)
            if not candidate_key:
                return None

            # Prefer the earliest source utterance whose remaining text can carry
            # this exact contiguous event. This gives each source occurrence a
            # finite budget and supports safe splitting of long utterances while
            # tolerating a terminal-punctuation difference from Qwen.
            for index, segment in enumerate(spoken_segments):
                source_text = str(segment.get("text", "") or "")
                progress = segment_progress[index]
                remaining = source_text[progress:]
                if not remaining:
                    continue

                remaining_key = _match_key(remaining)
                if candidate_key == remaining_key:
                    segment_progress[index] = min(
                        len(source_text),
                        progress + len(normalized_text),
                    )
                    display_text = str(segment.get("display", "") or "")
                    if len(display_text) == len(source_text):
                        last_display[0] = display_text[progress:segment_progress[index]].strip()
                    return set(segment.get("source_speakers", set()) or set())

                if remaining_key.startswith(candidate_key):
                    segment_progress[index] = min(
                        len(source_text),
                        progress + len(normalized_text),
                    )
                    display_text = str(segment.get("display", "") or "")
                    if len(display_text) == len(source_text):
                        last_display[0] = display_text[progress:segment_progress[index]].strip()
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

                matched_source_speakers = _consume_source_dialogue(normalized_text)
                if matched_source_speakers is None:
                    # Qwen sometimes emits a narrative sentence or repeats a source
                    # line in multiple shots. Neither is valid audio. Drop it before
                    # any timing/compilation stage can treat it as speech, and count
                    # the reason so dialogue loss is no longer silent.
                    dialogue_counts["unmatched_text"] += 1
                    continue

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
                if canonical is None:
                    # Explicit source dialogue is a protected semantic contract.
                    # Never silently delete a real spoken line because Qwen used
                    # an unresolved speaker surface. A later stage cannot recover
                    # an event that is discarded here, so fail closed with the
                    # exact shot/speaker context instead.
                    self._record_recovery(
                        "dialogue_speaker_unresolved",
                        f"shot={shot_id} speaker={speaker!r}",
                    )
                    raise RuntimeError(
                        "Explicit dialogue speaker could not be canonically resolved: "
                        f"shot={shot_id} speaker={speaker!r}"
                    )

                explicit_source_canonicals = {
                    resolved.lower()
                    for source_speaker in matched_source_speakers
                    if (resolved := _resolve(source_speaker)) is not None
                }
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

                if normalized_speaker not in bound:
                    # The speaker already resolved to a canonical identity. If Qwen
                    # omitted that identity from the shot binding, restore the existing
                    # canonical entity deterministically rather than deleting dialogue.
                    shot_characters = shot.get("characters")
                    if not isinstance(shot_characters, list):
                        shot_characters = list(shot_characters or [])
                        shot["characters"] = shot_characters
                    existing_norm = {
                        EntityResolver.normalize(str(value or ""))
                        for value in shot_characters
                        if str(value or "").strip()
                    }
                    if normalized_speaker not in existing_norm:
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
                        if normalized_speaker not in scene_norms:
                            scene_characters.append(canonical)

                repaired = dict(event)
                repaired["speaker"] = canonical
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
                if speaker not in bound:
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
        return critique

    def critique_plan(self, *, mode: str, user_input: str, plan: dict) -> dict | None:
        """Run one bounded cinematic critique when the effective feature flag is enabled.

        The critic itself does not mutate the canonical plan; the orchestrator may
        apply its whitelisted shot-field patches. H3_DIRECTOR_CRITIQUE overrides the
        legacy H3_DIRECTOR_CRITIC environment variable and runtime manifest setting.
        This method deliberately performs no story regeneration or retry.
        """
        if not director_critic_enabled():
            print(
                "[QWEN] director_critique skipped (effective director_critic setting is disabled)",
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

    OUTPUT RULES
    - overall_score is 1 (unusable) to 10 (no defects).
    - status is "review" if there is at least one finding, otherwise "pass".
    - Report at most 6 findings and at most 3 shot_patches. Merge repeats of the same defect
      (for example one bad roster name across many shots) into ONE finding.
    - Every finding names a shot_id or scene_id and states one actionable defect concisely; include only the short evidence quote needed to prove it.
    - Add a shot_patch only for a defect in checks 2, 3, or 5, rewriting only that field in the same
      style and length. Never change ids, characters, timing, or continuity. If no concrete defect
      exists, return empty arrays.
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

        dialogue = schema["properties"]["spoken_dialogue"]
        dialogue["maxItems"] = 128  # preserve existing capacity contract
        dialogue["items"]["properties"]["speaker"] = {
            "type": "string",
            "maxLength": 96,
        }
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


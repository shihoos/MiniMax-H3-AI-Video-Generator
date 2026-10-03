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

        generated_story = False
        source_character_names: list[str] = [
            str(character.get("name", "")).strip()
            for character in (base_plan.get("characters", []) or [])
            if isinstance(character, dict) and str(character.get("name", "")).strip()
        ]

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

            story_user = (
                self._story_text_user(
                    mode,
                    user_input,
                    source_character_names=source_character_names,
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
                    # Keep thinking enabled for the single creative story pass.
                    # This pass must solve multiple coupled constraints (goal, resistance,
                    # reversal, choice, consequence, cast, dialogue, and six-scene topology).
                    # 3200 tokens provide reasoning + final-story headroom; the existing
                    # single bounded no-think retry remains the only creative recovery path.
                    max_completion=3200,
                    disable_thinking=False,
                )
                story = self._validate_story_output_contracts(
                    mode,
                    user_input,
                    story,
                    source_character_names=source_character_names,
                )
                generated_story = True
            except RuntimeError as first_error:
                error_text = str(first_error)
                story_contract_repair = any(
                    marker in error_text
                    for marker in (
                        "exactly six paragraphs",
                        "420 to 560 words",
                    )
                )
                completion_repair = any(
                    marker in error_text
                    for marker in (
                        "unresolved future hook",
                        "underdeveloped final aftermath paragraph",
                        "does not end in a complete sentence",
                        "must contain at least one explicit quoted line",
                    )
                )
                expand_cast_repair = (
                    mode == EXPAND_USER_STORY_MODE
                    and "Expand Story introduced unanchored" in error_text
                )
                cast_contract_details = (
                    " For Expand Story, also remove or replace only unanchored character identities; "
                    "preserve source-grounded names and relationships exactly."
                    if expand_cast_repair
                    else ""
                )
                topology_only_repair = (
                    "exactly six paragraphs" in error_text
                    and "420 to 560 words" not in error_text
                )
                if topology_only_repair and story.strip():
                    # Paragraph topology is a deterministic formatting defect. Repair it
                    # without spending another Qwen call or altering any story prose.
                    fallback = self._coerce_story_to_six_paragraphs(story)
                    if fallback != str(story).strip():
                        try:
                            story = self._validate_story_output_contracts(
                                mode,
                                user_input,
                                fallback,
                                source_character_names=source_character_names,
                            )
                            generated_story = True
                        except RuntimeError as topology_error:
                            # Coercion can only repair paragraph topology. If validation
                            # exposes another genuine contract defect, let the existing
                            # bounded retry handle that defect below.
                            error_text = str(topology_error)
                            story_contract_repair = any(
                                marker in error_text
                                for marker in (
                                    "exactly six paragraphs",
                                    "420 to 560 words",
                                )
                            )
                            completion_repair = any(
                                marker in error_text
                                for marker in (
                                    "unresolved future hook",
                                    "underdeveloped final aftermath paragraph",
                                    "does not end in a complete sentence",
                                    "must contain at least one explicit quoted line",
                                )
                            )
                            expand_cast_repair = (
                                mode == EXPAND_USER_STORY_MODE
                                and "Expand Story introduced unanchored" in error_text
                            )
                            cast_contract_details = (
                                " For Expand Story, also remove or replace only unanchored character identities; "
                                "preserve source-grounded names and relationships exactly."
                                if expand_cast_repair
                                else ""
                            )

                if not generated_story:
                    if expand_cast_repair and story.strip():
                        # Expand Story can fail multiple contracts at once (for example,
                        # word count plus an invented relational character). When that
                        # happens, the cast-grounding defect must take precedence over the
                        # generic length repair so the single allowed retry fixes BOTH
                        # defects in one controlled edit rather than preserving the
                        # unanchored identity.
                        bad_cast_detail = error_text.partition(
                            "Expand Story introduced unanchored character(s):"
                        )[2].strip()
                        if not bad_cast_detail:
                            bad_cast_detail = error_text.partition(
                                "Expand Story introduced unanchored relational character(s):"
                            )[2].strip()
                        repair_user = (
                            "Perform a SURGICAL CAST + CONTRACT EDIT of the existing Expand Story. "
                            "Preserve the source-grounded plot, causal chain, setting, established characters, "
                            "relationships, reversal, choice, consequence, dialogue, paragraph structure, and ending. "
                            "The generated expansion contains an identity that is NOT grounded in the source: "
                            + bad_cast_detail
                            + ". Remove or replace ONLY that invented identity. If it is expressed as a relational noun "
                            "such as 'brother', 'sister', 'husband', 'wife', 'father', 'mother', 'friend', or similar, "
                            "do not turn that relation into a production character; rewrite the sentence generically "
                            "while preserving its story meaning. Do not introduce another person to replace it. "
                            "Use ONLY characters explicitly established by the SOURCE STORY. A relational noun in the "
                            "source is not permission to invent or name a person unless that person is explicitly established. "
                            "At the same time, fix the other failed contract: keep exactly six paragraphs and 420 to 560 words. "
                            "Add or remove only concrete physical action, visible reaction, sensory detail, or causal connective "
                            "tissue already supported by the source; do not add a new plot thread, object, mystery, reveal, "
                            "future hook, or character. Return the COMPLETE revised story only.\n\nSOURCE STORY:\n"
                            + str(user_input).strip()
                            + "\n\nCURRENT EXPANSION:\n"
                            + str(story).strip()
                        )
                        preserved_prefix = ""
                        minimum_completion = 520
                    elif story_contract_repair and story.strip():
                        # A word/paragraph failure is a format defect, not permission
                        # to rewrite the narrative. Keep the generated causal chain,
                        # identities, events, reversal, choice, dialogue, and ending
                        # intact, and ask Qwen for a surgical structural/length edit.
                        paragraphs = [
                            part.strip()
                            for part in re.split(r"\n\s*\n+", str(story).strip())
                            if part.strip()
                        ]
                        word_matches = re.findall(r"\b[\w'’-]+\b", str(story or ""))
                        current_word_count = len(word_matches)
                        paragraph_counts = [
                            len(re.findall(r"\b[\w'’-]+\b", paragraph))
                            for paragraph in paragraphs
                        ]
                        target_range = "450 to 520 words"
                        paragraph_guidance = "; ".join(
                            f"P{index}={count} words"
                            for index, count in enumerate(paragraph_counts, 1)
                        ) or "no usable paragraph boundaries"
                        if mode == AI_STORY_MODE:
                            contract_details = (
                                "Keep at least one short direct spoken line by an existing character. "
                            )
                        else:
                            contract_details = "Dialogue remains optional; never invent a speaker merely to pad length. "
                        repair_user = (
                            "Perform a SURGICAL EDIT of the existing story. Do not rewrite it from scratch and do not "
                            "change its plot. Preserve every existing character identity/name, relationship, setting, "
                            "event, causal connection, planted detail, reversal, choice, consequence, quoted dialogue, "
                            "and final outcome. "
                            f"The current story has {current_word_count} words across {len(paragraphs)} paragraphs. "
                            "Return the COMPLETE revised story, not an explanation. It must contain exactly six paragraphs "
                            f"separated by blank lines and land in {target_range}; target roughly 70-90 words per paragraph. "
                            "If the story is short, add only concrete physical action, visible reaction, sensory specificity, "
                            "or causal connective tissue to events that already exist. If it is long, remove only redundant "
                            "exposition or repetition. Do not add a new character, location, object, mystery, reveal, or "
                            "future hook. Do not move events to another paragraph unless necessary to restore exactly six "
                            "scene-sized beats. Keep the final paragraph at least 25 words and make it consequence/aftermath, "
                            "not new plot. The final sentence must remain a completed past-tense action in a settled place. "
                            + contract_details
                            + cast_contract_details
                            + "Current paragraph counts for guidance: "
                            + paragraph_guidance
                            + "\n\nEXISTING STORY:\n"
                            + str(story).strip()
                        )
                        preserved_prefix = ""
                        minimum_completion = 520
                    elif completion_repair and story.strip():
                        paragraphs = [
                            part.strip()
                            for part in re.split(r"\n\s*\n", str(story).strip())
                            if part.strip()
                        ]
                        if paragraphs:
                            final_paragraph = paragraphs[-1]
                            sentence_matches = list(
                                re.finditer(r"(?<=[.!?])\s+", final_paragraph)
                            )
                            if sentence_matches:
                                split_at = sentence_matches[-1].end()
                                preserved_final_body = final_paragraph[:split_at].strip()
                            else:
                                preserved_final_body = ""

                            if preserved_final_body:
                                preserved_prefix = "\n\n".join(paragraphs[:-1] + [preserved_final_body])
                                repair_user = (
                                    "Replace only the final sentence of the existing story. Keep every earlier sentence, "
                                    "character, relationship, event, and established fact unchanged. Write one concrete "
                                    "past-tense consequence that closes the conflict. Do not introduce anything new. "
                                    "Do not use would, will, could, might, should, or a future intention. "
                                    "End with what happened, not what the protagonist plans to do next."
                                    + (
                                        " Include one short direct spoken line by an existing character."
                                        if "must contain at least one explicit quoted line" in error_text
                                        else ""
                                    )
                                    + cast_contract_details
                                    + " Return only the replacement sentence.\n\nSTORY:\n"
                                    + str(story).strip()
                                )
                                minimum_completion = 24
                            else:
                                preserved_prefix = "\n\n".join(paragraphs[:-1])
                                repair_user = (
                                    "Replace only the final paragraph of the existing story. Preserve every earlier paragraph, "
                                    "character, relationship, event, and established fact. Close the conflict with a concrete "
                                    "past-tense consequence. Do not introduce anything new or future-oriented."
                                    + cast_contract_details
                                    + " Return only the replacement final paragraph.\n\nSTORY:\n" + str(story).strip()
                                )
                                minimum_completion = 60
                        else:
                            preserved_prefix = ""
                            repair_user = (
                                "Complete the existing story without changing its characters, relationships, or established events. "
                                "End with a concrete past-tense consequence, not a future intention. Return only the finished story."
                                + "\n\nSTORY:\n" + str(story).strip()
                            )
                            minimum_completion = 80
                    elif expand_cast_repair and story.strip():
                        preserved_prefix = ""
                        repair_user = (
                            "Perform a CAST-CORRECTION EDIT of the existing Expand Story. "
                            "Preserve the plot, causal chain, setting, source-grounded characters, relationships, "
                            "reversal, choice, consequence, dialogue, paragraph structure, and ending. "
                            "Remove or replace only character identities that were invented without support from the source story. "
                            "You may keep a relational character only when that relationship is explicitly grounded in the source. "
                            "Do not add any new person. Return the complete revised story only. "
                            "Keep exactly six paragraphs separated by blank lines and 420 to 560 words. "
                            "Do not introduce a new plot thread or future hook.\n\nSOURCE STORY:\n"
                            + str(user_input).strip()
                            + "\n\nCURRENT EXPANSION:\n"
                            + str(story).strip()
                        )
                        minimum_completion = 520
                    else:
                        if mode == AI_STORY_MODE:
                            retry_requirements = (
                                "Rewrite the story only as needed to satisfy the failed contract. Preserve every causally earned "
                                "character and do not add a person merely to satisfy validation. "
                            )
                        else:
                            retry_requirements = (
                                "Rewrite only as needed to complete the expansion. The SOURCE CHARACTER ANCHORS are a hard safety "
                                "reference, not a cast whitelist; preserve them and any additional character the story genuinely establishes. "
                                "Do not invent a person merely to satisfy validation. "
                            )
                        repair_user = (
                            story_user
                            + "\n\n"
                            + retry_requirements
                            + "Prioritize causal coherence and a finished resolution over extra detail. Return ONLY the finished story prose."
                        )
                        preserved_prefix = ""
                        minimum_completion = 350

                    try:
                        if story_contract_repair or completion_repair:
                            # Structural/completion repairs are constrained edits, not creative resampling.
                            retry_temperature = 0.15 if story_contract_repair else 0.20
                            retry_top_p = 0.68 if story_contract_repair else 0.70
                        else:
                            retry_temperature = max(0.50, min(0.68, temperature - 0.08))
                            retry_top_p = max(0.80, min(0.88, top_p - 0.04))
                        repaired = self._chat_text(
                            story_system,
                            repair_user,
                            minimum_completion=minimum_completion,
                            temperature=retry_temperature,
                            top_p=retry_top_p,
                            call_name=(
                                "ai_story_text_retry"
                                if mode == AI_STORY_MODE
                                else "expand_story_text_retry"
                            ),
                            max_completion=1800,
                            disable_thinking=True,
                        )
                        if completion_repair and preserved_prefix:
                            story = preserved_prefix + "\n\n" + str(repaired).strip()
                        else:
                            story = repaired
                        story = self._validate_story_output_contracts(
                            mode,
                            user_input,
                            story,
                            source_character_names=source_character_names,
                        )
                        generated_story = True
                    except RuntimeError as retry_error:
                        retry_error_text = str(retry_error)
                        if "exactly six paragraphs" in retry_error_text:
                            # Last-resort topology adapter only after Qwen has had the
                            # opportunity to repair the raw story. This adapter never
                            # rewrites prose; it only merges/splits at existing boundaries.
                            fallback = self._coerce_story_to_six_paragraphs(story)
                            if fallback != str(story).strip():
                                try:
                                    story = self._validate_story_output_contracts(
                                        mode,
                                        user_input,
                                        fallback,
                                        source_character_names=source_character_names,
                                    )
                                    generated_story = True
                                except RuntimeError as fallback_error:
                                    retry_error_text = str(fallback_error)
                                else:
                                    pass
                            if generated_story:
                                # Topology fallback succeeded and the final story is validated.
                                pass
                            else:
                                raise RuntimeError(
                                    (
                                        "AI Story generation failed validation after the controlled retry: "
                                        if mode == AI_STORY_MODE
                                        else "Expand Story generation failed validation after the controlled retry: "
                                    )
                                    + retry_error_text
                                ) from retry_error
                        else:
                            raise RuntimeError(
                                (
                                    "AI Story generation failed validation after the controlled retry: "
                                    if mode == AI_STORY_MODE
                                    else "Expand Story generation failed validation after the controlled retry: "
                                )
                                + retry_error_text
                            ) from retry_error



        # ----------------------------------------------------
        # PASS 1B: deterministic production foundation
        # ----------------------------------------------------
        #
        # The ProductionPlanner owns the canonical production roster and scene
        # topology. Qwen supplies the final narrative and creative shot direction;
        # it does not regenerate deterministic production identity here.
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
                canonical_characters.append(Character.from_dict(deepcopy(item)))
        else:
            # Named-only stories have a deterministic canonical roster already:
            # every identity is strongly grounded, no relational/descriptive
            # identity needs semantic adjudication, so avoid two redundant Qwen
            # calls. Ambiguous/relational/descriptive stories still use the
            # bounded semantic extraction + adjudication path.
            deterministic_descriptors = planner._canonicalize_character_descriptors(
                [
                    *planner.detect_character_descriptors(canonical_source_story),
                    *planner._explicit_source_character_names(canonical_source_story),
                ]
            )
            title_tokens = {
                "dr", "doctor", "prof", "professor", "mr", "mrs", "ms", "miss",
                "captain", "commander", "detective", "agent",
            }
            deterministic_descriptors = [
                name for name in deterministic_descriptors
                if str(name).strip().lower().rstrip(".") not in title_tokens
            ]
            relational_hints = planner._extract_relational_character_hints(
                canonical_source_story,
                deterministic_descriptors,
            )
            has_descriptive_identity = any(
                planner._descriptive_identity_is_grounded(
                    canonical_source_story,
                    name,
                )
                for name in deterministic_descriptors
            )
            named_only_deterministic = bool(deterministic_descriptors) and not relational_hints and not has_descriptive_identity and all(
                planner._high_confidence_deterministic_character(
                    canonical_source_story,
                    name,
                )
                for name in deterministic_descriptors
            )
            canonical_characters = planner.create_characters(
                canonical_source_story,
                qwen_character_extractor=None if named_only_deterministic else self.extract_character_entities,
                qwen_character_adjudicator=None if named_only_deterministic else self.adjudicate_character_entities,
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

    def _validate_story_output_contracts(
        self,
        mode: str,
        user_input: str,
        story: str,
        source_character_names: list[str] | None = None,
    ) -> str:
        """Validate the current story without rewriting its topology.

        Qwen receives the first opportunity to repair contract defects. The deterministic
        six-paragraph topology adapter is intentionally kept outside this validator as a
        last-resort fallback, so raw model output is authoritative before any topology
        normalization is applied.
        """
        working = self._normalize_story(story)
        errors: list[str] = []

        try:
            self._validate_mode_output(mode, user_input, working)
        except RuntimeError as exc:
            errors.append(str(exc))

        if mode == EXPAND_USER_STORY_MODE:
            try:
                self._validate_expand_story_cast(
                    user_input,
                    working,
                    source_character_names=source_character_names,
                )
            except RuntimeError as exc:
                errors.append(str(exc))

        if errors:
            raise RuntimeError(" | ".join(errors))
        return working

    def _validate_expand_story_cast(
        self,
        source_story: str,
        generated_story: str,
        source_character_names: list[str] | None = None,
    ) -> None:
        """Keep Expand Story identities grounded in the source narrative."""
        planner = self._planner()

        source_descriptors = planner._canonicalize_character_descriptors(
            [
                *planner.detect_character_descriptors(source_story),
                *planner._explicit_source_character_names(source_story),
            ]
        )
        allowed = {
            EntityResolver.normalize(str(name).strip())
            for name in (source_character_names or [])
            if str(name).strip()
        }
        allowed.update(
            EntityResolver.normalize(str(name).strip())
            for name in source_descriptors
            if str(name).strip()
        )

        # Source-grounded relations such as "Eli's father" are legitimate even
        # when the base production roster contains only "Eli". They must be
        # derived from the SOURCE, not accepted merely because Qwen invented them.
        source_relations = planner._extract_relational_character_hints(
            source_story,
            source_descriptors,
        )
        allowed.update(
            EntityResolver.normalize(str(item.get("name", "")).strip())
            for item in source_relations
            if item.get("strong") and str(item.get("name", "")).strip()
        )

        generated_descriptors = planner._canonicalize_character_descriptors(
            [
                *planner.detect_character_descriptors(generated_story),
                *planner._explicit_source_character_names(generated_story),
            ]
        )
        generated_named = {
            EntityResolver.normalize(str(name).strip())
            for name in generated_descriptors
            if str(name).strip()
            and planner._high_confidence_deterministic_character(
                generated_story,
                str(name).strip(),
            )
        }
        extra_named = sorted(name for name in generated_named if name not in allowed)
        if extra_named:
            raise RuntimeError(
                "Expand Story introduced unanchored character(s): "
                + ", ".join(extra_named)
            )

        generated_relations = planner._extract_relational_character_hints(
            generated_story,
            generated_descriptors,
        )
        extra_relations = sorted(
            {
                str(item.get("name", "")).strip()
                for item in generated_relations
                if item.get("strong")
                and str(item.get("name", "")).strip()
                and EntityResolver.normalize(str(item.get("name", "")).strip()) not in allowed
            }
        )
        if extra_relations:
            raise RuntimeError(
                "Expand Story introduced unanchored relational character(s): "
                + ", ".join(extra_relations)
            )

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
    def _normalize_dialogue_text(value: str) -> str:
        return (
            re.sub(
                r"\s+",
                " ",
                str(value or "").strip().strip('"“”‘’'),
            )
            .replace("’", "'")
            .replace("‘", "'")
            .replace("—", "-")
            .replace("–", "-")
            .lower()
        )

    _SPEECH_TAG_VERBS = (
        "said|says|asked|asks|replied|replies|whispered|whispers|shouted|shouts|"
        "called|calls|muttered|mutters|murmured|murmurs|warned|warns|snapped|snaps|"
        "answered|answers|cried|cries|yelled|yells|demanded|demands|breathed|added|"
        "adds|continued|insisted|insists|pleaded|pleads|growled|growls|hissed|hisses|"
        "gasped|gasps|stammered|stammers|announced|announces|ordered|orders"
    )
    _SPEECH_TAG_NAME = r"([A-Z][A-Za-z0-9'\u2019_-]*(?:\s+[A-Z][A-Za-z0-9'\u2019_-]*){0,2})"

    @classmethod
    def _speech_tag_speakers(cls, before: str, after: str) -> set[str]:
        """Return the speaker named by an adjacent speech tag, or an empty set.

        Recognises only unambiguous forms:
            "...," Mara said.   "..." said Mara.   Mara said, "..."
        Pronouns, articles and lowercase subjects ("the terminal said") never
        produce a speaker, so machine voices and unresolved tags stay empty.
        """
        verbs = cls._SPEECH_TAG_VERBS
        name = cls._SPEECH_TAG_NAME
        blocked = set(EntityResolver.PRONOUNS) | {"the", "a", "an", "then", "and", "but"}
        found: list[str] = []
        patterns = (
            (after, rf"^\s*[,;]?\s*(?:\u2014|-)?\s*{name}\s+(?:{verbs})\b"),
            (after, rf"^\s*[,;]?\s*(?:{verbs})\s+{name}\b"),
            (before, rf"{name}\s+(?:{verbs})(?:\s+[a-z]+ly)?\s*[,:]?\s*$"),
        )
        for window, pattern in patterns:
            match = re.search(pattern, window)
            if not match:
                continue
            candidate = match.group(1).strip()
            first = candidate.split()[0].lower()
            if first in blocked:
                continue
            found.append(EntityResolver.normalize(candidate))
            break
        return set(found)

    @classmethod
    def _extract_story_spoken_segments(cls, story: str) -> list[dict]:
        """Extract ordered source-speech segments with exact occurrence boundaries.

        Each quoted/scripted utterance is a finite source-text budget. The Director may
        split one utterance across adjacent shots, but it must not duplicate an utterance
        or turn narrative prose into speech.
        """
        text = str(story or "")
        segments: list[dict] = []
        quote_spans: list[tuple[int, int]] = []

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
            normalized = cls._normalize_dialogue_text(value)
            if not normalized:
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

            segments.append({
                "text": normalized,
                "source_speakers": set(),
                "tag_speakers": cls._speech_tag_speakers(
                    text[max(0, match.start() - 90):match.start()],
                    text[match.end():match.end() + 90],
                ),
                "start": match.start(),
                "end": match.end(),
            })
            quote_spans.append((match.start(), match.end()))

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

            speaker = match.group(1).strip()
            spoken = match.group(2).strip()
            normalized = cls._normalize_dialogue_text(spoken)
            if not speaker or not normalized:
                continue

            segments.append({
                "text": normalized,
                "source_speakers": {EntityResolver.normalize(speaker)},
                "start": match.start(),
                "end": match.end(),
            })

        segments.sort(key=lambda item: (item["start"], item["end"]))
        for segment in segments:
            segment.pop("start", None)
            segment.pop("end", None)
        return segments

    @classmethod
    def _extract_story_spoken_texts(cls, story: str) -> dict[str, set[str]]:
        """Return source dialogue anchors for compatibility with existing callers."""
        anchors: dict[str, set[str]] = {}
        for segment in cls._extract_story_spoken_segments(story):
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
        spoken_segments = self._extract_story_spoken_segments(story)
        segment_progress = [0 for _ in spoken_segments]

        last_tag_speakers: set[str] = set()

        def _consume_source_dialogue(normalized_text: str):
            last_tag_speakers.clear()
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
                    last_tag_speakers.update(segment.get("tag_speakers", set()) or set())
                    return set(segment.get("source_speakers", set()) or set())

                if remaining_key.startswith(candidate_key):
                    segment_progress[index] = min(
                        len(source_text),
                        progress + len(normalized_text),
                    )
                    last_tag_speakers.update(segment.get("tag_speakers", set()) or set())
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
                if not spoken_segments:
                    continue

                matched_source_speakers = _consume_source_dialogue(normalized_text)
                if matched_source_speakers is None:
                    # Qwen sometimes emits a narrative sentence or repeats a source
                    # line in multiple shots. Neither is valid audio. Drop it before
                    # any timing/compilation stage can treat it as speech.
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

                if not explicit_source_canonicals and last_tag_speakers:
                    # The prose names who spoke ("..." Mara said). That tag is
                    # stronger than the shot model's guess, so correct the
                    # speaker instead of trusting or failing on it.
                    tagged = {
                        resolved.lower()
                        for tag in last_tag_speakers
                        if (resolved := _resolve(tag)) is not None
                    }
                    if len(tagged) == 1 and canonical.lower() not in tagged:
                        remapped = canonical_by_norm[next(iter(tagged))]
                        self._record_recovery(
                            "dialogue_speaker_tag_remap",
                            f"shot={shot_id} from={canonical!r} to={remapped!r}",
                        )
                        canonical = remapped

                normalized_speaker = canonical.lower()
                if explicit_source_canonicals and normalized_speaker not in explicit_source_canonicals:
                    # The source gives explicit speaker provenance that conflicts
                    # with Qwen's attribution. Never silently delete or remap the
                    # line; fail closed so the semantic conflict is visible.
                    self._record_recovery(
                        "dialogue_speaker_source_mismatch",
                        f"shot={shot_id} speaker={speaker!r} source={sorted(explicit_source_canonicals)!r}",
                    )
                    raise RuntimeError(
                        "Explicit dialogue speaker conflicts with source attribution: "
                        f"shot={shot_id} speaker={speaker!r} source={sorted(explicit_source_canonicals)!r}"
                    )

                if bound and normalized_speaker not in bound:
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
    You audit a film production plan against its story. Check each item and report only defects you
    can point to.

    CHECKS
    1. ROSTER: enforce the roster in both directions. Every name in a scene or shot `characters` list
       must appear in `roster`, and every roster entry must correspond to an actual active person/character
       established by the story (not merely a remembered, missing, recorded, document-only, machine, or object entity).
    2. FIDELITY: each shot `action` and `visual_prompt` must depict events that happen in the story
       for that scene. Report invented events, objects, or people.
    3. MATCH: `visual_prompt` must describe the same moment as `action`. Report contradictions.
    4. CONTINUITY: lighting, location, and who is present must not change between consecutive shots
       of one scene without a story reason.
    5. FRAMING: shots of one scene must not repeat the same camera_shot and camera_movement.

    OUTPUT RULES
    - overall_score is 1 (unusable) to 10 (no defects).
    - status is "review" if there is at least one finding, otherwise "pass".
    - Every finding names a shot_id or scene_id and says what is wrong in under 25 words.
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

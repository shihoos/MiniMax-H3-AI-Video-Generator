from __future__ import annotations

import ctypes
import faulthandler
import sys
import gc
import hashlib
import json
import os
import re
import textwrap
import time
from functools import wraps
from copy import deepcopy
from pathlib import Path

from planner.cinematic_compiler import CinematicCompiler
from planner.entity_resolver import EntityResolver
from pipeline.production_checkpoint import ProductionCheckpoint

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
    DIRECTOR_THREADS_BATCH,
    DIRECTOR_TOP_P,
    DIRECTOR_SHOT_STORY_CONTEXT_CHARS,
    DIRECTOR_SHOT_SCENE_DESCRIPTION_CHARS,
    DIRECTOR_SHOT_SCENE_OBJECTIVE_CHARS,
    DIRECTOR_SHOT_SCENE_CONTINUITY_CHARS,
    DIRECTOR_SHOT_SCENE_ATMOSPHERE_CHARS,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    director_enabled,
)


SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT = '\nYou are the CINEMATOGRAPHY DIRECTOR for MiniMax H3.\n\nCreate exactly __SHOTS_PER_SCENE__ production-ready shots for EACH supplied scene.\n\nFor the `location` field, use ONLY the physical setting where the shot occurs.\nThe value must be a concrete place or environment, not an action, object, body part, emotion, event, clause, sentence fragment, or abstract phrase.\nWhen the shot remains in the same physical setting, preserve the supplied scene location exactly.\nWhen the supplied scene location is empty, infer the physical setting from the supplied story context, scene description, continuity notes, and environment details. Do not treat an arbitrary prepositional phrase as a location.\nOnly change `location` when the narrative explicitly moves to a different physical place.\n\nThe scenes are part of one coherent film. Use ONLY the supplied characters. Do not create new characters or invent character names.\nKeep action 10–30 words, visual_prompt 15–40 words, composition_notes <=18 words, lighting <=12 words, lens_and_depth_of_field <=10 words, mood <=5 words, camera_shot <=5 words, camera_movement <=5 words.\nlocation must be a specific physical place (e.g., "abandoned station platform", "stairwell", "underground chamber"), never a phrase from the story.\nPreserve:\n- character identity;\n- chronology;\n- visual continuity;\n- location continuity;\n- emotional progression;\n- visual-language consistency.\n- exact dialogue text; never paraphrase or summarize supplied dialogue.\n- stable speaker names from the supplied character roster.\n- if dialogue is present, represent each line in dialogue_events; do not put timestamps in the response.\n- describe the shot\'s required initial and ending continuity states in continuity_start_state and continuity_end_state.\n\nWithin each scene, the required shots must use meaningfully different\nframing/composition while describing the SAME narrative beat.\n\nSCENE-FUNCTION DIRECTING:\nEach supplied scene includes scene_function and obligatory_moment.\nUse them as directing constraints, not as new story events.\nsetup: establish geography and protagonist context.\ncatalyst: reveal the disruptive event, clue, or discovery.\ndevelopment: show objective, movement, complication, or escalation.\nmidpoint: emphasize new information and changed understanding.\nclimax: emphasize danger, decisive action, choice, and consequence.\nfinale: emphasize aftermath, resolution, and the closing emotional image.\nEvery required shots must visibly serve the supplied obligatory_moment.\n\nSHOT / FRAMING VOCABULARY:\nframing: extreme wide, wide, full, medium wide, medium, medium close-up, close-up, extreme close-up, over-the-shoulder, two-shot, POV, insert.\n\nCAMERA MOVEMENT VOCABULARY:\nmovement: static, pan, tilt, dolly, tracking, handheld, crane, push-in, orbit.\n\nLENS / DEPTH OF FIELD:\nwide-angle, normal, telephoto, shallow focus, deep focus, selective focus.\n\nCOMPOSITION VOCABULARY:\ncentered, rule of thirds, leading lines, foreground frame, negative space, silhouette, depth layering, subject isolation.\n\nLIGHTING VOCABULARY:\nlighting: warm tungsten, cool daylight, golden-hour, blue-hour, moonlight, practical neon, hard chiaroscuro, soft overcast, mixed practical/ambient.\n\nReturn JSON only in exactly this structure:\n\n{\n  "scene_shots": [\n    {\n      "scene_id": "scene_001",\n      "shots": [\n        {\n          "shot_id": "scene_001_shot_001",\n          "scene_id": "scene_001",\n          "duration_seconds": 5.2,\n          "characters": [],\n          "location": "<physical setting only>",\n          "action": "...",\n          "camera_shot": "...",\n          "camera_movement": "...",\n          "lens_and_depth_of_field": "...",\n          "composition_notes": "...",\n          "lighting": "...",\n          "color_temperature": "...",\n          "mood": "...",\n          "visual_prompt": "...",\n          "speaking_characters": [],\n          "speech_text": "",\n          "dialogue_events": [],\n          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "is_scene_boundary": false,\n          "character_spatial_bboxes": {},\n          "character_spatial_regions": {},\n          "character_spatial_bboxes_start": {},\n          "character_spatial_bboxes_end": {},\n          "character_spatial_regions_start": {},\n          "character_spatial_regions_end": {}\n        },\n        {\n          "shot_id": "scene_001_shot_002",\n          "scene_id": "scene_001",\n          "duration_seconds": 5.2,\n          "characters": [],\n          "location": "<physical setting only>",\n          "action": "...",\n          "camera_shot": "...",\n          "camera_movement": "...",\n          "lens_and_depth_of_field": "...",\n          "composition_notes": "...",\n          "lighting": "...",\n          "color_temperature": "...",\n          "mood": "...",\n          "visual_prompt": "...",\n          "speaking_characters": [],\n          "speech_text": "",\n          "dialogue_events": [],\n          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "is_scene_boundary": false,\n          "character_spatial_bboxes": {},\n          "character_spatial_regions": {},\n          "character_spatial_bboxes_start": {},\n          "character_spatial_bboxes_end": {},\n          "character_spatial_regions_start": {},\n          "character_spatial_regions_end": {}\n        }\n      ]\n    }\n  ]\n}\n\nThere must be exactly __SHOTS_PER_SCENE__ shots inside every scene_shots entry and\nexactly one entry for every supplied scene. Do not add prose outside JSON.\n\nDo NOT output compiler-owned fields.\nDo NOT add scenes.\nDo NOT omit scenes.\nReturn JSON only.\n'

class QwenDirectorPromptMixin:
    def _mode_instruction(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:

            return textwrap.dedent("""
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
    """).strip()

        if mode == EXPAND_USER_STORY_MODE:

            return textwrap.dedent("""
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
    """).strip()

        if mode == PRESERVE_USER_STORY_MODE:

            return textwrap.dedent("""
    PRESERVE STORY MODE.

    The supplied story text is immutable.

    Do not rewrite it.

    Do not add narrative events.

    Use the supplied story as-is and create only the production
    structure needed to visualize it.
    """).strip()

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    def _story_text_system(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:
            return textwrap.dedent("""
    You are the narrative writer for MiniMax H3.

    The user provides a premise.

    Write a complete cinematic short-film story with a clear beginning,
    escalating middle, irreversible choice or point of no return, climax,
    consequence, and explicit resolution. Completion is more important than
    reaching a target word count.

    Hard requirements:
    1. SUBVERT THE OBVIOUS. Introduce one unexpected reveal or reversal that
       is caused by a concrete detail established earlier in the story. Do not
       rely on a familiar default twist or introduce a random secret, artifact,
       monster, or organization only for surprise.
    2. INTERIORITY. Include at least two sentences that reveal the protagonist's
       specific fear, memory, desire, or private realization through concrete
       imagery or sensory association. Show why the moment matters to them.
    3. DIALOGUE. Include at least one short line of spoken dialogue by a named
       character. The line must change a decision, reveal information, create
       conflict, or foreshadow the central reversal. No filler dialogue.
    4. RESOLUTION. End with a complete aftermath paragraph showing what happened
       to the protagonist and what changed. Do not stop mid-action, mid-sentence,
       mid-word, or on an ellipsis.

    Additional constraints:
    - Aim for 400-650 words, but always finish the story completely.
    - Third person past tense.
    - One protagonist whose goal is stated in the first two sentences.
    - No camera directions, scene headings, shot descriptions, labels, or meta commentary.

    Output ONLY the story prose.
    Do not output JSON, JSON objects, labels, analysis, metadata, or explanations.
    """).strip()

        if mode == EXPAND_USER_STORY_MODE:
            return textwrap.dedent("""
    You are the narrative expansion writer for MiniMax H3.

    Expand the supplied story substantially while preserving its important characters,
    events, chronology, setting, outcome, and explicit constraints. Add meaningful
    new narrative material, cause-and-effect development, emotional depth, escalation,
    and consequences rather than merely rephrasing or lightly lengthening the source.
    Build a complete arc with escalation, point of no return, climax, consequence,
    and resolution.

    Hard requirements for the expansion:
    1. Preserve an existing twist if one exists. If none exists, introduce one
       unexpected reveal or reversal that is caused by a concrete detail already
       established in the source; do not add a random secret, artifact, monster,
       or organization solely for surprise.
    2. Add at least two sentences of meaningful interiority for the protagonist,
       tied to a specific fear, memory, desire, or private realization.
    3. Preserve or add at least one short line of spoken dialogue by a named
       character. The line must advance conflict, reveal information, or connect
       to the central reversal.
    4. End with a complete resolution paragraph describing the aftermath and what
       changed. Do not stop mid-action, mid-sentence, mid-word, or on an ellipsis.

    Additional constraints:
    - Aim for 400-650 words, but always finish the story completely.
    - Preserve source meaning and chronology; Do not replace the original plot.
    - Do not merely add adjectives.
    - Do not convert the story into camera directions, scene headings, or shot descriptions.

    Output ONLY the expanded story prose.
    Do not output JSON, JSON objects, labels, analysis, metadata, or explanations.
    """).strip()

        raise ValueError(
            "Preserve Story does not use a story-text pass."
        )

    def _story_text_user(
        self,
        mode: str,
        story: str,
    ) -> str:

        source_text = self._limit_text(
            story,
            7000,
        )
        return (
            "MODE: "
            + str(mode)
            + "\n\nSOURCE STORY / PREMISE:\n"
            + source_text
            + "\n\nFINAL OUTPUT REQUIREMENTS:\n"
            + "Return only the completed story. Prioritize finishing the full narrative, including the resolution, over adding extra detail. "
            + "End on a complete sentence with terminal punctuation. Include the required causal reversal, protagonist interiority, and functional dialogue."
        )

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

    @staticmethod
    def _metadata_json_schema() -> dict:
        scene_properties = {
            "scene_id": {"type": "string"},
            "title": {"type": "string"},
            "order": {"type": "integer"},
            "location": {"type": "string"},
            "time_of_day": {"type": "string"},
            "weather": {"type": "string"},
            "atmosphere": {"type": "string"},
            "description": {"type": "string"},
            "mood": {"type": "string"},
            "lighting": {"type": "string"},
            "color_temperature": {"type": "string"},
            "environment_details": {
                "type": "array",
                "minItems": 3,
                "maxItems": 6,
                "items": {"type": "string"},
            },
            "key_props": {
                "type": "array",
                "items": {"type": "string"},
            },
            "characters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "scene_objective": {"type": "string"},
            "continuity_notes": {"type": "string"},
        }

        character_properties = {
            "character_id": {"type": "string"},
            "name": {"type": "string"},
            "role": {"type": "string"},
            "description": {"type": "string"},
            "personality": {"type": "string"},
            "appearance": {"type": "object"},
            "clothing": {"type": "object"},
            "distinctive_features": {
                "type": "array",
                "items": {"type": "string"},
            },
            "character_state": {"type": "object"},
            "continuity_rules": {
                "type": "array",
                "items": {"type": "string"},
            },
        }

        return {
            "type": "object",
            "properties": {
                "director_notes": {"type": "string"},
                "visual_language": {
                    "type": "object",
                    "properties": {
                        "genre_tone": {"type": "string"},
                        "color_palette": {"type": "string"},
                        "lighting_philosophy": {"type": "string"},
                        "camera_philosophy": {"type": "string"},
                        "pacing": {"type": "string"},
                    },
                    "required": [
                        "genre_tone",
                        "color_palette",
                        "lighting_philosophy",
                        "camera_philosophy",
                        "pacing",
                    ],
                    "additionalProperties": False,
                },
                "characters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": character_properties,
                        "required": [
                            "name",
                            "role",
                        ],
                        "additionalProperties": False,
                    },
                },
                "scenes": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": scene_properties,
                        "required": [
                            "scene_id",
                            "title",
                            "order",
                            "location",
                            "time_of_day",
                            "weather",
                            "atmosphere",
                            "description",
                            "mood",
                            "lighting",
                            "color_temperature",
                            "environment_details",
                            "key_props",
                            "characters",
                            "scene_objective",
                            "continuity_notes",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "director_notes",
                "visual_language",
                "characters",
                "scenes",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _character_extraction_json_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "entity_type": {
                                "type": "string",
                                "enum": [
                                    "PERSON",
                                    "CHARACTER",
                                    "SENTIENT",
                                    "ORGANIZATION",
                                    "LOCATION",
                                    "FACILITY",
                                    "OBJECT",
                                    "ROLE",
                                    "EVENT",
                                    "OTHER",
                                ],
                            },
                            "is_character": {"type": "boolean"},
                            "aliases": {
                                "type": "array",
                                "maxItems": 6,
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "name",
                            "entity_type",
                            "is_character",
                            "aliases",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["candidates"],
            "additionalProperties": False,
        }

    def extract_character_entities(
        self,
        story: str,
        deterministic_candidates: list[str] | None = None,
    ) -> dict:
        """Use the loaded Qwen model as the semantic character authority.

        The planner performs only bounded production-safety validation and
        canonicalization after this call. No lower layer may call Qwen for
        character identity or alias semantics.
        """
        story = str(story or "").strip()
        if not story:
            return {"candidates": []}

        self._character_semantic_calls += 1
        if self._character_semantic_calls > 2:
            raise RuntimeError("Character semantic Qwen call budget exceeded (max 2).")

        candidate_hints = [
            str(value).strip()
            for value in (deterministic_candidates or [])
            if str(value).strip()
        ]

        system_prompt = textwrap.dedent("""
    You are a strict character/entity extraction component for a cinematic production planner.
    Return JSON only. Analyze the supplied story and classify named entities relevant to character identity.
    A character is a named human or named sentient fictional entity that can receive a stable identity lock.
    Do NOT invent names. Every returned name must literally occur in the story text, allowing only harmless
    case/punctuation/possessive normalization. Do NOT treat locations, organizations, facilities, projects,
    missions, events, objects, calendar words, weather, or unnamed role descriptors as characters.
    Titles such as Dr., Captain, Commander, etc. are not part of the canonical name. Preserve full names when
    present and use the most complete canonical name actually present in the story. When a character's full
    name and a shorter form (such as first name only or a title fragment) both appear as candidates, return
    only the single most complete canonical form; do not return the short form as a separate candidate. Review
    the deterministic candidate hints and explicitly mark any that are not characters, while also recovering
    named characters that the deterministic scan missed.
    """).strip()

        user_payload = json.dumps(
            {
                "story": self._limit_text(story, 7000),
                "deterministic_candidates": candidate_hints[:32],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        result = self._chat_json(
            system_prompt,
            user_payload,
            minimum_completion=96,
            temperature=0.05,
            top_p=0.70,
            call_name="character_entity_extraction",
            max_completion=384,
            json_mode=True,
            disable_thinking=True,
            response_schema=self._character_extraction_json_schema(),
        )
    
        if os.getenv("H3_DEBUG_CHARACTERS", "0").strip().lower() in {"1", "true", "yes", "on"}:
            print("\n[CHARACTER QWEN RAW RESULT]")
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    
        return result

    def adjudicate_character_entities(
        self,
        story: str,
        deterministic_candidates: list[str] | None,
        semantic_result,
    ) -> dict:
        """Run the single bounded semantic adjudication pass when extraction disagrees with safety evidence."""
        self._character_semantic_calls += 1
        if self._character_semantic_calls > 2:
            raise RuntimeError("Character semantic Qwen call budget exceeded (max 2).")

        candidates = [
            str(value).strip()
            for value in (deterministic_candidates or [])
            if str(value).strip()
        ][:12]
        supplied = semantic_result if isinstance(semantic_result, dict) else {}
        system_prompt = textwrap.dedent("""
    You are the final character-identity adjudicator. Return JSON only.
    Review only the supplied candidate names and the supplied semantic extraction.
    For each candidate decide whether it is a stable named character identity: CHARACTER,
    NOT_CHARACTER, or UNCERTAIN. Never invent or rename a candidate. A name is valid only
    when its name or an explicitly supplied alias is anchored in the story. UNCERTAIN is
    terminal and must not trigger another call.
    """).strip()
        payload = json.dumps(
            {
                "story": self._limit_text(story, 7000),
                "deterministic_candidates": candidates,
                "semantic_result": supplied,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return self._chat_json(
            system_prompt,
            payload,
            minimum_completion=96,
            temperature=0.05,
            top_p=0.70,
            call_name="character_entity_adjudication",
            max_completion=256,
            json_mode=True,
            disable_thinking=True,
            response_schema=self._character_extraction_json_schema(),
        )

    @staticmethod
    def _scene_json_schema() -> dict:

        metadata = QwenDirectorPromptMixin._metadata_json_schema()
        return {
            "type": "object",
            "properties": {
                "scenes": metadata["properties"]["scenes"],
            },
            "required": ["scenes"],
            "additionalProperties": False,
        }

    @staticmethod
    def _scene_compression_json_schema() -> dict:
        return QwenDirectorPromptMixin._scene_json_schema()

    @staticmethod
    def _shot_json_schema(
        min_items: int = 2,
        max_items: int = 2,
    ) -> dict:
        shot_properties = {
            "shot_id": {"type": "string"},
            "scene_id": {"type": "string"},
            "duration_seconds": {"type": "number"},
            "characters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "location": {"type": "string"},
            "action": {"type": "string"},
            "camera_shot": {"type": "string"},
            "camera_movement": {"type": "string"},
            "lens_and_depth_of_field": {"type": "string"},
            "composition_notes": {"type": "string"},
            "lighting": {"type": "string"},
            "color_temperature": {"type": "string"},
            "mood": {"type": "string"},
            "visual_prompt": {"type": "string"},
            "speaking_characters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "speech_text": {"type": "string"},
            "dialogue_events": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {"type": "string"},
                        "text": {"type": "string"},
                        "continues_from_previous_shot": {"type": "boolean"},
                        "continues_to_next_shot": {"type": "boolean"},
                    },
                    "required": [
                        "speaker",
                        "text",
                        "continues_from_previous_shot",
                        "continues_to_next_shot",
                    ],
                    "additionalProperties": False,
                },
            },
            "continuity_start_state": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "lighting": {"type": "string"},
                    "environment": {"type": "string"},
                    "props": {"type": "array", "items": {"type": "string"}},
                    "camera_side": {"type": "string"},
                    "state_description": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "continuity_end_state": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "lighting": {"type": "string"},
                    "environment": {"type": "string"},
                    "props": {"type": "array", "items": {"type": "string"}},
                    "camera_side": {"type": "string"},
                    "state_description": {"type": "string"},
                },
                "additionalProperties": False,
            },
            "is_scene_boundary": {"type": "boolean"},
            "character_spatial_bboxes": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "character_spatial_regions": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "character_spatial_bboxes_start": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "character_spatial_bboxes_end": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
            "character_spatial_regions_start": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "character_spatial_regions_end": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        }
        required = list(shot_properties.keys())
        return {
            "type": "object",
            "properties": {
                "shots": {
                    "type": "array",
                    "minItems": min_items,
                    "maxItems": max_items,
                    "items": {
                        "type": "object",
                        "properties": shot_properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["shots"],
            "additionalProperties": False,
        }

    def _shot_batch_json_schema(
        self,
        scene_count: int = 2,
    ) -> dict:
        shot_schema = QwenDirectorPromptMixin._shot_json_schema()[
            "properties"
        ]["shots"]["items"]
        return {
            "type": "object",
            "properties": {
                "scene_shots": {
                    "type": "array",
                    "minItems": scene_count,
                    "maxItems": scene_count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "scene_id": {"type": "string"},
                            "shots": {
                                "type": "array",
                                "minItems": self.SHOTS_PER_SCENE,
                                "maxItems": self.SHOTS_PER_SCENE,
                                "items": shot_schema,
                            },
                        },
                        "required": [
                            "scene_id",
                            "shots",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["scene_shots"],
            "additionalProperties": False,
        }

    def _shot_director_batch_system(
        self,
    ) -> str:
        return SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT.replace(
            "__SHOTS_PER_SCENE__",
            str(self.SHOTS_PER_SCENE),
        )

    @staticmethod
    def _compact_story_context(story: str, max_chars: int = DIRECTOR_SHOT_STORY_CONTEXT_CHARS) -> str:
        """Return a compact narrative spine for repeated shot-planning prompts."""
        value = str(story or "").strip()
        if len(value) <= max_chars:
            return value

        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", value)
            if part.strip()
        ]
        if not sentences:
            return value[:max_chars].rstrip() + "…"

        if len(sentences) == 1:
            return value[:max_chars].rstrip() + "…"

        first = sentences[0]
        last = sentences[-1]
        if len(first) + len(last) + 1 <= max_chars:
            return f"{first} {last}"

        first_budget = max(220, int(max_chars * 0.60))
        last_budget = max_chars - first_budget - 1
        return (
            first[:first_budget].rstrip()
            + " "
            + last[:max(120, last_budget)].rstrip()
        ).strip()[:max_chars].rstrip() + "…"

    def _shot_director_batch_user(
        self,
        story: str,
        characters: list[dict],
        scenes: list[dict],
        visual_language: dict | None = None,
        reference_visual_context: dict[str, dict] | None = None,
    ) -> str:
        compact_characters = []

        for item in characters:
            if not isinstance(item, dict):
                continue

            name = str(
                item.get("name", "") or ""
            ).strip()

            if not name:
                continue

            compact_characters.append(
                {
                    "name": name,
                    "role": str(
                        item.get("role", "") or ""
                    ).strip(),
                }
            )

        language = {}

        if isinstance(visual_language, dict):
            for key in (
                "genre_tone",
                "color_palette",
                "lighting_philosophy",
                "camera_philosophy",
                "pacing",
            ):
                value = str(
                    visual_language.get(key, "") or ""
                ).strip()

                if value:
                    language[key] = value

        scene_payloads = []

        for scene in scenes:
            scene_payload = {
                "scene_id": str(
                    scene.get("scene_id", "") or ""
                ).strip(),
                "title": str(
                    scene.get("title", "") or ""
                ).strip(),
                "location": str(
                    scene.get("location", "") or ""
                ).strip(),
                "description": self._limit_text(
                    scene.get("description", ""),
                    DIRECTOR_SHOT_SCENE_DESCRIPTION_CHARS,
                ),
                "time_of_day": str(
                    scene.get("time_of_day", "") or ""
                ).strip(),
                "weather": str(
                    scene.get("weather", "") or ""
                ).strip(),
                "atmosphere": self._limit_text(
                    scene.get("atmosphere", ""),
                    DIRECTOR_SHOT_SCENE_ATMOSPHERE_CHARS,
                ),
                "mood": str(
                    scene.get("mood", "") or ""
                ).strip(),
                "lighting": self._limit_text(
                    scene.get("lighting", ""),
                    180,
                ),
                "color_temperature": str(
                    scene.get("color_temperature", "") or ""
                ).strip(),
                "environment_details": self._clean_list(
                    scene.get("environment_details", []),
                    limit=4,
                ),
                "key_props": self._clean_list(
                    scene.get("key_props", []),
                    limit=4,
                ),
                "characters": self._clean_list(
                    scene.get("characters", []),
                    limit=6,
                ),
                "scene_objective": self._limit_text(
                    scene.get("scene_objective", ""),
                    DIRECTOR_SHOT_SCENE_OBJECTIVE_CHARS,
                ),
                "continuity_notes": self._limit_text(
                    scene.get("continuity_notes", ""),
                    DIRECTOR_SHOT_SCENE_CONTINUITY_CHARS,
                ),
                "scene_function": str(
                    scene.get("scene_function", "development")
                    or "development"
                ).strip(),
                "obligatory_moment": self._limit_text(
                    scene.get("obligatory_moment", scene.get("description", "")),
                    220,
                ),
            }

            # Lossless prompt compaction: omit only fields carrying no
            # information. Populated semantic/cinematic values are unchanged.
            scene_payload = {
                key: value
                for key, value in scene_payload.items()
                if value not in ("", [], {})
            }

            scene_payloads.append(scene_payload)

        visual_context = {}
        context_source = reference_visual_context or self._reference_visual_context
        for path, analysis in context_source.items():
            if isinstance(analysis, dict):
                visual_context[str(path)] = {
                    "description": str(analysis.get("description", "") or "")[:500],
                    "identity_features": [str(v) for v in (analysis.get("identity_features", []) or [])][:6],
                    "wardrobe": [str(v) for v in (analysis.get("wardrobe", []) or [])][:6],
                    "environment": [str(v) for v in (analysis.get("environment", []) or [])][:6],
                    "lighting": str(analysis.get("lighting", "") or "")[:220],
                    "composition": str(analysis.get("composition", "") or "")[:220],
                }

        return json.dumps(
            {
                "story_context": self._compact_story_context(story, DIRECTOR_SHOT_STORY_CONTEXT_CHARS),
                "characters": compact_characters,
                "visual_language": language,
                "reference_visual_analysis": visual_context,
                "scenes": scene_payloads,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

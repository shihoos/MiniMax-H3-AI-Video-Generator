from __future__ import annotations

import json
import os
import re
import textwrap

from planner.config import (
    DIRECTOR_STORY_CONTEXT_CHARS,
    AI_STORY_MODE,
    DIRECTOR_TEMPERATURE,
    DIRECTOR_TOP_P,
    DIRECTOR_SHOT_STORY_CONTEXT_CHARS,
    DIRECTOR_SHOTS_PER_SCENE,
    DIRECTOR_SHOT_SCENE_DESCRIPTION_CHARS,
    DIRECTOR_SHOT_SCENE_OBJECTIVE_CHARS,
    DIRECTOR_SHOT_SCENE_CONTINUITY_CHARS,
    DIRECTOR_SHOT_SCENE_ATMOSPHERE_CHARS,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
)


SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT = '\nYou are the CINEMATOGRAPHY DIRECTOR for MiniMax H3.\n\nCreate exactly __SHOTS_PER_SCENE__ production-ready shots for EACH supplied scene.\n\nFor the `location` field, use ONLY the physical setting where the shot occurs.\nThe value must be a concrete place or environment, not an action, object, body part, emotion, event, clause, sentence fragment, or abstract phrase.\nWhen the shot remains in the same physical setting, preserve the supplied scene location exactly.\nWhen the supplied scene location is empty, infer the physical setting from the supplied story context, scene description, continuity notes, and environment details. Do not treat an arbitrary prepositional phrase as a location.\nOnly change `location` when the narrative explicitly moves to a different physical place.\n\nThe scenes are part of one coherent film. Use ONLY the supplied characters. Do not create new characters or invent character names.\nKeep action 10–30 words, visual_prompt 15–40 words, composition_notes <=18 words, lighting <=12 words, lens_and_depth_of_field <=10 words, mood <=5 words, camera_shot <=5 words, camera_movement <=5 words.\nlocation must be a specific physical place (e.g., "abandoned station platform", "stairwell", "underground chamber"), never a phrase from the story.\n`action` describes only physical/narrative events; never describe camera behavior there.\n`camera_shot` and `camera_movement` contain only framing/camera decisions; do not duplicate them in `action`.\nPreserve:\n- character identity;\n- chronology;\n- visual continuity;\n- location continuity;\n- emotional progression;\n- visual-language consistency.\nDIALOGUE CONTRACT:\n- dialogue_events contain DIRECT SPOKEN DIALOGUE only; never convert narrative prose, action description, internal thoughts, exposition, screen text, or UI text into speech.\n- Every dialogue_events[].speaker MUST exactly match one supplied canonical character name.\n- Relational canonical identities are valid when supplied (for example, "Eli\'s father"). Use the exact canonical identity, not a bare role.\n- Never invent a new generic speaker such as "man", "woman", "boy", "girl", "doctor", "guard", or "officer" when that surface is not a supplied canonical identity or validated semantic alias.\n- Preserve spoken text exactly; never paraphrase, summarize, or invent dialogue.\n- When the story contains explicit spoken dialogue or script-style dialogue, copy only those spoken lines. If there is no explicit spoken-dialogue anchor, return dialogue_events as an empty array.\n- speaking_characters must contain exactly the unique dialogue speakers, and speech_text must be the dialogue event texts joined in order.\n- Do not put timestamps in the response.\n- describe the shot\'s required initial and ending continuity states in continuity_start_state and continuity_end_state.\n\nWithin each scene, the required shots must use meaningfully different\nframing/composition while describing the SAME narrative beat.\n\nSCENE-FUNCTION DIRECTING:\nEach supplied scene includes scene_function and obligatory_moment.\nUse them as directing constraints, not as new story events.\nsetup: establish geography and protagonist context.\ncatalyst: reveal the disruptive event, clue, or discovery.\ndevelopment: show objective, movement, complication, or escalation.\nmidpoint: emphasize new information and changed understanding.\nclimax: emphasize danger, decisive action, choice, and consequence.\nfinale: emphasize aftermath, resolution, and the closing emotional image.\nEvery required shots must visibly serve the supplied obligatory_moment.\n\nSHOT / FRAMING VOCABULARY:\nframing: extreme wide, wide, full, medium wide, medium, medium close-up, close-up, extreme close-up, over-the-shoulder, two-shot, POV, insert.\n\nCAMERA MOVEMENT VOCABULARY:\nmovement: static, pan, tilt, dolly, tracking, handheld, crane, push-in, orbit.\n\nLENS / DEPTH OF FIELD:\nwide-angle, normal, telephoto, shallow focus, deep focus, selective focus.\n\nCOMPOSITION VOCABULARY:\ncentered, rule of thirds, leading lines, foreground frame, negative space, silhouette, depth layering, subject isolation.\n\nLIGHTING VOCABULARY:\nlighting: warm tungsten, cool daylight, golden-hour, blue-hour, moonlight, practical neon, hard chiaroscuro, soft overcast, mixed practical/ambient.\n\nReturn JSON only in exactly this structure:\n\n{\n  "scene_shots": [\n    {\n      "scene_id": "scene_001",\n      "shots": [\n        {\n          "shot_id": "scene_001_shot_001",\n          "scene_id": "scene_001",\n          "duration_seconds": 5.2,\n          "characters": [],\n          "location": "<physical setting only>",\n          "action": "...",\n          "camera_shot": "...",\n          "camera_movement": "...",\n          "lens_and_depth_of_field": "...",\n          "composition_notes": "...",\n          "lighting": "...",\n          "color_temperature": "...",\n          "mood": "...",\n          "visual_prompt": "...",\n          "speaking_characters": [],\n          "speech_text": "",\n          "dialogue_events": [],\n          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "is_scene_boundary": false,\n          "character_spatial_bboxes": {},\n          "character_spatial_regions": {},\n          "character_spatial_bboxes_start": {},\n          "character_spatial_bboxes_end": {},\n          "character_spatial_regions_start": {},\n          "character_spatial_regions_end": {}\n        },\n        {\n          "shot_id": "scene_001_shot_002",\n          "scene_id": "scene_001",\n          "duration_seconds": 5.2,\n          "characters": [],\n          "location": "<physical setting only>",\n          "action": "...",\n          "camera_shot": "...",\n          "camera_movement": "...",\n          "lens_and_depth_of_field": "...",\n          "composition_notes": "...",\n          "lighting": "...",\n          "color_temperature": "...",\n          "mood": "...",\n          "visual_prompt": "...",\n          "speaking_characters": [],\n          "speech_text": "",\n          "dialogue_events": [],\n          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "is_scene_boundary": false,\n          "character_spatial_bboxes": {},\n          "character_spatial_regions": {},\n          "character_spatial_bboxes_start": {},\n          "character_spatial_bboxes_end": {},\n          "character_spatial_regions_start": {},\n          "character_spatial_regions_end": {}\n        }\n      ]\n    }\n  ]\n}\n\nThere must be exactly __SHOTS_PER_SCENE__ shots inside every scene_shots entry and\nexactly one entry for every supplied scene. Do not add prose outside JSON.\n\nDo NOT output compiler-owned fields.\nDo NOT add scenes.\nDo NOT omit scenes.\nReturn JSON only.\n'

class QwenDirectorPromptMixin:
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
    3. DIALOGUE. Include several short lines of spoken dialogue between the
       protagonist and the principal secondary character. Dialogue must change
       a decision, reveal information, create conflict, or force a response.
       No filler dialogue.
    4. SECONDARY CHARACTER ARC. Introduce the principal secondary character,
       antagonist, or relationship counterpart through an actual character action
       or direct interaction by roughly the first half of the story. Do NOT first
       introduce that character as a nameplate, recorded message, disembodied voice,
       screen label, logbook entry, or other exposition device and reveal them later.
       Give the secondary character a concrete goal or position that conflicts with
       or complicates the protagonist's objective. Make their action or revelation
       materially change the protagonist's next decision.
    5. ESCALATION AND CHOICE. The story must build through at least two meaningful
       complications before the climax, then force the protagonist to make a
       consequential choice rather than merely observe the final reveal.
    6. RESOLUTION. End with a concrete aftermath showing consequences for the
       protagonist and the central relationship or conflict. Avoid generic moral
       endings such as “he was now a guardian of something greater” unless the
       story has earned that exact transformation through a specific event.

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
       organization, or new persistent character solely for surprise.
       Do not introduce a new named or persistent unnamed human/sentient character
       unless that identity is already grounded by the supplied source story.
       Prefer the source's existing characters, relationships, and implications
       when creating the reversal.
    2. Add at least two sentences of meaningful interiority for the protagonist,
       tied to a specific fear, memory, desire, or private realization.
    3. Preserve or add at least one short line of spoken dialogue by a named
       character. The line must advance conflict, reveal information, or connect
       to the central reversal. In EXPAND-STORY output, every spoken line MUST be
       enclosed in quotation marks; never present spoken dialogue as unquoted narrative.
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
        source_character_names: list[str] | None = None,
    ) -> str:

        source_text = self._compact_story_context(
            story,
            DIRECTOR_STORY_CONTEXT_CHARS,
        )
        result = (
            "MODE: "
            + str(mode)
            + "\n\nSOURCE STORY / PREMISE:\n"
            + source_text
        )
        if mode == EXPAND_USER_STORY_MODE and source_character_names:
            anchors = list(dict.fromkeys(
                str(value).strip()
                for value in source_character_names
                if str(value).strip()
            ))
            if anchors:
                result += (
                    "\n\nSOURCE CHARACTER ANCHORS:\n"
                    + ", ".join(anchors[:16])
                    + "\nThese are the source-grounded persistent characters. Preserve them. "
                    + "Do not turn a transient role, prop, voice, or generic description into a new persistent character."
                )
        result += (
            "\n\nFINAL OUTPUT REQUIREMENTS:\n"
            + "Return only the completed story. Prioritize finishing the full narrative, including the resolution, over adding extra detail. "
            + "End on a complete sentence with terminal punctuation. Include the required causal reversal, protagonist interiority, and functional dialogue."
        )
        return result

    @staticmethod
    def _story_architecture_json_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "protagonist": {"type": "string"},
                "protagonist_goal": {"type": "string"},
                "secondary_character": {"type": "string"},
                "secondary_goal": {"type": "string"},
                "relationship_tension": {"type": "string"},
                "inciting_incident": {"type": "string"},
                "early_character_action": {"type": "string"},
                "escalation_beats": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
                "midpoint_reversal": {"type": "string"},
                "climax_choice": {"type": "string"},
                "ending_consequence": {"type": "string"},
                "cliche_traps_to_avoid": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 6,
                    "items": {"type": "string"},
                },
            },
            "required": [
                "protagonist",
                "protagonist_goal",
                "secondary_character",
                "secondary_goal",
                "relationship_tension",
                "inciting_incident",
                "early_character_action",
                "escalation_beats",
                "midpoint_reversal",
                "climax_choice",
                "ending_consequence",
                "cliche_traps_to_avoid",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _story_architecture_system(mode: str) -> str:
        if mode == AI_STORY_MODE:
            return textwrap.dedent("""
    You are the story architect for MiniMax H3.

    Before the prose writer drafts the story, design a compact causal architecture for a high-quality
    cinematic short film. The target quality is the successful two-character pattern used by the Director:
    a named protagonist with a concrete objective, a principal secondary character with an independent
    goal or position, direct two-way conflict, escalating complications, a reversal rooted in an earlier
    concrete detail, a costly climax choice, and a concrete aftermath.

    The secondary character must be a real dramatic participant. Do not make them a late nameplate,
    recording, disembodied voice, screen message, or exposition device. Avoid generic endings such as
    “guardian of something greater,” “the world had changed,” “some secrets were meant to stay buried,”
    or other vague moral transformations unless the premise specifically earns them.

    Every beat must causally alter the next decision. Avoid random monsters, secret organizations, destiny
    prophecies, or twists introduced only for surprise.
    """).strip()

        return textwrap.dedent("""
    You are the story architect for MiniMax H3 Expand Story mode.

    Preserve the source story's core characters, events, chronology, setting, and outcome. Plan an expansion
    that deepens the existing character relationship and causal chain rather than replacing the story.
    Keep the persistent character set stable and do not invent a new persistent human/sentient identity merely
    to create a twist. Build meaningful escalation, a grounded reversal, a consequential climax choice, and
    concrete aftermath.
    """).strip()

    @staticmethod
    def _story_architecture_user(
        mode: str,
        story: str,
        source_character_names: list[str] | None = None,
    ) -> str:
        payload = {
            "mode": mode,
            "source_story": str(story or "").strip(),
        }
        if source_character_names:
            payload["source_character_anchors"] = list(dict.fromkeys(
                str(value).strip()
                for value in source_character_names
                if str(value).strip()
            ))[:16]
        return (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\nReturn JSON only. Do not write the story prose yet."
        )

    @staticmethod
    def _story_writer_blueprint_instruction(blueprint: dict) -> str:
        return (
            "\n\nSTORY ARCHITECTURE BLUEPRINT:\n"
            + json.dumps(blueprint, ensure_ascii=False, indent=2)
            + "\n\nUse the blueprint as the causal skeleton, not as prose to copy. "
            + "Write one continuous finished story. Every major beat must change what the protagonist can do next. "
            + "The secondary character must act, pursue their own objective, and materially change the protagonist's decision. "
            + "The reversal must reinterpret an earlier concrete detail. The climax must force a real choice and the ending "
            + "must show concrete consequences."
        )

    @staticmethod
    def _story_quality_json_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "pass": {"type": "boolean"},
                "score": {"type": "integer", "minimum": 0, "maximum": 40},
                "secondary_character_arc": {"type": "integer", "minimum": 0, "maximum": 5},
                "character_conflict": {"type": "integer", "minimum": 0, "maximum": 5},
                "causal_reversal": {"type": "integer", "minimum": 0, "maximum": 5},
                "escalation_choice": {"type": "integer", "minimum": 0, "maximum": 5},
                "dialogue": {"type": "integer", "minimum": 0, "maximum": 5},
                "interiority": {"type": "integer", "minimum": 0, "maximum": 5},
                "resolution": {"type": "integer", "minimum": 0, "maximum": 5},
                "originality": {"type": "integer", "minimum": 0, "maximum": 5},
                "problems": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 8,
                },
            },
            "required": [
                "pass",
                "score",
                "secondary_character_arc",
                "character_conflict",
                "causal_reversal",
                "escalation_choice",
                "dialogue",
                "interiority",
                "resolution",
                "originality",
                "problems",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _story_quality_review_system(mode: str) -> str:
        return textwrap.dedent(f"""
    You are the strict narrative-quality gate for MiniMax H3 {mode}.

    Judge the supplied completed story as a short-film narrative, not as a list of
    style requirements. A story passes only when it is genuinely character-driven,
    causally coherent, dramatically escalating, and complete.

    QUALITY TARGET:
    - The protagonist has a concrete objective and a personal stake.
    - A principal secondary character is a real dramatic participant, not a late
      nameplate/voice/screen reveal. That character has a distinct position,
      motivation, or relationship to the protagonist and materially changes the plot.
    - Their interaction creates conflict or competing goals, not just exposition.
    - The reversal grows from earlier concrete details rather than arriving as a
      random monster/secret/device.
    - The climax contains a consequential choice.
    - Dialogue exposes conflict, information, or a decision; it is not filler.
    - Interiority is specific and tied to events.
    - The ending shows concrete consequences instead of a generic moral statement.
    - The story should feel at least as dramatically developed as a strong two-character
      short-film example: active secondary character, meaningful two-way dialogue,
      causal reveal, escalating stakes, and earned consequences.

    HARD FAIL CONDITIONS:
    - the secondary character first appears only through a nameplate, recording,
      disembodied voice, screen label, or exposition;
    - the protagonist simply learns a secret without having to make a consequential
      choice because of it;
    - the second character is mostly an information dispenser;
    - the final paragraph becomes a generic “guardian / greater purpose / changed man”
      moral without concrete consequences;
    - the story is substantially weaker than its own central character conflict.

    Score each dimension 0-5. Pass requires every core dimension to be at least 4,
    originality at least 3, and total score at least 32/40.

    Return JSON only.
    """).strip()

    @staticmethod
    def _story_quality_review_user(
        mode: str,
        story: str,
        source_character_names: list[str] | None = None,
    ) -> str:
        anchors = ""
        if source_character_names:
            values = [str(v).strip() for v in source_character_names if str(v).strip()]
            if values:
                anchors = "\nSOURCE CHARACTER ANCHORS:\n" + ", ".join(values[:16])
        return (
            "STORY:\n"
            + str(story).strip()
            + anchors
            + "\n\nReturn the strict JSON quality assessment."
        )

    @staticmethod
    def _story_quality_repair_system(mode: str) -> str:
        return textwrap.dedent(f"""
    You are the senior narrative editor for MiniMax H3 {mode}.

    Repair the supplied story into a stronger short-film narrative without throwing
    away its premise, core events, setting, or established characters.

    The repaired story MUST:
    - keep the protagonist's objective and personal stake;
    - introduce the principal secondary character through action/direct interaction,
      never first through a nameplate, recording, disembodied voice, screen label,
      or exposition-only reveal;
    - make the secondary character an active participant with a distinct position or
      goal that conflicts with or complicates the protagonist;
    - include several purposeful exchanges between them;
    - build at least two complications into an earned climax;
    - force a consequential protagonist choice;
    - make the central reversal causally emerge from earlier details;
    - retain concrete, specific interiority;
    - finish with concrete aftermath and consequences, not a generic moral;
    - preserve source characters/chronology/outcome in EXPAND-STORY mode;
    - never invent a new persistent character in EXPAND-STORY merely to create a twist.

    Preserve strong material from the supplied draft. Repair weak material rather than
    replacing the entire story with unrelated plot.

    Output ONLY the repaired story prose.
    """).strip()

    @staticmethod
    def _story_quality_repair_user(
        mode: str,
        story: str,
        review: dict,
        source_character_names: list[str] | None = None,
    ) -> str:
        anchors = ""
        if source_character_names:
            values = [str(v).strip() for v in source_character_names if str(v).strip()]
            if values:
                anchors = "\nSOURCE CHARACTER ANCHORS: " + ", ".join(values[:16])
        problems = review.get("problems", []) if isinstance(review, dict) else []
        return (
            "CURRENT STORY:\n"
            + str(story).strip()
            + anchors
            + "\n\nQUALITY FAILURES TO REPAIR:\n"
            + "\n".join(f"- {str(p).strip()}" for p in problems if str(p).strip())
            + "\n\nReturn only the fully repaired story."
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
                                "maxItems": 8,
                                "items": {"type": "string"},
                            },
                            "identity_type": {
                                "type": "string",
                                "enum": [
                                    "named_character",
                                    "relational_character",
                                    "descriptive_character",
                                ],
                            },
                            "relationship_to": {
                                "type": "string",
                            },
                            "relationship": {
                                "type": "string",
                            },
                        },
                        "required": [
                            "name",
                            "entity_type",
                            "is_character",
                            "aliases",
                            "identity_type",
                            "relationship_to",
                            "relationship",
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
    Return JSON only. Analyze the supplied story and classify stable human/sentient identities that can receive
    an identity lock. There are THREE valid character identity types:

    1) named_character: a proper/named character such as "Eli" or "Lin Mei".
    2) relational_character: a persistent unnamed character whose identity is grounded by a named story
       character and a concrete relationship, such as "Eli's father", "Sara's sister", or "Mira's commander".
    3) descriptive_character: a persistent unnamed person whose identity is grounded by a distinctive,
       recurring description, such as "the man in the suit" or "the woman with piercing eyes".

    Bare generic role labels are NOT canonical identities: "man", "woman", "boy", "girl", "person",
    "doctor", "scientist", "guard", "officer", and similar labels must not be returned as a canonical
    identity. A descriptive_character must contain a distinguishing description beyond the bare role.
    Relationship surfaces such as "father", "uncle", "his father", or "the man" are semantic references,
    not new canonical entities unless the story establishes a persistent relational/descriptive identity.

    Do NOT invent a relationship or identity. For a relational_character, relationship_to MUST name a canonical
    character actually grounded in the story, relationship MUST be one concrete family/role relation, and the
    supplied story must contain enough evidence to support that link. Prefer a canonical name in the form
    "<Canonical Character>'s <relationship>" and include grounded surface forms in aliases.
    For a descriptive_character, do not invent a relationship; preserve the most specific stable descriptor
    actually grounded in the story and include generic surface forms (for example "man") only as aliases.

    Do NOT treat locations, organizations, facilities, projects, missions, events, objects, calendar words, or
    weather as characters. Titles such as Dr., Captain, Commander, etc. are not part of the canonical name.
    For every deterministic candidate that is textually grounded, explicitly classify it; do not silently omit
    a supplied named or source-labeled character merely because the introduction uses a nameplate, badge, title,
    dialogue attribution, or another narrative surface. Qwen may reject a candidate only when the story evidence
    actually shows that it is not a character. Recover stable named, relational, and descriptive identities that
    the deterministic scan may not have named yet.
    """).strip()

        user_payload = json.dumps(
            {
                "story": self._compact_story_context(story, DIRECTOR_STORY_CONTEXT_CHARS),
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
            max_completion=768,
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
    You are the final character-identity adjudicator. Return JSON only and obey the supplied JSON schema exactly.
    Review only the supplied candidate names and the supplied semantic extraction. Never invent an unrelated person.
    For every supplied candidate, emit an explicit boolean `is_character` decision.
    Use `is_character=false` for NOT_CHARACTER or UNCERTAIN; there is no separate `decision` field.
    Use `entity_type=PERSON` and `identity_type=named_character` for named characters.
    Use `identity_type=relational_character` only when relationship_to is a grounded canonical character and the
    relationship is explicitly or unambiguously established by the story.
    Use `identity_type=descriptive_character` for a recurring unnamed person whose identity is grounded by a
    distinctive description (for example, `the woman with piercing eyes` or `the man in the suit`). The
    canonical descriptive name must contain the distinguishing description; bare `man`/`woman`/`stranger`
    are never canonical identities. If a candidate has no concrete supported relationship, do NOT force it into
    `relational_character`; classify it as `descriptive_character` when its distinguishing description is stable.
    Preserve grounded aliases without turning the alias itself into another canonical person.
    For possessive pronouns such as `his father`, follow the established discourse owner, not the nearest noun.
    Example: in `a sentient AI his father had created`, do not invent `AI's father` unless the story explicitly
    establishes that relationship; the pronoun must resolve from the previously established owner.
    A strongly story-grounded named, relational, or descriptive character must not be removed merely because a
    generic role label was classified negatively; explicit source evidence outranks a weak generic-role negative.
    """).strip()
        payload = json.dumps(
            {
                "story": self._compact_story_context(story, DIRECTOR_STORY_CONTEXT_CHARS),
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

    @staticmethod
    def _shot_batch_json_schema(
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
                                "minItems": DIRECTOR_SHOTS_PER_SCENE,
                                "maxItems": DIRECTOR_SHOTS_PER_SCENE,
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

            entry = {
                "name": name,
                "role": str(
                    item.get("role", "") or ""
                ).strip(),
            }
            profile = item.get("identity_profile")
            profile = profile if isinstance(profile, dict) else {}
            for key in ("identity_type", "relationship_to", "relationship"):
                value = str(item.get(key, profile.get(key, "")) or "").strip()
                if value:
                    entry[key] = value
            semantic_aliases = [
                str(value).strip()
                for value in (item.get("semantic_aliases", profile.get("semantic_aliases", [])) or [])
                if str(value).strip()
            ][:8]
            if semantic_aliases:
                entry["semantic_aliases"] = semantic_aliases
            compact_characters.append(entry)

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

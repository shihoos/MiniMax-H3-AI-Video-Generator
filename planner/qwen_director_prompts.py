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
    DIRECTOR_SHOT_SCENE_DESCRIPTION_CHARS,
    DIRECTOR_SHOT_SCENE_OBJECTIVE_CHARS,
    DIRECTOR_SHOT_SCENE_CONTINUITY_CHARS,
    DIRECTOR_SHOT_SCENE_ATMOSPHERE_CHARS,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
)
from planner.entity_resolver import EntityResolver


SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT = """
You are the CINEMATOGRAPHY DIRECTOR for MiniMax H3.

Create exactly __SHOTS_PER_SCENE__ production-ready shots for EACH supplied scene.

For the `location` field, use ONLY the physical setting where the shot occurs.
The value must be a concrete place or environment, not an action, object, body part, emotion, event, clause, sentence fragment, or abstract phrase.
When the shot remains in the same physical setting, preserve the supplied scene location exactly.
When the supplied scene location is empty, infer the physical setting from the supplied story context, scene description, continuity notes, and environment details. Do not treat an arbitrary prepositional phrase as a location.
Only change `location` when the narrative explicitly moves to a different physical place.
EVERY shot of one scene MUST use the identical `location` string (one scene = one place). Never use a different
wording for the same place between the shots of a scene, and never place an indoor scene on an outdoor platform or vice versa.

The scenes are part of one coherent film. Use ONLY the supplied characters. Do not create new characters or invent character names.
In `characters` and in every dialogue speaker use the EXACT canonical spelling and capitalization from the supplied `characters` list (never a lower-cased or shortened copy).
Only list a character in a shot if that character is visible in that shot's framing.
Keep action 10–30 words, visual_prompt 15–40 words, composition_notes <=18 words, lighting <=12 words, lens_and_depth_of_field <=10 words, mood <=5 words, camera_shot <=5 words, camera_movement <=5 words.
location must be a specific physical place (e.g., "abandoned station platform", "stairwell", "underground chamber"), never a phrase from the story.
`action` describes only physical/narrative events; never describe camera behavior there.
`camera_shot` and `camera_movement` contain only framing/camera decisions; do not duplicate them in `action`.
Preserve:
- character identity;
- chronology;
- visual continuity;
- location continuity;
- emotional progression;
- visual-language consistency.
DIALOGUE CONTRACT:
- dialogue_events contain DIRECT SPOKEN DIALOGUE ONLY; never convert narrative prose, action description, internal thoughts, exposition, screen text, or UI text into speech.
- A dialogue event MUST be an exact contiguous span of a quoted/scripted spoken line present in the supplied story context. If the text is not explicitly spoken, it is NOT dialogue.
- Text on signs, tags, labels, screens, monitors, keypads, canisters, files, and IDs is WRITTEN text, never dialogue; never assign it to any speaker.
- Copy dialogue with its original capitalization and punctuation exactly as listed in that scene's `source_dialogue` item; never lower-case it.
- Put a spoken line in the shot where it is actually said in the story, with its speaker visible and listed in `characters`; the `action` of that shot must show that character speaking. Do not spread unspoken shots across a line, and do not assign a line to a shot whose action is about something else.
- A sentence such as a character's action, breath, expression, realization, or narration is NEVER dialogue, even when it begins with the character's name.
- Every source utterance may be used only once. The only exception is splitting one longer source utterance into exact contiguous pieces across adjacent shots. Never repeat the same source line in multiple shots.
- Every dialogue_events[].speaker MUST exactly match one supplied canonical character name.
- Relational canonical identities are valid when supplied. Use the exact canonical identity, not a bare role.
- Never invent a new generic speaker such as "man", "woman", "boy", "girl", "doctor", "guard", or "officer" when that surface is not a supplied canonical identity or validated semantic alias.
- Preserve spoken text exactly; never paraphrase, summarize, invent, or transform narration into speech.
- When the story contains explicit spoken dialogue or script-style dialogue, copy only those spoken lines. If there is no explicit spoken-dialogue anchor, return dialogue_events as an empty array.
- A scene entry may contain a scene-local `source_dialogue` array. Each item binds an exact source `text` to its canonical `speaker`; that pairing is authoritative.
- A shot may use only entries from its own scene's `source_dialogue`. Never borrow a line from another scene or from the compact global story_context, even if its words appear there. Preserve the paired speaker and exact text; only split a longer line into contiguous pieces across adjacent shots in that same scene when necessary. If the scene has no applicable entry, return dialogue_events as an empty array.

- `speaking_characters` and `speech_text` are derived from `dialogue_events`; focus on producing correct dialogue_events and do not invent independent values for those fields.
- Do not put timestamps in the response.
- Treat H3 shot duration as a hard production constraint: target roughly 4–6 seconds of spoken dialogue per shot when natural, leaving timing headroom.
- When dialogue is present, keep `duration_seconds` conservative enough to leave legal timing headroom; the compiler will finalize the exact legal duration.
- Prefer one concise dialogue event per shot; use two only when the exchange genuinely needs both sides. Keep each spoken event short when the source permits, but never paraphrase or delete source dialogue.
- If a source utterance is longer, split its exact contiguous text across adjacent shots with continues_to_next_shot/continues_from_previous_shot rather than forcing an overlong single shot.
- A continuation edge is SAME-SPEAKER ONLY: if `continues_from_previous_shot` is true, the current speaker MUST be the same canonical character as the previous shot final dialogue speaker. If the speaker changes, both continuation flags at that boundary MUST be false and the new speaker starts a new dialogue turn.
- Never set `continues_to_next_shot=true` when the next shot begins with a different speaker.
- Do not pack dialogue into one shot when it can be distributed across the required shots of the same scene while preserving exact text and order.
- describe the shot's required initial and ending continuity states in continuity_start_state and continuity_end_state.

Within each scene, the required shots must use meaningfully different
framing/composition while describing the SAME narrative beat.
LIGHT AND PLACE CONTINUITY: all shots of one scene share ONE lighting setup. Give every shot of a scene the same
`lighting` and `color_temperature` strings (same source, direction, and color); vary only framing, lens, movement, and composition.
Choose lighting that the scene's own location, time of day, weather, and props physically motivate (storm daylight, flashlight beam,
emergency lamps, monitor glow, moonlight, firelight, sodium streetlight). The lighting vocabulary below is a spelling guide, NOT a menu to rotate:
never use "practical neon" or "warm tungsten" unless the scene visibly contains neon signs or tungsten lamps, and never use "soft overcast" indoors.
`visual_prompt` must name the setting and the light source so the image generator can reproduce it.

STORY FIDELITY: depict only events, objects, and states that the scene description or story context actually states.
Never add an opening, discovery, collapse, sound, object, or line of speech that the text does not contain (a vault described as
sealed stays sealed; an unnamed object stays unnamed). When a source is short and the same description appears in several scenes, the
scenes are different COVERAGE of the same moment: vary distance, angle, and detail (approach, threshold, close detail, reaction),
never advance the plot beyond what the description states, and never repeat the same action/visual in consecutive scenes.

SCENE-FUNCTION DIRECTING:
Each supplied scene includes scene_function and obligatory_moment.
Use them as directing constraints, not as new story events.
setup: establish geography and protagonist context.
catalyst: reveal the disruptive event, clue, or discovery.
development: show objective, movement, complication, or escalation.
midpoint: emphasize new information and changed understanding.
climax: emphasize danger, decisive action, choice, and consequence.
finale: emphasize aftermath, resolution, and the closing emotional image.
Every required shots must visibly serve the supplied obligatory_moment (when it is omitted, the scene description is the obligatory moment).

SHOT / FRAMING VOCABULARY:
framing: extreme wide, wide, full, medium wide, medium, medium close-up, close-up, extreme close-up, over-the-shoulder, two-shot, POV, insert.

CAMERA MOVEMENT VOCABULARY:
movement: static, pan, tilt, dolly, tracking, handheld, crane, push-in, orbit.

LENS / DEPTH OF FIELD:
wide-angle, normal, telephoto, shallow focus, deep focus, selective focus.

COMPOSITION VOCABULARY:
centered, rule of thirds, leading lines, foreground frame, negative space, silhouette, depth layering, subject isolation.

LIGHTING VOCABULARY (spelling guide only; see LIGHT AND PLACE CONTINUITY):
lighting: warm tungsten, cool daylight, golden-hour, blue-hour, moonlight, practical neon, hard chiaroscuro, soft overcast, mixed practical/ambient, flashlight beam, emergency lamps, monitor glow, firelight.
Write all framing, movement, lens, and composition terms in lowercase exactly as listed above.

Return JSON only in exactly this structure:

{
  "scene_shots": [
    {
      "scene_id": "scene_001",
      "shots": [
        {
          "shot_id": "scene_001_shot_001",
          "scene_id": "scene_001",
          "duration_seconds": 5.2,
          "characters": [],
          "location": "<physical setting only>",
          "action": "...",
          "camera_shot": "...",
          "camera_movement": "...",
          "lens_and_depth_of_field": "...",
          "composition_notes": "...",
          "lighting": "...",
          "color_temperature": "...",
          "mood": "...",
          "visual_prompt": "...",
          "speaking_characters": [],
          "speech_text": "",
          "dialogue_events": [],
          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},
          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},
          "is_scene_boundary": false,
          "character_spatial_bboxes": {},
          "character_spatial_regions": {},
          "character_spatial_bboxes_start": {},
          "character_spatial_bboxes_end": {},
          "character_spatial_regions_start": {},
          "character_spatial_regions_end": {}
        },
        {
          "shot_id": "scene_001_shot_002",
          "scene_id": "scene_001",
          "duration_seconds": 5.2,
          "characters": [],
          "location": "<physical setting only>",
          "action": "...",
          "camera_shot": "...",
          "camera_movement": "...",
          "lens_and_depth_of_field": "...",
          "composition_notes": "...",
          "lighting": "...",
          "color_temperature": "...",
          "mood": "...",
          "visual_prompt": "...",
          "speaking_characters": [],
          "speech_text": "",
          "dialogue_events": [],
          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},
          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},
          "is_scene_boundary": false,
          "character_spatial_bboxes": {},
          "character_spatial_regions": {},
          "character_spatial_bboxes_start": {},
          "character_spatial_bboxes_end": {},
          "character_spatial_regions_start": {},
          "character_spatial_regions_end": {}
        }
      ]
    }
  ]
}

There must be exactly __SHOTS_PER_SCENE__ shots inside every scene_shots entry and
exactly one entry for every supplied scene. Do not add prose outside JSON.

Do NOT output compiler-owned fields.
Do NOT add scenes.
Do NOT omit scenes.
Return JSON only.
"""


class QwenDirectorPromptMixin:
    _STORY_CRAFT_RULES = """
FORMAT AND DRAMATIC SPINE
- Write third-person past-tense cinematic prose only. Exactly six paragraphs separated by one blank line; each paragraph is one filmable scene in one place and continuous time. Keep 420-560 words. No headings, labels, camera directions, JSON, or commentary.
- Use the supplied STORY BIBLE as the single outline when present; do not create a competing second outline. If no plan is supplied, silently settle the protagonist's goal, resistance, planted evidence, reversal, consequential choice, and aftermath before writing.
- Paragraphs 1-3 establish the goal, first attempt, and growing pressure through actions and consequences. Paragraph 4 delivers a reversal caused by something established earlier. Paragraph 5 forces a consequential choice with a visible price. Paragraph 6 shows the concrete result and settled aftermath. This is scene-function guidance, not permission to force unrelated events into the premise.
- State the protagonist's concrete goal early. Make every important event follow from a visible decision, established fact, character action, or physical consequence. The turn must change what the protagonist does. End the central conflict; do not replace resolution with a future hook.

CHARACTERS AND DIALOGUE
- Qwen chooses all character names and cast membership. There is no fixed cast size, target, or maximum. Every planned cast member must appear and act in the current events, not merely be named in backstory, a document, a photograph, a recording, a screen, or a memory.
- Prefer a physically present counterpart with an independent immediate want when the premise can naturally support human conflict. Their action, refusal, bargain, deception, or competing duty should change the protagonist's options, and the protagonist should respond. A remote setting, dangerous job, or technical problem alone does not mean the story must be solitary. Use a solitary protagonist when the premise specifically depends on physical or social isolation, not merely because solitude is convenient. Never add a decorative character or person to satisfy a count.
- Include a short, purposeful line of live dialogue from a named, physically present character. If a consequential counterpart is present, prefer a brief exchange that changes a decision, reveals a conflict, or makes the turn land. Keep each spoken line natural and concise; dialogue should not explain the premise or deliver backstory. A recording, transmission, alarm, displayed message, or computer output is evidence, not live speech.
- Give each consequential person a distinct want and one stable canonical name. Avoid inventing relationships that the premise or story does not ground. Keep source-established people, chronology, location, outcome, and defining details intact in Expand Story; add new people only when they materially serve that source-specific causal story.

CAUSALITY, SPECIFICITY, AND PROSE
- Plant a concrete detail, skill, constraint, object, or relationship before the turn depends on it; pay it off through action or consequence. Track where people and important objects are; nothing teleports, vanishes, repeats, or changes hands without an event explaining it.
- Make the setting specific to this premise and protagonist's work. Prefer precise action and concrete evidence over generic warnings, hidden chambers, malfunctioning systems, countdowns, vague signals, stock genre twists, exposition dumps, purple prose, and named-feeling summaries. Use such devices only when the supplied premise genuinely calls for them.
- Include at least two concrete sentences of protagonist interiority grounded in a specific fear, memory, desire, or private realization. Show why the stakes matter; keep backstory subordinate to present action.
- Vary sentence rhythm and avoid repeated phrases. Spoken lines stay inside their scene paragraph, never on separate screenplay lines. Plain text only. The sixth paragraph must be a developed aftermath of at least 25 words and end with a completed action in a settled situation.
""".strip()

    def _story_text_system(
        self,
        mode: str,
    ) -> str:
        common = (
            "You are the narrative writer for MiniMax H3. Produce one complete, original, film-ready short-film story. "
            "Think through cause and effect before drafting, but output only the finished prose.\n\n"
            + self._STORY_CRAFT_RULES
        )
        if mode == AI_STORY_MODE:
            return (
                common
                + "\n\nAI STORY PREMISE FIDELITY\n"
                + "Treat the user input as a premise, not a completed story. Preserve its defining facts, profession, place, discovery, and constraints. "
                "Show expertise through the protagonist's skilled actions. Develop the premise into a causal narrative rather than paraphrasing it."
            )
        if mode == EXPAND_USER_STORY_MODE:
            return (
                common
                + "\n\nEXPAND STORY SOURCE FIDELITY (strict)\n"
                + "The supplied story is canon. Preserve its important characters, events in order, setting, outcome, and defining meaning. "
                "Deepen the existing story with motivation, resistance, escalation, a source-grounded reversal if appropriate, a difficult choice, and consequences. "
                "Do not replace the source plot with a generic genre plot or invent backstory/relationships just to manufacture a character."
            )
        raise ValueError("Preserve Story does not use a story-text pass.")

    def _story_text_user(
        self,
        mode: str,
        story: str,
        source_character_names: list[str] | None = None,
        story_bible: dict | None = None,
    ) -> str:
        source_text = self._compact_story_context(
            story,
            DIRECTOR_STORY_CONTEXT_CHARS,
        )
        result = "MODE: " + str(mode) + "\n\nSOURCE STORY / PREMISE:\n" + source_text
        if mode == EXPAND_USER_STORY_MODE:
            anchors = list(dict.fromkeys(
                str(value).strip()
                for value in (source_character_names or [])
                if str(value).strip()
            ))
            if anchors:
                result += (
                    "\n\nSOURCE CHARACTER ANCHORS (preserve; not a cast limit):\n"
                    + ", ".join(anchors[:16])
                )
        bible_text = self._format_story_bible(story_bible)
        if bible_text:
            result += "\n\n" + bible_text
        result += (
            "\n\nOUTPUT: exactly six paragraphs, 420-560 words, plain prose only. "
            "Use the supplied outline if present and resolve the central conflict with a concrete aftermath."
        )
        return result


    _STORY_BIBLE_SYSTEM = textwrap.dedent("""
    You are the head writer planning a short cinematic story for MiniMax H3 BEFORE any prose is written.
    Return JSON only. Plan ONE story that fills exactly six filmable scenes; each scene is one location and one continuous stretch of time.

    DIVERGE THEN SELECT (fill these fields first, in order): `obvious_version` is the one-sentence plot most writers would reach for with this premise.
    `concepts` are three different readings of the premise whose CAUSE of conflict differs from the obvious version and from each other (each under 30 words,
    each built on specific people, a specific profession, a specific object or place). `chosen_concept` is the one that gives the people the most incompatible wants
    and the most specific physical detail. Everything below must be built on the chosen concept, not on the obvious version.

    CAST: you decide the cast; there is no fixed number, target, or maximum. Prefer a physically present counterpart with an independent want when that person's choices can create meaningful resistance or change the protagonist's options. A remote location, a difficult job, or a broken machine alone is not enough to justify a solitary cast. Plan a lone protagonist only when physical or social isolation is essential to the premise's conflict, and state why in cast_rationale. Every planned cast member must appear and act in the present scenes, have a grounded want and one stable canonical full name, and participate in the causal chain. Nobody known only through a log, label, recording, display, or memory is a cast member.

    ORIGINALITY: do not take the nearest genre default for the premise. Find the reading of the premise that only this setting and these people
    could produce. A mysterious object, signal, or event must be defined concretely (what it is, who made or left it, what it does) and shown
    through use or consequence, never left as an unexplained glow, a classified project, or a hum.

    CONFLICT: people with incompatible wants, not a malfunctioning system. The environment may press on them but is not the antagonist.
    PLANTED DETAIL: one concrete fact, object, skill, or relationship shown in scene 1 or 2 that decides the choice in scene 5.
    TURN: lands in scene 4, grows out of something already shown, and forces a different decision.
    CHOICE and PRICE: two outcomes that cannot both be kept; the price is real and visible (it may be bleak or bittersweet).
    ENDING IMAGE: one concrete settled image that echoes a detail from scene 1.
    BEATS: exactly six strings, one per scene in order (open, first attempt, pressure, turn, choice, aftermath). Each names the place and states who does what and the visible consequence.
    The beats are one causal chain: the consequence of each beat is the problem of the next. Track every cast member and the planted object through all six beats; nobody and nothing
    vanishes or is used twice without a beat saying how. The ENDING IMAGE must restate what beat 6 shows (same person, same place, same object).
    Keep every field short and concrete (under 40 words each).
    """).strip()

    _STORY_BIBLE_EXPAND_NOTE = (
        "\n\nEXPAND MODE: the supplied source story is canon. Keep its people, setting, events in order, and outcome. "
        "The plan deepens it with cause, opposition, evidence, and consequence; it never replaces it with a different plot. "
        "Source characters are anchors, not a cast limit; add a new person only when the conflict needs them, and bring them on screen."
    )

    @staticmethod
    def _story_bible_json_schema() -> dict:
        def text(limit: int = 280) -> dict:
            return {"type": "string", "minLength": 1, "maxLength": limit}

        return {
            "type": "object",
            "properties": {
                "obvious_version": text(240),
                "concepts": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": text(240),
                },
                "chosen_concept": text(300),
                "cast": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 1, "maxLength": 64},
                            "role": text(120),
                            "want": text(200),
                            "flaw_or_secret": text(200),
                            "stance_toward_protagonist": text(160),
                        },
                        "required": ["name", "role", "want", "flaw_or_secret", "stance_toward_protagonist"],
                        "additionalProperties": False,
                    },
                },
                "cast_rationale": text(240),
                "central_conflict": text(300),
                "planted_detail": text(240),
                "turn": text(300),
                "choice_and_price": text(300),
                "ending_image": text(240),
                "beats": {
                    "type": "array",
                    "minItems": 6,
                    "maxItems": 6,
                    "items": text(320),
                },
            },
            "required": [
                "obvious_version", "concepts", "chosen_concept",
                "cast", "cast_rationale", "central_conflict", "planted_detail",
                "turn", "choice_and_price", "ending_image", "beats",
            ],
            "additionalProperties": False,
        }

    def _plan_story_bible(
        self,
        mode: str,
        story: str,
        source_character_names: list[str] | None = None,
    ) -> dict | None:
        """One compact thinking-mode planning call (schema-guided). Never fatal.

        Qwen decides the cast, conflict, planted detail, turn, price and six beats; the
        prose pass then writes from that plan. Any failure returns None and the prose
        pass runs exactly as before.
        """
        self._planned_cast = []
        try:
            system = self._STORY_BIBLE_SYSTEM
            if mode == EXPAND_USER_STORY_MODE:
                system += self._STORY_BIBLE_EXPAND_NOTE
            anchors = list(dict.fromkeys(
                str(value).strip()
                for value in (source_character_names or [])
                if str(value).strip()
            ))
            payload = {
                "mode": str(mode),
                "premise_or_source_story": self._compact_story_context(story, DIRECTOR_STORY_CONTEXT_CHARS),
                "source_character_anchors": anchors[:16],
            }
            print("[QWEN] story_bible_thinking=on", flush=True)
            bible = self._chat_json(
                system,
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                minimum_completion=300,
                temperature=0.60,
                top_p=0.95,
                call_name="story_bible",
                max_completion=2800,
                json_mode=True,
                disable_thinking=False,
                response_schema=self._story_bible_json_schema(),
            )
        except Exception as exc:  # planning is an enhancement, never a hard dependency
            print(f"[QWEN] story_bible skipped ({type(exc).__name__}: {str(exc)[:160]})", flush=True)
            return None

        if not isinstance(bible, dict):
            return None
        cast = [
            item for item in (bible.get("cast") or [])
            if isinstance(item, dict) and str(item.get("name", "") or "").strip()
        ]
        beats = [str(item).strip() for item in (bible.get("beats") or []) if str(item).strip()]
        if not cast or len(beats) != 6:
            print("[QWEN] story_bible skipped (incomplete plan)", flush=True)
            return None
        bible["cast"] = cast
        bible["beats"] = beats
        self._planned_cast = list(dict.fromkeys(str(item["name"]).strip() for item in cast))
        print(
            f"[QWEN] story_bible cast={len(cast)} names={self._planned_cast}",
            flush=True,
        )
        return bible

    @staticmethod
    def _format_story_bible(bible: dict | None) -> str:
        if not isinstance(bible, dict) or not bible.get("beats"):
            return ""
        lines = [
            "STORY BIBLE (your own plan: keep these exact names, never contradict it, and write one paragraph per beat):",
            "CHOSEN CONCEPT (the specific reading to write; avoid the obvious version: "
            + str(bible.get("obvious_version", "")) + "): " + str(bible.get("chosen_concept", "")),
            "CAST (every person below appears on screen and acts):",
        ]
        for person in bible.get("cast", []) or []:
            lines.append(
                f"- {person.get('name', '')}: {person.get('role', '')}. Wants: {person.get('want', '')}. "
                f"Flaw or secret: {person.get('flaw_or_secret', '')}. Toward the protagonist: {person.get('stance_toward_protagonist', '')}."
            )
        lines.append("CAST REASON: " + str(bible.get("cast_rationale", "")))
        lines.append("CONFLICT: " + str(bible.get("central_conflict", "")))
        lines.append("PLANTED EARLY, PAID LATE: " + str(bible.get("planted_detail", "")))
        lines.append("TURN (paragraph 4): " + str(bible.get("turn", "")))
        lines.append("CHOICE AND PRICE: " + str(bible.get("choice_and_price", "")))
        lines.append("ENDING IMAGE: " + str(bible.get("ending_image", "")))
        lines.append("SCENE BEATS:")
        for index, beat in enumerate(bible.get("beats", []) or [], start=1):
            lines.append(f"{index}. {beat}")
        return "\n".join(lines)

    def _sampling_for_mode(
        self,
        mode: str,
    ) -> tuple[float, float]:

        if mode in {AI_STORY_MODE, EXPAND_USER_STORY_MODE}:
            return (0.60, 0.95)

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
            0.60,
            0.90,
        )

    @staticmethod
    def _character_extraction_json_schema() -> dict:
        """Bound redundant character metadata without changing the planner contract.

        The extractor emits positive production-character candidates. Keep
        ``is_character`` required because the existing planner uses it as an
        explicit semantic gate; the prompt requires it to be true for every
        emitted candidate. The cast and spoken-dialogue arrays remain open to
        the existing high limits so the schema does not impose a cast/dialogue
        ceiling as a quality shortcut.
        """
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": 128,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "minLength": 1, "maxLength": 96},
                            "entity_type": {
                                "type": "string",
                                "enum": ["PERSON", "CHARACTER", "SENTIENT"],
                            },
                            "is_character": {"type": "boolean"},
                            "aliases": {
                                "type": "array",
                                "maxItems": 2,
                                "items": {"type": "string", "maxLength": 96},
                            },
                            "identity_type": {
                                "type": "string",
                                "enum": [
                                    "named_character",
                                    "relational_character",
                                    "descriptive_character",
                                ],
                            },
                            "relationship_to": {"type": "string", "maxLength": 96},
                            "relationship": {"type": "string", "maxLength": 64},
                        },
                        "required": ["name", "entity_type", "is_character", "identity_type"],
                        "additionalProperties": False,
                    },
                },
                "spoken_dialogue": {
                    "type": "array",
                    # Preserve the existing dialogue contract; do not cap it at 32.
                    "maxItems": 128,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "speaker": {"type": "string", "maxLength": 96},
                        },
                        "required": ["text", "speaker"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["candidates", "spoken_dialogue"],
            "additionalProperties": False,
        }

    def extract_character_entities(
        self,
        story: str,
        deterministic_candidates: list[str] | None = None,
    ) -> dict:
        """Use the loaded Qwen model as the semantic character authority.

        Qwen reads the completed story and returns the production character roster.
        The planner does only bounded validation/canonicalization afterward; it does
        not discover or invent additional characters from regex/prose heuristics.
        """
        story = str(story or "").strip()
        if not story:
            return {"candidates": [], "spoken_dialogue": []}

        self._character_semantic_calls += 1
        if self._character_semantic_calls > 1:
            raise RuntimeError("Character semantic Qwen call budget exceeded (max 1).")

        candidate_hints = [
            str(value).strip()
            for value in (deterministic_candidates or [])
            if str(value).strip()
        ]

        system_prompt = textwrap.dedent("""
    You are a strict character/production-character extraction component for a cinematic production planner.
    Return JSON only. The `candidates` array is a production-character allow-list, not a generic entity inventory.
    Emit only actual production-character candidates: do not enumerate locations, objects, organizations, events,
    UI/status labels, or other rejected surfaces as `is_character=false` records. Every emitted candidate MUST have
    `is_character=true` and `entity_type` PERSON, CHARACTER, or SENTIENT.

    Keep the payload compact without omitting actual characters or spoken dialogue: use a short canonical name,
    omit aliases when none are needed and include at most two short grounded aliases, and emit relationship fields
    only for a `relational_character`. Do not put explanations or evidence sentences in metadata fields. Complete
    the full candidate roster before emitting `spoken_dialogue`.

    Analyze the supplied story and identify stable human/sentient identities that can receive an identity lock.
    There are THREE valid character identity types:

    1) named_character: a proper/named character grounded directly in the supplied story.
    2) relational_character: a persistent unnamed character whose identity is grounded by a named story
       character and a concrete grounded relationship, expressed as a canonical relational identity.
    3) descriptive_character: a persistent unnamed person whose identity is grounded by a distinctive,
       recurring description, such as "the man in the suit" or "the woman with piercing eyes".

    CHARACTER COUNT IS OPEN. Return every recurring production character actually created by the story,
    whether the final cast contains one person, two people, or many. Do not impose a numeric cast limit.
    For a newly created AI Story, invent and return the canonical names the story itself uses. For Expand Story,
    preserve every supplied source anchor and also return any new character that the expanded narrative genuinely
    introduces and makes consequential. Do not suppress a valid new character merely because it was not in the source.

    Bare generic role labels are NOT canonical identities: "man", "woman", "boy", "girl", "person",
    "doctor", "scientist", "guard", "officer", and similar labels must not be returned as a canonical
    identity. A descriptive_character must contain a distinguishing description beyond the bare role.
    Reject pronouns, contractions, sentence fragments, ordinary prose tokens, UI/status words, and other
    non-identity surfaces as character candidates.
    Every emitted candidate must have `is_character=true` and use entity_type PERSON, CHARACTER, or SENTIENT.
    Do not use EVENT, LOCATION, OBJECT, ROLE, or OTHER for a character identity.
    Relationship surfaces such as "father", "uncle", "his father", or "the man" are semantic references,
    not new canonical entities unless the story establishes a persistent relational/descriptive identity.

    Do NOT invent a relationship or identity. For a relational_character, relationship_to MUST name a canonical
    character actually grounded in the story, relationship MUST be one concrete family/role relation, and the
    supplied story must contain enough evidence to support that link. Prefer a canonical name in the form
    "<Canonical Character>'s <relationship>" and include grounded surface forms in aliases.
    For a descriptive_character, do not invent a relationship; preserve the most specific stable descriptor
    actually grounded in the story and include generic surface forms only as aliases.

    Do NOT treat locations, organizations, facilities, projects, missions, events, objects, calendar words, or
    weather as characters. Titles such as Dr., Captain, Commander, etc. are not part of the canonical name.
    A person mentioned only as a historical incident, dead/missing subject, past researcher, archival reference,
    name on a container/document/photograph, or other backstory-only identity is NOT a production character
    unless that person is physically present in the story and takes meaningful action.
    Interrogative/function words such as "Why", "When", "Where", and "How" are never character names.
    Source anchors are preservation requirements, not a whitelist of permitted identities.
    Do not promote a weak textual surface into a canonical character merely because it is capitalized.
    Recover every stable named, relational, and descriptive identity actually created by the story.

    SPOKEN DIALOGUE SEMANTICS:
    Also return `spoken_dialogue` for direct speech actually spoken by a physically present production
    character in the supplied story. Each item must contain the exact spoken text and the speaker's
    canonical character name (or the grounded character surface when the character is descriptive/relational).
    Include quoted dialogue and explicit screenplay-style character lines only when they are genuinely spoken
    by that person. Exclude logs, notes, signs, labels, photographs, recordings, transmissions, alarms,
    terminal/computer output, narration, internal thoughts, remembered speech, and other text that is merely
    displayed, transmitted, reported, or read. Quotation marks alone do not make text spoken dialogue.
    Do not paraphrase, merge, split, or invent dialogue. The `text` field must match the source wording closely
    enough for deterministic exact-span reconciliation after this call. If there is no direct spoken dialogue,
    return an empty `spoken_dialogue` array.
    Every `spoken_dialogue` item must be text that sits inside quotation marks in the story or on an explicit
    `Name: line` screenplay label. Narration, description, and action sentences are never dialogue, even when a
    character is the subject. If the story has no quotation marks and no labelled lines, `spoken_dialogue` must be empty.

    PRESENCE RULE (apply before excluding anyone):
    A person IS a production character when they are physically present in ANY scene of the story: they act, speak in
    quotation marks, are addressed face to face, are touched, followed, or chased, or otherwise take part in the present
    action. A former title, earlier job, or earlier history ("the station's former lead scientist", "a retired pilot")
    only describes a person who is present now; it does not make them backstory-only. Exclude a person only when they NEVER
    appear in a present scene of the story.
    Every speaker named in `spoken_dialogue` MUST also appear in `candidates`.
    `deterministic_character_hints` are surface forms found by a simple scan and some are wrong: classify each one and
    include it, under its fullest grounded name, when that person is physically present.
    `screenwriter_planned_cast` (when supplied) lists the people the story's writer intended on screen: include each one
    who actually appears in the final story text.
    """).strip()

        user_payload = json.dumps(
            {
                "story": self._compact_story_context(story, DIRECTOR_STORY_CONTEXT_CHARS),
                "deterministic_character_hints": candidate_hints[:32],
                "screenwriter_planned_cast": [
                    str(value).strip()
                    for value in (getattr(self, "_planned_cast", None) or [])
                    if str(value).strip()
                ][:16],
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
        result = self._complete_semantic_roster(story, result)

        # Raw character-extractor payloads are intentionally hidden from normal production logs.
        # Use an explicit TRACE value only when low-level debugging is actually requested.
        if os.getenv("H3_DEBUG_CHARACTERS", "0").strip().lower() == "trace":
            print("\n[CHARACTER QWEN RAW RESULT]")
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

        self._semantic_spoken_dialogue = [
            {
                "text": str(item.get("text", "") or "").strip(),
                "speaker": str(item.get("speaker", "") or "").strip(),
            }
            for item in (result.get("spoken_dialogue", []) or [])
            if isinstance(item, dict)
            and str(item.get("text", "") or "").strip()
            and str(item.get("speaker", "") or "").strip()
        ]

        self._semantic_spoken_dialogue = self._anchor_semantic_dialogue_to_story(
            story,
            self._semantic_spoken_dialogue,
        )

        return result

    def _complete_semantic_roster(self, story: str, result: dict) -> dict:
        """Keep the extractor consistent with itself and with the story's own writer.

        A name that Qwen itself reports as a speaker, or that the story-planning pass
        listed as on-screen cast, must be offered to the planner as a candidate. This
        adds no name of our own: both sources are Qwen output, and the planner's
        grounding, mention-only and non-person checks still decide acceptance.
        """
        if not isinstance(result, dict):
            return result

        title_re = re.compile(
            r"^(?:dr|doctor|mr|mrs|ms|miss|prof|professor|captain|commander|detective|agent)\.?\s+",
            flags=re.IGNORECASE,
        )

        def tokens(value: str) -> frozenset:
            return frozenset(re.findall(r"[a-z0-9]+", title_re.sub("", str(value or "").strip()).lower()))

        candidates = [item for item in (result.get("candidates") or []) if isinstance(item, dict)]
        known = [
            tokens(surface)
            for item in candidates
            for surface in [item.get("name", ""), *(item.get("aliases", []) or [])]
            if str(surface or "").strip()
        ]
        known = [value for value in known if value]

        wanted = list(getattr(self, "_planned_cast", None) or [])
        wanted += [
            str(item.get("speaker", "") or "")
            for item in (result.get("spoken_dialogue") or [])
            if isinstance(item, dict)
        ]

        added: list[str] = []
        for name in wanted:
            name = str(name or "").strip()
            if not name or len(name) > 96 or not name[:1].isupper():
                continue
            wanted_tokens = tokens(name)
            if not wanted_tokens or any(
                wanted_tokens <= value or value <= wanted_tokens for value in known
            ):
                continue
            candidates.append({
                "name": name,
                "entity_type": "PERSON",
                "is_character": True,
                "identity_type": "named_character",
            })
            known.append(wanted_tokens)
            added.append(name)

        if added:
            print(f"[DIRECTOR] roster completion: Qwen speakers/planned cast re-offered {added}", flush=True)
        result["candidates"] = candidates
        return result

    def _anchor_semantic_dialogue_to_story(
        self,
        story: str,
        items: list[dict],
    ) -> list[dict]:
        """Keep only semantic dialogue items anchored to real speech in the story.

        An item is anchored when its text equals a quote-delimited span or an explicit
        `Name: line` label. This is the same exact-span test the dialogue reconciler applies
        later, so no valid dialogue is lost; unanchored narration sentences that the
        extractor mislabelled as speech are dropped before they can trigger a recovery.
        No word lists are used.
        """
        text = str(story or "")
        quote_pattern = re.compile(
            r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)',
            flags=re.UNICODE,
        )
        label_pattern = re.compile(
            r"(?m)^\s*([A-Z][A-Za-z0-9.'’\-]*(?:\s+[A-Z][A-Za-z0-9.'’\-]*){0,4})\s*(?::|—|–)\s*([^\n]+?)\s*$"
        )
        anchors: set[str] = set()
        for match in quote_pattern.finditer(text):
            key = self._dialogue_anchor_key(next((p for p in match.groups() if p), ""))
            if key:
                anchors.add(key)
        for match in label_pattern.finditer(text):
            display = self._normalize_dialogue_text(match.group(2), keep_case=True).rstrip(",;: ").strip()
            key = self._dialogue_anchor_key(display)
            if key:
                anchors.add(key)
        return [
            item
            for item in (items or [])
            if self._dialogue_anchor_key(str(item.get("text", "") or "")) in anchors
        ]

    def _shot_json_schema(
        self,
        min_items: int | None = None,
        max_items: int | None = None,
    ) -> dict:
        if min_items is None:
            min_items = self.SHOTS_PER_SCENE
        if max_items is None:
            max_items = self.SHOTS_PER_SCENE
        shot_properties = {
            "shot_id": {"type": "string"},
            "scene_id": {"type": "string"},
            "duration_seconds": {"type": "number"},
            "characters": {
                "type": "array",
                "items": {"type": "string"},
            },
            "location": {"type": "string", "minLength": 1},
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
        shot_schema = self._shot_json_schema(
            self.SHOTS_PER_SCENE,
            self.SHOTS_PER_SCENE,
        )["properties"]["shots"]["items"]
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
        """Compact by paragraph so every narrative turn remains represented."""
        value = str(story or "").strip()
        if len(value) <= max_chars:
            return value

        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n+", value)
            if part.strip()
        ]
        if not paragraphs:
            return value[:max_chars].rstrip() + "…"

        selected: list[str] = []
        remaining = max_chars
        separators_left = len(paragraphs) - 1

        for index, paragraph in enumerate(paragraphs):
            sentences = [
                part.strip()
                for part in re.split(r"(?<=[.!?])\s+", paragraph)
                if part.strip()
            ]
            if not sentences:
                continue

            first = sentences[0]
            last = sentences[-1]
            snippet = first if len(sentences) == 1 else f"{first} {last}"
            prefix = f"P{index + 1}: " if len(paragraphs) <= 12 else ""
            budget = max(40, (remaining - separators_left * 2) // max(1, len(paragraphs) - index))
            available = max(20, budget - len(prefix))

            if len(snippet) > available:
                if len(sentences) == 1:
                    snippet = snippet[: max(1, available - 1)].rstrip() + "…"
                else:
                    side = max(8, (available - 3) // 2)
                    snippet = first[:side].rstrip() + " … " + last[-side:].lstrip()
                    if len(snippet) > available:
                        snippet = snippet[: max(1, available - 1)].rstrip() + "…"

            candidate = prefix + snippet
            separator = "\n\n" if selected else ""
            needed = len(separator) + len(candidate)
            if needed > remaining:
                # The minimum budget should make this rare; still preserve a visible
                # fragment for the paragraph instead of dropping it completely.
                available = max(1, remaining - len(separator))
                candidate = (prefix + snippet)[:available].rstrip()
                if available > 1 and len(prefix + snippet) > available:
                    candidate = candidate[:-1].rstrip() + "…"
                needed = len(separator) + len(candidate)

            selected.append(candidate)
            remaining -= needed
            separators_left = max(0, separators_left - 1)

        compact = "\n\n".join(selected).strip()
        return compact or (value[:max_chars].rstrip() + "…")


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

            entry = {"name": name}
            role_value = str(item.get("role", "") or "").strip()
            # A role identical to the name (or the generic placeholder) carries no information.
            if role_value and role_value.lower() not in {name.lower(), "story character"}:
                entry["role"] = role_value
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

        canonical_names = {
            str(c.get("name")).strip().lower(): str(c.get("name")).strip()
            for c in compact_characters
        }
        # Placeholder values emitted by the deterministic scene builder. They describe nothing
        # and some are actively wrong (weather "natural", mood "cinematic"), so they are omitted
        # rather than sent as if they were story facts.
        placeholder_values = {
            "unspecified time", "natural", "cinematic", "cinematic naturalistic lighting",
            "cinematic depth", "stable environmental continuity",
        }

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
                "atmosphere": (
                    ""
                    if "natural cinematic environmental ambience" in str(scene.get("atmosphere", "") or "")
                    else self._limit_text(
                        scene.get("atmosphere", ""),
                        DIRECTOR_SHOT_SCENE_ATMOSPHERE_CHARS,
                    )
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
                "environment_details": [
                    detail
                    for detail in self._clean_list(
                        scene.get("environment_details", []),
                        limit=4,
                    )
                    if str(detail).strip().lower() not in placeholder_values
                ],
                "key_props": self._clean_list(
                    scene.get("key_props", []),
                    limit=4,
                ),
                "characters": [
                    canonical_names.get(str(name).strip().lower(), str(name).strip())
                    for name in self._clean_list(
                        scene.get("characters", []),
                        limit=6,
                    )
                ],
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
                and not (isinstance(value, str) and value.strip().lower() in placeholder_values)
            }

            # scene_objective / obligatory_moment are truncated copies of `description` when the
            # planner had nothing better. Sending them again triples the prompt for no signal.
            description_key = re.sub(r"\W+", " ", str(scene.get("description", "") or "").lower()).strip()
            for duplicate_key in ("scene_objective", "obligatory_moment"):
                duplicate_value = re.sub(
                    r"\W+", " ", str(scene_payload.get(duplicate_key, "") or "").lower().replace("…", "")
                ).strip()
                if duplicate_value and description_key.startswith(duplicate_value[:120]):
                    scene_payload.pop(duplicate_key, None)

            # Scene-level character binding is a candidate context, not a prerequisite.
            # The full canonical Qwen-approved roster is supplied separately above,
            # so an empty scene binding must still reach the shot director. Qwen may
            # select visible characters from that closed roster; _sanitize_shots()
            # and the final shot-character contract remain authoritative afterwards.
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

        # H3_SCENE_BOUND_DIALOGUE_V2_1: keep the established semantic extractor and
        # its non-fatal fallback contract, but bind each exact utterance to one scene.
        source_dialogue = []
        extract_dialogue = getattr(self, "_extract_story_spoken_segments", None)
        try:
            if callable(extract_dialogue):
                source_dialogue = [
                    segment
                    for segment in extract_dialogue(
                        story,
                        getattr(self, "_semantic_spoken_dialogue", None),
                        allowed_speakers=[
                            str(c.get("name", "")).strip()
                            for c in compact_characters
                            if isinstance(c, dict) and str(c.get("name", "")).strip()
                        ],
                        speaker_aliases=EntityResolver.build_character_alias_map(characters),
                    )
                    if isinstance(segment, dict)
                    and str(segment.get("display", "") or segment.get("text", "") or "").strip()
                ]

            def _dialogue_phrase_key(value):
                return re.sub(
                    r"[^a-z0-9]+", " ", str(value or "").casefold()
                ).strip()

            # Canonical speaker map includes the approved roster and its aliases.
            _canonical_by_norm = {
                EntityResolver.normalize(str(item.get("name", ""))): str(item.get("name", "")).strip()
                for item in characters
                if isinstance(item, dict) and str(item.get("name", "")).strip()
            }
            _alias_map = EntityResolver.build_character_alias_map(characters)

            # Do not leave stale source_dialogue entries on reused scene dictionaries.
            for _scene_payload in scene_payloads:
                if isinstance(_scene_payload, dict):
                    _scene_payload["source_dialogue"] = []

            # The shot scenes are derived from the six source-story paragraphs. Match the
            # exact semantic utterance to the FULL scene description, not compact story_context.
            _scene_text_by_id = {}
            for _scene in scenes:
                if not isinstance(_scene, dict):
                    continue
                _scene_id = str(_scene.get("scene_id", "") or "").strip()
                _scene_text = str(_scene.get("description", "") or _scene.get("story_summary", "") or "")
                if _scene_id and _scene_text:
                    _scene_text_by_id[_scene_id] = _scene_text

            _used_phrases = {}
            _scene_cursor = 0
            _unmapped_speakers = set()
            _unmatched_utterances = 0
            _full_story_key = _dialogue_phrase_key(story)
            for _segment in source_dialogue:
                _display = str(_segment.get("display", "") or _segment.get("text", "") or "").strip()
                _phrase = _dialogue_phrase_key(_display)
                _speakers = [
                    str(_value).strip()
                    for _value in (_segment.get("source_speakers", set()) or set())
                    if str(_value).strip()
                ]
                # Never guess if semantic source attribution is ambiguous.
                if not _phrase or len(_speakers) != 1:
                    continue
                _speaker_key = EntityResolver.normalize(_speakers[0])
                _canonical_speaker = _canonical_by_norm.get(_speaker_key)
                if _canonical_speaker is None:
                    _alias_target = _alias_map.get(_speaker_key)
                    if _alias_target:
                        _canonical_speaker = _canonical_by_norm.get(EntityResolver.normalize(_alias_target))
                if _canonical_speaker is None:
                    _unmapped_speakers.add(_speakers[0])
                    continue

                _attached = False
                for _scene_index in range(_scene_cursor, len(scene_payloads)):
                    _scene_payload = scene_payloads[_scene_index]
                    if not isinstance(_scene_payload, dict):
                        continue
                    _scene_id = str(_scene_payload.get("scene_id", "") or "").strip()
                    _raw_scene_text = _scene_text_by_id.get(
                        _scene_id, str(_scene_payload.get("description", "") or "")
                    )
                    _scene_key = _dialogue_phrase_key(_raw_scene_text)
                    _available = (" " + _scene_key + " ").count(" " + _phrase + " ")
                    _use_key = (_scene_id, _phrase)
                    _already_used = _used_phrases.get(_use_key, 0)
                    if _available <= _already_used:
                        continue

                    _scene_payload["source_dialogue"].append({
                        "speaker": _canonical_speaker,
                        "text": _display,
                    })
                    _used_phrases[_use_key] = _already_used + 1
                    _scene_cursor = _scene_index
                    _attached = True
                    break

                if not _attached:
                    # A line from another scene is expected to be absent from this
                    # two-scene batch. Count only a real source-mapping miss, not a
                    # correctly attributed utterance belonging to a different batch.
                    if (" " + _full_story_key + " ").count(" " + _phrase + " ") > 0:
                        continue
                    _unmatched_utterances += 1

            if _unmapped_speakers:
                print(
                    "[DIRECTOR] scene-bound dialogue: speaker absent from canonical roster; "
                    "line excluded from shot whitelist: " + repr(sorted(_unmapped_speakers)),
                    flush=True,
                )
            if _unmatched_utterances:
                print(
                    f"[DIRECTOR] scene-bound dialogue: {_unmatched_utterances} utterance(s) "
                    "did not match a source scene description",
                    flush=True,
                )
        except Exception as exc:
            recorder = getattr(self, "_record_recovery", None)
            if callable(recorder):
                recorder(
                    "shot_director_dialogue_extraction",
                    f"source dialogue extraction failed: {exc}",
                )
            else:
                print(
                    f"[DIRECTOR] source dialogue extraction failed: {exc}",
                    flush=True,
                )
            source_dialogue = []
            for _scene_payload in scene_payloads:
                if isinstance(_scene_payload, dict):
                    _scene_payload["source_dialogue"] = []

        payload = {
            "story_context": self._compact_story_context(story, DIRECTOR_SHOT_STORY_CONTEXT_CHARS),
            "characters": compact_characters,
            "visual_language": language,
            "reference_visual_analysis": visual_context,
            "scenes": scene_payloads,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

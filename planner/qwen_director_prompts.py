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
    DIRECTOR_SHOT_TEMPERATURE,
    DIRECTOR_SHOT_TOP_P,
    DIRECTOR_STORY_THINKING_TEMPERATURE,
    DIRECTOR_STORY_THINKING_TOP_P,
    DIRECTOR_STORY_WRITER_TEMPERATURE,
    DIRECTOR_STORY_WRITER_TOP_P,
    DIRECTOR_SHOT_STORY_CONTEXT_CHARS,
    DIRECTOR_SHOT_SCENE_DESCRIPTION_CHARS,
    DIRECTOR_SHOT_SCENE_OBJECTIVE_CHARS,
    DIRECTOR_SHOT_SCENE_CONTINUITY_CHARS,
    DIRECTOR_SHOT_SCENE_ATMOSPHERE_CHARS,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    DIRECTOR_STORY_SCENES,
    DIRECTOR_SHOTS_PER_SCENE,
    STORY_WORDS_PER_SCENE,
    STORY_TARGET_WORDS,
    STORY_MIN_WORDS,
    H3_FPS,
    H3_FRAMES_PER_SHOT,
)
from planner.entity_resolver import EntityResolver


# Canonical camera vocabulary shared by the prompt schema, sanitizer, and compiler.
import re

CAMERA_SHOT_OPTIONS = (
    "extreme wide",
    "wide",
    "full",
    "medium wide",
    "medium",
    "medium close-up",
    "close-up",
    "extreme close-up",
    "over-the-shoulder",
    "two-shot",
    "pov",
    "insert",
)

CAMERA_MOVEMENT_OPTIONS = (
    "static",
    "pan",
    "tilt",
    "dolly",
    "tracking",
    "handheld",
    "crane",
    "push-in",
    "orbit",
)

DEFAULT_CAMERA_SHOT = "medium wide"
DEFAULT_CAMERA_MOVEMENT = "static"


def _camera_key(value: object) -> str:
    text = str(value or "").strip().casefold()
    text = text.replace("‐", "-").replace("‑", "-").replace("‒", "-")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.!;,]+$", "", text).strip()
    return text


def normalize_camera_shot(value: object) -> tuple[str, str]:
    """Return (canonical value, status) where status is unchanged/normalized/fallback."""
    raw = _camera_key(value)
    aliases = {
        "wide establishing shot": "extreme wide",
        "establishing shot": "extreme wide",
        "extreme wide shot": "extreme wide",
        "wide shot": "wide",
        "full shot": "full",
        "medium wide shot": "medium wide",
        "medium shot": "medium",
        "medium close up": "medium close-up",
        "medium closeup": "medium close-up",
        "medium close-up shot": "medium close-up",
        "close up": "close-up",
        "closeup": "close-up",
        "close-up shot": "close-up",
        "close-up detail": "close-up",
        "extreme close up": "extreme close-up",
        "extreme closeup": "extreme close-up",
        "extreme close-up shot": "extreme close-up",
        "over the shoulder": "over-the-shoulder",
        "over-the-shoulder shot": "over-the-shoulder",
        "two shot": "two-shot",
        "two shot framing": "two-shot",
        "point of view": "pov",
        "p.o.v.": "pov",
        "insert shot": "insert",
    }
    candidate = aliases.get(raw, raw)
    # Remove a generic trailing word only when the remainder is itself canonical.
    if candidate.endswith(" shot") and candidate[:-5].strip() in CAMERA_SHOT_OPTIONS:
        candidate = candidate[:-5].strip()
    if candidate in CAMERA_SHOT_OPTIONS:
        return candidate, "unchanged" if candidate == raw else "normalized"
    return DEFAULT_CAMERA_SHOT, "fallback"


def normalize_camera_movement(value: object) -> tuple[str, str]:
    """Normalize clear camera-movement aliases; unknown values fail closed to static."""
    raw = _camera_key(value)
    aliases = {
        "static shot": "static",
        "no movement": "static",
        "locked off": "static",
        "locked-off": "static",
        "slow pan": "pan",
        "gentle pan": "pan",
        "slow tilt": "tilt",
        "tracking shot": "tracking",
        "camera tracking": "tracking",
        "dolly in": "push-in",
        "dolly forward": "push-in",
        "slow dolly in": "push-in",
        "slow dolly forward": "push-in",
        "slow push in": "push-in",
        "slow push-in": "push-in",
        "slow controlled push-in": "push-in",
        "gentle push-in": "push-in",
        "push in": "push-in",
        "dolly back": "dolly",
        "dolly out": "dolly",
        "slow dolly back": "dolly",
        "slow dolly out": "dolly",
        "pull back": "dolly",
        "camera orbit": "orbit",
        "crane shot": "crane",
        "hand-held": "handheld",
    }
    candidate = aliases.get(raw, raw)
    if candidate.endswith(" shot") and candidate[:-5].strip() in CAMERA_MOVEMENT_OPTIONS:
        candidate = candidate[:-5].strip()
    if candidate in CAMERA_MOVEMENT_OPTIONS:
        return candidate, "unchanged" if candidate == raw else "normalized"
    return DEFAULT_CAMERA_MOVEMENT, "fallback"



SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT = """
You are the CINEMATOGRAPHY DIRECTOR for MiniMax H3.

Create exactly __SHOTS_PER_SCENE__ production-ready shots for EACH supplied scene.

For each `scene_shots` item, return one `scene_metadata` object that is the canonical semantic interpretation of that scene. Qwen is the source of truth for its physical setting, time_of_day, weather, atmosphere, mood, lighting, color_temperature, environment_details, key_props, and dramatic_purpose. Interpret dramatic_purpose from what this specific scene does in the actual story and how it changes the protagonist's situation; write a concise natural-language phrase, not a fixed beat label. Do not infer purpose from scene index or force a conventional setup/catalyst/midpoint/climax sequence. A quiet aftermath, investigation, failed attempt, relationship shift, reveal, transition, or other story-specific function is valid when grounded by the text. Use the story and supplied scene prose, not keyword matching or a fixed vocabulary menu. Make the scene visually specific while preserving story facts. Do not invent a weather change, a new time-of-day event, or a prop that changes the plot. If the story does not establish time or weather and a choice is not strongly implied by the physical scene, use `unspecified`. For indoor scenes, weather can be `unspecified` unless it is visibly affecting the interior. All shots in the same scene must agree with this canonical setting; their per-shot location, lighting, color temperature, and mood must be consistent with `scene_metadata`.

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
location must be a specific physical place (e.g., "harbor fish market", "hospital corridor", "rooftop garden"), never a phrase from the story.
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
- Dialogue spoken only over a radio, intercom, loudspeaker, phone, recording, or screen is off-screen audio, not a visible speaking-character event. Do not bind that remote speaker to a shot as though physically present; omit that line from scene-local `dialogue_events`.
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

Within each scene, use meaningfully different framing and composition. Keep the same underlying story beat, but make the __SHOTS_PER_SCENE__ shot actions complementary (for example: establish the place, the main action, then a reaction or telling detail): show distinct observable facets, reactions, evidence, or state changes supported by the scene text; do not paraphrase the same action twice. If the source supports only one instant, use complementary coverage without inventing a new event.
LIGHT AND PLACE CONTINUITY: all shots of one scene share ONE lighting setup. Give every shot of a scene the same
`lighting` and `color_temperature` strings (same source, direction, and color); vary only framing, lens, movement, and composition.
Choose lighting that the scene's own location, time of day, weather, and props physically motivate (storm daylight, flashlight beam,
emergency lamps, monitor glow, moonlight, firelight, sodium streetlight). The lighting vocabulary below is a spelling guide, NOT a menu to rotate:
never use "practical neon" or "warm tungsten" unless the scene visibly contains neon signs or tungsten lamps, and never use "soft overcast" indoors.
`visual_prompt` must name the setting and the light source so the image generator can reproduce it.

STORY FIDELITY: depict only events, objects, and states that the scene description or story context actually states.
Never add an opening, discovery, collapse, sound, object, or line of speech that the text does not contain (a door described as
locked stays locked; an unnamed object stays unnamed). When a source is short and the same description appears in several scenes, the
scenes are different COVERAGE of the same moment: vary distance, angle, and detail (approach, threshold, close detail, reaction),
never advance the plot beyond what the description states, and never repeat the same action/visual in consecutive scenes.

PURPOSE-AWARE DIRECTING:
Infer the scene's actual dramatic purpose from the story context and its own scene description. Do not assume an early scene is necessarily setup, a middle scene is a midpoint, or the last scene is a finale; follow the story's actual causal and emotional function. Keep the purpose grounded in events that are stated or clearly implied, and never add a story event to make the scene fit a template. The supplied obligatory_moment identifies the central visible event or state that the required shots must cover; it is not permission to invent additional plot.

SHOT / FRAMING VOCABULARY:
framing: extreme wide, wide, full, medium wide, medium, medium close-up, close-up, extreme close-up, over-the-shoulder, two-shot, pov, insert.

CAMERA MOVEMENT VOCABULARY:
movement: static, pan, tilt, dolly, tracking, handheld, crane, push-in, orbit.

LENS / DEPTH OF FIELD:
wide-angle, normal, telephoto, shallow focus, deep focus, selective focus.

COMPOSITION VOCABULARY:
centered, rule of thirds, leading lines, foreground frame, negative space, silhouette, depth layering, subject isolation.

Choose natural, concrete lighting language supported by the actual environment and visible light sources. Do not select a lighting style from a fixed menu. Write framing, movement, lens, and composition terms in lowercase exactly as specified above.

Return JSON only in exactly this structure:

{
  "scene_shots": [
    {
      "scene_id": "scene_001",
      "scene_metadata": {
        "location": "<physical setting>",
        "time_of_day": "unspecified",
        "weather": "unspecified",
        "atmosphere": "<grounded atmosphere>",
        "mood": "<scene mood>",
        "lighting": "<physically motivated light source>",
        "color_temperature": "<scene color temperature>",
        "environment_details": [],
        "key_props": [],
        "dramatic_purpose": "<what this specific scene changes or accomplishes in the story>"
      },
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
    def _story_text_system(
        self,
        mode: str,
    ) -> str:
        """Configured scene-length prose contract with a craft brief; one author call, no retries."""
        return self._fill_story_length(self._story_text_system_raw(mode))

    @staticmethod
    def _fill_story_length(text: str) -> str:
        n = int(DIRECTOR_STORY_SCENES)
        per = int(STORY_WORDS_PER_SCENE)
        seconds = n * int(DIRECTOR_SHOTS_PER_SCENE) * float(H3_FRAMES_PER_SHOT) / float(H3_FPS)
        return (
            text.replace("__N__", str(n))
            .replace("__PER__", str(per))
            .replace("__TOTAL__", str(STORY_TARGET_WORDS))
            .replace("__MINW__", str(STORY_MIN_WORDS))
            .replace("__MINUTES__", f"{seconds / 60:.1f}")
        )

    def _story_text_system_raw(
        self,
        mode: str,
    ) -> str:
        fmt = """
    Format: plain prose in third person past tense, exactly __N__ paragraphs separated
    by blank lines, about __TOTAL__ words in all (never under __MINW__). The story
    becomes a film of about __MINUTES__ minutes, so each paragraph is ONE filmable scene
    of roughly __PER__ words: a specific place, visible action, and a change in the
    situation. Put any spoken line, in quotation marks, inside its scene paragraph,
    never in a paragraph of its own. No headings, labels, camera directions, or
    commentary. Output only the story."""
        if mode == AI_STORY_MODE:
            return textwrap.dedent("""
    You are a screenwriter writing a short film as prose.

    From the premise, invent the whole story yourself: who is in it and how many
    people, their names, their relationships, what each one wants, the setting, the
    conflict, and where it ends. Choose names and details that belong to this
    particular place, era, and culture, and avoid the first names and images that
    come to mind most easily. Use as many characters as the story truly needs.

    Write a complete, emotionally true story. The protagonist's goal is clear early.
    Each event follows from the one before it. One ordinary detail planted early
    pays off in a reversal the reader did not see coming but accepts as inevitable.
    A choice costs the protagonist something. People talk in distinct voices, and
    what they say changes something. The ending shows the consequences instead of
    stopping on mystery. Show action, objects, sounds, and behavior a camera can
    film; reveal feeling through them.
    """).strip() + "\n\n" + textwrap.dedent(fmt).strip()

        if mode == EXPAND_USER_STORY_MODE:
            return textwrap.dedent("""
    You are a screenwriter expanding a story into a full short film told as prose.

    Keep the source's characters, setting, chronology, events, and outcome, and keep
    an existing twist. Invent whatever is missing: connective events, obstacles,
    new people when the story needs them (you decide the final cast, their names, and
    relationships), and consequences. Choose new names and details that belong to
    this story's world.

    Write a complete, emotionally true story. The protagonist's goal is clear early.
    Each event follows from the one before it. One ordinary detail planted early
    pays off in a reversal that feels inevitable in hindsight. A choice costs the
    protagonist something. People talk in distinct voices, and what they say changes
    something. The ending shows the consequences. Show action, objects, sounds, and
    behavior a camera can film; reveal feeling through them.
    """).strip() + "\n\n" + textwrap.dedent(fmt).strip()

        raise ValueError("Preserve Story does not use a story-text pass.")

    def _story_blueprint_system(self, mode: str) -> str:
        """Build a compact causal beat sheet, not a Story Bible or alternate story."""
        n = int(DIRECTOR_STORY_SCENES)
        task_rule = (
            "Invent the cast, names, relationships, setting, conflict, and ending from the premise. "
            "Qwen owns all creative identity decisions; do not impose a fixed cast size."
            if mode == AI_STORY_MODE
            else "Preserve the source's established characters, setting, chronology, key events, and outcome. "
                 "Keep any existing twist; add only connective causality needed to film it."
        )
        return textwrap.dedent(f"""            You are the story architect for a short film, not the prose writer.
            Produce a concise, causally complete blueprint of exactly {n} scenes for the source premise.
            {task_rule}

            Plan for a finished, emotionally credible story: make the protagonist's goal concrete early;
            ensure each scene is caused by a previous choice or event; track who knows what; keep the
            location and important object states consistent; plant one ordinary detail early and pay it
            off in a fair reversal; make a consequential choice cost something; give the ending a clear,
            observable resolution in the final scene. Avoid unrelated subplots and unexplained turns.

            Output exactly {n} non-empty lines, one per scene, numbered `SCENE 01:` through
            `SCENE {n:02d}:` in order. Each line should be one compact beat (roughly 20–35 words) that
            states the visible action, the causal turn, and the important knowledge/object state carried
            forward. No preamble, JSON, character roster, prose paragraphs, or commentary.
        """).strip()

    def _story_blueprint_user(
        self,
        mode: str,
        story: str,
        source_character_names: list[str] | None = None,
    ) -> str:
        source_text = self._compact_story_context(story, DIRECTOR_STORY_CONTEXT_CHARS)
        result = "MODE: " + str(mode) + "\n\nSOURCE STORY / PREMISE:\n" + source_text
        if mode == EXPAND_USER_STORY_MODE:
            anchors = list(dict.fromkeys(
                str(value).strip()
                for value in (source_character_names or [])
                if str(value).strip()
            ))
            if anchors:
                result += "\n\nSOURCE CHARACTER ANCHORS (preserve; not a cast limit):\n" + ", ".join(anchors[:16])
        result += (
            f"\n\nCreate the {DIRECTOR_STORY_SCENES}-scene causal blueprint only. "
            "Every scene must change the situation and establish what causes the next scene."
        )
        return result

    @staticmethod
    def _normalize_story_blueprint(raw: str, expected_scenes: int | None = None) -> str:
        """Normalize scene-label formatting without regenerating or inventing beats."""
        count = int(expected_scenes or DIRECTOR_STORY_SCENES)
        header = re.compile(
            r"^\s*(?:SCENE\s*)?0*(\d{1,2})\s*[:.)\-–—]\s*(.*?)\s*$",
            re.IGNORECASE,
        )
        beats: dict[int, list[str]] = {}
        current: int | None = None
        for raw_line in str(raw or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = header.match(line)
            if match:
                number = int(match.group(1))
                if 1 <= number <= count:
                    if number in beats:
                        raise RuntimeError(
                            f"Story architect repeated scene label {number}; blueprint must contain "
                            f"exactly one beat per scene. No planner retry was attempted."
                        )
                    current = number
                    beats[number] = [match.group(2).strip()]
                else:
                    current = None
            elif current is not None:
                beats[current].append(line)
        if set(beats) != set(range(1, count + 1)):
            missing = sorted(set(range(1, count + 1)) - set(beats))
            raise RuntimeError(
                f"Story architect did not return all {count} numbered scene beats "
                f"(missing {missing}). No planner retry was attempted."
            )
        normalized = []
        for number in range(1, count + 1):
            text = " ".join(part for part in beats[number] if part).strip()
            if not text:
                raise RuntimeError(
                    f"Story architect returned an empty beat for scene {number}. No planner retry was attempted."
                )
            normalized.append(f"SCENE {number:02d}: {text}")
        return "\n".join(normalized)

    def _story_text_user(
        self,
        mode: str,
        story: str,
        source_character_names: list[str] | None = None,
        story_blueprint: str | None = None,
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
        if str(story_blueprint or "").strip():
            result += (
                "\n\nCAUSAL SCENE BLUEPRINT (structural reference; do not reproduce these labels):\n"
                + str(story_blueprint).strip()
                + "\n\nExpand every beat into vivid, natural filmable prose. Preserve its cause-and-effect, "
                  "character knowledge, and object/location continuity. Do not explain the blueprint or add headings."
            )
        result += "\n\nReturn only the completed story: " + str(DIRECTOR_STORY_SCENES) + " scene paragraphs, about " + str(STORY_TARGET_WORDS) + " words in total, never under " + str(STORY_MIN_WORDS) + " words, ending on a complete sentence with the resolution."
        return result


    def _sampling_for_mode(
        self,
        mode: str,
    ) -> tuple[float, float]:
        if mode in {AI_STORY_MODE, EXPAND_USER_STORY_MODE}:
            # The prose author runs non-thinking after the bounded thinking architect pass.
            return (
                float(DIRECTOR_STORY_WRITER_TEMPERATURE),
                float(DIRECTOR_STORY_WRITER_TOP_P),
            )
        if mode == PRESERVE_USER_STORY_MODE:
            return (0.10, 0.80)
        return (DIRECTOR_TEMPERATURE, DIRECTOR_TOP_P)

    def _shot_sampling(
        self,
    ) -> tuple[float, float]:
        """Return the shot-batch sampler configured in runtime_versions.yaml."""
        return (
            float(DIRECTOR_SHOT_TEMPERATURE),
            float(DIRECTOR_SHOT_TOP_P),
        )

    @staticmethod
    def _character_extraction_json_schema() -> dict:
        """Let Qwen return the complete cast without per-name or cast-count rules.

        The candidates array has no minItems/maxItems cast quota. Output tokens and
        the normal planner validations still bound the response operationally.
        """
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
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

        # Keep deterministic_candidates for caller compatibility only. Do not send
        # them to Qwen for per-name verdicts and never use them to add cast members.
        # The final prose itself is the complete source for Qwen's roster decision.
        _ = deterministic_candidates
        system_prompt = textwrap.dedent("""
    You are Qwen's character and dialogue extractor for a film-production planner. Return JSON only.

    Read the COMPLETE final story from its first paragraph to its last. Determine the actual production cast from the story itself. `candidates` is the set of people who physically participate in present-time story events: people who act, speak in person, oppose/help another character, touch or rescue someone, wake up and interact, enter or leave, or otherwise take part in an event shown happening now. Include a person first introduced as missing or in backstory if that person later physically appears and acts in present events. Do not stop after the protagonist.

    Review every paragraph for distinct people performing or receiving a present-time physical action, then merge aliases for the same person. Qwen alone decides the identities and number of production characters. There is no minimum cast target beyond the protagonist required for a story, no preference for one-person or multi-person stories, and no cast-count quota. Return the actual cast that the final prose depicts. Do not add people merely because a planning outline suggested them. Do not include people who appear only in memories, documents, photographs, recordings, logs, signs, or reported history. Do not treat places, objects, organizations, weather, labels, or generic role words as people.

    Use grounded canonical names and aliases from the story. Never invent names, relationships, or characters. Do not use a deterministic hint list or a planning cast as a separate source of truth. The `candidates` array itself is your final cast decision; no downstream step will add names from hints, planned cast, or dialogue speakers. A dialogue speaker must therefore also be represented in `candidates` when that person physically participates in the story.

    Return `spoken_dialogue` containing only exact direct speech actually spoken aloud by a physically present character in the current scene. A quotation mark is not sufficient evidence that text was spoken: exclude words read from or written in a note, journal, sign, screen, label, or document; messages or recordings; radio/intercom/speaker/static voices; remembered speech; inner monologue; and a voice described as echoing inside someone's mind. Exclude a line when the narration only repeats or recalls what someone said earlier. Every included speaker must both physically participate in the current story and exactly match one candidate's canonical name. Preserve exact utterance text; do not paraphrase or invent speech.

    Return exactly these top-level fields: `candidates` and `spoken_dialogue`. Before responding, re-read the complete story and check that every distinct person who physically participates in its current events is represented once, while mention-only and off-screen people are excluded. This is a semantic review of the story, not a requirement to classify any externally supplied names. Keep the JSON compact and output no explanation outside it.
    """).strip()

        user_payload = json.dumps(
            {"story": self._compact_story_context(story, DIRECTOR_STORY_CONTEXT_CHARS)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        extraction_schema = self._character_extraction_json_schema()

        result = self._chat_json(
            system_prompt,
            user_payload,
            minimum_completion=96,
            temperature=0.05,
            top_p=0.70,
            call_name="character_entity_extraction",
            max_completion=1536,
            json_mode=True,
            disable_thinking=True,
            response_schema=extraction_schema,
        )
        if not isinstance(result, dict):
            result = {"candidates": [], "spoken_dialogue": []}
        # Never supplement candidates from hints, the planning outline, or dialogue
        # speaker names. Qwen's `candidates` response is the roster authority.
        result = {
            "candidates": [item for item in (result.get("candidates") or []) if isinstance(item, dict)],
            "spoken_dialogue": [item for item in (result.get("spoken_dialogue") or []) if isinstance(item, dict)],
        }
        print(
            f"[DIRECTOR] Qwen cast candidates={len(result['candidates'])} "
            f"dialogue_events={len(result['spoken_dialogue'])}",
            flush=True,
        )

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
            "camera_shot": {"type": "string", "enum": list(CAMERA_SHOT_OPTIONS), "maxLength": 24},
            "camera_movement": {"type": "string", "enum": list(CAMERA_MOVEMENT_OPTIONS), "maxLength": 12},
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
        scene_metadata_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string", "minLength": 1},
                "time_of_day": {"type": "string"},
                "weather": {"type": "string"},
                "atmosphere": {"type": "string"},
                "mood": {"type": "string"},
                "lighting": {"type": "string", "minLength": 1},
                "color_temperature": {"type": "string"},
                "environment_details": {
                    "type": "array", "maxItems": 4,
                    "items": {"type": "string"},
                },
                "key_props": {
                    "type": "array", "maxItems": 6,
                    "items": {"type": "string"},
                },
                "dramatic_purpose": {"type": "string", "minLength": 1, "maxLength": 180},
            },
            "required": [
                "location", "time_of_day", "weather", "atmosphere",
                "mood", "lighting", "color_temperature",
                "environment_details", "key_props", "dramatic_purpose",
            ],
            "additionalProperties": False,
        }
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
                            "scene_metadata": scene_metadata_schema,
                            "shots": {
                                "type": "array",
                                "minItems": self.SHOTS_PER_SCENE,
                                "maxItems": self.SHOTS_PER_SCENE,
                                "items": shot_schema,
                            },
                        },
                        "required": [
                            "scene_id",
                            "scene_metadata",
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
        """Compact by paragraph while retaining a causal/action-bearing sentence.

        The old first+last-only strategy removed the middle actions that explained
        why a scene changed, leaving shot batches to invent visual bridges. This
        version preserves paragraph boundaries and selects first, last, and the
        strongest interior action/cause sentence when the character budget allows.
        """
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

        action_cues = re.compile(
            r"\b(?:but|because|when|until|after|before|then|instead|therefore|so|"
            r"decided|realized|refused|grabbed|pressed|opened|revealed|triggered|broke|"
            r"sealed|injected|activated|destroyed|fled|escaped|chose|lunged|snapped|"
            r"warned|shouted|slid|turned|reached|failed|collapsed|leaked|mutated|"
            r"stopped|followed|handed|pulled|cut|locked|unlocked|answered|lied|"
            r"admitted|betrayed|offered|threatened|confronted|blocked|struck|switched|"
            r"pocketed|pockets|tucked|dropped|stowed|carried|retrieved|returned|placed|"
            r"removed|held|took|kept|stored|transferred|hid|hidden|recovered|picked)\b",
            re.IGNORECASE,
        )
        quote_cue = re.compile(r"[\"“”‘’]")
        separator_cost = max(0, len(paragraphs) - 1) * 2
        prefix_cost = sum(len(f"P{i + 1}: ") for i in range(len(paragraphs)) if len(paragraphs) <= 12)
        budget_each = max(24, (max_chars - separator_cost - prefix_cost) // len(paragraphs))

        def quote_safe_truncate(text: str, limit: int) -> str:
            """Never leave a quoted dialogue span with an orphaned opening quote."""
            if len(text) <= limit:
                return text
            head = text[:max(0, limit - 1)]
            open_quote = None
            open_at = -1
            for pos, char in enumerate(head):
                if char == '"':
                    if open_quote == '"':
                        open_quote, open_at = None, -1
                    elif open_quote is None:
                        open_quote, open_at = '"', pos
                elif char == '“' and open_quote is None:
                    open_quote = '”'
                    open_at = pos
                elif char == '”' and char == open_quote:
                    open_quote, open_at = None, -1
            if open_quote is not None:
                head = head[:open_at].rstrip()
            return head.rstrip() + "…"

        selected: list[str] = []
        for index, paragraph in enumerate(paragraphs):
            # Protect common honorifics before sentence splitting: without this,
            # "Dr. Lena Voss" can become a standalone first sentence and the
            # compactor may keep "Dr." while dropping the identity that follows.
            protected = re.sub(
                r"\b(Dr|Mr|Mrs|Ms|Prof|Capt|Cmdr|Lt|Sgt|St|Rev|Hon|Sr|Jr)\.(?=\s)",
                lambda match: match.group(1) + "\u2024",
                paragraph,
            )
            sentences = [
                part.strip().replace("\u2024", ".")
                for part in re.split(r"(?<=[.!?])[\"\u201d\u2019']?\s+(?=[A-Z\"\u201c])", protected)
                if part.strip()
            ]
            if not sentences:
                continue
            chosen_indexes = {0, len(sentences) - 1}
            if len(sentences) > 2:
                middles = list(range(1, len(sentences) - 1))
                center = (len(sentences) - 1) / 2
                def sentence_score(i: int) -> tuple[int, float, int]:
                    sentence = sentences[i]
                    score = len(action_cues.findall(sentence)) * 3
                    score += 2 if quote_cue.search(sentence) else 0
                    score += 1 if len(sentence.split()) >= 9 else 0
                    return (score, -abs(i - center), -i)
                best_middle = max(middles, key=sentence_score)
                chosen_indexes.add(best_middle)

            # Keep only complete sentence spans. Never slice through a quoted
            # line or sentence to satisfy a context budget; omission is safer than
            # corrupting speaker text or feeding the shot writer a broken quote.
            chosen = [sentences[i] for i in sorted(chosen_indexes)]
            def total_len(items: list[str]) -> int:
                return sum(len(x) for x in items) + max(0, len(items) - 1)
            if total_len(chosen) > budget_each:
                opening = sentences[0]
                ending = sentences[-1] if len(sentences) > 1 else ""
                middle_candidates = [sentences[i] for i in sorted(chosen_indexes)
                                     if i not in {0, len(sentences) - 1}]
                compact_choice = []
                for sentence in [opening, *middle_candidates, ending]:
                    if not sentence:
                        continue
                    trial = compact_choice + [sentence]
                    if total_len(trial) <= budget_each:
                        compact_choice = trial
                if not compact_choice:
                    compact_choice = [sentence for sentence in sentences if len(sentence) <= budget_each][:1]
                chosen = compact_choice
            snippet = " ".join(chosen)
            prefix = f"P{index + 1}: " if len(paragraphs) <= 12 else ""
            available = max(1, budget_each)
            body_budget = max(1, available - len(prefix))
            if len(snippet) > body_budget:
                snippet = quote_safe_truncate(snippet, body_budget)
            candidate = prefix + snippet
            selected.append(candidate)

        compact = "\n\n".join(selected).strip()
        # Ensure strict cap under unusual paragraph counts or very short caps.
        if len(compact) > max_chars:
            compact = quote_safe_truncate(compact, max_chars)
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
            "unspecified", "unspecified time", "natural", "cinematic", "cinematic naturalistic lighting",
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

            _story_paragraphs = [
                _part.strip()
                for _part in re.split(r"\n\s*\n+", str(story or ""))
                if _part.strip()
            ]

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
                    # The batch receives only one/two scenes, but the story contains every
                    # source scene. First identify the paragraph(s) containing this exact
                    # utterance. If it uniquely belongs to a scene in this batch, attach it
                    # to that scene even if the expanded scene description omitted the line.
                    # If it belongs only to another batch, skip it intentionally. A phrase
                    # found in the story but not assignable to a scene is a real mapping miss
                    # and must not disappear silently.
                    story_paragraph_matches = []
                    for _paragraph_index, _paragraph in enumerate(_story_paragraphs):
                        _paragraph_key = _dialogue_phrase_key(_paragraph)
                        if (" " + _paragraph_key + " ").count(" " + _phrase + " ") > 0:
                            story_paragraph_matches.append(_paragraph_index)

                    batch_matches = []
                    for _scene_payload in scene_payloads:
                        _scene_match = re.search(r"(\d+)$", str(_scene_payload.get("scene_id", "")))
                        if not _scene_match:
                            continue
                        _paragraph_index = int(_scene_match.group(1)) - 1
                        if _paragraph_index in story_paragraph_matches:
                            batch_matches.append((_paragraph_index, _scene_payload))

                    if len(story_paragraph_matches) == 1 and len(batch_matches) == 1:
                        _paragraph_index, _target_scene = batch_matches[0]
                        _use_key = (str(_target_scene.get("scene_id", "") or ""), _phrase)
                        _already_used = _used_phrases.get(_use_key, 0)
                        _target_paragraph_key = _dialogue_phrase_key(_story_paragraphs[_paragraph_index])
                        _available = (" " + _target_paragraph_key + " ").count(" " + _phrase + " ")
                        if _already_used < _available:
                            _target_scene["source_dialogue"].append({
                                "speaker": _canonical_speaker,
                                "text": _display,
                            })
                            _used_phrases[_use_key] = _already_used + 1
                            _scene_cursor = max(_scene_cursor, scene_payloads.index(_target_scene))
                            _attached = True

                    if not _attached:
                        # A uniquely located utterance in a different scene batch is
                        # expected. Other cases remain visible as diagnostic misses.
                        if len(story_paragraph_matches) == 1 and not batch_matches:
                            # Uniquely maps to a different batch; its owner will handle it.
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

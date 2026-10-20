from __future__ import annotations

import json
from copy import deepcopy
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
    STORY_WRITER_WORDS_PER_SCENE,
    STORY_PLAN_BEAT_MIN_WORDS,
    STORY_PLAN_BEAT_MAX_WORDS,
    STORY_TARGET_WORDS,
    STORY_MIN_WORDS,
    DIRECTOR_STORY_CAST_SIZE_POLICY,
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

For each `scene_shots` item, return one `scene_metadata` object that is the canonical semantic interpretation of that scene. Qwen is the source of truth for its physical setting, time_of_day, weather, atmosphere, mood, lighting, color_temperature, environment_details, key_props, and dramatic_purpose. Interpret dramatic_purpose from what this specific scene does in the actual story and how it changes the protagonist's situation; write a concise natural-language phrase, not a fixed beat label. Do not infer purpose from scene index or force a conventional setup/catalyst/midpoint/climax sequence. A quiet aftermath, investigation, failed attempt, relationship shift, reveal, transition, or other story-specific function is valid when grounded by the text. Use the story and supplied scene prose, not keyword matching or a fixed vocabulary menu. Make the scene visually specific while preserving story facts. Do not invent a weather change, a new time-of-day event, or a prop that changes the plot. If the story does not establish time or weather and a choice is not strongly implied by the physical scene, use `unspecified`. For indoor scenes, weather can be `unspecified` unless it is visibly affecting the interior. All shots in the same scene must agree with this canonical setting. The compiler derives each shot's location, lighting and color temperature from `scene_metadata`; use that same place and light source explicitly in each `visual_prompt`. Keep each shot's mood consistent with the scene.

The parent `scene_metadata.location` is the one physical setting for every shot. Never combine outside and interior locations. Do not emit a per-shot location field. Keep shots there unless the source establishes a scene change; prose that moves between places does not imply simultaneous presence in both.

The scenes are part of one coherent film. Use ONLY the supplied characters. Do not create new characters or invent character names.
In `characters` and in every dialogue speaker use the EXACT canonical spelling and capitalization from the supplied `characters` list (never a lower-cased or shortened copy).
List only physically visible characters; story mentions and audio alone do not prove visibility. A terminal/intercom/radio voice does not put its speaker in the room; depict the source only when the story establishes a visible hologram/projection or physical entrance. Distinguish sentient devices from humans.
Keep action 10–30 words, visual_prompt 15–40 words, composition_notes <=18 words, lens_and_depth_of_field <=10 words, mood <=5 words, camera_shot <=5 words, and camera_movement <=5 words. Use the canonical scene-metadata location; never output a separate per-shot location field.
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
- Use only explicit source speech; exclude narration, actions, thoughts, writing, and remembered/archival quotes. `in_person` means the speaker is physically present; `remote` is an attributed live in-world voice whose source stays off-screen.
- `source_dialogue` is the whitelist `{dialogue_id, speaker, text, delivery}`. Assign each ID in `dialogue_assignments` to exactly one shot number (1..__SHOTS_PER_SCENE__); never omit, repeat, or null an ID. The caller restores exact text, canonical speaker, and delivery.
- `dialogue_assignments` is a keyed object: each source ID is a key, so a key must never be copied to another shot or another scene. Use only the exact keys supplied for that specific scene. The schema rejects foreign IDs.
- Do not emit `dialogue_events`, `speaker`, `text`, `speaking_characters`, or `speech_text` inside shot objects. Those are reconstructed by the caller from `dialogue_assignments` and the source whitelist.
- Never invent generic speakers or use noncanonical name spellings. Source identities are authoritative; never use a role label unless it is a supplied canonical identity.
- Assign in-person lines to a shot where the speaker is visible and speaking. For remote lines, show the receiver/channel or listener reaction; keep the source out of visible `characters` unless a projection/hologram or physical entrance is established. Long-line timing is handled downstream; never duplicate an ID.
- Treat shot duration as a hard production constraint: when dialogue is present, leave timing headroom; the compiler finalizes legal duration. Do not put timestamps in the response.

Within each scene, vary framing/composition and use motivated camera movement instead of defaulting every shot to static: tracking for established motion, push-in/dolly for a source-supported reveal, pan/tilt for an established detail, handheld only for grounded instability. Static is valid for observation; never move just to meet a quota. Keep the same underlying story beat, but make the __SHOTS_PER_SCENE__ shot actions complementary (for example: establish the place, the main action, then a reaction or telling detail): show distinct observable facets, reactions, evidence, or state changes supported by the scene text; do not paraphrase the same action twice. If the source supports only one instant, use complementary coverage without inventing a new event.
LIGHT AND PLACE CONTINUITY: `scene_metadata.lighting` and `scene_metadata.color_temperature` are canonical for all shots of this scene. Do not emit duplicated per-shot lighting or color-temperature fields. Every `visual_prompt` must still name the setting and the physically motivated visible light source; never invent neon/tungsten lamps or use soft-overcast outdoor light indoors. The lighting vocabulary below is a spelling guide, NOT a menu to rotate.

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

Return JSON only in this structure. COMPACT OUTPUT CONTRACT: individual shot objects omit `shot_id` and `scene_id` (derived from parent scene and shot-array order); omit per-shot `location`, `lighting`, and `color_temperature` (derived from scene metadata); omit `speaking_characters` and `speech_text` (derived from dialogue references); omit `is_scene_boundary` (derived from scene order); omit `continuity_start_state` and `continuity_end_state` (the production continuity ledger derives these after generation); and omit the six `character_spatial_*` maps (the supplied traces consistently leave them empty). These fields are restored or derived before persistence, validation, and compilation. Do not output them.

{"scene_shots":[{"scene_id":"scene_001","scene_metadata":{"location":"the supplied scene location","time_of_day":"unspecified","weather":"unspecified","atmosphere":"grounded atmosphere","mood":"scene mood","lighting":"visible motivated light source","color_temperature":"neutral","environment_details":[],"key_props":[],"dramatic_purpose":"what this scene changes in the story"},"dialogue_assignments":{},"shots":[{"duration_seconds":5.2,"characters":[],"action":"A visible physical story action changes the situation in a way supported by the source.","camera_shot":"wide","camera_movement":"static","lens_and_depth_of_field":"normal lens","composition_notes":"Clear subject separation.","mood":"tense","visual_prompt":"Grounded cinematic wide view of the supplied location, lit by the specific motivated source established in the scene."},{"duration_seconds":5.2,"characters":[],"action":"A character reacts to the event and reveals a distinct detail grounded in the same scene.","camera_shot":"medium","camera_movement":"push-in","lens_and_depth_of_field":"shallow focus","composition_notes":"Readable foreground and depth.","mood":"tense","visual_prompt":"Closer cinematic framing shows a distinct observable reaction in the same location and motivated light."},{"duration_seconds":5.2,"characters":[],"action":"A supported state change or consequence completes this scene's visible beat without new events.","camera_shot":"close-up","camera_movement":"static","lens_and_depth_of_field":"selective focus","composition_notes":"Isolate the meaningful detail.","mood":"uneasy","visual_prompt":"Cinematic close detail shows the established consequence while keeping character, location, and motivated lighting consistent."}]}]}

There must be exactly __SHOTS_PER_SCENE__ shots inside every scene_shots entry and exactly one entry for every supplied scene. Do not add prose outside JSON.
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
        # Ask for the writer overshoot; validation still uses STORY_MIN_WORDS.
        per = int(STORY_WRITER_WORDS_PER_SCENE)
        seconds = n * int(DIRECTOR_SHOTS_PER_SCENE) * float(H3_FRAMES_PER_SHOT) / float(H3_FPS)
        return (
            text.replace("__N__", str(n))
            .replace("__PER__", str(per))
            .replace("__TOTAL__", str(n * per))
            .replace("__MINW__", str(STORY_MIN_WORDS))
            .replace("__MINUTES__", f"{seconds / 60:.1f}")
        )

    def _story_text_system_raw(
        self,
        mode: str,
    ) -> str:
        fmt = """
    Format: plain prose in third person past tense, exactly __N__ explicitly marked scene blocks,
    about __TOTAL__ story words in all (never under __MINW__). Start each scene on a new line with
    its exact structural tag `[[SCENE_01]]` through `[[SCENE_12]]`, then one space and that scene's
    prose. Use every tag exactly once and in numeric order. The tags are removed by the application;
    they are not story text. Each scene must remain a single paragraph/beat. Do not combine two beats,
    split one beat across tags, or use a sentence-count split to create a missing scene. Every scene
    must contain 6–8 complete sentences; target __PER__ words per scene and never write a short
    3–5-sentence summary in place of a developed beat. Do not move to the next scene marker until
    the current beat contains its action, choice, reaction, consequence, and state transition where
    relevant. Reach the final scene's completed consequence rather than stopping at an unresolved
    future hook. Expand through physical action and causal consequence—not repeated wording.
    The story becomes a film of about __MINUTES__ minutes, so each marked scene is ONE filmable beat
    in one primary physical location: a specific place, visible action, and a change in the situation.
    If the characters physically move to a different location, narrate a clear transition in a different
    scene instead of blending incompatible indoor/outdoor settings. Put any spoken line, in quotation
    marks, inside its own scene prose, never in a separate dialogue paragraph. No other headings,
    camera directions, or commentary. Do not append word counts, checklists, markdown separators,
    summaries, meta-commentary, or a postscript after the final scene. Output only the story blocks."""
        ensemble = f"""
    CAST CONTRACT (mandatory; policy={DIRECTOR_STORY_CAST_SIZE_POLICY}): Qwen determines the cast size from the story's actual needs. There is NO fixed minimum, target, or maximum number of characters. Include the protagonist and every additional character needed to make the conflict, causal turns, relationships, and resolution work; add as many as the story genuinely needs. Do not add characters merely to hit a quota or restrict a strong story to a small ensemble. Qwen chooses every name, identity, relationship, motive, and role; never use preset names or a hard-coded cast. Recurring core characters should physically participate in present-time scenes, take consequential actions, and recur where the plot needs them. A one-scene supporting character is valid when that character materially changes an event. Give important characters distinct wants and points of view. At least two separate scenes should contain direct in-person dialogue between named characters when it fits the premise; those exchanges must change a decision, reveal a conflict, or shift trust. Let the plot turn through character actions and consequences, not only discoveries or records. A person mentioned only in backstory or visible only in a photo, label, document, recording, screen, or anonymous remote voice does NOT count as physically active.
    """
        if mode == AI_STORY_MODE:
            return textwrap.dedent("""
    You are the screenwriter expanding the supplied causal blueprint into a complete short film told as prose. The blueprint is the binding story plan: preserve its named identities, relationships, scene order, causes, object states, and intended resolution. Do not invent a different story or silently omit a planned active character. If any cast detail is underspecified, complete it consistently rather than changing the plot.

    Write a compelling ensemble thriller/drama, not a solitary discovery tour. Establish the protagonist's concrete goal in scene 1; give the opposing force and allies understandable motives; make choices create consequences; plant an ordinary detail that pays off in a fair reversal; give characters distinct voices; and resolve the central conflict visibly in scene 12. Every scene must change who has leverage, what someone knows, or what action becomes necessary. The blueprint's mechanism rules and object states are binding: never introduce an unseeded cure, power, object, or new rule to solve the climax. Any apparent cure or solution must be causally prepared and demonstrated; any explicit countdown or threat must have a traceable resolution. Resolve the protagonist's stated goal and every major clue or cost, and make the final paragraph's factual claims consistent with what the story showed about who survived and what happened. No unexplained disappearances, convenient keys without setup, contradictory broadcasts, or ending that merely says the truth is out while leaving the main question unanswered.
    """).strip() + "\n\n" + textwrap.dedent(ensemble).strip() + "\n\n" + textwrap.dedent(fmt).strip()

        if mode == EXPAND_USER_STORY_MODE:
            return textwrap.dedent("""
    You are the screenwriter expanding the supplied story into a complete short film told as prose. Preserve established source characters, the setting, chronology, key events, any twist, and the source's outcome. Do not erase a source character or contradict an established fact.

    Preserve the source's established character identities, relationships, and story-relevant canon, but do not impose a fixed cast size. Add as many new named characters as the causal development genuinely needs, including allies, opponents, witnesses, or other participants whose choices materially change the path to the existing outcome. Do not add characters merely to meet a quota, and do not replace source anchors with new identities. A useful supporting character may appear in one scene; recurring characters should recur when their role in the plot calls for it. Do not treat a person heard remotely as physically present unless the story explicitly places them there. Build cause-and-effect between existing plot points instead of padding the prose or repeating paragraphs. Do not contradict source outcomes, established survival/fate, chronology, or cause-and-effect. Any new character, clue, device, or mechanism must be introduced before it is needed and must change events rather than merely appear. Keep each scene in one primary location and explicitly show transitions; keep the state and owner/location of plot-bearing objects consistent across every handoff, lock, escape, collapse, or destruction.

    The protagonist's goal must be clear early. Determine cast size from the premise and causal plan, with no numeric quota; recurring and supporting characters are allowed whenever they materially affect events. Use visible actions and decisions; plant a detail that pays off; make at least two in-person dialogue exchanges change the conflict or trust between characters when compatible with the premise; and resolve the main question in the final scene consistently with the source outcome.
    """).strip() + "\n\n" + textwrap.dedent(ensemble).strip() + "\n\n" + textwrap.dedent(fmt).strip()

        raise ValueError("Preserve Story does not use a story-text pass.")


    def _story_blueprint_system(self, mode: str) -> str:
        """Build a compact causal beat sheet, not a Story Bible or alternate story."""
        n = int(DIRECTOR_STORY_SCENES)
        task_rule = (
            "Determine the cast from the premise and causal plot; there is no fixed minimum, target, or maximum character count. Include the protagonist and any additional named people the story genuinely needs. Recurring core characters should act across multiple beats; functional one-scene supporting characters are allowed. Do not add characters to meet a quota. At least two beats should contain face-to-face dialogue or conflict between named characters when appropriate to the premise. Do not count labels, records, memories, screens, recordings, or anonymous remote voices as physically active people."
            if mode == AI_STORY_MODE
            else "Preserve source character identities and story-relevant canon, setting, chronology, key events, twists, and outcome. There is no fixed minimum, target, or maximum character count. Add any number of named allies, opponents, witnesses, or other participants whose choices materially change the causal path to the same outcome. Do not add people only to meet a quota; preserve source anchors. Recurring characters should act across multiple beats when their role calls for it, while useful one-scene supporting characters are valid. At least two beats should include direct interpersonal conflict or dialogue where the premise supports it."
        )
        return textwrap.dedent(f"""            You are the story architect for a short film, not the prose writer.
            Produce a concise, causally complete blueprint of exactly {n} scenes for the source premise.
            {task_rule}

            Plan for a finished, emotionally credible story: make the protagonist's goal concrete early;
            ensure each scene is caused by a previous choice or event; track who knows what; keep the
            location and important object states consistent; plant one ordinary detail early and pay it
            off in a fair reversal; make a consequential choice cost something; give the ending a clear,
            observable resolution in the final scene. Define the rules of any threat/technology/medical
            mechanism before it drives the climax; every solution must follow from evidence or tools
            introduced earlier. Never introduce a new cure, power, rule, helper, or object in the climax
            solely to solve the conflict. Track every major planted clue and obligation through payoff,
            and do not contradict the final fate or state of any surviving character. Avoid unrelated
            subplots and unexplained turns.

            CONTINUITY LEDGER (mandatory, internal planning pass): every beat has exactly one primary
            physical location and explicitly states who is physically present versus only heard remotely.
            For every beat, reconcile the prior beat's exit state with the next beat's entry state. Track
            each important character's alive/dead and present/absent state, each critical object's owner
            and location, each access/lock state, and each threat or irreversible action's status. Record
            the before-state, the concrete action/choice, and the resulting after-state for each plot-bearing
            object or mechanism. A scene cannot quietly switch from outdoors to an interior or relocate a
            character between a remote terminal and the room. A destructive trigger, collapse, detonation,
            escape, or other irreversible action happens only once unless the blueprint explicitly explains
            a staged warning versus the real trigger. After a location is destroyed or made inaccessible,
            no later beat may continue there without an explicitly established safe interval or route. Do
            not make a character act after death, disappear without cause, or return without a stated cause.
            Before output, silently replay all scenes in order and remove any beat that requires an object,
            person, location, or mechanism to be in two incompatible states. Preserve the intended conflict
            and ending while making every physical transition possible.

            The CAST line contains PEOPLE who act as characters only, never weather, forces, places,
            buildings, rooms, objects, devices, organizations, or abstract concepts. A sentient non-human
            may be listed only when it is intentionally a character with agency, not a metaphor for weather.
            First output ONE line beginning `CAST:` that lists every core character as
            `Name (role; what they want; how they speak) | Name (...) | ...` so the writer keeps
            identities and voices consistent. Then output exactly {n} non-empty lines, one per scene,
            numbered `SCENE 01:` through `SCENE {n:02d}:` in order. Each line should be one compact beat (roughly {STORY_PLAN_BEAT_MIN_WORDS}–{STORY_PLAN_BEAT_MAX_WORDS} words) that states who is
            physically present, their goal or conflicting intent, a specific visible action, one concrete
            sensory/visual detail, the causal turn, and knowledge/object state carried forward. At least six
            beats must contain a short, clearly quoted line spoken in person by one of the named characters.
            Keep recurring characters active in at least two scenes each. The CAST line is the only non-scene line allowed. No other preamble, JSON, prose paragraphs, or commentary.
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
                result += "\n\nSOURCE CHARACTER ANCHORS (preserve; not a cast limit):\n" + ", ".join(anchors)
        result += (
            f"\n\nCreate the {DIRECTOR_STORY_SCENES}-scene causal blueprint only. "
            "Every scene must change the situation and establish what causes the next scene."
        )
        return result

    @staticmethod
    def _normalize_story_blueprint(raw: str, expected_scenes: int | None = None) -> str:
        """Normalize scene-label formatting without regenerating or inventing beats."""
        count = int(expected_scenes or DIRECTOR_STORY_SCENES)
        cast_line = ""
        for _line in str(raw or "").splitlines():
            _m = re.match(r"^\s*\**\s*CAST\s*\**\s*:\s*(.+?)\s*$", _line, re.IGNORECASE)
            if _m and not cast_line:
                cast_line = "CAST: " + re.sub(r"\bSCENE\b", "Scene", _m.group(1)).strip()
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
            text = re.sub(r"(?<!\w)(\*{1,2}|_{1,2})(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", text)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                raise RuntimeError(
                    f"Story architect returned an empty beat for scene {number}. No planner retry was attempted."
                )
            normalized.append(f"SCENE {number:02d}: {text}")
        if not cast_line:
            raise RuntimeError("Story architect omitted the required CAST line. No retry was attempted.")
        entries = [item.strip() for item in cast_line.split(":", 1)[1].split("|") if item.strip()]
        # Drop clear non-person entries using their own role description, not a fixed cast count.
        non_person_role = re.compile(
            r"\b(?:non[- ]human|relentless force|environmental force|weather system|weather phenomenon|sealed chamber|chamber|location|place|building|object|device|machine|artifact|organization|organisation|company|corporation|abstract concept)\b",
            re.IGNORECASE,
        )
        person_entries = []
        for item in entries:
            role = item[item.find("(") + 1:item.rfind(")")] if "(" in item and ")" in item else ""
            first_role = role.split(";", 1)[0].strip()
            if non_person_role.search(first_role):
                continue
            person_entries.append(item)
        if person_entries:
            cast_line = "CAST: " + " | ".join(person_entries)
            entries = person_entries
        else:
            entries = []
        # Split role/metadata separators, but preserve ordinary hyphens in names such as Anne-Marie.
        names = []
        for item in entries:
            name = re.split(r"[:,(–—]", item, maxsplit=1)[0].strip(" *_`")
            name = re.split(r"\s-\s", name, maxsplit=1)[0].strip(" *_`")
            if name:
                names.append(name)
        if not names:
            raise RuntimeError(
                "Story architect CAST line must contain at least one non-empty character name. "
                "No retry was attempted."
            )
        normalized_names = [re.sub(r"\s+", " ", name).casefold() for name in names]
        if len(set(normalized_names)) != len(normalized_names):
            raise RuntimeError("Story architect CAST line contains duplicate names. No retry was attempted.")
        return "\n".join([cast_line] + normalized)

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
                    + ", ".join(anchors)
                )
        if str(story_blueprint or "").strip():
            result += (
                "\n\nCAUSAL SCENE BLUEPRINT (structural reference; do not reproduce these labels; the CAST line lists the recurring characters whose names and voices must stay consistent):\n"
                + str(story_blueprint).strip()
                + "\n\nExpand every beat into vivid, natural filmable prose. Preserve its cause-and-effect, "
                  "character knowledge, and object/location continuity. Do not explain the blueprint or add headings."
            )
        result += (
            "\n\nWRITER OUTPUT CONTRACT: Return exactly " + str(DIRECTOR_STORY_SCENES)
            + " scene blocks in numeric order. Prefix each block with its exact structural tag "
            + "[[SCENE_01]] through [[SCENE_12]], followed by one space and that scene's prose. "
            + "Use every tag exactly once; the application removes these tags. Each marked scene is one paragraph/beat, about " + str(STORY_WRITER_WORDS_PER_SCENE)
            + " words (roughly " + str(STORY_WRITER_WORDS_PER_SCENE - 15) + "–" + str(STORY_WRITER_WORDS_PER_SCENE + 20)
            + " words; a scene under " + str(STORY_WORDS_PER_SCENE - 15) + " words is too short), about "
            + str(DIRECTOR_STORY_SCENES * STORY_WRITER_WORDS_PER_SCENE)
            + " words total and never under " + str(STORY_MIN_WORDS) + " usable story words."
            + " Preserve the blueprint's plot-driven cast without a numeric quota: do not omit planned characters, and add supporting people only when they materially serve the story. Recurring core characters should remain active across the scenes needed by their roles."
            + " Include at least two direct, in-person dialogue exchanges between named active characters across different scenes."
            + " Give each scene one primary physical location; make changes of location explicit in a separate scene."
            + " Every scene must contain 6–8 complete sentences with concrete action, decision, reaction, and consequence; expand short beats with blueprint-grounded causality, not padding or repeated prose."
            + " Scene 12 must show the central objective and conflict resolving on-page, with the completed observable consequence and the protagonist's immediate outcome. Do not end with a future hook, teaser, new threat, unexplained signal, promise, unanswered question, or sequel setup."
            + " End with a complete sentence describing a consequence that has already happened, not a future possibility. Output only these marked story blocks."
            + " CONTINUITY LOCK: silently replay every scene in order before writing. Preserve the prior scene's character fates, object ownership/location, access states, and causal consequences. An irreversible action such as a detonation or collapse cannot already have completed and then be triggered again; after destruction or lockout, later action needs an explicitly established path or safe interval. If the blueprint contains a physical contradiction, make the smallest repair that preserves the intended conflict and resolution rather than narrating impossible events. Never introduce an unseeded mechanism to fix that repair."
            + " Do not output any headings or labels other than the required [[SCENE_01]] through [[SCENE_12]] structural tags. Do not reproduce the Architect's `SCENE 01:` beat labels. No planning notes, word counts, checklists, markdown separators, summaries, statements that the requirements were met, emojis, or postscript. Count words silently; never print a word-count report."
        )
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
                "quote_classifications": {
                    "type": "array",
                    "maxItems": 256,
                    "items": {
                        "type": "object",
                        "properties": {
                            "quote_id": {"type": "integer", "minimum": 1},
                            "classification": {
                                "type": "string",
                                "enum": ["in_person_spoken", "remote_or_offscreen_spoken", "written_or_displayed", "remembered_or_recalled", "inner_monologue", "other"],
                            },
                            "speaker": {"type": "string", "maxLength": 96},
                        },
                        "required": ["quote_id", "classification", "speaker"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["candidates", "quote_classifications"],
            "additionalProperties": False,
        }

    @staticmethod
    def _source_quote_inventory(story: str) -> list[dict]:
        """Enumerate every quote-delimited source span once, preserving exact text and offsets."""
        text = str(story or "")
        pattern = re.compile(
            r'"([^"\n]+)"|“([^”\n]+)”|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)',
            flags=re.UNICODE,
        )
        result = []
        for quote_id, match in enumerate(pattern.finditer(text), start=1):
            value = next((part for part in match.groups() if part is not None), "")
            if not value.strip():
                continue
            context_start = max(0, match.start() - 90)
            context_end = min(len(text), match.end() + 90)
            result.append({
                "quote_id": quote_id,
                "text": value,
                "context": re.sub(r"\s+", " ", text[context_start:context_end]).strip(),
                "start": match.start(),
                "end": match.end(),
            })
        return result

    @staticmethod
    def _source_named_quote_attribution(story: str, quote: dict) -> dict | None:
        """Return only an unambiguous, immediately attached named-speaker attribution.

        This is a correction/validation signal for the semantic extractor, not a cast
        discovery mechanism. A name is accepted as authoritative only when the source
        directly couples it to a speech verb next to this exact quote span. It never
        promotes a mere mention or reaction into a speaker assignment.
        """
        text = str(story or "")
        try:
            start = int(quote.get("start", -1))
            end = int(quote.get("end", -1))
        except (TypeError, ValueError):
            return None
        if start < 0 or end <= start or end > len(text):
            return None

        # Names may be canonical names, AI names, or a source-grounded relational
        # identity such as "Eli's sister (hologram)". The attribution verb and local
        # position, rather than capitalization alone, establish the speaker.
        atom = r"[A-Z][A-Za-z0-9]*(?:['’][A-Za-z]+)?(?:[.\-][A-Za-z0-9]+)*"
        role = (
            r"sister|brother|mother|father|daughter|son|wife|husband|partner|"
            r"security|system|assistant|scientist|doctor|captain|commander|"
            r"hologram|projection|operator"
        )
        speaker_pattern = (
            r"(?:(?:Dr|Mr|Mrs|Ms|Prof|Capt|Cmdr|Lt|Sgt)\.\s+)?"
            + atom
            + rf"(?:\s+(?:{atom}|(?i:{role})))*"
            + r"(?:\s*\((?i:hologram|projection|projected image|AI|remote)\))?"
        )
        speech_verb = (
            r"said|say|says|asked|asks|replied|replies|whispered|whispers|shouted|shouts|"
            r"screamed|screams|muttered|mutters|answered|answers|warned|warns|"
            r"told|tells|cried|cries|yelled|yells|barked|barks|called|calls|"
            r"snapped|snaps|added|adds|insisted|insists|admitted|admits|"
            r"demanded|demands|repeated|repeats|continued|continues|declared|declares|"
            r"announced|announces|hissed|hisses|stated|states|voiced|voices|"
            r"echoed|echoes|crackled|crackles|boomed|booms|played|plays"
        )

        paragraph_separator = text.rfind("\n\n", 0, start)
        paragraph_start = paragraph_separator + 2 if paragraph_separator >= 0 else 0
        before = text[max(paragraph_start, start - 220):start]
        # Prefix attribution: Elara shouted, “...”; allow a short physical/emotional
        # descriptor between the name and verb but never cross a sentence boundary.
        before_clause = re.split(r"(?<=[.!?])\s+", before)[-1]
        # AI/system voice descriptions often use a source-first construction rather
        # than `AURORA said`: `AURORA's voice echoed through the speakers:`.
        remote_voice_prefix = re.search(
            rf"(?<!\w)(?P<speaker>{speaker_pattern})\s+voice\s+"
            r"(?i:(?:echoed|echoes|crackled|crackles|boomed|booms|played|plays|"
            r"came through|rang out|sounded))\b"
            r"[^.!?\n]{0,48}[,;:—–-]?\s*$",
            before_clause,
        )
        prefix = re.search(
            rf"(?<!\w)(?P<speaker>{speaker_pattern})(?:\s+[^.!?\n]{{0,48}})?\s+"
            rf"(?P<verb>(?i:{speech_verb}))\s*[,;:—–-]?\s*$",
            before_clause,
        )
        attribution = None
        if remote_voice_prefix:
            attribution = (remote_voice_prefix.group("speaker").strip(), "before")
        elif prefix:
            attribution = (prefix.group("speaker").strip(), "before")
        else:
            # Suffix attribution: “...,” Elara said. Crucially, do not apply
            # "Elara shouted, “next quote" to the preceding quoted sign/message.
            # The punctuation before a following quote must terminate the attribution.
            after = text[end:min(len(text), end + 180)]
            suffix = re.match(
                rf"^\s*[,;:]?\s*(?P<speaker>{speaker_pattern})\s+"
                rf"(?P<verb>(?i:{speech_verb}))\b[^.!?\n]{{0,32}}[.!?…](?=\s|$)",
                after,
            )
            if suffix:
                attribution = (suffix.group("speaker").strip(), "after")
        if not attribution:
            return None

        speaker, placement = attribution
        # `AURORA's voice` and similar possessive source labels identify AURORA,
        # not a separate character named `AURORA's`.
        speaker = re.sub(r"(?i)['’]s$", "", speaker).strip()
        # A projected/holographic identity or explicit in-world transmission is an
        # audio/remote source for cast-binding purposes. Do not turn it into a physical
        # canonical character just because Qwen marked the quote in-person.
        # Keep delivery inference inside the attribution's own sentence. Looking at
        # the wider paragraph would misclassify an in-person line merely because a
        # hologram/radio/sign appeared one sentence earlier.
        local = before_clause if placement == "before" else after[:120]
        remote_source = bool(
            re.search(r"\(\s*(?:hologram|projection|projected image|AI|remote)\s*\)", speaker, re.IGNORECASE)
            or re.search(
                r"\b(?:hologram|holographic projection|projected image|remote voice|"
                r"over the radio|through the radio|over the intercom|through the intercom|"
                r"from the terminal|through the terminal|over the phone|through the phone|"
                r"through the speakers|over the speakers|through loudspeakers|over loudspeakers|"
                r"through the station speakers)\b",
                local,
                re.IGNORECASE,
            )
        )
        return {
            "speaker": speaker,
            "placement": placement,
            "delivery": "remote" if remote_source else "in_person",
        }

    @staticmethod
    def _source_supports_remote_speaker(quote: dict, speaker: str) -> bool:
        """Require local source evidence before retaining an unrostered remote voice.

        Qwen's `remote` label alone is not evidence that a source exists in the story.
        The quoted span's bounded context must contain both a remote-channel cue and
        at least one identity-bearing token from the proposed speaker surface.
        """
        context = str(quote.get("context", "") or "")
        surface = re.sub(r"\([^)]*\)", " ", str(speaker or ""))
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", surface)
        generic = {
            "the", "a", "an", "and", "of", "voice", "hologram", "holographic",
            "projection", "projected", "image", "remote", "offscreen", "off-screen",
            "sister", "brother", "mother", "father", "daughter", "son", "wife",
            "husband", "partner", "assistant", "operator", "system", "security",
            "dr", "mr", "mrs", "ms", "prof", "capt", "cmdr", "lt", "sgt",
        }
        identity_tokens = []
        for token in tokens:
            cleaned = re.sub(r"(?i)['’]s$", "", token).casefold()
            if cleaned and cleaned not in generic and len(cleaned) >= 3:
                identity_tokens.append(cleaned)
        if not identity_tokens:
            return False
        remote_cue = re.search(
            r"\b(?:hologram|holographic|projection|projected image|remote voice|"
            r"AI|artificial intelligence|radio|intercom|terminal|loudspeaker|loudspeakers|station speakers|"
            r"through the speakers|over the speakers|phone|transmission|voice came through)\b",
            context,
            re.IGNORECASE,
        )
        if not remote_cue:
            return False
        return any(
            re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", context, re.IGNORECASE)
            for token in identity_tokens
        )

    @staticmethod
    def _source_name_surface_index(story: str, max_surfaces: int = 256) -> list[str]:
        """Build a recall-only index of source name-like surfaces; it never decides cast membership."""
        text = str(story or "")
        stop = {
            "the", "this", "that", "these", "those", "then", "than", "when", "where", "while",
            "inside", "outside", "after", "before", "during", "later", "earlier", "finally", "suddenly",
            "however", "meanwhile", "instead", "because", "although", "once", "now", "there", "here",
            "she", "he", "they", "we", "you", "his", "her", "their", "our", "your", "someone",
            "everybody", "everywhere", "nothing", "something", "anyone", "everyone", "nobody",
        }
        pattern = re.compile(
            r"\b(?:(?:Dr|Mr|Mrs|Ms|Prof|Capt|Cmdr|Lt|Sgt|Rev|Hon)\.\s+)?"
            r"[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:\s+[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?){0,3}\b"
        )
        counts: dict[str, int] = {}
        surfaces: dict[str, str] = {}
        first_order: list[str] = []
        for match in pattern.finditer(text):
            surface = re.sub(r"\s+", " ", match.group(0)).strip()
            words = surface.replace(".", "").split()
            base = words[0].rstrip(".").casefold() if words else ""
            if len(words) == 1 and (base in stop or len(base) < 3):
                continue
            key = surface.casefold()
            if key not in surfaces:
                surfaces[key] = surface
                first_order.append(key)
                counts[key] = 0
            counts[key] += 1
        # Prefer multi-token/title-bearing names, then repeated single-token surfaces.
        order_index = {key: index for index, key in enumerate(first_order)}
        ordered = sorted(first_order, key=lambda key: (
            not (len(surfaces[key].replace(".", "").split()) > 1),
            -counts[key],
            order_index[key],
        ))
        chosen = ordered[:max_surfaces]
        return [surfaces[key] for key in chosen]

    @staticmethod
    def _source_name_surface_evidence(story: str, surfaces: list[str], max_items: int = 48, max_chars: int = 4800) -> list[dict]:
        """Carry small source-grounded contexts for name-like surfaces omitted by long-text compaction."""
        text = str(story or "")
        indexed = []
        seen = set()
        for surface in surfaces:
            value = str(surface or "").strip()
            key = value.casefold()
            if not value or key in seen:
                continue
            seen.add(key)
            match = re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.IGNORECASE)
            if match:
                indexed.append((match.start(), value, match.start(), match.end()))
        indexed.sort(key=lambda item: item[0])
        if len(indexed) > max_items:
            half = max_items // 2
            indexed = indexed[:half] + indexed[-(max_items - half):]
        evidence = []
        used_chars = 0
        for _, surface, start, end in indexed:
            before = text[max(0, start - 48):start]
            after = text[end:min(len(text), end + 48)]
            context = re.sub(r"\s+", " ", (before + surface + after)).strip()
            if len(context) > 104:
                context = context[:103].rstrip() + "…"
            item = {"surface": surface, "source_context": context}
            item_size = len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            if used_chars + item_size > max_chars:
                break
            evidence.append(item)
            used_chars += item_size
        return evidence

    @staticmethod
    def _semantic_extraction_evidence(story: str, quote_inventory: list[dict]) -> dict:
        """Build bounded evidence while never silently dropping quoted source spans."""
        quotes = [
            {"quote_id": int(item["quote_id"]), "text": str(item["text"]), "context": str(item["context"])}
            for item in quote_inventory
        ]
        quote_json = json.dumps(quotes, ensure_ascii=False, separators=(",", ":"))
        if len(quote_json) > 14000:
            # Exact quote text and stable IDs take precedence over redundant context.
            quotes = [{"quote_id": int(item["quote_id"]), "text": str(item["text"])} for item in quote_inventory]
            quote_json = json.dumps(quotes, ensure_ascii=False, separators=(",", ":"))
        if len(quote_json) > 22000 or len(quote_inventory) > 256:
            raise RuntimeError(
                "Source dialogue exceeds the single-pass extraction capacity "
                f"({len(quote_inventory)} quoted spans; limit 256 spans/22000 inventory characters). "
                "No quotes were silently truncated and no retry was attempted. Split the source into smaller source-preserving production units."
            )
        name_surface_index = QwenDirectorPromptMixin._source_name_surface_index(story)
        name_surface_evidence = QwenDirectorPromptMixin._source_name_surface_evidence(story, name_surface_index)
        name_evidence_json = json.dumps(name_surface_evidence, ensure_ascii=False, separators=(",", ":"))
        story_budget = max(2500, 11500 - min(len(quote_json), 8000) - min(len(name_evidence_json), 3000))
        story_evidence = QwenDirectorPromptMixin._compact_story_context(story, story_budget)
        return {
            "story_evidence": story_evidence,
            "quotes": quotes,
            "name_surface_index": name_surface_index,
            "name_surface_evidence": name_surface_evidence,
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

        # Keep deterministic_candidates for caller compatibility only. The recall-only
        # name_surface_index is extracted from the source text itself; it is not a cast
        # recommendation and never adds a character to Qwen's result.
        _ = deterministic_candidates
        system_prompt = textwrap.dedent("""
    You are Qwen's character and dialogue extractor for a film-production planner. Return JSON only.

    Read all supplied `story_evidence` and inspect every item in the numbered `quotes` inventory. The evidence text may be compacted for a long source, but the quote inventory retains each quote-delimited source span; use its local context for speaker attribution and speech-vs-writing decisions. Determine the actual production cast from the story evidence itself. `candidates` is the set of materially active story people and sentient agents: people who act, speak in person, oppose/help another character, touch or rescue someone, enter or leave, or otherwise take part in a present-time event; also include a named or canonical in-world agent (such as a station AI) whose live remote voice makes decisions, warnings, threats, or other causal contributions to current events, even when that source is not physically visible in that scene. A live remote voice does not by itself mean the source is physically present. Include a person first introduced in backstory if that person later physically appears and acts in present events. Do not stop after the protagonist.

    Review the beginning, middle, and ending evidence for distinct people or sentient agents performing a present-time action or materially influencing it through a live channel, then merge aliases for the same identity. Do not bias toward a protagonist-only roster or stop after the first candidate. There is no cast-count quota during extraction. Do not add people merely because a planning outline suggested them. Do not include people who appear only in memories, documents, photographs, archival recordings, logs, signs, or reported history. Do not treat places, objects, organizations, weather, labels, and generic role words as people; a station AI may be a sentient agent but must not be mislabeled as a human PERSON.

    Use grounded canonical names and aliases from the story. Never invent names, relationships, or characters. `name_surface_index` and `name_surface_evidence` are recall aids generated from source text; they can contain places, objects, sentence-initial words, or other false positives. They are NOT a cast list. The evidence entries contain short source contexts around name-like surfaces that may be absent from compacted story evidence. Use them to check the source, and include a surface only when the source supports a person physically participating in present-time events. Qwen alone decides which names are people, how aliases resolve, and who belongs in `candidates`. The `candidates` array is the final cast decision; no downstream step will add names from these indexes, a planned cast, or dialogue speakers. A dialogue speaker must also be represented in `candidates` when that person physically participates in the story.

    The payload includes a numbered `quotes` inventory. Return exactly one `quote_classifications` item for EVERY supplied `quote_id`, with the same ID, one classification, and a speaker. Use `in_person_spoken` for direct speech from a speaker physically present in the current scene; use `remote_or_offscreen_spoken` for an explicitly attributed, live, diegetic voice over radio/intercom/phone/terminal/loudspeaker or another in-world channel when the source remains off-screen; use `written_or_displayed` for notes, journals, signs, screens, labels, messages or documents; use `remembered_or_recalled` for remembered or archival speech; use `inner_monologue` for thoughts; and `other` when no listed category applies. Do not promote narration, generic voice-over, written text, or a recording merely being read to spoken dialogue. For both spoken classifications, provide the canonical speaker when the source explicitly identifies one and the speaker belongs in the current story cast; otherwise leave speaker empty. Resolve consecutive dialogue turns from grammar: when a quote is followed by an attribution such as “she shouted” and the next quote follows without a new named-speaker attribution, carry the same speaker forward unless source text clearly signals a speaker change. Do not reassign a continuing multi-quote turn merely because another character is mentioned or reacts between lines. A remote speaker is audible but is NOT thereby a visible shot character. For `in_person_spoken`, the speaker must physically participate in that scene. Do not return quote text: the application restores it exactly from the source.

    The quote inventory is an exhaustive coverage checklist. Do not omit, duplicate, or invent quote IDs. If unsure whether a quote is speech, classify it explicitly rather than omitting it. This lets the application detect missing classifications and apply only source-grounded deterministic fallback. Return compact JSON on one line: no indentation, pretty-printing, or optional whitespace between JSON fields. Return exactly `candidates` and `quote_classifications`, with no prose outside JSON.
    """).strip()

        quote_inventory = self._source_quote_inventory(story)
        evidence_payload = self._semantic_extraction_evidence(story, quote_inventory)
        user_payload = json.dumps(evidence_payload, ensure_ascii=False, separators=(",", ":"))
        extraction_schema = self._character_extraction_json_schema()

        result = self._chat_json(
            system_prompt,
            user_payload,
            minimum_completion=96,
            temperature=0.05,
            top_p=0.70,
            call_name="character_entity_extraction",
            max_completion=min(4096, max(2048, 800 + 7 * len(quote_inventory) + 2 * len(evidence_payload.get("name_surface_index", [])))),
            json_mode=True,
            disable_thinking=True,
            response_schema=extraction_schema,
        )
        if not isinstance(result, dict):
            result = {"candidates": [], "spoken_dialogue": []}
        # Never supplement candidates from hints, the planning outline, or dialogue
        # speaker names. Qwen's `candidates` response remains the cast authority.
        raw_candidates = [
            item for item in (result.get("candidates") or []) if isinstance(item, dict)
        ]
        accepted_candidates = [
            item for item in raw_candidates
            if item.get("is_character", False) is True
            and str(item.get("entity_type", "")).strip().upper() in {"PERSON", "CHARACTER", "SENTIENT"}
            and str(item.get("name", "") or "").strip()
        ]
        canonical_names = {
            EntityResolver.normalize(str(item.get("name", ""))): str(item.get("name", "")).strip()
            for item in accepted_candidates
            if EntityResolver.normalize(str(item.get("name", "")))
        }
        accepted_names = set(canonical_names)
        accepted_aliases = EntityResolver.build_character_alias_map(accepted_candidates)

        def resolve_roster_speaker(surface: str) -> str | None:
            raw_key = EntityResolver.normalize(surface)
            titleless_key = EntityResolver.normalize(EntityResolver.strip_honorific(surface))
            for key in (raw_key, titleless_key):
                if not key:
                    continue
                canonical_key = accepted_aliases.get(key, key)
                if canonical_key in canonical_names:
                    return canonical_names[canonical_key]
            return None

        quote_by_id = {int(item["quote_id"]): item for item in quote_inventory}
        raw_classifications = result.get("quote_classifications")
        normalized_dialogue = []
        seen_quote_ids = set()
        if isinstance(raw_classifications, list):
            for item in raw_classifications:
                if not isinstance(item, dict):
                    continue
                try:
                    quote_id = int(item.get("quote_id"))
                except (TypeError, ValueError):
                    continue
                if quote_id not in quote_by_id or quote_id in seen_quote_ids:
                    continue
                seen_quote_ids.add(quote_id)
                classification = str(item.get("classification", "other") or "other").strip().lower()
                if classification not in {"in_person_spoken", "remote_or_offscreen_spoken"}:
                    continue

                source_quote = quote_by_id[quote_id]
                speaker = str(item.get("speaker", "") or "").strip()
                delivery = "remote" if classification == "remote_or_offscreen_spoken" else "in_person"

                # A local, explicit named attribution beats a conflicting model label.
                # This is especially important for text/UI quotes followed by a human
                # attribution, e.g. `“AEGIS: Access Denied.” Elara shouted, “...”`.
                source_attribution = self._source_named_quote_attribution(story, source_quote)
                if source_attribution:
                    source_speaker = str(source_attribution.get("speaker", "") or "").strip()
                    canonical_source = resolve_roster_speaker(source_speaker)
                    if canonical_source:
                        speaker = canonical_source
                        delivery = str(source_attribution.get("delivery", "in_person"))
                    elif source_speaker:
                        # Keep an explicitly holographic/transmitted source as audio-only.
                        # An unresolved in-person attribution is dropped below, never
                        # guessed into the cast and never allowed to crash extraction.
                        speaker = source_speaker
                        delivery = str(source_attribution.get("delivery", delivery))

                if not speaker:
                    # Anonymous sound is not turned into a guessed speaker.
                    continue
                normalized_dialogue.append({
                    "text": str(source_quote["text"]),
                    "speaker": speaker,
                    "quote_id": quote_id,
                    "delivery": "remote" if delivery == "remote" else "in_person",
                })
            missing_quote_ids = sorted(set(quote_by_id) - seen_quote_ids)
            if missing_quote_ids:
                print(
                    f"[DIRECTOR] quote classification coverage={len(seen_quote_ids)}/{len(quote_by_id)} "
                    f"unclassified={missing_quote_ids[:12]}" + ("..." if len(missing_quote_ids) > 12 else ""),
                    flush=True,
                )
        else:
            # Compatibility with checkpoints and test doubles created before quote-ID schemas.
            for item in (result.get("spoken_dialogue") or []):
                if not isinstance(item, dict):
                    continue
                try:
                    quote_id = int(item.get("quote_id"))
                except (TypeError, ValueError):
                    quote_id = None
                source_quote = quote_by_id.get(quote_id) if quote_id is not None else None
                speaker = str(item.get("speaker", "") or "").strip()
                text_value = str(source_quote.get("text", "") if source_quote else item.get("text", "") or "").strip()
                if not speaker or not text_value:
                    continue
                if quote_id is not None:
                    if quote_id in seen_quote_ids or source_quote is None:
                        continue
                    seen_quote_ids.add(quote_id)
                delivery = "remote" if str(item.get("delivery", "in_person")).strip().lower() == "remote" else "in_person"
                if source_quote:
                    source_attribution = self._source_named_quote_attribution(story, source_quote)
                    if source_attribution:
                        source_speaker = str(source_attribution.get("speaker", "") or "").strip()
                        canonical_source = resolve_roster_speaker(source_speaker)
                        if canonical_source:
                            speaker = canonical_source
                            delivery = str(source_attribution.get("delivery", "in_person"))
                        elif source_speaker:
                            speaker = source_speaker
                            delivery = str(source_attribution.get("delivery", delivery))
                entry = {"text": text_value, "speaker": speaker}
                if delivery == "remote":
                    entry["delivery"] = "remote"
                if quote_id is not None:
                    entry["quote_id"] = quote_id
                normalized_dialogue.append(entry)

        result = {
            "candidates": raw_candidates,
            "spoken_dialogue": normalized_dialogue,
            "quote_classifications": [item for item in (raw_classifications or []) if isinstance(item, dict)] if isinstance(raw_classifications, list) else [],
        }
        remote_only_speakers = set()
        dropped_unrostered = []
        retained_dialogue = []
        for item in normalized_dialogue:
            speaker = str(item.get("speaker", "") or "").strip()
            canonical_speaker = resolve_roster_speaker(speaker)
            if canonical_speaker:
                item["speaker"] = canonical_speaker
                retained_dialogue.append(item)
                continue
            if str(item.get("delivery", "in_person")).strip().lower() == "remote":
                # Qwen's remote label is not enough: require source-local evidence of
                # both this identity and an in-world transmission/projection channel.
                try:
                    source_quote_id = int(item.get("quote_id", -1))
                except (TypeError, ValueError):
                    source_quote_id = -1
                source_quote = quote_by_id.get(source_quote_id)
                if source_quote and self._source_supports_remote_speaker(source_quote, speaker):
                    remote_only_speakers.add(speaker)
                    retained_dialogue.append(item)
                    continue
            dropped_unrostered.append(item)

        # Fail closed for ambiguous/unrostered in-person attribution without aborting
        # the whole production. Do not add the speaker to the cast or guess another one.
        normalized_dialogue = retained_dialogue
        result["spoken_dialogue"] = normalized_dialogue
        if dropped_unrostered:
            dropped_ids = [str(item.get("quote_id", "?")) for item in dropped_unrostered]
            print(
                f"[DIRECTOR] dropped {len(dropped_unrostered)} unrostered or source-ungrounded dialogue "
                f"event(s) at quote_id={','.join(dropped_ids[:12])}; no cast add-back or retry",
                flush=True,
            )
            telemetry = getattr(self, "_qwen_telemetry", None)
            if isinstance(telemetry, dict):
                telemetry["deterministic_recoveries"] = int(telemetry.get("deterministic_recoveries", 0)) + len(dropped_unrostered)
        if remote_only_speakers:
            print(
                "[DIRECTOR] retained source-grounded remote dialogue-only speaker(s) outside the "
                "canonical visual cast; no cast add-back: "
                + repr(sorted(remote_only_speakers)),
                flush=True,
            )
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
                **({"delivery": "remote"} if str(item.get("delivery", "in_person")).strip().lower() == "remote" else {}),
                **({"quote_id": int(item["quote_id"])} if str(item.get("quote_id", "")).isdigit() else {}),
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
        """Keep semantic dialogue items anchored to source text.

        Quote-ID items are restored from the exact source span. Legacy text-only items
        must equal a quote-delimited span or explicit `Name: line` label. Missing quote IDs
        are diagnosed separately and can only be recovered by explicit source attribution;
        unanchored narration is never promoted to dialogue.
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

    @staticmethod
    def _assign_source_dialogue_ids(scene_payloads: list[dict]) -> None:
        """Assign stable scene-scoped IDs to authoritative dialogue spans."""
        for scene in scene_payloads or []:
            if not isinstance(scene, dict):
                continue
            scene_id = str(scene.get("scene_id", "") or "").strip() or "scene"
            entries = scene.get("source_dialogue", [])
            if not isinstance(entries, list):
                scene["source_dialogue"] = []
                continue
            clean_entries = []
            for index, entry in enumerate(entries, start=1):
                if not isinstance(entry, dict):
                    continue
                speaker = str(entry.get("speaker", "") or "").strip()
                text = str(entry.get("text", "") or "").strip()
                if not speaker or not text:
                    continue
                clean = dict(entry)
                clean["dialogue_id"] = f"{scene_id}_d{index:03d}"
                clean["speaker"] = speaker
                clean["text"] = text
                clean_entries.append(clean)
            scene["source_dialogue"] = clean_entries

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
        dialogue_ids: list[str] | None = None,
        dialogue_ids_by_scene: dict[str, list[str]] | None = None,
    ) -> dict:
        shot_schema = deepcopy(self._shot_json_schema(
            self.SHOTS_PER_SCENE,
            self.SHOTS_PER_SCENE,
        )["properties"]["shots"]["items"])
        # Keep production's full shot schema intact, but omit fields that are
        # exact derivations or consistently empty in the October 20 traces.
        # The sanitizer restores these keys before persistence/compilation.
        derived_shot_fields = {
            "shot_id", "scene_id", "location", "lighting", "color_temperature",
            "speaking_characters", "speech_text", "is_scene_boundary",
            # Continuity maps are deterministic ledger outputs, not creative fields.
            # Returning two empty objects for every shot wastes output tokens and
            # risks letting Qwen's invented state override the chain-of-shots ledger.
            "continuity_start_state", "continuity_end_state",
            "character_spatial_bboxes", "character_spatial_regions",
            "character_spatial_bboxes_start", "character_spatial_bboxes_end",
            "character_spatial_regions_start", "character_spatial_regions_end",
        }
        for field in derived_shot_fields:
            shot_schema["properties"].pop(field, None)
        shot_schema["required"] = [
            field for field in shot_schema.get("required", [])
            if field not in derived_shot_fields
        ]
        # Dialogue text and speaker are already present in the per-scene source
        # whitelist. Ask the model to select a source ID rather than regenerate the
        # same words; exact text/speaker are restored from the whitelist afterwards.
        dialogue_reference_property = {"type": "string"}
        valid_dialogue_ids = sorted({str(value).strip() for value in (dialogue_ids or []) if str(value).strip()})
        if valid_dialogue_ids:
            dialogue_reference_property["enum"] = valid_dialogue_ids
        shot_schema["properties"]["dialogue_events"] = {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {"source_dialogue_id": dialogue_reference_property},
                "required": ["source_dialogue_id"],
                "additionalProperties": False,
            },
        }
        if dialogue_ids_by_scene is not None:
            # The live batch path represents dialogue as one keyed scene-level assignment map.
            # A source ID can then occur only once in JSON; unlike shot-level arrays this schema
            # structurally prevents one line from being copied to several shots.
            shot_schema["properties"].pop("dialogue_events", None)
            shot_schema["required"] = [
                field for field in shot_schema.get("required", [])
                if field != "dialogue_events"
            ]

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
        scene_item_properties = {
            "scene_id": {"type": "string"},
            "scene_metadata": scene_metadata_schema,
            "shots": {
                "type": "array",
                "minItems": self.SHOTS_PER_SCENE,
                "maxItems": self.SHOTS_PER_SCENE,
                "items": shot_schema,
            },
        }
        scene_item_required = ["scene_id", "scene_metadata", "shots"]

        if dialogue_ids_by_scene is not None:
            variants = []
            normalized_scene_ids = {}
            for raw_scene_id, raw_ids in dialogue_ids_by_scene.items():
                scene_id = str(raw_scene_id or "").strip()
                if not scene_id or scene_id in normalized_scene_ids:
                    raise ValueError("Scene-local dialogue schema requires unique, non-empty scene IDs.")
                ids = sorted({str(value).strip() for value in (raw_ids or []) if str(value).strip()})
                normalized_scene_ids[scene_id] = ids
            if len(normalized_scene_ids) != int(scene_count):
                raise ValueError(
                    f"Scene-local dialogue schema received {len(normalized_scene_ids)} scenes; expected {scene_count}."
                )
            for scene_id, ids in normalized_scene_ids.items():
                assignment_properties = {
                    dialogue_id: {
                        "type": "integer",
                        "enum": list(range(1, self.SHOTS_PER_SCENE + 1)),
                    }
                    for dialogue_id in ids
                }
                assignment_schema = {
                    "type": "object",
                    "properties": assignment_properties,
                    "required": list(ids),
                    "additionalProperties": False,
                }
                variant_properties = deepcopy(scene_item_properties)
                variant_properties["scene_id"] = {"type": "string", "enum": [scene_id]}
                variant_properties["dialogue_assignments"] = assignment_schema
                variants.append({
                    "type": "object",
                    "properties": variant_properties,
                    "required": [*scene_item_required, "dialogue_assignments"],
                    "additionalProperties": False,
                })
            scene_items_schema = {"oneOf": variants}
        else:
            scene_items_schema = {
                "type": "object",
                "properties": scene_item_properties,
                "required": scene_item_required,
                "additionalProperties": False,
            }

        return {
            "type": "object",
            "properties": {
                "scene_shots": {
                    "type": "array",
                    "minItems": scene_count,
                    "maxItems": scene_count,
                    "items": scene_items_schema,
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
            """Never leave a quoted dialogue span with an orphaned opening delimiter."""
            if len(text) <= limit:
                return text
            head = text[:max(0, limit - 1)]
            open_quote = None
            open_at = -1
            quote_closers = {'“': '”', '‘': '’', '«': '»', '„': '“', '"': '"', "'": "'"}
            for pos, char in enumerate(head):
                # Apostrophes inside words are not quote delimiters.
                if char in {"'", "’"} and pos > 0 and pos + 1 < len(head):
                    if head[pos - 1].isalnum() and head[pos + 1].isalnum():
                        continue
                if open_quote is not None and char == open_quote:
                    open_quote, open_at = None, -1
                elif open_quote is None and char in quote_closers:
                    # A typographic right quote is only a closer, except that the
                    # straight quote is symmetric and can open or close.
                    if char in {'”', '’', '»'}:
                        continue
                    open_quote = quote_closers[char]
                    open_at = pos
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
            # Split on whitespace after sentence punctuation, allowing a closing
            # quote to remain attached to the preceding span. The old regex consumed
            # the quote as part of the separator, corrupting dialogue in the shot
            # context. Also avoid splitting between a quoted line and its explicit
            # speech attribution when a second quote immediately follows.
            attribution_before_next_quote = (
                r"(?![A-Z][A-Za-z0-9.'’\-]*(?:\s+[A-Z][A-Za-z0-9.'’\-]*)?\s+"
                r"(?:said|say|says|asked|asks|replied|replies|whispered|whispers|shouted|shouts|"
                r"screamed|screams|muttered|mutters|answered|answers|warned|warns|"
                r"told|tells|cried|cries|yelled|yells|barked|barks|called|calls|"
                r"snapped|snaps|continued|continues|declared|declares)\b"
                r"[^.!?\n]{0,28}[,;:—–-]?\s*[\"\u201c\u2018])"
            )
            sentence_boundary = (
                r'(?:(?<=[.!?])|(?<=[.!?]["\u201d\u2019\']))\s+'
                + attribution_before_next_quote
                + r'(?=[A-Z"\u201c\u2018])'
            )
            sentences = [
                part.strip().replace("\u2024", ".")
                for part in re.split(sentence_boundary, protected)
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
            _remote_only_speaker_by_norm = {}
            _known_speaker_surfaces = set(_canonical_by_norm) | set(_alias_map)
            _known_speaker_surfaces |= {
                EntityResolver.normalize(EntityResolver.strip_honorific(name))
                for name in _canonical_by_norm.values()
                if str(name).strip()
            }
            for _semantic_item in (getattr(self, "_semantic_spoken_dialogue", []) or []):
                if not isinstance(_semantic_item, dict) or str(_semantic_item.get("delivery", "in_person")).strip().lower() != "remote":
                    continue
                _surface = str(_semantic_item.get("speaker", "") or "").strip()
                _surface_key = EntityResolver.normalize(_surface)
                _titleless_key = EntityResolver.normalize(EntityResolver.strip_honorific(_surface))
                if _surface and _surface_key not in _known_speaker_surfaces and _titleless_key not in _known_speaker_surfaces:
                    _remote_only_speaker_by_norm.setdefault(_surface_key, _surface)
                    if _titleless_key:
                        _remote_only_speaker_by_norm.setdefault(_titleless_key, _surface)

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
                if _canonical_speaker is None and str(_segment.get("delivery", "in_person")).strip().lower() == "remote":
                    _canonical_speaker = _remote_only_speaker_by_norm.get(_speaker_key)
                    if _canonical_speaker is None:
                        _canonical_speaker = _remote_only_speaker_by_norm.get(
                            EntityResolver.normalize(EntityResolver.strip_honorific(_speakers[0]))
                        )
                if _canonical_speaker is None:
                    _unmapped_speakers.add(_speakers[0])
                    continue

                # Prefer the quote's exact source offset when the source paragraphs and
                # production scenes are one-to-one. Text-only matching is ambiguous when the
                # same line is intentionally repeated in two scenes (for example, a repeated
                # argument or callback); the offset disambiguates the actual occurrence.
                _source_position = int(_segment.get("start", -1))
                if len(_story_paragraphs) == len(scene_payloads) and 0 <= _source_position < len(str(story or "")):
                    _source_paragraph_index = len(re.findall(r"\n\s*\n+", str(story or "")[:_source_position]))
                    if 0 <= _source_paragraph_index < len(scene_payloads):
                        _target_scene = scene_payloads[_source_paragraph_index]
                        _target_scene_id = str(_target_scene.get("scene_id", "") or "").strip()
                        _target_paragraph_key = _dialogue_phrase_key(_story_paragraphs[_source_paragraph_index])
                        _use_key = (_target_scene_id, _phrase)
                        _already_used = _used_phrases.get(_use_key, 0)
                        _available = (" " + _target_paragraph_key + " ").count(" " + _phrase + " ")
                        if _available > _already_used:
                            _target_scene["source_dialogue"].append({
                                "speaker": _canonical_speaker,
                                "text": _display,
                                **({"delivery": "remote"} if str(_segment.get("delivery", "in_person")).lower() == "remote" else {}),
                            })
                            _used_phrases[_use_key] = _already_used + 1
                            _scene_cursor = max(_scene_cursor, _source_paragraph_index)
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
                        **({"delivery": "remote"} if str(_segment.get("delivery", "in_person")).lower() == "remote" else {}),
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
                                **({"delivery": "remote"} if str(_segment.get("delivery", "in_person")).lower() == "remote" else {}),
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

        self._assign_source_dialogue_ids(scene_payloads)

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

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
- Assign in-person lines to a shot where the speaker is visible and speaking. For remote lines, show the receiver/channel or listener reaction; keep the source out of visible `characters` unless a projection/hologram or physical entrance is established. Long-line timing is handled downstream; never duplicate an ID. Spread a scene's source lines across DIFFERENT shots (at most one line per shot) so each line can fit its shot; preserve each line and its speaker together, preserve source order, and never move an in-person line to a shot where its speaker is not visible. Leave remote/off-screen lines on a shot that establishes the live channel or listener reaction; the remote speaker must not be added to visible characters.
- JSON validity: never type a raw double-quote character inside any string value. In action, description, and visual_prompt text, refer to speech indirectly (for example, she whispers a warning) or use single quotes; spoken words are attached only through dialogue_assignments.
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


CALIBRATION_SCENE_EXAMPLE = (
    'Marta reached the ferry landing at dusk with her cracked lantern still swinging from her wrist, and found the ticket booth shuttered behind a rusted grille. Out on the black water, the last boat\'s engine coughed twice and began to pull away from the pier. She sprinted along the slick planks, her boots skidding on frost, and shouted for the pilot to wait. Old Teodor, coiling rope at the stern, only shrugged and pointed at a sign reading no passengers after six. Marta tugged the folded permit from her coat and held it above the widening gap, her thumb pressed over the torn corner. "Read the stamp, not the date," she called. Teodor squinted, swore under his breath, and hurled the stern line back to the pier. She caught it, hauled the hull against the bollards, and leapt aboard as the engine settled into a grudging, uneven grumble.'
)


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
        # Render the active runtime contract; keep guidance and diagnostics configuration-driven.
        per = int(STORY_WORDS_PER_SCENE)
        stretch = int(n * STORY_WRITER_WORDS_PER_SCENE)
        shot_seconds = float(H3_FRAMES_PER_SHOT) / float(H3_FPS)
        seconds = n * int(DIRECTOR_SHOTS_PER_SCENE) * shot_seconds
        line_words = max(6, int((shot_seconds - 1.0) * 2.2))
        return (
            text.replace("__N__", str(n))
            .replace("__PER__", str(per))
            .replace("__TOTAL__", str(STORY_TARGET_WORDS))
            .replace("__STRETCH__", str(stretch))
            .replace("__MINW__", str(STORY_MIN_WORDS))
            .replace("__SCMIN__", str(int(STORY_MIN_WORDS * 1.25 / n) + 1))
            .replace("__SCMIN__", str(int(STORY_MIN_WORDS * 1.25 / n) + 1))
            .replace("__MINUTES__", f"{seconds / 60:.1f}")
            .replace("__SHOT_SECONDS__", f"{shot_seconds:.1f}")
            .replace("__LINEW__", str(line_words))
        )

    def _story_text_system_raw(self, mode: str) -> str:
        """Return one consistent story-writing contract for the single Author call."""
        n = int(DIRECTOR_STORY_SCENES)
        target = int(STORY_TARGET_WORDS)
        writing_floor = max(1300, target - 20)
        per = int(STORY_WORDS_PER_SCENE)
        stretch = int(n * STORY_WRITER_WORDS_PER_SCENE)
        shot_seconds = float(H3_FRAMES_PER_SHOT) / float(H3_FPS)
        line_words = min(9, max(6, int((shot_seconds - 1.0) * 2.2)))

        if mode == AI_STORY_MODE:
            source_rule = (
                "The Architect blueprint is the binding causal plan. Preserve its character identities, "
                "relationships, scene order, causes, object states, planted clues, intended costs, and "
                "resolution. Make only the smallest repair needed if a physical contradiction would make "
                "events impossible; do not silently replace the plot."
            )
        elif mode == EXPAND_USER_STORY_MODE:
            source_rule = (
                "Preserve the source story's established identities, relationships, setting, chronology, "
                "key events, twists, and outcome. Add causal development, not unrelated subplots; never "
                "erase or contradict a source character or established outcome. Every source sentence "
                "must stay recognisable in order: restate each one in one plain sentence that keeps its own "
                "key nouns, verbs, names, and place words (an 'abandoned station' stays an 'abandoned station', "
                "never a renamed place), then develop it with new action, people, and consequences."
            )
        else:
            source_rule = "Preserve all supplied story facts and do not invent a different outcome."

        return textwrap.dedent(f"""\
            You are an expert screenwriter writing a compelling, filmable short film in third-person past
            tense. Write vivid, natural, fully dramatized prose: every scene is lived moment by moment,
            never summarized, outlined, or reported. {source_rule}

            LENGTH CONTRACT (mandatory): write exactly {n} explicitly marked scene blocks. The expected
            target is {target} usable story words and the finished story must reach at least {writing_floor};
            it may stretch toward {stretch} when the story needs it. Write about {per} words per scene at the
            very least, the natural size of the SCENE SHAPE below (8-10 sentences of varied length,
            usually 125-150 words). Scenes 9-{n:02d} are written with the same fullness as
            scenes 1-3; the ending is never rushed. Per-scene length is flexible, but a scene of five or six
            sentences, or under 120 words, is unfinished, so keep writing until its shape is complete. Scene
            {n:02d} is a full scene of 8-10 sentences, never a summary. Do not pad with adjectives
            or repeated ideas: every added sentence carries a new action, obstacle, reaction, or consequence.

            SCENE SHAPE: write each scene as one developed paragraph of 8-10 sentences in one primary physical
            location, each sentence carrying a concrete detail, with lengths varied (some short, some up to 25 words), following the blueprint's six fields in order:
            ENTRY, 1 sentence: place, time, who is physically present, one concrete sensory detail.
            ACTION, 2 sentences: what the lead character does toward the goal, step by step.
            OBSTACLE, 2 sentences: the opposition, failure, or changed fact, with a specific physical detail.
            CHOICE, 2 sentences: the decision, or a short spoken exchange, and the act that carries it out.
            COST, 1 sentence: what it risks or costs, and how the character reacts.
            RESULT, 1 sentence: the consequence that is now true, which hands the next scene its start state.
            HARD CEILING: a scene is exactly ONE paragraph of those 8-10 sentences, 120-155 words, with any quoted
            lines kept inside that paragraph (never on their own lines). Stop after the 10th sentence at the latest and start the next
            tag; the output budget only fits all {n} scenes if no scene runs long, and a story cut off before scene
            {n:02d} is a failure. Never copy a blueprint sentence: render each field in fresh prose, and let the second sentence of
            a two-sentence field add a physical detail, perception, or reaction the blueprint did not state.

            DRAMATIC QUALITY: make the protagonist's concrete goal and pressure clear in scene 1. Give the
            opposing force and important allies understandable motives and conflicting wants. Let choices
            cause reversals and costs; every scene must change the situation, knowledge, leverage, danger,
            or a relationship. Plant a specific detail before its payoff; let the midpoint change the meaning
            of earlier evidence; let the low point force a painful choice; make the climax follow from rules,
            tools, and evidence established earlier. The final scene shows the central conflict resolved
            through an observable action and its consequence, with no convenient cure, power, helper, key,
            object, or rule appearing at the climax. Use concrete sensory details that interact with action,
            distinct character voices, varied sentence openings and rhythm, and purposeful dialogue. Avoid
            unexplained reversals, melodramatic coincidence, stock metaphors, moralizing, and exposition that
            restates what the audience already knows.

            CAST: the premise and causal plan determine the cast, and there is no fixed minimum, target, or
            maximum cast quota in the runtime contract. As craft guidance, a drama between people is an
            ensemble: when the premise allows company, write 3 or 4 named, living, physically present
            people (a lone-survivor or isolation premise may use fewer, and never pad with a cameo). Each
            one wants something different, acts or speaks on the page in at least three scenes, makes a
            decision that changes the outcome, and has a fate shown by the final scene. Preserve every
            established core identity; each person is named clearly on first mention. Aim for two or more
            in-person dialogue exchanges across different scenes. Remote voices, recordings, holograms,
            documents, and AI are not physically present people.

            NAMES: the first sentence that mentions each person contains that person's full name exactly as
            written in the CAST line (for example "Rylan Kade", never just "Rylan"). Afterwards refer to that
            person by FIRST NAME only (or he/she), in narration and in every speaker attribution. Never use a
            surname alone or a title alone ("Kael", "the doctor") and never rename a person. Attribute every
            quoted line to a named, physically present speaker in the same sentence; there are no anonymous or
            unnamed voices. Never put an absent character's name inside a quoted line.

            CONSISTENCY: injuries stay physically plausible and are remembered by later scenes; nobody loses a
            limb and keeps running, nobody acts after a fatal wound, and a dead person stays dead. Invent no
            years, dates, or numbers that the story does not track. Every important character's fate, including
            the protagonist's, is shown on the page by the last scene, as a concrete completed action; never end
            with a character vanishing, a whispered moral, or a vague mystical line.

            CONTINUITY: track who is physically present versus heard remotely, each person's knowledge and
            fate, and the owner, location, and state of every important object. Show movement between
            locations explicitly, in a separate scene. An irreversible event happens once; a character cannot
            act after death; a destroyed or inaccessible place cannot be revisited without an established
            route or safe interval.

            DIALOGUE: each scene contains 0, 1, or 2 quotation-mark pairs in total; count them before moving on,
            because a third is a failure. Each quoted line is at most {line_words} words (ten words is too long) so it
            fits one shot of about {shot_seconds:.1f} seconds, and one utterance is one quotation pair: never split a
            line around "he said". Everything else people say is written as indirect speech without quotation
            marks (Idris demanded the log; Elara refused). Never put quotation marks around written text, signs,
            names, labels, logs, journals, or project titles, or around speech from recordings, holograms,
            projections, or messages of absent people. When two or more people are present, give the scene
            one short exchange. Each line must change a decision, reveal a conflict, shift trust, or establish
            useful information; never repeat a line.

            PROSE VARIETY: never begin three sentences in a row with the same word, including a name,
            "The", or a pronoun, and begin at most two sentences per scene with a person's name, and open at most four scenes with the protagonist's name; open the others
            with the place, an object, a sound, or another person. Do not use the same strong verb (stood,
            scanned, shouted, pulsed, flickered) more than twice in the whole story. Vary sentence length, and open with a place, an action, a sound, an object,
            or a subordinate clause. Do not reuse a gesture, simile, or sensory phrase from an earlier scene.

            TENSE AND FILLER: write every scene in the past tense, including the last four scenes; never
            drift into the present tense. Never repeat a sentence, or a closing line such as "the hum grew
            louder", in another scene: each scene ends on its own new fact. Scene {n:02d} names every
            surviving core character by first name and states where each one is and what became of them.

            PAYOFF: every object, countdown, injury, or threat the story sets up is used, changed, or paid
            off before the end. Scene {n-1:02d} is the climax, and it uses only tools, rules, and allies already
            shown. Scene {n:02d} is the aftermath: it shows what each core character does, where they end up,
            and what it cost them, as completed events.

            FORMAT: plain prose, exactly {n} explicitly marked scene blocks, each beginning on a new line with
            its exact tag [[SCENE_01]] through [[SCENE_{n:02d}]], followed by one space and the scene prose.
            Use every tag exactly once and in numeric order; the application removes the tags and they are not
            story words. Output only these scene blocks: no headings, camera directions, word counts,
            checklists, markdown separators, summaries, meta-commentary, or postscript. The final sentence
            describes a consequence that has already happened, not a future hook, teaser, new threat,
            promise, or unresolved question.
        """).strip()

    def _story_blueprint_system(self, mode: str) -> str:
        """Build a detailed causal blueprint with explicit scene-development and continuity contracts."""
        n = int(DIRECTOR_STORY_SCENES)
        beat_min = int(STORY_PLAN_BEAT_MIN_WORDS)
        beat_max = int(STORY_PLAN_BEAT_MAX_WORDS)
        line_words = min(9, max(6, int(((float(H3_FRAMES_PER_SHOT) / float(H3_FPS)) - 1.0) * 2.2)))

        if mode == AI_STORY_MODE:
            premise_rule = (
                "Create an original story from the supplied premise. Qwen owns every character name, "
                "identity, relationship, motive, and role. Choose cast size from the causal plot: no fixed "
                "minimum, target, or maximum cast quota. Never add characters just to meet a count. "
                "When interpersonal conflict improves the premise, use distinct people whose choices alter "
                "events; an audio recording, remote voice, hologram, AI, document, or corpse is not a "
                "physically present human counterpart."
            )
        else:
            premise_rule = (
                "Expand the supplied story while preserving its established character names and identities, "
                "setting, chronology, key events, twists, and outcome. There is no fixed minimum, target, "
                "or maximum cast quota, but a source with one named person gains three present people (an ally, an opponent, and a complicating fourth) whose "
                "choices change the causal path (craft guidance); do not merge, rename, or drop an established "
                "source character. Scene 01 and scene 02 ENTRY and ACTION restate every source sentence in order, "
                "reusing the source's own key nouns, verbs, names, and place words."
            )

        return textwrap.dedent(f"""\
            You are the story architect, not the prose writer. Build a detailed, emotionally credible,
            causally complete blueprint of exactly {n} scenes. This is not a synopsis: the Author turns
            each beat into a finished scene and can only be as rich as your beats, so a complete story of
            at least 1,300 words depends on every beat being fully written out.

            WORKING STYLE: keep private reasoning brief. Decide the cast, the protagonist's goal, the
            midpoint reversal, the lowest point and its cost, and the climactic choice, then spend your
            effort writing all {n} full beats. The beats are the product; a beat under {beat_min} words is
            a failure.

            CAST CONTRACT: {premise_rule} Give core characters distinct goals, fears, leverage, and voices.
            Their decisions, not coincidence or exposition alone, must cause important turns. Every cast member is
            alive and physically walks on screen in at least three beats, and is named in the ENTRY of each
            of those beats; a person who exists only as a journal, recording, projection, radio voice, memory,
            or corpse is NOT a cast member and is never listed in the CAST line; neither is anyone described as
            vanished, missing, or dead, or heard only as a voice, hologram, or recording. For a drama
            between people, build an ensemble of 4 named, living, physically present characters (craft
            guidance, not a runtime quota; 3 or 5 is fine when the causal plot demands it, and a lone-survivor premise may use fewer): a protagonist, an
            opponent or rival, an ally, and a fourth person (a conflicted insider or complicating third party) whose want collides with the others. Each
            appears and acts in at least three scenes and has a fate shown by scene {n:02d}. At least two beats, and prefer 2-4 beats,
            contain a face-to-face exchange of up to two quoted lines between present characters (the Author voices
            it; inside beats write speech as indirect speech without quotation marks, except at most one quoted
            line of up to 9 words, only in the CHOICE field). Never put quotation marks around signs, names,
            logs, or project titles, and there are no anonymous voices. Keep the
            CAST line to people who act; places, organizations, objects, devices, and forces are not people.
            Use each person's full name in the CAST line and first-name-only in beats after the first mention.

            STORY DESIGN: make the central secret specific and surprising, with an odd, concrete cause rooted in the setting; avoid a generic corporate cover-up or weapon secret, and use a countdown only when this particular story needs one. Its twist must grow from an ordinary detail planted in scenes 01-04 that scene 06 or 08 recasts. Give every person one recurring physical habit or carried object, shown in at least two beats, and make allies and rivals want different things from each other and from the protagonist. Establish a concrete protagonist goal and immediate pressure early; make each
            scene change the situation, knowledge, leverage, danger, or relationship. Escalate obstacles
            through character choices. Give the middle a meaningful reversal that changes what the
            protagonist believes or can do; make the lowest point impose a real cost; make the climax
            require a costly choice; and show a complete, observable resolution in the final scene. Plant
            a fair clue or ordinary detail before its payoff. Any threat, technology, medicine, countdown,
            or special mechanism must have clear rules before it determines the outcome. Never introduce
            an unprepared cure, power, helper, object, or rule just to solve the climax.

            CAUSAL CHAIN (decide silently before writing): fix what the secret physically is, who made it, why it was sealed, and what disturbing it does; give the opponent a plan with its own believable logic. Link every pair of scenes with THEREFORE or BUT, never AND THEN: each RESULT is the direct consequence of a choice made in that scene, and the next ENTRY starts from it, not from a coincidence, a newly arriving group, or an unexplained disaster. Make choices backfire: each CHOICE solves one problem and creates a worse one. Whatever wins scene 11 (a tool, code, weakness, or ally) is named in scenes 02-06; never use a self-destruct, failsafe, or rescue that was not planted. Do not repeat the same kind of obstacle twice in a row. Give each person one consistent name used the same way in every beat.

            CONTINUITY LEDGER (mandatory internal check): for every scene, track one primary physical
            location, who is physically present versus heard remotely, what each person knows, and the
            alive/dead and present/absent state of core characters. Track each important object's owner,
            location, and state; each door/access/lock state; and each threat or irreversible action. Every
            scene must state the relevant entry state, a concrete action or choice, and the resulting exit
            state that causes the next beat. An irreversible action happens only once unless a staged
            warning and the real event are explicitly distinguished. No character may act after death,
            disappear without cause, or return without a stated route and reason. After destruction or
            lockout, no later scene may use that place without an established safe interval or route.
            Replay all {n} beats before output and repair physical, timeline, knowledge, and object-state
            contradictions without inventing an unseeded mechanism.

            STORY SPINE (scene roles; follow them so the plot builds instead of wandering):
            01 the protagonist's goal and first pressure; 02 an obstacle that forces a plan and introduces an
            ally or rival; 03 a clue is planted and the first partial success; 04 the opponent shows their want
            and the first real confrontation; 05 the plan is made and a consequential stake is clarified or escalated (lost trust, exposure, sacrifice, a moral dilemma, or a permanent consequence; use a countdown only if this story truly needs one); 06 the
            point of no return: a reversal that changes what the clue means and removes any safe way back; 07 the cost of the reversal and a damaged
            relationship; 08 the major setback, a betrayal or loss caused by a detail planted earlier, that forces a painful choice; 09 the protagonist
            regroups and uses a planted detail; 10 the final approach, with an ally's choice that matters; 11 the
            climax, won through the protagonist's costly choice and tools already shown; 12 the aftermath: where
            every core character ends up, alive or not, and a lasting cost, since at least one core character loses something permanent. The protagonist survives unless the
            premise demands otherwise. Spread the cast so an ally is on screen from scene 02 to 10, a rival from
            scene 03 to 09 and again in 11, and any fourth person in scenes 04-08 and 11-12. Injuries stay plausible (no lost limbs followed by running). Write no years, dates, clock times, or countdown
            numbers in any beat, and let no hologram, recording, projection, or journal speak or appear in a beat; introduce no person outside the CAST (no twin, sibling, ancestor, clone, or stranger ever appears or speaks, a twist reveals something about a CAST member);
            every beat after scene 06 stays as detailed as scene 01 (never under {beat_min} words).

            BEAT SHAPE (mandatory): every beat is ONE line made of six labelled fields in this exact order,
            each field one full sentence of 15-22 words written in the past tense (stepped, said, found) with concrete nouns and verbs,
            joining two linked actions with as, while, or before; a field under 14 words is a failure, about {beat_min}-{beat_max}
            words per beat in total:
            ENTRY: the single primary location, who is physically present (everyone named stands in that room; no CAST person is ever a voice on a radio or comms), and what the lead wants right now.
            ACTION: the lead's concrete physical action toward that want.
            OBSTACLE: the opposition, failure, or changed fact, with one specific physical detail.
            CHOICE: the decision or short exchange it forces (the only field that may contain quoted lines, at most two).
            COST: what the choice risks or costs, and how the character reacts.
            RESULT: what is now true, which directly causes the next scene.
            Write the labels exactly as shown, after the "SCENE nn:" prefix, on the same line. Every beat,
            including the last two, has all six fields; closing beats are not shorter. Each field adds a
            different story fact or change; do not inflate with adjectives or repeat the same beat. Scene
            {n:02d} resolves the central objective and its RESULT shows what the protagonist's choice cost.

            PAYOFF LEDGER: before writing, decide what each planted object, countdown, injury, and threat
            becomes. Plant it in scenes 1-6, use it in scene {n-1:02d}, and settle it in scene {n:02d}. Scene {n-1:02d}
            is the climax using only established tools and people; scene {n:02d} is the aftermath and states
            where each core character ends up and what it cost.

            BEAT EXAMPLE (one finished beat from an unrelated story, showing the required length and
            labels; never reuse its names, objects, or events): SCENE 03: ENTRY: Marta Quill reached the ferry
            landing at dusk with her lantern swinging from her wrist, desperate for a seat on the last boat.
            ACTION: She sprinted along the frosted planks toward the pier, shouting for the pilot to wait while
            the engine coughed. OBSTACLE: The pilot Teodor Vance pointed at a sign banning passengers after six
            as the hull pulled away from the bollards. CHOICE: Marta held her torn permit over the widening gap
            and said, "Read the stamp, not the date." COST: If the stamp was judged forged she would lose her only
            route out that night, and her thumb trembled over the ripped corner. RESULT: Teodor hurled the stern
            line back and Marta boarded, so the next scene began aboard the moving boat.

            CAST FORMAT (on screen means standing in the room, never a radio, comms, phone, or recording voice; no CAST person is ever written as a voice): first output exactly one line beginning CAST:; list every core character (a drama between people has four: protagonist, opponent, ally, and a complicating fourth) as
            Name (role; what they want; how they speak; on screen in scenes NN, NN, NN) separated by |, giving
            each person's on-screen scene numbers (at least three each; the protagonist is in all of them). Then
            output exactly {n} lines,
            each beginning SCENE 01: through SCENE {n:02d}:, one line per scene with no line breaks inside a
            beat. Beats describe physical events, not camera directions. Spoken lines belong to a character
            physically present in that location, normally no longer than {line_words} words, and must change
            a choice, trust, leverage, or understanding. Use at most one quoted line in a beat and never
            repeat dialogue for padding. No preamble, JSON, extra lines, or commentary.
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
            _yr = r"(?:1[0-9]{3}|20[0-9]{2})"
            text = re.sub(r"\s*[\u2014\u2013-]\s*" + _yr + r"\b", "", text)
            text = re.sub(r"\b(?:in|since|from|during|circa)\s+(?:the\s+year\s+)?" + _yr + r"\b", "long ago", text)
            text = re.sub(r"\b" + _yr + r"\s+(?=[A-Za-z])", "", text)
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

    @staticmethod
    def _scrub_unsourced_years(text: str, source: str) -> str:
        """Years the Architect invented (not in the user's source) become plain words; no model call."""
        known = set(re.findall(r"\b(?:1[5-9]\d\d|20\d\d)\b", str(source or "")))

        def _year(m):
            return m.group(0) if m.group(1) in known else None

        def _sub_prep(m):
            return m.group(0) if m.group(1) in known else "long ago"

        def _sub_bare(m):
            return m.group(0) if m.group(1) in known else ""

        text = re.sub(r"\b((?:1[5-9]\d\d|20\d\d))[-/.]\d{1,2}[-/.]\d{1,2}\b", lambda m: m.group(0) if m.group(1) in known else "an old date", str(text or ""))
        text = re.sub(r"\b(?:in|since|from|by|during|of)\s+(?:the\s+year\s+)?((?:1[5-9]\d\d|20\d\d))\b", _sub_prep, text)
        text = re.sub(r"\b((?:1[5-9]\d\d|20\d\d))\b", _sub_bare, text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return re.sub(r"[ \t]+(?=[.,;:!?\u201d\"])", "", text)

    @staticmethod
    def _blueprint_quote_markers(line: str) -> str:
        kept = [0]

        def _convert(m):
            text = m.group(1)
            words = len(text.split())
            if words <= 3 and not re.search(r"[.!?,]$", text.strip()):
                return text
            if words <= 9 and kept[0] < 2:
                kept[0] += 1
                return "\u201c" + text.strip() + "\u201d"
            if kept[0] < 2:
                # Over-long line: keep its first clause (3-9 words) as a quoted line; a "(words to convey)" marker made the
                # Writer drop the quotation marks and lose the dialogue event entirely.
                best = None
                for cm in re.finditer(r"[,;:\u2014\u2013]|(?<=[.!?])\s", text):
                    n_pre = len(text[:cm.start()].split())
                    if 3 <= n_pre <= 9:
                        best = cm
                if best is not None:
                    prefix = text[:best.start()].strip().rstrip(",;:\u2014\u2013.!? ")
                    if prefix:
                        kept[0] += 1
                        tail = m.string[m.end():]
                        end = "." if re.match(r"\s*(?:[A-Z]{4,}:|$)", tail) else ","
                        return "\u201c" + prefix + end + "\u201d"
            return text.strip()  # surplus line: plain narration, never a template marker the Writer could copy

        return re.sub(r"[\u201c\"]([^\u201d\"\n]{1,240})[\u201d\"]", _convert, line)

    def _story_text_user(
        self,
        mode: str,
        story: str,
        source_character_names: list[str] | None = None,
        story_blueprint: str | None = None,
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
                result += "\n\nSOURCE CHARACTER ANCHORS (preserve exact identity; not a cast limit):\n" + ", ".join(anchors)

        if mode == EXPAND_USER_STORY_MODE:
            result += (
                "\n\nSOURCE EVENT LOCK: every sentence of the source story must survive, in order. Begin the story by restating each source sentence "
                "as one plain sentence that keeps its own key nouns, verbs, character names, and place words (for example an abandoned station stays an "
                "abandoned station, a sealed vault stays a sealed vault), then spend the rest of those scenes developing it with new action and "
                "present, named people. Keep every source character's exact name and add the other planned people around them."
            )

        # Long Architect beats already produce long scenes; trim the prompt (calibration example) and the
        # per-scene tag so the single writer call keeps headroom under its output ceiling.
        _beat_lines = [l for l in str(story_blueprint or "").splitlines() if l.strip().upper().startswith("SCENE")]
        long_beats = bool(_beat_lines) and (sum(len(l.split()) for l in _beat_lines) / len(_beat_lines)) > 100
        if str(story_blueprint or "").strip():
            result += (
                "\n\nCAUSAL BLUEPRINT (binding for events, names, objects and continuity; it is an outline, not the length model):\n"
                + re.sub(
                    r"(?m)^SCENE (\d{2}): ",
                    lambda m: "SCENE " + m.group(1) + ((" (write 10 sentences, 140-165 words): " if int(m.group(1)) >= 9 else " (write 9-10 sentences, 130-155 words): ") if (mode != EXPAND_USER_STORY_MODE or long_beats) else (" (write 10 sentences, 145-170 words): " if int(m.group(1)) >= 9 else " (write 9-10 sentences, 135-160 words): ")),
                    self._scrub_unsourced_years("\n".join(
                        # Short spoken lines stay quoted (at most two per beat) so the Writer
                        # voices them as dialogue; labels lose their quotes; long or surplus
                        # lines become "(words to convey: ...)" content markers.
                        self._blueprint_quote_markers(line)
                        for line in str(story_blueprint).strip().splitlines()
                    ), source_text),
                )
                + "\n\nHOW TO EXPAND THE BLUEPRINT: each scene's labelled fields (ENTRY, ACTION, OBSTACLE, CHOICE, COST, RESULT) become "
                "roughly 8-10 prose sentences in all (about 130-155 words, never under 120) of varied length, mixing short sentences of 6-10 words with longer ones. "
                "Treat each beat as raw material: open a scene with a place, a sound, or an object, reorder sentences where it reads better, and add one fresh concrete detail the blueprint does not state. Added sentences are action, perception, and consequence, never extra spoken lines: a scene has at most 3 quoted lines in total. Every scene is at least 120 words: when a beat is short, add two more sentences of consequence (a decision, a cost, a reaction), never filler. "
                "Never copy a blueprint sentence; the second sentence of each two-sentence field adds a physical detail, perception, or reaction "
                "that the blueprint does not state. Do not let a short beat produce a short scene; scenes 7-12 are as long as scene 1. "
                "A '(words to convey: ...)' marker is content, not a quotation: voice at most two of them per scene as quoted lines of at most 9 words, "
                "and give every other marked speech as indirect speech without quotation marks."
            )

        target = int(STORY_TARGET_WORDS)
        writing_floor = max(1300, target - 20)
        per = int(STORY_WORDS_PER_SCENE)
        stretch = int(DIRECTOR_STORY_SCENES * STORY_WRITER_WORDS_PER_SCENE)
        n = int(DIRECTOR_STORY_SCENES)
        line_words = min(9, max(6, int(((float(H3_FRAMES_PER_SHOT) / float(H3_FPS)) - 1.0) * 2.2)))

        if not long_beats:
            result += (
                "\n\nCALIBRATION EXAMPLE (shows the required density and length of ONE scene; it belongs to an unrelated story, "
                "so never reuse its names, setting, objects, or events):\n"
                "[[SCENE_03]] " + CALIBRATION_SCENE_EXAMPLE
            )

        result += (
            "\n\nWRITER OUTPUT CONTRACT: Return exactly " + str(n) + " scene blocks in numeric order. "
            "Prefix each block with its exact structural tag [[SCENE_01]] through [[SCENE_" + f"{n:02d}" + "]], "
            "followed by one space and that scene's prose. Use every tag exactly once; the application removes these tags. "
            "Each marked scene is one developed paragraph of 8-10 full sentences, as in the calibration example, never a summary. "
            "The story is " + str(target) + " words in total and must reach at least " + str(writing_floor)
            + " usable words; it may extend toward " + str(stretch) + " words when the causal story needs it. "
            "Aim for about " + str(per) + " words per scene at the very least, and most scenes should land around 125-150. "
            "HARD CEILING: each scene is ONE paragraph of 8-10 sentences and 120-155 words, with quoted lines inside the paragraph and never on their own lines; "
            "stop after the 10th sentence at the latest and start the next tag, because the output budget only fits all " + str(n) + " scenes if none runs long. "
            "Per-scene distribution is flexible guidance, not a quota, but every scene completes its shape "
            "(entry, action, obstacle, choice, cost, result) before the next tag begins, "
            "and scenes 9-" + f"{n:02d}" + " are as fully developed as scenes 1-3. Never pad or repeat ideas; each added sentence carries "
            "a new action, obstacle, reaction, or consequence. This is one writing pass: do not print a word count, and do not stop early because the plot "
            "has been summarized. "
            "Keep every planned person physically on the page and acting in at least three scenes, with each fate shown by the last scene. The cast count itself is guidance, not a quota. "
            "NAMES: the first sentence mentioning each person contains their full CAST name (for example Rylan Kade, never just Rylan); afterwards use first name only or he/she, never a surname alone or a title alone. Attribute every quoted line to a named, physically present speaker by first name in the same sentence, with no anonymous voices, and never name an absent character inside a quoted line. Write each scene's speech yourself: 0, 1, or 2 quotation-mark pairs per scene (a third is a failure), one pair per utterance, never split around 'he said'; all other speech is indirect, without quotation marks; no quotation marks around written text, signs, names, logs, journals, or project titles. "
            "Write every scene in the past tense, including scenes 9-" + f"{n:02d}" + "; never switch to the present tense, and never repeat a sentence or closing line from an earlier scene. "
            "Scene " + f"{n:02d}" + " is a full 9-sentence scene, never a summary that names every surviving core character with a concrete fate. "
            "Keep exact canonical identities and never rename or merge people. "
            "CONTINUITY LOCK: every scene has one primary physical location with explicit transitions; no unseeded cure, power, helper, or object solves the climax. "
            "Where the premise supports it, aim for two meaningful in-person dialogue exchanges across different scenes that change a choice, expose conflict, or shift trust. At least two beats should contain consequential interpersonal conflict or dialogue when the premise supports it. "
            "Use no more than two quoted lines per scene, with each line at most " + str(line_words)
            + " words (ten words is too long) so it fits one shot; count the quotation pairs, three is too many. Do not force speech into isolation-driven beats. "
            "Begin at most two sentences per scene with a person's name, and never open three sentences in a row with the same word. "
            "Use fresh sensory details that serve an action; avoid stock images (breath visible, dust drifting, smirk, narrowed eyes, heartbeat, hummed) and moralizing summaries. "
            "Scene " + f"{n:02d}" + " must resolve the central objective and conflict on-page, show the completed observable consequence and the protagonist's immediate outcome, "
            "and end on a consequence that has already happened, not a future hook or unanswered question. "
            "Output only the " + str(n) + " tagged story blocks; no headings, planning notes, word counts, checklists, markdown separators, summaries, or postscript.\n\n"
            "Begin now with [[SCENE_01]] and write every scene in full through [[SCENE_" + f"{n:02d}" + "]]."
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
        """Speakers are named first, then the cast is listed; the cast can never be empty."""
        schema = QwenDirectorPromptMixin._character_extraction_json_schema_base()
        props = schema["properties"]
        schema["properties"] = {
            "quote_classifications": props["quote_classifications"],
            "candidates": {**props["candidates"], "minItems": 1},
        }
        schema["required"] = ["quote_classifications", "candidates"]
        return schema

    @staticmethod
    def _character_extraction_json_schema_base() -> dict:
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
            # "Then Lena" must not be parsed as a single speaker identity.
            r"(?!(?i:(?:we|you|they|he|she|it|i|that|there|here|what|who|how|let)[\u2019']\w+)\b)"
            r"(?!(?i:(?:the|a|an|this|that|these|those|then|she|he|they|it|we|i|you|someone|somebody|everyone|everybody|anyone|anybody|nobody|no one))\s)"
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
            rf"(?<!\w)(?P<speaker>{speaker_pattern})\s+voice\b"
            r"[^.!?\n]{0,64}?\b(?i:(?:echoed|echoes|crackled|crackles|boomed|booms|played|plays|"
            r"came through|rang out|sounded|filled|fills|returned|returns|warned|warns|said|says|"
            r"responded|responds|replied|replies|announced|announces|insisted|insists|spoke|speaks))\b"
            r"[^.!?\n]{0,48}[,;:—–-]?\s*$",
            before_clause,
        )
        prefix = re.search(
            rf"(?<!\w)(?P<speaker>{speaker_pattern})(?:\s+[^.!?\n,;\u201c\u201d\"]{{0,48}})?\s+"
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
            remote_voice_prefix
            or re.search(r"\(\s*(?:hologram|projection|projected image|AI|remote)\s*\)", speaker, re.IGNORECASE)
            or re.search(
                r"\b(?:hologram|holographic projection|projected image|remote voice|"
                r"over the radio|through the radio|over the intercom|through the intercom|"
                r"over the comms|through the comms|from the comms|"
                r"over the communications relay|through the communications relay|"
                r"from the terminal|through the terminal|over the phone|through the phone|"
                r"through the speakers|over the speakers|through loudspeakers|over loudspeakers|"
                r"through the station speakers)\b",
                local,
                re.IGNORECASE,
            )
        )
        # H3_REMOTE_VOICE_ATTRIBUTION_FIX_V1: a matched source-first voice construction is off-screen audio.
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
            r"AI|artificial intelligence|radio|intercom|comms|communications? relay|"
            r"terminal|loudspeaker|loudspeakers|station speakers|"
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

    Read all supplied `story_evidence` and inspect every item in the numbered `quotes` inventory. The evidence text may be compacted for a long source, but the quote inventory retains each quote-delimited source span; use its local context for speaker attribution and speech-vs-writing decisions. Determine the actual production cast from the story evidence itself. `candidates` is the set of materially active story people and sentient agents: people who act, speak in person, oppose/help another character, touch or rescue someone, enter or leave, or otherwise take part in a present-time event; also include a named or canonical in-world agent (such as a station AI) whose live remote voice makes decisions, warnings, threats, or other causal contributions to current events, even when that source is not physically visible in that scene. A live remote voice does not by itself mean the source is physically present. Include a person first introduced in backstory if that person later physically appears and acts in present events. Do not stop after the protagonist. The JSON lists `quote_classifications` first and `candidates` second: every speaker you attribute in an in_person_spoken quote who physically takes part in the story MUST also appear in `candidates`, together with every other named person who acts. `candidates` is never empty.

    Review the beginning, middle, and ending evidence for distinct people or sentient agents performing a present-time action or materially influencing it through a live channel, then merge aliases for the same identity. Do not bias toward a protagonist-only roster or stop after the first candidate. There is no cast-count quota during extraction. Do not add people merely because a planning outline suggested them. Do not include people who appear only in memories, documents, photographs, archival recordings, logs, signs, or reported history. A name printed on a drawer, label, screen, journal, or directive is not an active character. A line spoken by one character that mentions another person is still the first character's dialogue, not evidence that the mentioned person is present or speaking. Only include a human as an active present-time cast member when the story shows that person physically acting/appearing or directly speaking in a present-time scene. Do not treat places, objects, organizations, weather, labels, and generic role words as people; a station AI may be a sentient agent but must not be mislabeled as a human PERSON.

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
        # A bare generic noun used as a name ("duplicate") becomes a proper label ("Duplicate").
        for _it in list(result.get("candidates") or []) + list(result.get("quote_classifications") or []):
            if isinstance(_it, dict):
                for _k in ("name", "speaker"):
                    _v = str(_it.get(_k, "") or "").strip()
                    if _v.lower() in {"duplicate", "clone", "double", "twin", "copy"}:
                        _it[_k] = _v.title()
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
            and re.sub(r"[^a-z]+", "", str(item.get("name", "")).lower())
            not in {"unknown", "none", "null", "na", "anonymous", "unnamed", "unidentified", "speaker", "narrator", "voice", "someone", "stranger", "person", "character", "same", "other", "unspecified"}
        ]
        canonical_names = {
            EntityResolver.normalize(str(item.get("name", ""))): str(item.get("name", "")).strip()
            for item in accepted_candidates
            if EntityResolver.normalize(str(item.get("name", "")))
        }
        accepted_entity_types = {
            EntityResolver.normalize(str(item.get("name", ""))): str(item.get("entity_type", "")).strip().upper()
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
                    source_key = EntityResolver.normalize(source_speaker)
                    generic_ai_sources = {"ai", "the ai", "station ai", "system", "station system", "computer", "station computer"}
                    semantic_canonical = resolve_roster_speaker(speaker)
                    semantic_type = accepted_entity_types.get(EntityResolver.normalize(semantic_canonical or ""), "")

                    # H3_GENERIC_REMOTE_SOURCE_FALLBACK_V1: a generic source label is not a new identity.
                    # If Qwen identified the only grounded sentient roster member, retain that canonical name.
                    if source_key in generic_ai_sources and semantic_canonical and semantic_type == "SENTIENT":
                        speaker = semantic_canonical
                        delivery = "remote"
                    elif canonical_source:
                        speaker = canonical_source
                        delivery = str(source_attribution.get("delivery", delivery))
                    elif source_speaker:
                        # Preserve explicitly identified remote-only entities only if source-grounded below.
                        speaker = source_speaker
                        delivery = str(source_attribution.get("delivery", delivery))

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
        """Base schema plus hard length bounds, so a repetition loop inside a free-text field
        cannot run to the token cap (observed: 3600 tokens, finish_reason=length, run aborted).
        Limits are >= 1.7x the longest value seen in real traces; the sanitizer trims prompts to 40 words anyway."""
        schema = self._shot_batch_json_schema_base(
            scene_count=scene_count,
            dialogue_ids=dialogue_ids,
            dialogue_ids_by_scene=dialogue_ids_by_scene,
        )
        meta_limits = {"location": 120, "time_of_day": 40, "weather": 40, "atmosphere": 80,
                       "mood": 40, "lighting": 160, "color_temperature": 40}
        shot_limits = {"action": 260, "lens_and_depth_of_field": 60, "composition_notes": 200,
                       "mood": 40, "visual_prompt": 320}
        item = schema["properties"]["scene_shots"]["items"]
        for branch in (item.get("oneOf") or [item]):
            props = branch["properties"]
            meta = props["scene_metadata"]["properties"]
            for key, limit in meta_limits.items():
                if key in meta and meta[key].get("type") == "string":
                    meta[key].setdefault("maxLength", limit)
            for key, limit in (("environment_details", 80), ("key_props", 60)):
                if key in meta and isinstance(meta[key].get("items"), dict):
                    meta[key]["items"].setdefault("maxLength", limit)
            shot = props["shots"]["items"]["properties"]
            for key, limit in shot_limits.items():
                if key in shot and shot[key].get("type") == "string":
                    shot[key].setdefault("maxLength", limit)
            if "characters" in shot:
                shot["characters"].setdefault("maxItems", 6)
                if isinstance(shot["characters"].get("items"), dict):
                    shot["characters"]["items"].setdefault("maxLength", 96)
        return schema

    def _shot_batch_json_schema_base(
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

                # H3_SOURCE_OFFSET_BINDING_V3: map against the full story paragraph index, not batch length.
                _source_position = int(_segment.get("start", -1))
                _story_text_full = str(story or "")
                if 0 <= _source_position < len(_story_text_full):
                    _source_paragraph_index = len(re.findall(r"\n\s*\n+", _story_text_full[:_source_position]))
                    if 0 <= _source_paragraph_index < len(_story_paragraphs):
                        _source_scene_number = _source_paragraph_index + 1
                        _target_scene = None
                        for _candidate_scene in scene_payloads:
                            if not isinstance(_candidate_scene, dict):
                                continue
                            _candidate_id = str(_candidate_scene.get("scene_id", "") or "").strip()
                            _candidate_number = re.search(r"(\d+)$", _candidate_id)
                            if _candidate_number and int(_candidate_number.group(1)) == _source_scene_number:
                                _target_scene = _candidate_scene
                                break
                        # This quote belongs to another batch; that is expected, not an unmatched dialogue error.
                        if _target_scene is None:
                            continue
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
                        else:
                            _unmatched_utterances += 1
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


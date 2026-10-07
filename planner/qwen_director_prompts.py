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
- Copy dialogue with its original capitalization and punctuation exactly as listed in `source_dialogue`; never lower-case it.
- Put a spoken line in the shot where it is actually said in the story, with its speaker visible and listed in `characters`; the `action` of that shot must show that character speaking. Do not spread unspoken shots across a line, and do not assign a line to a shot whose action is about something else.
- A sentence such as a character's action, breath, expression, realization, or narration is NEVER dialogue, even when it begins with the character's name.
- Every source utterance may be used only once. The only exception is splitting one longer source utterance into exact contiguous pieces across adjacent shots. Never repeat the same source line in multiple shots.
- Every dialogue_events[].speaker MUST exactly match one supplied canonical character name.
- Relational canonical identities are valid when supplied. Use the exact canonical identity, not a bare role.
- Never invent a new generic speaker such as "man", "woman", "boy", "girl", "doctor", "guard", or "officer" when that surface is not a supplied canonical identity or validated semantic alias.
- Preserve spoken text exactly; never paraphrase, summarize, invent, or transform narration into speech.
- When the story contains explicit spoken dialogue or script-style dialogue, copy only those spoken lines. If there is no explicit spoken-dialogue anchor, return dialogue_events as an empty array.
- The user payload may include a `source_dialogue` whitelist containing exact source utterances. Treat it as authoritative: dialogue_events may use only those utterances (or exact contiguous pieces of them) and must never invent a new line from narrative prose. If the supplied scenes contain no applicable source utterance, return dialogue_events as an empty array.

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
HARD FORMAT
- Third-person past tense, plain prose only: no markdown, asterisks, italics, headings, or labels.
- Exactly six paragraphs separated by one blank line. Each paragraph is ONE filmable scene:
  one location, one continuous stretch of time, one visible turn.
  Keep the complete story within the required 420-560 word contract. Prioritize a complete, causally coherent story over brevity.
  Do not stop while the story is still under the minimum. Before ending, silently check that the story has six paragraphs
  and is at least 420 words. If it is short, expand existing story beats with concrete action, consequences, sensory detail,
  or character decisions rather than adding filler or repetition. Do not pad the story merely to satisfy the word count,
  and do not truncate a necessary ending just to stay near an arbitrary target.
- The sixth paragraph is a full settled aftermath, not a compressed summary.

STORY PLAN (write these seven lines in your reasoning before drafting; never in the output)
WANT: what the protagonist is trying to do right now, and where.
RESISTANCE: what or who stands in the way, why, and how the protagonist's first attempt meets resistance.
CAST: who is physically present and what each person wants. The cast size is yours: one, two, or several. Include another person whenever their goal, action, or knowledge would change what the protagonist does. A solitary story is right when isolation or the environment is itself the pressure.
SETUP: a fact, object, skill, relationship, or constraint shown early that will matter later, if the story calls for one.
TURN (reversal): what the protagonist learns or sees that changes what they must do. It grows out of something already shown. New information is a turn only when it forces the protagonist to decide or act differently.
CHOICE: two outcomes that cannot both be kept, and the price paid for the one chosen. The price is real; the result may be bittersweet or bleak as well as hopeful.
RESULT: the concrete, settled situation after the choice.

STRUCTURE (one beat per paragraph; each beat is shown by what people do)
- Paragraph 1: open mid-action in a specific place. The protagonist has a concrete physical goal and meets an immediate obstacle. Use sensory detail that belongs to this setting and serves the action.
- Paragraph 2: the protagonist makes the first consequential attempt, and it meets resistance: it fails, succeeds at a cost, or succeeds and exposes a new problem. Either way the situation changes.
- Paragraph 3: the situation, or another person, answers that attempt. The protagonist adapts or is forced to, and the pressure rises. End on the consequence of what just happened.
- Paragraph 4: hold the TURN until here; paragraphs 1-3 build the pressure it overturns. Deliver it through action, evidence, another person's deeds or words, or a physical consequence, consistent with every earlier paragraph.
- Paragraph 5: the CHOICE, made through physical action, with its price made visible. Use what the story has already established, so that the way out comes from earlier paragraphs.
- Paragraph 6: one place, one moment. Two or three concrete actions show the direct result of the choice and end on a completed past-tense action in a settled situation. Nothing is left pending: no 'until', no 'yet to come', no next mission, no looming return.
- The protagonist causes, chooses, or misjudges something early enough that later events follow from that action or its consequence. Characters act, choose, refuse, misjudge, cooperate, deceive, or react to concrete consequences, and the protagonist makes meaningful decisions under pressure, whether it comes from another person or from the environment. Nothing important happens without a cause the story has shown.
- Establish facts, objects, relationships, and constraints before relying on them, and pay them off where the story naturally calls for it.
- Every named person is on screen. Someone known only through a document, label, message, or memory is evidence, not a character.

CHECK (in your reasoning, after drafting and before output): six paragraphs; at least 420 words; each paragraph's event follows from an earlier one; the turn rests on something shown earlier; the price is visible; the last sentence reports something that has already happened.

PROSE
- Concrete verbs and nouns over adjectives. Vary sentence length; mix short blunt sentences with longer ones.
- Do not begin more than two sentences in a row with the same word.
- Show emotion through what hands, eyes, breath, and posture do. Never name the emotion as a mood word.
- Avoid cliches such as "heart pounded", "the weight of", "pulse steadied", "a testament to", "shiver down",
  "sent a chill", "palpable", and the word "suddenly".
- Keep the prose filmable: convey private thoughts, fears, memories, and realizations through visible action,
  physical objects, or spoken lines rather than naming the internal state. Keep backstory brief and subordinate to the present action.
- Prefer a final physical image that echoes or transforms an important concrete detail from the opening when the story naturally supports it.
- Plain prose means plain text: no emoji, symbols, or decorative characters anywhere.
- Stop after the sixth paragraph. Write nothing after the story ends.

PEOPLE
- Every named person must be physically present in at least one scene and must act or speak there.
  Recordings, radios, screens, notes, memories, and holograms are evidence only: they never carry a character name and
  never count as a speaking character. Do not name a sound or machine as if it were a person.
- Give every recurring character one stable canonical name and use only that name.
""".strip()

    def _story_text_system(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:
            return (
                "You are the screenwriter of short cinematic stories for MiniMax H3. "
                "You write one complete, original, film-ready story from the premise. "
                "Before drafting, write the STORY PLAN lines below in your reasoning, then output only the finished prose.\n\n"
                + self._STORY_CRAFT_RULES
                + "\n\nPREMISE FIDELITY\n"
                "- Every concrete fact in the premise (the protagonist's profession, the place, the weather, "
                "the discovery) must appear explicitly in the story, using the premise's own key nouns. "
                "Show the profession through skilled actions the protagonist performs.\n\n"
                "CAST AND DIALOGUE\n"
                "- You decide the cast; there is no target size. Use the CAST line of your plan. Every recurring character materially affects the protagonist's goal, resistance, turn, choice, or consequence, wants something of their own, has one stable canonical name, and acts on screen. A character may change the story through action alone.\n"
                "- Dialogue is optional. Use short lines of direct speech in straight double quotes with a clear speaker only where speech carries conflict, information, refusal, deception, negotiation, or a relationship. Speakers want different things and interrupt, deflect, or lie instead of explaining the plot. Never add dialogue merely because several characters are present, and never invent a voice for a solitary story.\n"
                "- A log, recording, radio, screen, label, or alarm is evidence, never a speaking character, and never replaces a character's own action or decision.\n\n"
                "Output only the finished story prose."
            )

        if mode == EXPAND_USER_STORY_MODE:
            return (
                "You are the story editor-writer for MiniMax H3. You expand a short source story into a complete, "
                "film-ready cinematic story without changing who it is about or what happened. "
                "First identify silently the source's characters, setting, events, and outcome, then build the causal "
                "spine around them with the STORY PLAN lines below, written in your reasoning before drafting. "
                "Output only the finished prose.\n\n"
                + self._STORY_CRAFT_RULES
                + "\n\nSOURCE FIDELITY (strict)\n"
                "- Preserve every established source character that remains part of the story, the source setting, the source events in order, and the source outcome.\n"
                "- SOURCE CHARACTER ANCHORS are established identities to preserve, not a cast limit. There is no fixed cast size. The expanded cast may contain one person, two people, or any larger number according to Qwen's causal judgment. Keep established canonical names stable.\n"
                "- Qwen may introduce additional named, relational, or descriptive recurring characters when the expanded narrative genuinely requires them. "
                "A new character must materially affect the goal, resistance, reversal, choice, or consequence; do not add decorative cast.\n"
                "- Do not invent a relative, colleague, mentor, friend, or other relationship merely as backstory. If a new relationship matters to the causal story, establish that character explicitly on screen.\n"
                "- Dialogue is optional. Use short lines of direct speech in straight double quotes with a clear speaker only where speech carries conflict, information, refusal, deception, or a relationship. A character may change the story through action alone. Never add a character or a line of dialogue solely to create speech; a story with one person needs none.\n"
                "- If the source supports a present counterpart, let that person's action, choice, relationship, or spoken information drive the turn.\n"
                "- Add only cause, resistance, evidence, escalation, and consequence that deepen the source-specific causal chain. Preserve the source's setting and events in order; do not replace it with a generic genre plot.\n"
                "- Do not write the expansion as a compressed six-paragraph synopsis. Each paragraph should develop a concrete scene beat with visible action and consequence. If the story reaches its apparent ending before the minimum length, deepen the existing causal beats and aftermath rather than stopping early or introducing a disconnected subplot.\n\n"
                "Output only the finished expanded story prose."
            )

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
        if mode == EXPAND_USER_STORY_MODE:
            anchors = list(dict.fromkeys(
                str(value).strip()
                for value in (source_character_names or [])
                if str(value).strip()
            ))
            if anchors:
                result += (
                    "\n\nSOURCE CHARACTER ANCHORS (established characters to preserve; not a cast limit):\n"
                    + ", ".join(anchors[:16])
                )
        result += (
            "\n\nOUTPUT CONTRACT:\n"
            "Exactly six paragraphs. Keep the complete story within the required 420-560 word contract. Do not stop while under 420 words; "
            "if short, continue developing the existing causal beats with concrete action, consequences, sensory detail, character decisions, and a fully developed aftermath rather than filler or repetition. "
            "Do not compress the final consequence into a brief summary. Plain prose only. End on a completed consequence, not a future hook."
        )
        return result


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
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": 128,
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
                                "maxItems": 4,
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
                "spoken_dialogue": {
                    "type": "array",
                    "maxItems": 128,
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "speaker": {"type": "string"},
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
    You are a strict character/entity extraction component for a cinematic production planner.
    Return JSON only. Analyze the supplied story and classify stable human/sentient identities that can receive
    an identity lock. There are THREE valid character identity types:

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
    If `is_character=true`, `entity_type` MUST be PERSON, CHARACTER, or SENTIENT. Do not use EVENT,
    LOCATION, OBJECT, ROLE, or OTHER for a character identity. If `is_character=false`, do not emit a
    named_character, relational_character, or descriptive_character identity type.
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
    """).strip()

        user_payload = json.dumps(
            {
                "story": self._compact_story_context(story, DIRECTOR_STORY_CONTEXT_CHARS),
                "deterministic_character_hints": candidate_hints[:32],
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

            if not scene_payload.get("characters"):
                raise RuntimeError(
                    f"Scene {scene_payload.get('scene_id') or '<unknown>'} has no canonical character binding for shot planning."
                )
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

        source_dialogue = []
        extract_dialogue = getattr(self, "_extract_story_spoken_segments", None)
        if callable(extract_dialogue):
            try:
                source_dialogue = [
                    str(segment.get("display", "") or segment.get("text", "") or "").strip()
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

        payload = {
            "story_context": self._compact_story_context(story, DIRECTOR_SHOT_STORY_CONTEXT_CHARS),
            "characters": compact_characters,
            "visual_language": language,
            "reference_visual_analysis": visual_context,
            "scenes": scene_payloads,
        }
        if source_dialogue:
            payload["source_dialogue"] = source_dialogue

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

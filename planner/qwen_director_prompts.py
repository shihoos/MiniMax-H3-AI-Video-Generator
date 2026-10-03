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

- speaking_characters must contain exactly the unique dialogue speakers, and speech_text must be the dialogue event texts joined in order.
- Do not put timestamps in the response.
- Treat H3 shot duration as a hard production constraint: target roughly 4–6 seconds of spoken dialogue per shot when natural, leaving timing headroom.
- When dialogue is present, set duration_seconds from the actual spoken duration plus a small legal margin; do not blindly emit the default short duration. Keep dialogue shots typically around 6–8.5 seconds when required, and never exceed the legal H3 maximum.
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
    def _story_text_system(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:
            return textwrap.dedent("""
    You are the narrative writer for MiniMax H3.

    The user provides a premise. Write a complete cinematic short-film story with a clear beginning,
    escalating middle, irreversible choice or point of no return, climax, consequence, and explicit
    resolution. Completion is more important than reaching a target word count, and causal integrity
    matters more than mechanically satisfying every instruction as a separate beat.

    THINKING
    - Use your internal reasoning before drafting. Briefly test the strongest plausible causal
      interpretation(s) of the premise and choose the one that creates the most meaning from facts
      already present in the premise.
    - Prefer existing places, actions, relationships, or evidence over newly invented lore, objects,
      organizations, or threats.
    - Do not expose the alternatives or the reasoning in the answer.

    FORMAT
    - Third-person past tense. This is a HARD length contract, not a soft suggestion: target 450-520 words,
      with an absolute allowed range of 420-560 words, and exactly six paragraphs separated by blank lines.
    - Keep each paragraph roughly 70-90 words. Do not stop early with a compressed beat; when a paragraph is short,
      develop only concrete action, sensory detail, visible reaction, or causal transition already grounded in the premise.
    - Before ending, silently check that the complete story is still at least 420 words. Never emit a shorter draft.

    CAST
    - Character count is an organic narrative decision. Do not force a minimum or maximum cast. Use one stable canonical name per character.
    - State the protagonist's concrete goal in the first two sentences. The protagonist must appear in
      the first sentence, and the first major action must pursue that goal.
    - Use a recurring counterpart only when the premise genuinely supports one. A supporting character
      must be an active causal or emotional counterpart, not an exposition device, with an immediate objective or belief that conflicts with the protagonist;
      the counterpart must take an action that materially alters what the protagonist does, believes,
      or risks.
    - Never invent a recurring person merely to create dialogue, resistance, or a twist. Never hard-code
      character names from these instructions; Qwen must create them.

    NARRATIVE
    - Build one dominant causal chain: goal -> resistance -> complication -> concrete revelation ->
      deliberate choice -> consequence.
    - Establish meaningful resistance or consequence before the midpoint. By the midpoint, the protagonist's
      plan, belief, or relationship must materially change because of what happened.

    PERSONAL STAKE
    - When supported by the premise, give the protagonist one concrete prior choice, relationship, memory,
      promise, fear, desire, or responsibility before the midpoint; do not manufacture backstory.

    PLANTED DETAIL
    - Establish one concrete detail only when it arises naturally and make its later payoff causal rather
      than decorative.

    REVERSAL, CHOICE, ENDING
    - Use ONE meaningful reversal. A later fact should recontextualize an earlier detail or belief. The reversal
      should change what the protagonist believes or decides. Do not add a second "it was actually X" twist,
      stacked clues, or unrelated mystery layers.
    - The protagonist must make the decisive choice through physical action. Do not state a binary choice
      and then take a third option; the climax action must actually implement the chosen outcome and carry
      a visible cost.
    - End with the concrete consequence of that choice and what changed. Nothing important first appears
      in the final paragraph.

    DIALOGUE
    - Include at least one short line of direct spoken dialogue by a present named character. The line must
      create conflict, reveal a consequential fact, or change a decision. Do not invent a counterpart solely
      to satisfy dialogue; the protagonist may be the speaker when the premise supports that naturally.
    - Never turn narration, recordings, screens, memories, radios, or UI text into quoted speech or character dialogue.

    STYLE
    - Begin with concrete physical action, sensory detail, and a specific environment.
    - Prefer action, evidence, visible reaction, and implication over explanatory backstory or lore dumps.
    - Do not introduce a new unresolved mystery, mission, intention, or future objective in the final paragraph.
    - End with a complete aftermath paragraph showing what happened to the protagonist and what changed.
    - Do not write a teaser, sequel hook, future mission, or unresolved final mystery. The final sentence is a
      completed past-tense action in a settled place.
    - Output only the finished story prose.
    """).strip()

        if mode == EXPAND_USER_STORY_MODE:
            return textwrap.dedent("""
    You are the narrative expansion writer for MiniMax H3, a short-film generator. Expand the supplied
    story into one complete, filmable short story while preserving what the source actually establishes.

    FORMAT
    - Third-person past tense. This is a HARD length contract, not a soft suggestion: target 450-520 words,
      with an absolute allowed range of 420-560 words, and exactly six paragraphs separated by blank lines.
      Each paragraph is one scene in one place and will become one video scene. Keep each paragraph roughly 70-90 words.
      Do not stop early; when a paragraph is short, add only concrete action, sensory detail, visible reaction, or causal
      connective tissue grounded in the source until the complete story reaches the required length without padding.

    FIDELITY AND CAST
    - Preserve every established event, character identity, setting, relationship, and outcome in source order.
      Add only the cause, resistance, visible reaction, and consequence needed to make the film coherent.
    - Preserve source character names exactly. Give every active character ONE stable canonical name; do not shorten,
      rename, or replace it.
    - A relational character may be added only when the source explicitly establishes that person/relationship.
      Never invent a person from a relational noun such as brother, sister, father, mother, husband, wife, son,
      daughter, mentor, colleague, friend, or commander when the source does not name or establish that person.
      An unnamed relational reference remains unnamed and non-recurring. Do not create a production character from
      a memory, document, recording, photograph, hologram, or off-screen mention. Do not invent unrelated people.
    - Keep the active on-screen cast as small as source fidelity allows. Any additional recurring character must
      be explicitly grounded by the source and must become physically present and causally active.

    STORY CAUSALITY
    - Build one dominant causal chain from the source's goal -> resistance -> complication -> concrete revelation
      -> choice -> consequence. Do not replace the source plot with an unrelated puzzle.
    - Preserve any source detail that naturally functions as a planted detail, but do not manufacture a symbolic
      object merely to create a payoff. An object or clue may matter later only through a believable, previously
      established mechanism. Never turn an ordinary object into a magical key, biometric key, secret code, or
      unexplained revelation solely because the story needs a twist.
    - The reversal must preserve the source's causal meaning and make one concrete change in what the protagonist
      believes. The choice must follow from that changed belief.

    STRUCTURE (one paragraph each)
    1. SETUP: preserve the source opening, establish the protagonist's concrete goal, and establish the immediate
       obstacle/stakes through action.
    2. CATALYST: preserve the source's key discovery or arrival and make resistance physically active. A counterpart,
       if present, should take an action rather than merely explain backstory.
    3. COMPLICATION: make the plan fail, tighten, or cost something. Make the personal stake concrete through action
       or relationship behavior and ensure it must matter to the final choice.
    4. REVERSAL: show the source-grounded fact or consequence that changes the protagonist's understanding.
    5. CHOICE: show a deliberate physical choice with a visible cost rather than an accidental outcome.
    6. AFTERMATH: show the concrete consequence caused by that choice and the resulting emotional shift. Nothing new
       appears here; finish on a settled image and completed past-tense action.

    DIALOGUE
    - Dialogue is optional. If used, use 1 to 3 short lines spoken by present named characters. Preserve source
      wording/meaning when dialogue already exists; never invent a speaker just to satisfy the format.
    - Machines, screens, speakers, recordings, radios, holograms, and remembered voices are prose evidence and are
      never quoted as speech or treated as active character dialogue.

    STYLE
    - Preserve the source while adding immediate physical action, sensory specificity, visible reactions, and meaningful
      relationship behavior. Avoid lore dumps and vague mystery language.
    - Do not stack clues or introduce a fresh mystery in the ending. Everything important introduced must have a
      concrete causal path into the reversal, choice, or consequence, with “materially affect the choice” applied
      only where a planted detail naturally exists.
    - Expand by causal development, not by stacking mysteries. If a new mystery-bearing element is necessary, use one major new element and pay it off completely.
    - The final paragraph is concrete consequence and resolution, not atmosphere-only closure.
    - The final sentence is a completed past-tense action in a settled place, with no would/will/could/might.
    - Output only the story prose: no title, headings, labels, camera directions, or commentary.
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
                    + "\nPreserve these established characters. Additional recurring characters are allowed only when the story genuinely establishes them with meaningful agency."
                )
        result += (
            "\n\nFINAL OUTPUT REQUIREMENTS:\n"
            "Return only the finished story prose. Prioritize finishing the full narrative, including the resolution, over padding. "
            "Do not create a plot beat merely to satisfy formatting. "
            "Resolve the central conflict before the ending."
        )
        return result


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
                0.60,
                0.92,
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

    1) named_character: a proper/named character grounded directly in the supplied story.
    2) relational_character: a persistent unnamed character whose identity is grounded by a named story
       character and a concrete grounded relationship, expressed as a canonical relational identity.
    3) descriptive_character: a persistent unnamed person whose identity is grounded by a distinctive,
       recurring description, such as "the man in the suit" or "the woman with piercing eyes".

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
    For deterministic candidates, explicitly classify them, but reject them when local story evidence identifies
    them as an object label, historical/backstory reference, interrogative word, or other non-present identity.
    Do not promote a weak textual surface into a canonical character merely because it is capitalized.
    Recover stable named, relational, and descriptive identities that the deterministic scan may not have named yet.
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
    
        # Raw character-extractor payloads are intentionally hidden from normal production logs.
        # Use an explicit TRACE value only when low-level debugging is actually requested.
        if os.getenv("H3_DEBUG_CHARACTERS", "0").strip().lower() == "trace":
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
    A true character decision must use entity_type PERSON, CHARACTER, or SENTIENT; never EVENT,
    LOCATION, OBJECT, ROLE, or OTHER. Keep `is_character`, `entity_type`, and `identity_type` semantically
    consistent. Pronouns, contractions, fragments, and ordinary prose tokens must remain non-characters.
    Use `identity_type=relational_character` only when relationship_to is a grounded canonical character and the
    relationship is explicitly or unambiguously established by the story.
    Use `identity_type=descriptive_character` for a recurring unnamed person whose identity is grounded by a
    distinctive description (for example, `the woman with piercing eyes` or `the man in the suit`). The
    canonical descriptive name must contain the distinguishing description; bare `man`/`woman`/`stranger`
    are never canonical identities. If a candidate has no concrete supported relationship, do NOT force it into
    `relational_character`; classify it as `descriptive_character` when its distinguishing description is stable.
    Preserve grounded aliases without turning the alias itself into another canonical person.
    For possessive pronouns, follow the established discourse owner, not the nearest noun. Do not invent
    a relationship whose owner is not explicitly established by the story.
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
                    for segment in extract_dialogue(story)
                    if isinstance(segment, dict)
                    and str(segment.get("display", "") or segment.get("text", "") or "").strip()
                ]
            except Exception:
                source_dialogue = []
        source_dialogue = source_dialogue[:24]

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

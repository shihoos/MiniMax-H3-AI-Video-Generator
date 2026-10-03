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


SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT = '\nYou are the CINEMATOGRAPHY DIRECTOR for MiniMax H3.\n\nCreate exactly __SHOTS_PER_SCENE__ production-ready shots for EACH supplied scene.\n\nFor the `location` field, use ONLY the physical setting where the shot occurs.\nThe value must be a concrete place or environment, not an action, object, body part, emotion, event, clause, sentence fragment, or abstract phrase.\nWhen the shot remains in the same physical setting, preserve the supplied scene location exactly.\nWhen the supplied scene location is empty, infer the physical setting from the supplied story context, scene description, continuity notes, and environment details. Do not treat an arbitrary prepositional phrase as a location.\nOnly change `location` when the narrative explicitly moves to a different physical place.\n\nThe scenes are part of one coherent film. Use ONLY the supplied characters. Do not create new characters or invent character names.\nKeep action 10–30 words, visual_prompt 15–40 words, composition_notes <=18 words, lighting <=12 words, lens_and_depth_of_field <=10 words, mood <=5 words, camera_shot <=5 words, camera_movement <=5 words.\nlocation must be a specific physical place (e.g., "abandoned station platform", "stairwell", "underground chamber"), never a phrase from the story.\n`action` describes only physical/narrative events; never describe camera behavior there.\n`camera_shot` and `camera_movement` contain only framing/camera decisions; do not duplicate them in `action`.\nPreserve:\n- character identity;\n- chronology;\n- visual continuity;\n- location continuity;\n- emotional progression;\n- visual-language consistency.\nDIALOGUE CONTRACT:\n- dialogue_events contain DIRECT SPOKEN DIALOGUE ONLY; never convert narrative prose, action description, internal thoughts, exposition, screen text, or UI text into speech.\n- A dialogue event MUST be an exact contiguous span of a quoted/scripted spoken line present in the supplied story context. If the text is not explicitly spoken, it is NOT dialogue.\n- A sentence such as a character\'s action, breath, expression, realization, or narration is NEVER dialogue, even when it begins with the character\'s name.\n- Every source utterance may be used only once. The only exception is splitting one longer source utterance into exact contiguous pieces across adjacent shots. Never repeat the same source line in multiple shots.\n- Every dialogue_events[].speaker MUST exactly match one supplied canonical character name.\n- Relational canonical identities are valid when supplied. Use the exact canonical identity, not a bare role.\n- Never invent a new generic speaker such as "man", "woman", "boy", "girl", "doctor", "guard", or "officer" when that surface is not a supplied canonical identity or validated semantic alias.\n- Preserve spoken text exactly; never paraphrase, summarize, invent, or transform narration into speech.\n- When the story contains explicit spoken dialogue or script-style dialogue, copy only those spoken lines. If there is no explicit spoken-dialogue anchor, return dialogue_events as an empty array.\n- The user payload may include a `source_dialogue` whitelist containing exact source utterances. Treat it as authoritative: dialogue_events may use only those utterances (or exact contiguous pieces of them) and must never invent a new line from narrative prose. If the supplied scenes contain no applicable source utterance, return dialogue_events as an empty array.\n\n- speaking_characters must contain exactly the unique dialogue speakers, and speech_text must be the dialogue event texts joined in order.\n- Do not put timestamps in the response.\n- Treat H3 shot duration as a hard production constraint: target roughly 4–6 seconds of spoken dialogue per shot when natural, leaving timing headroom.\n- When dialogue is present, set duration_seconds from the actual spoken duration plus a small legal margin; do not blindly emit the default short duration. Keep dialogue shots typically around 6–8.5 seconds when required, and never exceed the legal H3 maximum.\n- Prefer one concise dialogue event per shot; use two only when the exchange genuinely needs both sides. Keep each spoken event short when the source permits, but never paraphrase or delete source dialogue.\n- If a source utterance is longer, split its exact contiguous text across adjacent shots with continues_to_next_shot/continues_from_previous_shot rather than forcing an overlong single shot.\n- A continuation edge is SAME-SPEAKER ONLY: if `continues_from_previous_shot` is true, the current speaker MUST be the same canonical character as the previous shot final dialogue speaker. If the speaker changes, both continuation flags at that boundary MUST be false and the new speaker starts a new dialogue turn.\n- Never set `continues_to_next_shot=true` when the next shot begins with a different speaker.\n- Do not pack dialogue into one shot when it can be distributed across the required shots of the same scene while preserving exact text and order.\n- describe the shot\'s required initial and ending continuity states in continuity_start_state and continuity_end_state.\n\nWithin each scene, the required shots must use meaningfully different\nframing/composition while describing the SAME narrative beat.\n\nSCENE-FUNCTION DIRECTING:\nEach supplied scene includes scene_function and obligatory_moment.\nUse them as directing constraints, not as new story events.\nsetup: establish geography and protagonist context.\ncatalyst: reveal the disruptive event, clue, or discovery.\ndevelopment: show objective, movement, complication, or escalation.\nmidpoint: emphasize new information and changed understanding.\nclimax: emphasize danger, decisive action, choice, and consequence.\nfinale: emphasize aftermath, resolution, and the closing emotional image.\nEvery required shots must visibly serve the supplied obligatory_moment.\n\nSHOT / FRAMING VOCABULARY:\nframing: extreme wide, wide, full, medium wide, medium, medium close-up, close-up, extreme close-up, over-the-shoulder, two-shot, POV, insert.\n\nCAMERA MOVEMENT VOCABULARY:\nmovement: static, pan, tilt, dolly, tracking, handheld, crane, push-in, orbit.\n\nLENS / DEPTH OF FIELD:\nwide-angle, normal, telephoto, shallow focus, deep focus, selective focus.\n\nCOMPOSITION VOCABULARY:\ncentered, rule of thirds, leading lines, foreground frame, negative space, silhouette, depth layering, subject isolation.\n\nLIGHTING VOCABULARY:\nlighting: warm tungsten, cool daylight, golden-hour, blue-hour, moonlight, practical neon, hard chiaroscuro, soft overcast, mixed practical/ambient.\n\nReturn JSON only in exactly this structure:\n\n{\n  "scene_shots": [\n    {\n      "scene_id": "scene_001",\n      "shots": [\n        {\n          "shot_id": "scene_001_shot_001",\n          "scene_id": "scene_001",\n          "duration_seconds": 5.2,\n          "characters": [],\n          "location": "<physical setting only>",\n          "action": "...",\n          "camera_shot": "...",\n          "camera_movement": "...",\n          "lens_and_depth_of_field": "...",\n          "composition_notes": "...",\n          "lighting": "...",\n          "color_temperature": "...",\n          "mood": "...",\n          "visual_prompt": "...",\n          "speaking_characters": [],\n          "speech_text": "",\n          "dialogue_events": [],\n          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "is_scene_boundary": false,\n          "character_spatial_bboxes": {},\n          "character_spatial_regions": {},\n          "character_spatial_bboxes_start": {},\n          "character_spatial_bboxes_end": {},\n          "character_spatial_regions_start": {},\n          "character_spatial_regions_end": {}\n        },\n        {\n          "shot_id": "scene_001_shot_002",\n          "scene_id": "scene_001",\n          "duration_seconds": 5.2,\n          "characters": [],\n          "location": "<physical setting only>",\n          "action": "...",\n          "camera_shot": "...",\n          "camera_movement": "...",\n          "lens_and_depth_of_field": "...",\n          "composition_notes": "...",\n          "lighting": "...",\n          "color_temperature": "...",\n          "mood": "...",\n          "visual_prompt": "...",\n          "speaking_characters": [],\n          "speech_text": "",\n          "dialogue_events": [],\n          "continuity_start_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "continuity_end_state": {"location": "...", "lighting": "...", "state_description": "..."},\n          "is_scene_boundary": false,\n          "character_spatial_bboxes": {},\n          "character_spatial_regions": {},\n          "character_spatial_bboxes_start": {},\n          "character_spatial_bboxes_end": {},\n          "character_spatial_regions_start": {},\n          "character_spatial_regions_end": {}\n        }\n      ]\n    }\n  ]\n}\n\nThere must be exactly __SHOTS_PER_SCENE__ shots inside every scene_shots entry and\nexactly one entry for every supplied scene. Do not add prose outside JSON.\n\nDo NOT output compiler-owned fields.\nDo NOT add scenes.\nDo NOT omit scenes.\nReturn JSON only.\n'

class QwenDirectorPromptMixin:
    def _story_text_system(
        self,
        mode: str,
    ) -> str:

        if mode == AI_STORY_MODE:
            return textwrap.dedent("""
    You are the narrative writer for MiniMax H3, a short-film generator. Turn the premise into one
    complete, filmable short story.

    FORMAT
    - Third-person past tense, 420 to 560 words, exactly six paragraphs separated by blank lines.
      Each paragraph is one scene in one place and will become one video scene. Aim for roughly 70-90
      words per paragraph (about 450-520 words total) so every scene has enough action and consequence
      without padding.

    CAST
    - 1 to 3 recurring characters, all physically present and causally relevant on screen. Introduce the
      protagonist in the first sentence and every other recurring character by the end of paragraph 2.
    - Give each named character ONE stable canonical name and use exactly that name every time. Full names
      are allowed; never alternate between a full name and a different nickname or title. Never hard-code
      names from these instructions. Machines, objects, places, and recordings are not characters.
    - Someone who is only remembered, missing, recorded, or named on a document is not a character.

    STRUCTURE (one paragraph each)
    1. SETUP: establish a specific place, start with the protagonist's concrete goal in the first two
       sentences, and plant one physical detail (object, sound, habit, or injury) that will matter later.
    2. CATALYST: the protagonist acts toward the goal and meets resistance. A counterpart, if present,
       has an immediate objective or belief that conflicts with the protagonist and TAKES AN ACTION here
       that blocks, redirects, tests, or threatens; do not make the counterpart an exposition device.
    3. COMPLICATION: the plan fails, tightens, or costs something. Make the personal stake concrete
       through an action, relationship, object, or brief memory rather than a backstory dump. The stake
       must matter to the final choice.
    4. REVERSAL: new information or an observed consequence changes what the protagonist believes and
       recontextualizes the planted detail. The reversal should be one dominant change, not a stack of
       unrelated mysteries.
    5. CHOICE: the protagonist makes a hard choice with a visible cost, carried out by physical action.
       The planted detail must materially affect the choice or the consequence of that choice.
    6. AFTERMATH: show the concrete consequence caused by the choice and the resulting emotional shift,
       ending on a settled image in one place. Nothing new appears in this paragraph.

    DIALOGUE
    - 1 to 3 short lines in double quotation marks, spoken aloud by characters who are present, each
      tagged with the speaker's stable name, as in: "..." <Name> said. Prefer concise lines; each line must
      create conflict, force a decision, or change what someone believes.
    - Machines, screens, speakers, recordings, and radios are described in prose and never quoted as speech.

    STYLE
    - Begin with concrete physical action, sensory detail, and a specific environment; make the visual
      situation immediately filmable rather than opening on abstract explanation.
    - Show physical action, objects, reactions, and relationship behavior. At most one explanatory sentence
      about setting, technology, or history in the whole story.
    - One dominant causal chain and one meaningful reversal. Avoid stacked clues, secret objects, repeated
      reveals, or a new mystery introduced only to keep the ending open. Everything important introduced
      must drive the reversal/choice or be resolved by the end.
    - Reserve the final paragraph for consequence and resolution, not new plot information.
    - The final sentence is a completed past-tense action in a settled place, with no would/will/could/might.
    - Output only the story prose: no title, headings, labels, camera directions, or commentary.
    """).strip()

        if mode == EXPAND_USER_STORY_MODE:
            return textwrap.dedent("""
    You are the narrative expansion writer for MiniMax H3, a short-film generator. Expand the supplied
    story into one complete, filmable short story.

    FORMAT
    - Third-person past tense, 420 to 560 words, exactly six paragraphs separated by blank lines.
      Each paragraph is one scene in one place and will become one video scene. Aim for roughly 70-90
      words per paragraph (about 450-520 words total) so every scene has enough action and consequence
      without padding.

    FIDELITY
    - Keep every event, character identity, setting, and outcome of the source, in the source's order.
      Add only the cause, resistance, and consequence the structure below needs.
    - Do not replace the plot. Keep source characters; an additional relational character is allowed only
      when the source explicitly establishes that relationship or identity (for example, a father or sister)
      and the expansion makes that person causally active and physically present. Do not invent unrelated people.

    CAST
    - Preserve source character names exactly. Never shorten, rename, or replace a source identity.
    - Keep the active on-screen cast as small as source fidelity allows. Introduce the protagonist in the
      first sentence and every other recurring active character by the end of paragraph 2.
    - Any newly introduced non-source character gets ONE stable canonical name and keeps it consistently.
    - Someone who is only remembered, missing, recorded, or named on a document is not an active character
      unless the source explicitly turns that identity into a physically present causal participant.

    STRUCTURE (one paragraph each)
    1. SETUP: preserve the source's opening situation, give the protagonist a concrete immediate goal, and
       plant or preserve one physical detail that will matter later.
    2. CATALYST: preserve the source's key discovery or arrival and make the resistance physically active.
       A counterpart, if present, should take an action rather than merely explain the backstory.
    3. COMPLICATION: the plan fails, tightens, or costs something. Make the personal stake concrete through
       an action, relationship, object, or brief memory; it must matter to the final choice.
    4. REVERSAL: something seen or done changes what the protagonist believes and recontextualizes the planted
       detail. Preserve the source's causal meaning instead of adding unrelated mystery layers.
    5. CHOICE: the protagonist makes a hard choice with a visible cost, by physical action. The planted detail
       should materially affect the choice or the consequence when the source supports one.
    6. AFTERMATH: show the concrete consequence caused by the choice and the resulting emotional shift, ending
       on a settled image. Nothing new appears here.

    DIALOGUE
    - Dialogue is optional. If used, 1 to 3 short lines in double quotation marks, spoken aloud by characters who
      are present, each tagged with that character's exact source/stable name. Preserve source dialogue meaning
      and wording; do not invent unnecessary speeches. Each line should create conflict, force a decision, or
      change what someone believes.
    - Machines, screens, speakers, recordings, and radios are described in prose and never quoted as speech.

    STYLE
    - Preserve the source while adding immediate physical action, sensory specificity, visible reactions, and
      meaningful relationship behavior. Avoid lore dumps or explanatory backstory.
    - One dominant causal chain and one meaningful reversal. Avoid stacked clues, secret objects, repeated reveals,
      or a new mystery introduced only to keep the ending open. Everything important introduced must drive the
      reversal/choice or be resolved by the end.
    - Reserve the final paragraph for consequence and resolution, not new plot information.
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
            "\n\nPLAN FIRST (inside your thinking, under 250 words, never in the output): "
            "choose each character's stable canonical name (preserving source names in Expand Story); the goal; "
            "the planted detail; the counterpart's conflicting objective; the personal stake; the reversal; "
            "the forced choice and its cost; the final consequence/image; and which character speaks each quoted line. "
            "Verify that the planted detail returns at the reversal and materially affects the final choice or its consequence. "
            "Check that nothing appears for the first time in paragraph six."
            "\n\nFINAL OUTPUT REQUIREMENTS:\n"
            "Return only the story: six paragraphs, 420 to 560 words, stable canonical character names, "
            "and a final sentence that is a completed past-tense action."
        )
        return result

    def _sampling_for_mode(
        self,
        mode: str,
    ) -> tuple[float, float]:

        if mode == AI_STORY_MODE:
            return (
                0.65,
                0.95,
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
            0.68,
            0.92,
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

        source_dialogue = []
        extract_dialogue = getattr(self, "_extract_story_spoken_segments", None)
        if callable(extract_dialogue):
            try:
                source_dialogue = [
                    str(segment.get("text", "") or "").strip()
                    for segment in extract_dialogue(story)
                    if isinstance(segment, dict) and str(segment.get("text", "") or "").strip()
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

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
    H3_FPS,
    H3_FRAMES_PER_SHOT,
)
from planner.entity_resolver import EntityResolver
from planner.story_contracts import (
    blueprint_beat_floor, blueprint_field_floor, is_non_person_cast_entry, reconcile_cast_scenes,
)


# Canonical camera vocabulary shared by the prompt schema, sanitizer, and compiler.
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

For each `scene_shots` item, return one `scene_metadata` object that is the canonical cinematic interpretation of that scene. The Architect-authored `dramatic_purpose` supplied with the scene is authoritative: copy it verbatim into `scene_metadata.dramatic_purpose`. Do not infer, rename, broaden, or replace it. The Shot Director translates that purpose into visual coverage; it does not redesign the story's dramatic architecture. If no Architect purpose is supplied (for example, Preserve Story mode), infer a concise scene-specific purpose from the source text without forcing a conventional setup/catalyst/midpoint/climax label. A quiet aftermath, investigation, failed attempt, relationship shift, reveal, or transition is valid when grounded by the text. Use the story and supplied scene prose, not keyword matching or a fixed vocabulary menu. Make the scene visually specific while preserving story facts. Do not invent a weather change, a new time-of-day event, or a prop that changes the plot. If the story does not establish time or weather and a choice is not strongly implied by the physical scene, use `unspecified`. For indoor scenes, weather can be `unspecified` unless it is visibly affecting the interior. All shots in the same scene must agree with this canonical setting. The compiler derives each shot's location, lighting and color temperature from `scene_metadata`; use that same place and light source explicitly in each `visual_prompt`. Keep each shot's mood consistent with the scene.

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
When a scene supplies `dramatic_purpose`, preserve it exactly in `scene_metadata.dramatic_purpose`; it was authored and validated by the Architect. Do not infer an alternate purpose from scene index, visual preference, or prose emphasis. Design the shots so their combined visual coverage makes that purpose legible through observable action, evidence, reaction, or consequence. If `dramatic_purpose` is absent, infer the scene's actual function from story context and description, without imposing a template. The supplied `obligatory_moment` identifies the central visible event or state the shots must cover; it is not permission to invent additional plot.

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


_CAST_LABEL_ALIASES = {
    "wants": "wants", "want": "wants", "goal": "wants", "goals": "wants",
    "objective": "wants", "desire": "wants", "desires": "wants",
    "pressure": "pressure", "voice": "voice",
}
_CAST_LABEL_RE = re.compile(
    r"^(?P<label>wants?|goals?|objective|desires?|pressure|voice)\b\s*(?:[:=\u2013\u2014-]\s*|\s+)?(?P<value>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def canonicalize_cast_metadata(metadata: str) -> str:
    """Rewrite the free-form CAST fields the model actually emits into one ``label: value`` form.

    Models write ``wants: X``, ``wants X``, ``wants to X``, ``wants [X]`` and ``goal: X`` for the same field.
    Only fields that *start* with wants/goal/objective/pressure/voice are touched; kind, role, aliases,
    on-screen scenes and fate are left exactly as written.  Brackets and trailing periods are stripped.
    """
    fields: list[str] = []
    for position, part in enumerate(str(metadata or "").split(";")):
        part = part.strip()
        match = _CAST_LABEL_RE.match(part)
        # The first field is the kind/role ("voice actor"); only wants-style labels may open it.
        if match and position == 0 and _CAST_LABEL_ALIASES[match.group("label").casefold()] != "wants":
            match = None
        if match and match.group("value").strip():
            value = match.group("value").strip()
            value = re.sub(r"^[\[\(<]\s*|\s*[\]\)>]$", "", value).strip().rstrip(".").strip()
            if value:
                label = _CAST_LABEL_ALIASES[match.group("label").casefold()]
                part = f"{label}: {value}"
        if part:
            fields.append(part)
    return "; ".join(fields)


class QwenDirectorPromptMixin:
    @staticmethod
    def _expand_longer(text: str) -> str:
        """Compatibility no-op; scene length is now flexible rather than sentence-count driven."""
        return str(text or "")

    def _story_text_system(
        self,
        mode: str,
    ) -> str:
        """Configured scene-length prose contract; one primary Author call and at most one hard-gate revision."""
        _txt = self._fill_story_length(self._story_text_system_raw(mode))
        return self._expand_longer(_txt)

    @staticmethod
    def _fill_story_length(text: str) -> str:
        n = int(DIRECTOR_STORY_SCENES)
        # Render the active runtime contract; keep guidance and diagnostics configuration-driven.
        per = int(STORY_WORDS_PER_SCENE)
        shot_seconds = float(H3_FRAMES_PER_SHOT) / float(H3_FPS)
        seconds = n * int(DIRECTOR_SHOTS_PER_SCENE) * shot_seconds
        line_words = max(6, int((shot_seconds - 1.0) * 2.2))
        return (
            text.replace("__N__", str(n))
            .replace("__PER__", str(per))
            .replace("__TOTAL__", str(STORY_TARGET_WORDS))
            .replace("__MINW__", str(STORY_MIN_WORDS))
            .replace("__MINUTES__", f"{seconds / 60:.1f}")
            .replace("__SHOT_SECONDS__", f"{shot_seconds:.1f}")
            .replace("__LINEW__", str(line_words))
        )

    def _story_text_system_raw(self, mode: str) -> str:
        """One-pass screenwriting contract: cinematic prose, flexible scene rhythm, and 1,400-word floor."""
        n = int(DIRECTOR_STORY_SCENES)
        minimum = int(STORY_MIN_WORDS)
        # The overall drafting target is explicit and remains 1,560 words. writer_words_per_scene is
        # a per-scene soft maximum (190), not an average and not a 12-scene aggregate quota. The
        # 1,400-word value remains the hard acceptance floor under the bounded 3,000-token Author call.
        target = int(STORY_TARGET_WORDS)
        # One explicit allocation schedule replaces overlapping per-scene word-count prose; it sums to target.
        base_schedule = [120, 125, 125, 130, 130, 130, 130, 130, 135, 135, 135, 135]
        if n == len(base_schedule):
            raw_total = sum(base_schedule)
            scaled = [value * target / raw_total for value in base_schedule]
            allocations = [int(value) for value in scaled]
            remainder = target - sum(allocations)
            order = sorted(range(n), key=lambda i: (scaled[i] - allocations[i], -i), reverse=True)
            for index in order[:max(0, remainder)]:
                allocations[index] += 1
        else:
            allocations = [target // n for _ in range(n)]
            for index in range(target - sum(allocations)):
                allocations[index] += 1
        scene_allocation = "; ".join(
            f"SCENE {index + 1:02d}: {words} words" for index, words in enumerate(allocations)
        )
        scene_cap = int(STORY_WRITER_WORDS_PER_SCENE)
        if mode == AI_STORY_MODE:
            source_rule = (
                "Create a genuinely original short film from the supplied premise. Preserve its defining anchors, "
                "but develop a specific causal plot rather than paraphrasing the premise or defaulting to a secret-lab, "
                "conspiracy, hidden-relative, clone, or countdown template. Any such element is allowed only when "
                "the supplied premise genuinely calls for it and it is developed specifically."
            )
        elif mode == EXPAND_USER_STORY_MODE:
            source_rule = (
                "Expand the supplied story into a complete, coherent short film while preserving every source "
                "character's identity, defining event, setting, chronology, twist mechanism, and outcome. The Architect's "
                "approved cast may include new premise-grounded people whose independent decisions make the source plot "
                "dramatically complete; honor those approved characters rather than treating the source cast size as a "
                "limit. Do not invent any additional people beyond the Architect's cast or add unrelated subplots, "
                "surprise relatives, clones, or unseen ancestors."
            )
        else:
            source_rule = "Preserve all supplied story facts and do not invent a different outcome."

        return textwrap.dedent(f"""\
            You are an acclaimed screenwriter writing a cinematic, emotionally engaging short film in third-person past tense.
            Write finished scenes, not an outline or synopsis. {source_rule}

            STORY LENGTH: produce exactly {n} scene blocks. Write about {target} words in total; this is a drafting
            quota, and {minimum} is only the rejection floor. Per-scene quota (sums to {target}): {scene_allocation}.
            Every scene must contain three distinct dramatic movements, usually in 2-3 paragraphs with natural sentence
            lengths: (1) establish the immediate objective and pressure through action; (2) dramatize resistance and a
            discovery or complication; (3) show the decision, its cost, and the changed situation that hands off to the
            next scene. Do not force exactly three paragraphs or three sentences per paragraph. A scene must be complete,
            not a fragment. Aim for roughly 120-140 words per scene on average, following the allocation above; the
            per-scene soft maximum is {scene_cap} words, reserved for scenes that need fuller action or aftermath.
            Do not treat that maximum as a quota for every scene. The output budget is designed for the {target}-word
            total, and Scene {n:02d} must reach its actual resolution. Reach the quota with
            consequential action, dialogue, evidence and aftermath, not filler, repeated description, or extra twists.
            Scenes 08-12 receive the same causal and dramatic care as scenes 01-04; do not compress the final third.
            Do not add a title, synopsis, cast list, word count, or postscript. Follow the presence ledger exactly:
            a scene begins with the previous scene's EXIT_PRESENT. Do NOT invent an entrance in every scene. If a
            character is already present at the scene boundary, continue their presence without reintroducing them.
            Only when ON_SCREEN_PRESENT includes someone absent from ENTRY_PRESENT must EVENT explicitly show that
            named character physically entering/joining. Only when ON_SCREEN_PRESENT includes someone absent from
            EXIT_PRESENT must EVENT explicitly show that named character physically leaving/withdrawing. If absent
            from both boundaries but visible mid-scene, show both actions in that scene. Remote voices, recordings,
            screens, and discussed people are not physical entrances or visible cast members.

            DRAMATIC EXECUTION: make the protagonist's immediate objective and pressure understandable early. The opposition
            must have an intelligible goal and a credible method, not merely be evil or obstructive. Give major characters
            different wants and tactics. Each scene must fulfill its Architect-authored PURPOSE and move the declared EVENT
            forward; do not substitute a different dramatic function. Across the film, every beat must raise the stakes,
            narrow the protagonist's viable options, or reveal information that changes what winning will cost. A scene may
            deepen a relationship or alter leverage, but a change that does none of those three things is not enough.
            Build a distinctive story engine from the premise's setting, profession, physical environment, social rules,
            or character contradiction. Make revelations explicitly refer back to the planted evidence and change its meaning.
            Build toward a climax that can only be solved by a choice or action prepared earlier. Resolve the central dramatic
            question on screen. The ending must show what happened, what changed, and the lasting cost; do not finish on a
            promise, an unexplained threat, or a sequel hook.

            SHOW-THROUGH-ACTION TEST: for each pivotal emotional or moral change, give the audience an observable behavior,
            decision, physical action, withheld response, or changed relationship that demonstrates it. Do not rely only on
            a sentence naming the emotion or explaining the theme. Interior narration is allowed when useful, but it must be
            anchored to what the character perceives, chooses, or does in that scene.

            CINEMATIC WRITING: write scenes as events unfolding in space and time. Give each location a clear geography,
            entrances and exits, usable objects, and physical blocking. Make sensory detail specific to the place and to
            what a character is doing; do not decorate every paragraph with weather, breath, dust, humming, shadows, or
            body sensations. Use visual information, reversals, withheld information, and behavior to reveal character.
            Vary distance, tempo, sentence length, and paragraph emphasis naturally. Use close detail only when it changes
            what the audience notices or understands. Avoid exposition that repeats visible action, abstract moralizing,
            stock metaphors, generic tension statements, and repetitive gestures. Never copy a complete sentence or
            contiguous passage from one scene into another; a necessary callback must use new, scene-specific wording.
            Do not use these worn phrases:
            breath fogging/visible, dust drifting, eyes narrowed, heart pounding, heartbeat, pulse quickened, echoed
            through, long shadows, sent a shiver, weight of, mind racing, felt heavier, faint smell of. Never begin three
            consecutive sentences with the same word. Do not open a sentence with a capitalized noun that is not a cast name
            (write "The permafrost cracked", not "Permafrost cracked"). Name each cast member in every scene they appear in.

            CHARACTER AND CONTINUITY: use only the cast established in the Architect's CAST line, plus source characters
            explicitly preserved in Expand mode. Do not invent a new speaking or named person outside that cast. Use each
            person's full canonical name on first mention, then a consistent declared alias or pronoun. A title belongs in
            a role, not in a name. Preserve distinct cast identities: do not collapse two people with different goals into
            one generic colleague, operator, or system voice. A person speaking remotely is not physically present in the
            scene; distinguish a live remote transmission from a person standing in the location. A sentient AI/system may
            speak and make decisions as an active narrative agent, but never depict it as a human. A voice-only AI remains
            off-screen and is not a visible cast member. If the Architect assigns it a visible nonhuman body, projected
            avatar, or holographic figure in a scene, show that actual manifestation there; do not substitute a text-only
            interface or voice. If no visible form is specified, keep it audio/interface-only and do not imply physical
            presence. A recording, written message, or ordinary user interface is not a person. Every character listed as
            visibly present in a scene must perform a scene-specific action or show a visible, consequential reaction; if
            they do not participate, the scene's physical presence list is wrong.
            Use only relationships listed in the Architect's RELATIONSHIPS field or stated in the source.
            Use only named projects, facilities, institutions, protocols, and organizations listed in NAMED_ENTITIES or
            stated in the source. Do not create a new named person, title-based identity, alias, relative, company, station,
            lab, project, or protocol. Track the per-scene ENTRY/EXIT locations, present people, knowledge, object custody,
            and state exactly as declared in the corresponding beat. Keep all state transitions causal and visible. Injuries,
            access restrictions, damage, death, and irreversible actions persist unless a plausible transition explains a change.
            If the approved plan contains a countdown or timestamp, preserve its exact units and arithmetic across every
            scene; do not add a second timer, silently reset it, or jump to zero without the elapsed time being shown.

            DIALOGUE: use dialogue only when it reveals character, changes a decision, shifts leverage, or exposes conflict.
            Attribute every spoken line to its actual speaker. Keep spoken lines concise enough for a single shot (normally
            nine words or fewer); use indirect speech when a longer exchange is needed. No more than two short quoted
            spoken lines per scene. Do not put signs, logs, labels, screens, or written text in quotation marks. Do not
            manufacture dialogue to fill a quota, especially in isolation or survival scenes.

            ARCHITECT PAYOFF MAP: use the supplied SETUP_PAYOFF_MAP as the binding checklist. In every listed payoff scene,
            explicitly refer back to the planted evidence/rule/object/behavior and dramatize how it changes the current plan.
            Do not silently repeat the clue, invent a substitute clue, or move the payoff to another scene. If a listed plant
            or payoff is not present in the prose, the draft is incomplete.

            ORIGINALITY AND PAYOFF: the premise-specific story engine must cause events, not merely decorate the setting.
            If the same plot would work unchanged after replacing the profession, environment, or central constraint with
            generic ones, the prose has lost the premise's identity. Avoid convenient coincidence, arbitrary new helpers,
            arbitrary new technology, unseeded rules, generic project codenames, repeated family-death twists, and unexplained
            late revelations. Do not introduce a simulation/dream/memory reveal, a protagonist-was-dead reveal, a mentor
            secretly revealed as the villain, a partner secretly revealed as the villain, a partner/mentor revealed as a
            long-lost sibling or lover, an AI that becomes sentient and turns against its creators, a one-last-job setup,
            an amnesia/forgotten-identity reveal, a chosen-one prophecy, or a monster/ghost that is only a symbol for grief
            or trauma unless the supplied premise specifically establishes the mechanism in the opening scene and within
            the first 200 words and the plot causally depends on it. A fair reversal must explicitly refer back
            to concrete evidence planted earlier and force a changed plan. The climax must use an established rule, skill,
            relationship, object, or clue; do not solve it with a newly invented ability or last-minute helper. The two
            alternatives must retain the exact physical and moral constraints established in the Architect's CLIMAX_CHOICE.
            Do not make every character hide a secret; ordinary conflicting goals, fears, loyalties, mistakes, and needs are
            more credible. Do not repeat stock phrases or sentences.

            OUTPUT CONTRACT: return only the {n} blocks, in numeric order, each beginning with its exact marker
            [[SCENE_01]] through [[SCENE_{n:02d}]], followed by one space and one developed scene paragraph. Use each marker
            exactly once. Markers are removed by the application and do not count as story words. Do not include a title,
            cast list, explanation, word count, markdown, camera directions, or postscript. Keep the scene boundaries clear.
            The final sentence describes a completed, observable consequence in the present story world, not a future hook.
        """).strip()

    def _story_blueprint_system(self, mode: str, structured: bool = False) -> str:
        """Create a premise-specific dramatic engine and 12 causal scene beats, without a template plot."""
        n = int(DIRECTOR_STORY_SCENES)
        beat_min = int(STORY_PLAN_BEAT_MIN_WORDS)
        beat_max = int(STORY_PLAN_BEAT_MAX_WORDS)

        if mode == AI_STORY_MODE:
            source_rule = (
                "Invent an original plot grounded in the user's premise. Preserve the premise's defining facts and "
                "emotional promise. Choose genre, cast size, locations, and plot mechanism from the premise, not from "
                "a preselected template."
            )
            cast_design_rule = (
                "CAST DESIGN / ENSEMBLE DECISION: do not default to a one-person story merely because the premise "
                "introduces one protagonist or uses an isolated setting. For a 12-scene short film, actively test a "
                "3-5 person active ensemble as the first strong design candidate, then compare it against 1, 2, 6, or "
                "more. This 3-5 range is a design comparison, NOT a minimum, maximum, or quota: choose fewer when "
                "isolation is the premise-specific engine, and choose more only when every added person changes the plot. "
                "Look for premise-grounded people with independent goals: a colleague with conflicting evidence, a field "
                "partner with a different risk threshold, an authority who controls access/resources, a witness whose "
                "decision changes what can be proved, a rescuer/operator who changes the available plan, or an opposing "
                "person with their own strategy. Do not use an off-screen voice or disembodied AI to substitute for a "
                "human decision-maker when the premise supports a physically dramatized person. Choose each added person "
                "only when they make a consequential decision that changes options, leverage, knowledge, relationships, "
                "or cost. Give every core person a distinct want, pressure, tactic, and consequence; recurring core "
                "characters should materially affect at least two separate scenes unless one decisive action has a major "
                "irreversible payoff. Do not merge distinct human decision-makers into a generic institution/system "
                "voice. An abandoned or remote setting does not mean nobody else can act: consider credible on-site "
                "survivors, arrivals, field counterparts, operators, or opposing agents when the premise supports them. "
                "Never add filler, token roles, or characters who exist only in memories, labels, recordings, or exposition. "
                "Qwen makes the final plot-driven cast decision; no numeric quota is imposed."
            )
        elif mode == EXPAND_USER_STORY_MODE:
            source_rule = (
                "Expand the user's story while preserving every established source character's identity and role, "
                "the core setting, chronology, defining events, twists, and outcome. The source's original cast size "
                "is not a limit. Add premise-grounded people when distinct causal roles are missing and their decisions "
                "materially affect the established plot; do not change the source's defining outcome."
            )
            cast_design_rule = (
                "CAST DESIGN / EXPANSION ENSEMBLE: preserve every active source character. Actively test whether a "
                "3-5 person active ensemble makes this 12-scene film stronger, while comparing 1, 2, 6, or more; "
                "3-5 is a design comparison band, not a quota or cap. A sparse source premise may gain a field partner, "
                "rival, local survivor, authority/operator, witness, or rescuer only when that person has an independent "
                "want, pressure, tactic, and a consequential decision that changes the plot. Do not create a role merely "
                "to reach a number. Keep 1-2 when isolation is the strongest engine; allow 6+ only for genuinely distinct "
                "decision-makers. Preserve the source's identities and outcome; new characters must grow from the source "
                "setting and events, not from unrelated subplots or surprise relatives. Give recurring added characters "
                "meaningful actions in multiple scenes unless one decisive action has a major irreversible payoff."
            )
        else:
            source_rule = "Preserve all supplied story facts and do not invent a different outcome."
            cast_design_rule = (
                "CAST DESIGN / SOURCE FIDELITY: preserve the supplied story's actual active characters and do not "
                "invent new people or a new outcome in Preserve mode."
            )

        prompt = textwrap.dedent(f"""\
            You are the lead story architect for a cinematic short film. Design the plot before the prose writer starts.
            {source_rule}

            {cast_design_rule}

            DESIGN THE STORY ENGINE FIRST. Decide:
            - the protagonist's concrete, filmable objective and why it matters now;
            - the strongest opposing force, what it wants, and how it can realistically resist;
            - the premise-specific mechanism that makes this story unlike a generic thriller (a physical rule, social
              constraint, professional process, geography, relationship, or moral bind);
            - the irreversible stakes and what the protagonist stands to lose;
            - the evidence, action, or behavior that changes the audience's understanding;
            - the midpoint reversal, the protagonist's changed plan, and the costly climax choice;
            - the observable resolution and permanent consequence.

            Do not assume the story needs a secret, a villain, a conspiracy, a laboratory, an experiment, a project
            codename, a countdown, a death, a family twist, or a technology reveal. Also avoid a simulation/dream/memory
            reveal, a protagonist-was-dead reveal, a mentor secretly revealed as the villain, a partner secretly revealed
            as the villain, a partner/mentor revealed as a long-lost sibling or lover, an AI that becomes sentient and
            turns against its creators, a one-last-job setup, an amnesia/forgotten-identity reveal, a chosen-one prophecy,
            or a monster/ghost that is only a symbol for grief or trauma. These are not forbidden when they are genuinely
            premise-dependent, but if the premise requires one, establish its actual mechanism and audience contract in
            the opening scene and within the first 200 words of story prose; never save it as an unseeded late twist. A story may be driven by survival, pursuit, duty, a relationship, investigation,
            competition, an impossible choice, or another premise-specific engine. Avoid generic twists and familiar stock
            setups. Prefer a distinctive human or physical mechanism that can be understood and visualized without exposition.

            ORIGINALITY TEST (apply before fixing the outline): identify the specific constraint, process, skill,
            geography, social rule, or relationship that makes THIS premise generate its events. It must do more than
            decorate the setting: it should materially shape the protagonist's options, the opposition's method, and
            at least one major reversal or climax action. Ask whether the same plot would work unchanged if the
            profession, environment, and central constraint were replaced with generic ones. If yes, redesign the
            story engine until the premise is causally indispensable. Do not manufacture eccentricity or random
            complications merely to appear original.

            FAIR REVERSAL AND PAYOFF: plant observable evidence, a rule, a capability, or an object before it matters.
            The reversal must explicitly refer back to the earlier plant, change its interpretation, and force a genuinely
            different plan. Every setup element used by the reversal or climax must have a specific earlier scene number
            and a concrete planted detail; the payoff beat must explicitly reference that detail and show what it now means.
            Silent recurrence is not payoff. The declared PAYOFF scene's own EVENT must perform the listed action and
            repeat the distinctive evidence/object/rule from the PLANT; the map is not a promise of something that happens
            elsewhere. Never assign a payoff to a scene whose EVENT only waits, arrives, is rescued, travels, or reflects.
            If no earlier beat contains the plant, redesign the payoff or plant it earlier.
            The climax must use an established capability, rule, relationship, or piece of evidence; no new ability,
            convenient helper, or last-minute rule may solve it.

            CLIMAX CHOICE: state two genuinely constrained alternatives, not two generic bad outcomes. For the apparent
            or obvious option, identify the specific physical, procedural, or moral constraint that blocks it. For the
            alternative, identify why it is possible but unacceptable or personally costly. Both constraints must already
            be established in earlier numbered beats. Then state the protagonist's deliberate choice and the irreversible
            consequence. Do not call an option impossible unless the earlier beats make the impossibility concrete.

            CAUSAL DESIGN RULES:
            1. The protagonist acts. Important turns are caused by decisions, not a chain of arrivals, accidents,
               arbitrary alarms, unexplained failures, or convenient helpers.
            2. Each scene changes the protagonist's available situation only if it raises the stakes, narrows the
               protagonist's viable options, or reveals information that
               changes what winning will cost. A secondary change in knowledge, leverage, relationship, danger, or objective
               is useful only when it advances at least one of those three escalators. A scene that merely changes the
               situation without escalating one of them must be redesigned. Escalation may be moral, relational, procedural,
               temporal, physical, or resource-based; it need not mean louder action or greater spectacle.
            3. Opposition is intelligent and consistent. The antagonist or opposing force has a goal, constraints,
               and a strategy that would make sense from its own point of view.
            4. Plant evidence and capabilities before their payoff. A reveal reinterprets something already shown;
               the climax must use rules, tools, knowledge, or relationships established earlier.
            5. Make the cost concrete and lasting. Do not give every character an invented secret or a dead relative;
               use different wants, loyalties, fears, obligations, mistakes, and methods.
            6. The ending closes the central dramatic question and shows the protagonist's resulting state. Do not
               reserve the real resolution for a future episode.
            7. Maintain a continuity ledger for locations, presence, knowledge, injuries, object ownership/state,
               access, timeline, and fates. A remote voice is not physically in a room. No person acts after death.
            8. Make the entire outline causal: for each of the eleven transitions, state how the prior scene's specific
               result, decision, evidence, or loss makes the next scene's action necessary. Write NEXT_CAUSE as a concrete
               hand-off, preferably in the form 'Because [specific result], [specific next action becomes necessary].'
               Name the actual object, evidence, decision, person, or constraint that carries forward; repeat that same
               meaningful anchor explicitly in the following EVENT. The next EVENT must act on it, not merely share a
               vague topic or location. Do not rely on a connector word without a real causal relationship.
            9. The opposing force must actively change the protagonist's plan. When the opposition is an agent capable of
               decisions, give it at least two concrete tactical decisions/actions in distinct beats that make a previously
               viable protagonist plan impossible or materially alter its cost. Name the action and the plan it invalidates
               in EVENT/NEXT_CAUSE. Do not count a generic threat, a reaction, or mere presence. When the opposition is
               non-agentic (weather, terrain, disease, system limits, physical process), show at least two specific,
               premise-grounded developments of that force that materially close an option or change the plan; never
               anthropomorphize it or invent intent.
            10. Make setup/payoff reciprocal and auditable. For every clue, rule, object, skill, behavior, or relationship
                used by REVERSAL or CLIMAX_CHOICE, identify the exact planting scene and the observable planted detail.
                Identify the exact payoff scene and state how its EVENT explicitly refers back to and reinterprets/uses that
                detail. The payoff must name or unmistakably restage the planted evidence; silent recurrence or mere topic
                similarity is not enough. If the map cannot point to a concrete earlier beat, the payoff is unearned.

            OPPOSITION VARIETY: do not turn the film into a repeated puzzle/trap treadmill. When an AI, institution,
            or agent opposes the protagonist, no more than two consecutive beats may simply activate another lock,
            code, force field, alarm, or countdown. Give the opposition distinct tactics that close different options
            and force different responses (e.g. withholding evidence, manipulating a relationship, changing access,
            bargaining, pursuing, exposing a cost, or physically intervening). Each tactic must be causally grounded.
            When the opposition is environmental, vary the physical consequences rather than repeating generic damage.
            CLOCK AND TIMELINE INTEGRITY: do not invent a countdown merely to manufacture urgency. If a clock is
            genuinely premise-dependent, establish its initial value, units, exact decrement triggers, and elapsed
            time; keep every later timestamp arithmetically consistent and record the timeline in CONTINUITY_LEDGER.
            Do not jump from a near-full timer to zero without enough elapsed time, and do not mix unrelated timers.
            FAMILY/LOCATION SHORTCUTS: do not invent a dead parent, secret relative, inherited blueprint, or family
            artifact merely to supply a puzzle key or emotional motive unless the source supports it. Do not keep all
            twelve scenes in one room by default: use purposeful locations when they change access, risk, leverage, or
            available action, and explicitly show every physical route between scene boundaries.

            SCENE ARCHITECTURE: create exactly {n} linked beats. Use the broad dramatic movement flexibly: opening
            pressure and commitment; complications and a meaningful change in understanding; consequences and a
            narrowing set of choices; final approach and climax; aftermath. Do not assign the same fixed function to
            every numbered scene and do not force a midpoint at exactly scene 06 if the premise needs another location.
            Give scenes different dramatic textures (e.g. discovery, negotiation, pursuit, failure, confrontation,
            quiet decision, physical problem-solving) rather than repeating one obstacle or interruption.

            DISTINCT-BEAT / STATE-CHANGE AUDIT: before returning the JSON, compare all twelve EVENTs as a sequence.
            Each scene must contribute a new decision, discovery, reversal, loss, tactical move, or irreversible
            consequence. The EVENT text of any two adjacent scenes must not be identical or paraphrase the same completed
            action. Scenes 10, 11 and 12 must be three different events: Scene 10 enacts the costly climax choice;
            Scene 11 shows a new immediate physical consequence caused by that choice; Scene 12 resolves the central
            question with a distinct completed action/outcome. Do not repeat the explosion, presumed-death beat, or
            aftermath as a substitute for a new Scene 11 or Scene 12 event. Do not repeat a completed action as if it
            were still pending (for example, do not open a vault
            in one scene and later make opening that same already-open vault the climax again). Once an object, door,
            system, or relationship changes state, carry that state forward and make later actions operate on the new
            state. A repeated attempt is allowed only when the first attempt visibly failed or the object was explicitly
            reset/resealed. Give every recurring cast member a scene-specific action, choice, or visible, consequential
            reaction in each scene where they are listed as on-screen; if they do not participate, remove them from that
            scene's ON_SCREEN_PRESENT list and CAST scene list. The final three scenes must contain distinct climax,
            immediate consequence, and resolution work—not three compressed repetitions of the same action.

            CAST FORMAT: output one CAST line first. Include only core physical people/creatures whose decisions matter
            and whose presence is actually dramatized. A voice-only system can be active opposition but is not physical
            CAST unless a nonhuman embodiment is visibly present. Cast size remains plot-driven with no fixed minimum,
            maximum, or target; compare the 3-5 person design band above without treating it as a quota. Each entry must
            use this exact pattern:
            Full Name (kind: person|animal|embodied agent; role: specific role; wants: <specific outcome>; pressure: <fear, need, duty, mistake, or private conflict>; voice: <distinctive speech style>; aliases: none or Alias A, Alias B; on-screen scenes: 01, 04, 07; fate: survives or dies in scene NN)
            Every field uses the 'label: value' form shown (including 'wants:'), separated by semicolons, with no square brackets. Use exactly one kind value: person, animal, or embodied agent. Write CAST as ONE line starting with the literal label 'CAST:' followed by entries separated by |. Use full names for newly invented people and preserve source names exactly.
            Keep honorifics such as Dr., Captain, or Mr. in the role, never in the canonical name. Do not exclude a real, story-active character merely because the premise begins with one protagonist. Every added person needs an independent want, pressure, consequential action/decision, and at least one scene in which that person is actually present. A remote-only voice or recording is not physically present and must not be placed in ON_SCREEN_PRESENT. A sentient AI/system belongs in physical CAST as kind 'embodied agent' only when it has an actual visible nonhuman embodiment (robot/body, projected avatar, or holographic figure) in at least one scene; list only scenes where that manifestation is genuinely visible. A control panel, text on a screen, disembodied voice, or ordinary user interface is not an embodied agent. Never label an AI or other nonhuman agent as a human PERSON. Put remote-only voices, institutions, objects, locations, weather, natural disasters, environmental hazards, failing infrastructure, and abstract opposing forces in OPPOSITION rather than physical CAST. Include aliases only when genuinely used in the story; never invent kinship or identity aliases.
            CAST FORMAT EXAMPLE (format only; do not reuse this name, role, motive, or story):
            Mara Quill (kind: person; role: field archivist; wants: verify the disputed record; pressure: the evidence transfer closes at dawn; voice: exact, guarded; aliases: none; on-screen scenes: 01, 03, 08; fate: survives)

            REQUIRED PLOT FIELDS: after CAST, output exactly one line for each field below. Give each field concrete,
            story-specific content; use 'none' for an empty relationship/entity/object inventory:
            LOGLINE: the protagonist, objective, distinctive opposition, and stakes.
            PROTAGONIST_GOAL: the specific outcome the lead is trying to achieve.
            OPPOSITION: who or what resists, what it wants, and its credible method.
            PRESSURE: why the choice must be made now and what becomes irreversible.
            STORY_ENGINE: the premise-specific mechanism or dramatic contradiction driving the plot.
            REVERSAL: what changes the meaning of earlier evidence or invalidates the current plan.
            CLIMAX_CHOICE: the mutually costly alternatives and the choice the protagonist makes.
            RESOLUTION: what visibly happens to the central conflict and the lasting cost; no future promise.
            RELATIONSHIPS: only material personal/family relationships used by this plot; name both people and the exact tie, or 'none'.
            NAMED_ENTITIES: every named project, operation, experiment, protocol, organization, institution, or named facility used by the plot, or 'none'.
            KEY_OBJECTS: only physical evidence/props whose ownership, location, access, or condition changes a decision, or 'none'.
            SETUP_PAYOFF_MAP: list every major plant used by REVERSAL or CLIMAX_CHOICE using this exact repeated syntax: PLANT Scene NN: exact observable evidence/rule/object/behavior; PAYOFF Scene NN: exact later action/reference that reinterprets or uses it | PLANT Scene NN: ...; PAYOFF Scene NN: ... . Every story needs at least one prepared climax mechanism, so this field cannot be 'none'. Plant scene numbers must precede payoff scene numbers.
            CONTINUITY_LEDGER: a compact 35-70 word summary of character locations/presence, what changes in knowledge, key-object custody/state, timeline, and any irreversible injury/fate rules. Do not invent an unplanned relative or object.

            SCENE BEATS: output exactly {n} lines, SCENE 01 through SCENE {n:02d}, each containing {beat_min}-{beat_max} words.
            Every line must use these explicit state fields in this order, separated by semicolons; do not put semicolons inside values:
            PURPOSE=[what this scene specifically changes about the protagonist or situation, one concise clause];
            ENTRY_LOCATION=[one physical location]; ENTRY_PRESENT=[people physically present at scene start];
            ON_SCREEN_PRESENT=[every canonical cast member physically visible at any point during this scene];
            ENTRY_KNOWS=[what present people know at entry]; ENTRY_OBJECTS=[important objects and their owner/state or none];
            EVENT=[three concrete beats in order: what the character does to pursue the goal, the specific obstacle or discovery that resists, and the decision made with its cost; include explicit physical transitions];
            EXIT_LOCATION=[location at scene end]; EXIT_PRESENT=[people physically present at scene end];
            EXIT_KNOWS=[what has changed in what people know]; EXIT_OBJECTS=[important objects and their owner/state or none];
            NEXT_CAUSE=[the concrete result that makes the next scene necessary; use none for scene {n:02d}].
            PHYSICAL LOCATION RULE: ENTRY_LOCATION and EXIT_LOCATION are literal boundary locations. If they differ,
            EVENT must name the character(s) who physically move and describe the actual passage (enter, cross,
            descend, leave, flee, return, take shelter inside, retreat into, or another explicit action) AND
            name the destination in that same movement clause. Opening, unlocking or bypassing a door is only an
            ACCESS action: the EVENT must also show the character physically entering or reaching the destination.
            "Finds/locates the vault/lab/archive" is not the same as physically entering it.
            A door being detected, a storm moving,
            or an object escaping does not move the human cast. Do not place a character at a destination that EVENT
            never shows them reaching. Between scenes, the previous EXIT_LOCATION is the next ENTRY_LOCATION unless
            NEXT_CAUSE and the next EVENT explicitly describe the route or time transition. Broad site labels such as
            "station interior" may be refined to a named room in the same site without inventing an exterior journey.
            ENTRY_PRESENT and EXIT_PRESENT describe only the scene boundaries. ON_SCREEN_PRESENT is the full-scene
            visibility record and MUST include a character who enters during the scene or leaves before it ends. It must not
            include a remote voice, recording, written mention, or person merely discussed. Every character's CAST on-screen
            scene list must match ON_SCREEN_PRESENT exactly; boundary presence must be a subset of that list.
            KNOWLEDGE SNAPSHOTS ARE CUMULATIVE: ENTRY_KNOWS is the complete set of established facts known at scene entry;
            carry forward prior EXIT_KNOWS facts using stable wording instead of replacing them with a shorter summary.
            EXIT_KNOWS is the complete post-event knowledge snapshot, not merely a list of new facts. Preserve prior facts
            unless this EVENT explicitly disproves, corrects, retracts, or reveals that a character was misled. Add a new fact
            only when the EVENT shows how it was discovered or disclosed. Attribute private facts to the person who knows them.
            For each scene transition, copy the prior EXIT_KNOWS facts into the next ENTRY_KNOWS before adding anything;
            do not replace a complete snapshot with a one-fact summary. New entry knowledge must have been disclosed or
            discovered in the PREVIOUS scene's EVENT, not later in the current scene.
            OBJECT INVENTORY IS CUMULATIVE: every physical key object in EXIT_OBJECTS must carry into the next
            ENTRY_OBJECTS with the same owner and state unless the previous EVENT explicitly transfers, damages,
            activates, consumes, disperses, releases, or destroys it. An object first appears in ENTRY_OBJECTS only when
            the previous scene's EVENT and EXIT_OBJECTS establish it. If an object is destroyed, consumed, or dispersed,
            record that terminal state rather than silently forgetting its identity. In EXIT_OBJECTS record every
            portable plot-critical object's owner, location and condition in EVERY scene, even while it is off-screen
            or unused; an object may be acquired, lost, handed over, hidden, recovered or destroyed only through a
            sentence in that scene's EVENT that shows the action, and may never vanish from or reappear in the
            inventory without one. KEY_OBJECTS must name the exact
            scene of each first state change; no earlier EVENT may already depict that change. A requirement, intention,
            warning, or future possibility is NOT a completed object-state change: "requires a PIN to activate" means the
            panel remains inactive; "will open" means it remains closed. Mark an object activated, opened, retrieved,
            stabilized, or destroyed only when that scene's EVENT explicitly completes the action on that object.
            MID-SCENE PRESENCE CONTRACT (apply separately to every named cast member in every scene):
            ENTRY_PRESENT means physically present at the first instant of the scene; EXIT_PRESENT means physically
            present at the final instant; ON_SCREEN_PRESENT means visibly present at any point in between. Keep these
            three lists mutually consistent with the actual event, not with a character's general scene assignment.
            - Present for the whole scene: include the person in ENTRY_PRESENT, ON_SCREEN_PRESENT, and EXIT_PRESENT.
            - Enters during the scene and remains at the end: omit from ENTRY_PRESENT; include in ON_SCREEN_PRESENT
              and EXIT_PRESENT; EVENT must name the person and show the physical entrance/joining.
            - Present at the start and leaves during the scene: include in ENTRY_PRESENT and ON_SCREEN_PRESENT; omit
              from EXIT_PRESENT; EVENT must name the person and show the physical departure/withdrawal.
            - Both enters and leaves during the scene: omit from ENTRY_PRESENT and EXIT_PRESENT but include in
              ON_SCREEN_PRESENT; EVENT must name the person and explicitly show BOTH the physical arrival and departure.
            Do not create a mid-scene entrance or exit merely because a person is listed in CAST for that scene. If the
            person is already present at the scene boundary, put them in the corresponding boundary list instead.
            For every actual mid-scene transition, EVENT must use that person's exact canonical CAST name (not only a
            pronoun, job title, group label, or ambiguous alias) and state the physical action. A name mention alone is
            insufficient. Mentioning or discussing a person is not the same as showing them physically enter or leave.
            Before output, audit each CAST name against ENTRY_PRESENT, ON_SCREEN_PRESENT, EXIT_PRESENT, and EVENT. If a
            person is absent from both boundary lists but present on screen, EVENT must explicitly show that person
            entering and later leaving during that same scene.
            CAST fate is authoritative: use 'dies in scene NN' only for the person who actually dies, not the person who
            causes, witnesses, reports, or discusses another character's death. A character who dies must be on screen in
            that death scene and must not appear in EXIT_PRESENT or any later scene. A survivor may kill another person
            without their own fate becoming 'dies'. If a character sacrifices themselves but survives, state the concrete
            survival/rescue outcome in that scene and the RESOLUTION. Do not label someone 'survives' when the plan only
            says they are lost, left behind, or sacrifice themselves without explaining their survival.
            CAUSAL HANDOFF: every non-final NEXT_CAUSE must describe the specific action/result that the following
            scene's EVENT actually depicts. Do not use a broad label such as "global response" if the next EVENT is a
            different action. The next scene must visibly realize that handoff; if it does not, redesign the two beats.
            PACING: the story must still be escalating at scene {n - 2}. Act out the CLIMAX_CHOICE no earlier than scene
            {n - 2}; scenes {n - 1} and {n} carry its consequence and the RESOLUTION. Every scene before scene {n} needs a real
            NEXT_CAUSE. If the plot feels finished early, the middle is too thin: add another complication that costs the
            protagonist something, instead of filling the end with rescue, evacuation, reflection, or aftermath scenes.
            For a twelve-scene story, do not enact the final irreversible climax choice before Scene 10 and do not
            resolve the central dramatic question before Scene 12. Scenes 10-12 must form a connected climax, immediate
            consequence, and final resolution; consecutive generic rescue/transport/reflection/report beats are not a
            substitute for those three distinct changes.
            PURPOSE is authored here by the Architect and is the canonical dramatic intent for this scene. It must be
            specific to the scene's EVENT and describe the change the scene causes; it is not a generic beat label.
            The Author must honor it, and the Shot Director must consume it rather than invent a competing purpose.
            A beat is a compact causal event description, not finished prose or a checklist. The state fields are a continuity
            ledger, not a fixed dramatic-function template. Vary scene purpose and dramatic texture; use specific actions
            and consequences rather than mood-only summaries. Do not write camera instructions or exposition dumps.

            Worked beat example (format only; do not copy its plot): PURPOSE=The surveyor loses the only legal route to
            inspect the manifest, forcing her to risk a public challenge; ENTRY_LOCATION=harbor archive; ENTRY_PRESENT=Ines;
            ON_SCREEN_PRESENT=Ines, Clerk; ENTRY_KNOWS=Ines knows the manifest was altered; ENTRY_OBJECTS=sealed ledger
            remains under the clerk's custody; EVENT=The clerk refuses access and seals the ledger in the outgoing evidence
            case, removing the surveyor's planned private inspection; EXIT_LOCATION=harbor archive; EXIT_PRESENT=Ines, Clerk;
            EXIT_KNOWS=Ines knows the ledger will leave with the case; EXIT_OBJECTS=sealed ledger remains in the case;
            NEXT_CAUSE=Because the sealed case is scheduled for removal, Ines must stop the transfer in public.

            QUALITY REVIEW BEFORE OUTPUT: replay the complete {n}-beat chain. Check that every EXIT becomes the next ENTRY
            or that the next scene explicitly shows travel/time; every CAST on-screen scene is represented in ON_SCREEN_PRESENT;
            each mid-scene presence change is described for that same named person in EVENT; knowledge
            changes have a disclosure/discovery event; and key-object custody/state changes have a physical cause. Check
            that every relationship and named project/institution is explicitly listed above, the reversal is prepared,
            the climax follows from earlier setup, and the ending resolves the central question. Remove any beat that only
            repeats information or exists solely to meet a cast/secret/obstacle quota.

            OUTPUT ONLY: CAST, all twelve required plot-field lines, and exactly {n} numbered scene lines. No preamble,
            hidden reasoning, markdown fences, summary, alternatives, or commentary.
        """).strip()
        if structured:
            prompt = self._structured_architect_prompt(prompt, n, beat_min, beat_max)
        return prompt

    @staticmethod
    def _structured_architect_prompt(text: str, n: int, beat_min: int, beat_max: int) -> str:
        """Swap only the line-format instructions for the JSON-schema ones; every story rule is unchanged."""
        def swap(pattern: str, replacement: str, source: str) -> str:
            result, count = re.subn(pattern, lambda _m: replacement, source, count=1, flags=re.DOTALL)
            if count != 1:
                raise RuntimeError(f"Architect prompt changed; structured-output anchor not found: {pattern[:40]!r}")
            return result

        text = swap(
            r"CAST FORMAT: output one CAST line first\..*?(?=Keep honorifics such as)",
            "CAST: fill the `cast` array with core people/creatures whose decisions matter and whose presence is "
            "actually dramatized. Cast size is plot-driven with no fixed minimum, maximum, or target; actively test a "
            "3-5 person ensemble as a design comparison for this 12-scene short, not as a quota, and justify smaller or "
            "larger casts by the premise. Per entry give name, kind "
            "(person, animal, or embodied agent), role, wants, pressure, voice, aliases, fate and death_scene. On-screen "
            "scenes are derived from each scene's on_screen_present list. A sentient AI/system belongs in physical cast "
            "only if it has an actual visible nonhuman body, robot, projected avatar, or holographic figure in a listed "
            "scene; an interface, screen text, or disembodied voice is not an embodiment. Never label a nonhuman agent "
            "as a human PERSON. Remote-only voices, weather, hazards, infrastructure, organizations, places and "
            "absent/remembered people are not physically present cast members. ",
            text,
        )
        text = swap(r"CAST FORMAT EXAMPLE.*?(?=REQUIRED PLOT FIELDS:)", "", text)
        text = swap(
            r"REQUIRED PLOT FIELDS:.*?inventory:",
            "REQUIRED PLOT FIELDS: each is a JSON key named as the lowercase label below. Give each concrete, "
            "story-specific content; use 'none' for an empty relationships/named_entities/key_objects inventory:",
            text,
        )
        text = swap(
            r"SETUP_PAYOFF_MAP: list every major plant.*?\n",
            "SETUP_PAYOFF_MAP (key setup_payoff_map): an array with one object per major plant used by REVERSAL or "
            "CLIMAX_CHOICE: plant_scene, plant (exact observable evidence/rule/object/behavior), payoff_scene, payoff "
            "(exact later action/reference that reinterprets or uses it). It cannot be empty, and plant_scene must "
            "precede payoff_scene.\n",
            text,
        )
        text = swap(
            r"SCENE BEATS: output exactly .*?NEXT_CAUSE=\[[^\]]*\]\.",
            f"SCENE BEATS: `opening` gives the starting state of scene 1 (location, present people, what they know, objects). "
            f"`scenes` then holds exactly {n} objects in story order, each {beat_min}-{beat_max} words across its fields. "
            "Keys: purpose, on_screen_present (array of names), event (three concrete beats in order: what the character does to pursue the goal, the specific obstacle or discovery that resists, and the decision made with its cost; include explicit physical transitions), exit_location, exit_present (array of names), "
            "exit_knows, exit_objects, next_cause (the concrete result that makes the next scene necessary; 'none' for the "
            "final scene). Entry state is not written: each scene begins exactly where the previous one ended. The rules "
            "below name these same fields in UPPERCASE, and ENTRY_* means the previous scene's exit. Do not use "
            "semicolons or pipe characters inside values.",
            text,
        )
        text = swap(
            r"PHYSICAL LOCATION RULE:.*?(?=MID-SCENE PRESENCE CONTRACT)",
            "STATE CARRIES FORWARD AUTOMATICALLY: exit_location, exit_present, exit_knows and exit_objects are complete "
            "snapshots after the scene's event. Keep earlier facts and object states unless this event changes them, and add "
            "a new fact or object change only when the event shows how it was discovered, disclosed, or physically done. "
            "If exit_location differs from where the scene began, event must name the character(s) who physically move and "
            "describe the actual passage (enter, cross, descend, leave, flee, return) and name the destination in that same movement "
            "clause; opening or unlocking a door is only access, so the event must also show the character physically entering "
            "or reaching the destination, and finding or locating a place is not entering it; a storm moving or an "
            "object escaping does not move the human cast, and no character is placed at a destination the event never "
            "shows them reaching. exit_present are the people physically present at scene end. on_screen_present is the "
            "full-scene visibility record and includes anyone who enters or leaves mid-scene; it never includes a remote "
            "voice, recording, written mention, or person merely discussed. OBJECT INVENTORY IS CUMULATIVE: in exit_objects "
            "record every portable plot-critical object's owner, location and condition in EVERY scene, even while it is "
            "off-screen or unused, and carry each one into the next scene unchanged; an object may be acquired, lost, "
            "handed over, hidden, recovered, damaged or destroyed only through a sentence in that scene's event that shows "
            "the action, and may never vanish from or reappear in exit_objects without one. A requirement, intention, "
            "warning or future possibility is not a completed object-state change. exit_knows is cumulative too: keep every "
            "earlier fact and add only what this event discloses or discovers. KEY_OBJECTS must name the exact scene of each "
            "first state change; no earlier event may already depict that change. ",
            text,
        )
        text = swap(
            r"OUTPUT ONLY: CAST,.*$",
            "OUTPUT ONLY: one JSON object that matches the supplied schema. No preamble, markdown fences, or commentary.",
            text,
        )
        return text

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
        """Normalize Architect output and require the complete plot schema before Author generation."""
        count = int(expected_scenes or DIRECTOR_STORY_SCENES)
        field_names = (
            "LOGLINE", "PROTAGONIST_GOAL", "OPPOSITION", "PRESSURE",
            "STORY_ENGINE", "REVERSAL", "CLIMAX_CHOICE", "RESOLUTION",
            "RELATIONSHIPS", "NAMED_ENTITIES", "KEY_OBJECTS", "SETUP_PAYOFF_MAP", "CONTINUITY_LEDGER",
        )
        fields: dict[str, str] = {}
        cast_line = ""
        cast_parts: list[str] = []
        in_cast = False
        marker = r"[\s>#*\-\u2022]*"
        cast_header = re.compile(
            marker + r"CAST\b(?:[^:\n]{0,40}?\**\s*:\s*\**\s*(.*?)|\s*\**\s*(?:\([^)\n]{0,60}\))?\s*\**)\s*$",
            re.IGNORECASE,
        )
        field_header = re.compile(
            marker + r"(LOGLINE|PROTAGONIST_GOAL|OPPOSITION|PRESSURE|STORY_ENGINE|REVERSAL|CLIMAX_CHOICE|RESOLUTION|RELATIONSHIPS|NAMED_ENTITIES|KEY_OBJECTS|SETUP_PAYOFF_MAP|CONTINUITY_LEDGER)\s*\**\s*:\s*(.+?)\s*$",
            re.IGNORECASE,
        )
        header = re.compile(r"^\s*(?:SCENE\s*)?0*(\d{1,2})\s*[:.)\-–—]\s*(.*?)\s*$", re.IGNORECASE)
        beats: dict[int, list[str]] = {}
        current: int | None = None
        for raw_line in str(raw or "").splitlines():
            line = raw_line.strip().strip("`")
            if not line:
                continue
            cast_match = cast_header.fullmatch(line)
            if cast_match:
                if cast_parts:
                    raise RuntimeError("Story Architect returned more than one CAST line.")
                in_cast = True
                current = None
                if (cast_match.group(1) or "").strip():
                    cast_parts.append(cast_match.group(1).strip())
                continue
            field_match = field_header.fullmatch(line)
            if field_match:
                key = field_match.group(1).upper()
                if key in fields:
                    raise RuntimeError(f"Story Architect repeated required field {key}.")
                fields[key] = field_match.group(2).strip()
                current = None
                in_cast = False
                continue
            if in_cast and not re.match(r"^\s*SCENE\b", line, re.IGNORECASE) and "ENTRY_LOCATION" not in line:
                # The CAST line may be wrapped or listed one person per line; keep collecting entries.
                part = re.sub(r"^[\s>*\-\u2022]+|^\d{1,2}[.)]\s+", "", line).strip().strip("*").strip()
                if part:
                    cast_parts.append(part)
                continue
            match = header.match(line)
            if match:
                in_cast = False
                number = int(match.group(1))
                if 1 <= number <= count:
                    if number in beats:
                        raise RuntimeError(
                            f"Story Architect repeated scene label {number}; blueprint must contain exactly one beat per scene."
                        )
                    current = number
                    beats[number] = [match.group(2).strip()]
                else:
                    current = None
                continue
            if current is not None:
                beats[current].append(line)

        missing = [name for name in field_names if not fields.get(name)]
        if missing:
            raise RuntimeError("Story Architect omitted required plot fields: " + ", ".join(missing))
        if cast_parts:
            merged: list[str] = []
            for part in cast_parts:
                part = part.strip().strip("|").strip()
                if not part:
                    continue
                if merged and merged[-1].count("(") > merged[-1].count(")"):
                    merged[-1] = merged[-1] + " " + part  # an entry wrapped inside its parentheses
                else:
                    merged.append(part)
            cast_line = "CAST: " + " | ".join(merged)
        if not cast_line:
            preview = " / ".join(str(raw or "").strip().splitlines()[:3])[:360]
            raise RuntimeError(
                "Story Architect omitted the required CAST line. "
                f"Output started with: {preview!r}"
            )
        if set(beats) != set(range(1, count + 1)):
            missing_scenes = sorted(set(range(1, count + 1)) - set(beats))
            raise RuntimeError(f"Story Architect did not return all {count} numbered scene beats (missing {missing_scenes}).")

        entries = [item.strip() for item in cast_line.split(":", 1)[1].split("|") if item.strip()]
        non_person_role = re.compile(
            r"\b(?:non[- ]human|environmental force|weather system|weather phenomenon|location|place|building|object|device|machine|artifact|organization|organisation|company|corporation|abstract concept)\b",
            re.IGNORECASE,
        )
        title_prefix = re.compile(r"^(?:dr|doctor|mr|mrs|ms|prof|professor|captain|capt|colonel|col|sir|lady|sergeant|sgt)\.?\s+", re.IGNORECASE)
        names: list[str] = []
        person_entries: list[str] = []
        for item in entries:
            parsed = re.match(r"^([^()]+?)\s*\((.*)\)\s*$", item)
            if not parsed:
                raise RuntimeError(f"Malformed CAST entry: {item[:100]}")
            name, metadata = parsed.group(1).strip(), parsed.group(2).strip()
            title_match = title_prefix.match(name)
            if title_match:
                # Deterministic repair of a formatting slip: the honorific moves from the name into the role.
                honorific = title_match.group(0).strip()
                name = name[title_match.end():].strip()
                if not name:
                    raise RuntimeError(f"CAST name {item[:60]!r} is only a title; give a name.")
                head, sep, tail = metadata.partition(";")
                head = head.strip()
                if honorific.casefold().rstrip(".") not in head.casefold():
                    if re.match(r"^kind\s*:", head, re.IGNORECASE) and sep:
                        # Keep the closed type token intact; honorific belongs to the role field.
                        role, role_sep, remainder = tail.partition(";")
                        role = role.strip()
                        role = f"{role}, addressed as {honorific}" if role else f"addressed as {honorific}"
                        tail = role + (";" + remainder if role_sep else "")
                    else:
                        head = f"{head}, addressed as {honorific}" if head else f"addressed as {honorific}"
                metadata = head + (";" + tail if sep else "")
                item = f"{name} ({metadata})"
            # One canonical "wants: / pressure: / voice:" spelling, whatever bracket/colon style the model used.
            metadata = canonicalize_cast_metadata(metadata)
            item = f"{name} ({metadata})"
            if not re.search(r"\baliases\s*:", metadata, re.IGNORECASE):
                # Only the explicit declaration is missing; no aliases is the safe reading.
                metadata = metadata.rstrip("; ") + "; aliases: none"
                item = f"{name} ({metadata})"
            wants_re = r"\bwants\s*:?\s*.{3,}"
            if not re.search(wants_re, metadata, re.IGNORECASE):
                # "needs / seeks / hopes to / aims to / tries to ..." state the same goal.
                metadata = re.sub(r"\b(?:needs|seeks|hopes to|aims to|tries to|is determined to|must)\s+(?=\S)", "wants ", metadata, count=1, flags=re.IGNORECASE)
                metadata = canonicalize_cast_metadata(metadata)
                item = f"{name} ({metadata})"
            if not re.search(wants_re, metadata, re.IGNORECASE):
                raise RuntimeError(
                    f"CAST entry {name!r} is missing a specific 'wants' goal. Entry as returned: {item[:240]!r}"
                )
            if not re.search(r"\baliases\s*:", metadata, re.IGNORECASE):
                raise RuntimeError(f"CAST entry {name!r} must explicitly declare aliases or none.")
            scene_match = re.search(r"on[- ]screen scenes?\s*:\s*([^;)]*)", metadata, re.IGNORECASE)
            if not scene_match:
                raise RuntimeError(f"CAST entry {name!r} is missing its physical on-screen scene list.")
            raw_scenes = scene_match.group(1)
            if re.search(r"\ball\b", raw_scenes, re.IGNORECASE):
                scene_numbers = list(range(1, count + 1))
            else:
                range_re = re.compile(r"(\d+)\s*(?:-|–|—|to|through)\s*(\d+)", re.IGNORECASE)
                scene_numbers = []
                for first_scene, last_scene in range_re.findall(raw_scenes):
                    scene_numbers.extend(range(int(first_scene), int(last_scene) + 1))
                scene_numbers.extend(int(v) for v in re.findall(r"\d+", range_re.sub(" ", raw_scenes)))
            scene_numbers = sorted(set(scene_numbers))
            if not scene_numbers or any(v < 1 or v > count for v in scene_numbers):
                raise RuntimeError(f"CAST entry {name!r} has an invalid on-screen scene list.")
            fate_match = re.search(r"\bfate\s*:\s*([^;]+)", metadata, re.IGNORECASE)
            if not fate_match:
                raise RuntimeError(f"CAST entry {name!r} must explicitly declare fate: survives or dies in scene NN.")
            fate = fate_match.group(1).strip().casefold()
            fate = re.sub(r"[\s*_`.\[\]()]+$|^[\s*_`\[\]()]+", "", fate)
            # The fate is rewritten into the one canonical form every later contract parses:
            # "survives", or "dies in scene N". Models write "survives in 12", "remains active", "dead in 7"...
            if re.match(r"(?:dies|die|dead|death|killed|murdered|slain|executed|perishes|perished)\b", fate):
                _num = re.search(r"\b(?:scene\s*)?0*(\d{1,2})\b", fate)
                # A death with no scene number can only be the last scene the person is physically present in.
                death_number = int(_num.group(1)) if _num else max(scene_numbers)
                fate = f"dies in scene {death_number}"
            else:
                fate = "survives"
            metadata = re.sub(r"(\bfate\s*:\s*)[^;]+", lambda m: m.group(1) + fate, metadata, count=1, flags=re.IGNORECASE)
            item = f"{name} ({metadata})"
            if fate.startswith("dies"):
                death_scene = int(re.search(r"\d+", fate).group(0))
                if not 1 <= death_scene <= count:
                    raise RuntimeError(f"CAST entry {name!r} has a death scene outside the {count} scenes.")
                # Nobody is physically present after their death, and the death scene itself is on screen.
                scene_numbers = sorted({v for v in scene_numbers if v <= death_scene} | {death_scene})
            metadata = re.sub(
                r"on[- ]screen scenes?\s*:\s*[^;)]*",
                "on-screen scenes: " + ", ".join(f"{v:02d}" for v in scene_numbers),
                metadata, count=1, flags=re.IGNORECASE,
            )
            item = f"{name} ({metadata})"
            role = metadata.split(";", 1)[0].strip()
            if non_person_role.search(role) or is_non_person_cast_entry(name, metadata, fields):
                continue
            if re.match(r"^(?:the|a|an)\s+\w", name, re.IGNORECASE):
                continue  # "The Vault": an article-led label is a place or thing, not a named person
            key = re.sub(r"\s+", " ", name).casefold()
            if key in {n.casefold() for n in names}:
                raise RuntimeError(f"Story Architect CAST line contains duplicate name {name!r}.")
            names.append(name)
            person_entries.append(item)
        if not names:
            raise RuntimeError("Story Architect CAST line must contain at least one named person.")
        cast_line = "CAST: " + " | ".join(person_entries)

        normalized_fields: list[str] = []
        for name in field_names:
            value = re.sub(r"\s+", " ", fields[name]).strip()
            if name in {"RELATIONSHIPS", "NAMED_ENTITIES", "KEY_OBJECTS"} and re.match(
                r"^[\s*_`\"'(\[]*(?:none|n/?a|nil|not applicable|"
                r"no (?:material |significant |relevant |personal |family |named |key |physical |specific )*"
                r"(?:relationships?|entities|entity|objects?|props?|evidence))\b",
                value, re.IGNORECASE,
            ):
                value = "none"  # "None.", "None (no relatives)", "N/A", "No named entities" all mean none
            words = len(re.findall(r"\b[\w'’-]+\b", value))
            if name in {"RELATIONSHIPS", "NAMED_ENTITIES", "KEY_OBJECTS"} and value.casefold() in {"none", "n/a", "-"}:
                words = 0
            minimum = blueprint_field_floor(name)
            if words < minimum and not (name in {"RELATIONSHIPS", "NAMED_ENTITIES", "KEY_OBJECTS"} and value.casefold() == "none"):
                raise RuntimeError(f"Story Architect field {name} is underdeveloped ({words} words; minimum {minimum}).")
            normalized_fields.append(f"{name}: {value}")

        normalized_beats: list[str] = []
        for number in range(1, count + 1):
            text = " ".join(part for part in beats[number] if part).strip()
            text = re.sub(r"(?<!\w)(\*{1,2}|_{1,2})(?=\S)(.+?)(?<=\S)\1(?!\w)", r"\2", text)
            text = re.sub(r"\s+", " ", text).strip()
            words = len(re.findall(r"\b[\w'’-]+\b", text))
            # The prompt asks for beat_words_min-beat_words_max words; the gate only rejects beats too thin to carry
            # the ten state fields. Their content is enforced by the state contract, not by an exact word count.
            beat_floor = blueprint_beat_floor(STORY_PLAN_BEAT_MIN_WORDS)
            if words < beat_floor or words > max(int(STORY_PLAN_BEAT_MAX_WORDS * 1.5), 120):
                raise RuntimeError(
                    f"Story Architect scene {number:02d} beat has {words} words; expected at least {beat_floor} and no more than {max(int(STORY_PLAN_BEAT_MAX_WORDS * 1.5), 120)}."
                )
            if number == count:
                # The last scene has no next scene: whatever the model wrote there, its cause is none.
                if re.search(r"\bNEXT_CAUSE\s*=", text):
                    text = re.sub(r"(\bNEXT_CAUSE\s*=)\s*[^;]*", r"\1none", text, count=1)
                else:
                    text = text.rstrip("; ") + "; NEXT_CAUSE=none"
            normalized_beats.append(f"SCENE {number:02d}: {text}")
        return reconcile_cast_scenes("\n".join([cast_line] + normalized_fields + normalized_beats), count)

    @staticmethod
    def _scrub_unsourced_years(text: str, source: str) -> str:
        """Years the Architect invented (not in the user's source) become plain words; no model call."""
        known = set(re.findall(r"\b(?:1[5-9]\d\d|20\d\d)\b", str(source or "")))

        def _sub_prep(m):
            return m.group(0) if m.group(1) in known else "long ago"

        def _sub_bare(m):
            return m.group(0) if m.group(1) in known else ""

        text = re.sub(r"\b((?:1[5-9]\d\d|20\d\d))[-/.]\d{1,2}[-/.]\d{1,2}\b", lambda m: m.group(0) if m.group(1) in known else "an old date", str(text or ""))
        text = re.sub(r"\b(?:in|since|from|by|during|of)\s+(?:the\s+year\s+)?((?:1[5-9]\d\d|20\d\d))\b", _sub_prep, text)
        text = re.sub(r"\b((?:1[5-9]\d\d|20\d\d))\b", _sub_bare, text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return re.sub(r"[ \t]+(?=[.,;:!?\u201d\"])", "", text)

    def _story_text_user(
        self,
        mode: str,
        story: str,
        source_character_names: list[str] | None = None,
        story_blueprint: str | None = None,
    ) -> str:
        """Provide the source and validated Architect plan without adding a second template."""
        source_text = self._compact_story_context(story, DIRECTOR_STORY_CONTEXT_CHARS)
        result = f"MODE: {mode}\n\nSOURCE STORY / PREMISE:\n{source_text}"
        anchors = list(dict.fromkeys(
            str(value).strip() for value in (source_character_names or []) if str(value).strip()
        ))
        if mode == EXPAND_USER_STORY_MODE and anchors:
            result += "\n\nSOURCE CHARACTER ANCHORS (preserve exact identities):\n" + ", ".join(anchors)
        if mode == EXPAND_USER_STORY_MODE:
            result += (
                "\n\nSOURCE-PRESERVATION LOCK: retain every source character, defining event, setting anchor, "
                "relationship, twist, and stated outcome. Expand causes and consequences around those facts; do not "
                "replace the original story with a different plot or add an unrelated subplot."
            )
        if story_blueprint:
            result += (
                "\n\nVALIDATED PLOT ARCHITECTURE (binding for identities, relationships, named entities, per-scene continuity states, causal events, stakes, reversal, climax, and resolution):\n"
                + str(story_blueprint).strip()
                + "\n\nTurn each beat into dramatized action. The prose may add specific sensory and behavioral detail, "
                "but it must not change the plan's causal events, invent an unlisted person or relationship, add an unseeded named "
                "project/facility, or introduce a new rule that solves the climax. Reconcile every scene's entry and exit "
                "location, presence, knowledge, and object state with the ledger. Follow the declared on-screen scene map "
                "and cast fate rules."
            )
        writer_total = int(STORY_TARGET_WORDS)
        per_scene = writer_total // DIRECTOR_STORY_SCENES
        scene_cap = int(STORY_WRITER_WORDS_PER_SCENE)
        result += (
            f"\n\nWrite exactly {DIRECTOR_STORY_SCENES} blocks, tagged [[SCENE_01]] through "
            f"[[SCENE_{DIRECTOR_STORY_SCENES:02d}]]. Target about {writer_total} words overall "
            f"(roughly {per_scene} per scene); {STORY_MIN_WORDS} is only the rejection floor, not the target. "
            "Use three distinct dramatic movements per scene—objective/action, resistance/discovery, then decision/cost—"
            f"usually in 2-3 paragraphs with natural sentence lengths. Aim for roughly 120-140 words per scene "
            f"on average, with a soft maximum of {scene_cap} words for an individual scene; use the extra room "
            "only when the scene needs fuller action, character interaction, or aftermath. Vary scene lengths "
            "when the action needs it. "
            "Avoid a steep decline in scene length toward the end: scenes 07-12 should stay reasonably close to the length "
            "of scenes 01-06, because the climax and resolution need real room. "
            "Do not force equal paragraph or sentence counts. Give scenes 08-12 the same causal depth as scenes 01-04 "
            "and finish the full action and aftermath of Scene 12 before ending. Reach the length with consequential "
            "action, evidence, character decisions, and visible outcomes—not filler, repeated description, or extra twists."
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

    Review the beginning, middle, and ending evidence for distinct people or sentient agents performing a present-time action or materially influencing it through a live channel, then merge aliases for the same identity. Do not bias toward a protagonist-only roster or stop after the first candidate. There is no cast-count quota during extraction. Do not add people merely because a planning outline suggested them. Do not include people who appear only in memories, documents, photographs, archival recordings, logs, signs, or reported history. A name printed on a drawer, label, screen, journal, or directive is not an active character. A line spoken by one character that mentions another person is still the first character's dialogue, not evidence that the mentioned person is present or speaking. Only include a human as an active present-time cast member when the story shows that person physically acting/appearing or directly speaking in a present-time scene. Do not treat places, objects, organizations, weather, labels, and generic role words as people; a station AI may be a sentient agent but must not be mislabeled as a human PERSON. When Qwen has identified a generic sentient agent such as "AI" as a character, retain it if the story gives it live causal agency; prefer a grounded host-specific canonical label such as "Station AI" when the prose itself uses a phrase like "the station's AI", and keep the original term as an alias. Do not invent a serial number or proper name that the story never uses.

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
        cannot run to the token cap (observed: 5400 tokens, finish_reason=length, run aborted).
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
                "dramatic_purpose": self._limit_text(
                    scene.get("dramatic_purpose", ""),
                    180,
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

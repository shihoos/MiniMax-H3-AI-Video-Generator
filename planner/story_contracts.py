"""Deterministic story, entity, continuity and cinematic coverage contracts.

The contracts validate explicit planning metadata and high-confidence prose facts. They
intentionally fail closed on identity, scene-state and dialogue-integrity violations,
without pretending that lexical checks can replace human judgement of dramatic quality.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

TITLE_RE = re.compile(
    r"^(?:dr|doctor|mr|mrs|ms|miss|prof|professor|captain|capt|colonel|col|"
    r"sir|lady|sergeant|sgt|officer|detective|inspector|reverend|rev)\.?\s+",
    re.IGNORECASE,
)

RELATIONSHIP_TERMS = {
    "mother", "father", "mom", "mum", "dad", "brother", "sister", "sibling",
    "twin", "husband", "wife", "spouse", "son", "daughter", "child", "children",
    "grandmother", "grandfather", "grandma", "grandpa", "aunt", "uncle", "cousin",
    "nephew", "niece", "stepmother", "stepfather", "stepsister", "stepbrother",
    "fiance", "fiancee", "exwife", "exhusband", "girlfriend", "boyfriend", "partner",
    "grandson", "granddaughter", "halfbrother", "halfsister", "fatherinlaw", "motherinlaw",
    "brotherinlaw", "sisterinlaw",
}

PROJECT_ENTITY_RE = re.compile(
    r"\b(?:Project|Operation|Experiment|Protocol|Program|Programme|Initiative|Campaign|"
    r"Facility|Institute|Laboratory|Lab|Station|Hospital|Corporation|Company|Agency|"
    r"Department|Unit|Mission|Case|Subject)\s+[A-Z][A-Za-z0-9'’\-]*(?:\s+[A-Z][A-Za-z0-9'’\-]*){0,2}\b"
)

NAMED_FACILITY_RE = re.compile(
    r"\b([A-Z][a-zA-Z0-9'’\-]*(?:\s+[A-Z][a-zA-Z0-9'’\-]*){0,1}(?:'s|’s))\s+"
    r"(Lab|Laboratory|Station|Facility|Institute|Hospital|Archive|Observatory)\b"
)

SPEECH_VERBS = (
    "said|asked|replied|answered|whispered|shouted|called|warned|hissed|muttered|"
    "demanded|insisted|snapped|announced|added|told|ordered|argued|explained|"
    "promised|admitted|confessed|cried|barked|interrupted|radioed|transmitted"
)
ACTION_VERBS = (
    r"enter\w*|arriv\w*|appear\w*|stands?|standing|stood|walk\w*|runs?|running|ran|cross\w*|climb\w*|"
    r"follow\w*|reach\w*|turn\w*|grab\w*|holds?|holding|held|carry\w*|open\w*|close\w*|"
    r"lift\w*|pull\w*|push\w*|kneel\w*|sits?|sitting|sat|wait\w*|watch\w*|face\w*|"
    r"step\w*|mov\w*|emerge\w*|join\w*|leaves?|leaving|left|depart\w*|return\w*|look\w*|"
    r"listen\w*|remain\w*|work\w*|lean\w*|crouch\w*|slide\w*|pause\w*|"
    r"inspect\w*|examin\w*|compare\w*|record\w*|write\w*|read\w*|"
    r"speak\w*|address\w*|signal\w*|meet\w*|introduc\w*|identify\w*|"
    r"recogniz\w*|gesture\w*|embrac\w*|point\w*|nod\w*|smil\w*|frown\w*"
)

# Capitalized only because they open a sentence: seasons, weather, light and abstract nouns are not people.
COMMON_NOUN_STARTERS = {
    "winter", "summer", "spring", "autumn", "fall", "morning", "evening", "night", "midnight", "dawn", "dusk",
    "silence", "darkness", "light", "shadow", "shadows", "rain", "snow", "wind", "fog", "mist", "smoke", "dust",
    "water", "ice", "fire", "steam", "thunder", "lightning", "storm", "tide", "sunlight", "moonlight", "daylight",
    "cold", "heat", "time", "memory", "fear", "hope", "grief", "doubt", "panic", "dread", "relief", "anger",
    "nothing", "everything", "something", "someone", "everyone", "anyone", "nobody", "somebody", "everybody",
    "history", "paper", "metal", "glass", "stone", "salt", "rust", "echoes", "echo", "ash", "frost", "heat",
}

COMMON_SENTENCE_STARTERS = {
    "after", "although", "and", "as", "at", "before", "beneath", "but", "by",
    "during", "even", "finally", "for", "from", "he", "her", "his", "however",
    "in", "inside", "it", "later", "meanwhile", "near", "next", "once", "outside",
    "she", "since", "that", "the", "their", "then", "there", "they", "this", "though",
    "through", "throughout", "under", "until", "when", "while", "with", "without",
}

STOP_WORDS = {
    "about", "after", "again", "against", "along", "although", "among", "and", "another",
    "around", "because", "before", "being", "between", "could", "every", "from", "have",
    "herself", "himself", "into", "itself", "might", "other", "ourselves", "should", "their",
    "theirs", "there", "these", "they", "this", "those", "through", "under", "until", "very",
    "was", "were", "what", "when", "where", "which", "while", "whose", "would", "with",
    "without", "your", "yours", "then", "than", "them", "therefore", "that", "has", "had",
    "onto", "over", "upon", "also", "only", "more", "most", "some", "such", "many", "much",
    "each", "both", "same", "still", "just", "like", "once", "back", "away", "here", "will",
    "shall", "from", "have", "into", "the", "who", "whom", "whose", "where", "why", "how",
}

SCENE_STATE_KEYS = (
    "ENTRY_LOCATION", "ENTRY_PRESENT", "ENTRY_KNOWS", "ENTRY_OBJECTS", "EVENT",
    "EXIT_LOCATION", "EXIT_PRESENT", "EXIT_KNOWS", "EXIT_OBJECTS", "NEXT_CAUSE",
)

TRAVEL_RE = re.compile(
    r"\b(arriv\w*|enter\w*|reach\w*|travel\w*|walk\w*|drive\w*|cross\w*|climb\w*|"
    r"descend\w*|return\w*|follow\w*|move\w*|head\w*|board\w*|sail\w*|ride\w*|"
    r"hours later|next morning|by dawn|after sunset|at daybreak|the following day)\b",
    re.I,
)
# Arrival of a new person also covers meetings, introductions and rejoining, not just travel verbs.
ARRIVAL_RE = re.compile(
    TRAVEL_RE.pattern[:-3]
    + r"|meet\w*|met|join\w*|rejoin\w*|introduc\w*|encounter\w*|greet\w*|reunit\w*|"
    r"appear\w*|emerge\w*|come|comes|came|coming|show\w* up|turn\w* up|burst\w*|rush\w*|hurr\w*|"
    r"bring\w*|brought|escort\w*|lead\w*|led|summon\w*|call\w* in|wait\w* for)\b",
    re.I,
)
KNOWLEDGE_RE = re.compile(
    r"\b(learn\w*|discover\w*|realiz\w*|recogniz\w*|reveal\w*|tell\w*|show\w*|read\w*|"
    r"find\w*|decipher\w*|overhear\w*|explain\w*|admit\w*|confess\w*|prove\w*|identify\w*)\b", re.I
)
OBJECT_CHANGE_RE = re.compile(
    r"\b(take\w*|hand\w*|give\w*|steal\w*|drop\w*|leave\w*|carry\w*|move\w*|hide\w*|"
    r"open\w*|close\w*|break\w*|destroy\w*|transfer\w*|recover\w*|remove\w*|place\w*|"
    r"return\w*|seize\w*|lock\w*|unlock\w*|burn\w*|mark\w*|sign\w*)\b", re.I
)
DEATH_RE = re.compile(r"\b(die[sd]?|dead|killed|murdered|executed|perished|drowned|shot dead|succumbed)\b", re.I)
CAUSAL_LINK_RE = re.compile(
    r"\b(because|therefore|so that|so|forces?|forcing|leads? to|which means|"
    r"as a result|consequently|prompting|requiring|making|leaving|causing|"
    r"compels?|compelling|drives?|driving)\b", re.I
)


# Size gates. The prompt still asks the Architect for full-size fields and beats; these floors only reject output
# too thin to carry its structure (60% of the prompted size). Content is enforced by the contracts, not by a word count.
SOFT_SIZE_RATIO = 0.6


def blueprint_field_floor(name: str) -> int:
    prompted = 35 if name == "CONTINUITY_LEDGER" else (10 if name in {"LOGLINE", "STORY_ENGINE", "CLIMAX_CHOICE"} else 7)
    return max(4, int(prompted * SOFT_SIZE_RATIO))


def blueprint_beat_floor(prompted_minimum: int) -> int:
    return max(30, int(prompted_minimum * SOFT_SIZE_RATIO))


def word_tokens(value: str) -> list[str]:
    return [w.casefold() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", str(value or ""))]


def content_tokens(value: str) -> set[str]:
    return {w for w in word_tokens(value) if len(w) > 2 and w not in STOP_WORDS and not w.isdigit()}


def _fields_and_beats(blueprint: str) -> tuple[dict[str, str], dict[int, str], str]:
    fields: dict[str, str] = {}
    beats: dict[int, str] = {}
    cast_line = ""
    for line in str(blueprint or "").splitlines():
        if line.startswith("CAST:"):
            cast_line = line
            continue
        m = re.match(r"^([A-Z_]+):\s*(.*)$", line)
        if m and m.group(1) != "SCENE":
            fields[m.group(1)] = m.group(2).strip()
            continue
        scene = re.match(r"^SCENE\s+(\d{2}):\s*(.+)$", line)
        if scene:
            beats[int(scene.group(1))] = scene.group(2).strip()
    return fields, beats, cast_line


def parse_scene_state(beat: str) -> dict[str, str]:
    """Parse the explicit ENTRY/EVENT/EXIT/NEXT_CAUSE scene-state fields."""
    result: dict[str, str] = {}
    text = str(beat or "")
    pattern = re.compile(
        r"(?:^|;\s*)(ENTRY_LOCATION|ENTRY_PRESENT|ENTRY_KNOWS|ENTRY_OBJECTS|EVENT|"
        r"EXIT_LOCATION|EXIT_PRESENT|EXIT_KNOWS|EXIT_OBJECTS|NEXT_CAUSE)\s*=\s*", re.I
    )
    matches = list(pattern.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result[match.group(1).upper()] = text[match.end():end].strip(" ;|")
    return result


def parse_cast(blueprint: str) -> list[dict[str, Any]]:
    """Parse canonical names, aliases, on-screen scenes, and terminal fate metadata."""
    _, _, cast_line = _fields_and_beats(blueprint)
    result: list[dict[str, Any]] = []
    if not cast_line:
        return result
    for raw in cast_line.partition(":")[2].split("|"):
        item = raw.strip()
        m = re.match(r"^([^()]+?)\s*\((.*)\)\s*$", item)
        if not m:
            continue
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        if TITLE_RE.match(name):
            name = TITLE_RE.sub("", name).strip()
        metadata = m.group(2)
        alias_match = re.search(r"aliases\s*:\s*([^;]+)", metadata, re.I)
        aliases: list[str] = []
        if alias_match and alias_match.group(1).strip().casefold() not in {"none", "n/a", "-"}:
            aliases = [v.strip() for v in re.split(r",|/", alias_match.group(1)) if v.strip()]
        scene_match = re.search(r"on-screen scenes\s*:\s*([0-9, ]+)", metadata, re.I)
        scenes = [int(v) for v in re.findall(r"\d+", scene_match.group(1))] if scene_match else []
        fate_match = re.search(r"fate\s*:\s*([^;]+)", metadata, re.I)
        fate = fate_match.group(1).strip() if fate_match else "unspecified"
        result.append({"name": name, "aliases": aliases, "scenes": scenes, "fate": fate, "metadata": metadata})
    return result


def reconcile_cast_scenes(blueprint: str, expected_scenes: int = 12) -> str:
    """Make each CAST on-screen scene list agree with the per-scene ENTRY/EXIT presence ledger.

    The Architect writes both and they often drift apart. The ledger is the more detailed record the
    prose is written from, so a person's CAST scene list becomes exactly the scenes the ledger shows
    them in. Dropping an unsupported claim never adds an obligation; a person who never appears in the
    ledger is left unchanged so the contract still reports them.
    """
    fields, beats, cast_line = _fields_and_beats(blueprint)
    if not cast_line or set(beats) != set(range(1, expected_scenes + 1)):
        return blueprint
    states = {number: parse_scene_state(beats[number]) for number in beats}
    rebuilt: list[str] = []
    for raw in cast_line.partition(":")[2].split("|"):
        raw = raw.strip()
        parsed = parse_cast("CAST: " + raw)
        if len(parsed) == 1:
            item = parsed[0]
            present = [
                number for number in sorted(states)
                if _name_is_present(states[number].get("ENTRY_PRESENT", ""), item)
                or _name_is_present(states[number].get("EXIT_PRESENT", ""), item)
            ]
            if present and present != item["scenes"]:
                raw = re.sub(
                    r"on-screen scenes\s*:\s*[0-9, ]+",
                    "on-screen scenes: " + ", ".join(f"{v:02d}" for v in present),
                    raw, count=1, flags=re.IGNORECASE,
                )
        rebuilt.append(raw)
    return blueprint.replace(cast_line, "CAST: " + " | ".join(rebuilt), 1)


def _name_surfaces(name: str, aliases: list[str] | None = None) -> list[str]:
    values = [str(name or "").strip(), *(str(v).strip() for v in (aliases or []))]
    surfaces: list[str] = []
    for value in values:
        if not value:
            continue
        titleless = TITLE_RE.sub("", value).strip()
        surfaces.extend([value, titleless])
        first = titleless.split()[0] if titleless.split() else ""
        if len(first) >= 3:
            surfaces.append(first)
    return list(dict.fromkeys(v for v in surfaces if v))


def _name_is_present(value: str, item: dict[str, Any]) -> bool:
    raw = str(value or "")
    return any(
        re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", raw, re.I)
        for surface in _name_surfaces(item["name"], item.get("aliases", []))
    )


def _present_names(state: str, cast: list[dict[str, Any]]) -> set[str]:
    value = str(state or "").casefold()
    if value.strip() in {"none", "nobody", "no one", "empty"}:
        return set()
    return {item["name"].casefold() for item in cast if _name_is_present(value, item)}


def _unknown_named_surfaces(value: str, cast: list[dict[str, Any]]) -> list[str]:
    """Find capitalized person-like surfaces in a presence list not in the canonical roster."""
    known = [surface.casefold() for item in cast for surface in _name_surfaces(item["name"], item["aliases"])]
    # An honorific before a name is not a second person: "Dr. Elara Voss" is Elara Voss.
    value = re.sub(
        r"\b(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Prof|Professor|Captain|Capt|Colonel|Col|Sir|Lady|Sergeant|Sgt|Officer|Detective|Inspector|Reverend|Rev)\.?\s+(?=[A-Z])",
        "", str(value or ""),
    )
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b", str(value or ""))
    out = []
    for candidate in candidates:
        normalized = candidate.casefold()
        if normalized in known or normalized in {"none", "no one", "nobody"}:
            continue
        # Ignore sentence-start/generic labels and names embedded within a known full name.
        if normalized.split()[0] in COMMON_SENTENCE_STARTERS:
            continue
        if any(normalized in full or full in normalized for full in known):
            continue
        out.append(candidate)
    return list(dict.fromkeys(out))


def _stemmed_content_tokens(value: str) -> set[str]:
    """Small conservative suffix normalizer for prose/plan anchor matching."""
    result = set()
    for token in content_tokens(value):
        word = token
        if len(word) > 5 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 5 and word.endswith("ing"):
            word = word[:-3]
            if len(word) > 2 and word[-1:] == word[-2:-1]:
                word = word[:-1]
        elif len(word) > 4 and word.endswith("ed"):
            word = word[:-2]
        elif len(word) > 4 and word.endswith("es"):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        result.add(word)
    return result


def _has_plot_anchor(field_value: str, prose: str, minimum: int = 2) -> bool:
    planned = _stemmed_content_tokens(field_value)
    actual = _stemmed_content_tokens(prose)
    return len(planned & actual) >= minimum


def validate_blueprint_state_contract(blueprint: str, expected_scenes: int = 12) -> list[str]:
    """Hard-check explicit scene entry/exit states and high-confidence transitions."""
    fields, beats, cast_line = _fields_and_beats(blueprint)
    required_fields = {
        "LOGLINE", "PROTAGONIST_GOAL", "OPPOSITION", "PRESSURE", "STORY_ENGINE", "REVERSAL",
        "CLIMAX_CHOICE", "RESOLUTION", "RELATIONSHIPS", "NAMED_ENTITIES", "KEY_OBJECTS", "CONTINUITY_LEDGER",
    }
    errors: list[str] = []
    missing = sorted(name for name in required_fields if not fields.get(name))
    if missing:
        errors.append("missing structured plot/continuity fields: " + ", ".join(missing))
    if not cast_line:
        errors.append("missing canonical CAST line")
    if set(beats) != set(range(1, expected_scenes + 1)):
        errors.append(f"expected exactly {expected_scenes} scene-state beats")
        return errors

    cast = parse_cast(blueprint)
    if not cast:
        errors.append("CAST contains no named core people")
    cast_names = [c["name"].casefold() for c in cast]
    if len(cast_names) != len(set(cast_names)):
        errors.append("CAST contains duplicate canonical character identities")
    for item in cast:
        if not item["scenes"]:
            errors.append(f"CAST character {item['name']!r} has no planned on-screen scenes")
        if any(scene < 1 or scene > expected_scenes for scene in item["scenes"]):
            errors.append(f"CAST character {item['name']!r} has an out-of-range on-screen scene")
        if len(item["scenes"]) != len(set(item["scenes"])):
            errors.append(f"CAST character {item['name']!r} repeats an on-screen scene")
        if item["fate"].casefold() != "unspecified":
            fate = item["fate"].casefold()
            if fate not in {"survives", "alive"} and not re.fullmatch(r"dies in scene\s+\d{1,2}", fate):
                errors.append(f"CAST character {item['name']!r} has unsupported fate metadata {item['fate']!r}")
            if fate.startswith("dies"):
                death_scene = int(re.search(r"\d+", fate).group(0)) if re.search(r"\d+", fate) else None
                if death_scene is None or death_scene not in item["scenes"]:
                    errors.append(f"CAST character {item['name']!r} death scene must be included in on-screen scenes")
                elif any(scene > death_scene for scene in item["scenes"]):
                    errors.append(f"CAST character {item['name']!r} has physical appearances after their planned death")

    for number in range(1, expected_scenes + 1):
        state = parse_scene_state(beats[number])
        missing_state = [k for k in SCENE_STATE_KEYS if not state.get(k)]
        if missing_state:
            errors.append(f"scene {number:02d} missing state fields: {', '.join(missing_state)}")
            continue
        if len(word_tokens(state["EVENT"])) < 6:
            errors.append(f"scene {number:02d} event is too thin to dramatize")
        if number < expected_scenes and len(word_tokens(state["NEXT_CAUSE"])) < 5:
            errors.append(f"scene {number:02d} next-scene cause is too vague")
        if number == expected_scenes and state["NEXT_CAUSE"].casefold() not in {"none", "resolved", "complete", "n/a"}:
            errors.append("scene 12 NEXT_CAUSE must be none/resolved; the final beat must not promise another episode")
        for key in ("ENTRY_PRESENT", "EXIT_PRESENT"):
            raw = state[key].casefold()
            if raw.strip() not in {"none", "nobody", "no one", "empty"}:
                unknown = _unknown_named_surfaces(state[key], cast)
                if unknown:
                    errors.append(f"scene {number:02d} {key} contains unrostered identity: {unknown[0]}")
        for item in cast:
            present = _name_is_present(state["ENTRY_PRESENT"], item) or _name_is_present(state["EXIT_PRESENT"], item)
            if number in item["scenes"] and not present:
                errors.append(f"scene {number:02d} CAST appearance for {item['name']!r} is absent from entry/exit state")
            if present and number not in item["scenes"]:
                errors.append(
                    f"scene {number:02d} state places {item['name']!r} physically present, but the CAST on-screen scene list omits it"
                )

    # Explicit death facts in the planned event must agree with the scene's exit state.
    for number in range(1, expected_scenes + 1):
        state = parse_scene_state(beats[number])
        for item in cast:
            surfaces = _name_surfaces(item["name"], item["aliases"])
            if not any(re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", state.get("EVENT", ""), re.I) for s in surfaces):
                continue
            if DEATH_RE.search(state.get("EVENT", "")):
                if _name_is_present(state.get("EXIT_PRESENT", ""), item):
                    errors.append(f"scene {number:02d} kills {item['name']!r} but leaves them physically present at exit")
                if item["fate"].casefold() == "survives":
                    errors.append(f"scene {number:02d} kills {item['name']!r} despite CAST fate=survives")

    # The compact global ledger must be anchored to the canonical protagonist and key world state.
    ledger = fields.get("CONTINUITY_LEDGER", "")
    ledger_names = [item for item in cast if _name_is_present(ledger, item)]
    if cast and not ledger_names:
        errors.append("CONTINUITY_LEDGER does not identify any canonical cast member")
    key_objects = fields.get("KEY_OBJECTS", "")
    if key_objects.casefold() not in {"none", "n/a", "-"}:
        object_anchors = content_tokens(key_objects) & content_tokens(ledger)
        if len(object_anchors) < 2:
            errors.append("CONTINUITY_LEDGER does not anchor the declared key objects")

    # Every non-final NEXT_CAUSE must express a causal consequence and feed the next event.
    for number in range(1, expected_scenes):
        current = parse_scene_state(beats[number])
        following = parse_scene_state(beats[number + 1])
        cause = current.get("NEXT_CAUSE", "")
        if not CAUSAL_LINK_RE.search(cause):
            errors.append(f"scene {number:02d} NEXT_CAUSE lacks an explicit causal connector")
        cause_tokens = content_tokens(cause)
        next_event_tokens = content_tokens(following.get("EVENT", ""))
        shared = cause_tokens & next_event_tokens
        if len(shared) < 1:
            errors.append(f"scene {number + 1:02d} event is not anchored to the previous scene's NEXT_CAUSE")

    # Scene-to-scene continuity for physical location, cast presence, knowledge and object state.
    for number in range(1, expected_scenes):
        prior = parse_scene_state(beats[number])
        current = parse_scene_state(beats[number + 1])
        old_location = re.sub(r"\W+", " ", prior.get("EXIT_LOCATION", "").casefold()).strip()
        new_location = re.sub(r"\W+", " ", current.get("ENTRY_LOCATION", "").casefold()).strip()
        bridge = current.get("EVENT", "") + " " + prior.get("NEXT_CAUSE", "")
        if old_location and new_location and old_location != new_location and not TRAVEL_RE.search(bridge):
            errors.append(f"scene {number + 1:02d} changes location without a route/time transition")

        old_present = _present_names(prior.get("EXIT_PRESENT", ""), cast)
        new_present = _present_names(current.get("ENTRY_PRESENT", ""), cast)
        arriving = new_present - old_present
        if arriving and not ARRIVAL_RE.search(current.get("EVENT", "")):
            errors.append(f"scene {number + 1:02d} introduces {', '.join(sorted(arriving))} without an arrival/meeting event")
        departing = old_present - new_present
        if departing and not re.search(r"\b(leave\w*|depart\w*|exit\w*|return\w*|separate\w*|stay\w*|remain\w*|fall\w*|die\w*|killed|drowned|captured|taken)\b", bridge, re.I):
            errors.append(f"scene {number:02d} removes {', '.join(sorted(departing))} without an exit/fate transition")

        before_knowledge = content_tokens(prior.get("EXIT_KNOWS", ""))
        after_knowledge = content_tokens(current.get("ENTRY_KNOWS", ""))
        gained = after_knowledge - before_knowledge
        lost = before_knowledge - after_knowledge
        if len(gained) >= 3 and not KNOWLEDGE_RE.search(current.get("EVENT", "")):
            errors.append(f"scene {number + 1:02d} changes established knowledge without a discovery/disclosure event")
        if len(lost) >= 3 and not re.search(r"\b(forget\w*|misremember\w*|doubt\w*|retract\w*|disprove\w*|correct\w*|mislead\w*)\b", current.get("EVENT", ""), re.I):
            errors.append(f"scene {number + 1:02d} removes established knowledge without a reason")

        before_objects = content_tokens(prior.get("EXIT_OBJECTS", ""))
        after_objects = content_tokens(current.get("ENTRY_OBJECTS", ""))
        object_delta = before_objects ^ after_objects
        if len(object_delta) >= 3 and not OBJECT_CHANGE_RE.search(current.get("EVENT", "")):
            errors.append(f"scene {number + 1:02d} changes important object custody/state without a physical action")

    return list(dict.fromkeys(errors))


def _entity_allowlist(fields: dict[str, str], source_story: str) -> str:
    return " ".join((str(source_story or ""), fields.get("NAMED_ENTITIES", ""), fields.get("RELATIONSHIPS", ""))).casefold()


def _is_sentence_initial_common_word(story: str, match: re.Match, candidate: str, first: str, allowed_text: str) -> bool:
    """A single capitalized word that only opens a sentence ("Winter followed...") is a word, not a person."""
    if len(candidate.split()) != 1:
        return False
    before = story[:match.start()]
    at_sentence_start = not before.strip() or bool(re.search(r"(?:[.!?]['\"”’)]*|\n)\s*['\"“‘(]*$", before))
    if not at_sentence_start:
        return False
    if first in COMMON_NOUN_STARTERS:
        return True
    # The same word used in lower case elsewhere is an ordinary word; a person's name is never lower-cased.
    # A word the source or the plan itself spells as a name is not excused.
    if re.search(r"(?<![A-Za-z])" + re.escape(first) + r"(?![A-Za-z])", allowed_text):
        return False
    return bool(re.search(r"(?<![A-Za-z])" + re.escape(first) + r"(?![A-Za-z])", story))


POSSESSIVE_BODY = (
    r"hands?|fingers?|eyes?|voice|breath|arms?|shoulders?|face|lips?|feet|footsteps?|gaze|head|knees?|"
    r"mouth|jaw|palms?|pulse|heartbeat|chest|legs?|thumbs?|knuckles?|wrists?|fists?"
)
POSSESSIVE_ACTION_EXTRA = (
    r"tighten\w*|trembl\w*|clench\w*|curl\w*|twitch\w*|flutter\w*|shak\w*|grip\w*|grasp\w*|press\w*|"
    r"trac\w*|brush\w*|rais\w*|wrap\w*|squeez\w*|rest\w*|slip\w*|drum\w*|steady\w*|steadied|fumbl\w*"
)


def _person_acts_in(paragraph: str, surfaces: list[str]) -> bool:
    """True when the paragraph shows the person acting, speaking, or a body part acting for them."""
    for surface in surfaces:
        name = r"(?<!\w)" + re.escape(surface) + r"(?!\w)"
        if re.search(name + r"\s+(?:" + SPEECH_VERBS + r"|" + ACTION_VERBS + r")\b", paragraph, re.I):
            return True
        if re.search(
            name + r"(?:'s|’s)\s+(?:" + POSSESSIVE_BODY + r")\s+(?:" + ACTION_VERBS + r"|" + POSSESSIVE_ACTION_EXTRA + r")\b",
            paragraph, re.I,
        ):
            return True
    return False


def audit_story_against_blueprint(
    story: str,
    blueprint: str,
    source_story: str = "",
    expected_scenes: int = 12,
) -> list[str]:
    """Audit the prose against canonical entities, relationships, scene beats and states."""
    fields, beats, _ = _fields_and_beats(blueprint)
    cast = parse_cast(blueprint)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", str(story or "")) if p.strip()]
    if len(paragraphs) != expected_scenes:
        return [f"story/blueprint audit expected {expected_scenes} paragraphs, found {len(paragraphs)}"]
    allowed_text = _entity_allowlist(fields, source_story)
    errors: list[str] = []

    # Plot fields are not decorative metadata: the finished prose must visibly carry
    # the protagonist goal, opposition, story engine, reversal, climax choice, and resolution.
    early_story = " ".join(paragraphs[:4])
    middle_story = " ".join(paragraphs[3:10])
    late_story = " ".join(paragraphs[8:])
    ending_story = " ".join(paragraphs[-2:])
    plot_anchors = (
        ("PROTAGONIST_GOAL", early_story, 2),
        ("OPPOSITION", " ".join(paragraphs[:9]), 2),
        ("STORY_ENGINE", " ".join(paragraphs), 2),
        ("REVERSAL", middle_story, 2),
        ("CLIMAX_CHOICE", late_story, 2),
        ("RESOLUTION", ending_story, 2),
    )
    for field_name, prose, minimum in plot_anchors:
        value = fields.get(field_name, "")
        if value and not _has_plot_anchor(value, prose, minimum):
            errors.append(f"story prose does not visibly realize Architect field {field_name}")

    # Named programs, facilities and institutions must originate in source or Architect plan.
    for regex in (PROJECT_ENTITY_RE, NAMED_FACILITY_RE):
        for match in regex.finditer(str(story or "")):
            entity = re.sub(r"\s+", " ", match.group(0)).strip()
            if entity.casefold() not in allowed_text:
                errors.append(f"unseeded named project/facility/organization: {entity}")

    # Kinship labels and personal relationship types must be grounded in source or RELATIONSHIPS.
    allowed_relationships = set(re.findall(r"[a-z]+", allowed_text))
    used_relationships = set(re.findall(r"[a-z]+", str(story or "").casefold())) & RELATIONSHIP_TERMS
    for term in sorted(used_relationships - allowed_relationships):
        errors.append(f"unseeded relationship/relative term in story: {term}")
    # Catch owner-specific contradictions such as the plan seeding a brother while prose
    # silently changes that person into a sister. The broad term test above catches additions;
    # this pass catches a change in the named relationship owner/type pair.
    relation_pattern = re.compile(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)(?:'s|’s)\s+"
        r"(" + "|".join(sorted(RELATIONSHIP_TERMS, key=len, reverse=True)) + r")\b", re.I
    )
    allowed_relation_pairs = {
        (m.group(1).casefold(), m.group(2).casefold())
        for m in relation_pattern.finditer(str(source_story or "") + " " + fields.get("RELATIONSHIPS", ""))
    }
    for match in relation_pattern.finditer(str(story or "")):
        pair = (match.group(1).casefold(), match.group(2).casefold())
        if pair not in allowed_relation_pairs:
            errors.append(f"unseeded/altered relationship for {match.group(1)}: {match.group(2)}")

    allowed_names = {c["name"].casefold() for c in cast}
    allowed_names.update(a.casefold() for c in cast for a in c["aliases"])
    source_names = {
        value.casefold() for value in re.findall(
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
            str(source_story or "") + " " + fields.get("RELATIONSHIPS", ""),
        )
    }
    source_attribution_re = re.compile(
        rf"\b([A-Z][a-z]{{2,}}(?:\s+[A-Z][a-z]{{2,}}){{0,2}})\s+(?:{SPEECH_VERBS}|{ACTION_VERBS})\b"
    )
    source_names.update(m.group(1).casefold() for m in source_attribution_re.finditer(str(source_story or "")))
    # Source names may be mentioned, but an unrostered source-only person cannot silently
    # become a physical actor or an unqualified in-person speaker in the new story.
    allowed_surfaces = {s.casefold() for c in cast for s in _name_surfaces(c["name"], c["aliases"])}
    source_only_names = source_names - allowed_surfaces - allowed_names
    unknown_people: set[str] = set()

    honorific = re.compile(
        r"\b(?:Dr\.?|Doctor|Mr\.?|Mrs\.?|Ms\.?|Professor|Prof\.?|Captain|Capt\.?|Colonel|Col\.?|Officer|Detective|Inspector)\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
    )
    for match in honorific.finditer(str(story or "")):
        candidate = match.group(1).strip().casefold()
        if candidate in allowed_names or candidate in allowed_surfaces:
            continue
        sentence_start = max(str(story or "").rfind(mark, 0, match.start()) for mark in ".!?\n") + 1
        sentence_end_candidates = [str(story or "").find(mark, match.end()) for mark in ".!?\n"]
        sentence_end_candidates = [v for v in sentence_end_candidates if v >= 0]
        sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(str(story or ""))
        sentence_context = str(story or "")[sentence_start:sentence_end].casefold()
        source_only = candidate in source_only_names or candidate.split()[0] in source_only_names
        remote_audio = bool(re.search(r"\b(radio|comms|intercom|phone|transmission|receiver|earpiece|broadcast|channel|voice|hologram)\b", sentence_context))
        has_speech = bool(re.search(rf"\b(?:{SPEECH_VERBS})\b", sentence_context))
        if not (source_only and remote_audio and has_speech):
            unknown_people.add(match.group(1).strip())

    name_attribution = re.compile(
        rf"\b([A-Z][a-z]{{2,}}(?:\s+[A-Z][a-z]{{2,}}){{0,2}})\s+(?:{SPEECH_VERBS}|{ACTION_VERBS})\b"
    )
    for match in name_attribution.finditer(str(story or "")):
        candidate = re.sub(r"\s+", " ", match.group(1)).strip()
        first = candidate.split()[0].casefold()
        if first in COMMON_SENTENCE_STARTERS:
            continue
        if _is_sentence_initial_common_word(str(story or ""), match, candidate, first, allowed_text):
            continue
        candidate_key = candidate.casefold()
        role_words = {
            "clerk", "guard", "officer", "captain", "driver", "pilot", "doctor", "nurse", "engineer",
            "technician", "archivist", "surveyor", "operator", "worker", "soldier", "detective", "teacher",
            "porter", "servant", "attendant", "stranger", "witness", "sailor", "mechanic", "assistant",
        }
        if len(candidate.split()) > 1 and set(candidate_key.split()) & role_words:
            continue
        source_text = str(source_story or "")
        sentence_start = max(str(story or "").rfind(mark, 0, match.start()) for mark in ".!?\n") + 1
        sentence_end_candidates = [str(story or "").find(mark, match.end()) for mark in ".!?\n"]
        sentence_end_candidates = [v for v in sentence_end_candidates if v >= 0]
        sentence_end = min(sentence_end_candidates) if sentence_end_candidates else len(str(story or ""))
        context = str(story or "")[sentence_start:sentence_end].casefold()
        remote_context = bool(re.search(
            r"\b(radio|comms|communication|intercom|phone|telephone|transmission|transmitter|receiver|"
            r"earpiece|speaker|broadcast|channel|static|voice|recording|hologram|monitor|screen)\b", context
        ))
        is_physical_action = bool(re.search(rf"\b(?:{ACTION_VERBS})\b", match.group(0), re.I))
        is_speech = bool(re.search(rf"\b(?:{SPEECH_VERBS})\b", match.group(0), re.I))
        source_only_match = candidate_key in source_only_names or first in source_only_names
        if source_only_match and not is_physical_action and is_speech and remote_context:
            continue
        if candidate_key not in allowed_names and candidate_key not in allowed_surfaces and first not in allowed_surfaces:
            unknown_people.add(candidate)

    named_intro = re.compile(
        r"\b(?:a|an|the)\s+(?:man|woman|boy|girl|person|officer|clerk|driver|guard|doctor|nurse|stranger)\s+"
        r"(?:named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
    )
    for match in named_intro.finditer(str(story or "")):
        candidate = match.group(1).strip()
        if candidate.casefold() not in allowed_names and candidate.casefold() not in allowed_surfaces:
            unknown_people.add(candidate)
    for person in sorted(unknown_people):
        errors.append(f"unseeded named person/title-contaminated identity: {person}")

    # Every planned physical appearance must be represented in its declared scene, not just elsewhere.
    for item in cast:
        if not item["scenes"]:
            errors.append(f"CAST character {item['name']!r} has no planned on-screen scenes")
            continue
        surfaces = _name_surfaces(item["name"], item["aliases"])
        for scene_number in item["scenes"]:
            if not 1 <= scene_number <= len(paragraphs):
                errors.append(f"CAST character {item['name']!r} has out-of-range appearance scene {scene_number}")
                continue
            paragraph = paragraphs[scene_number - 1]
            mentioned = any(re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", paragraph, re.I) for surface in surfaces)
            if not mentioned:
                errors.append(f"planned character {item['name']!r} is missing from physical scene {scene_number:02d}")
                continue
            # If the person is only mentioned but never acts, speaks, or is described as physically present,
            # the CAST on-screen claim is not supported by the prose.
            physical_cue = re.compile(
                r"\b(" + ACTION_VERBS + r"|" + SPEECH_VERBS + r"|stood|sat|waited|watched|listened|"
                r"remained|knelt|leaned|crouched|was beside|stood beside|sat beside|entered|exited)\b", re.I
            )
            sentence_contexts = re.split(r"(?<=[.!?])\s+", paragraph)
            context_has_action = False
            for sentence in sentence_contexts:
                if any(re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", sentence, re.I) for surface in surfaces):
                    if physical_cue.search(sentence):
                        context_has_action = True
                        break
            if not context_has_action:
                errors.append(f"planned on-screen character {item['name']!r} has no physical/speech action in scene {scene_number:02d}")

    # Compare story paragraphs with planned events and hand-offs using content words, not field labels.
    for number in range(1, expected_scenes + 1):
        state = parse_scene_state(beats.get(number, ""))
        paragraph = paragraphs[number - 1]
        event_tokens = content_tokens(state.get("EVENT", ""))
        story_tokens = content_tokens(paragraph)
        overlap = event_tokens & story_tokens
        if len(event_tokens) >= 5 and (len(overlap) < 2 or len(overlap) / max(1, len(event_tokens)) < 0.10):
            errors.append(f"scene {number:02d} prose does not clearly realize its planned event")
        if number > 1:
            cause_tokens = content_tokens(parse_scene_state(beats[number - 1]).get("NEXT_CAUSE", ""))
            if len(cause_tokens) >= 5 and len(cause_tokens & story_tokens) < 2:
                errors.append(f"scene {number:02d} does not visibly realize the previous scene's causal hand-off")

        if number > 1:
            prior = parse_scene_state(beats[number - 1])
            prior_loc = content_tokens(prior.get("EXIT_LOCATION", ""))
            current_loc = content_tokens(state.get("ENTRY_LOCATION", ""))
            if prior_loc and current_loc and prior_loc != current_loc:
                opening = " ".join(paragraph.split()[:90])
                if not TRAVEL_RE.search(opening):
                    errors.append(f"scene {number:02d} prose changes location without showing travel or elapsed time")

            # If the planned state changed, prose must show the corresponding transition.
            prior_people = _present_names(prior.get("EXIT_PRESENT", ""), cast)
            entry_people = _present_names(state.get("ENTRY_PRESENT", ""), cast)
            newly_present = entry_people - prior_people
            if newly_present and not ARRIVAL_RE.search(" ".join(paragraph.split()[:90])):
                errors.append(f"scene {number:02d} prose does not show arrival of {', '.join(sorted(newly_present))}")
            departed = prior_people - entry_people
            prior_paragraph = paragraphs[number - 2]
            if departed and not re.search(r"\b(leave\w*|depart\w*|exit\w*|return\w*|separate\w*|stay\w*|remain\w*|fall\w*|die\w*|killed|drowned|captured|taken)\b", prior_paragraph, re.I):
                errors.append(f"scene {number - 1:02d} removes a planned person without an exit/fate cue")

            prior_knowledge = content_tokens(prior.get("EXIT_KNOWS", ""))
            entry_knowledge = content_tokens(state.get("ENTRY_KNOWS", ""))
            if len(entry_knowledge - prior_knowledge) >= 3 and not KNOWLEDGE_RE.search(paragraph):
                errors.append(f"scene {number:02d} prose does not show the planned knowledge transition")
            if len(prior_knowledge - entry_knowledge) >= 3 and not re.search(r"\b(forget\w*|misremember\w*|doubt\w*|retract\w*|disprove\w*|correct\w*|mislead\w*)\b", paragraph, re.I):
                errors.append(f"scene {number:02d} prose does not explain the changed knowledge state")

            prior_objects = content_tokens(prior.get("EXIT_OBJECTS", ""))
            entry_objects = content_tokens(state.get("ENTRY_OBJECTS", ""))
            object_delta = prior_objects ^ entry_objects
            if len(object_delta) >= 3:
                if not OBJECT_CHANGE_RE.search(paragraph):
                    errors.append(f"scene {number:02d} prose does not show the planned object custody/state change")
                if not (object_delta & content_tokens(paragraph)):
                    errors.append(f"scene {number:02d} prose does not name the object involved in its state change")

    # A person may act, speak or be physically described only in the scenes their CAST entry lists, and never
    # after their planned death. Possessive forms ("Ines's fingers close...") count as physical activity.
    for item in cast:
        fate = item.get("fate", "unspecified").casefold()
        death_scene = int(re.search(r"\d+", fate).group(0)) if fate.startswith("dies in scene") and re.search(r"\d+", fate) else None
        listed = set(item["scenes"])
        for scene_number in range(1, expected_scenes + 1):
            after_death = death_scene is not None and scene_number > death_scene
            unlisted = bool(listed) and scene_number not in listed
            if not (after_death or unlisted):
                continue
            if _person_acts_in(paragraphs[scene_number - 1], _name_surfaces(item["name"], item["aliases"])):
                if after_death:
                    errors.append(f"dead character {item['name']!r} physically acts/speaks after scene {death_scene:02d}")
                else:
                    errors.append(
                        f"{item['name']!r} acts/speaks in scene {scene_number:02d}, which the CAST on-screen scene list omits"
                    )

    return list(dict.fromkeys(errors))


def ensure_cinematic_coverage(shots: list[dict], scene_id: str = "") -> tuple[list[dict], list[str]]:
    """Guarantee distinct shot framings without forcing camera movement or inventing action."""
    if not isinstance(shots, list):
        return [], [f"{scene_id}: shots must be a list"]
    result = [dict(s) if isinstance(s, dict) else s for s in shots]
    try:
        from planner.qwen_director_prompts import CAMERA_SHOT_OPTIONS, CAMERA_MOVEMENT_OPTIONS, DEFAULT_CAMERA_SHOT
        valid_frames = list(CAMERA_SHOT_OPTIONS)
        valid_moves = set(CAMERA_MOVEMENT_OPTIONS)
    except Exception:
        valid_frames = [
            "extreme wide", "wide", "full", "medium wide", "medium", "medium close-up",
            "close-up", "extreme close-up", "over-the-shoulder", "two-shot", "pov", "insert",
        ]
        valid_moves = {"static", "pan", "tilt", "dolly", "tracking", "handheld", "crane", "push-in", "orbit"}
        DEFAULT_CAMERA_SHOT = "medium wide"

    object_re = re.compile(r"\b(key|map|journal|diary|photograph|photo|screen|monitor|panel|console|document|letter|file|note|device|instrument|blade|ring|token|seal|permit|record|evidence|switch|lever|handle|lock)\b", re.I)
    motion_re = re.compile(r"\b(run|ran|walk|walked|cross|crossed|climb|climbed|drive|drove|move|moved|enter|entered|leave|left|follow|followed|approach|approached|carry|carried|chase|chased|sprint|sprinted)\b", re.I)
    reaction_re = re.compile(r"\b(face|eyes|expression|realiz\w*|react\w*|hesitat\w*|fear|grief|shock|stare|watched|looked|turn\w*|whisper\w*)\b", re.I)

    def preference(shot: dict) -> list[str]:
        text = " ".join(str(shot.get(k, "") or "") for k in ("action", "visual_prompt", "composition_notes"))
        chars = shot.get("characters") or []
        if object_re.search(text):
            return ["insert", "close-up", "extreme close-up", "medium", "over-the-shoulder"]
        if motion_re.search(text):
            return ["full", "wide", "medium wide", "over-the-shoulder"]
        if len(chars) >= 2:
            return ["two-shot", "over-the-shoulder", "medium", "medium wide"]
        if reaction_re.search(text):
            return ["close-up", "medium close-up", "over-the-shoulder", "medium"]
        return ["wide", "medium", "medium close-up", "close-up", "full", "extreme wide"]

    used: set[str] = set()
    adjustments: list[str] = []
    coverage_notes = {
        "extreme wide": "Use the frame to clarify the full environment and spatial relationships.",
        "wide": "Keep the subject grounded in the surrounding geography.",
        "full": "Keep full-body movement and blocking readable.",
        "medium wide": "Show the subject and nearby physical context together.",
        "medium": "Prioritize the character's action within the scene geography.",
        "medium close-up": "Emphasize the character's response while retaining context.",
        "close-up": "Emphasize the supported facial response or decisive detail.",
        "extreme close-up": "Isolate only a detail already established in the shot text.",
        "over-the-shoulder": "Clarify the established eyeline and relationship between subjects.",
        "two-shot": "Keep both established participants and their spatial relationship visible.",
        "pov": "Use only the established viewpoint and its visible target.",
        "insert": "Isolate only an object or evidence item already established in the shot text.",
    }
    for index, shot in enumerate(result):
        if not isinstance(shot, dict):
            continue
        raw = str(shot.get("camera_shot", "") or "").strip().lower()
        frame = raw if raw in valid_frames else DEFAULT_CAMERA_SHOT
        if frame in used:
            candidates = [v for v in preference(shot) if v in valid_frames and v not in used]
            # Never invent a POV, second participant, or isolated prop merely to meet
            # a framing-diversity requirement. The generic fallback uses neutral
            # coverage sizes; specific shot types require evidence in the shot.
            candidates += [
                v for v in (
                    "extreme wide", "wide", "full", "medium wide", "medium",
                    "medium close-up", "close-up", "extreme close-up",
                ) if v in valid_frames and v not in used
            ]
            replacement = candidates[0] if candidates else frame
            if replacement != frame:
                frame = replacement
                adjustments.append(f"{scene_id or 'scene'} shot {index + 1}: repeated framing normalized to {frame!r}")
        if raw not in valid_frames or str(shot.get("camera_shot", "")).strip().lower() != frame:
            shot["camera_shot"] = frame
            if raw not in valid_frames:
                adjustments.append(f"{scene_id or 'scene'} shot {index + 1}: unsupported framing normalized to {frame!r}")
        used.add(frame)

        raw_move = str(shot.get("camera_movement", "static") or "static").strip().lower()
        if raw_move not in valid_moves:
            shot["camera_movement"] = "static"
            adjustments.append(f"{scene_id or 'scene'} shot {index + 1}: unsupported movement normalized to static")

        existing = str(shot.get("composition_notes", "") or "").strip()
        note = coverage_notes.get(frame, "Use a distinct, story-grounded composition.")
        if note.casefold() not in existing.casefold():
            shot["composition_notes"] = (existing + " " + note).strip()[:240]
    return result, adjustments


def validate_cinematic_coverage(shots: list[dict], scene_id: str = "") -> list[str]:
    """Hard-check framing diversity; report rather than rewrite near-duplicate actions."""
    errors: list[str] = []
    valid = [s for s in shots if isinstance(s, dict)]
    frames = [str(s.get("camera_shot", "") or "").casefold() for s in valid]
    if len(valid) >= 3 and len(set(frames)) < 3:
        errors.append(f"{scene_id}: fewer than three distinct framings")
    actions = [re.sub(r"\W+", " ", str(s.get("action", "") or "").casefold()).strip() for s in valid]
    for i in range(len(actions)):
        for j in range(i + 1, len(actions)):
            if len(actions[i].split()) >= 8 and SequenceMatcher(None, actions[i], actions[j]).ratio() >= 0.88:
                errors.append(f"{scene_id}: shots {i + 1} and {j + 1} describe near-duplicate actions")
    return errors

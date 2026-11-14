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
    r"lieutenant|lt|major|maj|commander|cmdr|admiral|adm|general|gen|"
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
    r"hesitat\w*|pause\w*|volunteer\w*|"
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
    "PURPOSE", "ENTRY_LOCATION", "ENTRY_PRESENT", "ON_SCREEN_PRESENT", "ENTRY_KNOWS", "ENTRY_OBJECTS",
    "EVENT", "EXIT_LOCATION", "EXIT_PRESENT", "EXIT_KNOWS", "EXIT_OBJECTS", "NEXT_CAUSE",
)

TRAVEL_RE = re.compile(
    r"\b(arriv\w*|enter\w*|reach\w*|travel\w*|walk\w*|drive\w*|cross\w*|climb\w*|"
    r"descend\w*|return\w*|follow\w*|move\w*|head\w*|board\w*|sail\w*|ride\w*|"
    r"disappear\w*|withdraw\w*|vanish\w*|depart\w*|slip away|go\w*|separate\w*|"
    r"take shelter|seek shelter|retreat\w*|duck inside|shelter\w* inside|"
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
    r"find\w*|decipher\w*|overhear\w*|explain\w*|admit\w*|confess\w*|prove\w*|identify\w*|"
    r"detect\w*|observ\w*|witness\w*|confirm\w*)\b", re.I
)
OBJECT_CHANGE_RE = re.compile(
    r"\b(take\w*|hand\w*|give\w*|steal\w*|drop\w*|leave\w*|carry\w*|move\w*|hide\w*|"
    r"open\w*|close\w*|break\w*|destroy\w*|transfer\w*|recover\w*|retriev\w*|remove\w*|place\w*|"
    r"find\w*|discover\w*|locate\w*|return\w*|seize\w*|lock\w*|unlock\w*|burn\w*|mark\w*|sign\w*|"
    r"inject\w*|activat\w*|dispers\w*|releas\w*|escap\w*|consum\w*|use\w*|stabiliz\w*|"
    r"dissolv\w*|absorb\w*|seal\w*|unseal\w*)\b", re.I
)
CAUSAL_LINK_RE = re.compile(
    r"\b(because|therefore|so that|so|thus|hence|forces?|forcing|forced|leads? to|leading to|led to|which means|"
    r"which (?:reveals?|forces?|exposes?|triggers?|requires?|demands?|leaves?|puts?|sends?|pushes?)|"
    r"as a result|consequently|prompting|prompts?|requiring|requires?|making|leaving|causing|"
    r"compels?|compelling|drives?|driving|triggers?|triggering|sets? off|results? in|resulting in|"
    r"means|demands?|pushes?|pushing|sends?|sending|exposes?|exposing|reveals?|revealing|"
    r"now that|once|since|after|unless|before|in order to|must|needs? to|has to|have to)\b", re.I
)


def _event_explicitly_assigns_death(event: str, item: dict[str, Any]) -> bool:
    """Match an explicit death predicate to its named subject/object, not any name in the beat."""
    value = str(event or "")
    for surface in _name_surfaces(item.get("name", ""), item.get("aliases", [])):
        name = re.escape(surface)
        # Intransitive/predicative death: Kael dies; Kael is dead; Kael, pinned under debris, dies.
        subject = (
            r"\b" + name + r"\b(?:\s*,\s*[^,;.!?]{1,48},)?\s+"
            r"(?:dies|died|is dead|was dead|perishes|perished|drowns|drowned|succumbs|succumbed)\b"
        )
        passive = (
            r"\b" + name + r"\b\s+(?:is|was|gets|got)\s+"
            r"(?:killed|murdered|executed|drowned|shot dead|crushed)\b"
        )
        # Transitive death where this character is the victim: the beam kills Kael / killing Kael.
        object_verb = (
            r"\b(?:kills?|killing|murders?|murdering|executes?|executing|drowns?|drowning|"
            r"shoots?|shooting|crushes?|crushing|strikes down|striking down)\b"
            r"[^.;!?]{0,40}\b" + name + r"\b"
        )
        if any(re.search(pattern, value, re.I) for pattern in (subject, passive, object_verb)):
            return True
    return False

# Printed in the diagnostic artifacts so the code version that ran is unambiguous.
CONTRACTS_REVISION = "2026-11-14.5.1"

# Hard findings cover structural facts and continuity contradictions: missing/invalid structure,
# impossible cast state, ungrounded setup/payoff evidence, physical location jumps, and declared key-object
# custody/state changes that the EVENT does not perform. Purely lexical NEXT_CAUSE paraphrase checks remain
# advisory; the validator must not confuse wording variation with a broken causal chain.
HARD_FINDING_MARKERS = (
    "missing structured plot/continuity fields",
    "SETUP_PAYOFF_MAP must contain",
    "SETUP_PAYOFF_MAP plant and payoff details",
    "SETUP_PAYOFF_MAP has invalid ordering",
    "PURPOSE is too vague",
    "missing canonical CAST line",
    "expected exactly",
    "CAST contains no named",
    "duplicate canonical character",
    "has no planned on-screen scenes",
    "out-of-range",
    "repeats an on-screen scene",
    "repeats the preceding scene beat",
    "unsupported fate metadata",
    "death scene must be included",
    "physical appearances after their planned death",
    "missing state fields",
    "NEXT_CAUSE must be none/resolved",
    "kills ",
    "CONTINUITY_LEDGER does not identify any canonical cast member",
    "without a character-specific arrival",
    "without a character-specific departure",
    "has no character-specific arrival",
    "has no character-specific departure",
    "has no character-specific EVENT anchor",
    "entry object inventory introduces",
    "drops terminal object identity",
    "key object is missing from scene inventory",
    "survivor fate is unresolved",
    "CAST fate says",
    "death of ",
    "object inventory drops",
    "object inventory introduces",
    "object identity is not carried",
    "object state change is not anchored",
    "key-object state change is not anchored",
    "has no NEXT_CAUSE handoff",
    # These findings describe explicit structural contradictions, not stylistic wording preferences.
    "SETUP_PAYOFF_MAP plant in scene",
    "SETUP_PAYOFF_MAP payoff in scene",
    "SETUP_PAYOFF_MAP payoff scene",
    "changes its physical location without an explicit transition",
    "object state change is not anchored to the named object and a physical action",
)
# Prose findings that may stop a run: invented people/organisations/relationships and the dead acting.
HARD_PROSE_MARKERS = (
    "unseeded",
    "dead character",
    "title-contaminated",
    "has out-of-range appearance scene",
)


def split_contract_findings(findings: list[str], hard_markers: tuple[str, ...] = HARD_FINDING_MARKERS) -> tuple[list[str], list[str]]:
    """Return (fatal, warnings), keeping lexical/style preferences advisory and state contradictions fatal."""
    fatal: list[str] = []
    soft: list[str] = []
    for item in findings:
        (fatal if any(marker in item for marker in hard_markers) else soft).append(item)
    return fatal, soft


# Size gates. The prompt still asks the Architect for full-size fields and beats; these floors only reject output
# too thin to carry its structure (60% of the prompted size). Content is enforced by the contracts, not by a word count.
SOFT_SIZE_RATIO = 0.6


def blueprint_field_floor(name: str) -> int:
    if name == "CONTINUITY_LEDGER":
        prompted = 35
    elif name == "SETUP_PAYOFF_MAP":
        prompted = 12
    elif name in {"LOGLINE", "STORY_ENGINE", "CLIMAX_CHOICE"}:
        prompted = 10
    else:
        prompted = 7
    return max(4, int(prompted * SOFT_SIZE_RATIO))


def blueprint_beat_floor(prompted_minimum: int) -> int:
    return max(30, int(prompted_minimum * SOFT_SIZE_RATIO))


def word_tokens(value: str) -> list[str]:
    # Normalize possessives to their lexical base. Without this, "Elara's" and
    # "Elara" never intersect, creating false causal-handoff warnings.
    result: list[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", str(value or "")):
        normalized = token.casefold()
        normalized = re.sub(r"['’]s$", "", normalized)
        if normalized:
            result.append(normalized)
    return result


def content_tokens(value: str) -> set[str]:
    return {
        w for w in word_tokens(value)
        if len(w) > 2 and w not in STOP_WORDS and w not in {"scene", "scenes"} and not w.isdigit()
    }


_LOCATION_GENERIC_WORDS = {
    "the", "a", "an", "inside", "interior", "outside", "exterior", "room", "corridor",
    "entrance", "exit", "lower", "upper", "main", "central", "north", "south",
    "east", "west", "area", "sector", "point", "platform", "level", "floor",
    # Site descriptors are shared by every room in a facility; they are not proof
    # that the EVENT names the actual destination. Keep room identifiers such as
    # "control", "archive" and "hidden" as destination anchors.
    "abandoned", "research", "station",
}

# Physical-PASSAGE verbs. Opening/unlocking/bypassing/breaching a door is only an access action: it must be
# followed by an actual crossing in the same clause. Discovery/search verbs (explore, find, locate, discover,
# search, investigate, approach, follow) are deliberately NOT movement: finding a vault is not entering it.
# Verbs that are ambiguous without a direction (proceed, venture, rush, ...) require a directional preposition.
_SCENE_LOCAL_MOVE_RE = re.compile(
    r"\b(?:enter\w*|go(?:es|ing)?\s+inside|step(?:s|ped|ping)?\s+inside|cross\w*|pass(?:es|ed|ing)?\s+through|"
    r"mov(?:e|es|ed|ing)\s+through|navigat\w*|reach\w*|transport\w*|escort\w*|transfer\w*|"
    r"descend\w*|climb\w*|leave|leaves|leaving|left|exit\w*|"
    r"depart\w*|return\w*|travel\w*|walk\w*|run\w*|runs|ran|flee\w*|fled|escape\w*|emerge\w*|"
    r"evacuate\w*|withdraw\w*|withdrew|"
    r"tak(?:e|es|ing)\s+shelter|took\s+shelter|seek(?:s|ing)?\s+shelter|sought\s+shelter|"
    r"retreat\w*|duck(?:s|ed|ing)?\s+inside|shelter\w*\s+inside|"
    r"arriv\w*\s+(?:at|in|inside|to)|"
    r"(?:proceed\w*|venture\w*|crawl\w*|slip\w*|sneak\w*|stumbl\w*|rush\w*|dash\w*)\s+"
    r"(?:to|into|inside|through|down|up|toward|towards|past|across)|"
    r"head(?:s|ed|ing)?\s+(?:to|for|toward|towards|down|up|into|out)|"
    r"go(?:es|ing)?\s+(?:to|into|down|up|through)|went\s+(?:to|into|inside|down|up|through)|"
    r"(?:make|makes|made|making)\s+(?:her|his|their)\s+way)\b", re.I,
)
_CLAUSE_SPLIT_RE = re.compile(
    r"[.;!?]+|,|\b(?:while|as|but|and|then|before|after|when|although|though|whereas|yet|so)\b", re.I,
)
_SCENE_LOCAL_ACCESS_RE = re.compile(
    r"\b(?:enter\w*|go(?:es|ing)?\s+inside|step(?:s|ped|ping)?\s+inside|cross\w*|pass(?:es|ed|ing)?\s+through|"
    r"open\w*|unlock\w*|bypass\w*|breach\w*|descend\w*|climb\w*|"
    r"tak(?:e|es|ing)\s+shelter|took\s+shelter|seek(?:s|ing)?\s+shelter|sought\s+shelter|"
    r"retreat\w*|duck(?:s|ed|ing)?\s+inside|shelter\w*\s+inside)\b", re.I,
)


def _location_tokens(value: str) -> set[str]:
    return {
        token for token in content_tokens(value)
        if token not in _LOCATION_GENERIC_WORDS and token not in {"arctic", "storm-ravaged"}
    }


def _is_nested_station_location(old_location: str, new_location: str) -> bool:
    """Treat broad site labels (e.g. station interior) and their named rooms as one site."""
    old = re.sub(r"\W+", " ", str(old_location or "").casefold()).strip()
    new = re.sub(r"\W+", " ", str(new_location or "").casefold()).strip()
    if not old or not new:
        return False
    old_tokens = set(old.split())
    new_tokens = set(new.split())
    common_site = old_tokens & new_tokens
    if not common_site:
        return False
    if ("exterior" in old_tokens) != ("exterior" in new_tokens):
        return False
    # A broad "<site> interior" label may be refined to a room in that same site.
    if "interior" in old_tokens and ("room" in new_tokens or "control" in new_tokens):
        return True
    if "interior" in new_tokens and ("room" in old_tokens or "control" in old_tokens):
        return True
    return False


def _strip_cast_surfaces(text: str, cast: list[dict[str, Any]]) -> str:
    value = str(text or "")
    surfaces: list[str] = []
    for item in cast:
        surfaces.extend(_name_surfaces(item.get("name", ""), item.get("aliases", [])))
    for surface in sorted(set(surfaces), key=len, reverse=True):
        if surface:
            value = re.sub(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", " ", value, flags=re.I)
    return value


_CAUSE_SYNONYM_GROUPS = (
    {"global", "world", "worldwide", "planetwide"},
    {"response", "react", "reacts", "reacted", "reaction"},
    {"stay", "stays", "staying", "behind", "sacrifice", "sacrifices", "sacrificed"},
    {"disperse", "disperses", "dispersed", "spread", "spreads", "spreading"},
)
_CAUSE_SYNONYM_MAP = {token: f"synonym_{index}" for index, group in enumerate(_CAUSE_SYNONYM_GROUPS) for token in group}


def _cause_event_is_anchored(cause: str, event: str, cast: list[dict[str, Any]]) -> bool:
    """Compare consequential terms, not shared character names or grammatical wording."""
    cause_text = _strip_cast_surfaces(cause, cast)
    event_text = _strip_cast_surfaces(event, cast)
    cause_tokens = content_tokens(cause_text)
    event_tokens = content_tokens(event_text)
    cause_norm = {_CAUSE_SYNONYM_MAP.get(token, token) for token in cause_tokens}
    event_norm = {_CAUSE_SYNONYM_MAP.get(token, token) for token in event_tokens}
    return bool(cause_norm & event_norm)


_INVARIANT_PLURALS = {"species", "series", "news", "means", "crossroads", "headquarters", "analysis", "basis", "axis"}


def _plural_variants(tokens: set[str]) -> set[str]:
    """Every token plus its conservative singular form (levels->level, archives->archive, laboratories->laboratory).

    Invariant plurals (species, series) and words ending in ss/us/is (glass, process, status) are left alone.
    """
    variants: set[str] = set()
    for token in tokens:
        variants.add(token)
        if token in _INVARIANT_PLURALS or len(token) <= 3:
            continue
        if token.endswith(("ss", "us", "is")):
            continue
        if token.endswith("ies") and len(token) > 4:
            variants.add(token[:-3] + "y")
        elif token.endswith(("sses", "xes", "zes", "ches", "shes")):
            variants.add(token[:-2])
        elif token.endswith("s"):
            variants.add(token[:-1])
    return variants


def _scene_local_transition_is_explained(
    entry_location: str,
    exit_location: str,
    event: str,
    cast: list[dict[str, Any]],
) -> bool:
    old = re.sub(r"\W+", " ", str(entry_location or "").casefold()).strip()
    new = re.sub(r"\W+", " ", str(exit_location or "").casefold()).strip()
    if not old or not new or old == new:
        return True
    if _is_nested_station_location(old, new):
        return True
    # Title abbreviations ("Dr. Hale") must not be mistaken for sentence ends.
    event_text = re.sub(r"\b(Dr|Mr|Mrs|Ms|Prof|St|Col|Gen|Capt|Lt|Sgt|Cmdr)\.", r"\1", str(event or ""))
    destination_tokens = _plural_variants(_location_tokens(exit_location))
    target_is_interior = bool(re.search(r"\b(?:interior|inside|room|corridor|vault)\b", new, re.I))
    source_is_outside = bool(re.search(r"\b(?:tundra|exterior|outside|entrance)\b", old, re.I))

    def _has_named_mover(text: str) -> bool:
        # Movement by an object/weather system is not proof that the visible cast moved.
        return any(
            re.search(r"(?<!\w)" + re.escape(surface) + r"(?!\w)", text, re.I)
            for item in cast
            for surface in _name_surfaces(item.get("name", ""), item.get("aliases", []))
        )

    # A passage verb and the destination must appear in the SAME clause ("descends to the lower level"),
    # inside a sentence that names a canonical cast member. Opening a door, or a monitor displaying a room
    # elsewhere in the sentence, does not count as reaching the destination.
    cast_named_earlier = False
    for sentence in re.split(r"(?<=[.!?])\s+", event_text):
        named_here = _has_named_mover(sentence)
        # A pronoun sentence ("...forces her to take shelter inside") continues the cast member named earlier.
        pronoun_continues = cast_named_earlier and bool(
            re.search(r"\b(?:she|he|they|her|his|him|their|them)\b", sentence, re.I)
        )
        cast_named_earlier = cast_named_earlier or named_here
        if not (named_here or pronoun_continues):
            continue
        for clause in _CLAUSE_SPLIT_RE.split(sentence):
            if not clause or not _SCENE_LOCAL_MOVE_RE.search(clause):
                continue
            if destination_tokens and not (_plural_variants(content_tokens(clause)) & destination_tokens):
                continue
            if target_is_interior and source_is_outside and not _SCENE_LOCAL_ACCESS_RE.search(clause):
                continue
            return True
    return False


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
    """Parse boundary state plus full-scene physical visibility and causal hand-off.

    Mid-scene transitions are derived from presence deltas between ON_SCREEN_PRESENT
    and the ENTRY/EXIT boundary fields (entry = ON_SCREEN_PRESENT - ENTRY_PRESENT;
    exit = ON_SCREEN_PRESENT - EXIT_PRESENT). The Architect is not asked to emit a
    redundant second list, so a stale duplicate cannot drift out of sync with the
    ledger.
    """
    result: dict[str, str] = {}
    text = str(beat or "")
    pattern = re.compile(
        r"(?:^|;\s*)(PURPOSE|ENTRY_LOCATION|ENTRY_PRESENT|ON_SCREEN_PRESENT|ENTRY_KNOWS|ENTRY_OBJECTS|"
        r"EVENT|EXIT_LOCATION|EXIT_PRESENT|EXIT_KNOWS|EXIT_OBJECTS|NEXT_CAUSE)\s*=\s*", re.I
    )
    matches = list(pattern.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result[match.group(1).upper()] = text[match.end():end].strip(" ;|")
    return result


def is_non_person_cast_entry(name: str, metadata: str, fields: dict[str, str] | None = None) -> bool:
    """Recognize non-embodied opposition entries so forces/systems are never audited as people.

    Explicit ``kind`` metadata is authoritative. For older Architect output without kind,
    a narrowly-scoped compatibility rule excludes a generic, voiceless ``opposition`` entry
    only when the same named entity is independently declared in OPPOSITION. This avoids
    treating a storm, infrastructure failure, or other impersonal antagonist as a physical
    cast member while retaining ordinary named human antagonists with a real role.
    """
    metadata = str(metadata or "")
    kind_match = re.search(r"\b(?:kind|entity[_ ]type|category)\s*:\s*([^;,]+)", metadata, re.I)
    if kind_match:
        kind = re.sub(r"[\s_/-]+", " ", kind_match.group(1).strip().casefold())
        if kind in {"person", "human", "animal", "creature", "embodied agent", "sentient embodied character"}:
            return False
        # An explicit but unrecognized type is not silently promoted to a physical person.
        # The prompt's allowed values are closed; add a new embodied type deliberately.
        return True

    role = metadata.split(";", 1)[0].strip().casefold()
    known_non_person_role = re.compile(
        r"\b(?:non[- ]human|relentless force|environmental force|weather system|weather phenomenon|"
        r"natural phenomenon|natural disaster|environmental hazard|infrastructure|system|location|place|"
        r"building|object|device|machine|artifact|organization|organisation|company|corporation|abstract concept)\b",
        re.I,
    )
    if known_non_person_role.search(role):
        return True

    voice_match = re.search(r"\bvoice\s*:\s*([^;]+)", metadata, re.I)
    voice_is_none = bool(voice_match and voice_match.group(1).strip().casefold() in {"none", "n/a", "-", "silent/no voice"})
    generic_opposition_role = role in {"opposition", "opposing force", "environmental opposition", "natural force"}
    opposition_text = str((fields or {}).get("OPPOSITION", "") or "")
    name = re.sub(r"\s+", " ", str(name or "")).strip()
    name_is_grounded_as_opposition = bool(
        name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", opposition_text, re.I)
    )
    return generic_opposition_role and voice_is_none and name_is_grounded_as_opposition


def parse_cast(blueprint: str) -> list[dict[str, Any]]:
    """Parse physical character entries; exclude non-person opposition forces/systems."""
    fields, _, cast_line = _fields_and_beats(blueprint)
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
        if is_non_person_cast_entry(name, metadata, fields):
            continue
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
    """Normalize non-person opponents and reconcile physical presence with boundaries.

    ENTRY_PRESENT and EXIT_PRESENT are boundary snapshots. Every listed physical person
    must therefore occur in ON_SCREEN_PRESENT. A declared death is also a physical scene
    appearance and must be represented there. The model should author the complete ledger;
    this deterministic normalization repairs only these logically implied inclusions.
    """
    fields, beats, cast_line = _fields_and_beats(blueprint)
    if not cast_line or set(beats) != set(range(1, expected_scenes + 1)):
        return blueprint

    cast_entries: list[tuple[str, dict[str, Any]]] = []
    for raw in cast_line.partition(":")[2].split("|"):
        raw = raw.strip()
        parsed_match = re.match(r"^([^()]+?)\s*\((.*)\)\s*$", raw)
        if parsed_match and is_non_person_cast_entry(parsed_match.group(1).strip(), parsed_match.group(2), fields):
            continue
        parsed = parse_cast("CAST: " + raw)
        if len(parsed) == 1:
            cast_entries.append((raw, parsed[0]))

    updated_beats = dict(beats)
    states = {number: parse_scene_state(beats[number]) for number in beats}
    for number, state in states.items():
        original_on_screen = state.get("ON_SCREEN_PRESENT", "").strip()
        additions: list[str] = []
        for _raw, item in cast_entries:
            must_be_visible = (
                _name_is_present(state.get("ENTRY_PRESENT", ""), item)
                or _name_is_present(state.get("EXIT_PRESENT", ""), item)
            )
            fate_match = re.fullmatch(r"dies in scene\s+(\d+)", str(item.get("fate", "")).casefold())
            death_scene = int(fate_match.group(1)) if fate_match else None
            if death_scene == number:
                must_be_visible = True
            if must_be_visible and not _name_is_present(original_on_screen, item):
                additions.append(item["name"])
        if not additions:
            continue
        base = "" if original_on_screen.casefold() in {"none", "nobody", "no one", "empty", "-"} else original_on_screen
        additions = [name for name in dict.fromkeys(additions) if not _name_is_present(base, {"name": name, "aliases": []})]
        revised_on_screen = ", ".join(part for part in (base, ", ".join(additions)) if part)
        state["ON_SCREEN_PRESENT"] = revised_on_screen
        beat_text = beats[number]
        beat_text, count = re.subn(
            r"(\bON_SCREEN_PRESENT\s*=\s*)(.*?)(?=\s*;\s*(?:PURPOSE|ENTRY_KNOWS|ENTRY_OBJECTS|EVENT|EXIT_LOCATION|EXIT_PRESENT|EXIT_KNOWS|EXIT_OBJECTS|NEXT_CAUSE)\s*=|$)",
            lambda match: match.group(1) + revised_on_screen,
            beat_text,
            count=1,
            flags=re.IGNORECASE,
        )
        if count:
            updated_beats[number] = beat_text

    rebuilt: list[str] = []
    for raw, item in cast_entries:
        present = [
            number for number in sorted(states)
            if _name_is_present(states[number].get("ON_SCREEN_PRESENT", ""), item)
        ]
        fate_match = re.fullmatch(r"dies in scene\s+(\d+)", str(item.get("fate", "")).casefold())
        if fate_match:
            death_scene = int(fate_match.group(1))
            present = sorted({value for value in present if value <= death_scene} | {death_scene})
        # Preserve the original CAST scene list when the full-scene ledger has no
        # evidence; the validator will expose the unresolved mismatch rather than erase it.
        if present and present != item["scenes"]:
            raw = re.sub(
                r"on-screen scenes\s*:\s*[0-9, ]+",
                "on-screen scenes: " + ", ".join(f"{value:02d}" for value in present),
                raw,
                count=1,
                flags=re.IGNORECASE,
            )
        rebuilt.append(raw)

    revised = blueprint.replace(cast_line, "CAST: " + " | ".join(rebuilt), 1)
    for number, old_beat in beats.items():
        new_beat = updated_beats[number]
        if new_beat != old_beat:
            revised = revised.replace(old_beat, new_beat, 1)
    return revised


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
        # People are also referred to by surname ("Petrov") and by a quoted call sign ('K-9 Unit "Rook"' -> Rook).
        unquoted = re.sub(r"[\"“”‘’]", "", titleless).strip()
        tokens = unquoted.split()
        if len(tokens) >= 2 and len(tokens[-1]) >= 3 and re.search(r"[A-Za-z]", tokens[-1]):
            surfaces.append(tokens[-1])
        if unquoted != titleless:
            surfaces.append(unquoted)
        for quoted in re.findall(r"[\"“‘]([^\"”’]{2,30})[\"”’]", titleless):
            surfaces.append(quoted.strip())
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
        r"\b(?:Dr|Doctor|Mr|Mrs|Ms|Miss|Prof|Professor|Captain|Capt|Colonel|Col|Lieutenant|Lt|Major|Maj|Commander|Cmdr|Admiral|Adm|General|Gen|Sir|Lady|Sergeant|Sgt|Officer|Detective|Inspector|Reverend|Rev)\.?\s+(?=[A-Z])",
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


_PLOT_GENERIC_ANCHORS = {
    "scene", "plant", "payoff", "event", "story", "plot", "character", "person",
    "find", "finds", "found", "discover", "discovers", "discovered", "locate", "locates",
    "reveal", "reveals", "revealed", "show", "shows", "showed", "confirm", "confirms",
    "confirmed", "use", "uses", "used", "attempt", "attempts", "attempted", "make", "makes",
    "made", "must", "need", "needs", "needed", "because", "therefore", "warning", "warns",
    "warned", "link", "links", "linked", "death", "dies", "died", "true", "nature",
    "security", "protocol", "protocols", "contents", "access", "station", "vault", "room",
    "control", "archive", "archives", "laboratory", "laboratories", "lab", "arctic",
    "storm", "experiment", "instability", "unstable", "result", "results", "consequence",
}


def _distinctive_plot_tokens(value: str, cast: list[dict[str, Any]]) -> set[str]:
    """Return evidence/object/rule anchors, excluding cast names and generic story verbs/nouns."""
    stripped = _strip_cast_surfaces(str(value or ""), cast)
    return {
        token for token in _stemmed_content_tokens(stripped)
        if token not in _PLOT_GENERIC_ANCHORS and len(token) >= 4
    }


def _has_plot_anchor(
    field_value: str,
    prose: str,
    cast: list[dict[str, Any]] | int | None = None,
    minimum: int = 2,
) -> bool:
    # Backward compatibility: the prose-audit caller historically passed minimum as arg 3.
    if isinstance(cast, int):
        minimum, cast = cast, None
    cast = cast or []
    planned = _distinctive_plot_tokens(field_value, cast)
    actual = _distinctive_plot_tokens(prose, cast)
    return len(planned & actual) >= minimum


_PLOT_EVIDENCE_ACTION_RE = re.compile(
    r"\b(?:use\w*|apply\w*|activat\w*|open\w*|unlock\w*|break\w*|destroy\w*|"
    r"take\w*|hand\w*|give\w*|steal\w*|drop\w*|carry\w*|hide\w*|recover\w*|"
    r"retriev\w*|remove\w*|place\w*|find\w*|discover\w*|locat\w*|return\w*|"
    r"seize\w*|lock\w*|burn\w*|read\w*|decipher\w*|interpret\w*|confirm\w*|"
    r"prove\w*|reveal\w*|expos\w*|trigger\w*|explain\w*|identify\w*|recogniz\w*|"
    r"demonstrat\w*|establish\w*|connect\w*|link\w*|trace\w*)\b", re.I,
)


def _plot_event_anchor_is_grounded(
    field_value: str, event: str, cast: list[dict[str, Any]],
) -> bool:
    """Require the distinctive evidence/object and its actual use, discovery or interpretation in one EVENT sentence."""
    anchors = _distinctive_plot_tokens(field_value, cast)
    if not anchors:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+", str(event or "")):
        sentence_anchors = _distinctive_plot_tokens(sentence, cast)
        if anchors & sentence_anchors and (
            OBJECT_CHANGE_RE.search(sentence)
            or KNOWLEDGE_RE.search(sentence)
            or _PLOT_EVIDENCE_ACTION_RE.search(sentence)
        ):
            return True
    return False


_OBJECT_STATE_WORDS = {
    "held", "holds", "holding", "owned", "owner", "control", "controlled", "carried",
    "carry", "carrying", "found", "finds", "discovered", "discover", "retrieved", "recovered",
    "activated", "deactivated", "opened", "closed", "sealed", "unsealed", "broken", "damaged",
    "destroyed", "intact", "stabilized", "consumed", "dispersed", "released", "transferred",
    "hidden", "exposed", "inside", "outside", "under", "near", "remains", "remained", "still",
    "elena", "voss", "kael", "hale", "marcus", "dr", "by", "the", "with", "from",
}
_OBJECT_ENVIRONMENT_WORDS = {"storm", "weather", "wind", "snow", "ice", "darkness", "lightning", "thunder", "outside", "inside"}


def _inventory_items(value: str) -> list[str]:
    """Split the comma-delimited state ledger into individual physical-object entries."""
    text = str(value or "").strip()
    if text.casefold() in {"", "none", "n/a", "-", "no objects", "nobody"}:
        return []
    result = []
    for raw in re.split(r",\s*", text):
        item = raw.strip(" ;.")
        if not item or item.casefold() in {"none", "n/a", "-", "no objects"}:
            continue
        # Weather and broad environmental conditions are not physical inventory items.
        if content_tokens(item) and content_tokens(item).issubset(_OBJECT_ENVIRONMENT_WORDS):
            continue
        result.append(item)
    return result


def _object_identity_tokens(item: str) -> set[str]:
    return {
        token for token in _stemmed_content_tokens(item)
        if token not in _OBJECT_STATE_WORDS and token not in _OBJECT_ENVIRONMENT_WORDS and len(token) >= 3
    }


def _same_inventory_object(left: str, right: str) -> bool:
    a, b = _object_identity_tokens(left), _object_identity_tokens(right)
    if not a or not b:
        return False
    shared = a & b
    return bool(shared) and (len(shared) >= 2 or any(len(token) >= 5 for token in shared))


def _object_action_is_anchored(item: str, event: str) -> bool:
    """Require an object-specific action in the same sentence, not an unrelated verb elsewhere."""
    anchors = _object_identity_tokens(item)
    if not anchors:
        anchors = _stemmed_content_tokens(item) - _OBJECT_STATE_WORDS
    if not anchors:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+", str(event or "")):
        sentence_tokens = _stemmed_content_tokens(sentence)
        if anchors & sentence_tokens and OBJECT_CHANGE_RE.search(sentence):
            return True
    return False


def _object_state_signature(item: str) -> set[str]:
    return _stemmed_content_tokens(item) & _OBJECT_STATE_WORDS


_NONPORTABLE_KEY_OBJECTS = {
    "sealed underground vault", "sealed vault", "vault", "station", "research station",
    "lab", "laboratory", "archive", "station archives", "control room", "reactor chamber",
}


def _tracked_key_object_items(inventory: str, key_objects: str) -> list[str]:
    """Restrict continuity enforcement to props explicitly declared as story key objects.

    Scene ledgers may also mention weather, voices, debris, bodies, doors and fixed set pieces.
    Those are not automatically portable plot props and should not trigger a false object-custody failure.
    """
    declared = _inventory_items(key_objects)
    portable = [
        item for item in declared
        if re.sub(r"\s+", " ", item.casefold()).strip() not in _NONPORTABLE_KEY_OBJECTS
        and not _object_identity_tokens(item).issubset({"vault", "station", "lab", "laboratory", "archive", "room", "chamber"})
    ]
    return [
        item for item in _inventory_items(inventory)
        if any(_same_inventory_object(item, key_item) for key_item in portable)
    ]


def validate_blueprint_state_contract(blueprint: str, expected_scenes: int = 12) -> list[str]:
    """Hard-check scene-state continuity using presence deltas, without lexical transition vocabularies.

    Returns a single ordered list of findings. Structural continuity defects and
    unresolved character/object state contradictions are fatal; lexical wording preferences
    remain advisory and are not emitted by this state validator.
    """
    fields, beats, cast_line = _fields_and_beats(blueprint)
    required_fields = {
        "LOGLINE", "PROTAGONIST_GOAL", "OPPOSITION", "PRESSURE", "STORY_ENGINE", "REVERSAL",
        "CLIMAX_CHOICE", "RESOLUTION", "RELATIONSHIPS", "NAMED_ENTITIES", "KEY_OBJECTS",
        "SETUP_PAYOFF_MAP", "CONTINUITY_LEDGER",
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

    payoff_map = str(fields.get("SETUP_PAYOFF_MAP", "") or "").strip()
    payoff_pairs = re.findall(
        r"PLANT\s+Scene\s+(\d{1,2})\s*:\s*(.+?)\s*;\s*"
        r"PAYOFF\s+Scene\s+(\d{1,2})\s*:\s*(.+?)"
        r"(?=\s*\|\s*PLANT\s+Scene\s+|$)",
        payoff_map,
        flags=re.IGNORECASE,
    )
    if not payoff_pairs:
        errors.append("SETUP_PAYOFF_MAP must contain at least one explicit PLANT Scene NN / PAYOFF Scene NN pair")
    for plant_raw, plant_detail, payoff_raw, payoff_detail in payoff_pairs:
        plant_scene, payoff_scene = int(plant_raw), int(payoff_raw)
        if not plant_detail.strip() or not payoff_detail.strip():
            errors.append("SETUP_PAYOFF_MAP plant and payoff details must both be concrete")
        if not (1 <= plant_scene < payoff_scene <= expected_scenes):
            errors.append(
                f"SETUP_PAYOFF_MAP has invalid ordering/range: plant scene {plant_scene:02d}, payoff scene {payoff_scene:02d}"
            )
            continue
        plant_event = parse_scene_state(beats[plant_scene]).get("EVENT", "")
        payoff_event = parse_scene_state(beats[payoff_scene]).get("EVENT", "")
        # Character names, locations and generic verbs cannot prove that an object/evidence
        # plant or payoff actually occurs. Require a distinctive planted anchor in the named EVENT.
        if not _plot_event_anchor_is_grounded(plant_detail, plant_event, cast):
            errors.append(
                f"SETUP_PAYOFF_MAP plant in scene {plant_scene:02d} is not grounded in that scene's EVENT"
            )
        if not _plot_event_anchor_is_grounded(payoff_detail, payoff_event, cast):
            errors.append(
                f"SETUP_PAYOFF_MAP payoff in scene {payoff_scene:02d} is not grounded in that scene's EVENT"
            )
        if not _plot_event_anchor_is_grounded(plant_detail, payoff_event, cast):
            errors.append(
                f"SETUP_PAYOFF_MAP payoff scene {payoff_scene:02d} does not explicitly refer back to the planted evidence from scene {plant_scene:02d}"
            )

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
        if len(word_tokens(state["PURPOSE"])) < 5:
            errors.append(f"scene {number:02d} PURPOSE is too vague; provide a scene-specific dramatic change")
        if len(word_tokens(state["EVENT"])) < 6:
            errors.append(f"scene {number:02d} event is too thin to dramatize")
        normalized_cause = re.sub(r"[\W_]+", " ", state["NEXT_CAUSE"].casefold()).strip()
        if number < expected_scenes and normalized_cause in {
            "none", "n a", "na", "resolved", "complete", "no next cause", "no next scene", "-",
        }:
            errors.append(f"scene {number:02d} has no NEXT_CAUSE handoff")
        if number == expected_scenes and state["NEXT_CAUSE"].casefold() not in {"none", "resolved", "complete", "n/a"}:
            errors.append("scene 12 NEXT_CAUSE must be none/resolved; the final beat must not promise another episode")

        entry_location = state.get("ENTRY_LOCATION", "")
        exit_location = state.get("EXIT_LOCATION", "")
        if (
            entry_location
            and exit_location
            and re.sub(r"\W+", " ", entry_location.casefold()).strip()
            != re.sub(r"\W+", " ", exit_location.casefold()).strip()
            and not _scene_local_transition_is_explained(
                entry_location, exit_location, state.get("EVENT", ""), cast,
            )
        ):
            errors.append(
                f"scene {number:02d} changes its physical location without an explicit transition "
                f"({entry_location!r} to {exit_location!r})"
            )
        for key in ("ENTRY_PRESENT", "ON_SCREEN_PRESENT", "EXIT_PRESENT"):
            raw = state[key].casefold()
            if raw.strip() not in {"none", "nobody", "no one", "empty"}:
                unknown = _unknown_named_surfaces(state[key], cast)
                if unknown:
                    errors.append(f"scene {number:02d} {key} contains unrostered identity: {unknown[0]}")
        for item in cast:
            on_screen = _name_is_present(state["ON_SCREEN_PRESENT"], item)
            at_entry = _name_is_present(state["ENTRY_PRESENT"], item)
            at_exit = _name_is_present(state["EXIT_PRESENT"], item)
            if (at_entry or at_exit) and not on_screen:
                errors.append(
                    f"scene {number:02d} boundary presence for {item['name']!r} is missing from ON_SCREEN_PRESENT"
                )
            if number in item["scenes"] and not on_screen:
                errors.append(f"scene {number:02d} CAST appearance for {item['name']!r} is absent from ON_SCREEN_PRESENT")
            if on_screen and number not in item["scenes"]:
                errors.append(
                    f"scene {number:02d} ON_SCREEN_PRESENT includes {item['name']!r}, but the CAST on-screen scene list omits it"
                )
            # Mid-scene transitions are derived, not requested as duplicate model-authored fields:
            #   entry = ON_SCREEN_PRESENT - ENTRY_PRESENT
            #   exit  = ON_SCREEN_PRESENT - EXIT_PRESENT
            # This makes the ledger deterministic and prevents a second redundant list from drifting.
            event_mentions_person = _name_is_present(state["EVENT"], item)
            if on_screen and not at_entry and not event_mentions_person:
                errors.append(
                    f"scene {number:02d} mid-scene entry for {item['name']!r} has no character-specific EVENT anchor"
                )
            if on_screen and not at_exit and not event_mentions_person:
                errors.append(
                    f"scene {number:02d} mid-scene exit for {item['name']!r} has no character-specific EVENT anchor"
                )

            # CAST fate is the single authoritative death declaration. Do not infer that
            # a named character died merely because a death verb occurs somewhere in an
            # EVENT: that character may be the killer, witness, or person discussing another death.
            death_match = re.fullmatch(r"dies in scene\s+(\d+)", item["fate"].casefold())
            if death_match and int(death_match.group(1)) == number:
                if at_exit:
                    errors.append(
                        f"scene {number:02d} CAST fate says {item['name']!r} dies here but EXIT_PRESENT still includes them"
                    )
                if not event_mentions_person:
                    errors.append(
                        f"scene {number:02d} death of {item['name']!r} is not anchored in EVENT"
                    )

            # Soft warning only: a character whose CAST fate is survives/alive should
            # not have a death cue co-occurring with their name in EVENT. This string
            # contains no HARD_FINDING_MARKERS substring, so split_contract_findings()
            # routes it to warnings, not fatals.
            if item["fate"].casefold() in {"survives", "alive"}:
                if _event_explicitly_assigns_death(state["EVENT"], item):
                    errors.append(
                        f"scene {number:02d} potential fate contradiction: CAST says {item['name']!r} survives "
                        f"but EVENT explicitly assigns a death to them (review)"
                    )
                # "Sacrifices herself/himself" is materially ambiguous when the CAST
                # asserts survival and the scene removes the character from the exit
                # boundary. Require a character-specific survival/rescue statement.
                sacrifice = re.search(
                    r"\bsacrific(?:e|es|ed|ing)\s+(?:herself|himself|themselves|their\s+life|"
                    r"her\s+life|his\s+life)\b",
                    state["EVENT"], re.I,
                )
                if sacrifice and not at_exit:
                    explicitly_survives = any(
                        re.search(
                            r"(?<!\w)" + re.escape(surface) + r"(?!\w)[^.;!?]{0,55}\b"
                            r"(?:survives?|survived|is rescued|gets rescued|is recovered|gets recovered)\b",
                            state["EVENT"], re.I,
                        )
                        for surface in _name_surfaces(item["name"], item.get("aliases", []))
                    )
                    if not explicitly_survives:
                        errors.append(
                            f"scene {number:02d} survivor fate is unresolved for {item['name']!r}: "
                            "the character self-sacrifices and exits the scene without an explicit survival outcome"
                        )

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
        if not _cause_event_is_anchored(cause, following.get("EVENT", ""), cast):
            errors.append(f"scene {number + 1:02d} event is not anchored to the previous scene's NEXT_CAUSE")

    # Scene-to-scene continuity for physical location, cast presence, knowledge and object state.
    for number in range(1, expected_scenes):
        prior = parse_scene_state(beats[number])
        current = parse_scene_state(beats[number + 1])
        old_location = re.sub(r"\W+", " ", prior.get("EXIT_LOCATION", "").casefold()).strip()
        new_location = re.sub(r"\W+", " ", current.get("ENTRY_LOCATION", "").casefold()).strip()
        bridge = current.get("EVENT", "") + " " + prior.get("NEXT_CAUSE", "")
        old_normalized = re.sub(r"\W+", " ", old_location).strip()
        new_normalized = re.sub(r"\W+", " ", new_location).strip()
        nested_same_site = _is_nested_station_location(old_location, new_location)
        location_terms = _location_tokens(new_location)
        bridge_tokens = content_tokens(bridge)
        transition_named = bool(location_terms & bridge_tokens)
        if (
            old_location
            and new_location
            and old_normalized != new_normalized
            and not nested_same_site
            and (not TRAVEL_RE.search(bridge) or not transition_named)
        ):
            errors.append(f"scene {number + 1:02d} changes location without a route/time transition")

        old_present = _present_names(prior.get("EXIT_PRESENT", ""), cast)
        new_present = _present_names(current.get("ENTRY_PRESENT", ""), cast)
        arriving = new_present - old_present
        for canonical_name in sorted(arriving):
            person = next((item for item in cast if item["name"].casefold() == canonical_name), None)
            if person is not None and not _name_is_present(current.get("EVENT", ""), person):
                errors.append(
                    f"scene {number + 1:02d} introduces {person['name']!r} without a character-specific arrival EVENT anchor"
                )
        departing = old_present - new_present
        for canonical_name in sorted(departing):
            person = next((item for item in cast if item["name"].casefold() == canonical_name), None)
            if person is not None and not _name_is_present(bridge, person):
                errors.append(
                    f"scene {number:02d} removes {person['name']!r} without a character-specific departure EVENT anchor"
                )

        before_knowledge = content_tokens(prior.get("EXIT_KNOWS", ""))
        after_knowledge = content_tokens(current.get("ENTRY_KNOWS", ""))
        gained = after_knowledge - before_knowledge
        lost = before_knowledge - after_knowledge
        # ENTRY_KNOWS is knowledge already held at the scene boundary. A fact added
        # there must have been discovered/disclosed by the preceding scene's EVENT,
        # not by the current scene's later action.
        if len(gained) >= 2 and not KNOWLEDGE_RE.search(prior.get("EVENT", "")):
            errors.append(f"scene {number + 1:02d} changes established knowledge without a discovery/disclosure event")
        if len(lost) >= 2 and not re.search(
            r"\b(forget\w*|misremember\w*|doubt\w*|retract\w*|disprove\w*|correct\w*|mislead\w*|"
            r"conceal\w*|withhold\w*|hide\w*|revise\w*)\b", prior.get("EVENT", ""), re.I,
        ):
            errors.append(f"scene {number + 1:02d} removes established knowledge without a reason")

        prior_exit_text = str(prior.get("EXIT_OBJECTS", "") or "").strip()
        current_entry_text = str(current.get("ENTRY_OBJECTS", "") or "").strip()
        key_objects = fields.get("KEY_OBJECTS", "")
        prior_items = _tracked_key_object_items(prior_exit_text, key_objects)
        entry_items = _tracked_key_object_items(current_entry_text, key_objects)
        unmatched_entry = list(entry_items)
        for prior_item in prior_items:
            match_index = next((i for i, item in enumerate(unmatched_entry)
                                if _same_inventory_object(prior_item, item)), None)
            if match_index is None:
                errors.append(
                    f"scene {number + 1:02d} object inventory drops key object '{prior_item}' from prior EXIT_OBJECTS; "
                    "carry its identity and terminal state forward or show the explicit physical disposition"
                )
            else:
                unmatched_entry.pop(match_index)
        if unmatched_entry:
            errors.append(
                f"scene {number + 1:02d} object inventory introduces key object '{unmatched_entry[0]}' in ENTRY_OBJECTS "
                "without the object being present in the previous EXIT_OBJECTS"
            )

        # Validate only declared plot-critical props. Fixed set pieces, atmosphere, voices and debris
        # may appear in the descriptive inventory without becoming portable custody-tracked objects.
        entry_items = _tracked_key_object_items(current.get("ENTRY_OBJECTS", ""), key_objects)
        exit_items = _tracked_key_object_items(current.get("EXIT_OBJECTS", ""), key_objects)
        remaining_exit = list(exit_items)
        changed_items: list[str] = []
        for entry_item in entry_items:
            match_index = next((i for i, item in enumerate(remaining_exit)
                                if _same_inventory_object(entry_item, item)), None)
            if match_index is None:
                changed_items.append(entry_item)  # declared key object disappears from the scene ledger
            else:
                exit_item = remaining_exit.pop(match_index)
                if _object_state_signature(entry_item) != _object_state_signature(exit_item):
                    changed_items.append(exit_item)
        changed_items.extend(remaining_exit)  # declared key objects newly appear in the scene ledger
        for changed_item in changed_items:
            if not _object_action_is_anchored(changed_item, current.get("EVENT", "")):
                errors.append(
                    f"scene {number + 1:02d} key-object state change is not anchored to the named object and a physical action "
                    f"('{changed_item}')"
                )

    # KEY_OBJECTS may declare when a consequential state begins (e.g. "damaged in
    # Scene 05"). Reject an earlier scene that already depicts that same state.
    key_objects = fields.get("KEY_OBJECTS", "")
    for match in re.finditer(
        r"(?P<label>[A-Za-z][^()|;]{2,48})\([^)]*?\bstate\s*:\s*"
        r"(?P<state>damaged|broken|activated|dispersed|destroyed|sealed|unsealed)\b"
        r"[^)]*?\bscene\s*(?P<scene>\d{1,2})\b[^)]*\)",
        key_objects, re.I,
    ):
        label_tokens = content_tokens(match.group("label"))
        state_word = match.group("state").casefold()
        declared_scene = int(match.group("scene"))
        # The final specific noun is the most reliable anchor for the object name.
        anchors = {token for token in label_tokens if len(token) >= 4}
        for earlier_scene in range(1, min(declared_scene, expected_scenes + 1)):
            event = parse_scene_state(beats[earlier_scene]).get("EVENT", "")
            event_tokens = content_tokens(event)
            state_patterns = {
                "damaged": r"damag\w*", "broken": r"brok\w*", "activated": r"activat\w*",
                "dispersed": r"dispers\w*", "destroyed": r"destroy\w*", "sealed": r"seal\w*",
                "unsealed": r"unseal\w*",
            }
            if anchors & event_tokens and re.search(state_patterns.get(state_word, re.escape(state_word) + r"\w*"), event, re.I):
                errors.append(
                    f"declared object state timing conflicts: {match.group('label').strip()} is already "
                    f"{state_word} in scene {earlier_scene:02d}, before its declared scene {declared_scene:02d}"
                )
                break

    # Every declared key object must be represented in a scene inventory, not only in metadata.
    for segment in re.split(r"\s*;\s*(?=[A-Za-z][^;()]{2,48}\()", key_objects):
        object_match = re.match(
            r"\s*(?P<label>[A-Za-z][^()|;]{2,48})\([^)]*?\bscene\s*(?P<scene>\d{1,2})\b[^)]*\)",
            segment, re.I,
        )
        if not object_match:
            continue
        label_tokens = {token for token in content_tokens(object_match.group("label")) if len(token) >= 4}
        scene_number = int(object_match.group("scene"))
        if not label_tokens or not 1 <= scene_number <= expected_scenes:
            continue
        scene_state = parse_scene_state(beats[scene_number])
        inventory_tokens = content_tokens(
            scene_state.get("ENTRY_OBJECTS", "") + " " + scene_state.get("EXIT_OBJECTS", "")
        )
        if not (label_tokens & inventory_tokens):
            errors.append(
                f"scene {scene_number:02d} key object is missing from scene inventory: "
                f"{object_match.group('label').strip()}"
            )

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


def _is_isolated_sentence_opener(story: str, match: re.Match, candidate: str) -> bool:
    """One capitalized word at a sentence start, used once in the whole story, with no speech verb."""
    if len(candidate.split()) != 1:
        return False
    before = story[:match.start()]
    if before.strip() and not re.search(r"(?:[.!?]['\"”’)]*|\n)\s*['\"“‘(]*$", before):
        return False
    if re.search(rf"\b(?:{SPEECH_VERBS})\b", match.group(0), re.I):
        return False
    return len(re.findall(r"(?<![A-Za-z])" + re.escape(candidate) + r"(?![A-Za-z])", story)) < 2


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

    # Named programs, facilities and institutions must originate in the source or the Architect plan. "The plan"
    # is the whole plan: a prop the Architect declared in KEY_OBJECTS, or named in a beat or the setup/payoff
    # map, is seeded even though it is not listed under NAMED_ENTITIES.
    plan_text = " ".join([allowed_text, *(str(v) for v in fields.values()), *(str(v) for v in beats.values())]).casefold()
    for regex in (PROJECT_ENTITY_RE, NAMED_FACILITY_RE):
        for match in regex.finditer(str(story or "")):
            entity = re.sub(r"\s+", " ", match.group(0)).strip()
            if entity.casefold() not in plan_text:
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
    # A person the Architect named but never put on screen is seeded: the story may refer to them.
    mentioned_only = re.search(r"Named but never on screen:\s*(.+)$", fields.get("RELATIONSHIPS", ""), re.S)
    for mentioned_name in (re.split(r"\s*,\s*", mentioned_only.group(1)) if mentioned_only else []):
        mentioned_name = mentioned_name.strip().rstrip(".")
        if mentioned_name:
            allowed_names.add(mentioned_name.casefold())
            allowed_surfaces.update(s.casefold() for s in _name_surfaces(mentioned_name, []))
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
        if _is_isolated_sentence_opener(str(story or ""), match, candidate):
            # "Permafrost cracked under the rails." A single capitalized word that opens one sentence, never speaks and is
            # never repeated cannot be told apart from an ordinary noun without a dictionary: report it, do not discard the draft.
            errors.append(f"possible unrostered name opening a sentence (advisory): {candidate}")
            continue
        candidate_key = candidate.casefold()
        role_words = {
            "clerk", "guard", "officer", "captain", "driver", "pilot", "doctor", "nurse", "engineer",
            "technician", "archivist", "surveyor", "operator", "worker", "soldier", "detective", "teacher",
            "porter", "servant", "attendant", "stranger", "witness", "sailor", "mechanic", "assistant",
        }
        if len(candidate.split()) > 1 and set(candidate_key.split()) & role_words:
            continue
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
            # Reuse the shared movement/transition vocabulary; the former small bespoke
            # list missed ordinary prose such as "Kael disappeared down the corridor".
            if departed and not TRAVEL_RE.search(prior_paragraph):
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

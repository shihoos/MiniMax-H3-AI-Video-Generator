"""Schema-constrained Architect output.

The Architect used to emit a free-text blueprint that regexes had to re-parse, so every formatting
variation by the model (``wants:`` vs ``wants [..]`` ...) became a hard failure. The model now returns
JSON that vLLM enforces with a JSON schema (the same mechanism the character and shot passes already use),
and :func:`render_blueprint_text` serialises it back to the existing text blueprint so every downstream
contract, validator and Author prompt keeps consuming exactly the format it always did.

Nothing here judges story quality. Presence lists drive a cast member's on-screen scenes mechanically.
"""
from __future__ import annotations

import json
import re

PLOT_TEXT_FIELDS = (
    "logline", "protagonist_goal", "opposition", "pressure", "story_engine", "reversal",
    "climax_choice", "resolution", "relationships", "named_entities", "key_objects",
)
SCENE_TEXT_FIELDS = (
    "purpose", "entry_location", "entry_knows", "entry_objects", "event",
    "exit_location", "exit_knows", "exit_objects", "next_cause",
)
SCENE_LIST_FIELDS = ("entry_present", "on_screen_present", "exit_present")
SCENE_ORDER = (
    "purpose", "entry_location", "entry_present", "on_screen_present", "entry_knows", "entry_objects",
    "event", "exit_location", "exit_present", "exit_knows", "exit_objects", "next_cause",
)
# What the model writes per scene. Entry state is never authored after scene 1: every scene starts exactly
# where the previous one ended, so the boundary continuity the checkers verify holds by construction.
AUTHORED_SCENE_FIELDS = (
    "purpose", "on_screen_present", "event", "exit_location", "exit_present", "exit_knows", "exit_objects", "next_cause",
)
OPENING_FIELDS = ("location", "present", "knows", "objects")
_ENTRY_FROM_EXIT = {
    "entry_location": "exit_location", "entry_present": "exit_present",
    "entry_knows": "exit_knows", "entry_objects": "exit_objects",
}
KINDS = ("person", "animal", "embodied agent")


def blueprint_json_schema(scene_count: int) -> dict:
    def text(max_length: int | None = None) -> dict:
        value = {"type": "string", "minLength": 1}
        if max_length is not None:
            value["maxLength"] = int(max_length)
        return value

    names = {"type": "array", "items": text(80)}
    cast_item = {
        "type": "object",
        "properties": {
            "name": text(80),
            "kind": {"type": "string", "enum": list(KINDS)},
            "role": text(100),
            "wants": text(200),
            "pressure": text(200),
            "voice": text(100),
            "aliases": names,
            "fate": {"type": "string", "enum": ["survives", "dies"]},
            "death_scene": {"type": "integer", "minimum": 0, "maximum": scene_count},
        },
        "required": ["name", "kind", "role", "wants", "pressure", "voice", "aliases", "fate", "death_scene"],
        "additionalProperties": False,
    }
    scene_text_limits = {
        "purpose": 180,
        "event": 520,
        "exit_location": 120,
        "exit_knows": 600,
        "exit_objects": 480,
        "next_cause": 280,
    }
    scene_props = {
        key: (names if key in SCENE_LIST_FIELDS else text(scene_text_limits.get(key, 320)))
        for key in AUTHORED_SCENE_FIELDS
    }
    scene_item = {
        "type": "object",
        "properties": scene_props,
        "required": list(AUTHORED_SCENE_FIELDS),
        "additionalProperties": False,
    }
    opening = {
        "type": "object",
        "properties": {"location": text(120), "present": names, "knows": text(600), "objects": text(480)},
        "required": list(OPENING_FIELDS),
        "additionalProperties": False,
    }
    payoff_item = {
        "type": "object",
        "properties": {
            "plant_scene": {"type": "integer", "minimum": 1, "maximum": scene_count},
            "plant": text(260),
            "payoff_scene": {"type": "integer", "minimum": 1, "maximum": scene_count},
            "payoff": text(260),
        },
        "required": ["plant_scene", "plant", "payoff_scene", "payoff"],
        "additionalProperties": False,
    }
    properties = {"cast": {"type": "array", "minItems": 1, "items": cast_item}}
    plot_text_limits = {
        "logline": 320, "protagonist_goal": 240, "opposition": 420, "pressure": 360,
        "story_engine": 420, "reversal": 360, "climax_choice": 480, "resolution": 360,
        "relationships": 300, "named_entities": 300, "key_objects": 360,
    }
    for key in PLOT_TEXT_FIELDS:
        properties[key] = text(plot_text_limits.get(key, 420))
    properties["setup_payoff_map"] = {"type": "array", "minItems": 1, "items": payoff_item}
    properties["opening"] = opening
    properties["scenes"] = {"type": "array", "minItems": scene_count, "maxItems": scene_count, "items": scene_item}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def blueprint_response_format(scene_count: int) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {"name": "story_blueprint", "strict": True, "schema": blueprint_json_schema(scene_count)},
    }


def parse_blueprint_json(raw: str) -> dict | None:
    """Return the decoded plan, or None when ``raw`` is not a JSON object (e.g. a legacy text plan)."""
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RuntimeError(f"Story Architect returned malformed JSON ({exc}).") from exc
    return data if isinstance(data, dict) else None


def _clean(value: object) -> str:
    # ';' separates scene state fields and '|' separates CAST entries in the text blueprint.
    text = re.sub(r"\s+", " ", str(value if value is not None else "")).strip()
    return text.replace(";", ",").replace("|", "/").strip()


def _join(values: object) -> str:
    items = [_clean(v) for v in (values if isinstance(values, list) else [values]) if _clean(v)]
    return ", ".join(items) if items else "none"


def derive_on_screen_scenes(
    name: str, aliases: list[str], scenes: list[dict], shared_parts: set[str] | None = None,
) -> list[int]:
    """Scenes whose ON_SCREEN_PRESENT list names this person (full name, alias, or an unshared name part)."""
    shared_parts = shared_parts or set()
    parts = [t for t in re.split(r"\s+", name) if len(t) > 2 and t.casefold() not in shared_parts]
    needles = [x.casefold() for x in [name] + [a for a in aliases if a] + parts]
    found: list[int] = []
    for number, scene in enumerate(scenes, 1):
        present = scene.get("on_screen_present") or []
        present = present if isinstance(present, list) else [present]
        haystack = [re.sub(r"\s+", " ", str(p)).strip().casefold() for p in present]
        if any(h == n or re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", h) for h in haystack for n in needles):
            found.append(number)
    return found


def _as_list(value: object) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [re.sub(r"\s+", " ", str(v)).strip() for v in values if str(v or "").strip()]


def resolve_scenes(data: dict) -> list[dict]:
    """Complete scene states from the authored fields.

    Scene 1 starts in ``opening``; every later scene starts exactly where the previous one ended, and anyone
    present at either boundary is on screen by definition. Plans that still carry explicit entry_* fields
    (older traces) keep them.
    """
    opening = data.get("opening") if isinstance(data.get("opening"), dict) else {}
    resolved: list[dict] = []
    for index, authored in enumerate(data.get("scenes") or []):
        scene = dict(authored) if isinstance(authored, dict) else {}
        for entry_key, exit_key in _ENTRY_FROM_EXIT.items():
            if scene.get(entry_key):
                continue
            if index == 0:
                scene[entry_key] = opening.get(entry_key[len("entry_"):], "")
            else:
                scene[entry_key] = resolved[-1].get(exit_key, "")
        union: list[str] = []
        for name in _as_list(scene.get("entry_present")) + _as_list(scene.get("on_screen_present")) + _as_list(scene.get("exit_present")):
            if name.casefold() not in {u.casefold() for u in union} and name.casefold() not in {"none", "nobody", "no one"}:
                union.append(name)
        scene["on_screen_present"] = union
        resolved.append(scene)
    return resolved


def derive_continuity_ledger(data: dict, scenes: list[dict] | None = None) -> str:
    """Derive the redundant global continuity summary from canonical scene state.

    The model must not spend output tokens inventing a second continuity ledger. The opening,
    CAST, KEY_OBJECTS and per-scene snapshots are authoritative; this compact summary is rendered
    deterministically after JSON decoding and always stays within the prompt's 35-70 word contract.
    """
    opening = data.get("opening") if isinstance(data.get("opening"), dict) else {}
    resolved = scenes if scenes is not None else resolve_scenes(data)
    first = resolved[0] if resolved else {}
    last = resolved[-1] if resolved else {}

    def clean(value: object, fallback: str) -> str:
        result = _clean(value)
        return result if result and result.casefold() not in {"none", "n/a", "-"} else fallback

    def clip_words(value: object, maximum: int, fallback: str) -> str:
        text = clean(value, fallback)
        matches = list(re.finditer(r"\b[\w'’-]+\b", text))
        if len(matches) <= maximum:
            return text
        return text[:matches[maximum - 1].end()].rstrip(" ,;:.") + "…"

    opening_present = _as_list(opening.get("present"))
    if not opening_present and first:
        opening_present = _as_list(first.get("on_screen_present"))
    lead = clip_words(opening_present[0] if opening_present else "The protagonist", 2, "The protagonist")
    start_location = clip_words(
        opening.get("location") or first.get("entry_location"), 4, "the opening location",
    )
    end_location = clip_words(last.get("exit_location"), 4, "the final location")
    def first_clause(value: object, fallback: str, maximum: int) -> str:
        text = clean(value, fallback)
        # Keep the first complete proposition rather than truncating in the middle of a clause.
        text = re.split(r"[;.]|,\s+(?:but|yet|while|whereas)\b", text, maxsplit=1, flags=re.I)[0].strip()
        return clip_words(text, maximum, fallback)

    opening_knows = first_clause(opening.get("knows"), "no additional facts at the opening", 7)
    final_knows = first_clause(last.get("exit_knows"), "no additional facts at the ending", 7)
    key_objects = clip_words(data.get("key_objects"), 18, "none declared")

    # Record the protagonist's declared fate without trying to infer whether a death happened.
    # Detailed CAST and per-scene state remain authoritative for every individual character.
    lead_fate = "survives"
    for member in data.get("cast") or []:
        if not isinstance(member, dict):
            continue
        if str(member.get("name", "")).casefold() == str(opening_present[0] if opening_present else "").casefold():
            lead_fate = "dies" if str(member.get("fate", "survives")).casefold() == "dies" else "survives"
            break

    summary = (
        f"{lead} begins at {start_location} and ends at {end_location}. "
        f"Opening knowledge: {opening_knows}. Ending knowledge: {final_knows}. "
        f"Key objects: {key_objects}. {lead} {lead_fate}. "
        "Scene snapshots govern object ownership, location, condition, injuries and presence; CAST governs character fate, "
        "and numbered scenes preserve chronology."
    )
    summary = re.sub(r"\s+", " ", summary).strip()
    # Defensive trim for unusually long names/locations while retaining all authoritative detail elsewhere.
    words = list(re.finditer(r"\b[\w'’-]+\b", summary))
    if len(words) > 70:
        summary = summary[:words[69].end()].rstrip(" ,;:.") + "."
    if len(re.findall(r"\b[\w'’-]+\b", summary)) < 35:
        summary = (
            f"{summary} Each scene records the inherited starting state and completed ending state, including who is present, "
            "what is known, and where every portable plot-critical object remains."
        )
    return summary


def render_blueprint_text(data: dict, scene_count: int, dropped: list[str] | None = None) -> str:
    """Serialise the validated JSON plan into the text blueprint the rest of the pipeline consumes.

    Cast means beings that are physically on screen. A listed "cast member" that no scene's
    on_screen_present names (a storm, an institution, a person only mentioned) is not cast; it is left out
    and its name appended to ``dropped`` so the caller can report it. Opposition already carries such forces.
    """
    scenes = resolve_scenes(data)
    if len(scenes) != scene_count:
        raise RuntimeError(
            f"Story Architect returned {len(scenes)} scenes; exactly {scene_count} are required."
        )
    entries: list[str] = []
    gone: list[str] = []
    cast = [m for m in (data.get("cast") or []) if isinstance(m, dict)]
    part_owners: dict[str, int] = {}
    for member in cast:
        for token in {t.casefold() for t in re.split(r"\s+", _clean(member.get("name"))) if len(t) > 2}:
            part_owners[token] = part_owners.get(token, 0) + 1
    shared = {token for token, count in part_owners.items() if count > 1}
    for member in cast:
        name = re.sub(r"[()]", "", _clean(member.get("name")))
        aliases = [_clean(a) for a in (member.get("aliases") or []) if _clean(a)]
        scene_numbers = derive_on_screen_scenes(name, aliases, scenes, shared)
        if not scene_numbers:
            gone.append(name)
            if dropped is not None:
                dropped.append(name)
            continue
        fate = "survives"
        if str(member.get("fate", "")).casefold() == "dies":
            try:
                death = int(member.get("death_scene") or 0)
            except (TypeError, ValueError):
                death = 0
            death = death if 1 <= death <= scene_count else (max(scene_numbers) if scene_numbers else scene_count)
            fate = f"dies in scene {death:02d}"
        kind = _clean(member.get("kind")).casefold()
        entries.append(
            f"{name} (kind: {kind}; role: {_clean(member.get('role'))}; wants: {_clean(member.get('wants'))}; "
            f"pressure: {_clean(member.get('pressure'))}; voice: {_clean(member.get('voice'))}; "
            f"aliases: {', '.join(aliases) if aliases else 'none'}; "
            f"on-screen scenes: {', '.join(f'{v:02d}' for v in scene_numbers)}; fate: {fate})"
        )
    lines = ["CAST: " + " | ".join(entries)]
    if gone:
        # The Architect named these people, so the story may refer to them even though no scene shows them.
        relationships = _clean(data.get("relationships"))
        prefix = "" if not relationships or relationships.casefold() == "none" else relationships + ". "
        data = dict(data, relationships=prefix + "Named but never on screen: " + ", ".join(gone))

    def field(label: str, key: str) -> str:
        return f"{label}: {_clean(data.get(key)) or 'none'}"

    for label, key in (
        ("LOGLINE", "logline"), ("PROTAGONIST_GOAL", "protagonist_goal"), ("OPPOSITION", "opposition"),
        ("PRESSURE", "pressure"), ("STORY_ENGINE", "story_engine"), ("REVERSAL", "reversal"),
        ("CLIMAX_CHOICE", "climax_choice"), ("RESOLUTION", "resolution"),
        ("RELATIONSHIPS", "relationships"), ("NAMED_ENTITIES", "named_entities"), ("KEY_OBJECTS", "key_objects"),
    ):
        lines.append(field(label, key))
    pairs = []
    for item in data.get("setup_payoff_map") or []:
        pairs.append(
            f"PLANT Scene {int(item.get('plant_scene', 0)):02d}: {_clean(item.get('plant'))}; "
            f"PAYOFF Scene {int(item.get('payoff_scene', 0)):02d}: {_clean(item.get('payoff'))}"
        )
    lines.append("SETUP_PAYOFF_MAP: " + " | ".join(pairs))
    ledger = derive_continuity_ledger(data, scenes)
    lines.append("CONTINUITY_LEDGER: " + ledger)
    for number, scene in enumerate(scenes, 1):
        parts = []
        for key in SCENE_ORDER:
            value = _join(scene.get(key)) if key in SCENE_LIST_FIELDS else (_clean(scene.get(key)) or "none")
            parts.append(f"{key.upper()}={value}")
        lines.append(f"SCENE {number:02d}: " + "; ".join(parts))
    return "\n".join(lines)

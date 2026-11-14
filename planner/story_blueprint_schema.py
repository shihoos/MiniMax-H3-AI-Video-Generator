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
    text = {"type": "string", "minLength": 1}
    names = {"type": "array", "items": text}
    cast_item = {
        "type": "object",
        "properties": {
            "name": text,
            "kind": {"type": "string", "enum": list(KINDS)},
            "role": text,
            "wants": text,
            "pressure": text,
            "voice": text,
            "aliases": names,
            "fate": {"type": "string", "enum": ["survives", "dies"]},
            "death_scene": {"type": "integer", "minimum": 0, "maximum": scene_count},
        },
        "required": ["name", "kind", "role", "wants", "pressure", "voice", "aliases", "fate", "death_scene"],
        "additionalProperties": False,
    }
    scene_props = {k: (names if k in SCENE_LIST_FIELDS else text) for k in AUTHORED_SCENE_FIELDS}
    scene_item = {
        "type": "object",
        "properties": scene_props,
        "required": list(AUTHORED_SCENE_FIELDS),
        "additionalProperties": False,
    }
    opening = {
        "type": "object",
        "properties": {"location": text, "present": names, "knows": text, "objects": text},
        "required": list(OPENING_FIELDS),
        "additionalProperties": False,
    }
    payoff_item = {
        "type": "object",
        "properties": {
            "plant_scene": {"type": "integer", "minimum": 1, "maximum": scene_count},
            "plant": text,
            "payoff_scene": {"type": "integer", "minimum": 1, "maximum": scene_count},
            "payoff": text,
        },
        "required": ["plant_scene", "plant", "payoff_scene", "payoff"],
        "additionalProperties": False,
    }
    properties = {"cast": {"type": "array", "minItems": 1, "items": cast_item}}
    for key in PLOT_TEXT_FIELDS:
        properties[key] = text
    properties["setup_payoff_map"] = {"type": "array", "minItems": 1, "items": payoff_item}
    properties["continuity_ledger"] = text
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
    lines.append(field("CONTINUITY_LEDGER", "continuity_ledger"))
    for number, scene in enumerate(scenes, 1):
        parts = []
        for key in SCENE_ORDER:
            value = _join(scene.get(key)) if key in SCENE_LIST_FIELDS else (_clean(scene.get(key)) or "none")
            parts.append(f"{key.upper()}={value}")
        lines.append(f"SCENE {number:02d}: " + "; ".join(parts))
    return "\n".join(lines)

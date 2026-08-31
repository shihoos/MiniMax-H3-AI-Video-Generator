from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any


def ensure_shot_uid(shot: dict[str, Any], production_id: str) -> str:
    """Create a stable shot identity once; never derive it from mutable order."""
    existing = str(shot.get("shot_uid", "") or "").strip()
    if existing:
        return existing
    shot_id = str(shot.get("shot_id", "") or "").strip()
    if not shot_id:
        raise ValueError("Cannot create shot_uid without shot_id.")
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"minimax-h3:{production_id}:{shot_id}"))
    shot["shot_uid"] = uid
    return uid


def semantic_content_digest(shot: dict[str, Any]) -> str:
    payload = {key: shot.get(key) for key in (
        "scene_id", "characters", "location", "action", "camera_shot",
        "camera_movement", "lens_and_depth_of_field", "composition_notes",
        "lighting", "color_temperature", "mood", "visual_prompt",
    )}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def stable_seed(
    production_seed: str | int,
    shot: dict[str, Any],
) -> int:
    uid = ensure_shot_uid(shot, str(production_seed))
    material = f"{production_seed}\x1f{uid}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & 0x7FFFFFFFFFFFFFFF


def ensure_plan_lineage(plan: dict[str, Any], production_id: str, production_seed: str | int = "h3") -> dict[str, Any]:
    for shot in plan.get("shots", []) or []:
        if not isinstance(shot, dict):
            continue
        ensure_shot_uid(shot, production_id)
        shot["semantic_content_digest"] = semantic_content_digest(shot)
        if shot.get("seed") in (None, ""):
            shot["seed"] = stable_seed(production_seed, shot)
    return plan

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from schemas.character import Character


class ContinuityViolation(RuntimeError):
    def __init__(self, message: str, *, shot_id: str = "", previous_shot_id: str = "", details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.shot_id = shot_id
        self.previous_shot_id = previous_shot_id
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "shot_id": self.shot_id,
            "previous_shot_id": self.previous_shot_id,
            "details": deepcopy(self.details),
        }


class ContinuityLedger:
    """Deterministic scene-aware continuity state machine.

    Same-scene shots inherit the previous end-state. A scene boundary flushes
    environment/lighting/props/spatial state and previous-frame relay; only
    canonical character identity survives the cut.
    """

    SPATIAL_KEYS = ("character_spatial_bboxes", "character_spatial_regions")

    @staticmethod
    def _ordered_shots(plan: dict) -> list[dict]:
        scene_order = {
            str(scene.get("scene_id", "")).strip(): index
            for index, scene in enumerate(plan.get("scenes", []) or [])
            if isinstance(scene, dict) and str(scene.get("scene_id", "")).strip()
        }
        return sorted(
            [shot for shot in (plan.get("shots", []) or []) if isinstance(shot, dict)],
            key=lambda shot: (
                scene_order.get(str(shot.get("scene_id", "")).strip(), 10**9),
                int(shot.get("order", 0)),
            ),
        )

    def __init__(self, project_root: Path, production_id: str | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.production_id = str(production_id or "").strip() or "default"
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.production_id)
        self.root = self.project_root / "data" / "production" / "continuity" / safe
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict[str, Any]] = []

    @staticmethod
    def _identity_fingerprints(characters: list[Character | dict]) -> dict[str, str]:
        result: dict[str, str] = {}
        for item in characters:
            if isinstance(item, Character):
                result[item.character_id] = item.identity_fingerprint
            else:
                cid = str(item.get("character_id", "")).strip()
                if cid:
                    result[cid] = Character(
                        character_id=cid,
                        name=str(item.get("name", cid)),
                        role=str(item.get("role", "story character")),
                        description=str(item.get("description", "")),
                        personality=str(item.get("personality", "")),
                        appearance=dict(item.get("appearance", {}) or {}),
                        clothing=dict(item.get("clothing", {}) or {}),
                        distinctive_features=list(item.get("distinctive_features", []) or []),
                        character_state=dict(item.get("character_state", {}) or {}),
                        continuity_rules=list(item.get("continuity_rules", []) or []),
                    ).identity_fingerprint
        return result

    @staticmethod
    def _character_lookup(characters: list[Character | dict]):
        by_name: dict[str, Character | dict] = {}
        by_id: dict[str, Character | dict] = {}
        for item in characters:
            if isinstance(item, Character):
                by_name[item.name.strip().lower()] = item
                by_id[item.character_id] = item
            else:
                name = str(item.get("name", "")).strip().lower()
                cid = str(item.get("character_id", "")).strip()
                if name:
                    by_name[name] = item
                if cid:
                    by_id[cid] = item
        return by_name, by_id

    @staticmethod
    def _state_for_shot(shot: dict, prefix: str) -> dict[str, Any]:
        explicit = shot.get(prefix)
        if isinstance(explicit, dict) and explicit:
            state = deepcopy(explicit)
        else:
            state = {
                "location": str(shot.get("location", "") or ""),
                "lighting": str(shot.get("lighting", "") or ""),
                "characters": list(shot.get("characters", []) or []),
                "camera_side": str(shot.get("camera_side", "") or ""),
                "props": deepcopy(shot.get("props", []) or []),
                "state_description": str(
                    shot.get("continuity_start_state" if prefix.endswith("start_state") else "continuity_end_state", "") or ""
                ).strip(),
                "character_spatial_bboxes": deepcopy(shot.get("character_spatial_bboxes", {}) or {}),
                "character_spatial_regions": deepcopy(shot.get("character_spatial_regions", {}) or {}),
                "character_spatial_bboxes_start": deepcopy(shot.get("character_spatial_bboxes_start", {}) or {}),
                "character_spatial_bboxes_end": deepcopy(shot.get("character_spatial_bboxes_end", {}) or {}),
                "character_spatial_regions_start": deepcopy(shot.get("character_spatial_regions_start", {}) or {}),
                "character_spatial_regions_end": deepcopy(shot.get("character_spatial_regions_end", {}) or {}),
            }
        return state

    @staticmethod
    def _identity_map_for_shot(shot: dict, fingerprint_map: dict[str, str], name_to_id: dict[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in shot.get("characters", []) or []:
            label = str(name).strip()
            cid = name_to_id.get(label.lower(), "")
            result[label] = fingerprint_map.get(cid, "")
        return result

    @staticmethod
    def _scene_boundary(shot: dict, previous: dict[str, Any] | None) -> bool:
        if previous is None:
            return True
        return bool(shot.get("is_scene_boundary", False))

    @classmethod
    def validate_proposed(cls, plan: dict) -> None:
        previous_by_scene: dict[str, dict[str, Any]] = {}
        previous_by_order: dict[str, dict[str, Any]] = {}
        for shot in cls._ordered_shots(plan):
            scene_id = str(shot.get("scene_id", "")).strip()
            previous = previous_by_scene.get(scene_id)
            if cls._scene_boundary(shot, previous):
                previous_by_scene[scene_id] = shot
                continue
            if not isinstance(shot.get("continuity_start_state") or {}, dict):
                raise ContinuityViolation(
                    f"{shot.get('shot_id', '')}: continuity_start_state must be a dictionary.",
                    shot_id=str(shot.get("shot_id", "")),
                    previous_shot_id=str(previous.get("shot_id", "") if previous else ""),
                    details={"field": "continuity_start_state"},
                )
            if not isinstance(previous.get("continuity_end_state") or {}, dict):
                raise ContinuityViolation(
                    f"{previous.get('shot_id', '')}: continuity_end_state must be a dictionary.",
                    shot_id=str(shot.get("shot_id", "")),
                    previous_shot_id=str(previous.get("shot_id", "")),
                    details={"field": "continuity_end_state"},
                )
            start = deepcopy(shot.get("continuity_start_state") or {})
            end = deepcopy(previous.get("continuity_end_state") or {})
            if start and end and start != end:
                raise ContinuityViolation(
                    f"Continuity mismatch: {previous.get('shot_id')} -> {shot.get('shot_id')}",
                    shot_id=str(shot.get("shot_id", "")),
                    previous_shot_id=str(previous.get("shot_id", "")),
                    details={"expected_start_state": end, "proposed_start_state": start},
                )
            previous_ids = previous.get("identity_fingerprints") or {}
            current_ids = shot.get("identity_fingerprints") or {}
            for character, fingerprint in previous_ids.items():
                if character in current_ids and fingerprint and current_ids.get(character) != fingerprint:
                    raise ContinuityViolation(
                        f"Identity fingerprint mismatch for {character}: {previous.get('shot_id')} -> {shot.get('shot_id')}",
                        shot_id=str(shot.get("shot_id", "")),
                        previous_shot_id=str(previous.get("shot_id", "")),
                        details={"character": character, "expected_fingerprint": fingerprint, "proposed_fingerprint": current_ids.get(character)},
                    )

            previous_boxes = previous.get("character_spatial_bboxes_end") or previous.get("character_spatial_bboxes") or {}
            current_start_boxes = shot.get("character_spatial_bboxes_start") or {}
            for name, previous_box in previous_boxes.items():
                if name in current_start_boxes and current_start_boxes[name] != previous_box:
                    raise ContinuityViolation(
                        f"Spatial continuity mismatch for {name}: {previous.get('shot_id')} -> {shot.get('shot_id')}",
                        shot_id=str(shot.get("shot_id", "")),
                        previous_shot_id=str(previous.get("shot_id", "")),
                        details={"character": name, "expected_start_bbox": previous_box, "proposed_start_bbox": current_start_boxes[name]},
                    )
            previous_by_scene[scene_id] = shot
            previous_by_order[str(shot.get("shot_id", ""))] = shot

    @classmethod
    def _scene_reset_state(cls, shot: dict, identity_fingerprints: dict[str, str]) -> dict[str, Any]:
        return {
            "scene_boundary_reset": True,
            "environment": {},
            "lighting": {},
            "props": [],
            "camera_side": "",
            "character_spatial_bboxes": {},
            "character_spatial_regions": {},
            "canonical_character_identity_fingerprints": identity_fingerprints,
        }

    def apply(self, plan: dict, characters: list[dict] | list[Character]) -> dict:
        fingerprint_map = self._identity_fingerprints(characters)
        name_to_id: dict[str, str] = {}
        for item in characters:
            if isinstance(item, Character):
                name_to_id[item.name.strip().lower()] = item.character_id
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip().lower()
                cid = str(item.get("character_id", "")).strip()
                if name and cid:
                    name_to_id[name] = cid

        self.validate_proposed(plan)
        previous_by_scene: dict[str, dict[str, Any] | None] = {}
        self.entries = []

        for shot in self._ordered_shots(plan):
            scene_id = str(shot.get("scene_id", "")).strip()
            previous = previous_by_scene.get(scene_id)
            is_boundary = self._scene_boundary(shot, previous)
            proposed_end = self._state_for_shot(shot, "continuity_end_state")
            if is_boundary:
                start = self._scene_reset_state(shot, self._identity_map_for_shot(shot, fingerprint_map, name_to_id))
                repaired = bool(previous is not None)
                shot["is_scene_boundary"] = True
            else:
                start = deepcopy(previous.get("continuity_end_state") or {}) if previous else self._state_for_shot(shot, "continuity_start_state")
                repaired = bool(previous is not None)
                shot["is_scene_boundary"] = False

            shot["continuity_start_state"] = start
            shot["continuity_end_state"] = proposed_end
            shot["continuity_repair_applied"] = repaired
            shot["identity_fingerprints"] = self._identity_map_for_shot(shot, fingerprint_map, name_to_id)

            if not is_boundary:
                previous_boxes = previous.get("character_spatial_bboxes_end") or previous.get("character_spatial_bboxes") or {}
                previous_regions = previous.get("character_spatial_regions_end") or previous.get("character_spatial_regions") or {}
                if previous_boxes:
                    shot["character_spatial_bboxes_start"] = deepcopy(previous_boxes)
                if previous_regions:
                    shot["character_spatial_regions_start"] = deepcopy(previous_regions)
            if shot.get("character_spatial_bboxes_end"):
                shot["character_spatial_bboxes"] = deepcopy(shot["character_spatial_bboxes_end"])
            if shot.get("character_spatial_regions_end"):
                shot["character_spatial_regions"] = deepcopy(shot["character_spatial_regions_end"])

            entry = {
                "scene_id": scene_id,
                "shot_id": str(shot.get("shot_id", "")),
                "order": int(shot.get("order", 0)),
                "is_scene_boundary": bool(shot["is_scene_boundary"]),
                "start_state": deepcopy(start),
                "end_state": deepcopy(proposed_end),
                "identity_fingerprints": deepcopy(shot["identity_fingerprints"]),
                "character_spatial_bboxes": deepcopy(shot.get("character_spatial_bboxes", {}) or {}),
                "character_spatial_regions": deepcopy(shot.get("character_spatial_regions", {}) or {}),
                "character_spatial_bboxes_start": deepcopy(shot.get("character_spatial_bboxes_start", {}) or {}),
                "character_spatial_bboxes_end": deepcopy(shot.get("character_spatial_bboxes_end", {}) or {}),
                "character_spatial_regions_start": deepcopy(shot.get("character_spatial_regions_start", {}) or {}),
                "character_spatial_regions_end": deepcopy(shot.get("character_spatial_regions_end", {}) or {}),
                "repair_applied": repaired,
            }
            self.entries.append(entry)
            previous_by_scene[scene_id] = shot

        path = self.root / "continuity_ledger.json"
        path.write_text(json.dumps({"production_id": self.production_id, "entries": self.entries}, indent=2, ensure_ascii=False), encoding="utf-8")
        plan["continuity_ledger_path"] = str(path)
        return plan

    def apply_field_level_fallback(self, plan: dict, characters: list[dict] | list[Character]) -> dict:
        """Repair only continuity-carrying fields after bounded Qwen repair attempts."""
        shots = self._ordered_shots(plan)
        previous_by_scene: dict[str, dict[str, Any] | None] = {}
        for shot in shots:
            scene_id = str(shot.get("scene_id", "")).strip()
            previous = previous_by_scene.get(scene_id)
            if previous is not None and not bool(shot.get("is_scene_boundary", False)):
                shot["continuity_start_state"] = deepcopy(previous.get("continuity_end_state") or {})
                shot["character_spatial_bboxes_start"] = deepcopy(previous.get("character_spatial_bboxes_end") or previous.get("character_spatial_bboxes") or {})
                shot["character_spatial_regions_start"] = deepcopy(previous.get("character_spatial_regions_end") or previous.get("character_spatial_regions") or {})
                shot["continuity_repair_applied"] = True
            previous_by_scene[scene_id] = shot
        return self.apply(plan, characters)

    @staticmethod
    def validate(plan: dict) -> None:
        previous_by_scene: dict[str, dict[str, Any]] = {}
        for shot in ContinuityLedger._ordered_shots(plan):
            scene_id = str(shot.get("scene_id", "")).strip()
            previous = previous_by_scene.get(scene_id)
            if previous is not None and not bool(shot.get("is_scene_boundary", False)):
                if previous.get("continuity_end_state") != shot.get("continuity_start_state"):
                    raise RuntimeError(
                        f"Continuity mismatch: {previous.get('shot_id')} -> {shot.get('shot_id')}"
                    )
            previous_by_scene[scene_id] = shot

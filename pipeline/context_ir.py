from __future__ import annotations

from typing import Any


class H3ContextIRCompiler:
    """Translate richer production IR into the authoritative H3 execution context."""

    VERSION = 1

    @classmethod
    def validate(cls, context_ir: dict[str, Any]) -> None:
        if not isinstance(context_ir, dict):
            raise TypeError("H3 Context-IR must be a mapping.")
        version = int(context_ir.get("version", 0) or 0)
        if version != cls.VERSION:
            raise ValueError(
                f"Unsupported H3 Context-IR version: {version}; expected {cls.VERSION}."
            )
        prompt = str(context_ir.get("h3_prompt", "") or "").strip()
        if not prompt:
            raise ValueError("H3 Context-IR contains an empty h3_prompt.")

    @classmethod
    def prompt(cls, context_ir: dict[str, Any]) -> str:
        cls.validate(context_ir)
        return str(context_ir["h3_prompt"]).strip()

    def compile(self, plan: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
        prompt = str(shot.get("h3_prompt", "") or shot.get("visual_prompt", "") or "").strip()
        return {
            "version": self.VERSION,
            "story": str(plan.get("story", "") or ""),
            "scene": {
                "scene_id": str(shot.get("scene_id", "") or ""),
                "location": str(shot.get("location", "") or ""),
                "time_of_day": str(shot.get("time_of_day", "") or ""),
                "mood": str(shot.get("mood", "") or ""),
            },
            "h3_prompt": prompt,
            "shot": {
                "shot_id": str(shot.get("shot_id", "") or ""),
                "duration_seconds": float(shot.get("duration_seconds", 0.0) or 0.0),
                "camera": {
                    "shot": str(shot.get("camera_shot", "") or ""),
                    "movement": str(shot.get("camera_movement", "") or ""),
                    "lens": str(shot.get("lens_and_depth_of_field", "") or ""),
                },
                "composition": str(shot.get("composition_notes", "") or ""),
                "lighting": str(shot.get("lighting", "") or ""),
                "action": str(shot.get("action", "") or ""),
            },
            "continuity": shot.get("continuity_start_state", {}) or {},
            "references": [
                {"index": index, "binding": binding}
                for index, binding in enumerate(shot.get("reference_bindings", []) or [], start=1)
            ],
            "audio": {
                "soundscape": str(shot.get("overall_soundscape", "") or ""),
                "music": str(shot.get("non_diegetic_music", "") or ""),
            },
        }

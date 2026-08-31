from __future__ import annotations

from typing import Any


class H3ContextIRCompiler:
    """Translate the richer production IR into a compact H3-compatible context representation."""

    VERSION = 1

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

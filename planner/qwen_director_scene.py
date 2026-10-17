from __future__ import annotations

import re


class QwenDirectorSceneMixin:
    """Prepare deterministic scene context without guessing story semantics.

    Scene IDs and scene boundaries are structural. Dramatic purpose, mood,
    location, weather, time of day, and lighting are story-dependent and are
    authored by the existing Qwen shot-batch response. This layer only keeps
    the source-grounded visible moment compact enough for that response.
    """

    _LEGACY_POSITIONAL_FUNCTIONS = frozenset({
        "setup", "catalyst", "development", "midpoint", "climax", "finale",
    })

    @classmethod
    def _annotate_scene_context(cls, scenes: list[dict]) -> list[dict]:
        result: list[dict] = []
        for raw_scene in scenes:
            if not isinstance(raw_scene, dict):
                continue
            scene = dict(raw_scene)

            # Discard only obsolete fixed-role labels emitted by earlier versions.
            # Do not delete Qwen-authored dramatic_purpose or other semantic fields.
            legacy_function = str(scene.get("scene_function", "") or "").strip().casefold()
            if legacy_function in cls._LEGACY_POSITIONAL_FUNCTIONS:
                scene.pop("scene_function", None)

            objective = str(scene.get("scene_objective", "") or "").strip()
            description = str(scene.get("description", "") or "").strip()
            existing_moment = str(scene.get("obligatory_moment", "") or "").strip()
            moment = existing_moment or objective or description
            moment = re.sub(r"\s+", " ", moment).strip()
            if len(moment) > 220:
                moment = moment[:220].rsplit(" ", 1)[0].strip()
            if moment:
                scene["obligatory_moment"] = moment
            else:
                scene.pop("obligatory_moment", None)
            result.append(scene)
        return result

    @classmethod
    def _annotate_scene_functions(cls, scenes: list[dict]) -> list[dict]:
        """Backward-compatible alias; never assigns scene roles by position/keywords."""
        return cls._annotate_scene_context(scenes)

    def _planner(self):
        if self._fallback_planner is None:
            from planner.production_planner import ProductionPlanner
            self._fallback_planner = ProductionPlanner(self.project_root)
        return self._fallback_planner

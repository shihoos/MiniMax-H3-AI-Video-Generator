from __future__ import annotations

import re


class QwenDirectorSceneMixin:
    """Prepare deterministic scene context without guessing story semantics.

    Scene IDs and scene boundaries are structural. In AI Story/Expand mode,
    dramatic purpose is authored by the Architect and must survive this layer
    unchanged. The shot-batch response supplies cinematic setting, mood, weather,
    time of day, and lighting. This layer only keeps the source-grounded visible
    moment compact enough for that response.
    """

    @classmethod
    def _annotate_scene_context(cls, scenes: list[dict]) -> list[dict]:
        result: list[dict] = []
        for raw_scene in scenes:
            if not isinstance(raw_scene, dict):
                continue
            scene = dict(raw_scene)

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

    def _planner(self):
        if self._fallback_planner is None:
            from planner.production_planner import ProductionPlanner
            self._fallback_planner = ProductionPlanner(self.project_root)
        return self._fallback_planner

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pipeline.continuity_manager import (
    ContinuityManager,
)

from planner.config import (
    PRODUCTION_DIR,
    WORKFLOW_AUTO,
    ensure_directories,
)

from planner.production_planner import (
    ProductionPlanner,
)


class ProductionOrchestrator:

    def __init__(self):

        ensure_directories()

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        self.planner = ProductionPlanner(
            self.project_root
        )

        self.continuity_manager = (
            ContinuityManager()
        )

    def create_production_plan(
        self,
        mode: str,
        user_input: str,
        workflow_mode: str = WORKFLOW_AUTO,
        profile: str = "base",
    ) -> dict:

        plan = self.planner.build(
            mode=mode,
            user_input=user_input,
            workflow_mode=workflow_mode,
            profile=profile,
        )

        # Apply existing project continuity state.
        previous_shot = None

        all_shots = []

        scenes_by_id = {
            scene["scene_id"]: scene
            for scene in plan["scenes"]
        }

        for shot in plan["shots"]:

            continuity_context = (
                self.continuity_manager
                .build_context(
                    previous_shot
                )
            )

            if continuity_context:
                shot["continuity_notes"] = (
                    (
                        shot.get(
                            "continuity_notes",
                            "",
                        )
                        + "\n"
                        + continuity_context
                    )
                    .strip()
                )

            if previous_shot is not None:
                shot["previous_shot"] = (
                    previous_shot["shot_id"]
                )

                previous_shot["next_shot"] = (
                    shot["shot_id"]
                )

            previous_shot = shot

            all_shots.append(
                shot
            )

        plan["shots"] = all_shots

        plan["created_at"] = (
            datetime.now()
            .isoformat()
        )

        output_path = (
            PRODUCTION_DIR
            / "production_plan.json"
        )

        output_path.write_text(
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        plan[
            "production_plan_path"
        ] = str(
            output_path
        )

        return plan

    def unload_models(self):
        # No external planner model is loaded.
        return None

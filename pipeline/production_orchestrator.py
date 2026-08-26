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

        previous_shot = None

        for shot in plan["shots"]:

            context = (
                self.continuity_manager
                .build_context(
                    previous_shot
                )
            )

            if context:
                shot["continuity_notes"] = (
                    (
                        shot.get(
                            "continuity_notes",
                            "",
                        )
                        + "\n"
                        + context
                    ).strip()
                )

            if previous_shot is not None:

                shot["previous_shot"] = (
                    previous_shot[
                        "shot_id"
                    ]
                )

                previous_shot[
                    "next_shot"
                ] = shot[
                    "shot_id"
                ]

            previous_shot = shot

        plan["preview_ready"] = True
        plan["created_at"] = (
            datetime.now().isoformat()
        )

        preview_path = (
            PRODUCTION_DIR
            / "story_preview.json"
        )

        preview_path.write_text(
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
            preview_path
        )

        return plan

    def unload_models(self):
        # There is no separate planner model.
        # H3 loads its single locked Qwen encoder
        # through the ComfyUI workflow.
        return None

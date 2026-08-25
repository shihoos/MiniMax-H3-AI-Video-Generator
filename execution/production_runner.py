from __future__ import annotations

from pathlib import Path
from typing import Any

from execution.shot_executor import (
    ShotExecutor,
)

from pipeline.h3_scene_continuity import (
    H3SceneContinuity,
)

from scheduler.gpu_scheduler import (
    GPUScheduler,
)


class ProductionRunner:

    def __init__(
        self,
        project_root: Path,
        comfy_clients: dict[int, object],
    ):

        self.project_root = Path(
            project_root
        )

        self.clients = dict(
            comfy_clients
        )

        if not self.clients:
            raise ValueError(
                "At least one ComfyUI client is required."
            )

        self.input_root = (
            self.project_root
            / "ComfyUI"
            / "input"
        )

        self.output_root = (
            self.project_root
            / "data"
            / "production"
            / "h3"
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.continuity = (
            H3SceneContinuity(
                self.project_root
            )
        )

    def _executor(
        self,
        gpu_id: int,
        scene_id: str,
    ) -> ShotExecutor:

        input_dir = (
            self.input_root
            / f"gpu_{gpu_id}"
            / scene_id
        )

        return ShotExecutor(
            comfy_client=self.clients[gpu_id],
            project_root=self.project_root,
            comfy_input_dir=input_dir,
        )

    @staticmethod
    def _workflow_for_shot(
        shot: dict,
        profile: str,
    ) -> str:

        explicit = (
            str(
                shot.get(
                    "workflow_mode",
                    ""
                )
            ).strip()
        )

        if explicit in {
            "ref2v",
            "turbo_ref2v",
        }:
            return explicit

        if profile == "turbo":
            return "turbo_ref2v"

        return "ref2v"

    def _run_scene(
        self,
        gpu_id: int,
        scene_id: str,
        shots: list[dict],
        profile: str,
    ):

        executor = self._executor(
            gpu_id,
            scene_id,
        )

        ordered = sorted(
            shots,
            key=lambda shot:
            int(
                shot.get(
                    "order",
                    0,
                )
            ),
        )

        output_dir = (
            self.output_root
            / f"gpu_{gpu_id}"
            / scene_id
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = []

        previous_video = None
        previous_shot = None

        for shot in ordered:

            shot = dict(
                shot
            )

            if previous_video is not None:

                last_frame = (
                    self.continuity.prepare_next_shot(
                        previous_video,
                        scene_id,
                        str(
                            previous_shot[
                                "shot_id"
                            ]
                        ),
                    )
                )

                shot[
                    "reference_images"
                ] = [
                    str(last_frame),
                    *shot.get(
                        "reference_images",
                        [],
                    ),
                ][:9]

            workflow_mode = (
                self._workflow_for_shot(
                    shot,
                    profile,
                )
            )

            result = executor.execute_shot(
                shot=shot,
                workflow_mode=workflow_mode,
                output_dir=output_dir,
            )

            results.append(
                result
            )

            previous_video = Path(
                result
            )

            previous_shot = shot

        return results

    def run(
        self,
        production_plan: dict[str, Any],
    ):

        shots = list(
            production_plan.get(
                "shots",
                [],
            )
        )

        if not shots:
            raise RuntimeError(
                "Production plan contains no shots."
            )

        profile = str(
            production_plan.get(
                "profile",
                "base",
            )
        )

        scenes = {}

        for shot in shots:

            scene_id = str(
                shot.get(
                    "scene_id",
                    "",
                )
            )

            if not scene_id:
                raise RuntimeError(
                    "Shot is missing scene_id."
                )

            scenes.setdefault(
                scene_id,
                [],
            ).append(
                shot
            )

        scene_ids = list(
            scenes
        )

        scheduler = GPUScheduler(
            gpu_ids=sorted(
                self.clients
            )
        )

        # Independent scenes may run in parallel.
        scene_jobs = [
            (
                scene_id,
                scenes[scene_id]
            )
            for scene_id in scene_ids
        ]

        results = scheduler.run_independent(
            scene_jobs,
            lambda gpu_id, job:
                (
                    job[0],
                    self._run_scene(
                        gpu_id,
                        job[0],
                        job[1],
                        profile,
                    ),
                ),
        )

        results.sort(
            key=lambda item:
            min(
                int(
                    shot.get(
                        "order",
                        0,
                    )
                )
                for shot in scenes[
                    item[0]
                ]
            )
        )

        return [
            video
            for _scene, videos
            in results
            for video
            in videos
        ]

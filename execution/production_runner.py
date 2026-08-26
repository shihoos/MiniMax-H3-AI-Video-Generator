from __future__ import annotations

from pathlib import Path
from typing import Any

from execution.assembly_manager import (
    AssemblyManager,
)
from execution.shot_executor import (
    ShotExecutor,
)
from pipeline.h3_scene_continuity import (
    H3SceneContinuity,
)
from pipeline.identity_anchor_store import (
    IdentityAnchorStore,
)
from planner.config import (
    DELIVERY_FPS,
    DELIVERY_HEIGHT,
    DELIVERY_WIDTH,
    PROFILE_BASE,
    PROFILE_TURBO,
    PROFILE_UPSCALE,
    WORKFLOW_REF2V,
    WORKFLOW_TURBO_REF2V,
    WORKFLOW_UPSCALE,
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

        self.identity_anchors = (
            IdentityAnchorStore(
                self.project_root
            )
        )

    def _executor(
        self,
        gpu_id: int,
        scene_id: str,
    ):

        input_dir = (
            self.input_root
            / f"gpu_{gpu_id}"
            / scene_id
        )

        return ShotExecutor(
            comfy_client=self.clients[
                gpu_id
            ],
            project_root=self.project_root,
            comfy_input_dir=input_dir,
        )

    @staticmethod
    def _workflow_for_shot(
        shot: dict,
        profile: str,
    ) -> str:

        explicit = str(
            shot.get(
                "workflow_mode",
                "",
            )
        ).strip()

        if explicit in {
            WORKFLOW_REF2V,
            WORKFLOW_TURBO_REF2V,
            WORKFLOW_UPSCALE,
        }:
            return explicit

        if profile == PROFILE_TURBO:
            return WORKFLOW_TURBO_REF2V

        if profile == PROFILE_UPSCALE:
            return WORKFLOW_UPSCALE

        return WORKFLOW_REF2V

    @staticmethod
    def _sort_shots(
        shots,
    ):
        return sorted(
            shots,
            key=lambda shot: int(
                shot.get(
                    "order",
                    0,
                )
            ),
        )

    def _character_map(
        self,
        production_plan: dict,
    ) -> dict:

        return {
            str(
                character.get(
                    "name",
                    "",
                )
            ).strip().lower():
                character
            for character in production_plan.get(
                "characters",
                [],
            )
            if str(
                character.get(
                    "name",
                    "",
                )
            ).strip()
        }

    def _add_identity_anchors(
        self,
        shot: dict,
        character_map: dict,
    ):

        refs = list(
            shot.get(
                "reference_images",
                [],
            )
            or []
        )

        for name in (
            shot.get(
                "characters",
                [],
            )
            or []
        ):

            character = character_map.get(
                str(name).lower()
            )

            if not character:
                continue

            if (
                character.get(
                    "reference_mode"
                )
                == "provided"
            ):
                continue

            character_id = character.get(
                "character_id"
            )

            if not character_id:
                continue

            anchor = (
                self.identity_anchors.latest_anchor(
                    character_id
                )
            )

            if (
                anchor
                and str(anchor)
                not in refs
                and len(refs) < 9
            ):
                refs.append(
                    str(anchor)
                )

        shot[
            "reference_images"
        ] = refs[:9]

    def _persist_first_appearance_anchors(
        self,
        shot: dict,
        character_map: dict,
        frame_path: Path,
    ):

        for name in (
            shot.get(
                "characters",
                [],
            )
            or []
        ):

            character = character_map.get(
                str(name).lower()
            )

            if not character:
                continue

            if (
                character.get(
                    "reference_mode"
                )
                == "provided"
            ):
                continue

            character_id = character.get(
                "character_id"
            )

            if not character_id:
                continue

            self.identity_anchors.save_first_anchor(
                character_id=character_id,
                shot_id=str(
                    shot["shot_id"]
                ),
                source_frame=frame_path,
            )

    def _run_scene(
        self,
        gpu_id,
        scene_id,
        shots,
        profile,
        character_map,
    ):

        executor = self._executor(
            gpu_id,
            scene_id,
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

        for original in self._sort_shots(
            shots
        ):

            shot = dict(
                original
            )

            # Reuse the first successful generated
            # appearance as a persistent identity anchor.
            self._add_identity_anchors(
                shot,
                character_map,
            )

            if previous_video is not None:

                last_frame = (
                    self.continuity
                    .prepare_next_shot(
                        previous_video,
                        scene_id,
                        previous_shot[
                            "shot_id"
                        ],
                    )
                )

                refs = list(
                    shot.get(
                        "reference_images",
                        [],
                    )
                    or []
                )

                if (
                    str(last_frame)
                    not in refs
                ):
                    refs.insert(
                        0,
                        str(last_frame),
                    )

                shot[
                    "reference_images"
                ] = refs[:9]

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

            result = Path(
                result
            )

            results.append(
                result
            )

            # The final frame becomes the continuity frame
            # and, on first appearance, the character anchor.
            anchor_frame = (
                self.continuity
                .prepare_next_shot(
                    result,
                    scene_id,
                    shot[
                        "shot_id"
                    ],
                )
            )

            self._persist_first_appearance_anchors(
                shot,
                character_map,
                anchor_frame,
            )

            previous_video = result
            previous_shot = shot

        return results

    def _parallel_safe(
        self,
        production_plan: dict,
    ) -> bool:

        return bool(
            production_plan.get(
                "parallel_safe",
                False,
            )
        )

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
                PROFILE_BASE,
            )
        ).strip().lower()

        if profile not in {
            PROFILE_BASE,
            PROFILE_TURBO,
            PROFILE_UPSCALE,
        }:
            raise RuntimeError(
                f"Unsupported profile: {profile}"
            )

        scenes = {}

        for shot in shots:

            scene_id = str(
                shot.get(
                    "scene_id",
                    "",
                )
            ).strip()

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

        character_map = (
            self._character_map(
                production_plan
            )
        )

        scene_jobs = list(
            scenes.items()
        )

        if (
            self._parallel_safe(
                production_plan
            )
            and len(self.clients) > 1
        ):

            scheduler = GPUScheduler(
                gpu_ids=sorted(
                    self.clients
                )
            )

            scene_results = (
                scheduler.run_independent(
                    scene_jobs,
                    lambda gpu_id, job: (
                        job[0],
                        self._run_scene(
                            gpu_id,
                            job[0],
                            job[1],
                            profile,
                            character_map,
                        ),
                    ),
                )
            )

        else:

            # Story continuity takes precedence over parallelism.
            # Use one worker sequentially when scenes share
            # characters or continuity dependencies.
            gpu_id = sorted(
                self.clients
            )[0]

            scene_results = []

            for job in scene_jobs:
                scene_results.append(
                    (
                        job[0],
                        self._run_scene(
                            gpu_id,
                            job[0],
                            job[1],
                            profile,
                            character_map,
                        ),
                    )
                )

        scene_results.sort(
            key=lambda item: min(
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

        videos = []

        for _scene_id, scene_videos in (
            scene_results
        ):
            videos.extend(
                scene_videos
            )

        assembly_dir = (
            self.project_root
            / "data"
            / "production"
            / "final"
        )

        assembler = AssemblyManager(
            assembly_dir
        )

        final_video = assembler.assemble(
            videos,
            final_name=(
                f"h3_{profile}_720p.mp4"
            ),
            width=int(
                production_plan.get(
                    "delivery_width",
                    DELIVERY_WIDTH,
                )
            ),
            height=int(
                production_plan.get(
                    "delivery_height",
                    DELIVERY_HEIGHT,
                )
            ),
            fps=int(
                production_plan.get(
                    "delivery_fps",
                    DELIVERY_FPS,
                )
            ),
        )

        return {
            "shot_outputs": videos,
            "final_video": final_video,
            "profile": profile,
        }

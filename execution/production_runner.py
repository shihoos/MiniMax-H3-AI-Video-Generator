from __future__ import annotations

import uuid
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
from pipeline.production_checkpoint import (
    ProductionCheckpoint,
)
from pipeline.identity_anchor_store import (
    IdentityAnchorStore,
)
from planner.config import (
    DELIVERY_FPS,
    DELIVERY_HEIGHT,
    DELIVERY_WIDTH,
    H3_MAX_REFERENCE_IMAGES,
    PROFILE_BASE,
    PROFILE_TURBO,
    PROFILE_UPSCALE,
    WORKFLOW_REF2V,
    WORKFLOW_TURBO_REF2V,
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
        ).resolve()

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

        self.production_id = None

        self.production_input_root = (
            self.input_root
        )

        self.output_root = (
            self.project_root
            / "data"
            / "production"
            / "h3"
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

    @staticmethod
    def _safe_name(
        value,
    ) -> str:

        text = str(
            value or ""
        ).strip()

        cleaned = "".join(
            char
            if (
                char.isalnum()
                or char in "._-"
            )
            else "_"
            for char in text
        )

        return (
            cleaned[:96]
            or "production"
        )

    def _resolve_production_id(
        self,
        production_plan: dict,
    ) -> str:

        requested = str(
            production_plan.get(
                "production_id",
                "",
            )
            or ""
        ).strip()

        if requested:
            return self._safe_name(
                requested
            )

        return (
            "production_"
            f"{uuid.uuid4().hex}"
        )

    def _prepare_production_paths(
        self,
        production_id: str,
    ) -> None:

        self.production_id = (
            production_id
        )

        self.production_input_root = (
            self.input_root
            / production_id
        )

        self.output_root = (
            self.project_root
            / "data"
            / "production"
            / production_id
            / "h3"
        )

        self.production_input_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Production-scoped continuity state.
        #
        # The production ID is only known when run() starts,
        # so these stores must be created here rather than left
        # as the global instances created during __init__.
        self.continuity = (
            H3SceneContinuity(
                self.project_root,
                production_id=production_id,
            )
        )

        self.identity_anchors = (
            IdentityAnchorStore(
                self.project_root,
                production_id=production_id,
            )
        )

    def _executor(
        self,
        gpu_id: int,
        scene_id: str,
    ):

        return ShotExecutor(
            comfy_client=self.clients[
                gpu_id
            ],
            project_root=self.project_root,
            comfy_input_dir=(
                self.production_input_root
                / f"gpu_{gpu_id}"
                / self._safe_name(scene_id)
            ),
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
            or ""
        ).strip()

        if explicit in {
            WORKFLOW_REF2V,
            WORKFLOW_TURBO_REF2V,
        }:
            return explicit

        if profile == PROFILE_TURBO:
            return WORKFLOW_TURBO_REF2V

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
            for character
            in production_plan.get(
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
    ) -> None:

        references = list(
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

            character_id = (
                character.get(
                    "character_id"
                )
            )

            if not character_id:
                continue

            anchor = (
                self.identity_anchors
                .latest_anchor(
                    character_id
                )
            )

            if (
                anchor
                and str(anchor)
                not in references
            ):
                references.insert(
                    0,
                    str(anchor),
                )

        shot[
            "reference_images"
        ] = references[
            :H3_MAX_REFERENCE_IMAGES
        ]

    def _persist_first_appearance_anchors(
        self,
        shot: dict,
        character_map: dict,
        frame_path: Path,
    ) -> None:

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

            character_id = (
                character.get(
                    "character_id"
                )
            )

            if not character_id:
                continue

            self.identity_anchors.save_first_anchor(
                character_id=character_id,
                shot_id=str(
                    shot[
                        "shot_id"
                    ]
                ),
                source_frame=frame_path,
            )

    # ========================================================
    # RENDER CHECKPOINT / RESUME
    # ========================================================

    def _checkpoint_store(self) -> ProductionCheckpoint:
        return ProductionCheckpoint(
            self.project_root
        )

    def _load_render_checkpoint(
        self,
        production_id: str,
    ) -> dict | None:
        try:
            store = self._checkpoint_store()
            return store.load(
                production_id
            )
        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
            TypeError,
        ):
            return None

    def _update_render_checkpoint(
        self,
        production_id: str,
        *,
        status: str,
        stage: str,
        current_scene_id: str = "",
        completed_shot_ids: list[str] | None = None,
        completed_shot: dict[str, dict] | None = None,
        error: str = "",
        final_video: str = "",
        shot_outputs: list[str] | None = None,
    ) -> None:
        store = self._checkpoint_store()

        updates = {
            "status": status,
            "stage": stage,
            "current_scene_id": current_scene_id,
            "completed_shot_ids": list(
                completed_shot_ids or []
            ),
            "error": str(error or ""),
        }

        if completed_shot is not None:
            updates[
                "completed_shots"
            ] = dict(
                completed_shot
            )

        if final_video:
            updates[
                "final_video"
            ] = str(
                final_video
            )

        if shot_outputs is not None:
            updates[
                "shot_outputs"
            ] = [
                str(path)
                for path in shot_outputs
            ]

        store.update(
            production_id,
            updates,
        )

    @staticmethod
    def _existing_shot_output(
        output: Path,
        shot_id: str | None = None,
    ) -> Path | None:
        path = (
            Path(output)
            if shot_id is None
            else Path(output) / f"{shot_id}.mp4"
        )

        if (
            path.is_file()
            and path.stat().st_size > 0
        ):
            return path.resolve()

        return None

    def _run_scene(
        self,
        gpu_id,
        scene_id,
        shots,
        profile,
        character_map,
        upscale_enabled,
        production_id: str,
        completed_shots: dict[str, dict],
    ):

        executor = self._executor(
            gpu_id,
            scene_id,
        )

        output_dir = (
            self.output_root
            / f"gpu_{gpu_id}"
            / self._safe_name(scene_id)
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

            shot_id = str(
                shot.get(
                    "shot_id",
                    "",
                )
                or ""
            ).strip()

            if not shot_id:
                raise RuntimeError(
                    "Shot is missing shot_id."
                )

            # ------------------------------------------------
            # RESUME BY RECORDED OUTPUT PATH + GPU.
            # Never assume the original output was rendered
            # on the GPU currently executing the resume.
            # ------------------------------------------------
            record = completed_shots.get(
                shot_id
            )

            existing = None

            if isinstance(
                record,
                dict,
            ):

                recorded_output = str(
                    record.get(
                        "output",
                        "",
                    )
                    or ""
                ).strip()

                if recorded_output:
                    existing = (
                        self._existing_shot_output(
                            Path(
                                recorded_output
                            )
                        )
                    )

                # Backward compatibility for older checkpoints that
                # have GPU ownership but no absolute output path.
                if existing is None:
                    recorded_gpu = record.get(
                        "gpu_id",
                        gpu_id,
                    )

                    legacy_output_dir = (
                        self.output_root
                        / f"gpu_{recorded_gpu}"
                        / self._safe_name(
                            scene_id
                        )
                    )

                    existing = (
                        self._existing_shot_output(
                            legacy_output_dir,
                            shot_id,
                        )
                    )

                if existing is not None:

                    result = Path(
                        existing
                    )

                    results.append(
                        result
                    )

                    previous_video = result
                    previous_shot = shot

                    print(
                        "[H3 RESUME] skipping completed shot",
                        shot_id,
                        "->",
                        result,
                        flush=True,
                    )

                    continue

            self._update_render_checkpoint(
                production_id,
                status="rendering",
                stage="rendering",
                current_scene_id=scene_id,
            )

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

                references = list(
                    shot.get(
                        "reference_images",
                        [],
                    )
                    or []
                )

                if (
                    str(last_frame)
                    not in references
                ):
                    references.insert(
                        0,
                        str(last_frame),
                    )

                shot[
                    "reference_images"
                ] = references[
                    :H3_MAX_REFERENCE_IMAGES
                ]

            workflow_mode = (
                self._workflow_for_shot(
                    shot,
                    profile,
                )
            )

            result = (
                executor.execute_shot(
                    shot=shot,
                    workflow_mode=workflow_mode,
                    output_dir=output_dir,
                    upscale=upscale_enabled,
                )
            )

            result = Path(
                result
            ).resolve()

            if (
                not result.is_file()
                or result.stat().st_size <= 0
            ):
                raise RuntimeError(
                    "Shot execution returned an invalid output: "
                    f"{result}"
                )

            results.append(
                result
            )

            anchor_frame = (
                self.continuity
                .prepare_next_shot(
                    result,
                    scene_id,
                    shot["shot_id"],
                )
            )

            self._persist_first_appearance_anchors(
                shot,
                character_map,
                anchor_frame,
            )

            previous_video = result
            previous_shot = shot

            completed_shots[
                shot_id
            ] = {
                "gpu_id": int(gpu_id),
                "scene_id": str(
                    scene_id
                ),
                "output": str(
                    result
                ),
            }

            self._update_render_checkpoint(
                production_id,
                status="rendering",
                stage="rendering",
                current_scene_id=scene_id,
                completed_shot_ids=[
                    shot_id
                ],
                completed_shot={
                    shot_id: completed_shots[
                        shot_id
                    ],
                },
            )

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

        production_id = (
            self._resolve_production_id(
                production_plan
            )
        )

        production_plan[
            "production_id"
        ] = production_id

        self._prepare_production_paths(
            production_id
        )

        profile = str(
            production_plan.get(
                "profile",
                PROFILE_BASE,
            )
            or PROFILE_BASE
        ).strip().lower()

        if profile not in {
            PROFILE_BASE,
            PROFILE_TURBO,
            PROFILE_UPSCALE,
        }:
            raise RuntimeError(
                f"Unsupported profile: {profile}"
            )

        upscale_enabled = bool(
            production_plan.get(
                "upscale_enabled",
                profile
                == PROFILE_UPSCALE,
            )
        )

        checkpoint = (
            self._load_render_checkpoint(
                production_id
            )
        )

        checkpoint_status = str(
            (checkpoint or {}).get(
                "status",
                "",
            )
            or ""
        ).strip().lower()

        if checkpoint_status == "completed":
            final_video = str(
                (checkpoint or {}).get(
                    "final_video",
                    "",
                )
                or ""
            ).strip()

            if final_video:
                final_path = Path(
                    final_video
                ).resolve()

                if (
                    final_path.is_file()
                    and final_path.stat().st_size > 0
                ):
                    raise RuntimeError(
                        "Production is already completed. "
                        "Start a new production instead of "
                        "rendering the completed session: "
                        f"{production_id}"
                    )

        resume_active = checkpoint_status in {
            "rendering",
            "interrupted",
            "failed",
        }

        completed_shots = {
            str(shot_id): dict(
                record
            )
            for shot_id, record
            in (
                (checkpoint or {}).get(
                    "completed_shots",
                    {},
                )
                or {}
            ).items()
            if isinstance(
                record,
                dict,
            )
        }

        # Backward-compatible migration from checkpoints that only have IDs.
        legacy_ids = {
            str(value).strip()
            for value in (
                (checkpoint or {}).get(
                    "completed_shot_ids",
                    [],
                )
                or []
            )
            if str(value).strip()
        }

        for shot_id in legacy_ids:
            completed_shots.setdefault(
                shot_id,
                {},
            )

        all_planned_shot_ids = {
            str(
                shot.get(
                    "shot_id",
                    "",
                )
                or ""
            ).strip()
            for shot in shots
            if str(
                shot.get(
                    "shot_id",
                    "",
                )
                or ""
            ).strip()
        }

        completed_shots = {
            shot_id: record
            for shot_id, record
            in completed_shots.items()
            if shot_id in all_planned_shot_ids
        }

        # Establish the durable render state before workers start.
        self._update_render_checkpoint(
            production_id,
            status="rendering",
            stage="rendering",
            current_scene_id=str(
                (checkpoint or {}).get(
                    "current_scene_id",
                    "",
                )
                or ""
            ),
            completed_shot_ids=list(
                completed_shots.keys()
            ),
        )

        scenes: dict[
            str,
            list[dict],
        ] = {}

        for shot in shots:

            scene_id = str(
                shot.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()

            if not scene_id:
                raise RuntimeError(
                    "Shot is missing scene_id."
                )

            scenes.setdefault(
                scene_id,
                []
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

        scene_results = []

        try:

            if (
                not resume_active
                and bool(
                    production_plan.get(
                        "parallel_safe",
                        False,
                    )
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
                                upscale_enabled,
                                production_id,
                                completed_shots,
                            ),
                        ),
                    )
                )

            else:

                gpu_id = sorted(
                    self.clients
                )[0]

                for scene_id, scene_shots in (
                    scene_jobs
                ):

                    scene_results.append(
                        (
                            scene_id,
                            self._run_scene(
                                gpu_id,
                                scene_id,
                                scene_shots,
                                profile,
                                character_map,
                                upscale_enabled,
                                production_id,
                                completed_shots,
                            ),
                        )
                    )

                    self._update_render_checkpoint(
                        production_id,
                        status="rendering",
                        stage="rendering",
                        current_scene_id=scene_id,
                        completed_shot_ids=list(
                            completed_shots.keys()
                        ),
                    )

            # Re-read the checkpoint after all workers finish so the durable
            # merged state, rather than one worker's local dictionary, is
            # authoritative.
            final_checkpoint = (
                self._load_render_checkpoint(
                    production_id
                )
            )

            completed_shots = (
                self._completed_shot_records(
                    final_checkpoint
                )
            )

            # Every planned shot must have a durable record and a real file.
            videos = []

            for shot in self._sort_shots(
                shots
            ):

                shot_id = str(
                    shot.get(
                        "shot_id",
                        "",
                    )
                    or ""
                ).strip()

                record = (
                    completed_shots.get(
                        shot_id
                    )
                )

                if not isinstance(
                    record,
                    dict,
                ):
                    raise RuntimeError(
                        "Missing completed render record for shot: "
                        + shot_id
                    )

                output = str(
                    record.get(
                        "output",
                        "",
                    )
                    or ""
                ).strip()

                if not output:
                    raise RuntimeError(
                        "Completed render record has no output path for shot: "
                        + shot_id
                    )

                output_path = (
                    self._existing_shot_output(
                        Path(
                            output
                        )
                    )
                )

                if output_path is None:
                    raise RuntimeError(
                        "Completed render record points to a missing/invalid "
                        f"output for shot {shot_id}: {output}"
                    )

                videos.append(
                    output_path
                )

            if len(videos) != len(shots):
                raise RuntimeError(
                    "Rendered output count does not match "
                    f"planned shot count: {len(videos)} != {len(shots)}"
                )

            assembly_dir = (
                self.project_root
                / "data"
                / "production"
                / production_id
                / "final"
            )

            assembly_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            assembler = AssemblyManager(
                assembly_dir
            )

            final_video = (
                assembler.assemble(
                    videos,
                    final_name="final.mp4",
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
            )

            final_video = Path(
                final_video
            ).resolve()

            if (
                not final_video.is_file()
                or final_video.stat().st_size <= 0
            ):
                raise RuntimeError(
                    "Final video assembly produced no valid file."
                )

            self._update_render_checkpoint(
                production_id,
                status="completed",
                stage="render_complete",
                current_scene_id="",
                completed_shot_ids=sorted(
                    all_planned_shot_ids
                ),
                final_video=str(
                    final_video
                ),
                shot_outputs=[
                    str(path)
                    for path in videos
                ],
            )

            return {
                "production_id": production_id,
                "shot_outputs": videos,
                "final_video": final_video,
                "profile": profile,
                "upscale_enabled": upscale_enabled,
            }

        except Exception as exc:

            try:
                checkpoint_now = (
                    self._load_render_checkpoint(
                        production_id
                    )
                )

                persisted = (
                    self._completed_shot_records(
                        checkpoint_now
                    )
                )

                self._update_render_checkpoint(
                    production_id,
                    status="failed",
                    stage="rendering",
                    current_scene_id=str(
                        (checkpoint_now or {}).get(
                            "current_scene_id",
                            "",
                        )
                        or ""
                    ),
                    completed_shot_ids=list(
                        persisted.keys()
                    ),
                    error=str(
                        exc
                    ),
                )

            except Exception as checkpoint_error:

                print(
                    "[CHECKPOINT] render failure update failed:",
                    str(
                        checkpoint_error
                    ),
                    flush=True,
                )

            raise

    def _completed_shot_records(
        self,
        checkpoint: dict | None,
    ) -> dict[str, dict]:
        records = (
            (checkpoint or {}).get(
                "completed_shots",
                {},
            )
            or {}
        )

        if isinstance(
            records,
            dict,
        ):
            return {
                str(shot_id): dict(
                    record
                )
                for shot_id, record
                in records.items()
                if isinstance(
                    record,
                    dict,
                )
            }

        # Legacy checkpoints can still provide completed_shot_ids, but without
        # output paths those records are not enough for safe cross-GPU resume.
        return {}


from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from execution.shot_executor import (
    ShotExecutor,
)

from postprocess.final_export import (
    FinalExporter,
)

from postprocess.h3_regenerate_2k import (
    H3Regenerate2K,
)

from planner.config import (
    H3_REGENERATE_2K_ENABLED,
    WORKFLOW_AUTO,
    WORKFLOW_HARD_R2V,
    WORKFLOW_HARD_CHAINED,
    WORKFLOW_SEAMLESS_V2,
    WORKFLOW_SEAMLESS_CORE,
    WORKFLOW_KEYFRAMES,
    WORKFLOW_EXTEND_TAKE,
    WORKFLOW_TURBO_I2V,
    WORKFLOW_TURBO_REF2V,
    WORKFLOW_TURBO_T2V,
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

    @staticmethod
    def select_auto_workflow(
        shots,
        profile,
    ):

        if profile == "turbo":

            has_reference = any(
                (
                    shot.get(
                        "reference_images"
                    )
                    or shot.get(
                        "reference_videos"
                    )
                    or shot.get(
                        "reference_audio_paths"
                    )
                )
                for shot in shots
            )

            if has_reference:
                return WORKFLOW_TURBO_REF2V

            if any(
                shot.get(
                    "reference_images"
                )
                for shot in shots
            ):
                return WORKFLOW_TURBO_I2V

            return WORKFLOW_TURBO_T2V

        # Special modes explicitly requested by the planner.
        for shot in shots:

            if shot.get(
                "keyframe_images"
            ):
                return WORKFLOW_KEYFRAMES

            if shot.get(
                "extend_take_source_video"
            ):
                return WORKFLOW_EXTEND_TAKE

        if len(shots) == 1:
            return WORKFLOW_HARD_R2V

        # Hard Mode Chained is used for shorter continuous
        # scenes; Seamless Chain v2 is the long-form path.
        if len(shots) <= 8:
            return WORKFLOW_HARD_CHAINED

        return WORKFLOW_SEAMLESS_V2

    @staticmethod
    def character_reference_map(
        production_plan,
    ):

        result = {}

        for character in (
            production_plan.get(
                "characters",
                [],
            )
        ):

            name = str(
                character.get(
                    "name",
                    "",
                )
            ).strip()

            if not name:
                continue

            paths = list(
                character.get(
                    "reference_paths",
                    [],
                )
            )

            if paths:
                result[name] = paths[:3]

        return result

    def _executor(
        self,
        gpu_id,
        scene_id,
    ):

        input_dir = (
            self.input_root
            / f"gpu_{gpu_id}"
            / str(scene_id)
        )

        input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return ShotExecutor(
            comfy_client=self.clients[
                gpu_id
            ],
            project_root=self.project_root,
            comfy_input_dir=input_dir,
        )

    def run_scene(
        self,
        gpu_id,
        scene_id,
        shots,
        workflow_mode,
        profile,
        character_reference_map,
        production_plan,
    ):

        shots = sorted(
            shots,
            key=lambda item:
                int(
                    item.get(
                        "order",
                        0,
                    )
                )
        )

        if workflow_mode == WORKFLOW_AUTO:

            workflow_mode = (
                self.select_auto_workflow(
                    shots,
                    profile,
                )
            )

        if (
            workflow_mode
            in {
                WORKFLOW_HARD_R2V,
                WORKFLOW_KEYFRAMES,
                WORKFLOW_EXTEND_TAKE,
            }
            and len(shots) != 1
        ):
            if workflow_mode == WORKFLOW_HARD_R2V:
                raise RuntimeError(
                    "hard_r2v is a single-shot workflow. "
                    "Use hard_chained or seamless_v2 for "
                    "a multi-shot scene."
                )

            if workflow_mode == WORKFLOW_KEYFRAMES:
                raise RuntimeError(
                    "keyframes is a single-generation workflow. "
                    "Use it for one scene-generation pass."
                )

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

        width = int(
            production_plan.get(
                "width",
                1344,
            )
        )

        height = int(
            production_plan.get(
                "height",
                768,
            )
        )

        frames = int(
            production_plan.get(
                "frames_per_shot",
                243,
            )
        )

        steps = int(
            production_plan.get(
                "steps",
                14,
            )
        )

        result = (
            executor.execute_scene(
                scene_id=scene_id,
                shots=shots,
                workflow_mode=workflow_mode,
                profile=profile,
                character_reference_map=(
                    character_reference_map
                ),
                output_dir=output_dir,
                width=width,
                height=height,
                frames_per_shot=frames,
                steps=steps,
            )
        )

        return (
            scene_id,
            workflow_mode,
            result,
        )

    @staticmethod
    def concat(
        videos,
        destination,
    ):

        import subprocess

        if not videos:
            raise ValueError(
                "No videos supplied."
            )

        destination = Path(
            destination
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest = (
            destination.with_suffix(
                ".txt"
            )
        )

        lines = []

        for video in videos:

            path = (
                Path(video)
                .resolve()
            )

            escaped = str(
                path
            ).replace(
                "'",
                "'\\''",
            )

            lines.append(
                f"file '{escaped}'"
            )

        manifest.write_text(
            "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest),
                "-c",
                "copy",
                str(destination),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        manifest.unlink(
            missing_ok=True
        )

        if result.returncode != 0:
            raise RuntimeError(
                "FFmpeg concat failed:\n"
                + result.stderr[-5000:]
            )

        return destination

    def run(
        self,
        production_plan: dict[str, Any],
    ):

        shots = production_plan.get(
            "shots",
            [],
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

        workflow_mode = str(
            production_plan.get(
                "workflow_mode",
                WORKFLOW_AUTO,
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

            scenes.setdefault(
                scene_id,
                [],
            ).append(
                shot
            )

        character_reference_map = (
            self.character_reference_map(
                production_plan
            )
        )

        jobs = [
            SimpleNamespace(
                scene_id=scene_id,
                shots=scene_shots,
            )
            for scene_id, scene_shots
            in scenes.items()
        ]

        scheduler = GPUScheduler(
            gpu_ids=sorted(
                self.clients.keys()
            )
        )

        def worker(
            gpu_id,
            job,
        ):

            return self.run_scene(
                gpu_id=gpu_id,
                scene_id=job.scene_id,
                shots=job.shots,
                workflow_mode=workflow_mode,
                profile=profile,
                character_reference_map=(
                    character_reference_map
                ),
                production_plan=production_plan,
            )

        results = scheduler.run(
            jobs,
            worker,
        )

        # Narrative order.
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

        scene_videos = []

        for (
            _scene_id,
            _workflow_mode,
            path,
        ) in results:
            scene_videos.append(
                Path(path)
            )

        native_master = (
            self.output_root
            / "master_native.mp4"
        )

        self.concat(
            scene_videos,
            native_master,
        )

        master = native_master

        if H3_REGENERATE_2K_ENABLED:

            regenerated = (
                self.output_root
                / "master_h3_2k.mp4"
            )

            master = (
                H3Regenerate2K().regenerate(
                    source_video=native_master,
                    destination=regenerated,
                    prompt=production_plan.get(
                        "story",
                        "",
                    ),
                )
            )

        final = (
            self.project_root
            / "data"
            / "production"
            / "final_h3_720p.mp4"
        )

        return FinalExporter.export_720p(
            source=master,
            destination=final,
        )

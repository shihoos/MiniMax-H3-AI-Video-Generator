from __future__ import annotations

import shutil
from pathlib import Path

from planner.config import (
    H3_HEIGHT,
    H3_MAX_REFERENCE_AUDIO,
    H3_MAX_REFERENCE_FILES,
    H3_MAX_REFERENCE_IMAGES,
    H3_MAX_REFERENCE_VIDEOS,
    H3_WIDTH,
    TURBO_STEPS,
)


class ShotExecutor:

    MAX_IMAGES = H3_MAX_REFERENCE_IMAGES
    MAX_VIDEOS = H3_MAX_REFERENCE_VIDEOS
    MAX_AUDIO = H3_MAX_REFERENCE_AUDIO
    MAX_TOTAL_REFERENCES = H3_MAX_REFERENCE_FILES

    def __init__(
        self,
        comfy_client,
        project_root,
        comfy_input_dir,
    ):

        from execution.h3_workflow_builder import (
            H3WorkflowBuilder,
        )

        from execution.h3_upscaled_workflow_builder import (
            H3UpscaledWorkflowBuilder,
        )

        self.client = comfy_client

        self.project_root = Path(
            project_root
        ).resolve()

        self.comfy_input_root = (
            self.project_root
            / "ComfyUI"
            / "input"
        ).resolve()

        self.comfy_input_dir = Path(
            comfy_input_dir
        ).resolve()

        try:
            self.comfy_input_dir.relative_to(
                self.comfy_input_root
            )
        except ValueError as exc:
            raise ValueError(
                "ComfyUI input directory must be inside "
                f"{self.comfy_input_root}: "
                f"{self.comfy_input_dir}"
            ) from exc

        self.comfy_input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.builder = H3WorkflowBuilder(
            project_root=self.project_root,
            comfy_client=self.client,
        )

        self.upscaled_builder = (
            H3UpscaledWorkflowBuilder(
                project_root=self.project_root,
                comfy_client=self.client,
            )
        )

    @staticmethod
    def _safe_name(
        value,
    ) -> str:

        return "".join(
            char
            if (
                char.isalnum()
                or char in "._-"
            )
            else "_"
            for char in str(value)
        )

    def copy_input(
        self,
        source,
        prefix: str,
    ) -> str:

        source = Path(
            source
        ).resolve()

        if not source.is_file():
            raise FileNotFoundError(
                f"Media input does not exist:\n{source}"
            )

        destination = (
            self.comfy_input_dir
            / (
                f"{prefix}_"
                f"{self._safe_name(source.name)}"
            )
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        if (
            not destination.is_file()
            or destination.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Copied media is missing or empty:\n"
                f"{destination}"
            )

        try:
            relative = (
                destination.relative_to(
                    self.comfy_input_root
                )
            )
        except ValueError as exc:
            raise RuntimeError(
                "Copied media escaped the ComfyUI input root:\n"
                f"{destination}"
            ) from exc

        # IMPORTANT:
        # ComfyUI LoadImage / VHS loaders resolve files
        # relative to ComfyUI/input. Never return only
        # destination.name when the file lives in a subdirectory.
        return relative.as_posix()

    def _prepare_media(
        self,
        shot,
    ):

        images = list(
            shot.get(
                "reference_images",
                [],
            )
            or []
        )

        videos = list(
            shot.get(
                "reference_videos",
                [],
            )
            or []
        )

        audio = list(
            shot.get(
                "reference_audio_paths",
                [],
            )
            or []
        )

        if len(images) > self.MAX_IMAGES:
            raise RuntimeError(
                f"{shot.get('shot_id')}: "
                "maximum 9 reference images."
            )

        if len(videos) > self.MAX_VIDEOS:
            raise RuntimeError(
                f"{shot.get('shot_id')}: "
                "maximum 3 reference videos."
            )

        if len(audio) > self.MAX_AUDIO:
            raise RuntimeError(
                f"{shot.get('shot_id')}: "
                "maximum 3 reference audio clips."
            )

        if (
            len(images)
            + len(videos)
            + len(audio)
            > self.MAX_TOTAL_REFERENCES
        ):
            raise RuntimeError(
                f"{shot.get('shot_id')}: "
                "maximum 12 reference files."
            )

        copied_images = [
            self.copy_input(
                value,
                f"{shot['shot_id']}_image_{index + 1}",
            )
            for index, value in enumerate(
                images
            )
        ]

        copied_videos = [
            self.copy_input(
                value,
                f"{shot['shot_id']}_video_{index + 1}",
            )
            for index, value in enumerate(
                videos
            )
        ]

        copied_audio = [
            self.copy_input(
                value,
                f"{shot['shot_id']}_audio_{index + 1}",
            )
            for index, value in enumerate(
                audio
            )
        ]

        return (
            copied_images,
            copied_videos,
            copied_audio,
        )

    @staticmethod
    def _number(
        value,
        default,
        cast,
    ):

        if value is None:
            return default

        try:
            return cast(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return default

    def execute_shot(
        self,
        *,
        shot,
        workflow_mode,
        output_dir,
        upscale=False,
    ):

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        prompt = (
            shot.get(
                "h3_prompt",
                "",
            )
            or shot.get(
                "visual_prompt",
                "",
            )
        ).strip()

        if not prompt:
            raise RuntimeError(
                f"{shot.get('shot_id')}: empty H3 prompt."
            )

        seed = self._number(
            shot.get(
                "seed"
            ),
            135791113,
            int,
        )

        width = self._number(
            shot.get(
                "width"
            ),
            H3_WIDTH,
            int,
        )

        height = self._number(
            shot.get(
                "height"
            ),
            H3_HEIGHT,
            int,
        )

        duration = self._number(
            shot.get(
                "duration_seconds"
            ),
            5.2,
            float,
        )

        (
            images,
            videos,
            audio,
        ) = self._prepare_media(
            shot
        )

        if upscale:

            if workflow_mode not in {
                "ref2v",
                "turbo_ref2v",
            }:
                raise RuntimeError(
                    "Combined upscale requires "
                    "ref2v or turbo_ref2v generation mode."
                )

            workflow = (
                self.upscaled_builder.build_upscaled(
                    generation_mode=workflow_mode,
                    prompt=prompt,
                    seed=seed,
                    turbo_steps=TURBO_STEPS,
                    reference_images=images,
                    reference_videos=videos,
                    reference_audio=audio,
                    width=width,
                    height=height,
                    duration_seconds=duration,
                )
            )

        else:

            workflow = self.builder.build(
                mode=workflow_mode,
                prompt=prompt,
                seed=seed,
                turbo_steps=TURBO_STEPS,
                reference_images=images,
                reference_videos=videos,
                reference_audio=audio,
                width=width,
                height=height,
                duration_seconds=duration,
            )

        prompt_id = (
            self.client.queue_prompt(
                workflow
            )
        )

        history = (
            self.client.wait_for_prompt(
                prompt_id,
                timeout=14400,
            )
        )

        outputs = (
            self.client.find_video_outputs(
                history
            )
        )

        if not outputs:
            raise RuntimeError(
                f"No video output for "
                f"{shot.get('shot_id')}"
            )

        output = outputs[-1]

        destination = (
            output_dir
            / f"{shot['shot_id']}.mp4"
        )

        return self.client.download_file(
            filename=output[
                "filename"
            ],
            subfolder=output[
                "subfolder"
            ],
            file_type=output[
                "type"
            ],
            destination=destination,
        )

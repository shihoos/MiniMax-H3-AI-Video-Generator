from __future__ import annotations

import shutil
from pathlib import Path


class ShotExecutor:

    MAX_IMAGES = 9
    MAX_VIDEOS = 3
    MAX_AUDIO = 3

    def __init__(
        self,
        comfy_client,
        project_root,
        comfy_input_dir,
    ):
        from execution.h3_workflow_builder import (
            H3WorkflowBuilder,
        )

        self.client = comfy_client
        self.project_root = Path(
            project_root
        )
        self.comfy_input_dir = Path(
            comfy_input_dir
        )

        self.comfy_input_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.builder = H3WorkflowBuilder(
            project_root=self.project_root,
            comfy_client=self.client,
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
        )

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

        shutil.copy2(
            source,
            destination,
        )

        return destination.name

    def _prepare_media(
        self,
        shot,
    ):
        raw_images = (
            list(
                shot.get(
                    "reference_images",
                    [],
                )
                or []
            )
        )

        raw_videos = (
            list(
                shot.get(
                    "reference_videos",
                    [],
                )
                or []
            )
        )

        raw_audio = (
            list(
                shot.get(
                    "reference_audio_paths",
                    [],
                )
                or []
            )
        )

        if len(raw_images) > self.MAX_IMAGES:
            raise RuntimeError(
                f"{shot.get('shot_id')}: "
                "maximum 9 reference images."
            )

        if len(raw_videos) > self.MAX_VIDEOS:
            raise RuntimeError(
                f"{shot.get('shot_id')}: "
                "maximum 3 reference videos."
            )

        if len(raw_audio) > self.MAX_AUDIO:
            raise RuntimeError(
                f"{shot.get('shot_id')}: "
                "maximum 3 reference audio clips."
            )

        if (
            len(raw_images)
            + len(raw_videos)
            + len(raw_audio)
            > 12
        ):
            raise RuntimeError(
                f"{shot.get('shot_id')}: "
                "maximum 12 reference files."
            )

        images = [
            self.copy_input(
                value,
                f"{shot['shot_id']}_image_{i + 1}",
            )
            for i, value in enumerate(
                raw_images
            )
        ]

        videos = [
            self.copy_input(
                value,
                f"{shot['shot_id']}_video_{i + 1}",
            )
            for i, value in enumerate(
                raw_videos
            )
        ]

        audio = [
            self.copy_input(
                value,
                f"{shot['shot_id']}_audio_{i + 1}",
            )
            for i, value in enumerate(
                raw_audio
            )
        ]

        return images, videos, audio

    @staticmethod
    def _number(
        value,
        default,
        cast,
    ):
        if value is None:
            return default

        try:
            return cast(value)
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
                f"{shot.get('shot_id')}: "
                "empty H3 prompt."
            )

        seed = self._number(
            shot.get("seed"),
            135791113,
            int,
        )

        width = self._number(
            shot.get("width"),
            1344,
            int,
        )

        height = self._number(
            shot.get("height"),
            768,
            int,
        )

        duration = self._number(
            shot.get("duration_seconds"),
            5.2,
            float,
        )

        images, videos, audio = (
            self._prepare_media(
                shot
            )
        )

        workflow = self.builder.build(
            mode=workflow_mode,
            prompt=prompt,
            seed=seed,
            turbo_steps=8,
            reference_images=images,
            reference_videos=videos,
            reference_audio=audio,
            width=width,
            height=height,
            duration_seconds=duration,
        )

        prompt_id = self.client.queue_prompt(
            workflow
        )

        history = self.client.wait_for_prompt(
            prompt_id,
            timeout=14400,
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
            filename=output["filename"],
            subfolder=output["subfolder"],
            file_type=output["type"],
            destination=destination,
        )

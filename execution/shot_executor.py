from __future__ import annotations

import shutil
from pathlib import Path


class ShotExecutor:

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
    def _safe_name(value) -> str:

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

        source = Path(source)

        if not source.is_file():
            raise FileNotFoundError(
                f"Reference input does not exist:\n{source}"
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

    def _prepare_reference_images(
        self,
        shot,
    ) -> list[str]:

        raw = shot.get(
            "reference_images",
            [],
        )

        if raw is None:
            return []

        if not isinstance(raw, list):
            raise RuntimeError(
                f"{shot.get('shot_id')}: "
                "reference_images must be a list."
            )

        prepared: list[str] = []

        for index, value in enumerate(
            raw[:9]
        ):

            if value is None:
                continue

            source = Path(
                str(value)
            )

            if not source.is_file():
                raise FileNotFoundError(
                    f"{shot.get('shot_id')}: "
                    f"reference image does not exist:\n"
                    f"{source}"
                )

            filename = self.copy_input(
                source,
                prefix=(
                    f"{shot.get('shot_id', 'shot')}"
                    f"_ref{index + 1}"
                ),
            )

            prepared.append(
                filename
            )

        return prepared

    @staticmethod
    def _float_or_none(
        value,
    ):

        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_or_none(
        value,
    ):

        if value is None:
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

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

        seed = int(
            shot.get(
                "seed",
                135791113,
            )
        )

        width = self._int_or_none(
            shot.get("width")
        )

        height = self._int_or_none(
            shot.get("height")
        )

        duration = self._float_or_none(
            shot.get("duration_seconds")
        )

        if duration is None:
            duration = self._float_or_none(
                shot.get("duration")
            )

        reference_images = (
            self._prepare_reference_images(
                shot
            )
        )

        workflow = self.builder.build(
            mode=workflow_mode,
            prompt=prompt,
            seed=seed,
            turbo_steps=8,
            reference_images=reference_images,
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

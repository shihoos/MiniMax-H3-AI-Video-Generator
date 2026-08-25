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
            H3WorkflowBuilder
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

        self.builder = (
            H3WorkflowBuilder(
                self.project_root,
                self.client,
            )
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
                source
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

        workflow = (
            self.builder.build(
                mode=workflow_mode,
                prompt=prompt,
                seed=seed,
            )
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
            filename=output["filename"],
            subfolder=output["subfolder"],
            file_type=output["type"],
            destination=destination,
        )

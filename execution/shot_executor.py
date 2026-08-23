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

        self.builder = (
            H3WorkflowBuilder(
                self.project_root,
                comfy_client=self.client,
            )
        )

    @staticmethod
    def _safe_name(
        value,
    ):

        return "".join(
            char
            if (
                char.isalnum()
                or char in "._-"
            )
            else "_"
            for char in str(value)
        )

    def _copy(
        self,
        source,
        prefix,
    ):

        source = Path(
            source
        )

        if not source.is_file():
            raise FileNotFoundError(
                f"Missing reference media: {source}"
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

    def _copy_images(
        self,
        paths,
    ):

        return [
            self._copy(
                path,
                "reference_image",
            )
            for path in paths[:9]
        ]

    def _copy_videos(
        self,
        paths,
    ):

        return [
            self._copy(
                path,
                "reference_video",
            )
            for path in paths[:3]
        ]

    def _copy_audio(
        self,
        paths,
    ):

        return [
            self._copy(
                path,
                "reference_audio",
            )
            for path in paths[:3]
        ]

    def _prepare_auto_refs(
        self,
        scene_id,
        character_reference_map,
    ) -> str:

        root = (
            self.comfy_input_dir
            / "h3_refs"
            / self._safe_name(
                scene_id
            )
        )

        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        used = 0

        for character_name, paths in (
            character_reference_map.items()
        ):

            if used >= 9:
                break

            character_dir = (
                root
                / self._safe_name(
                    character_name
                )
            )

            character_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for index, path in enumerate(
                paths[:3],
                start=1,
            ):

                if used >= 9:
                    break

                source = Path(
                    path
                )

                if not source.is_file():
                    continue

                destination = (
                    character_dir
                    / (
                        f"{index:02d}_"
                        f"{self._safe_name(source.name)}"
                    )
                )

                shutil.copy2(
                    source,
                    destination,
                )

                used += 1

        # ComfyUI's input-relative path.
        relative = (
            root.relative_to(
                self.project_root
                / "ComfyUI"
                / "input"
            )
        )

        return str(
            relative
        ).replace(
            "\\",
            "/",
        )

    @staticmethod
    def _seed(
        shots,
    ) -> int:

        for shot in shots:
            if shot.get("seed") is not None:
                return int(
                    shot["seed"]
                )

        # Fixed fallback.
        return 135791113

    @staticmethod
    def build_script(
        shots,
    ) -> str:

        ordered = sorted(
            shots,
            key=lambda item: int(
                item.get(
                    "order",
                    0,
                )
            ),
        )

        prompts = []

        for shot in ordered:

            prompt = str(
                shot.get(
                    "h3_prompt",
                    "",
                )
                or
                shot.get(
                    "visual_prompt",
                    "",
                )
            ).strip()

            if not prompt:
                raise RuntimeError(
                    f"{shot.get('shot_id')}: "
                    "empty H3 prompt."
                )

            prompts.append(
                prompt
            )

        return "\n---\n".join(
            prompts
        )

    def execute_scene(
        self,
        *,
        scene_id,
        shots,
        workflow_mode,
        profile,
        character_reference_map,
        output_dir,
        width,
        height,
        frames_per_shot,
        steps,
    ):

        if not shots:
            raise ValueError(
                "Cannot execute empty scene."
            )

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_paths = []
        video_paths = []
        audio_paths = []
        keyframe_paths = []

        for shot in shots:

            image_paths.extend(
                shot.get(
                    "reference_images",
                    [],
                )
            )

            video_paths.extend(
                shot.get(
                    "reference_videos",
                    [],
                )
            )

            audio_paths.extend(
                shot.get(
                    "reference_audio_paths",
                    [],
                )
            )

            keyframe_paths.extend(
                shot.get(
                    "keyframe_images",
                    [],
                )
            )

        image_paths = list(
            dict.fromkeys(
                image_paths
            )
        )[:9]

        video_paths = list(
            dict.fromkeys(
                video_paths
            )
        )[:3]

        audio_paths = list(
            dict.fromkeys(
                audio_paths
            )
        )[:3]

        keyframe_paths = list(
            dict.fromkeys(
                keyframe_paths
            )
        )

        image_files = (
            self._copy_images(
                image_paths
            )
        )

        video_files = (
            self._copy_videos(
                video_paths
            )
        )

        audio_files = (
            self._copy_audio(
                audio_paths
            )
        )

        keyframe_files = (
            self._copy_images(
                keyframe_paths
            )
        )

        refs_root = (
            self._prepare_auto_refs(
                scene_id,
                character_reference_map,
            )
        )

        script = self.build_script(
            shots
        )

        seed = self._seed(
            shots
        )

        workflow = (
            self.builder.build(
                mode=workflow_mode,
                profile=profile,
                script=script,
                shot_count=len(shots),
                width=width,
                height=height,
                frames_per_shot=frames_per_shot,
                steps=steps,
                seed=seed,
                image_files=image_files,
                video_files=video_files,
                audio_files=audio_files,
                keyframe_files=keyframe_files,
                refs_root=refs_root,
                output_prefix=(
                    f"h3/{scene_id}/master"
                ),
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
                f"No video output for scene "
                f"{scene_id}."
            )

        result = outputs[-1]

        destination = (
            output_dir
            / f"{scene_id}.mp4"
        )

        return self.client.download_file(
            filename=result[
                "filename"
            ],
            subfolder=result[
                "subfolder"
            ],
            file_type=result[
                "type"
            ],
            destination=destination,
        )

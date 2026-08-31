from __future__ import annotations

import os
import shutil
import time
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
        gpu_id: int | None = None,
        metrics_path: Path | None = None,
    ):

        from execution.h3_workflow_builder import (
            H3WorkflowBuilder,
        )

        from execution.h3_upscaled_workflow_builder import (
            H3UpscaledWorkflowBuilder,
        )

        self.client = comfy_client
        self.gpu_id = int(gpu_id) if gpu_id is not None else -1
        self.metrics = None
        if metrics_path is not None:
            from execution.metrics import MetricsRecorder
            self.metrics = MetricsRecorder(Path(metrics_path))

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

        # Reference media is immutable during a shot. Prefer a symlink so a
        # large image/video/audio asset is not physically copied into
        # ComfyUI/input for every shot. Fall back to a hard link, then a real
        # copy on filesystems that do not support links.
        if destination.exists() or destination.is_symlink():
            destination.unlink()

        linked = False
        try:
            destination.symlink_to(source)
            linked = True
        except (OSError, NotImplementedError):
            try:
                os.link(source, destination)
                linked = True
            except OSError:
                shutil.copy2(source, destination)

        if (
            not destination.is_file()
            or destination.stat().st_size <= 0
        ):
            raise RuntimeError(
                f"Prepared media is missing or empty:\n"
                f"{destination}"
            )

        if linked and destination.is_symlink():
            resolved_target = destination.resolve()
            if resolved_target != source:
                raise RuntimeError(
                    "Reference symlink resolved to an unexpected target:\n"
                    f"{resolved_target}"
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

    @staticmethod
    def _resolve_ref_image_size(shot: dict) -> str:
        explicit = str(shot.get("ref_image_size", "") or "").strip().lower()
        if explicit in {"match", "max"}:
            return explicit
        camera = " ".join(
            str(shot.get(key, "") or "")
            for key in ("camera_shot", "shot_type", "framing")
        ).lower()
        close_tokens = ("extreme close", "extreme-close", "ecu", "close-up", "close up", "closeup")
        return "max" if any(token in camera for token in close_tokens) else "match"

    @staticmethod
    def _select_savevideo_output(
        workflow: dict,
        history: dict,
        *,
        shot_id: str,
    ) -> dict:
        """Select the video artifact emitted by the workflow's SaveVideo node.

        Never rely on list ordering from ComfyUI history: multiple nodes can
        produce video artifacts and their ordering is not a stable API.
        """
        if not isinstance(workflow, dict):
            raise RuntimeError(f"{shot_id}: submitted workflow is not a mapping.")
        if not isinstance(history, dict):
            raise RuntimeError(f"{shot_id}: ComfyUI history is not a mapping.")

        save_nodes = []
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == "SaveVideo":
                save_nodes.append(str(node_id))

        if len(save_nodes) != 1:
            raise RuntimeError(
                f"{shot_id}: expected exactly one SaveVideo node in the submitted "
                f"workflow, found {len(save_nodes)}."
            )

        save_node_id = save_nodes[0]
        node_history = history.get("outputs", {}).get(save_node_id)
        if not isinstance(node_history, dict):
            raise RuntimeError(
                f"{shot_id}: ComfyUI history has no output record for SaveVideo "
                f"node {save_node_id}."
            )

        candidates = []
        for output_list in node_history.values():
            if not isinstance(output_list, list):
                continue
            for item in output_list:
                if not isinstance(item, dict):
                    continue
                filename = str(item.get("filename") or "").strip()
                if filename and Path(filename).suffix.lower() in {
                    ".mp4", ".mov", ".mkv", ".webm"
                }:
                    candidates.append({
                        "filename": filename,
                        "subfolder": str(item.get("subfolder") or ""),
                        "type": str(item.get("type") or "output"),
                    })

        if len(candidates) != 1:
            raise RuntimeError(
                f"{shot_id}: expected exactly one video artifact from SaveVideo "
                f"node {save_node_id}, found {len(candidates)}."
            )

        return candidates[0]

    @staticmethod
    def _is_oom_error(error: BaseException) -> bool:
        text = str(error).lower()
        return any(token in text for token in (
            "out of memory",
            "cuda oom",
            "cuda outofmemory",
            "outofmemoryerror",
            "cublas_status_alloc_failed",
            "not enough memory",
        ))

    def _record(self, event: str, **fields) -> None:
        if self.metrics is not None:
            self.metrics.record(event, gpu_id=(None if self.gpu_id < 0 else self.gpu_id), **fields)

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
                    ref_image_size=self._resolve_ref_image_size(shot),
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
                ref_image_size=self._resolve_ref_image_size(shot),
            )

        shot_id = str(shot.get("shot_id") or "")
        max_oom_retries = max(0, int(os.getenv("H3_OOM_MAX_RETRIES", "1")))
        oom_retries = 0
        started = time.monotonic()

        while True:
            attempt_started = time.monotonic()
            self._record("shot_attempt_started", shot_id=shot_id, workflow_mode=workflow_mode, upscale=bool(upscale), attempt=oom_retries + 1)
            try:
                prompt_id = self.client.queue_prompt(workflow)
                queued_at = time.monotonic()
                self._record("shot_queued", shot_id=shot_id, prompt_id=prompt_id, queue_submit_seconds=queued_at - attempt_started, attempt=oom_retries + 1)

                history = self.client.wait_for_prompt(
                    prompt_id,
                    poll_interval=float(os.getenv("H3_COMFY_POLL_INTERVAL", "2")),
                    timeout=float(os.getenv("H3_COMFY_JOB_TIMEOUT", "14400")),
                )
                output = self._select_savevideo_output(
                    workflow,
                    history,
                    shot_id=shot_id,
                )
                destination = output_dir / f"{shot['shot_id']}.mp4"
                result = self.client.download_file(
                    filename=output["filename"],
                    subfolder=output["subfolder"],
                    file_type=output["type"],
                    destination=destination,
                )
                self._record(
                    "shot_completed",
                    shot_id=shot_id,
                    prompt_id=prompt_id,
                    attempt=oom_retries + 1,
                    oom_retries=oom_retries,
                    total_seconds=time.monotonic() - started,
                    output_bytes=Path(result).stat().st_size,
                )
                return result
            except Exception as error:
                self._record(
                    "shot_failed_attempt",
                    shot_id=shot_id,
                    attempt=oom_retries + 1,
                    oom=self._is_oom_error(error),
                    error_type=type(error).__name__,
                    error=str(error)[-2000:],
                    attempt_seconds=time.monotonic() - attempt_started,
                )
                if not self._is_oom_error(error) or oom_retries >= max_oom_retries:
                    raise
                oom_retries += 1
                self._record("shot_oom_recovery", shot_id=shot_id, attempt=oom_retries + 1)
                try:
                    self.client.free_memory(unload_models=False, free_memory=True)
                except Exception as recovery_error:
                    self._record("shot_oom_recovery_free_failed", shot_id=shot_id, error=str(recovery_error)[-1000:])
                time.sleep(min(5.0, 1.0 * (2 ** (oom_retries - 1))))



    def validate_rendered_media(self, output_path):
        """Return stream-level A/V validation for a completed shot."""
        from pipeline.dialogue_duration import FFProbeMediaDurationProvider
        return FFProbeMediaDurationProvider().validate_video_audio_sync(output_path)

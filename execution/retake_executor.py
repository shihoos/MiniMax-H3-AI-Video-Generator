from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline.context_ir import H3ContextIRCompiler
from pipeline.dialogue_duration import FFProbeMediaDurationProvider
from pipeline.h3_scene_continuity import H3SceneContinuity
from pipeline.retake_manager import RetakeManager
from pipeline.seed_lineage import semantic_content_digest, stable_seed


class RetakeExecutor:
    """Execute a bounded selective retake using the current H3 Ref2VA production path.

    H3's official FL2VA task is the ideal primitive for a boundary-anchored retake,
    but this repository deliberately has an exact Ref2VA production inventory. Until
    an FL2VA checkpoint/workflow is explicitly added to that inventory, this executor
    uses two extracted boundary frames as highest-priority Ref2VA references and then
    stitches the replacement back into the existing shot.
    """

    MIN_SECONDS = 4.0
    MAX_SECONDS = 15.0

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()
        self.retake_manager = RetakeManager(self.project_root)
        self.continuity = H3SceneContinuity(self.project_root)
        self.probe = FFProbeMediaDurationProvider()

    @staticmethod
    def _normalize_range(base_duration: float, start: float, end: float) -> tuple[float, float]:
        start = max(0.0, min(float(start), base_duration))
        end = max(start, min(float(end), base_duration))
        requested = end - start
        if requested >= RetakeExecutor.MIN_SECONDS:
            return start, min(end, start + RetakeExecutor.MAX_SECONDS)
        # H3 requires at least 4 seconds. Expand around the requested center while
        # staying within the existing shot. This guarantees a legal replacement.
        center = (start + end) / 2.0
        half = RetakeExecutor.MIN_SECONDS / 2.0
        expanded_start = max(0.0, center - half)
        expanded_end = min(base_duration, expanded_start + RetakeExecutor.MIN_SECONDS)
        if expanded_end - expanded_start < RetakeExecutor.MIN_SECONDS:
            expanded_start = max(0.0, base_duration - RetakeExecutor.MIN_SECONDS)
            expanded_end = base_duration
        return expanded_start, expanded_end

    def execute(
        self,
        *,
        production_id: str,
        scene_id: str,
        shot: dict[str, Any],
        base_video: Path,
        start_seconds: float,
        end_seconds: float,
        shot_executor,
        workflow_mode: str,
        upscale: bool,
    ) -> dict[str, Any]:
        base_video = Path(base_video).resolve()
        if not base_video.is_file():
            raise FileNotFoundError(base_video)
        base_duration = self.probe.duration_seconds(base_video, stream_selector="v:0")
        start, end = self._normalize_range(base_duration, start_seconds, end_seconds)
        duration = end - start
        if not (self.MIN_SECONDS <= duration <= self.MAX_SECONDS):
            raise RuntimeError(f"Retake range is outside H3 limits after normalization: {duration:.3f}s")

        request_path = self.retake_manager.request(
            production_id,
            str(shot["shot_id"]),
            start_seconds=start,
            end_seconds=end,
            reason=str(shot.get("retake_reason", "Automatic quality-gate retake") or "Automatic quality-gate retake"),
            preserve_audio=True,
        )

        first_frame = self.continuity.extract_frame_at(
            base_video, start, scene_id=scene_id, shot_id=str(shot["shot_id"]), label="retake_start"
        )
        end_frame_time = min(max(0.0, end), max(0.0, base_duration - 1.0 / 24.0))
        last_frame = self.continuity.extract_frame_at(
            base_video, end_frame_time, scene_id=scene_id, shot_id=str(shot["shot_id"]), label="retake_end"
        )

        replacement = dict(shot)
        original_refs = list(replacement.get("reference_images", []) or [])
        original_roles = [dict(x) for x in (replacement.get("reference_roles", []) or []) if isinstance(x, dict)]
        original_role_map = {str(x.get("path", "")): x for x in original_roles if str(x.get("path", "")).strip()}
        boundary_items = [
            (
                str(first_frame),
                {
                    "path": str(first_frame),
                    "media_type": "picture",
                    "role": "retake_start_frame",
                    "relationship": "fully_preserved",
                    "priority": 200,
                    "label": "Exact base-shot frame at the retake start; preserve the visual state entering the replacement.",
                },
            ),
            (
                str(last_frame),
                {
                    "path": str(last_frame),
                    "media_type": "picture",
                    "role": "retake_end_frame",
                    "relationship": "fully_preserved",
                    "priority": 199,
                    "label": "Exact base-shot frame at the retake end; preserve the visual state leaving the replacement.",
                },
            ),
        ]
        remaining = []
        for path in original_refs:
            p = str(path)
            if p in {str(first_frame), str(last_frame)}:
                continue
            remaining.append((p, dict(original_role_map.get(p, {"path": p, "role": "visual_reference", "priority": 50}))))
        ordered = boundary_items + remaining
        ordered = ordered[:9]
        replacement["reference_images"] = [p for p, _ in ordered]
        replacement["reference_roles"] = [r for _, r in ordered]
        replacement["reference_bindings"] = [
            f"<Picture {i}> = {r.get('label', r.get('role', 'visual reference'))}"
            for i, (_, r) in enumerate(ordered, start=1)
        ]
        base_prompt = str(replacement.get("visual_prompt", replacement.get("action", "")) or "").strip()
        replacement["context_ir_instruction"] = (
            "Selective retake: Picture 1 is the exact frame at the start boundary and Picture 2 is "
            "the exact frame at the end boundary. Recreate only the requested action inside this range "
            "while preserving character identity, wardrobe, environment, lighting, camera continuity, "
            "and all non-target story state. Do not invent a new scene. "
            + base_prompt
        ).strip()
        replacement["duration_seconds"] = duration
        replacement["seed"] = stable_seed(f"{production_id}:retake", replacement)
        replacement["semantic_content_digest"] = semantic_content_digest(replacement)
        replacement["shot_id"] = f"{shot['shot_id']}__retake"
        replacement["retake_source_shot_id"] = str(shot["shot_id"])
        replacement["retake_start_seconds"] = start
        replacement["retake_end_seconds"] = end
        replacement["retake_request_path"] = str(request_path)
        replacement_context_ir = H3ContextIRCompiler().compile(
            {"production_id": production_id, "story": str(shot.get("story", "") or "")},
            replacement,
        )
        replacement["h3_context_ir"] = replacement_context_ir

        out_dir = self.project_root / "data" / "production" / str(production_id) / "retakes" / str(shot["shot_id"])
        out_dir.mkdir(parents=True, exist_ok=True)
        replacement_video = shot_executor.execute_shot(
            shot=replacement,
            workflow_mode=workflow_mode,
            output_dir=out_dir,
            upscale=upscale,
            context_ir=replacement_context_ir,
        )
        replacement_video = Path(replacement_video).resolve()
        retake_duration = self.probe.duration_seconds(replacement_video, stream_selector="v:0")
        if abs(retake_duration - duration) > 0.45:
            raise RuntimeError(
                f"Retake duration mismatch: requested {duration:.3f}s, generated {retake_duration:.3f}s."
            )

        stitched = out_dir / f"{shot['shot_id']}__replaced.mp4"
        self.retake_manager.stitch(
            base_video,
            replacement_video,
            stitched,
            start_seconds=start,
            end_seconds=end,
            preserve_audio=True,
        )
        self.retake_manager.mark_completed(request_path, stitched, replacement_video)
        return {
            "output": stitched.resolve(),
            "replacement_video": replacement_video,
            "start_seconds": start,
            "end_seconds": end,
            "request_path": request_path,
        }

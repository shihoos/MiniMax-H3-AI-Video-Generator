from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pipeline.production_checkpoint import ProductionCheckpoint


class ProductionManifest:
    """Immutable-ish audit artifact describing exactly what produced a film."""

    VERSION = 2

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    @staticmethod
    def _file_hash(path: Path) -> str:
        if not path.is_file():
            return ""
        return ProductionCheckpoint.digest_file(path)

    def build(self, plan: dict[str, Any]) -> dict[str, Any]:
        files = {}
        for rel in (
            "configs/runtime_versions.yaml",
            "configs/model_inventory.yaml",
            "configs/custom_nodes.yaml",
            "execution/retake_executor.py",
            "planner/qwen_director.py",
            "planner/cinematic_compiler.py",
            "planner/production_planner.py",
            "execution/h3_workflow_builder.py",
            "execution/h3_upscaled_workflow_builder.py",
            "execution/production_runner.py",
            "execution/shot_executor.py",
            "execution/execution_policy.py",
            "pipeline/timeline.py",
            "pipeline/context_ir.py",
            "pipeline/vlm_analyzer.py",
            "pipeline/quality_gate.py",
            "pipeline/retake_manager.py",
            "pipeline/runtime_diagnostics.py",
            "pipeline/comfy_preview.py",
            "pipeline/visual_feedback.py",
            "pipeline/visual_state_observer.py",
            "pipeline/production_checkpoint.py",
            "ui/storyboard_gradio.py",
            "ui/shot_view_model.py",
        ):
            files[rel] = self._file_hash(self.project_root / rel)
        manifest = {
            "version": self.VERSION,
            "production_id": str(plan.get("production_id", "")),
            "plan_sha256": ProductionCheckpoint.plan_digest(plan),
            "story_sha256": ProductionCheckpoint.digest_text(str(plan.get("story", "") or "")),
            "director_notes_sha256": ProductionCheckpoint.digest_text(str(plan.get("director_notes", "") or "")),
            "files": files,
            "models": plan.get("model_manifest", plan.get("models", {})) or {},
            "runtime": plan.get("runtime_diagnostics", {}) or {},
            "timeline_version": (plan.get("timeline", {}) or {}).get("version", 1),
            "execution": {
                "mode": str(plan.get("execution_mode", "production") or "production"),
                "context_ir_version": (plan.get("features", {}) or {}).get("context_ir_version", 2),
                "profile": str(plan.get("profile", "base") or "base"),
            },
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return manifest

    def write(self, plan: dict[str, Any], path: Path) -> dict[str, Any]:
        manifest = self.build(plan)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return manifest

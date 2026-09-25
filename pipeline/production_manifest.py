from __future__ import annotations

import hashlib
import json
import os
import tempfile

import yaml
from pathlib import Path
from typing import Any

from pipeline.production_checkpoint import ProductionCheckpoint


class ProductionManifest:
    """Immutable-ish audit artifact describing exactly what produced a film."""

    VERSION = 4

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    @staticmethod
    def _file_hash(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-tracked file is missing: {path}")
        return ProductionCheckpoint.digest_file(path)

    @staticmethod
    def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


    def default_model_manifest(self) -> dict[str, Any]:
        """Return authoritative model provenance from the repository inventories."""
        inventory_path = self.project_root / "configs" / "model_inventory.yaml"
        runtime_path = self.project_root / "configs" / "runtime_versions.yaml"
        inventory: dict[str, Any] = {}
        runtime: dict[str, Any] = {}
        if inventory_path.is_file():
            with inventory_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            if isinstance(data, dict):
                inventory = data
        if runtime_path.is_file():
            with runtime_path.open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            if isinstance(data, dict):
                runtime = data
        production_models = dict(inventory.get("models", {}) or {})
        # The Director is a non-ComfyUI runtime asset. Keep its provenance in
        # the top-level inventory block, with Eagle3 recorded as its speculative
        # decoding companion rather than as a standalone H3 production model.
        director = dict(inventory.get("director_model", {}) or {})
        speculative = dict(director.get("speculative", {}) or {})
        runtime_director = dict(runtime.get("director", {}) or {})
        if runtime_director.get("model_path"):
            director["path"] = str(runtime_director["model_path"])
        if runtime_director.get("backend"):
            director["runtime"] = str(runtime_director["backend"])
        if runtime_director.get("format"):
            director["format"] = str(runtime_director["format"])
        if runtime_director.get("vllm_version"):
            director["runtime_version"] = str(runtime_director["vllm_version"])
        if runtime_director.get("speculative_method"):
            speculative["method"] = str(runtime_director["speculative_method"])
        if runtime_director.get("speculative_model_path"):
            speculative["path"] = str(runtime_director["speculative_model_path"])
        if runtime_director.get("speculative_tokens") is not None:
            speculative["tokens"] = int(runtime_director["speculative_tokens"])
        if speculative:
            director["speculative"] = speculative
        return {
            "production": production_models,
            "director": director,
            "inventory_policy": dict(inventory.get("policy", {}) or {}),
        }

    def build(self, plan: dict[str, Any], *, require_context_ir_results: bool = False) -> dict[str, Any]:
        files: dict[str, str] = {}
        roots = ("planner", "pipeline", "execution", "schemas", "scheduler", "ui", "kaggle")
        for root_name in roots:
            root = self.project_root / root_name
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                rel = path.relative_to(self.project_root).as_posix()
                files[rel] = self._file_hash(path)
        for pattern in ("configs/*.yaml", "configs/*.yml", "workflows/**/*.json", "requirements*.txt", ".github/workflows/*.yml"):
            for path in sorted(self.project_root.glob(pattern)):
                if path.is_file():
                    files[path.relative_to(self.project_root).as_posix()] = self._file_hash(path)

        authoritative_models = self.default_model_manifest()
        supplied_models = plan.get("model_manifest") or plan.get("models")
        if supplied_models is not None:
            if not isinstance(supplied_models, dict):
                raise RuntimeError("Production model provenance must be a mapping.")
            if ProductionCheckpoint.digest_object(supplied_models) != ProductionCheckpoint.digest_object(authoritative_models):
                raise RuntimeError("Plan-supplied model provenance differs from the authoritative repository inventory.")
        model_manifest = authoritative_models
        production_models = model_manifest.get("production")
        director_model = model_manifest.get("director")
        if not isinstance(production_models, dict) or not production_models:
            raise RuntimeError("Production model provenance is missing the production model inventory.")
        if not isinstance(director_model, dict) or not (director_model.get("path") or director_model.get("filename")):
            raise RuntimeError("Production model provenance is missing the Director model path.")

        effective_prompts: dict[str, str] = {}
        context_ir_artifacts: dict[str, dict[str, Any]] = {}
        for shot in plan.get("shots", []) or []:
            if not isinstance(shot, dict):
                continue
            sid = str(shot.get("shot_id", "")).strip()
            ctx = shot.get("h3_context_ir") or {}
            if require_context_ir_results and not isinstance(ctx, dict):
                raise RuntimeError(f"Shot {sid} has no production Context-IR record.")
            provenance = ctx.get("context_ir_provenance") if isinstance(ctx, dict) else None
            if isinstance(provenance, dict):
                artifact = {
                    "backend": provenance.get("backend", ""),
                    "status": provenance.get("status", ""),
                    "compiler": provenance.get("compiler", ""),
                    "compiler_version": provenance.get("compiler_version", ""),
                    "base_prompt_sha256": provenance.get("base_prompt_sha256", ""),
                    "effective_prompt_sha256": provenance.get("effective_prompt_sha256", ""),
                    "capture_path": provenance.get("capture_path", ""),
                }
                capture_path = (
                    Path(str(provenance.get("capture_path", "")).strip())
                    if provenance.get("capture_path")
                    else None
                )
                if capture_path and capture_path.is_file():
                    artifact["capture_sha256"] = self._file_hash(capture_path)
                    try:
                        persisted = json.loads(capture_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        raise RuntimeError(
                            f"Shot {sid} has an invalid persisted Context-IR artifact: {capture_path}"
                        ) from exc
                    if persisted.get("context_ir_input") != ctx.get("context_ir_input"):
                        raise RuntimeError(
                            f"Shot {sid} persisted Context-IR does not match the in-memory compiler output."
                        )
                context_ir_artifacts[sid] = artifact

                effective_prompt = str(shot.get("h3_effective_prompt", "") or "").strip()
                expected_base_sha = hashlib.sha256(
                    str(ctx.get("context_ir_input", "")).encode("utf-8")
                ).hexdigest()
                expected_effective_sha = hashlib.sha256(
                    effective_prompt.encode("utf-8")
                ).hexdigest() if effective_prompt else ""
                if require_context_ir_results and (
                    provenance.get("backend") != "local_compiler"
                    or provenance.get("status") != "compiled"
                    or str(provenance.get("base_prompt_sha256", "")) != expected_base_sha
                    or str(provenance.get("effective_prompt_sha256", "")) != expected_effective_sha
                    or not capture_path
                    or not capture_path.is_file()
                ):
                    raise RuntimeError(
                        f"Shot {sid} has inconsistent or missing local Context-IR provenance."
                    )
                if effective_prompt:
                    effective_prompts[sid] = effective_prompt
                elif require_context_ir_results:
                    raise RuntimeError(
                        f"Shot {sid} is missing its effective local H3 prompt."
                    )
            elif require_context_ir_results:
                raise RuntimeError(
                    f"Shot {sid} is missing its local Context-IR provenance record."
                )

        manifest = {
            "version": self.VERSION,
            "production_id": str(plan.get("production_id", "")),
            "plan_sha256": ProductionCheckpoint.plan_digest(plan),
            "story_sha256": ProductionCheckpoint.digest_text(str(plan.get("story", "") or "")),
            "director_notes_sha256": ProductionCheckpoint.digest_text(str(plan.get("director_notes", "") or "")),
            "files": files,
            "models": model_manifest,
            "runtime": plan.get("runtime_diagnostics", {}) or {},
            "timeline_version": (plan.get("timeline", {}) or {}).get("version", 1),
            "effective_h3_prompts": effective_prompts,
            "context_ir": context_ir_artifacts,
            "execution": dict(plan.get("execution", {}) or {}),
            "workflow_files": {k: v for k, v in files.items() if k.startswith("workflows/")},
        }
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return manifest

    def write(self, plan: dict[str, Any], path: Path, *, require_context_ir_results: bool = True) -> dict[str, Any]:
        manifest = self.build(plan, require_context_ir_results=require_context_ir_results)
        self._atomic_write_json(Path(path), manifest)
        return manifest

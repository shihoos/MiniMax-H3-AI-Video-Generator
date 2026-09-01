from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from pipeline.vram_profile import resolve_vram_profile
from planner.config import RUNTIME


class RuntimeDiagnostics:
    """Collect a machine-readable runtime health/fingerprint without mutating the environment."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    @staticmethod
    def _version(module_name: str) -> str:
        try:
            module = importlib.import_module(module_name)
            return str(getattr(module, "__version__", "present"))
        except Exception as exc:
            return f"unavailable: {exc}"

    @staticmethod
    def _command(command: list[str]) -> str:
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False, timeout=10)
            return result.stdout.strip()[-4000:]
        except Exception as exc:
            return f"unavailable: {exc}"

    def collect(self, *, comfy_url: str | None = None) -> dict[str, Any]:
        runtime_vram = dict(RUNTIME.get("runtime", {}).get("vram", {}) or {})
        gpu_id = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_id = torch.cuda.current_device()
        except Exception:
            gpu_id = None
        profile = resolve_vram_profile(runtime_vram, gpu_id=gpu_id)
        features = dict(RUNTIME.get("features", {}) or {})
        report: dict[str, Any] = {
            "python": platform.python_version(),
            "features": {
                "live_preview": bool(features.get("live_preview", True)),
                "qa_enabled": bool(features.get("qa_enabled", True)),
                "vlm_enabled": bool(features.get("vlm_enabled", True)),
                "vlm_reference_analysis": bool(features.get("vlm_reference_analysis", True)),
                "vlm_visual_qa": bool(features.get("vlm_visual_qa", True)),
                "director_critic": bool(features.get("director_critic", True)),
                "selective_retake": bool(features.get("selective_retake", True)),
                "auto_retake": bool(features.get("auto_retake", True)),
                "max_auto_retakes_per_shot": int(features.get("max_auto_retakes_per_shot", 1) or 0),
                "context_ir_version": int(features.get("context_ir_version", 2) or 0),
                "preview_execution_mode": bool(features.get("preview_execution_mode", True)),
                "diagnostic_execution_mode": bool(features.get("diagnostic_execution_mode", True)),
                "retake_execution_mode": bool(features.get("retake_execution_mode", True)),
                "context_ir_required": bool(features.get("context_ir_required", True)),
            },
            "vram_profile": {
                "name": profile.name,
                "async_offload_streams": profile.async_offload_streams,
                "cpu_vae": profile.cpu_vae,
                "disable_pinned_memory": profile.disable_pinned_memory,
                "fast_disk": profile.fast_disk,
                "reserve_vram_gb": profile.reserve_vram_gb,
                "gpu_id": gpu_id,
                "reason": profile.reason,
                "execution_policy": {
                    "production": True,
                    "preview": True,
                    "retake": True,
                    "diagnostic": True,
                },
            },
            "platform": platform.platform(),
            "gradio": self._version("gradio"),
            "pillow": self._version("PIL"),
            "yaml": self._version("yaml"),
            "websocket_client": self._version("websocket"),
            "llama_cpp": self._version("llama_cpp"),
            "environment": {key: value for key, value in os.environ.items() if key.startswith("H3_")},
            "nvidia_smi": self._command(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
        }
        try:
            import torch
            report["torch"] = str(torch.__version__)
            report["torch_cuda"] = str(torch.version.cuda)
            report["cuda_available"] = bool(torch.cuda.is_available())
            report["gpu_count"] = int(torch.cuda.device_count())
            report["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except Exception as exc:
            report["torch"] = f"unavailable: {exc}"
        if comfy_url:
            try:
                from execution.comfy_client import ComfyClient
                client = ComfyClient(comfy_url)
                report["comfyui"] = {"healthy": client.health_check(), "system_stats": client.get_system_stats()}
            except Exception as exc:
                report["comfyui"] = {"healthy": False, "error": str(exc)}
        return report

    def write(self, path: Path, *, comfy_url: str | None = None) -> dict[str, Any]:
        report = self.collect(comfy_url=comfy_url)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return report

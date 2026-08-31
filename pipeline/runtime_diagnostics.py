from __future__ import annotations

import importlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


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
        report: dict[str, Any] = {
            "python": platform.python_version(),
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

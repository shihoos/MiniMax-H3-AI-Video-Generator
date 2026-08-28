from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from threading import RLock
from typing import Any


class MetricsRecorder:
    """Small dependency-free JSONL recorder for per-shot/per-scene telemetry."""

    _lock = RLock()

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _gpu_snapshot(gpu_id: int | None) -> dict[str, Any]:
        if gpu_id is None or shutil.which("nvidia-smi") is None:
            return {}
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode != 0:
                return {}
            for line in result.stdout.splitlines():
                fields = [part.strip() for part in line.split(",")]
                if len(fields) != 4 or int(fields[0]) != int(gpu_id):
                    continue
                return {
                    "gpu_utilization_percent": int(fields[1]),
                    "gpu_memory_used_mb": int(fields[2]),
                    "gpu_memory_total_mb": int(fields[3]),
                }
        except Exception:
            return {}
        return {}

    def record(self, event: str, *, gpu_id: int | None = None, **fields: Any) -> None:
        payload = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "event": str(event),
        }
        payload.update(self._gpu_snapshot(gpu_id))
        payload.update(fields)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())

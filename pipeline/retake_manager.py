from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import Any


class RetakeManager:
    """Create durable selective-retake requests and stitch externally rendered segments."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    def request(
        self,
        production_id: str,
        shot_id: str,
        *,
        start_seconds: float = 0.0,
        end_seconds: float | None = None,
        reason: str = "",
        preserve_audio: bool = True,
    ) -> Path:
        if start_seconds < 0:
            raise ValueError("retake start must be >= 0")
        if end_seconds is not None and end_seconds <= start_seconds:
            raise ValueError("retake end must be greater than start")
        root = self.project_root / "data" / "production" / str(production_id) / "retakes"
        root.mkdir(parents=True, exist_ok=True)
        request_id = "retake_" + uuid.uuid4().hex[:12]
        path = root / f"{request_id}.json"
        payload = {
            "version": 1,
            "request_id": request_id,
            "production_id": str(production_id),
            "shot_id": str(shot_id),
            "start_seconds": float(start_seconds),
            "end_seconds": None if end_seconds is None else float(end_seconds),
            "reason": str(reason or "").strip(),
            "preserve_audio": bool(preserve_audio),
            "status": "requested",
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def stitch(
        self,
        base_video: Path,
        retake_video: Path,
        output_video: Path,
        *,
        start_seconds: float,
        end_seconds: float,
        preserve_audio: bool = True,
    ) -> Path:
        base_video = Path(base_video).resolve()
        retake_video = Path(retake_video).resolve()
        output_video = Path(output_video).resolve()
        if not base_video.is_file() or not retake_video.is_file():
            raise FileNotFoundError("Base or retake video is missing.")
        if end_seconds <= start_seconds:
            raise ValueError("Retake range must have positive duration.")
        output_video.parent.mkdir(parents=True, exist_ok=True)
        tail = output_video.with_name(output_video.stem + ".tail.mp4")
        head = output_video.with_name(output_video.stem + ".head.mp4")
        concat = output_video.with_name(output_video.stem + ".concat.txt")
        try:
            self._run_ffmpeg([
                "ffmpeg", "-y", "-i", str(base_video), "-t", str(start_seconds), "-c", "copy", str(head)
            ])
            self._run_ffmpeg([
                "ffmpeg", "-y", "-ss", str(end_seconds), "-i", str(base_video), "-c", "copy", str(tail)
            ])
            def _concat_line(path: Path) -> str:
                # ffmpeg concat demuxer uses single-quoted file names; escape
                # backslashes, single quotes and control newlines explicitly.
                value = path.as_posix().replace("\\", "\\\\").replace("'", "'\\''").replace("\n", "\\n")
                return f"file '{value}'"

            concat.write_text("\n".join(_concat_line(p) for p in (head, retake_video, tail)) + "\n", encoding="utf-8")
            self._run_ffmpeg([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(output_video)
            ])
            return output_video
        finally:
            for path in (head, tail, concat):
                try:
                    path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _run_ffmpeg(command: list[str]) -> None:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=180)
        if result.returncode != 0:
            raise RuntimeError("ffmpeg retake operation failed:\n" + result.stderr[-5000:])

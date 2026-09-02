from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any


class RetakeManager:
    """Persist selective-retake requests and safely stitch replacements.

    A preserve-audio retake replaces only the requested video interval while
    keeping the original production audio timeline intact. The replacement
    video is normalized to the exact requested interval before it is stitched;
    this prevents a generated retake that is a few frames long/short from
    shifting the original soundtrack relative to the tail.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    @staticmethod
    def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                if os.path.exists(temp_name):
                    os.unlink(temp_name)
            except OSError:
                pass

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
            "version": 2,
            "request_id": request_id,
            "production_id": str(production_id),
            "shot_id": str(shot_id),
            "start_seconds": float(start_seconds),
            "end_seconds": None if end_seconds is None else float(end_seconds),
            "reason": str(reason or "").strip(),
            "preserve_audio": bool(preserve_audio),
            "status": "requested",
        }
        self._atomic_json_write(path, payload)
        return path

    @staticmethod
    def _filter(width: int, height: int, rate: str, *, duration: float | None = None, pad_to_duration: bool = False) -> str:
        filters = [
            f"fps={rate}",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",
        ]
        if pad_to_duration and duration is not None:
            # Clone the final frame when a generated retake is shorter than
            # the requested interval, then trim to the exact target duration.
            filters.append(f"tpad=stop_mode=clone:stop_duration={duration:.6f}")
        return ",".join(filters)

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

        replacement_duration = float(end_seconds) - float(start_seconds)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        head = output_video.with_name(output_video.stem + ".head.mp4")
        tail = output_video.with_name(output_video.stem + ".tail.mp4")
        middle = output_video.with_name(output_video.stem + ".middle.mp4")
        concat = output_video.with_name(output_video.stem + ".concat.txt")
        stitched_video = output_video.with_name(output_video.stem + ".video.mp4")

        try:
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,r_frame_rate",
                    "-of", "json", str(base_video),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=30.0,
            )
            if probe.returncode != 0:
                raise RuntimeError(
                    "Unable to inspect base video for retake stitching: "
                    + probe.stderr[-2000:]
                )
            streams = json.loads(probe.stdout or "{}").get("streams") or []
            if not streams:
                raise RuntimeError("Base video has no video stream.")

            info = streams[0]
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            rate = str(info.get("r_frame_rate") or "24/1")
            if width <= 0 or height <= 0:
                raise RuntimeError("Base video has invalid dimensions for retake stitching.")

            video_args = [
                "-c:v", "libx264",
                "-preset", os.getenv("H3_RETAKE_FFMPEG_PRESET", "fast"),
                "-crf", os.getenv("H3_RETAKE_FFMPEG_CRF", "17"),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]

            if preserve_audio:
                # Preserve the original soundtrack. Critically, force the
                # middle replacement video to the exact requested duration so
                # the tail resumes on the same audio timestamp as the base.
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(base_video), "-t", f"{start_seconds:.6f}", "-an",
                    "-vf", self._filter(width, height, rate),
                    *video_args, str(head),
                ])
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(retake_video), "-an",
                    "-vf", self._filter(width, height, rate, duration=replacement_duration, pad_to_duration=True),
                    "-t", f"{replacement_duration:.6f}",
                    *video_args, str(middle),
                ])
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(base_video), "-ss", f"{end_seconds:.6f}",
                    "-an", "-vf", self._filter(width, height, rate),
                    *video_args, str(tail),
                ])
            else:
                audio_args = [
                    "-c:a", "aac",
                    "-ar", "32000",
                    "-ac", "2",
                    "-b:a", os.getenv("H3_RETAKE_AUDIO_BITRATE", "192k"),
                ]
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(base_video), "-t", f"{start_seconds:.6f}",
                    "-vf", self._filter(width, height, rate),
                    *video_args, *audio_args, str(head),
                ])
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(retake_video),
                    "-vf", self._filter(width, height, rate, duration=replacement_duration, pad_to_duration=True),
                    "-t", f"{replacement_duration:.6f}",
                    *video_args, *audio_args, str(middle),
                ])
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(base_video), "-ss", f"{end_seconds:.6f}",
                    "-vf", self._filter(width, height, rate),
                    *video_args, *audio_args, str(tail),
                ])

            parts = [
                path for path in (head, middle, tail)
                if path.is_file() and path.stat().st_size > 0
            ]
            if not parts:
                raise RuntimeError("No retake stitch segments were produced.")

            lines = []
            for item in parts:
                value = (
                    item.as_posix()
                    .replace("\\", "\\\\")
                    .replace("'", "'\\''")
                    .replace("\n", "\\n")
                )
                lines.append(f"file '{value}'")
            concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

            self._run_ffmpeg([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat),
                "-c", "copy", "-movflags", "+faststart", str(stitched_video),
            ])

            if preserve_audio:
                # Replace only the video while keeping the ORIGINAL complete
                # audio timeline. Since the video now has exactly the same
                # duration as the base, the original soundtrack stays aligned.
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(stitched_video),
                    "-i", str(base_video),
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", os.getenv("H3_RETAKE_AUDIO_BITRATE", "192k"),
                    "-movflags", "+faststart",
                    str(output_video),
                ])
            else:
                # The segments already contain matching A/V parameters.
                os.replace(stitched_video, output_video)

            if not output_video.is_file() or output_video.stat().st_size <= 0:
                raise RuntimeError(
                    f"Retake stitching produced no valid output: {output_video}"
                )
            return output_video
        finally:
            for path in (head, tail, middle, concat, stitched_video):
                try:
                    path.unlink()
                except OSError:
                    pass

    def mark_completed(
        self,
        request_path: Path,
        stitched_video: Path,
        replacement_video: Path,
    ) -> None:
        path = Path(request_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Retake request is not a JSON object: {path}")
        payload["status"] = "completed"
        payload["stitched_video"] = str(Path(stitched_video).resolve())
        payload["replacement_video"] = str(Path(replacement_video).resolve())
        self._atomic_json_write(path, payload)

    @staticmethod
    def _run_ffmpeg(command: list[str]) -> None:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=300.0,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg retake operation failed:\n" + result.stderr[-5000:]
            )

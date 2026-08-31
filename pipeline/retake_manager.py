from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path


class RetakeManager:
    """Persist selective-retake requests and safely stitch replacements."""

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

        head = output_video.with_name(output_video.stem + ".head.mp4")
        tail = output_video.with_name(output_video.stem + ".tail.mp4")
        middle = output_video.with_name(output_video.stem + ".middle.mp4")
        concat = output_video.with_name(output_video.stem + ".concat.txt")
        silent_concat = output_video.with_name(output_video.stem + ".video.mp4")

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

            vf = (
                f"fps={rate},"
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1"
            )
            video_args = [
                "-c:v", "libx264",
                "-preset", os.getenv("H3_RETAKE_FFMPEG_PRESET", "fast"),
                "-crf", os.getenv("H3_RETAKE_FFMPEG_CRF", "17"),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
            ]
            if preserve_audio:
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(base_video), "-t", str(start_seconds),
                    "-an", "-vf", vf, *video_args, str(head),
                ])
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(retake_video), "-an", "-vf", vf, *video_args, str(middle),
                ])
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(end_seconds), "-i", str(base_video),
                    "-an", "-vf", vf, *video_args, str(tail),
                ])
            else:
                audio_args = ["-c:a", "aac", "-ar", "32000", "-ac", "2", "-b:a", os.getenv("H3_RETAKE_AUDIO_BITRATE", "192k")]
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(base_video), "-t", str(start_seconds),
                    "-vf", vf, *video_args, *audio_args, str(head),
                ])
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(retake_video),
                    "-vf", vf, *video_args, *audio_args, str(middle),
                ])
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", str(end_seconds), "-i", str(base_video),
                    "-vf", vf, *video_args, *audio_args, str(tail),
                ])

            parts = [path for path in (head, middle, tail) if path.is_file() and path.stat().st_size > 0]
            if not parts:
                raise RuntimeError("No retake stitch segments were produced.")

            lines = []
            for item in parts:
                value = item.as_posix().replace("\\", "\\\\").replace("'", "'\\''").replace("\n", "\\n")
                lines.append(f"file '{value}'")
            concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

            if preserve_audio:
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", "-movflags", "+faststart", str(silent_concat),
                ])
            else:
                self._run_ffmpeg([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c", "copy", "-movflags", "+faststart", str(silent_concat),
                ])

            if preserve_audio:
                command = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(silent_concat),
                    "-i", str(base_video),
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", os.getenv("H3_RETAKE_AUDIO_BITRATE", "192k"),
                    "-shortest", "-movflags", "+faststart",
                    str(output_video),
                ]
            else:
                # Each head/middle/tail segment was encoded with matching video
                # and audio parameters above, so the concat result already
                # contains the correct segmented soundtrack. Do not replace it
                # with the middle clip's audio.
                command = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(silent_concat),
                    "-map", "0:v:0", "-map", "0:a:0?",
                    "-c", "copy",
                    "-movflags", "+faststart",
                    str(output_video),
                ]
            self._run_ffmpeg(command)

            if not output_video.is_file() or output_video.stat().st_size <= 0:
                raise RuntimeError(f"Retake stitching produced no valid output: {output_video}")
            return output_video
        finally:
            for path in (head, tail, middle, concat, silent_concat):
                try:
                    path.unlink()
                except OSError:
                    pass

    def mark_completed(self, request_path: Path, stitched_video: Path, replacement_video: Path) -> None:
        path = Path(request_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "completed"
        payload["stitched_video"] = str(Path(stitched_video).resolve())
        payload["replacement_video"] = str(Path(replacement_video).resolve())
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _run_ffmpeg(command: list[str]) -> None:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=180.0,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg retake operation failed:\n" + result.stderr[-5000:]
            )

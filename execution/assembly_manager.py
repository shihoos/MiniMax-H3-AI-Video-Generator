from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


class AssemblyManager:
    """Validate, concatenate and atomically publish the final video."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _binary_version(binary: str) -> str:
        if shutil.which(binary) is None:
            raise RuntimeError(f"{binary} is not installed or not on PATH.")
        result = subprocess.run(
            [binary, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{binary} failed version check: {result.stderr[-2000:]}"
            )
        return (result.stdout or result.stderr).splitlines()[0].strip()

    @classmethod
    def check_ffmpeg(cls) -> None:
        cls._binary_version("ffmpeg")
        cls._binary_version("ffprobe")

    @staticmethod
    def _validate_inputs(videos: Iterable[Path]) -> list[Path]:
        result: list[Path] = []
        for video in videos:
            path = Path(video).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size <= 0:
                raise RuntimeError(f"Video is empty: {path}")
            result.append(path)
        if not result:
            raise ValueError("No videos supplied.")
        return result

    @staticmethod
    def _probe(path: Path) -> dict:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_streams", "-show_format",
                "-of", "json", str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed for {path}: {result.stderr[-2000:]}"
            )
        try:
            import json
            return json.loads(result.stdout)
        except Exception as exc:
            raise RuntimeError(f"Invalid ffprobe output for {path}.") from exc

    def assemble(
        self,
        videos: list[Path],
        final_name: str = "final_video.mp4",
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
        *,
        video_preset: str | None = None,
        video_crf: int | None = None,
        video_codec: str | None = None,
        audio_codec: str | None = None,
        audio_bitrate: str | None = None,
    ) -> Path:
        video_preset = str(video_preset or os.getenv("H3_FFMPEG_PRESET", "medium"))
        video_crf = int(video_crf if video_crf is not None else os.getenv("H3_FFMPEG_CRF", "17"))
        video_codec = str(video_codec or os.getenv("H3_FFMPEG_VIDEO_CODEC", "libx264"))
        audio_codec = str(audio_codec or os.getenv("H3_FFMPEG_AUDIO_CODEC", "aac"))
        audio_bitrate = str(audio_bitrate or os.getenv("H3_FFMPEG_AUDIO_BITRATE", "192k"))
        self.check_ffmpeg()
        inputs = self._validate_inputs(videos)

        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Invalid final delivery parameters.")
        if not 0 <= int(video_crf) <= 51:
            raise ValueError("video_crf must be between 0 and 51.")
        if not str(final_name).lower().endswith(".mp4"):
            raise ValueError("Final output must be an .mp4 file.")

        # Probe every source before launching a long assembly job. This catches
        # corrupt zero-byte/incomplete MP4s before spending CPU on concat.
        for path in inputs:
            probe = self._probe(path)
            streams = probe.get("streams", [])
            if not any(s.get("codec_type") == "video" for s in streams):
                raise RuntimeError(f"No video stream found in {path}")

        concat_fd, concat_name = tempfile.mkstemp(
            prefix="concat_", suffix=".txt", dir=self.output_dir
        )
        os.close(concat_fd)
        concat_file = Path(concat_name)
        destination = (self.output_dir / final_name).resolve()
        temp_output = self.output_dir / f".{destination.stem}.tmp.mp4"

        lines = []
        for path in inputs:
            escaped = str(path).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        vf = (
            f"fps={int(fps)},"
            f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase,"
            f"crop={int(width)}:{int(height)}:"
            f"({int(width)}-iw)/2:({int(height)}-ih)/2"
        )
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-vf", vf,
            "-c:v", video_codec,
            "-preset", video_preset,
            "-crf", str(int(video_crf)),
            "-pix_fmt", "yuv420p",
            "-c:a", audio_codec,
            "-b:a", str(audio_bitrate),
            "-movflags", "+faststart",
            str(temp_output),
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "FFmpeg assembly failed:\n" + result.stderr[-5000:]
                )
            if not temp_output.is_file() or temp_output.stat().st_size <= 0:
                raise RuntimeError("FFmpeg reported success but produced no output.")
            os.replace(temp_output, destination)
            return destination
        finally:
            concat_file.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)

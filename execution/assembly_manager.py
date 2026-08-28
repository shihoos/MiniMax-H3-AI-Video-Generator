from __future__ import annotations

import json
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
            return json.loads(result.stdout)
        except Exception as exc:
            raise RuntimeError(f"Invalid ffprobe output for {path}.") from exc

    @staticmethod
    def _stream_signature(probe: dict) -> tuple | None:
        video = next(
            (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
            None,
        )
        audio = next(
            (s for s in probe.get("streams", []) if s.get("codec_type") == "audio"),
            None,
        )
        if video is None:
            return None
        return (
            video.get("codec_name"),
            int(video.get("width") or 0),
            int(video.get("height") or 0),
            str(video.get("r_frame_rate") or ""),
            str(video.get("pix_fmt") or ""),
            video.get("profile"),
            audio.get("codec_name") if audio else None,
            audio.get("sample_rate") if audio else None,
            int(audio.get("channels") or 0) if audio else 0,
        )

    @classmethod
    def _can_stream_copy(
        cls,
        probes: list[dict],
        width: int,
        height: int,
        fps: int,
    ) -> bool:
        if not probes:
            return False
        signatures = [cls._stream_signature(p) for p in probes]
        if not signatures or any(sig is None for sig in signatures):
            return False
        if len(set(signatures)) != 1:
            return False
        video = next(
            s for s in probes[0].get("streams", [])
            if s.get("codec_type") == "video"
        )
        try:
            source_width = int(video.get("width") or 0)
            source_height = int(video.get("height") or 0)
            num, den = (str(video.get("r_frame_rate") or "0/1").split("/", 1))
            source_fps = float(num) / float(den) if float(den) else 0.0
        except Exception:
            return False
        return (
            source_width == int(width)
            and source_height == int(height)
            and abs(source_fps - float(fps)) < 0.01
        )

    @staticmethod
    def _write_concat_file(inputs: list[Path], output_dir: Path) -> Path:
        fd, name = tempfile.mkstemp(prefix="concat_", suffix=".txt", dir=output_dir)
        os.close(fd)
        path = Path(name)
        lines = []
        for item in inputs:
            lines.append(f"file '{str(item).replace(chr(39), chr(39)+chr(92)+chr(39)+chr(39))}'")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

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
        nvenc_cq = int(os.getenv("H3_FFMPEG_NVENC_CQ", "19"))
        require_audio = os.getenv("H3_REQUIRE_AUDIO", "1").strip().lower() in {"1", "true", "yes", "on"}
        allow_copy_fallback = os.getenv("H3_FFMPEG_COPY_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}

        self.check_ffmpeg()
        inputs = self._validate_inputs(videos)

        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Invalid final delivery parameters.")
        if not 0 <= video_crf <= 51:
            raise ValueError("video_crf must be between 0 and 51.")
        if not 0 <= nvenc_cq <= 51:
            raise ValueError("H3_FFMPEG_NVENC_CQ must be between 0 and 51.")
        if not str(final_name).lower().endswith(".mp4"):
            raise ValueError("Final output must be an .mp4 file.")

        probes: list[dict] = []
        for path in inputs:
            probe = self._probe(path)
            streams = probe.get("streams", [])
            if not any(s.get("codec_type") == "video" for s in streams):
                raise RuntimeError(f"No video stream found in {path}")
            if require_audio and not any(s.get("codec_type") == "audio" for s in streams):
                raise RuntimeError(f"No audio stream found in H3 shot output: {path}")
            probes.append(probe)

        concat_file = self._write_concat_file(inputs, self.output_dir)
        destination = (self.output_dir / final_name).resolve()
        temp_output = self.output_dir / f".{destination.stem}.{os.getpid()}.tmp.mp4"
        stream_copy = (
            self._can_stream_copy(probes, width, height, fps)
            and os.getenv("H3_FFMPEG_FORCE_TRANSCODE", "0").strip().lower()
            not in {"1", "true", "yes", "on"}
        )

        def transcode_command() -> list[str]:
            vf = (
                f"fps={int(fps)},"
                f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase,"
                f"crop={int(width)}:{int(height)}:"
                f"({int(width)}-iw)/2:({int(height)}-ih) / 2"
            ).replace(" / ", "/")
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-vf", vf,
                "-c:v", video_codec,
                "-preset", video_preset,
            ]
            if video_codec.lower().endswith("_nvenc"):
                command += ["-cq", str(nvenc_cq), "-rc", "vbr"]
            else:
                command += ["-crf", str(video_crf)]
            command += [
                "-pix_fmt", "yuv420p",
                "-c:a", audio_codec,
                "-b:a", audio_bitrate,
                "-movflags", "+faststart",
                str(temp_output),
            ]
            return command

        def copy_command() -> list[str]:
            return [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(temp_output),
            ]

        try:
            commands = [copy_command()] if stream_copy else [transcode_command()]
            if stream_copy and allow_copy_fallback:
                commands.append(transcode_command())

            last_error = ""
            for index, command in enumerate(commands):
                temp_output.unlink(missing_ok=True)
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    last_error = result.stderr[-5000:]
                    if index + 1 < len(commands):
                        continue
                    raise RuntimeError("FFmpeg assembly failed:\n" + last_error)

                if not temp_output.is_file() or temp_output.stat().st_size <= 0:
                    last_error = "FFmpeg reported success but produced no output."
                    if index + 1 < len(commands):
                        continue
                    raise RuntimeError(last_error)

                final_probe = self._probe(temp_output)
                final_streams = final_probe.get("streams", [])
                video_stream = next((s for s in final_streams if s.get("codec_type") == "video"), None)
                audio_stream = next((s for s in final_streams if s.get("codec_type") == "audio"), None)
                if video_stream is None:
                    last_error = "Final output contains no video stream."
                    if index + 1 < len(commands):
                        continue
                    raise RuntimeError(last_error)
                if require_audio and audio_stream is None:
                    last_error = "Final output contains no audio stream."
                    if index + 1 < len(commands):
                        continue
                    raise RuntimeError(last_error)

                try:
                    final_width = int(video_stream.get("width") or 0)
                    final_height = int(video_stream.get("height") or 0)
                    if final_width != int(width) or final_height != int(height):
                        raise RuntimeError(
                            f"Final output resolution is {final_width}x{final_height}; expected {int(width)}x{int(height)}."
                        )
                except ValueError as exc:
                    raise RuntimeError("Final output has invalid video dimensions.") from exc

                os.replace(temp_output, destination)
                return destination

            raise RuntimeError("FFmpeg assembly failed: " + (last_error or "unknown error"))
        finally:
            concat_file.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)


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
            timeout=30.0,
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
            timeout=60.0,
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
    def _has_audio(probe: dict) -> bool:
        return any(s.get("codec_type") == "audio" for s in probe.get("streams", []))

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
        try:
            video_crf = int(video_crf if video_crf is not None else os.getenv("H3_FFMPEG_CRF", "17"))
            nvenc_cq = int(os.getenv("H3_FFMPEG_NVENC_CQ", "19"))
        except (TypeError, ValueError) as exc:
            raise ValueError("FFmpeg CRF/CQ environment values must be integers.") from exc
        video_codec = str(video_codec or os.getenv("H3_FFMPEG_VIDEO_CODEC", "libx264"))
        audio_codec = str(audio_codec or os.getenv("H3_FFMPEG_AUDIO_CODEC", "aac"))
        audio_bitrate = str(audio_bitrate or os.getenv("H3_FFMPEG_AUDIO_BITRATE", "192k"))
        self.check_ffmpeg()
        inputs = self._validate_inputs(videos)

        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("Invalid final delivery parameters.")
        if not 0 <= int(video_crf) <= 51:
            raise ValueError("video_crf must be between 0 and 51.")
        if not 0 <= nvenc_cq <= 51:
            raise ValueError("H3_FFMPEG_NVENC_CQ must be between 0 and 51.")
        if not str(final_name).lower().endswith(".mp4"):
            raise ValueError("Final output must be an .mp4 file.")

        probes = []
        for path in inputs:
            probe = self._probe(path)
            streams = probe.get("streams", [])
            if not any(s.get("codec_type") == "video" for s in streams):
                raise RuntimeError(f"No video stream found in {path}")
            probes.append(probe)

        require_audio = os.getenv("H3_FFMPEG_REQUIRE_AUDIO", "1").strip().lower() in {"1", "true", "yes", "on"}
        if require_audio and any(not self._has_audio(probe) for probe in probes):
            missing = [str(path) for path, probe in zip(inputs, probes) if not self._has_audio(probe)]
            raise RuntimeError("H3 shot output is missing required audio:\n" + "\n".join(missing))

        destination = (self.output_dir / final_name).resolve()
        if destination.parent != self.output_dir:
            raise ValueError("final_name must not escape output_dir.")
        concat_file = self._write_concat_file(inputs, self.output_dir)
        temp_output = self.output_dir / f".{destination.stem}.{os.getpid()}.tmp.mp4"
        stream_copy = self._can_stream_copy(probes, width, height, fps) and os.getenv("H3_FFMPEG_FORCE_TRANSCODE", "0").strip().lower() not in {"1", "true", "yes", "on"}

        if stream_copy:
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(temp_output),
            ]
        else:
            vf = (
                f"fps={int(fps)},"
                f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase,"
                f"crop={int(width)}:{int(height)}:"
                f"({int(width)}-iw)/2:({int(height)}-ih)/2,setsar=1"
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
                command += ["-crf", str(int(video_crf))]
            command += [
                "-pix_fmt", "yuv420p",
                "-c:a", audio_codec,
                "-b:a", audio_bitrate,
                "-movflags", "+faststart",
                str(temp_output),
            ]

        try:
            try:
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=1800.0,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "FFmpeg assembly timed out after 1800 seconds."
                ) from exc
            if result.returncode != 0 and stream_copy:
                vf = (
                    f"fps={int(fps)},scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase,"
                    f"crop={int(width)}:{int(height)}:({int(width)}-iw)/2:({int(height)}-ih)/2,setsar=1"
                )
                command = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(concat_file),
                    "-vf", vf, "-c:v", video_codec, "-preset", video_preset,
                ]
                if video_codec.lower().endswith("_nvenc"):
                    command += ["-cq", str(nvenc_cq), "-rc", "vbr"]
                else:
                    command += ["-crf", str(int(video_crf))]
                command += [
                    "-pix_fmt", "yuv420p", "-c:a", audio_codec,
                    "-b:a", audio_bitrate, "-movflags", "+faststart", str(temp_output),
                ]
                try:
                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=1800.0,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        "FFmpeg assembly timed out after 1800 seconds."
                    ) from exc
            if result.returncode != 0:
                raise RuntimeError("FFmpeg assembly failed:\n" + result.stderr[-5000:])
            if not temp_output.is_file() or temp_output.stat().st_size <= 0:
                raise RuntimeError("FFmpeg reported success but produced no output.")
            final_probe = self._probe(temp_output)
            final_streams = final_probe.get("streams", [])
            if not any(s.get("codec_type") == "video" for s in final_streams):
                raise RuntimeError("Final output contains no video stream.")
            if require_audio and not self._has_audio(final_probe):
                raise RuntimeError("Final output contains no audio stream.")
            final_video = next(s for s in final_streams if s.get("codec_type") == "video")
            if int(final_video.get("width") or 0) != int(width) or int(final_video.get("height") or 0) != int(height):
                raise RuntimeError(
                    f"Final output resolution mismatch: expected {width}x{height}, "
                    f"got {final_video.get('width')}x{final_video.get('height')}"
                )
            os.replace(temp_output, destination)
            return destination
        finally:
            concat_file.unlink(missing_ok=True)
            temp_output.unlink(missing_ok=True)

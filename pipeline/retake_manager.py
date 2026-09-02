from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any


class RetakeManager:
    """Persist selective-retake requests and stitch replacements safely.

    preserve_audio=True replaces only video in the requested interval and keeps
    the original production soundtrack. preserve_audio=False builds segmented
    A/V with one deterministic audio contract for every segment.
    """

    AUDIO_SAMPLE_RATE = 32000
    AUDIO_CHANNELS = 2
    VIDEO_FPS_DEFAULT = 24

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    @staticmethod
    def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except OSError:
                pass

    @staticmethod
    def _run(command: list[str], timeout: float = 300.0) -> None:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ffmpeg/ffprobe operation failed:\n" + result.stderr[-5000:]
            )

    @classmethod
    def _probe(cls, path: Path) -> dict[str, Any]:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30.0,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Unable to inspect media {path}: {result.stderr[-3000:]}")
        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid ffprobe output for {path}.") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"Invalid media probe for {path}.")
        return data

    @staticmethod
    def _stream_duration(probe: dict[str, Any], codec_type: str) -> float:
        for stream in probe.get("streams", []) or []:
            if stream.get("codec_type") != codec_type:
                continue
            try:
                value = float(stream.get("duration"))
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value
        try:
            value = float((probe.get("format") or {}).get("duration"))
        except (TypeError, ValueError):
            value = 0.0
        return max(0.0, value)

    @staticmethod
    def _fps(rate: str) -> float:
        try:
            numerator, denominator = str(rate).split("/", 1)
            value = float(numerator) / float(denominator)
            return value if value > 0 else 24.0
        except (TypeError, ValueError, ZeroDivisionError):
            return 24.0

    @staticmethod
    def _safe_duration(value: float) -> float:
        return max(0.0, float(value))

    @classmethod
    def _video_filter(
        cls,
        width: int,
        height: int,
        rate: str,
        *,
        target_duration: float | None = None,
        source_duration: float | None = None,
    ) -> str:
        filters = [
            f"fps={rate}",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",
            "setpts=PTS-STARTPTS",
        ]
        if target_duration is not None:
            target = cls._safe_duration(target_duration)
            source = cls._safe_duration(source_duration or 0.0)
            pad_duration = max(0.0, target - source)
            if pad_duration > 0:
                filters.append(f"tpad=stop_mode=clone:stop_duration={pad_duration + (1.0 / max(1.0, cls._fps(rate))):.6f}")
            filters.extend([
                f"trim=duration={target:.6f}",
                "setpts=PTS-STARTPTS",
            ])
        return ",".join(filters)

    @classmethod
    def _audio_filter(cls, duration: float) -> str:
        return (
            "aresample=async=1:first_pts=0,"
            "apad,"
            f"atrim=duration={duration:.6f},"
            "asetpts=PTS-STARTPTS"
        )

    @classmethod
    def _has_audio(cls, path: Path) -> bool:
        probe = cls._probe(path)
        return any(s.get("codec_type") == "audio" for s in probe.get("streams", []) or [])

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
        self._atomic_json_write(path, {
            "version": 2,
            "request_id": request_id,
            "production_id": str(production_id),
            "shot_id": str(shot_id),
            "start_seconds": float(start_seconds),
            "end_seconds": None if end_seconds is None else float(end_seconds),
            "reason": str(reason or "").strip(),
            "preserve_audio": bool(preserve_audio),
            "status": "requested",
        })
        return path

    def _encode_video_segment(
        self,
        source: Path,
        destination: Path,
        *,
        width: int,
        height: int,
        rate: str,
        target_duration: float,
        start: float | None = None,
        end: float | None = None,
    ) -> None:
        probe = self._probe(source)
        source_duration = self._stream_duration(probe, "video")
        if source_duration <= 0:
            raise RuntimeError(f"Source has no usable video duration: {source}")
        target = max(0.0, float(target_duration))
        if target <= 0:
            raise ValueError("Target video segment duration must be positive.")

        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        if start is not None:
            command.extend(["-ss", f"{start:.6f}"])
        command.extend(["-i", str(source), "-map", "0:v:0"])
        if end is not None:
            command.extend(["-t", f"{max(0.0, end - (start or 0.0)):.6f}"])

        command.extend([
            "-vf",
            self._video_filter(width, height, rate),
            "-r", rate,
            "-t", f"{target:.6f}",
            "-c:v", "libx264",
            "-preset", os.getenv("H3_RETAKE_FFMPEG_PRESET", "fast"),
            "-crf", os.getenv("H3_RETAKE_FFMPEG_CRF", "17"),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(destination),
        ])
        if source_duration + 1e-6 >= target:
            self._run(command)
            return

        # H3 can occasionally return a replacement shorter than the requested
        # retake interval. Extend it deterministically by holding its final
        # frame, then trim the result to the exact target frame count.
        frame_path = destination.with_name(destination.stem + ".last.png")
        try:
            self._run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-sseof", "-0.05", "-i", str(source),
                "-map", "0:v:0", "-frames:v", "1", "-update", "1",
                str(frame_path),
            ])
            pad_duration = max(0.0, target - source_duration)
            if not frame_path.is_file() or frame_path.stat().st_size <= 0:
                raise RuntimeError(f"Unable to extract final replacement frame: {source}")
            duration = min(source_duration, target)
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(source),
                "-loop", "1", "-framerate", rate, "-i", str(frame_path),
                "-filter_complex",
                (
                    f"[0:v:0]trim=duration={duration:.6f},setpts=PTS-STARTPTS[v0];"
                    f"[1:v:0]trim=duration={pad_duration:.6f},setpts=PTS-STARTPTS[v1];"
                    "[v0][v1]concat=n=2:v=1:a=0,"
                    "setpts=PTS-STARTPTS[outv]"
                ),
                "-map", "[outv]",
                "-t", f"{target:.6f}",
                "-r", rate,
                "-c:v", "libx264",
                "-preset", os.getenv("H3_RETAKE_FFMPEG_PRESET", "fast"),
                "-crf", os.getenv("H3_RETAKE_FFMPEG_CRF", "17"),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(destination),
            ]
            self._run(command)
        finally:
            try:
                frame_path.unlink()
            except OSError:
                pass

    def _encode_av_segment(
        self,
        source: Path,
        destination: Path,
        *,
        width: int,
        height: int,
        rate: str,
        target_duration: float,
        start: float | None = None,
        end: float | None = None,
    ) -> None:
        probe = self._probe(source)
        source_duration = self._stream_duration(probe, "video")
        if source_duration <= 0:
            raise RuntimeError(f"Source has no usable video duration: {source}")
        target = max(0.0, float(target_duration))
        has_audio = self._has_audio(source)

        # Build a video-only normalized segment first. This keeps the final-frame
        # extension logic identical to preserve_audio mode and prevents A/V from
        # diverging when the retake video is short.
        video_tmp = destination.with_name(destination.stem + ".video.mp4")
        try:
            self._encode_video_segment(
                source, video_tmp,
                width=width, height=height, rate=rate,
                target_duration=target,
                start=start, end=end,
            )
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_tmp)]
            if has_audio:
                if start is not None:
                    command.extend(["-ss", f"{start:.6f}"])
                command.extend(["-i", str(source)])
                audio_input = "1:a:0"
            else:
                command.extend([
                    "-f", "lavfi", "-i",
                    f"anullsrc=channel_layout=stereo:sample_rate={self.AUDIO_SAMPLE_RATE}",
                ])
                audio_input = "1:a:0"
            command.extend([
                "-map", "0:v:0",
                "-map", audio_input,
                "-c:v", "copy",
                "-c:a", "aac",
                "-ar", str(self.AUDIO_SAMPLE_RATE),
                "-ac", str(self.AUDIO_CHANNELS),
                "-b:a", os.getenv("H3_RETAKE_AUDIO_BITRATE", "192k"),
                "-af", self._audio_filter(target),
                "-t", f"{target:.6f}",
                "-movflags", "+faststart",
                str(destination),
            ])
            if start is not None and not has_audio:
                # No source-audio seek is needed when the second input is silence.
                pass
            self._run(command)
        finally:
            try:
                video_tmp.unlink()
            except OSError:
                pass

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

        base_probe = self._probe(base_video)
        base_duration = self._stream_duration(base_probe, "video")
        if base_duration <= 0:
            raise RuntimeError("Base video has no usable video duration.")

        start = max(0.0, min(float(start_seconds), base_duration))
        end = max(start, min(float(end_seconds), base_duration))
        if end <= start:
            raise ValueError("Retake interval is outside the base video duration.")

        replacement_duration = end - start
        video_stream = next((s for s in base_probe.get("streams", []) or [] if s.get("codec_type") == "video"), None)
        if video_stream is None:
            raise RuntimeError("Base video has no video stream.")

        width = int(video_stream.get("width") or 0)
        height = int(video_stream.get("height") or 0)
        rate = str(video_stream.get("r_frame_rate") or f"{self.VIDEO_FPS_DEFAULT}/1")
        if width <= 0 or height <= 0:
            raise RuntimeError("Base video dimensions are invalid.")

        output_video.parent.mkdir(parents=True, exist_ok=True)
        head = output_video.with_name(output_video.stem + ".head.mp4")
        middle = output_video.with_name(output_video.stem + ".middle.mp4")
        tail = output_video.with_name(output_video.stem + ".tail.mp4")
        concat = output_video.with_name(output_video.stem + ".concat.txt")
        stitched = output_video.with_name(output_video.stem + ".stitched.mp4")

        try:
            head_duration = start
            tail_duration = max(0.0, base_duration - end)
            if head_duration > 0:
                if preserve_audio:
                    self._encode_video_segment(base_video, head, width=width, height=height, rate=rate, target_duration=head_duration, start=0.0, end=start)
                else:
                    self._encode_av_segment(base_video, head, width=width, height=height, rate=rate, target_duration=head_duration, start=0.0, end=start)
            if replacement_duration > 0:
                if preserve_audio:
                    self._encode_video_segment(retake_video, middle, width=width, height=height, rate=rate, target_duration=replacement_duration)
                else:
                    self._encode_av_segment(retake_video, middle, width=width, height=height, rate=rate, target_duration=replacement_duration)
            if tail_duration > 0:
                if preserve_audio:
                    self._encode_video_segment(base_video, tail, width=width, height=height, rate=rate, target_duration=tail_duration, start=end, end=base_duration)
                else:
                    self._encode_av_segment(base_video, tail, width=width, height=height, rate=rate, target_duration=tail_duration, start=end, end=base_duration)

            parts = [p for p in (head, middle, tail) if p.is_file() and p.stat().st_size > 0]
            if not parts:
                raise RuntimeError("No retake stitch segments were produced.")

            lines = []
            for item in parts:
                value = item.as_posix().replace("\\", "\\\\").replace("'", "'\\''").replace("\n", "\\n")
                lines.append(f"file '{value}'")
            concat.write_text("\n".join(lines) + "\n", encoding="utf-8")

            self._run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat),
                "-c", "copy", "-movflags", "+faststart", str(stitched),
            ])

            if preserve_audio:
                base_audio = next((s for s in base_probe.get("streams", []) or [] if s.get("codec_type") == "audio"), None)
                command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(stitched)]
                if base_audio is not None:
                    command.extend([
                        "-i", str(base_video),
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "copy", "-c:a", "aac",
                        "-b:a", os.getenv("H3_RETAKE_AUDIO_BITRATE", "192k"),
                    ])
                else:
                    command.extend(["-map", "0:v:0", "-c:v", "copy"])
                command.extend(["-t", f"{base_duration:.6f}", "-movflags", "+faststart", str(output_video)])
                self._run(command)
            else:
                os.replace(stitched, output_video)

            final_probe = self._probe(output_video)
            final_duration = self._stream_duration(final_probe, "video")
            fps = self._fps(rate)
            tolerance = max(1.0 / fps, 0.02) + 0.02
            if abs(final_duration - base_duration) > tolerance:
                raise RuntimeError(
                    f"Retake stitch duration mismatch: base={base_duration:.6f}s output={final_duration:.6f}s"
                )
            return output_video
        finally:
            for path in (head, middle, tail, concat, stitched):
                try:
                    path.unlink()
                except OSError:
                    pass

    def mark_completed(self, request_path: Path, stitched_video: Path, replacement_video: Path) -> None:
        path = Path(request_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Retake request is not a JSON object: {path}")
        payload["status"] = "completed"
        payload["stitched_video"] = str(Path(stitched_video).resolve())
        payload["replacement_video"] = str(Path(replacement_video).resolve())
        self._atomic_json_write(path, payload)

    _run_ffmpeg = _run

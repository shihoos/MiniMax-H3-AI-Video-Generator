from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class DurationEstimate:
    seconds: float
    source: str
    exact_for_source: bool = False


class DialogueDurationProvider(Protocol):
    def estimate(self, text: str, event: dict[str, Any] | None = None) -> DurationEstimate:
        ...


class ExplicitOrWPMDurationProvider:
    """Use explicit duration when supplied; otherwise a conservative estimate.

    The estimate is intentionally never presented as H3-native audio timing.
    Actual rendered audio is measured separately with FFprobe.
    """

    WORDS_PER_MINUTE = 145.0
    MIN_SECONDS = 0.60
    PADDING_SECONDS = 0.15

    def estimate(self, text: str, event: dict[str, Any] | None = None) -> DurationEstimate:
        event = event or {}
        for key in ("expected_duration_seconds", "duration_seconds"):
            value = event.get(key)
            if value not in (None, ""):
                seconds = float(value)
                if seconds <= 0:
                    raise ValueError(f"{key} must be greater than zero when supplied.")
                return DurationEstimate(seconds=seconds, source="explicit", exact_for_source=True)
        expected_ms = event.get("expected_duration_ms")
        if expected_ms not in (None, ""):
            milliseconds = int(expected_ms)
            if milliseconds <= 0:
                raise ValueError("expected_duration_ms must be greater than zero.")
            return DurationEstimate(
                seconds=milliseconds / 1000.0,
                source="explicit_ms",
                exact_for_source=True,
            )
        words = max(1, len(re.findall(r"\S+", str(text or ""))))
        seconds = max(
            self.MIN_SECONDS,
            words / self.WORDS_PER_MINUTE * 60.0 + self.PADDING_SECONDS,
        )
        return DurationEstimate(seconds=seconds, source="wpm_estimate", exact_for_source=False)


class FFProbeMediaDurationProvider:
    """Measure actual encoded media duration using ffprobe."""

    def __init__(self, ffprobe_path: str | None = None) -> None:
        self.ffprobe_path = ffprobe_path or shutil.which("ffprobe")

    def duration_seconds(
        self,
        media_path: str | Path,
        *,
        stream_selector: str = "",
    ) -> float:
        """Measure the selected stream duration, never the container duration.

        Some codecs do not expose a stream-level duration. In that case we
        fall back to packet timestamps for the selected stream rather than
        silently comparing the container duration for both audio and video.
        """
        if not self.ffprobe_path:
            raise RuntimeError(
                "ffprobe is required for actual media-duration validation."
            )

        path = Path(media_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        selector = str(stream_selector or "").strip()
        if not selector:
            raise ValueError(
                "stream_selector is required for stream-specific duration measurement."
            )

        command = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", selector,
            "-show_entries", "stream=duration,start_time",
            "-of", "json",
            str(path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=15.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffprobe timed out while analyzing {path}. The media file may be corrupted."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe stream-duration query failed for {path}: "
                f"{result.stderr[-4000:]}"
            )

        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams") or []
        if streams:
            stream = streams[0]
            value = stream.get("duration")
            if value not in (None, "", "N/A"):
                seconds = float(value)
                if seconds > 0:
                    return seconds

        # Fallback to packet timestamps so a missing stream.duration does not
        # silently degrade to container duration.
        packet_command = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", selector,
            "-show_entries", "packet=pts_time,dts_time,duration_time",
            "-of", "json",
            str(path),
        ]
        try:
            packet_result = subprocess.run(
                packet_command,
                capture_output=True,
                text=True,
                check=False,
                timeout=15.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffprobe timed out while analyzing {path}. The media file may be corrupted."
            ) from exc
        if packet_result.returncode != 0:
            raise RuntimeError(
                f"ffprobe packet-duration fallback failed for {path}: "
                f"{packet_result.stderr[-4000:]}"
            )

        packets = json.loads(packet_result.stdout or "{}").get("packets") or []
        timestamps: list[float] = []
        final_end: float | None = None
        for packet in packets:
            start_value = packet.get("pts_time", packet.get("dts_time"))
            if start_value not in (None, "N/A", ""):
                try:
                    start = float(start_value)
                except (TypeError, ValueError):
                    start = None
                if start is not None:
                    timestamps.append(start)
                    duration_value = packet.get("duration_time")
                    try:
                        packet_duration = float(duration_value) if duration_value not in (None, "N/A", "") else 0.0
                    except (TypeError, ValueError):
                        packet_duration = 0.0
                    final_end = max(
                        final_end or 0.0,
                        start + max(packet_duration, 0.0),
                    )

        if final_end is not None and timestamps:
            start_time = min(timestamps)
            seconds = final_end - start_time
            if seconds > 0:
                return seconds

        raise RuntimeError(
            f"ffprobe returned no usable stream duration for {selector} in {path}."
        )

    def has_stream(self, media_path: str | Path, stream_type: str) -> bool:
        if not self.ffprobe_path:
            return False
        path = Path(media_path).resolve()
        command = [
            self.ffprobe_path,
            "-v", "error",
            "-select_streams", stream_type,
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=15.0,
            )
        except subprocess.TimeoutExpired:
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def validate_video_audio_sync(
        self,
        media_path: str | Path,
        *,
        tolerance_seconds: float = 0.30,
    ) -> dict[str, Any]:
        path = Path(media_path).resolve()
        has_video = self.has_stream(path, "v:0")
        has_audio = self.has_stream(path, "a:0")
        result: dict[str, Any] = {
            "path": str(path),
            "video_stream_present": has_video,
            "audio_stream_present": has_audio,
            "video_duration_seconds": None,
            "audio_duration_seconds": None,
            "duration_delta_seconds": None,
            "within_tolerance": None,
        }
        if has_video:
            result["video_duration_seconds"] = self.duration_seconds(path, stream_selector="v:0")
        if has_audio:
            result["audio_duration_seconds"] = self.duration_seconds(path, stream_selector="a:0")
        if result["video_duration_seconds"] is not None and result["audio_duration_seconds"] is not None:
            delta = abs(result["video_duration_seconds"] - result["audio_duration_seconds"])
            result["duration_delta_seconds"] = delta
            result["within_tolerance"] = delta <= float(tolerance_seconds)
        return result

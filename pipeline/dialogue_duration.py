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

    def duration_seconds(self, media_path: str | Path, *, stream_selector: str = "") -> float:
        if not self.ffprobe_path:
            raise RuntimeError("ffprobe is required for actual media-duration validation.")
        path = Path(media_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        command = [
            self.ffprobe_path,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
        ]
        if stream_selector:
            command[4:4] = ["-select_streams", stream_selector]
        command.append(str(path))
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[-4000:]}")
        payload = json.loads(result.stdout or "{}")
        value = (payload.get("format") or {}).get("duration")
        if value in (None, ""):
            raise RuntimeError(f"ffprobe returned no duration for {path}.")
        seconds = float(value)
        if seconds <= 0:
            raise RuntimeError(f"ffprobe returned invalid duration for {path}: {seconds}")
        return seconds

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
        result = subprocess.run(command, capture_output=True, text=True, check=False)
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

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class VisualStateObserver:
    """Cheap deterministic observation of a rendered frame; no model inference required."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root).resolve()

    def observe_frame(self, frame_path: Path) -> dict[str, Any]:
        path = Path(frame_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        result: dict[str, Any] = {"frame_path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        try:
            from PIL import Image, ImageStat
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                stat = ImageStat.Stat(rgb)
                result["width"], result["height"] = rgb.size
                result["mean_rgb"] = [round(float(v), 3) for v in stat.mean]
                result["mean_luminance"] = round(sum(stat.mean) / 3.0, 3)
        except Exception as exc:
            result["observer_warning"] = str(exc)
        return result

    def observe_video_tail(self, video_path: Path, frame_path: Path) -> dict[str, Any]:
        result = self.observe_frame(frame_path)
        result["video_path"] = str(Path(video_path).resolve())
        return result

    def extract_review_frames(self, video_path: Path, output_dir: Path, count: int = 3) -> list[Path]:
        """Extract deterministic representative frames for optional semantic QA."""
        import subprocess
        video_path = Path(video_path).resolve()
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if count < 1:
            return []
        # Use evenly spaced timestamps only when ffprobe can determine duration.
        try:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=10)
            duration = max(0.01, float(probe.stdout.strip()))
        except Exception:
            return []
        positions = [0.05, 0.50, 0.95] if count >= 3 else [0.5]
        positions = positions[:count]
        frames: list[Path] = []
        for index, ratio in enumerate(positions, start=1):
            path = output_dir / f"review_{index:02d}.jpg"
            timestamp = min(max(duration * ratio, 0.0), max(0.0, duration - 0.01))
            result = subprocess.run([
                "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(video_path),
                "-frames:v", "1", "-q:v", "3", str(path)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=30)
            if result.returncode == 0 and path.is_file() and path.stat().st_size > 0:
                frames.append(path)
        return frames


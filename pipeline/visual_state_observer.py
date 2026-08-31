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

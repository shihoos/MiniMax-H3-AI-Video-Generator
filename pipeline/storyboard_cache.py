from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


class StoryboardCache:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.cache_dir = self.root / "storyboard_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def digest(structural_payload: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(structural_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def paths(self, digest: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{digest}.png", self.cache_dir / f"{digest}.json"

    def restore(self, digest: str, image_path: Path, manifest_path: Path) -> dict[str, Any] | None:
        cached_image, cached_manifest = self.paths(digest)
        if not cached_image.is_file() or not cached_manifest.is_file():
            return None
        shutil.copy2(cached_image, image_path)
        shutil.copy2(cached_manifest, manifest_path)
        try:
            payload = json.loads(cached_manifest.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def store(self, digest: str, image_path: Path, manifest_path: Path) -> None:
        cached_image, cached_manifest = self.paths(digest)
        shutil.copy2(image_path, cached_image)
        shutil.copy2(manifest_path, cached_manifest)

from __future__ import annotations

from pathlib import Path
import json
import hashlib
import os
import shutil
from contextlib import contextmanager
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


class StoryboardReferenceBuilder:
    """Build a deterministic single-image storyboard/reference sheet."""

    WIDTH = 2048
    HEIGHT = 1365
    HEADER_H = 145
    GAP = 18
    PANEL_BG = (245, 245, 245)
    BORDER = (45, 45, 45)
    TEXT = (20, 20, 20)
    MUTED = (95, 95, 95)

    def __init__(self, project_root: Path, production_id: str) -> None:
        self.project_root = Path(project_root).resolve()
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(production_id))
        self.root = self.project_root / "data" / "production" / safe / "storyboard"
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _font(size: int):
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _crop(path: str, size: tuple[int, int]) -> Image.Image | None:
        p = Path(path)
        if not p.is_file():
            return None
        try:
            with Image.open(p) as src:
                image = src.convert("RGB")
                return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
        except Exception:
            return None

    @staticmethod
    def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
        words = str(text or "").split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    @staticmethod
    def structural_digest(plan: dict, characters: list[dict]) -> str:
        """Hash only storyboard-defining structure; dialogue/checkpoint state is excluded."""
        scenes = []
        scene_order = {
            str(scene.get("scene_id", "")): index
            for index, scene in enumerate(plan.get("scenes", []) or [], start=1)
            if isinstance(scene, dict)
        }
        for shot in sorted(
            [item for item in (plan.get("shots", []) or []) if isinstance(item, dict)],
            key=lambda item: (scene_order.get(str(item.get("scene_id", "")), 10**9), int(item.get("order", 0))),
        ):
            scenes.append({
                "scene_id": str(shot.get("scene_id", "")),
                "order": int(shot.get("order", 0)),
                "shot_id": str(shot.get("shot_id", "")),
                "characters": list(shot.get("characters", []) or []),
                "location": str(shot.get("location", "")),
                "action": str(shot.get("action", "")),
                "camera_shot": str(shot.get("camera_shot", "")),
                "camera_movement": str(shot.get("camera_movement", "")),
                "composition_notes": str(shot.get("composition_notes", "")),
                "lighting": str(shot.get("lighting", "")),
                "storyboard_reference": bool(shot.get("storyboard_reference")),
            })
        payload = {
            "characters": [
                {
                    "character_id": str(item.get("character_id", "")),
                    "name": str(item.get("name", "")),
                    "appearance": item.get("appearance", {}) or {},
                    "reference_paths": item.get("reference_paths", []) or [],
                }
                for item in characters if isinstance(item, dict)
            ],
            "shots": scenes,
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def build(self, plan: dict, characters: list[dict]) -> dict[str, Any]:
        digest = self.structural_digest(plan, characters)
        cache_dir = self.root / "storyboard_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_image = cache_dir / f"{digest}.png"
        cache_manifest = cache_dir / f"{digest}.json"
        image_path = self.root / "unified_storyboard_reference.png"
        manifest_cache_path = self.root / "reference_role_manifest.json"
        if cache_image.is_file() and cache_manifest.is_file():
            shutil.copy2(cache_image, image_path)
            shutil.copy2(cache_manifest, manifest_cache_path)
            cached = json.loads(manifest_cache_path.read_text(encoding="utf-8"))
            return {
                "path": str(image_path),
                "manifest_path": str(manifest_cache_path),
                "panels": cached.get("storyboard", {}).get("panels", []),
                "manifest": cached,
                "cache_hit": True,
            }

        scene_order = {
            str(scene.get("scene_id", "")).strip(): index
            for index, scene in enumerate(plan.get("scenes", []) or [])
            if isinstance(scene, dict) and str(scene.get("scene_id", "")).strip()
        }
        shots = sorted(
            [s for s in (plan.get("shots", []) or []) if isinstance(s, dict)],
            key=lambda s: (
                scene_order.get(str(s.get("scene_id", "")).strip(), 10**9),
                int(s.get("order", 0)),
            ),
        )
        image = Image.new("RGB", (self.WIDTH, self.HEIGHT), "white")
        draw = ImageDraw.Draw(image)
        title_font = self._font(42)
        body_font = self._font(23)
        small_font = self._font(19)
        draw.text((42, 28), "MiniMax H3 — Unified Storyboard Reference", fill=self.TEXT, font=title_font)
        subtitle = "Sequencing / composition guide. Character identity remains anchored by the separate canonical references."
        draw.text((44, 86), subtitle, fill=self.MUTED, font=small_font)

        body_y = self.HEADER_H
        available_h = self.HEIGHT - body_y - 35
        columns = min(4, max(1, len(shots)))
        rows = (len(shots) + columns - 1) // columns
        panel_w = (self.WIDTH - 2 * 35 - (columns - 1) * self.GAP) // columns
        panel_h = (available_h - max(0, rows - 1) * self.GAP) // max(1, rows)

        char_by_name = {
            str(c.get("name", "")).strip().lower(): c
            for c in characters
            if isinstance(c, dict)
        }
        panels = []
        for index, shot in enumerate(shots):
            col = index % columns
            row = index // columns
            x = 35 + col * (panel_w + self.GAP)
            y = body_y + row * (panel_h + self.GAP)
            draw.rounded_rectangle((x, y, x + panel_w, y + panel_h), radius=12, outline=self.BORDER, width=3, fill=self.PANEL_BG)

            sid = str(shot.get("shot_id", f"shot_{index+1}"))
            scene = str(shot.get("scene_id", ""))
            headline = f"{index + 1:02d}  {scene} / {sid}"
            draw.text((x + 18, y + 14), headline, fill=self.TEXT, font=body_font)

            thumb = None
            for name in shot.get("characters", []) or []:
                char = char_by_name.get(str(name).strip().lower())
                if not char:
                    continue
                paths = list(char.get("reference_paths", []) or [])
                if paths:
                    thumb = self._crop(paths[0], (260, 185))
                    if thumb:
                        break
            if thumb:
                image.paste(thumb, (x + 18, y + 58))
                text_x = x + 302
                text_w = panel_w - 320
            else:
                text_x = x + 18
                text_w = panel_w - 36

            text_y = y + 62
            action = str(shot.get("action", "") or shot.get("visual_prompt", ""))
            camera = ", ".join(v for v in (
                str(shot.get("camera_shot", "")),
                str(shot.get("camera_movement", "")),
                str(shot.get("lens_and_depth_of_field", "")),
            ) if v)
            for label, value in (
                ("ACTION", action),
                ("CAMERA", camera),
                ("LOCATION", str(shot.get("location", ""))),
            ):
                draw.text((text_x, text_y), f"{label}", fill=self.MUTED, font=small_font)
                text_y += 24
                lines = self._wrap(draw, value, small_font, text_w)
                for line in lines[:3]:
                    draw.text((text_x, text_y), line, fill=self.TEXT, font=small_font)
                    text_y += 22
                text_y += 6
                if text_y > y + panel_h - 35:
                    break

            panels.append({
                "index": index + 1,
                "shot_id": sid,
                "scene_id": scene,
            })

        path = self.root / "unified_storyboard_reference.png"
        image.save(path, format="PNG", optimize=True)
        manifest_path = self.root / "reference_role_manifest.json"
        manifest = {
            "manifest_version": 2,
            "production_id": self.root.parent.name,
            "storyboard": {
                "path": str(path),
                "role": "storyboard",
                "structural_digest": digest,
                "description": "Unified storyboard for shot sequencing, composition and blocking; not the canonical character identity source.",
                "panels": panels,
            },
            "shots": {},
        }
        for shot in shots:
            sid = str(shot.get("shot_id", "")).strip()
            refs = [str(value).strip() for value in (shot.get("reference_images", []) or []) if str(value).strip()]
            roles = [dict(item) for item in (shot.get("reference_roles", []) or []) if isinstance(item, dict)]
            by_path = {str(item.get("path", "")).strip(): item for item in roles if str(item.get("path", "")).strip()}

            if str(path) not in refs:
                if len(refs) >= 9:
                    ranked = []
                    for index, ref in enumerate(refs):
                        role = by_path.get(ref, {"priority": 50})
                        # Preserve canonical identity references preferentially.
                        score = 1000 + int(role.get("priority", 50)) if role.get("role") == "character_identity" else int(role.get("priority", 50))
                        ranked.append((score, -index, ref))
                    _, _, dropped = min(ranked, key=lambda item: (item[0], item[1]))
                    refs = [ref for ref in refs if ref != dropped]
                    roles = [role for role in roles if str(role.get("path", "")).strip() != dropped]
                refs.append(str(path))
                roles.append({
                    "path": str(path),
                    "role": "storyboard",
                    "label": "Unified storyboard for sequencing, composition and blocking; not the canonical character identity source.",
                    "priority": 90,
                })

            # This is the initial manifest order. The runner may later add/reorder
            # the previous-shot final frame; it will update this same manifest.
            roles_by_path = {str(role.get("path", "")).strip(): dict(role) for role in roles if str(role.get("path", "")).strip()}
            normalized_roles = []
            for ref in refs[:9]:
                role = dict(roles_by_path.get(ref, {"path": ref, "role": "visual_reference", "priority": 50}))
                role["path"] = ref
                normalized_roles.append(role)

            shot["reference_images"] = [role["path"] for role in normalized_roles]
            shot["reference_roles"] = normalized_roles
            shot["storyboard_reference"] = str(path)
            shot["reference_role_manifest"] = str(manifest_path)

            bindings = []
            for index, role in enumerate(normalized_roles, start=1):
                kind = str(role.get("role", "")).strip().lower()
                if kind == "storyboard":
                    label = "Unified storyboard for sequencing, composition and blocking; not the canonical character identity source."
                elif kind == "previous_shot_last_frame":
                    label = "Previous shot final-frame continuity reference."
                else:
                    character = str(role.get("character_name", "")).strip()
                    label = f"Canonical visual identity reference for {character}; use for stable identity only." if character else "Production visual reference."
                bindings.append(f"<Picture {index}> = {label}")

            manifest["shots"][sid] = {
                "shot_id": sid,
                "reference_images": list(shot["reference_images"]),
                "references": [
                    {"picture_index": i + 1, **dict(role), "path": str(shot["reference_images"][i])}
                    for i, role in enumerate(normalized_roles)
                ],
                "picture_bindings": bindings,
                "actual_runtime_order": False,
                "invariant_verified": False,
            }
        temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        with temporary.open("r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(manifest_path)
        shutil.copy2(path, cache_image)
        shutil.copy2(manifest_path, cache_manifest)
        return {"path": str(path), "manifest_path": str(manifest_path), "panels": panels, "manifest": manifest, "cache_hit": False}

    @staticmethod
    @contextmanager
    def _manifest_file_lock(path: Path):
        lock_path = path.with_name(f".{path.name}.lock")
        handle = lock_path.open("a+")
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            handle.close()

    @staticmethod
    def update_manifest(
        manifest_path: str | Path,
        shot_id: str,
        reference_images: list[str],
        reference_roles: list[dict],
        picture_bindings: list[str],
        *,
        actual_runtime_order: bool = True,
    ) -> dict[str, Any]:
        path = Path(manifest_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with StoryboardReferenceBuilder._manifest_file_lock(path):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            shots = manifest.setdefault("shots", {})
            sid = str(shot_id).strip()
            refs = [str(value).strip() for value in reference_images if str(value).strip()]
            roles = [dict(role) for role in reference_roles]
            bindings = [str(value) for value in picture_bindings]
            if len(refs) != len(roles):
                raise RuntimeError(
                    f"Reference manifest mismatch for {sid}: {len(refs)} images vs {len(roles)} roles."
                )
            if len(refs) != len(bindings):
                raise RuntimeError(
                    f"Picture binding mismatch for {sid}: {len(refs)} refs vs {len(bindings)} bindings."
                )
            normalized_roles = []
            for index, (ref, role) in enumerate(zip(refs, roles), start=1):
                item = dict(role)
                item["path"] = ref
                item["picture_index"] = index
                normalized_roles.append(item)
            entry = shots.setdefault(sid, {})
            entry.update({
                "shot_id": sid,
                "reference_images": refs,
                "references": normalized_roles,
                "picture_bindings": bindings,
                "actual_runtime_order": bool(actual_runtime_order),
                "invariant_verified": False,
            })
            manifest["shots"] = shots
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            with temporary.open("r+b") as handle:
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
            return dict(entry)

    @staticmethod
    def assert_manifest_invariant(
        manifest_entry: dict[str, Any],
        reference_images: list[str],
        picture_bindings: list[str],
    ) -> None:
        manifest_refs = [str(value) for value in manifest_entry.get("reference_images", [])]
        actual_refs = [str(value) for value in reference_images]
        manifest_bindings = list(manifest_entry.get("picture_bindings", []))
        manifest_reference_records = list(manifest_entry.get("references", []))
        if manifest_refs != actual_refs:
            raise RuntimeError("H3 reference order != reference-role manifest.")
        if manifest_bindings != list(picture_bindings):
            raise RuntimeError("reference-role manifest != <Picture N> prompt binding.")
        if len(manifest_reference_records) != len(actual_refs):
            raise RuntimeError("Reference-role record count does not match H3 reference order.")
        for index, (ref, record) in enumerate(zip(actual_refs, manifest_reference_records), start=1):
            if str(record.get("path", "")) != str(ref):
                raise RuntimeError(f"Manifest reference path mismatch at Picture {index}.")
            if int(record.get("picture_index", 0)) != index:
                raise RuntimeError(f"Manifest Picture index mismatch at Picture {index}.")


def json_dumps(value: Any) -> str:
    import json
    return json.dumps(value, indent=2, ensure_ascii=False)

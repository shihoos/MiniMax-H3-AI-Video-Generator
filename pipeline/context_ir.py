from __future__ import annotations

from pathlib import Path
import re
from typing import Any


class H3ContextIRCompiler:
    """Compile the project's canonical shot state into H3 Ref2VA Context-IR.

    The compiler is deterministic. It does not replace MiniMax's hosted
    Context-IR service; it prepares a semantically precise Ref2VA prompt and
    reference contract for the existing official API bridge in the ComfyUI
    workflow.
    """

    VERSION = 2
    REQUIRED_SECTIONS = (
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    _LABEL_RE = re.compile(r"^<(Picture|Video|Audio)\s+(\d+)>$")
    _TOKEN_RE = re.compile(r"<(?:Picture|Video|Audio)\s+\d+>")

    @staticmethod
    def _clean(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @classmethod
    def _list_text(cls, value: Any, limit: int = 8) -> str:
        if isinstance(value, dict):
            value = value.values()
        values: list[str] = []
        for item in value or []:
            text = cls._clean(item)
            if text and text not in values:
                values.append(text)
            if len(values) >= limit:
                break
        return "; ".join(values)

    @classmethod
    def _reference_kind(cls, role: dict[str, Any], path: str, fallback: str = "Picture") -> str:
        raw = cls._clean(role.get("media_type") or role.get("type") or role.get("kind")).lower()
        if raw in {"video", "movie", "mp4", "mov", "webm", "mkv", "avi"}:
            return "Video"
        if raw in {"audio", "sound", "wav", "mp3", "m4a", "aac", "flac", "ogg"}:
            return "Audio"
        suffix = Path(path).suffix.lower()
        if suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
            return "Video"
        if suffix in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
            return "Audio"
        return fallback

    @classmethod
    def _role_for_path(cls, roles: list[dict[str, Any]], path: str) -> dict[str, Any] | None:
        target = cls._clean(path)
        for role in roles:
            if cls._clean(role.get("path") or role.get("source")) == target:
                return role
        return None

    @classmethod
    def _character_for_reference(cls, shot: dict[str, Any], path: str, media_type: str) -> dict[str, str]:
        mapping_key = {
            "Video": "reference_video_by_character",
            "Audio": "reference_audio_by_character",
        }.get(media_type)
        if not mapping_key:
            return {}
        mapping = shot.get(mapping_key) or {}
        if not isinstance(mapping, dict):
            return {}
        target = cls._clean(path)
        for name, values in mapping.items():
            if isinstance(values, (list, tuple, set)) and target in {cls._clean(v) for v in values}:
                return {"character_name": cls._clean(name)}
            if isinstance(values, str) and cls._clean(values) == target:
                return {"character_name": cls._clean(name)}
        return {}

    @classmethod
    def _default_role(cls, media_type: str) -> str:
        return {
            "Picture": "visual_reference",
            "Video": "motion_reference",
            "Audio": "audio_reference",
        }[media_type]

    @classmethod
    def _default_relationship(cls, media_type: str, role: str) -> str:
        role_l = cls._clean(role).lower()
        if media_type == "Audio":
            if "copy" in role_l or "reuse" in role_l:
                return "fully_copy"
            return "reference"
        if "identity" in role_l or "retake_start" in role_l or "retake_end" in role_l or "continuity" in role_l:
            return "fully_preserved"
        if "storyboard" in role_l:
            return "partially_preserved"
        if "motion" in role_l or "style" in role_l:
            return "weak_reference" if "motion" in role_l else "attribute_transfer"
        return "weak_reference"

    @classmethod
    def _relationship(cls, media_type: str, role: dict[str, Any]) -> str:
        value = cls._clean(role.get("relationship") or role.get("reference_relationship"))
        if value:
            allowed = (
                {"fully_copy", "partially_copy", "reference", "weak_reference"}
                if media_type == "Audio"
                else {"fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"}
            )
            if value in allowed:
                return value
        return cls._default_relationship(media_type, cls._clean(role.get("role")))

    @classmethod
    def _canonical_references(cls, shot: dict[str, Any]) -> list[dict[str, Any]]:
        """Build a single semantic registry while preserving runtime media order.

        Existing runtime fields are intentionally treated as the source of truth:
        reference_images/reference_videos/reference_audio_paths are the exact media
        lists consumed by the H3 workflow, while reference_roles enrich picture roles
        and the per-character maps provide semantic links for video/audio references.
        """
        roles = [r for r in (shot.get("reference_roles") or []) if isinstance(r, dict)]
        refs: list[dict[str, Any]] = []
        counters = {"Picture": 0, "Video": 0, "Audio": 0}
        seen: set[tuple[str, str]] = set()

        media_lists = (
            ("Picture", list(shot.get("reference_images") or [])),
            ("Video", list(shot.get("reference_videos") or [])),
            ("Audio", list(shot.get("reference_audio_paths") or [])),
        )
        # Legacy shots sometimes expose one audio path through reference_audio only.
        if not media_lists[2][1] and cls._clean(shot.get("reference_audio")):
            media_lists = media_lists[:2] + (("Audio", [shot["reference_audio"]]),)

        for media_type, paths in media_lists:
            for raw_path in paths:
                path = cls._clean(raw_path)
                if not path:
                    continue
                key = (media_type, path)
                if key in seen:
                    continue
                seen.add(key)
                role = dict(cls._role_for_path(roles, path) or {})
                role["path"] = path
                role["media_type"] = media_type.lower()
                role_name = cls._clean(role.get("role")) or cls._default_role(media_type)
                character = cls._character_for_reference(shot, path, media_type)
                character_name = cls._clean(role.get("character_name")) or character.get("character_name", "")
                counters[media_type] += 1
                refs.append(
                    {
                        "index": len(refs) + 1,
                        "media_type": media_type.lower(),
                        "media_kind": media_type,
                        "label": f"<{media_type} {counters[media_type]}>",
                        "path": path,
                        "role": role_name,
                        "description": cls._clean(role.get("description") or role.get("label") or role_name),
                        "relationship": cls._relationship(media_type, role),
                        "character_name": character_name,
                        "character_id": cls._clean(role.get("character_id")),
                        "priority": int(role.get("priority", 0) or 0),
                    }
                )

        # Legacy bindings are only a fallback when there are no runtime media paths.
        if not refs:
            for raw in shot.get("reference_bindings") or []:
                binding = cls._clean(raw)
                if not binding:
                    continue
                match = re.search(r"<(Picture|Video|Audio)\s+(\d+)>", binding, flags=re.I)
                media_type = match.group(1).capitalize() if match else "Picture"
                requested = int(match.group(2)) if match else counters[media_type] + 1
                counters[media_type] = max(counters[media_type], requested)
                refs.append(
                    {
                        "index": len(refs) + 1,
                        "media_type": media_type.lower(),
                        "media_kind": media_type,
                        "label": f"<{media_type} {requested}>",
                        "path": "",
                        "role": cls._default_role(media_type),
                        "description": binding,
                        "relationship": cls._default_relationship(media_type, cls._default_role(media_type)),
                        "character_name": "",
                        "character_id": "",
                        "priority": 0,
                    }
                )
        return refs

    @classmethod
    def _character_index(cls, plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for char in plan.get("characters") or []:
            if not isinstance(char, dict):
                continue
            name = cls._clean(char.get("name"))
            if name:
                out[name.casefold()] = char
        return out

    @classmethod
    def _subject_definitions(cls, plan: dict[str, Any], shot: dict[str, Any]) -> str:
        chars = cls._character_index(plan)
        refs = cls._canonical_references(shot)
        referenced_names: list[str] = []
        for ref in refs:
            name = cls._clean(ref.get("character_name"))
            if name and name.casefold() not in {n.casefold() for n in referenced_names}:
                referenced_names.append(name)

        subject_ids = {name.casefold(): i + 1 for i, name in enumerate(referenced_names)}
        lines: list[str] = []
        for name in [cls._clean(v) for v in (shot.get("characters") or []) if cls._clean(v)]:
            char = chars.get(name.casefold(), {})
            appearance = char.get("appearance", {}) if isinstance(char.get("appearance"), dict) else {}
            clothing = char.get("clothing", {}) if isinstance(char.get("clothing"), dict) else {}
            details = [
                name,
                cls._clean(char.get("role")),
                cls._clean(char.get("description")),
                cls._list_text(appearance.values()),
                cls._list_text(clothing.values()),
                cls._list_text(char.get("distinctive_features")),
                cls._list_text(char.get("continuity_rules")),
            ]
            details = [v for v in details if v]
            if name.casefold() in subject_ids:
                lines.append(f"<Subject {subject_ids[name.casefold()]}> is " + ". ".join(details) + ".")
            elif details:
                lines.append("Character: " + ". ".join(details) + ".")

        for ref in refs:
            relation = ref["relationship"]
            role = ref["role"]
            char_name = cls._clean(ref.get("character_name"))
            target = f" for {char_name}" if char_name else ""
            description = ref["description"] or role
            subject_link = ""
            if char_name and char_name.casefold() in subject_ids:
                subject_link = f" It defines <Subject {subject_ids[char_name.casefold()]}>."
            lines.append(
                f"{ref['label']} is the {role} reference{target}; {description}.{subject_link} "
                f"Its reference relationship is {relation}."
            )

        if not lines:
            return "No explicit external references are required for this shot."
        return "\n".join(lines)

    @classmethod
    def _task_type(cls, refs: list[dict[str, Any]], shot: dict[str, Any]) -> str:
        has_reference = bool(refs)
        has_audio = any(ref["media_kind"] == "Audio" for ref in refs)
        if has_reference and has_audio:
            return "[reference generation + audio reference]"
        if has_reference:
            return "[reference generation]"
        return "[text generation]"

    @classmethod
    def _summary(cls, plan: dict[str, Any], shot: dict[str, Any], refs: list[dict[str, Any]]) -> str:
        parts = [
            cls._task_type(refs, shot),
            f"Scene: {cls._clean(shot.get('scene_id'))}",
            f"Shot: {cls._clean(shot.get('shot_id'))}",
        ]
        if cls._clean(plan.get("production_id")):
            parts.append(f"Production: {cls._clean(plan.get('production_id'))}")
        if cls._clean(shot.get("location")):
            parts.append(f"Target location: {cls._clean(shot.get('location'))}")
        if cls._clean(shot.get("time_of_day")):
            parts.append(f"Time of day: {cls._clean(shot.get('time_of_day'))}")
        duration = float(shot.get("duration_seconds", 0.0) or 0.0)
        parts.append(f"Target duration: {duration:.3f} seconds")
        if refs:
            parts.append("References: " + ", ".join(ref["label"] for ref in refs))
        return "; ".join(p for p in parts if p).strip() + "."

    @classmethod
    def _retention_analysis(cls, shot: dict[str, Any], refs: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for ref in refs:
            target = f" for {ref['character_name']}" if ref.get("character_name") else ""
            lines.append(f"{ref['label']}: {ref['relationship']}{target} — {ref['description']}.")

        continuity = shot.get("continuity_start_state") or {}
        if isinstance(continuity, dict):
            for key, label in (
                ("location", "location"),
                ("lighting", "lighting"),
                ("environment", "environment"),
                ("camera_side", "camera side"),
                ("state_description", "state"),
            ):
                value = cls._clean(continuity.get(key))
                if value:
                    lines.append(f"Preserve {label}: {value}.")
            props = cls._list_text(continuity.get("props"), 12)
            if props:
                lines.append(f"Preserve required props: {props}.")

        notes = cls._clean(shot.get("continuity_notes"))
        if notes:
            lines.append(f"Continuity instruction: {notes}.")
        if not lines:
            return "No external content requires preservation beyond the shot specification."
        return "\n".join(lines)

    @classmethod
    def _speaker_language(cls, event: dict[str, Any], shot: dict[str, Any], plan: dict[str, Any]) -> str:
        return (
            cls._clean(event.get("language"))
            or cls._clean(shot.get("dialogue_language"))
            or cls._clean(plan.get("dialogue_language"))
            or cls._clean(plan.get("language"))
            or "English"
        )

    @classmethod
    def _dialogue_lines(cls, plan: dict[str, Any], shot: dict[str, Any]) -> list[str]:
        events = [e for e in (shot.get("dialogue_events") or []) if isinstance(e, dict)]
        speaker_map: dict[str, str] = {}
        next_id = 1
        lines: list[str] = []
        for event in events:
            speaker_name = cls._clean(event.get("speaker_name") or event.get("speaker"))
            text = str(event.get("text", "") or "").strip()
            if not text:
                continue
            raw_id = cls._clean(event.get("speaker_id"))
            if re.fullmatch(r"S\d+", raw_id, flags=re.IGNORECASE):
                speaker_id = raw_id.upper()
                if speaker_name:
                    speaker_map.setdefault(speaker_name.casefold(), speaker_id)
                    try:
                        next_id = max(next_id, int(speaker_id[1:]) + 1)
                    except ValueError:
                        pass
            else:
                key = speaker_name.casefold() or f"speaker_{next_id}"
                if key not in speaker_map:
                    speaker_map[key] = f"S{next_id}"
                    next_id += 1
                speaker_id = speaker_map[key]
            language = cls._speaker_language(event, shot, plan)
            start = float(event.get("start_seconds", 0.0) or 0.0)
            end = float(event.get("end_seconds", 0.0) or 0.0)
            continuation = " Continue this dialogue into the next shot." if event.get("continues_to_next_shot") else ""
            speaker = f"{speaker_name} ({speaker_id})" if speaker_name else f"({speaker_id})"
            timing = f"At {start:.2f} seconds" if end <= start else f"From {start:.2f} to {end:.2f} seconds"
            lines.append(f"{timing}, {speaker} says: <d>[{language}] {text}</d>.{continuation}")
        if not lines:
            speech = cls._clean(shot.get("speech_text"))
            if speech:
                language = cls._clean(shot.get("dialogue_language")) or cls._clean(plan.get("language")) or "English"
                lines.append(f"<d>[{language}] {speech}</d>")
        return lines

    @classmethod
    def _scene(cls, plan: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
        for scene in plan.get("scenes") or []:
            if isinstance(scene, dict) and cls._clean(scene.get("scene_id")) == cls._clean(shot.get("scene_id")):
                return scene
        return {}

    @classmethod
    def _detailed_description(cls, plan: dict[str, Any], shot: dict[str, Any], refs: list[dict[str, Any]]) -> str:
        scene = cls._scene(plan, shot)
        base = (
            cls._clean(shot.get("detailed_description"))
            or cls._clean(shot.get("visual_prompt"))
            or cls._clean(shot.get("action"))
            or "Depict the required story action in playback order."
        )
        lines: list[str] = []
        style = cls._clean(shot.get("mood") or scene.get("mood"))
        if style:
            lines.append(f"Overall visual tone: {style}.")
        lines.append(f"[Shot 1] {base}")

        camera = "; ".join(
            p for p in (
                cls._clean(shot.get("camera_shot")),
                cls._clean(shot.get("camera_movement")),
                cls._clean(shot.get("lens_and_depth_of_field")),
                cls._clean(shot.get("composition_notes")),
            ) if p
        )
        if camera:
            lines.append(f"Camera and composition: {camera}.")

        environment = "; ".join(
            p for p in (
                cls._clean(shot.get("location")),
                cls._clean(scene.get("atmosphere")),
                cls._list_text(scene.get("environment_details"), 10),
                cls._list_text(scene.get("key_props"), 10),
            ) if p
        )
        if environment:
            lines.append(f"Environment: {environment}.")

        lighting = "; ".join(
            p for p in (
                cls._clean(shot.get("lighting") or scene.get("lighting")),
                cls._clean(shot.get("color_temperature") or scene.get("color_temperature")),
            ) if p
        )
        if lighting:
            lines.append(f"Lighting: {lighting}.")

        if refs:
            lines.append(
                "Reference application in this shot: "
                + "; ".join(
                    f"{r['label']} ({r['role']}, {r['relationship']})"
                    + (f" for {r['character_name']}" if r.get("character_name") else "")
                    for r in refs
                )
                + "."
            )

        dialogue = cls._dialogue_lines(plan, shot)
        lines.extend(dialogue)
        sound = cls._clean(shot.get("sound_effects"))
        if sound:
            lines.append(f"Current physical sound: {sound}.")
        return " ".join(line.strip() for line in lines if line.strip())

    @classmethod
    def _soundscape(cls, shot: dict[str, Any]) -> str:
        value = cls._clean(shot.get("overall_soundscape"))
        return value or "Natural location ambience and physical sounds appropriate to the described action."

    @classmethod
    def _music(cls, shot: dict[str, Any]) -> str:
        value = cls._clean(shot.get("non_diegetic_music"))
        return value or "No non-diegetic music unless explicitly required by the production plan."

    @classmethod
    def validate(cls, context_ir: dict[str, Any]) -> None:
        if not isinstance(context_ir, dict):
            raise TypeError("H3 Context-IR must be a mapping.")
        if int(context_ir.get("version", 0) or 0) != cls.VERSION:
            raise ValueError(f"Unsupported H3 Context-IR version; expected {cls.VERSION}.")
        if cls._clean(context_ir.get("mode")).lower() != "ref2va":
            raise ValueError("H3 Context-IR mode must be ref2va.")
        prompt = cls._clean(context_ir.get("h3_prompt"))
        if not prompt:
            raise ValueError("H3 Context-IR contains an empty h3_prompt.")
        sections = context_ir.get("sections")
        if not isinstance(sections, dict) or tuple(sections.keys()) != cls.REQUIRED_SECTIONS:
            raise ValueError("H3 Context-IR must expose the six sections in canonical order.")
        positions: list[int] = []
        for name in cls.REQUIRED_SECTIONS:
            marker = f"{name}:"
            pos = prompt.find(marker)
            if pos < 0:
                raise ValueError(f"H3 Context-IR prompt is missing required section: {marker}")
            positions.append(pos)
            if not cls._clean(sections.get(name)):
                raise ValueError(f"H3 Context-IR section is empty: {name}")
        if positions != sorted(positions):
            raise ValueError("H3 Context-IR sections are out of order.")

        refs = context_ir.get("references") or []
        if not isinstance(refs, list):
            raise ValueError("H3 Context-IR references must be a list.")
        declared: dict[str, dict[str, Any]] = {}
        for ref in refs:
            if not isinstance(ref, dict):
                raise ValueError("H3 Context-IR reference entries must be objects.")
            label = cls._clean(ref.get("label"))
            match = cls._LABEL_RE.fullmatch(label)
            if not match:
                raise ValueError(f"Invalid canonical reference label: {label!r}")
            if label in declared:
                raise ValueError(f"Duplicate Context-IR reference label: {label}")
            media_kind = match.group(1).lower()
            media_type = cls._clean(ref.get("media_type")).lower()
            if media_type and media_type != media_kind:
                raise ValueError(f"Context-IR reference type mismatch for {label}.")
            relationship = cls._clean(ref.get("relationship"))
            if media_kind == "audio":
                allowed = {"fully_copy", "partially_copy", "reference", "weak_reference"}
            else:
                allowed = {"fully_preserved", "partially_preserved", "attribute_transfer", "weak_reference"}
            if relationship not in allowed:
                raise ValueError(f"Invalid relationship {relationship!r} for {label}.")
            declared[label] = ref

        used = set(cls._TOKEN_RE.findall(prompt))
        if used - set(declared):
            raise ValueError("H3 Context-IR uses undeclared reference labels: " + ", ".join(sorted(used - set(declared))))
        if set(declared) - used:
            raise ValueError("H3 Context-IR declares unused reference labels: " + ", ".join(sorted(set(declared) - used)))

        counts = {"picture": 0, "video": 0, "audio": 0}
        for ref in declared.values():
            counts[cls._clean(ref.get("media_type")).lower()] += 1
        if counts["picture"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
            raise ValueError("H3 Context-IR reference counts exceed Ref2VA limits.")
        if sum(counts.values()) > 12:
            raise ValueError("H3 Context-IR reference count exceeds the 12-file Ref2VA limit.")
        if counts["audio"] and not (counts["picture"] or counts["video"]):
            raise ValueError("Ref2VA audio references require at least one visual reference.")

        for section_name in ("subject_definitions", "retention_analysis", "detailed_description"):
            section_text = str(sections[section_name])
            for label in declared:
                if label not in section_text:
                    raise ValueError(f"Reference {label} is missing from {section_name}.")

        # Speaker IDs must remain stable throughout the prompt and dialogue tags
        # must carry an explicit language marker.
        speaker_ids = re.findall(r"\((S\d+)\)", sections["detailed_description"])
        if speaker_ids and any(not speaker_ids.count(sid) for sid in set(speaker_ids)):
            raise ValueError("Invalid speaker-id sequence in detailed_description.")
        for tag in re.findall(r"<d>(.*?)</d>", sections["detailed_description"], flags=re.S):
            if not re.match(r"\[[^\]]+\]\s+", tag.strip()):
                raise ValueError("Dialogue tags must contain an explicit [Language] marker.")

    @classmethod
    def prompt(cls, context_ir: dict[str, Any]) -> str:
        cls.validate(context_ir)
        return str(context_ir["h3_prompt"]).strip()

    def compile(self, plan: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
        refs = self._canonical_references(shot)
        sections = {
            "subject_definitions": self._subject_definitions(plan, shot),
            "summary": self._summary(plan, shot, refs),
            "retention_analysis": self._retention_analysis(shot, refs),
            "detailed_description": self._detailed_description(plan, shot, refs),
            "overall_soundscape": self._soundscape(shot),
            "non_diegetic_music": self._music(shot),
        }
        prompt = "\n\n".join(f"{name}:\n{sections[name]}" for name in self.REQUIRED_SECTIONS)
        registry = []
        for ref in refs:
            registry.append(
                {
                    "index": int(ref["index"]),
                    "label": ref["label"],
                    "media_type": ref["media_type"],
                    "path": ref["path"],
                    "role": ref["role"],
                    "description": ref["description"],
                    "relationship": ref["relationship"],
                    "character_name": ref["character_name"],
                    "character_id": ref["character_id"],
                    "priority": int(ref["priority"]),
                    "binding": f"{ref['label']} -> {ref['role']}",
                }
            )

        result = {
            "version": self.VERSION,
            "mode": "ref2va",
            "story": str(plan.get("story", "") or ""),
            "scene": {
                "scene_id": self._clean(shot.get("scene_id")),
                "location": self._clean(shot.get("location")),
                "time_of_day": self._clean(shot.get("time_of_day")),
                "mood": self._clean(shot.get("mood")),
            },
            "h3_prompt": prompt,
            "sections": sections,
            "shot": {
                "shot_id": self._clean(shot.get("shot_id")),
                "duration_seconds": float(shot.get("duration_seconds", 0.0) or 0.0),
                "camera": {
                    "shot": self._clean(shot.get("camera_shot")),
                    "movement": self._clean(shot.get("camera_movement")),
                    "lens": self._clean(shot.get("lens_and_depth_of_field")),
                },
                "composition": self._clean(shot.get("composition_notes")),
                "lighting": self._clean(shot.get("lighting")),
                "action": self._clean(shot.get("action")),
            },
            "continuity": shot.get("continuity_start_state", {}) or {},
            "references": registry,
            "audio": {
                "soundscape": self._soundscape(shot),
                "music": self._music(shot),
            },
        }
        self.validate(result)
        return result

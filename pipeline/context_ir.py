from __future__ import annotations

from typing import Any
from pathlib import Path
import re


class H3ContextIRCompiler:
    """Compile production state into MiniMax H3 Ref2VA Context-IR style text.

    The repository deliberately stays Ref2VA-only. This compiler therefore
    follows MiniMax's current full-reference prompt organization:
    subject_definitions, summary, retention_analysis, detailed_description,
    overall_soundscape, and non_diegetic_music. The resulting text is the
    authoritative prompt consumed by the H3 workflow builder.
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

    @staticmethod
    def _clean(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @classmethod
    def _list_text(cls, value: Any, limit: int = 8) -> str:
        values = []
        for item in value or []:
            text = cls._clean(item)
            if text and text not in values:
                values.append(text)
            if len(values) >= limit:
                break
        return "; ".join(values)

    @classmethod
    def _reference_kind(cls, role: dict[str, Any], path: str) -> str:
        raw = cls._clean(
            role.get("media_type")
            or role.get("type")
            or role.get("kind")
        ).lower()
        if raw in {"video", "movie", "mp4", "mov", "webm", "mkv", "avi"}:
            return "Video"
        if raw in {"audio", "sound", "wav", "mp3", "m4a", "aac", "flac", "ogg"}:
            return "Audio"
        suffix = Path(path).suffix.lower()
        if suffix in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
            return "Video"
        if suffix in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
            return "Audio"
        return "Picture"

    @classmethod
    def _canonical_references(cls, shot: dict[str, Any]) -> list[dict[str, Any]]:
        """Return one deterministic reference registry for the shot.

        ``reference_roles`` is the authoritative modern contract.
        ``reference_bindings`` is retained as a legacy compatibility source only
        when roles are absent; it can never renumber an authoritative role list.
        Labels are media-specific (Picture/Video/Audio) and counted independently,
        matching the Ref2VA reference-label convention.
        """
        roles = [
            role for role in (shot.get("reference_roles", []) or [])
            if isinstance(role, dict)
        ]
        bindings = [
            cls._clean(raw)
            for raw in (shot.get("reference_bindings", []) or [])
            if cls._clean(raw)
        ]

        refs: list[dict[str, Any]] = []
        counters = {"Picture": 0, "Video": 0, "Audio": 0}
        seen_paths: set[str] = set()

        for role in roles:
            path = cls._clean(
                role.get("path")
                or role.get("source")
                or ""
            )
            if path and path in seen_paths:
                continue
            if path:
                seen_paths.add(path)

            media_kind = cls._reference_kind(role, path)
            counters[media_kind] += 1
            label = f"<{media_kind} {counters[media_kind]}>"
            role_name = cls._clean(role.get("role")) or "visual reference"
            description = cls._clean(
                role.get("description")
                or role.get("label")
                or role_name
            )

            refs.append({
                "index": len(refs) + 1,
                "label": label,
                "media_type": media_kind.lower(),
                "path": path,
                "role": role_name,
                "description": description,
                "priority": int(role.get("priority", 0) or 0),
            })

        # Legacy plans may contain only textual bindings. Preserve them without
        # inventing a conflicting modern role list. Parse an existing canonical
        # media label when present; otherwise retain the historical Picture label.
        if not refs and bindings:
            counters = {"Picture": 0, "Video": 0, "Audio": 0}
            for binding in bindings:
                match = re.search(
                    r"<(Picture|Video|Audio)\s+(\d+)>\s*=",
                    binding,
                    flags=re.IGNORECASE,
                )
                if match:
                    media_kind = match.group(1).capitalize()
                    counters[media_kind] = max(
                        counters[media_kind],
                        int(match.group(2)),
                    )
                else:
                    media_kind = "Picture"
                    counters[media_kind] += 1
                if match:
                    label = f"<{media_kind} {int(match.group(2))}>"
                else:
                    label = f"<{media_kind} {counters[media_kind]}>"
                refs.append({
                    "index": len(refs) + 1,
                    "label": label,
                    "media_type": media_kind.lower(),
                    "path": "",
                    "role": "visual reference",
                    "description": binding,
                    "priority": 0,
                })

        # Prevent duplicate canonical labels from malformed legacy bindings while
        # retaining the first deterministic declaration.
        unique: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        for ref in refs:
            if ref["label"] in seen_labels:
                continue
            seen_labels.add(ref["label"])
            unique.append(ref)
        return unique

    @classmethod
    def _reference_definitions(
        cls,
        shot: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        definitions: list[str] = []
        bindings: list[str] = []
        refs = cls._canonical_references(shot)
        for ref in refs:
            detail = f"{ref['label']} is used as {ref['role']}"
            if ref.get("description") and ref["description"] != ref["role"]:
                detail += f": {ref['description']}"
            if ref.get("path"):
                detail += f" from {ref['path']}"
            definitions.append(detail + ".")
            bindings.append(f"{ref['label']} = {ref['role']}")
        return definitions, bindings

    @classmethod
    def _subject_definitions(cls, plan: dict[str, Any], shot: dict[str, Any]) -> str:
        characters_by_name = {
            cls._clean(c.get("name")): c
            for c in (plan.get("characters", []) or [])
            if isinstance(c, dict) and cls._clean(c.get("name"))
        }
        names = [cls._clean(v) for v in (shot.get("characters", []) or []) if cls._clean(v)]
        lines = []
        for name in names:
            character = characters_by_name.get(name, {})
            appearance = character.get("appearance", {}) if isinstance(character.get("appearance"), dict) else {}
            clothing = character.get("clothing", {}) if isinstance(character.get("clothing"), dict) else {}
            details = [
                name,
                cls._clean(character.get("role")),
                cls._clean(character.get("description")),
                cls._clean(character.get("personality")),
                cls._list_text(appearance.values() if appearance else []),
                cls._list_text(clothing.values() if clothing else []),
                cls._list_text(character.get("distinctive_features", [])),
                cls._list_text(character.get("continuity_rules", [])),
            ]
            details = [x for x in details if x]
            if details:
                lines.append(f"<Subject {len(lines) + 1}> is {'. '.join(details)}.")
        refs, bindings = cls._reference_definitions(shot)
        lines.extend(refs)
        if bindings:
            lines.extend(bindings)
        if not lines:
            lines.append("No explicit external reference subject is required for this shot.")
        return "\n".join(lines)

    @classmethod
    def _summary(cls, plan: dict[str, Any], shot: dict[str, Any]) -> str:
        parts = [
            "[cinematic shot generation]",
            f"Scene: {cls._clean(shot.get('scene_id'))}",
            f"Shot: {cls._clean(shot.get('shot_id'))}",
            f"Location: {cls._clean(shot.get('location'))}",
            f"Time: {cls._clean(shot.get('time_of_day'))}",
            f"Duration: {float(shot.get('duration_seconds', 0.0) or 0.0):.3f} seconds",
            f"Continuity mode: {cls._clean(shot.get('continuity_mode') or 'chained')}",
        ]
        production = cls._clean(plan.get("production_id"))
        if production:
            parts.insert(1, f"Production: {production}")
        return "; ".join(x for x in parts if x and not x.endswith(": ")) + "."

    @classmethod
    def _retention_analysis(cls, shot: dict[str, Any]) -> str:
        continuity = shot.get("continuity_start_state", {})
        if not isinstance(continuity, dict):
            continuity = {}
        required = []
        for ref in cls._canonical_references(shot):
            relationship = (
                "partially_preserved"
                if ref["role"] == "storyboard"
                else "fully_preserved"
            )
            description = ref["description"] or ref["role"]
            required.append(
                f"{ref['label']}: {relationship} - {description}."
            )
        for key, label in (
            ("location", "location"),
            ("lighting", "lighting"),
            ("environment", "environment"),
            ("camera_side", "camera side"),
            ("state_description", "state"),
        ):
            value = cls._clean(continuity.get(key))
            if value:
                required.append(f"Preserve {label}: {value}.")
        props = cls._list_text(continuity.get("props", []))
        if props:
            required.append(f"Preserve required props: {props}.")
        if not required:
            required.append("Preserve established character identity, wardrobe, setting, lighting, chronology, and scene-state continuity.")
        return " ".join(required)

    @classmethod
    def _detailed_description(cls, plan: dict[str, Any], shot: dict[str, Any]) -> str:
        scene = None
        for candidate in plan.get("scenes", []) or []:
            if isinstance(candidate, dict) and cls._clean(candidate.get("scene_id")) == cls._clean(shot.get("scene_id")):
                scene = candidate
                break
        scene = scene or {}

        description = (
            cls._clean(shot.get("detailed_description"))
            or cls._clean(shot.get("visual_prompt"))
            or cls._clean(shot.get("action"))
        )
        visual_analysis = plan.get("reference_visual_analysis", {}) or {}
        if isinstance(visual_analysis, dict) and visual_analysis:
            cues = []
            for item in visual_analysis.values():
                if not isinstance(item, dict):
                    continue
                cue_text = cls._clean(item.get("description"))
                if cue_text:
                    cues.append(cue_text)
            if cues:
                description = (description + " Reference-derived visible cues: " + "; ".join(cues[:4])).strip()
        camera = ", ".join(
            x for x in (
                cls._clean(shot.get("camera_shot")),
                cls._clean(shot.get("camera_movement")),
                cls._clean(shot.get("lens_and_depth_of_field")),
                cls._clean(shot.get("composition_notes")),
            ) if x
        )
        environment = ", ".join(
            x for x in (
                cls._clean(scene.get("atmosphere")),
                cls._list_text(scene.get("environment_details", []), 8),
                cls._list_text(scene.get("key_props", []), 8),
            ) if x
        )
        lines = [
            "Style: " + cls._clean(shot.get("mood") or scene.get("mood")) + ".",
            f"[Shot 1] {description or 'Depict the required story action in playback order.'}",
        ]
        if camera:
            lines.append(f"Camera: {camera}.")
        if environment:
            lines.append(f"Environment: {environment}.")
        lighting = ", ".join(
            x for x in (
                cls._clean(shot.get("lighting") or scene.get("lighting")),
                cls._clean(shot.get("color_temperature") or scene.get("color_temperature")),
            ) if x
        )
        if lighting:
            lines.append(f"Lighting: {lighting}.")
        ref_entries = cls._canonical_references(shot)
        if ref_entries:
            lines.append(
                "References: "
                + "; ".join(
                    f"{entry['label']} ({entry['role']})"
                    for entry in ref_entries
                )
                + "."
            )
        dialogue = []
        for event in shot.get("dialogue_events", []) or []:
            if not isinstance(event, dict):
                continue
            speaker = cls._clean(event.get("speaker"))
            text = str(event.get("text", "") or "").strip()
            if speaker and text:
                dialogue.append(f"(S1) {speaker}: <d>{text}</d>")
        speech = cls._clean(shot.get("speech_text"))
        if speech and not dialogue:
            dialogue.append(f"<d>{speech}</d>")
        lines.extend(dialogue)
        return " ".join(x for x in lines if x).strip()

    @classmethod
    def _soundscape(cls, shot: dict[str, Any]) -> str:
        value = cls._clean(shot.get("overall_soundscape"))
        return value or "Natural location ambience and physical sounds appropriate to the described action."

    @classmethod
    def _music(cls, shot: dict[str, Any]) -> str:
        value = cls._clean(shot.get("non_diegetic_music"))
        return value or "No non-diegetic music unless required by the production plan."

    @classmethod
    def validate(cls, context_ir: dict[str, Any]) -> None:
        if not isinstance(context_ir, dict):
            raise TypeError("H3 Context-IR must be a mapping.")
        version = int(context_ir.get("version", 0) or 0)
        if version != cls.VERSION:
            raise ValueError(
                f"Unsupported H3 Context-IR version: {version}; expected {cls.VERSION}."
            )
        if str(context_ir.get("mode", "")).strip().lower() != "ref2va":
            raise ValueError("H3 Context-IR mode must be ref2va for this repository.")
        prompt = str(context_ir.get("h3_prompt", "") or "").strip()
        if not prompt:
            raise ValueError("H3 Context-IR contains an empty h3_prompt.")
        expected = [f"{name}:" for name in cls.REQUIRED_SECTIONS]
        positions = []
        for marker in expected:
            position = prompt.find(marker)
            if position < 0:
                raise ValueError(f"H3 Context-IR prompt is missing required section: {marker}")
            positions.append(position)
        if positions != sorted(positions):
            raise ValueError("H3 Context-IR sections are out of order.")
        sections = context_ir.get("sections")
        if not isinstance(sections, dict):
            raise ValueError("H3 Context-IR must expose its six named sections.")
        if tuple(sections.keys()) != cls.REQUIRED_SECTIONS:
            raise ValueError("H3 Context-IR section mapping is incomplete or out of order.")
        for section in cls.REQUIRED_SECTIONS:
            if not str(sections.get(section, "") or "").strip():
                raise ValueError(f"H3 Context-IR section is empty: {section}")

        # Every declared reference must use a canonical media-specific label,
        # and every canonical label used by the six-section prompt must be declared
        # exactly once in the structured registry.
        references = context_ir.get("references", []) or []
        if not isinstance(references, list):
            raise ValueError("H3 Context-IR references must be a list.")

        declared: dict[str, dict[str, Any]] = {}
        expected_pattern = re.compile(r"^<(Picture|Video|Audio)\s+(\d+)>$")
        for item in references:
            if not isinstance(item, dict):
                raise ValueError("H3 Context-IR reference entries must be objects.")
            label = self_label = cls._clean(item.get("label"))
            match = expected_pattern.fullmatch(label)
            if not match:
                raise ValueError(
                    f"Invalid canonical reference label: {label!r}"
                )
            media_kind = match.group(1).lower()
            if label in declared:
                raise ValueError(
                    f"Duplicate Context-IR reference label: {label}"
                )
            item_type = cls._clean(item.get("media_type"))
            if item_type and item_type != media_kind:
                raise ValueError(
                    f"Context-IR reference type mismatch for {label}: "
                    f"media_type={item_type!r}."
                )
            declared[label] = item

        prompt = str(context_ir.get("h3_prompt", "") or "")
        used = set(
            re.findall(
                r"<(?:Picture|Video|Audio)\s+\d+>",
                prompt,
            )
        )
        unknown = sorted(used - set(declared))
        missing = sorted(set(declared) - used)
        if unknown:
            raise ValueError(
                "H3 Context-IR contains undeclared reference labels: "
                + ", ".join(unknown)
            )
        if missing:
            raise ValueError(
                "H3 Context-IR declares references never used: "
                + ", ".join(missing)
            )

        sections = context_ir.get("sections", {})
        semantic_sections = (
            "subject_definitions",
            "retention_analysis",
            "detailed_description",
        )
        for label in declared:
            missing_sections = [
                name
                for name in semantic_sections
                if label not in str(sections.get(name, "") or "")
            ]
            if missing_sections:
                raise ValueError(
                    f"Context-IR reference {label} is missing from required "
                    f"section(s): {', '.join(missing_sections)}."
                )
    @classmethod
    def prompt(cls, context_ir: dict[str, Any]) -> str:
        cls.validate(context_ir)
        return str(context_ir["h3_prompt"]).strip()

    def compile(self, plan: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
        sections = {
            "subject_definitions": self._subject_definitions(plan, shot),
            "summary": self._summary(plan, shot),
            "retention_analysis": self._retention_analysis(shot),
            "detailed_description": self._detailed_description(plan, shot),
            "overall_soundscape": self._soundscape(shot),
            "non_diegetic_music": self._music(shot),
        }
        prompt = "\n\n".join(
            f"{name}:\n{sections[name]}"
            for name in self.REQUIRED_SECTIONS
        ).strip()

        refs = []
        for ref in self._canonical_references(shot):
            refs.append({
                "index": int(ref["index"]),
                "label": ref["label"],
                "media_type": ref["media_type"],
                "binding": f"{ref['label']} = {ref['role']}",
                "role": ref["role"],
                "path": ref["path"],
                "description": ref["description"],
                "priority": int(ref["priority"]),
            })

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
            "references": refs,
            "audio": {
                "soundscape": self._soundscape(shot),
                "music": self._music(shot),
            },
        }
        self.validate(result)
        return result

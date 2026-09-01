from __future__ import annotations

from typing import Any
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
    def _reference_definitions(cls, shot: dict[str, Any]) -> tuple[list[str], list[str]]:
        bindings = []
        for raw in shot.get("reference_bindings", []) or []:
            text = cls._clean(raw)
            if text:
                bindings.append(text)

        roles = [x for x in (shot.get("reference_roles", []) or []) if isinstance(x, dict)]
        if roles:
            labels = []
            for index, role in enumerate(roles, start=1):
                label = cls._clean(role.get("label"))
                kind = cls._clean(role.get("role")) or "visual reference"
                path = cls._clean(role.get("path"))
                if not label:
                    label = f"Picture {index}"
                if label.lower().startswith("picture"):
                    ref_label = f"<{label}>"
                elif label.lower().startswith(("video ", "audio ")):
                    ref_label = f"<{label}>"
                else:
                    ref_label = f"<Picture {index}>"
                detail = f"{ref_label} is used as {kind}"
                if path:
                    detail += f" from {path}"
                labels.append(detail + ".")
            return labels, bindings

        return [], bindings

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
        roles = [r for r in (shot.get("reference_roles", []) or []) if isinstance(r, dict)]
        for index, role in enumerate(roles, start=1):
            label = cls._clean(role.get("label")) or f"Picture {index}"
            role_name = cls._clean(role.get("role")) or "visual reference"
            relationship = "fully_preserved"
            if role_name in {"storyboard", "visual_reference"}:
                relationship = "partially_preserved" if role_name == "storyboard" else "fully_preserved"
            description = cls._clean(role.get("description") or role.get("label") or role_name)
            required.append(f"<{label}>: {relationship} - {description}.")
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

        # Every declared picture/video/audio label must be canonical and every
        # angle-bracket reference used anywhere in the six sections must refer
        # to one of those declarations. This prevents silent reference drift.
        reference_labels = []
        for item in context_ir.get("references", []) or []:
            if not isinstance(item, dict):
                raise ValueError("H3 Context-IR reference entries must be objects.")
            index = int(item.get("index", 0) or 0)
            if index <= 0:
                raise ValueError("H3 Context-IR reference index must be positive.")
            label = f"<Picture {index}>"
            if label in reference_labels:
                raise ValueError(f"Duplicate Context-IR reference label: {label}")
            reference_labels.append(label)

        declared = set(reference_labels)
        tokens = set(re.findall(r"<(?:Picture|Video|Audio)\s+\d+>", prompt))
        unknown = sorted(tokens - declared)
        if unknown:
            raise ValueError(
                "H3 Context-IR contains undeclared reference labels: "
                + ", ".join(unknown)
            )
        if reference_labels:
            for label in reference_labels:
                if label not in prompt:
                    raise ValueError(f"Declared Context-IR reference is never used: {label}")

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
        for index, binding in enumerate(shot.get("reference_bindings", []) or [], start=1):
            refs.append({"index": index, "binding": self._clean(binding)})

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

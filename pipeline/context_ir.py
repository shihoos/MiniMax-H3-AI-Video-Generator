from __future__ import annotations

from pathlib import Path
from typing import Any
import re


class H3ContextIRCompiler:
    """Compile canonical production state into MiniMax H3 Ref2VA Context-IR.

    This compiler is intentionally deterministic.  The repository's creative
    planner remains responsible for story/camera choices; this layer is the
    semantic adapter that makes entities, references, speakers, timing, and
    sound relationships explicit before the official MiniMax Context-IR API
    refines the multimodal prompt for H3.

    Public compatibility surface is intentionally unchanged:
      H3ContextIRCompiler().compile(plan, shot)
      H3ContextIRCompiler.validate(context_ir)
      H3ContextIRCompiler.prompt(context_ir)
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

    VISUAL_RELATIONSHIPS = {
        "fully_preserved",
        "partially_preserved",
        "attribute_transfer",
        "weak_reference",
    }
    AUDIO_RELATIONSHIPS = {
        "fully_copy",
        "partially_copy",
        "reference",
        "weak_reference",
    }
    REF_PATTERN = re.compile(r"^<(Picture|Video|Audio)\s+(\d+)>$")
    REF_TOKEN_PATTERN = re.compile(r"<(?:Picture|Video|Audio)\s+\d+>")
    SPEAKER_TOKEN_PATTERN = re.compile(r"\(S\d+\)")

    @staticmethod
    def _clean(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip())

    @classmethod
    def _list_text(cls, value: Any, limit: int = 8) -> str:
        values: list[str] = []
        if isinstance(value, dict):
            iterable = value.values()
        elif isinstance(value, (list, tuple, set)):
            iterable = value
        else:
            iterable = []
        for item in iterable:
            text = cls._clean(item)
            if text and text not in values:
                values.append(text)
            if len(values) >= limit:
                break
        return "; ".join(values)

    @classmethod
    def _path_key(cls, value: Any) -> str:
        text = cls._clean(value)
        if not text:
            return ""
        try:
            return str(Path(text).resolve())
        except (OSError, RuntimeError, ValueError):
            return text

    @classmethod
    def _iter_mapping_paths(cls, mapping: Any) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        if not isinstance(mapping, dict):
            return rows
        for owner, paths in mapping.items():
            owner_name = cls._clean(owner)
            if isinstance(paths, str):
                paths = [paths]
            if not isinstance(paths, (list, tuple, set)):
                continue
            for path in paths:
                p = cls._clean(path)
                if p:
                    rows.append((owner_name, p))
        return rows

    @classmethod
    def _find_character(cls, plan: dict[str, Any], name: str) -> dict[str, Any]:
        target = cls._clean(name).lower()
        for character in plan.get("characters", []) or []:
            if not isinstance(character, dict):
                continue
            if cls._clean(character.get("name")).lower() == target:
                return character
        return {}

    @classmethod
    def _reference_kind(cls, role: dict[str, Any], path: str) -> str:
        raw = cls._clean(
            role.get("media_type") or role.get("type") or role.get("kind")
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
    def _visual_role_for_path(cls, shot: dict[str, Any], path: str) -> dict[str, Any]:
        key = cls._path_key(path)
        for role in shot.get("reference_roles", []) or []:
            if not isinstance(role, dict):
                continue
            if cls._path_key(role.get("path") or role.get("source")) == key:
                return dict(role)
        return {}

    @classmethod
    def _relationship_for(cls, media_type: str, role: str, explicit: Any = None) -> str:
        chosen = cls._clean(explicit).lower()
        if media_type == "audio":
            if chosen in cls.AUDIO_RELATIONSHIPS:
                return chosen
            if "copy" in role or "reuse" in role or "source audio" in role:
                return "fully_copy"
            return "reference"

        if chosen in cls.VISUAL_RELATIONSHIPS:
            return chosen
        role_lower = role.lower()
        if "storyboard" in role_lower or "scene layout" in role_lower:
            return "partially_preserved"
        if "identity" in role_lower or "continuity" in role_lower:
            return "fully_preserved"
        if "motion" in role_lower or "camera" in role_lower or "style" in role_lower:
            return "weak_reference"
        if "attribute" in role_lower:
            return "attribute_transfer"
        if "retake_start" in role_lower or "retake_end" in role_lower:
            return "fully_preserved"
        return "weak_reference"

    @classmethod
    def _canonical_references(cls, shot: dict[str, Any]) -> list[dict[str, Any]]:
        """Build one deterministic semantic registry in runtime input order.

        The existing runtime schema already owns the ordered media lists and the
        per-character video/audio mappings.  We derive semantic metadata from
        those fields rather than introducing another competing reference store.
        """
        raw_roles = [
            dict(role)
            for role in (shot.get("reference_roles", []) or [])
            if isinstance(role, dict)
        ]
        role_by_path: dict[str, dict[str, Any]] = {}
        for role in raw_roles:
            key = cls._path_key(role.get("path") or role.get("source"))
            if key and key not in role_by_path:
                role_by_path[key] = role

        video_owner: dict[str, str] = {}
        for owner, path in cls._iter_mapping_paths(shot.get("reference_video_by_character")):
            video_owner.setdefault(cls._path_key(path), owner)

        audio_owner: dict[str, str] = {}
        for owner, path in cls._iter_mapping_paths(shot.get("reference_audio_by_character")):
            audio_owner.setdefault(cls._path_key(path), owner)

        refs: list[dict[str, Any]] = []
        counters = {"Picture": 0, "Video": 0, "Audio": 0}
        seen: set[str] = set()

        def append_ref(path: str, media_kind: str, extra: dict[str, Any] | None = None) -> None:
            clean_path = cls._clean(path)
            if not clean_path:
                return
            key = f"{media_kind}:{cls._path_key(clean_path)}"
            if key in seen:
                return
            seen.add(key)
            counters[media_kind] += 1
            role = dict(extra or role_by_path.get(cls._path_key(clean_path), {}))
            role_name = cls._clean(role.get("role"))
            owner = cls._clean(
                role.get("character_name")
                or role.get("entity_name")
                or (video_owner if media_kind == "Video" else audio_owner).get(cls._path_key(clean_path), "")
            )
            description = cls._clean(
                role.get("description")
                or role.get("label")
                or role_name
            )
            if not role_name:
                if media_kind == "Video" and owner:
                    role_name = "motion reference"
                    description = description or f"Motion and temporal reference for {owner}."
                elif media_kind == "Audio" and owner:
                    role_name = "voice timbre reference"
                    description = description or f"Voice timbre reference for {owner}."
                else:
                    role_name = "visual reference" if media_kind == "Picture" else "reference"
            relationship = cls._relationship_for(
                media_kind.lower(),
                role_name,
                role.get("relationship") or role.get("retention"),
            )
            refs.append({
                "index": len(refs) + 1,
                "label": f"<{media_kind} {counters[media_kind]}>",
                "media_type": media_kind.lower(),
                "path": clean_path,
                "role": role_name,
                "description": description or role_name,
                "relationship": relationship,
                "character_name": owner,
                "character_id": cls._clean(role.get("character_id") or role.get("entity_id")),
                "priority": int(role.get("priority", 0) or 0),
            })

        for path in shot.get("reference_images", []) or []:
            append_ref(path, "Picture")
        for path in shot.get("reference_videos", []) or []:
            append_ref(
                path,
                "Video",
                role_by_path.get(cls._path_key(path), {})
                or {"character_name": video_owner.get(cls._path_key(path), "")},
            )
        for path in shot.get("reference_audio_paths", []) or []:
            append_ref(
                path,
                "Audio",
                role_by_path.get(cls._path_key(path), {})
                or {"character_name": audio_owner.get(cls._path_key(path), "")},
            )

        # Legacy fallback only when the runtime media lists are absent.
        if not refs:
            for role in raw_roles:
                path = cls._clean(role.get("path") or role.get("source"))
                if path:
                    append_ref(path, cls._reference_kind(role, path), role)

        if not refs:
            bindings = [
                cls._clean(raw)
                for raw in (shot.get("reference_bindings", []) or [])
                if cls._clean(raw)
            ]
            for binding in bindings:
                match = re.search(
                    r"<(Picture|Video|Audio)\s+(\d+)>\s*=\s*(.+)$",
                    binding,
                    flags=re.IGNORECASE,
                )
                if match:
                    kind = match.group(1).capitalize()
                    label_no = int(match.group(2))
                    counters[kind] = max(counters[kind], label_no)
                    refs.append({
                        "index": len(refs) + 1,
                        "label": f"<{kind} {label_no}>",
                        "media_type": kind.lower(),
                        "path": "",
                        "role": cls._clean(match.group(3)) or "reference",
                        "description": cls._clean(match.group(3)) or "reference",
                        "relationship": cls._relationship_for(kind.lower(), cls._clean(match.group(3))),
                        "character_name": "",
                        "character_id": "",
                        "priority": 0,
                    })

        return refs

    @classmethod
    def _reference_map(cls, refs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(ref["label"]): ref for ref in refs}

    @classmethod
    def _speaker_map(cls, shot: dict[str, Any]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        next_id = 1
        for event in shot.get("dialogue_events", []) or []:
            if not isinstance(event, dict):
                continue
            raw_id = cls._clean(event.get("speaker_id") or event.get("speaker") or event.get("speaker_name"))
            if not raw_id:
                continue
            if raw_id.startswith("S") and raw_id[1:].isdigit():
                mapping.setdefault(raw_id, raw_id)
                continue
            if raw_id not in mapping:
                mapping[raw_id] = f"S{next_id}"
                next_id += 1
        return mapping

    @classmethod
    def _speaker_name(cls, event: dict[str, Any]) -> str:
        return cls._clean(event.get("speaker_name") or event.get("speaker") or event.get("character_name"))

    @classmethod
    def _language(cls, plan: dict[str, Any], shot: dict[str, Any], event: dict[str, Any] | None = None) -> str:
        event = event or {}
        for candidate in (
            event.get("language"),
            shot.get("language"),
            shot.get("dialogue_language"),
            plan.get("language"),
            plan.get("dialogue_language"),
        ):
            value = cls._clean(candidate)
            if value:
                return value
        return "English"

    @classmethod
    def _subject_definitions(
        cls,
        plan: dict[str, Any],
        shot: dict[str, Any],
        refs: list[dict[str, Any]],
    ) -> str:
        names = [cls._clean(v) for v in (shot.get("characters", []) or []) if cls._clean(v)]
        refs_by_owner: dict[str, list[dict[str, Any]]] = {}
        for ref in refs:
            owner = cls._clean(ref.get("character_name"))
            if owner:
                refs_by_owner.setdefault(owner.lower(), []).append(ref)

        speaker_names: set[str] = set()
        for event in shot.get("dialogue_events", []) or []:
            if isinstance(event, dict):
                speaker = cls._speaker_name(event)
                if speaker:
                    speaker_names.add(speaker.lower())

        speaker_map = cls._speaker_map(shot)
        lines: list[str] = []
        subject_index = 0
        defined_reference_labels: set[str] = set()

        for name in names:
            character = cls._find_character(plan, name)
            appearance = character.get("appearance", {}) if isinstance(character.get("appearance"), dict) else {}
            clothing = character.get("clothing", {}) if isinstance(character.get("clothing"), dict) else {}
            details = [
                cls._clean(character.get("role")),
                cls._clean(character.get("description")),
                cls._clean(character.get("personality")),
                cls._list_text(appearance.values() if appearance else []),
                cls._list_text(clothing.values() if clothing else []),
                cls._list_text(character.get("distinctive_features", [])),
                cls._list_text(character.get("continuity_rules", [])),
            ]
            details = [x for x in details if x]
            owner_refs = refs_by_owner.get(name.lower(), [])
            visual_refs = [r for r in owner_refs if r["media_type"] in {"picture", "video"}]
            audio_refs = [r for r in owner_refs if r["media_type"] == "audio"]

            # A <Subject N> is reserved for reusable referenced visible content.
            if visual_refs:
                subject_index += 1
                subject_label = f"<Subject {subject_index}>"
                sentence = f"{subject_label} is {name}"
                if details:
                    sentence += ", " + ", ".join(details)
                sentence += "."
                lines.append(sentence)
                for ref in visual_refs:
                    defined_reference_labels.add(ref["label"])
                    if ref["media_type"] == "picture":
                        lines.append(
                            f"{ref['label']} is the visual identity or appearance reference for {name}; "
                            f"its role is {ref['role']}, with a {ref['relationship']} relationship."
                        )
                    else:
                        lines.append(
                            f"{ref['label']} is the temporal or motion reference for {name}; "
                            f"its role is {ref['role']}, with a {ref['relationship']} relationship."
                        )
            elif details:
                # Unreferenced subjects remain ordinary textual subjects rather than
                # inventing a reference-backed <Subject N> label.
                sentence = f"{name} is " + ", ".join(details) + "."
                lines.append(sentence)

            if name.lower() in speaker_names:
                raw_candidates = [
                    cls._clean(event.get("speaker_id"))
                    for event in (shot.get("dialogue_events", []) or [])
                    if isinstance(event, dict) and cls._speaker_name(event).lower() == name.lower()
                ]
                raw_candidates = [x for x in raw_candidates if x]
                speaker_id = speaker_map.get(raw_candidates[0], "") if raw_candidates else ""
                if speaker_id:
                    lines.append(f"{name} is the speaker identified as ({speaker_id}) for dialogue in this shot.")

            for ref in audio_refs:
                defined_reference_labels.add(ref["label"])
                lines.append(
                    f"{ref['label']} is an audio reference for {name}; its role is {ref['role']}, "
                    f"with a {ref['relationship']} relationship."
                )

        # References with no character owner still need a stable definition.
        for ref in refs:
            if ref["label"] in defined_reference_labels:
                continue
            role = ref["role"]
            description = ref["description"] or role
            if ref["media_type"] == "picture":
                lines.append(
                    f"{ref['label']} is a concrete visual reference for the shot; "
                    f"role: {role}; {description}; relationship: {ref['relationship']}."
                )
            elif ref["media_type"] == "video":
                lines.append(
                    f"{ref['label']} is a source video or temporal reference for the shot; "
                    f"role: {role}; {description}; relationship: {ref['relationship']}."
                )
            else:
                lines.append(
                    f"{ref['label']} is an audio reference for the shot; "
                    f"role: {role}; {description}; relationship: {ref['relationship']}."
                )

        if not lines:
            lines.append("No explicit external reference is required for this shot.")
        return "\n".join(lines)

    @classmethod
    def _task_types(cls, refs: list[dict[str, Any]], shot: dict[str, Any]) -> list[str]:
        types: list[str] = []
        role_blob = " ".join(cls._clean(r.get("role")) for r in refs).lower()
        relation_blob = " ".join(cls._clean(r.get("relationship")) for r in refs).lower()
        if any("edit" in cls._clean(r.get("role")).lower() or "source video" in cls._clean(r.get("role")).lower() for r in refs if r["media_type"] == "video"):
            types.append("video editing")
        elif any(r["media_type"] == "video" for r in refs) and any("motion" in cls._clean(r.get("role")).lower() or "camera" in cls._clean(r.get("role")).lower() for r in refs):
            types.append("reference generation")
        elif any(r["media_type"] == "picture" for r in refs):
            types.append("reference generation")
        elif refs:
            types.append("reference generation")
        if any(r["media_type"] == "audio" and r["relationship"] in {"fully_copy", "partially_copy"} for r in refs):
            types.append("audio reuse")
        if any(r["media_type"] == "audio" and r["relationship"] in {"reference", "weak_reference"} for r in refs):
            types.append("audio reference")
        if cls._clean(shot.get("continuity_mode")).lower() in {"continuation", "chained"} and any(r["role"] in {"retake_start_frame", "retake_end_frame"} for r in refs):
            types.append("video continuation")
        if not types:
            types.append("cinematic shot generation")
        # stable de-duplication
        return list(dict.fromkeys(types))

    @classmethod
    def _summary(cls, plan: dict[str, Any], shot: dict[str, Any], refs: list[dict[str, Any]]) -> str:
        parts = [
            "[" + " + ".join(cls._task_types(refs, shot)) + "]",
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
        if refs:
            relationships = "; ".join(
                f"{ref['label']} {ref['role']} ({ref['relationship']})"
                for ref in refs
            )
            parts.append(f"Reference relationships: {relationships}")
        return "; ".join(x for x in parts if x and not x.endswith(": ")) + "."

    @classmethod
    def _retention_analysis(cls, shot: dict[str, Any], refs: list[dict[str, Any]]) -> str:
        continuity = shot.get("continuity_start_state", {})
        if not isinstance(continuity, dict):
            continuity = {}
        required: list[str] = []
        for ref in refs:
            required.append(
                f"{ref['label']}: {ref['relationship']} - {ref['description'] or ref['role']}."
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
        if cls._clean(shot.get("continuity_notes")):
            required.append(f"Continuity notes: {cls._clean(shot.get('continuity_notes'))}.")
        if not required:
            required.append("Preserve established identity, wardrobe, setting, lighting, chronology, and scene-state continuity.")
        return " ".join(required)

    @classmethod
    def _dialogue_lines(cls, plan: dict[str, Any], shot: dict[str, Any]) -> list[str]:
        lines: list[str] = []
        speaker_map = cls._speaker_map(shot)
        for event in shot.get("dialogue_events", []) or []:
            if not isinstance(event, dict):
                continue
            text = str(event.get("text", "") or "").strip()
            speaker_name = cls._speaker_name(event)
            raw_speaker = cls._clean(event.get("speaker_id") or speaker_name)
            speaker_id = speaker_map.get(raw_speaker, raw_speaker if re.fullmatch(r"S\d+", raw_speaker) else "")
            if not text or not speaker_id:
                continue
            language = cls._language(plan, shot, event)
            start = float(event.get("start_seconds", 0.0) or 0.0)
            end = float(event.get("end_seconds", 0.0) or 0.0)
            timing = f" at {start:.2f} seconds" if end <= start else f" from {start:.2f} to {end:.2f} seconds"
            speaker_phrase = f"{speaker_name} ({speaker_id})" if speaker_name else f"({speaker_id})"
            continuation = " Continue this dialogue into the next shot." if bool(event.get("continues_to_next_shot", False)) else ""
            lines.append(
                f"At{timing}, {speaker_phrase} says <d>[{language}] {text}</d>.{continuation}"
            )
        if not lines:
            speech = cls._clean(shot.get("speech_text"))
            if speech:
                lines.append(f"Spoken dialogue: <d>[{cls._language(plan, shot)}] {speech}</d>.")
        return lines

    @classmethod
    def _detailed_description(cls, plan: dict[str, Any], shot: dict[str, Any], refs: list[dict[str, Any]]) -> str:
        scene: dict[str, Any] = {}
        target_scene = cls._clean(shot.get("scene_id"))
        for candidate in plan.get("scenes", []) or []:
            if isinstance(candidate, dict) and cls._clean(candidate.get("scene_id")) == target_scene:
                scene = candidate
                break

        description = (
            cls._clean(shot.get("detailed_description"))
            or cls._clean(shot.get("visual_prompt"))
            or cls._clean(shot.get("action"))
        )
        if not description:
            description = "Depict the required story action continuously in playback order."

        visual_analysis = plan.get("reference_visual_analysis", {}) or {}
        if isinstance(visual_analysis, dict):
            cues: list[str] = []
            for item in visual_analysis.values():
                if isinstance(item, dict):
                    cue = cls._clean(item.get("description"))
                    if cue:
                        cues.append(cue)
            if cues:
                description += " Reference-derived visible cues: " + "; ".join(cues[:4]) + "."

        paragraphs = [f"[Shot 1] {description}"]
        camera = ", ".join(
            x for x in (
                cls._clean(shot.get("camera_shot")),
                cls._clean(shot.get("camera_movement")),
                cls._clean(shot.get("lens_and_depth_of_field")),
                cls._clean(shot.get("composition_notes")),
            ) if x
        )
        if camera:
            paragraphs.append(f"Camera and composition: {camera}.")

        environment = ", ".join(
            x for x in (
                cls._clean(shot.get("location")),
                cls._clean(scene.get("atmosphere")),
                cls._list_text(scene.get("environment_details", []), 8),
                cls._list_text(scene.get("key_props", []), 8),
            ) if x
        )
        if environment:
            paragraphs.append(f"Environment: {environment}.")

        lighting = ", ".join(
            x for x in (
                cls._clean(shot.get("lighting") or scene.get("lighting")),
                cls._clean(shot.get("color_temperature") or scene.get("color_temperature")),
                cls._clean(shot.get("mood") or scene.get("mood")),
            ) if x
        )
        if lighting:
            paragraphs.append(f"Lighting and visual tone: {lighting}.")

        continuity = shot.get("continuity_start_state", {})
        if isinstance(continuity, dict):
            continuity_items = []
            for key in ("location", "lighting", "environment", "camera_side", "state_description"):
                value = cls._clean(continuity.get(key))
                if value:
                    continuity_items.append(f"{key.replace('_', ' ')}={value}")
            props = cls._list_text(continuity.get("props", []), 8)
            if props:
                continuity_items.append(f"props={props}")
            if continuity_items:
                paragraphs.append("Continuity state: " + "; ".join(continuity_items) + ".")

        if refs:
            effect_lines = []
            for ref in refs:
                owner = f" for {ref['character_name']}" if ref.get("character_name") else ""
                effect_lines.append(
                    f"{ref['label']} affects the shot{owner} as {ref['role']} with {ref['relationship']} semantics."
                )
            paragraphs.append("Reference application: " + " ".join(effect_lines))

        dialogue = cls._dialogue_lines(plan, shot)
        if dialogue:
            paragraphs.extend(dialogue)

        action = cls._clean(shot.get("action"))
        if action and action.lower() not in description.lower():
            paragraphs.append(f"Action and state change: {action}.")

        return " ".join(p.strip() for p in paragraphs if p.strip())

    @classmethod
    def _soundscape(cls, shot: dict[str, Any], refs: list[dict[str, Any]]) -> str:
        value = cls._clean(shot.get("overall_soundscape"))
        audio_refs = [r for r in refs if r["media_type"] == "audio"]
        if value:
            if audio_refs:
                return value.rstrip(".") + ". Audio references in effect: " + "; ".join(
                    f"{r['label']} ({r['role']}, {r['relationship']})" for r in audio_refs
                ) + "."
            return value
        if audio_refs:
            return "Natural location ambience and physical sounds appropriate to the described action, with " + "; ".join(
                f"{r['label']} used as {r['role']} ({r['relationship']})" for r in audio_refs
            ) + "."
        return "Natural location ambience and physical sounds appropriate to the described action."

    @classmethod
    def _music(cls, shot: dict[str, Any], refs: list[dict[str, Any]]) -> str:
        value = cls._clean(shot.get("non_diegetic_music"))
        music_refs = [
            r for r in refs
            if r["media_type"] == "audio" and any(token in r["role"].lower() for token in ("music", "score", "soundtrack"))
        ]
        if value:
            if music_refs:
                return value.rstrip(".") + ". Related references: " + "; ".join(
                    f"{r['label']} ({r['relationship']})" for r in music_refs
                ) + "."
            return value
        if music_refs:
            return "Non-diegetic music uses " + "; ".join(
                f"{r['label']} as {r['role']} ({r['relationship']})" for r in music_refs
            ) + "."
        return "No non-diegetic music unless required by the production plan."

    @classmethod
    def validate(cls, context_ir: dict[str, Any]) -> None:
        if not isinstance(context_ir, dict):
            raise TypeError("H3 Context-IR must be a mapping.")
        version = int(context_ir.get("version", 0) or 0)
        if version != cls.VERSION:
            raise ValueError(f"Unsupported H3 Context-IR version: {version}; expected {cls.VERSION}.")
        if cls._clean(context_ir.get("mode")).lower() != "ref2va":
            raise ValueError("H3 Context-IR mode must be ref2va for this repository.")

        prompt = cls._clean(context_ir.get("h3_prompt"))
        if not prompt:
            raise ValueError("H3 Context-IR contains an empty h3_prompt.")
        sections = context_ir.get("sections")
        if not isinstance(sections, dict):
            raise ValueError("H3 Context-IR must expose its six named sections.")
        if tuple(sections.keys()) != cls.REQUIRED_SECTIONS:
            raise ValueError("H3 Context-IR section mapping is incomplete or out of order.")
        positions: list[int] = []
        for section in cls.REQUIRED_SECTIONS:
            marker = f"{section}:"
            pos = prompt.find(marker)
            if pos < 0:
                raise ValueError(f"H3 Context-IR prompt is missing required section: {marker}")
            positions.append(pos)
            if not cls._clean(sections.get(section)):
                raise ValueError(f"H3 Context-IR section is empty: {section}")
        if positions != sorted(positions):
            raise ValueError("H3 Context-IR sections are out of order.")

        references = context_ir.get("references", []) or []
        if not isinstance(references, list):
            raise ValueError("H3 Context-IR references must be a list.")
        declared: dict[str, dict[str, Any]] = {}
        media_counts = {"picture": 0, "video": 0, "audio": 0}
        for item in references:
            if not isinstance(item, dict):
                raise ValueError("H3 Context-IR reference entries must be objects.")
            label = cls._clean(item.get("label"))
            match = cls.REF_PATTERN.fullmatch(label)
            if not match:
                raise ValueError(f"Invalid canonical reference label: {label!r}")
            if label in declared:
                raise ValueError(f"Duplicate Context-IR reference label: {label}")
            media_kind = match.group(1).lower()
            media_counts[media_kind] += 1
            if cls._clean(item.get("media_type")) not in {"", media_kind}:
                raise ValueError(f"Context-IR reference type mismatch for {label}.")
            relationship = cls._clean(item.get("relationship")).lower()
            allowed = cls.AUDIO_RELATIONSHIPS if media_kind == "audio" else cls.VISUAL_RELATIONSHIPS
            if relationship not in allowed:
                raise ValueError(f"Invalid {media_kind} relationship for {label}: {relationship!r}")
            declared[label] = item

        if media_counts["picture"] > 9 or media_counts["video"] > 3 or media_counts["audio"] > 3:
            raise ValueError("Context-IR exceeds Ref2VA per-media reference limits (9 pictures, 3 videos, 3 audio).")
        if sum(media_counts.values()) > 12:
            raise ValueError("Context-IR exceeds the Ref2VA mixed reference limit of 12 files.")
        if media_counts["audio"] and not (media_counts["picture"] or media_counts["video"]):
            raise ValueError("Ref2VA audio references cannot be the only reference modality.")

        used = set(cls.REF_TOKEN_PATTERN.findall(prompt))
        unknown = sorted(used - set(declared))
        if unknown:
            raise ValueError("H3 Context-IR contains undeclared reference labels: " + ", ".join(unknown))
        missing = sorted(set(declared) - used)
        if missing:
            raise ValueError("H3 Context-IR declares references never used: " + ", ".join(missing))

        # Reference labels should be visible in the reference-aware semantic sections.
        for label in declared:
            for section in ("subject_definitions", "summary", "retention_analysis", "detailed_description"):
                if label not in str(sections.get(section, "") or ""):
                    raise ValueError(f"Context-IR reference {label} is missing from {section}.")

        # Reject internal runtime paths leaking into the multimodal prompt.
        for item in references:
            path = cls._clean(item.get("path"))
            if path and path in prompt:
                raise ValueError(f"Internal reference path leaked into Context-IR prompt: {path}")

        dialogue_section = str(sections.get("detailed_description", "") or "")
        dialogue_events = context_ir.get("dialogue", []) or []
        if dialogue_events:
            for event in dialogue_events:
                speaker_id = cls._clean(event.get("speaker_id"))
                if speaker_id and not re.fullmatch(r"S\d+", speaker_id):
                    raise ValueError(f"Non-canonical speaker ID in Context-IR: {speaker_id}")
        for marker in re.findall(r"<d>.*?</d>", dialogue_section):
            if not re.search(r"<d>\[[^\]]+\]\s*.+</d>", marker, flags=re.DOTALL):
                raise ValueError("Dialogue must use <d>[Language] text</d> formatting.")
        speaker_ids = cls.SPEAKER_TOKEN_PATTERN.findall(dialogue_section)
        if speaker_ids:
            # Repeated speakers are allowed; the mapping itself must be stable.
            first_seen: dict[str, str] = {}
            for token in speaker_ids:
                first_seen.setdefault(token, token)

    @classmethod
    def prompt(cls, context_ir: dict[str, Any]) -> str:
        cls.validate(context_ir)
        return str(context_ir["h3_prompt"]).strip()

    def compile(self, plan: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any]:
        refs = self._canonical_references(shot)
        speaker_map = self._speaker_map(shot)
        sections = {
            "subject_definitions": self._subject_definitions(plan, shot, refs),
            "summary": self._summary(plan, shot, refs),
            "retention_analysis": self._retention_analysis(shot, refs),
            "detailed_description": self._detailed_description(plan, shot, refs),
            "overall_soundscape": self._soundscape(shot, refs),
            "non_diegetic_music": self._music(shot, refs),
        }
        prompt = "\n\n".join(
            f"{name}:\n{sections[name]}" for name in self.REQUIRED_SECTIONS
        ).strip()
        reference_rows = []
        for ref in refs:
            reference_rows.append({
                "index": int(ref["index"]),
                "label": ref["label"],
                "media_type": ref["media_type"],
                "binding": f"{ref['label']} {ref['role']}; relationship={ref['relationship']}",
                "role": ref["role"],
                "relationship": ref["relationship"],
                "path": ref["path"],
                "description": ref["description"],
                "character_name": ref.get("character_name", ""),
                "character_id": ref.get("character_id", ""),
                "priority": int(ref["priority"]),
            })

        dialogue_rows = []
        for event in shot.get("dialogue_events", []) or []:
            if not isinstance(event, dict):
                continue
            raw = self._clean(event.get("speaker_id") or event.get("speaker_name") or event.get("speaker"))
            if not raw:
                continue
            dialogue_rows.append({
                "speaker_id": speaker_map.get(raw, raw if re.fullmatch(r"S\d+", raw) else ""),
                "speaker_name": self._speaker_name(event),
                "language": self._language(plan, shot, event),
                "text": str(event.get("text", "") or ""),
                "start_seconds": float(event.get("start_seconds", 0.0) or 0.0),
                "end_seconds": float(event.get("end_seconds", 0.0) or 0.0),
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
            "references": reference_rows,
            "speakers": speaker_map,
            "dialogue": dialogue_rows,
            "audio": {
                "soundscape": self._soundscape(shot, refs),
                "music": self._music(shot, refs),
            },
        }
        self.validate(result)
        return result

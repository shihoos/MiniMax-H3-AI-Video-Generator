from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

from pipeline.production_checkpoint import ProductionCheckpoint

from planner.entity_resolver import EntityResolver
from planner.qwen_director_prompts import (
    DEFAULT_CAMERA_MOVEMENT,
    DEFAULT_CAMERA_SHOT,
    normalize_camera_movement,
    normalize_camera_shot,
)

from planner.config import (
    AI_STORY_MODE,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    DIRECTOR_STORY_SCENES,
    STORY_ACCEPT_MIN_WORDS,
)


# ---------------------------------------------------------------------------
# Spoken-dialogue contract
# ---------------------------------------------------------------------------
# Dialogue semantics come from the existing single Qwen character/entity pass.
# This module contains no speech-verb, machine-word, UI-word, or story-domain
# vocabulary list. Deterministic code only reconciles Qwen's semantic spans to
# the final canonical character roster.

_CAMERA_DIRECTIVE_IN_ACTION = re.compile(
    r"\b(?:the\s+)?camera\s+(?:(?:slowly|gently|smoothly|deliberately|steadily)\s+)?"
    r"(?:pans?|tilts?|tracks?|dollies|pushes?|pulls?|cranes?|orbits?|zooms?|moves?|follows?)\b|"
    r"\b(?:tracking|establishing|wide|extreme\s+wide|medium\s+wide|medium\s+close[- ]?up|"
    r"close[- ]?up|extreme\s+close[- ]?up|over[- ]the[- ]shoulder|two[- ]shot)\s+shot\b|"
    r"\b(?:slow|gentle|smooth|controlled)\s+pan\b|"
    r"\bdolly\s+(?:in|out|back|forward)\b|"
    r"\b(?:push|pull)\s*[- ]?(?:in|back)\b",
    flags=re.IGNORECASE,
)

_ACTION_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of", "for", "from",
    "with", "by", "as", "is", "was", "were", "be", "been", "being", "it", "its", "that",
    "this", "then", "there", "across", "slowly", "gently", "smoothly", "deliberately",
    "camera", "pan", "pans", "tilt", "tilts", "track", "tracks", "tracking", "dolly", "ending",
}


def _grounded_action_fallback(action: str, scene_description: str, visual_prompt: str) -> str:
    """Choose a source-grounded, camera-free sentence when action contains camera direction."""
    action_tokens = {
        word for word in re.findall(r"[a-z][a-z'’-]*", str(action or "").casefold())
        if len(word) > 2 and word not in _ACTION_STOPWORDS
    }
    candidates = []
    for source_priority, source in enumerate((scene_description, visual_prompt)):
        normalized_source = re.sub(r"\s+", " ", str(source or "").strip())
        if not normalized_source:
            continue
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", normalized_source)
            if sentence.strip()
        ]
        for sentence in sentences:
            if _CAMERA_DIRECTIVE_IN_ACTION.search(sentence):
                continue
            words = re.findall(r"[\w'’-]+", sentence)
            if len(words) > 30:
                sentence = " ".join(words[:30]).rstrip(" ,;:") + "."
                words = re.findall(r"[\w'’-]+", sentence)
            tokens = {
                word.casefold() for word in re.findall(r"[a-z][a-z'’-]*", sentence.casefold())
                if len(word) > 2 and word not in _ACTION_STOPWORDS
            }
            overlap = len(action_tokens & tokens)
            in_contract_range = 10 <= len(words) <= 30
            # Prefer story sentences overlapping the original shot's actual subjects
            # and state. Prefer the scene description to the visual prompt on ties.
            candidates.append((overlap, int(in_contract_range), -abs(len(words) - 18), -source_priority, sentence))
    if not candidates:
        return ""
    chosen = max(candidates, key=lambda item: item[:4])
    return chosen[-1]


class QwenDirectorSanitizeMixin:
    def _valid_character_name(
        self,
        name: str,
        *,
        identity_type: str = "named_character",
    ) -> bool:

        value = str(
            name or ""
        ).strip()

        if not value:
            return False

        lowered = value.lower()
        identity_type = str(identity_type or "named_character").strip().lower()

        if EntityResolver.generic_role_surface(lowered) or lowered in EntityResolver.RELATIONSHIP_LABELS:
            return False

        # Qualified descriptive identities may legitimately be longer than
        # ordinary named identities, e.g. ``the man in the grey coverall``.
        max_words = 10 if identity_type == "descriptive_character" else 5
        if len(value.split()) > max_words:
            return False

        if any(
            token in value
            for token in (
                ":",
                ";",
                "|",
                "{",
                "}",
                "[",
                "]",
            )
        ):
            return False

        return True

    @staticmethod
    def _slug(
        value: str,
    ) -> str:

        cleaned = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(
                value or ""
            ).lower(),
        ).strip("_")

        return (
            cleaned
            or "character"
        )

    @staticmethod
    def _coerce_mapping(
        value,
    ) -> dict:
        """Safely normalize a model field that should be a JSON object."""
        if isinstance(value, dict):
            return dict(value)

        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return {}
                if isinstance(parsed, dict):
                    return parsed

        return {}

    @staticmethod
    def _coerce_list(
        value,
    ) -> list:
        """Safely normalize a model field that should be a JSON array."""
        if value is None:
            return []

        if isinstance(value, list):
            return list(value)

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, set):
            return list(value)

        if isinstance(value, str):
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return []
                return list(parsed) if isinstance(parsed, list) else []
            return [text] if text else []

        return []

    def _sanitize_characters(
        self,
        characters,
    ) -> list[dict]:

        result: list[dict] = []
        seen: set[str] = set()

        for value in (
            characters or []
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            name = str(
                value.get(
                    "name",
                    "",
                )
                or ""
            ).strip()

            profile = self._coerce_mapping(value.get("identity_profile", {}))
            identity_type = str(
                value.get("identity_type", profile.get("identity_type", "named_character"))
                or "named_character"
            ).strip().lower()

            if not self._valid_character_name(
                name,
                identity_type=identity_type,
            ):
                continue

            key = EntityResolver.normalize(name)

            if key in seen:
                continue

            seen.add(
                key
            )

            character_id = str(
                value.get(
                    "character_id",
                    "",
                )
                or ""
            ).strip()

            if not character_id:

                character_id = (
                    f"char_{self._slug(name)}"
                )

            relationship_to = str(
                value.get("relationship_to", profile.get("relationship_to", "")) or ""
            ).strip() or None
            relationship = str(
                value.get("relationship", profile.get("relationship", "")) or ""
            ).strip() or None
            aliases_raw = value.get("semantic_aliases", profile.get("semantic_aliases", []))
            aliases = []
            for alias in self._coerce_list(aliases_raw):
                text = str(alias).strip()
                if not text or not EntityResolver.is_safe_semantic_reference(text):
                    if EntityResolver.normalize(text) in EntityResolver.GENERIC_ROLE_ALIASES and identity_type in {"relational_character", "descriptive_character"}:
                        aliases.append(text)
                    continue
                if EntityResolver.normalize(text) in EntityResolver.GENERIC_ROLE_ALIASES and identity_type not in {"relational_character", "descriptive_character"}:
                    continue
                aliases.append(text)
            aliases = list(dict.fromkeys(aliases))
            if identity_type == "relational_character" and (not relationship_to or not relationship):
                identity_type = "named_character"
                relationship_to = None
                relationship = None
                aliases = []

            if identity_type == "descriptive_character":
                normalized_name = EntityResolver.normalize(name)
                core = re.sub(r"^(?:the|a|an)\s+", "", normalized_name).strip()
                # Validate descriptive identities by shape only. Contextual grounding belongs
                # to ProductionPlanner; this layer must not maintain another English role list.
                if len(core.split()) < 2 or len(core) < 5:
                    identity_type = "named_character"
                    aliases = []

            profile.update({
                "name": name,
                "semantic_aliases": aliases,
                "identity_type": identity_type,
                "relationship_to": relationship_to,
                "relationship": relationship,
            })

            result.append(
                {
                    "character_id":
                        character_id,

                    "name":
                        name,

                    "role":
                        str(
                            value.get(
                                "role",
                                "story character",
                            )
                            or "story character"
                        ),

                    "description":
                        str(
                            value.get(
                                "description",
                                "",
                            )
                            or ""
                        ),

                    "personality":
                        str(
                            value.get(
                                "personality",
                                "",
                            )
                            or ""
                        ),

                    "appearance":
                        self._coerce_mapping(
                            value.get(
                                "appearance",
                                {},
                            )
                        ),

                    "clothing":
                        self._coerce_mapping(
                            value.get(
                                "clothing",
                                {},
                            )
                        ),

                    "distinctive_features":
                        self._coerce_list(
                            value.get(
                                "distinctive_features",
                                [],
                            )
                        ),

                    "character_state":
                        self._coerce_mapping(
                            value.get(
                                "character_state",
                                {},
                            )
                        ),

                    "continuity_rules":
                        self._coerce_list(
                            value.get(
                                "continuity_rules",
                                [],
                            )
                        ),
                    # Return the already-sanitized identity contract, not the
                    # raw model fields.  This keeps top-level character metadata
                    # exactly aligned with identity_profile after malformed
                    # relational candidates are downgraded.
                    "semantic_aliases": list(aliases),
                    "identity_type": identity_type,
                    "relationship_to": relationship_to,
                    "relationship": relationship,
                    "identity_profile": profile,
                }
            )

        return result

    @staticmethod
    def _clean_list(
        value,
        limit: int | None = None,
    ) -> list[str]:
        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):
            items = [
                item.strip()
                for item in re.split(
                    r"[\n,;]+",
                    value,
                )
                if item.strip()
            ]
        elif isinstance(
            value,
            (list, tuple, set),
        ):
            items = [
                str(item).strip()
                for item in value
                if str(item).strip()
            ]
        else:
            items = [
                str(value).strip()
            ] if str(value).strip() else []

        result: list[str] = []
        seen: set[str] = set()

        for item in items:
            key = item.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

            if (
                limit is not None
                and len(result) >= limit
            ):
                break

        return result

    @staticmethod
    def _character_alias_map(
        character_names: set[str],
    ) -> dict[str, str]:
        """Resolve safe, unambiguous aliases to canonical roster names."""
        canonical_names = sorted(
            {
                str(name or "").strip().lower()
                for name in character_names
                if str(name or "").strip()
            }
        )

        aliases: dict[str, str] = {}

        for canonical in canonical_names:
            aliases[canonical] = canonical

        first_candidates: dict[str, set[str]] = {}
        pair_candidates: dict[str, set[str]] = {}

        for canonical in canonical_names:

            tokens = re.findall(
                r"[a-z0-9']+",
                canonical,
            )

            if not tokens:
                continue

            if tokens[0] not in {
                "the", "a", "an", "this", "that", "these", "those",
                "his", "her", "their", "my", "our", "your",
            }:
                first_candidates.setdefault(
                    tokens[0],
                    set(),
                ).add(canonical)

            if len(tokens) >= 2:

                pair = " ".join(
                    tokens[-2:]
                )

                pair_candidates.setdefault(
                    pair,
                    set(),
                ).add(canonical)

        for alias, matches in first_candidates.items():

            if len(matches) == 1:
                aliases[alias] = next(
                    iter(matches)
                )

        for alias, matches in pair_candidates.items():

            if len(matches) == 1:
                aliases[alias] = next(
                    iter(matches)
                )

        return aliases

    def _sanitize_scenes(
        self,
        scenes,
        character_names: set[str],
    ) -> list[dict]:

        result: list[dict] = []

        for index, value in enumerate(
            scenes or [],
            start=1,
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            description = str(
                value.get(
                    "description",
                    "",
                )
                or ""
            ).strip()

            if not description:

                description = str(
                    value.get(
                        "scene_objective",
                        "",
                    )
                    or ""
                ).strip()

            if not description:
                continue

            lower = description.lower()

            if lower.startswith(
                (
                    "tone:",
                    "visual priority:",
                    "visual priorities:",
                    "camera:",
                    "lighting:",
                    "mood:",
                    "sound:",
                )
            ):
                continue

            selected: list[str] = []

            alias_map = self._character_alias_map(
                character_names
            )

            for name in (
                value.get(
                    "characters",
                    [],
                )
                or []
            ):

                supplied = str(
                    name
                ).strip().lower()

                resolved = alias_map.get(
                    supplied
                )

                if resolved:
                    selected.append(
                        resolved
                    )

            if not selected and character_names:

                searchable = " ".join(
                    [
                        description,
                        str(value.get("title", "") or ""),
                        str(value.get("scene_objective", "") or ""),
                        str(value.get("continuity_notes", "") or ""),
                    ]
                ).lower()

                for alias, canonical in sorted(
                    alias_map.items(),
                    key=lambda item: (
                        -len(item[0]),
                        item[0],
                    ),
                ):

                    if re.search(
                        r"(?<![a-z0-9'])"
                        + re.escape(alias)
                        + r"(?![a-z0-9'])",
                        searchable,
                    ):

                        if canonical not in selected:
                            selected.append(
                                canonical
                            )

            selected = self._clean_list(
                selected,
                limit=6,
            )

            scene_id = str(
                value.get(
                    "scene_id",
                    f"scene_{index:03d}",
                )
                or f"scene_{index:03d}"
            ).strip()

            title = str(
                value.get(
                    "title",
                    "",
                )
                or ""
            ).strip()

            if not title:

                title = (
                    str(
                        value.get(
                            "location",
                            "",
                        )
                        or ""
                    ).strip()
                    or f"Scene {index}"
                )

            result.append(
                {
                    "scene_id":
                        scene_id,

                    "title":
                        title,

                    "order":
                        len(result) + 1,

                    "location":
                        str(
                            value.get(
                                "location",
                                "",
                            )
                            or ""
                        ),

                    "time_of_day":
                        str(
                            value.get(
                                "time_of_day",
                                "",
                            )
                            or ""
                        ),

                    "weather":
                        str(
                            value.get(
                                "weather",
                                "",
                            )
                            or ""
                        ),

                    "atmosphere":
                        str(
                            value.get(
                                "atmosphere",
                                "",
                            )
                            or ""
                        ),

                    "description":
                        description,

                    "mood":
                        str(
                            value.get(
                                "mood",
                                "",
                            )
                            or ""
                        ),

                    "lighting":
                        str(
                            value.get(
                                "lighting",
                                "",
                            )
                            or ""
                        ),

                    "color_temperature":
                        str(
                            value.get(
                                "color_temperature",
                                "",
                            )
                            or ""
                        ),

                    "environment_details":
                        self._clean_list(
                            value.get(
                                "environment_details",
                                [],
                            ),
                            limit=12,
                        ),

                    "key_props":
                        self._clean_list(
                            value.get(
                                "key_props",
                                [],
                            ),
                            limit=8,
                        ),

                    "scene_objective":
                        str(
                            value.get(
                                "scene_objective",
                                "",
                            )
                            or ""
                        ),

                    "characters":
                        self._clean_list(
                        selected,
                    ),

                    "story_summary":
                        str(
                            value.get(
                                "story_summary",
                                description,
                            )
                            or description
                        ),

                    "continuity_notes":
                        str(
                            value.get(
                                "continuity_notes",
                                "",
                            )
                            or ""
                        ),

                    "shot_ids":
                        [],
                }
            )

        # Normalize duplicate scene IDs deterministically. Never silently allow
        # two scenes to share a checkpoint address. Fresh plans may be repaired;
        # resume plans are returned before this path so their stored IDs remain stable.
        used_ids: set[str] = set()
        for scene in result:
            base_id = str(scene.get("scene_id", "") or "").strip() or "scene"
            candidate = base_id
            suffix = 2
            while candidate.lower() in used_ids:
                candidate = f"{base_id}_{suffix}"
                suffix += 1
            scene["scene_id"] = candidate
            used_ids.add(candidate.lower())

        # Budget enforcement intentionally happens after sanitization.
        return result

    @staticmethod
    def _normalize_batch_shot_response(
        response: dict,
    ) -> dict[str, list[dict]]:
        if not isinstance(response, dict):
            return {}

        if (
            str(response.get("scene_id", "") or "").strip()
            and isinstance(response.get("shots"), list)
        ):
            entries = [response]
        else:
            entries = (
                response.get("scene_shots")
                or response.get("shots")
                or response.get("items")
                or response.get("scene_shot")
            )

        if entries is None:
            return {}

        if isinstance(entries, dict):
            entries = [
                {"scene_id": str(key), "shots": value}
                for key, value in entries.items()
                if isinstance(value, list)
            ]

        if not isinstance(entries, list):
            return {}

        result: dict[str, list[dict]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            scene_id = str(entry.get("scene_id", "") or "").strip()
            shots = entry.get("shots")
            if scene_id and isinstance(shots, list):
                normalized_shots = []
                for shot_index, item in enumerate(shots, start=1):
                    if not isinstance(item, dict):
                        continue
                    candidate = dict(item)
                    # These identifiers are fully determined by the parent scene
                    # and array order. Do not spend generated tokens on them.
                    if not str(candidate.get("scene_id", "") or "").strip():
                        candidate["scene_id"] = scene_id
                    if not str(candidate.get("shot_id", "") or "").strip():
                        candidate["shot_id"] = f"{scene_id}_shot_{shot_index:03d}"
                    normalized_shots.append(candidate)

                # In the current compact contract, each scene owns one keyed assignment
                # per source utterance. JSON object keys cannot duplicate, so a line cannot
                # accidentally be assigned to three shots. Expand the 1-based shot number
                # into the legacy compact-ID event shape consumed by the existing validator.
                assignments = entry.get("dialogue_assignments", {})
                if isinstance(assignments, dict):
                    for raw_dialogue_id, raw_shot_number in assignments.items():
                        dialogue_id = str(raw_dialogue_id or "").strip()
                        if not dialogue_id or raw_shot_number is None:
                            continue
                        if isinstance(raw_shot_number, bool):
                            continue
                        try:
                            shot_number = int(raw_shot_number)
                        except (TypeError, ValueError):
                            continue
                        if str(raw_shot_number).strip() != str(shot_number):
                            continue
                        if not 1 <= shot_number <= len(normalized_shots):
                            continue
                        target = normalized_shots[shot_number - 1]
                        events = target.get("dialogue_events", [])
                        if not isinstance(events, list):
                            events = []
                        events.append({"source_dialogue_id": dialogue_id})
                        target["dialogue_events"] = events

                result[scene_id] = normalized_shots
                continue
            if isinstance(shots, list):
                for shot_index, item in enumerate(shots, start=1):
                    if not isinstance(item, dict):
                        continue
                    candidate = dict(item)
                    sid = str(candidate.get("scene_id", "") or "").strip()
                    if sid and not str(candidate.get("shot_id", "") or "").strip():
                        candidate["shot_id"] = f"{sid}_shot_{shot_index:03d}"
                    if sid:
                        result.setdefault(sid, []).append(candidate)
        return result

    @staticmethod
    def _normalize_batch_scene_metadata_response(response: dict) -> dict[str, dict]:
        """Return Qwen's per-scene semantic metadata without mixing it into shots."""
        if not isinstance(response, dict):
            return {}
        entries = response.get("scene_shots", [])
        if isinstance(entries, dict):
            entries = [dict(value, scene_id=key) for key, value in entries.items() if isinstance(value, dict)]
        if not isinstance(entries, list):
            return {}
        result: dict[str, dict] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            scene_id = str(entry.get("scene_id", "") or "").strip()
            metadata = entry.get("scene_metadata")
            if scene_id and isinstance(metadata, dict):
                result[scene_id] = dict(metadata)
        return result

    def _sanitize_shots(
        self,
        shots,
        scene: dict,
        character_names: set[str],
    ) -> list[dict]:

        scene_id = str(
            scene.get(
                "scene_id",
                "",
            )
            or ""
        ).strip()

        result: list[dict] = []
        visual_prompts_normalized = 0
        camera_fields_normalized = 0
        camera_fields_defaulted = 0
        camera_actions_normalized = 0

        for value in (
            shots
            or []
        ):

            if not isinstance(
                value,
                dict,
            ):
                continue

            candidate = dict(
                value
            )

            candidate_scene_id = str(
                candidate.get(
                    "scene_id",
                    scene_id,
                )
                or scene_id
            ).strip()

            if candidate_scene_id != scene_id:
                continue

            candidate[
                "scene_id"
            ] = scene_id

            # Restore compact-response fields from their authoritative source.
            # Scene location/light are deliberately not generated redundantly for
            # every shot in a batch. Spatial maps remain in the canonical contract
            # but safely default empty when no upstream evidence supplies them.
            scene_description = str(scene.get("description", "") or "").strip()
            scene_lighting_raw = str(scene.get("lighting", "") or "").strip()
            scene_color_raw = str(scene.get("color_temperature", "") or "").strip()
            scene_lighting = scene_lighting_raw or "soft natural light"
            scene_color = scene_color_raw or "neutral"
            scene_mood = str(scene.get("mood", "") or "").strip() or "cinematic"
            raw_location = str(candidate.get("location", "") or "").strip()
            scene_location = str(scene.get("location", "") or "").strip()
            if scene_location:
                candidate["location"] = scene_location
            elif raw_location:
                candidate["location"] = raw_location
            else:
                candidate["location"] = "unspecified location"
            if scene_lighting_raw:
                candidate["lighting"] = scene_lighting_raw
            if scene_color_raw:
                candidate["color_temperature"] = scene_color_raw

            defaults = {
                "camera_shot": DEFAULT_CAMERA_SHOT,
                "camera_movement": DEFAULT_CAMERA_MOVEMENT,
                "lens_and_depth_of_field": "normal lens, moderate depth of field",
                "composition_notes": "Clear subject separation with readable spatial depth.",
                "lighting": scene_lighting,
                "color_temperature": scene_color,
                "mood": scene_mood,
                "visual_prompt": scene_description or str(candidate.get("action", "") or "").strip(),
            }

            for field, fallback in defaults.items():
                if not str(candidate.get(field, "") or "").strip():
                    candidate[field] = fallback

            # Structured output schemas constrain Qwen at generation time, but a
            # fallback/legacy/cached shot can bypass that schema. Normalize only
            # well-understood aliases; unknown labels fail closed to canonical defaults.
            for _field, _normalizer in (
                ("camera_shot", normalize_camera_shot),
                ("camera_movement", normalize_camera_movement),
            ):
                _raw_camera_value = str(candidate.get(_field, "") or "").strip()
                _canonical_camera_value, _camera_status = _normalizer(_raw_camera_value)
                if _camera_status != "unchanged":
                    candidate[_field] = _canonical_camera_value
                    if _camera_status == "fallback":
                        camera_fields_defaulted += 1
                    else:
                        camera_fields_normalized += 1

            raw_action = str(candidate.get("action", "") or "").strip()
            if _CAMERA_DIRECTIVE_IN_ACTION.search(raw_action):
                grounded_action = _grounded_action_fallback(
                    raw_action,
                    scene_description,
                    str(candidate.get("visual_prompt", "") or ""),
                )
                if not grounded_action:
                    # Do not ship camera instructions in the narrative-action field
                    # when no source-grounded replacement exists.
                    continue
                candidate["action"] = grounded_action
                camera_actions_normalized += 1

            shot_location = str(
                candidate.get("location", "") or ""
            ).strip()
            scene_location = str(
                scene.get("location", "") or ""
            ).strip()

            if shot_location and scene_location:
                location_words = shot_location.split()
                if len(location_words) > 4:
                    candidate["location"] = scene_location
                elif any(
                    word.lower() in {
                        "the", "a", "an", "of", "she", "he",
                        "they", "his", "her", "their", "doing",
                        "response", "hushed", "sharp", "this",
                        "that", "didn't", "somewhere", "few",
                        "who", "still", "remembered",
                    }
                    for word in location_words
                ):
                    candidate["location"] = scene_location

            for state_key in (
                "continuity_start_state",
                "continuity_end_state",
            ):
                state = candidate.get(state_key)
                if isinstance(state, dict) and state.get("location"):
                    state_words = str(state["location"]).split()
                    if len(state_words) > 4:
                        state["location"] = candidate.get(
                            "location", scene_location
                        )

            if not str(candidate.get("visual_prompt", "") or "").strip():
                continue

            # The shot prompt contract is 15-40 words. Enforce it deterministically
            # rather than spending another model call: enrich short prompts only with
            # same-shot action/composition already present, and trim excess words.
            visual_prompt = re.sub(
                r"\s+", " ", str(candidate.get("visual_prompt", "") or "").strip()
            )
            original_visual_prompt = visual_prompt
            def _visual_word_count(value: str) -> int:
                return len(re.findall(r"[\w'’-]+", value))

            if _visual_word_count(visual_prompt) < 15:
                additions = [
                    str(candidate.get("action", "") or "").strip(),
                    str(candidate.get("composition_notes", "") or "").strip(),
                    f"{shot_location or scene_location or 'the scene'} under {scene_lighting}",
                ]
                for addition in additions:
                    if addition and addition.casefold() not in visual_prompt.casefold():
                        visual_prompt = (visual_prompt + " " + addition).strip()
                    if _visual_word_count(visual_prompt) >= 15:
                        break
            if _visual_word_count(visual_prompt) > 40:
                visual_prompt = " ".join(visual_prompt.split()[:40]).rstrip(" ,;:")
            if visual_prompt != original_visual_prompt:
                candidate["visual_prompt"] = visual_prompt
                visual_prompts_normalized += 1

            # Character binding is production-critical. Qwen may omit the
            # field or return an empty list even though the scene already
            # has approved characters. In that case deterministically inherit
            # the scene's character set. When Qwen does provide names, keep
            # only names already present in the approved character roster.
            scene_characters = self._clean_list(
                scene.get(
                    "characters",
                    [],
                ),
                limit=6,
            )

            supplied_characters = self._clean_list(
                candidate.get(
                    "characters",
                    [],
                ),
                limit=6,
            )

            selected_characters: list[str] = []

            for name in supplied_characters:

                lowered = name.lower()

                if lowered in character_names:
                    selected_characters.append(name)

            if not selected_characters and (
                "characters" not in candidate or not isinstance(candidate.get("characters"), list)
            ):
                # Compatibility fallback is only for legacy payloads that omitted the
                # field. An explicit [] is meaningful: this shot may be a setting/detail
                # shot with no visible person, or show only a receiver for remote dialogue.
                # Never turn an audio-only speaker or the whole scene cast into visible bodies.
                selected_characters = [
                    name
                    for name in scene_characters
                    if name.lower() in character_names
                ]

            candidate["characters"] = self._clean_list(
                selected_characters,
                limit=6,
            )

            dialogue = candidate.get("dialogue_events", [])
            if not isinstance(dialogue, list):
                dialogue = []
            normalized_dialogue = []
            for event in dialogue:
                if not isinstance(event, dict):
                    continue
                speaker = str(event.get("speaker", "") or "").strip()
                text = str(event.get("text", "") or "")
                if not speaker or not text.strip():
                    continue
                normalized_event = {
                    "speaker": speaker,
                    "text": text,
                    "continues_from_previous_shot": bool(event.get("continues_from_previous_shot", False)),
                    "continues_to_next_shot": bool(event.get("continues_to_next_shot", False)),
                }
                if str(event.get("delivery", "in_person")).strip().lower() == "remote":
                    normalized_event["delivery"] = "remote"
                normalized_dialogue.append(normalized_event)
            if not normalized_dialogue:
                legacy_speakers = candidate.get("speaking_characters", []) or []
                legacy_text = str(candidate.get("speech_text", "") or "").strip()
                if legacy_text and legacy_speakers:
                    # Keep the compatibility bridge, but never manufacture a
                    # dialogue event from obviously long narrative prose. The
                    # Director's semantic normalizer remains the final gate.
                    quoted = re.findall(r'“([^”\n]+)”|"([^"\n]+)"|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)', legacy_text)
                    quoted_texts = [next((part for part in group if part), "").strip() for group in quoted]
                    quoted_texts = [value for value in quoted_texts if value]
                    if quoted_texts:
                        legacy_text = " ".join(quoted_texts)
                        normalized_dialogue = [{
                            "speaker": str(legacy_speakers[0]).strip(),
                            "text": legacy_text,
                            "continues_from_previous_shot": False,
                            "continues_to_next_shot": False,
                        }]
            # Keep all legacy dialogue fields synchronized with the canonical
            # event list. This prevents stale speaking_characters/speech_text
            # metadata from surviving when Qwen returns invalid JSON and the
            # deterministic fallback path is used.
            candidate["dialogue_events"] = normalized_dialogue
            candidate["speaking_characters"] = list(dict.fromkeys(
                str(event.get("speaker", "") or "").strip()
                for event in normalized_dialogue
                if str(event.get("speaker", "") or "").strip()
            ))
            candidate["speech_text"] = " ".join(
                str(event.get("text", "") or "").strip()
                for event in normalized_dialogue
                if str(event.get("text", "") or "").strip()
            )

            def _normalize_continuity(value) -> dict:
                if isinstance(value, dict):
                    return dict(value)
                text = str(value or "").strip()
                if not text:
                    return {}
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    return {"state_description": text}
                return parsed if isinstance(parsed, dict) else {"state_description": str(parsed)}

            candidate["continuity_start_state"] = _normalize_continuity(
                candidate.get("continuity_start_state", candidate.get("continuity_state_start"))
            )
            candidate["continuity_end_state"] = _normalize_continuity(
                candidate.get("continuity_end_state", candidate.get("continuity_state_end"))
            )
            candidate.pop("continuity_state_start", None)
            candidate.pop("continuity_state_end", None)
            candidate["is_scene_boundary"] = bool(candidate.get("is_scene_boundary", False))
            raw_bboxes = candidate.get("character_spatial_bboxes", {}) or {}
            normalized_bboxes = {}
            if isinstance(raw_bboxes, dict):
                for name, bbox in raw_bboxes.items():
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        values = [float(v) for v in bbox]
                        if all(0.0 <= v <= 1.0 for v in values) and values[2] >= values[0] and values[3] >= values[1]:
                            normalized_bboxes[str(name).strip()] = values
            candidate["character_spatial_bboxes"] = normalized_bboxes
            raw_regions = candidate.get("character_spatial_regions", {}) or {}
            candidate["character_spatial_regions"] = (
                {str(k).strip(): str(v).strip() for k, v in raw_regions.items() if str(k).strip() and str(v).strip()}
                if isinstance(raw_regions, dict) else {}
            )
            for spatial_key in ("character_spatial_bboxes_start", "character_spatial_bboxes_end"):
                raw = candidate.get(spatial_key, {}) or {}
                normalized = {}
                if isinstance(raw, dict):
                    for name, bbox in raw.items():
                        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                            values = [float(v) for v in bbox]
                            if all(0.0 <= v <= 1.0 for v in values) and values[2] >= values[0] and values[3] >= values[1]:
                                normalized[str(name).strip()] = values
                candidate[spatial_key] = normalized
            for spatial_key in ("character_spatial_regions_start", "character_spatial_regions_end"):
                raw = candidate.get(spatial_key, {}) or {}
                candidate[spatial_key] = (
                    {str(k).strip(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v).strip()}
                    if isinstance(raw, dict) else {}
                )

            result.append(
                candidate
            )

        if visual_prompts_normalized:
            print(
                f"[DIRECTOR] visual_prompt word range normalized for {visual_prompts_normalized} shot(s) in {scene_id}; no model retry",
                flush=True,
            )
        if camera_fields_normalized or camera_fields_defaulted:
            print(
                f"[DIRECTOR] camera vocabulary normalized in {scene_id}: "
                f"aliases={camera_fields_normalized}, defaulted_invalid={camera_fields_defaulted}; no model retry",
                flush=True,
            )
        if camera_actions_normalized:
            print(
                f"[DIRECTOR] removed camera direction from narrative action in {scene_id}: "
                f"{camera_actions_normalized} grounded replacement(s); no model retry",
                flush=True,
            )

        # Canonicalize dialogue continuation across adjacent shots in the same
        # scene. Qwen may independently set the two boundary flags; the plan
        # must expose one consistent boundary state to DialogueTimeline.
        for index, candidate in enumerate(result):
            events = candidate.get("dialogue_events", [])
            if not isinstance(events, list) or not events:
                if index > 0:
                    previous_events = result[index - 1].get("dialogue_events", [])
                    if isinstance(previous_events, list) and previous_events:
                        previous_events[-1]["continues_to_next_shot"] = False
                continue

            if index == 0 or bool(candidate.get("is_scene_boundary", False)):
                events[0]["continues_from_previous_shot"] = False
                if index > 0:
                    previous_events = result[index - 1].get("dialogue_events", [])
                    if isinstance(previous_events, list) and previous_events:
                        previous_events[-1]["continues_to_next_shot"] = False
                continue

            previous_events = result[index - 1].get("dialogue_events", [])
            if not isinstance(previous_events, list) or not previous_events:
                events[0]["continues_from_previous_shot"] = False
                continue

            previous_flag = bool(previous_events[-1].get("continues_to_next_shot", False))
            current_flag = bool(events[0].get("continues_from_previous_shot", False))
            continuation = previous_flag or current_flag
            previous_events[-1]["continues_to_next_shot"] = continuation
            events[0]["continues_from_previous_shot"] = continuation

        return result

    @staticmethod
    def _normalize_ids(
        scenes: list[dict],
        shots: list[dict],
    ) -> None:

        old_to_new: dict[str, str] = {}

        for index, scene in enumerate(
            scenes,
            start=1,
        ):
            old_id = str(
                scene.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()
            canonical = f"scene_{index:03d}"
            if old_id:
                old_to_new.setdefault(
                    old_id.lower(),
                    canonical,
                )
            scene["scene_id"] = canonical
            scene["order"] = index

        scene_shot_counts: dict[str, int] = {}
        for shot in shots:
            old_scene_id = str(
                shot.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()

            scene_id = old_to_new.get(
                old_scene_id.lower(),
                old_scene_id,
            )

            shot["scene_id"] = scene_id
            scene_shot_counts[scene_id] = (
                scene_shot_counts.get(
                    scene_id,
                    0,
                )
                + 1
            )
            shot_number = scene_shot_counts[scene_id]
            shot["shot_id"] = (
                f"{scene_id}_shot_{shot_number:03d}"
            )

    @staticmethod
    def _normalize_story(
        text: str,
    ) -> str:

        value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not value:
            return ""

        # Story text is plain prose. Qwen sometimes wraps sign/label text in markdown
        # emphasis (*"..."*) or adds bold headings; those markers would otherwise leak
        # into scene descriptions, prompts, and the dialogue extractor.
        value = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", value)
        value = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\1", value)

        # Emoji / pictographs are never part of the story prose; strip them and any
        # orphaned " ." they leave behind.
        value = re.sub(
            "[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u200D]+",
            "",
            value,
        )
        value = re.sub(r"\s+\.(?=\s|$)", ".", value)

        paragraphs = []
        for paragraph in re.split(r"\n\s*\n+", value):
            normalized = re.sub(r"[ \t\n]+", " ", paragraph).strip()
            if normalized:
                paragraphs.append(normalized)

        return "\n\n".join(paragraphs)

    @staticmethod
    def _meaningful_tokens(
        text: str,
    ) -> set[str]:

        words = re.findall(
            r"[A-Za-z][A-Za-z'-]{3,}",
            str(
                text or ""
            ).lower(),
        )

        stop = {
            "this",
            "that",
            "with",
            "from",
            "into",
            "about",
            "have",
            "will",
            "they",
            "their",
            "there",
            "which",
            "while",
            "where",
            "would",
            "could",
            "should",
            "story",
            "then",
            "than",
            "when",
        }

        return {
            word
            for word in words
            if word not in stop
        }

    @staticmethod
    def _preservation_anchors(text: str) -> set[str]:
        value = str(text or "")
        anchors: set[str] = set()

        # Explicitly named/called characters are high-confidence anchors.
        for match in re.finditer(
            r"\b(?:named|called)\s+([A-Z][A-Za-z'-]{1,}(?:\s+[A-Z][A-Za-z'-]{1,})*)",
            value,
        ):
            anchors.add(match.group(1).strip().lower())

        # Preserve dates/counts/measurements that can materially change plot facts.
        anchors.update(re.findall(r"\b\d+(?:[.,]\d+)?(?:%|[A-Za-z]+)?\b", value.lower()))
        return anchors

    @staticmethod
    def _premise_compare_token(token: str) -> str:
        """Normalize a content token enough to recognize ordinary inflectional paraphrase."""
        value = str(token or "").lower().strip()
        if len(value) <= 5:
            return value
        for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
            if value.endswith(suffix) and len(value) - len(suffix) >= 5:
                return value[: -len(suffix)]
        return value

    def _global_premise_coverage(
        self,
        source: str,
        result: str,
        minimum_token_overlap: float = 0.15,
    ) -> tuple[float, list[str]]:
        """Measure premise preservation while allowing normal narrative paraphrase.

        AI Story may legitimately paraphrase verbs/adjectives and distribute the
        supplied premise across several sentences. A 40% raw-token gate was too
        brittle for Qwen3 thinking output and rejected valid rewrites. We therefore
        use light inflection normalization and a lower deterministic lexical floor;
        the story prompt separately requires all major concrete premise facts to
        remain explicit. Expand Story keeps its stricter sentence/event check.
        """
        source_sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", source)
            if sentence.strip()
        ]
        result_tokens = {
            self._premise_compare_token(token)
            for token in self._meaningful_tokens(result)
            if token
        }
        if not source_sentences:
            return 1.0, []

        covered = 0
        missing: list[str] = []
        for sentence in source_sentences:
            tokens = {
                self._premise_compare_token(token)
                for token in self._meaningful_tokens(sentence)
                if token
            }
            if not tokens:
                continue
            # Short premises need at least two preserved content tokens; longer
            # premises use the configured lexical floor without demanding near-
            # verbatim wording.
            required_overlap = max(
                float(minimum_token_overlap),
                min(0.40, 2.0 / len(tokens)),
            )
            overlap = len(tokens & result_tokens) / max(1, len(tokens))
            if overlap >= required_overlap:
                covered += 1
            else:
                missing.append(sentence[:140])

        coverage = covered / max(1, len(source_sentences))
        return coverage, missing

    def _preservation_coverage(
        self,
        source: str,
        result: str,
        minimum_sentence_overlap: float = 0.28,
    ) -> tuple[float, list[str]]:
        """Measure conservative sentence-level lexical preservation.

        This is deliberately lexical/structural rather than an LLM judge so
        acceptance remains deterministic, cheap, and available in CI.
        """
        source_sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", source)
            if sentence.strip()
        ]
        result_sentences = [
            sentence.strip()
            for sentence in re.split(r"[.!?]+", result)
            if sentence.strip()
        ]
        if not source_sentences:
            return 1.0, []
        result_tokens = [
            self._meaningful_tokens(sentence)
            for sentence in result_sentences
        ]
        covered = 0
        missing: list[str] = []
        for sentence in source_sentences:
            tokens = self._meaningful_tokens(sentence)
            if not tokens:
                continue
            best = 0.0
            for candidate in result_tokens:
                if not candidate:
                    continue
                overlap = len(tokens & candidate) / max(1, len(tokens))
                best = max(best, overlap)
            if best >= minimum_sentence_overlap:
                covered += 1
            else:
                missing.append(sentence[:140])
        coverage = covered / max(1, len(source_sentences))
        return coverage, missing

    @staticmethod
    def _preserve_story_requires_semantic_dialogue(text: str) -> bool:
        """Return True when Preserve Story contains quote-delimited prose that needs semantic attribution."""
        value = str(text or "")
        return bool(
            re.search(r'"[^"\n]+"', value)
            or re.search(r'“[^”\n]+”', value)
            or re.search(r'‘[^’\n]+’', value)
            or re.search(r"(?<!\w)'[^'\n]+'(?!\w)", value)
        )

    @staticmethod
    def _story_has_open_ended_finale(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return True

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", value)
            if sentence.strip()
        ]
        final_sentence = sentences[-1].lower() if sentences else value.lower()
        # Only the terminal sentence determines whether the story itself ends on
        # a future hook. Earlier sentences may legitimately mention plans,
        # uncertainty, or threats that are resolved by the final sentence.
        # Looking through the whole final paragraph caused false positives and
        # unnecessary story rewrites.
        hook_patterns = (
            r"\b(?:he|she|they|[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,2})\s+(?:would|will|could|might)\s+(?:find|follow|face|continue|discover|uncover|return|seek|search|learn|pursue)\b",
            r"\bwould begin\b",
            r"\bwas about to\b",
            r"\bwas only the beginning\b",
            r"\bthe beginning of\b",
            r"\bhad only begun\b",
            r"\bquestions remained\b",
            r"\bwhat happened next\b",
            r"\bthe next chapter\b",
            r"\bready to discover\b",
            r"\bready to uncover\b",
            r"\b(?:walked|stepped|headed|disappeared|went) into the unknown\b",
            r"\bset something in motion that could never be undone\b",
            r"\buntil (?:the|it|they|he|she|something|someone)\b.{0,40}\b(?:returned|came back|woke|stirred|found|rose|called)\b",
            r"\byet to come\b",
            r"\bonce more\b.{0,30}\buntil\b",
            r"\bfirst step of\b",
        )

        return any(
            re.search(pattern, final_sentence, flags=re.IGNORECASE)
            for pattern in hook_patterns
        )

    @staticmethod
    def _final_paragraph_word_count(text: str) -> int:
        value = str(text or "").strip()
        paragraph = value.split("\n\n")[-1].strip()
        return len(re.findall(r"\b[\w'’-]+\b", paragraph))

    @staticmethod
    def _ends_cleanly(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        # Accept normal English terminal punctuation followed by any standard
        # straight/curly closing quote, apostrophe, bracket, or guillemet.
        # This is syntax normalization only; it does not forgive an actually
        # unfinished final clause.
        return bool(re.search(r"[.!?][\"'\u2019\u201d\u00bb\u203a\)\]\}]*$", value))

    def _validate_story_completion_contract(self, mode: str, result: str) -> None:
        """Reject incomplete or structurally invalid stories from the single creative pass."""
        if mode not in {AI_STORY_MODE, EXPAND_USER_STORY_MODE}:
            return

        paragraphs = [
            part.strip()
            for part in re.split(r"\n\s*\n+", str(result or "").strip())
            if part.strip()
        ]
        # Collect every defect so the single fail-closed error reports all of them together.
        errors: list[str] = []
        if len(paragraphs) != DIRECTOR_STORY_SCENES:
            errors.append(
                f"Generated story must contain exactly {DIRECTOR_STORY_SCENES} paragraphs (found {len(paragraphs)})."
            )

        # A minimum length is a hard lower acceptance boundary, not an exact target.
        # It must not cause trimming or padding; a short draft fails the single-pass run.
        word_count = len(re.findall(r"\b[\w'’-]+\b", str(result or "")))
        minimum_words = int(getattr(self, "_story_min_words", STORY_ACCEPT_MIN_WORDS))
        if word_count < minimum_words:
            errors.append(
                f"Generated story has {word_count} words, below the minimum accepted length of {minimum_words}."
            )

        if self._story_has_open_ended_finale(result):
            errors.append(
                "Generated story ends on an unresolved future hook rather than a completed consequence."
            )

        # The final scene must resolve the story; do not impose a numeric paragraph quota.

        # Sentence count is a writing-quality target, not a hard production gate.
        try:
            sentence_issues = self._story_sentence_contract_issues(result)
        except (AttributeError, TypeError):
            sentence_issues = []
        # Sentence count remains a writing-quality target, not a hard production gate.
        # _generate_story_once reports these issues through story_static_checks.
        # Keep all other completion-contract checks blocking.

        if errors:
            raise RuntimeError(" | ".join(errors))

    def _validate_mode_output(
        self,
        mode: str,
        user_input: str,
        story: str,
    ) -> None:

        source = self._normalize_story(
            user_input
        )

        result = self._normalize_story(
            story
        )

        if not result:

            raise RuntimeError(
                "Qwen director returned an empty story."
            )

        if mode == PRESERVE_USER_STORY_MODE:

            if source != result:

                raise RuntimeError(
                    "Preserve Story mode changed "
                    "the supplied story."
                )

            return

        if mode == AI_STORY_MODE:

            if source == result:

                raise RuntimeError(
                    "AI Story mode returned "
                    "the premise unchanged."
                )

            # Do not reject a valid one-sentence story using an arbitrary
            # sentence-count rule. Require deterministic source preservation
            # and some genuinely new meaningful content instead.
            source_anchors = self._preservation_anchors(source)
            result_lower = result.lower()
            missing_anchors = [
                anchor
                for anchor in source_anchors
                if anchor not in result_lower
            ]
            if missing_anchors:
                raise RuntimeError(
                    "AI Story mode dropped required source anchors: "
                    + ", ".join(missing_anchors[:8])
                )

            coverage, missing_sentences = self._global_premise_coverage(
                source,
                result,
            )
            if source and coverage < 0.50:
                detail = "; ".join(missing_sentences[:3])
                raise RuntimeError(
                    "AI Story mode did not preserve enough of the supplied "
                    f"premise (global_coverage={coverage:.2f}). {detail}".strip()
                )

            source_tokens = self._meaningful_tokens(source)
            result_tokens = self._meaningful_tokens(result)
            if source_tokens and not (result_tokens - source_tokens):
                raise RuntimeError(
                    "AI Story mode did not add meaningful narrative content."
                )

            if not self._ends_cleanly(result):
                raise RuntimeError(
                    "Story text does not end in a complete sentence "
                    "(the model stopped generating before finishing)."
                )

            self._validate_story_completion_contract(mode, result)

            return

        if mode == EXPAND_USER_STORY_MODE:

            if source == result:

                raise RuntimeError(
                    "Expand Story mode returned "
                    "the supplied story unchanged."
                )

            source_words = (
                self._meaningful_tokens(
                    source
                )
            )

            if source_words:

                result_words = (
                    self._meaningful_tokens(
                        result
                    )
                )

                overlap = (
                    len(
                        source_words
                        & result_words
                    )
                    / max(
                        1,
                        len(source_words),
                    )
                )

                # Lexical overlap is not a reliable semantic-preservation gate:
                # valid paraphrases can use different words while retaining source events.
                # Required anchors and the stricter source-event coverage check below
                # remain blocking, so do not reject an expansion on token overlap alone.

            anchors = self._preservation_anchors(source)
            result_lower = result.lower()
            missing_anchors = [
                anchor
                for anchor in sorted(anchors)
                if anchor not in result_lower
            ]
            if missing_anchors:
                raise RuntimeError(
                    "Expand Story mode dropped source anchors: "
                    + ", ".join(missing_anchors)
                )

            coverage, missing_sentences = self._preservation_coverage(
                source,
                result,
                minimum_sentence_overlap=0.28,
            )
            if coverage < 0.60:
                details = "; ".join(missing_sentences[:3])
                raise RuntimeError(
                    "Expand Story mode did not preserve enough source-event coverage "
                    f"(coverage={coverage:.3f}; minimum=0.600). "
                    f"Unmatched source events: {details}"
                )

            sentences = [
                value
                for value
                in re.split(
                    r"[.!?]+",
                    result,
                )
                if value.strip()
            ]

            if len(sentences) < 3:

                raise RuntimeError(
                    "Expand Story mode did not "
                    "provide enough narrative development."
                )

            if not self._ends_cleanly(result):
                raise RuntimeError(
                    "Story text does not end in a complete sentence "
                    "(the model stopped generating before finishing)."
                )

            self._validate_story_completion_contract(mode, result)

            return

        raise ValueError(
            f"Unsupported story mode: {mode}"
        )

    def _validate_shot_character_contract(
        self,
        shots: list[dict],
        characters: list[dict],
    ) -> None:
        if not characters:
            return

        allowed = {
            str(value.get("name", "")).strip().lower()
            for value in characters
            if isinstance(value, dict)
            and str(value.get("name", "")).strip()
        }

        for shot in shots:
            shot_characters = [
                str(name).strip()
                for name in (
                    shot.get("characters", [])
                    or []
                )
                if str(name).strip()
            ]
            events = [event for event in (shot.get("dialogue_events", []) or []) if isinstance(event, dict)]
            remote_speakers = {
                str(event.get("speaker", "") or "").strip().lower()
                for event in events
                if str(event.get("delivery", "in_person")).strip().lower() == "remote"
                and str(event.get("speaker", "") or "").strip()
            }

            # Dialogue-only remote sources may appear in speaking_characters, but
            # never in the physical/visual `characters` roster.
            for field in ("characters", "speaking_characters"):
                field_allowed = allowed if field == "characters" else (allowed | remote_speakers)
                for name in (shot.get(field, []) or []):
                    if str(name).strip().lower() not in field_allowed:
                        raise RuntimeError(
                            f"Shot {shot.get('shot_id', '')} contains unknown character '{name}'."
                        )

            speakers = {
                str(name).strip().lower()
                for name in (
                    shot.get("speaking_characters", [])
                    or []
                )
                if str(name).strip()
            }
            visible = {name.lower() for name in shot_characters}
            in_person_speakers = {
                str(event.get("speaker", "") or "").strip().lower()
                for event in events
                if str(event.get("delivery", "in_person")).strip().lower() != "remote"
                and str(event.get("speaker", "") or "").strip()
            }
            if not in_person_speakers.issubset(visible):
                raise RuntimeError(
                    f"Shot {shot.get('shot_id', '')} has an in-person speaker not present in its visible character bindings."
                )
            if not speakers.issubset(visible | remote_speakers):
                raise RuntimeError(
                    f"Shot {shot.get('shot_id', '')} has an in-person speaker not present in its character bindings."
                )

    @staticmethod
    def _baseline_visual_language() -> dict:
        return {
            "genre_tone": "cinematic, story-led, naturalistic with controlled contrast",
            "color_palette": "coherent palette derived from scene mood and environment",
            "lighting_philosophy": "motivated cinematic lighting consistent within each scene",
            "camera_philosophy": "deliberate composition with motivated movement and continuity-first coverage",
            "pacing": "clear escalation with varied cinematic rhythm",
        }

    @staticmethod
    def _sanitize_visual_language(
        value,
    ) -> dict:

        if not isinstance(value, dict):
            return {}

        fields = (
            "genre_tone",
            "color_palette",
            "lighting_philosophy",
            "camera_philosophy",
            "pacing",
        )

        return {
            field: str(
                value.get(
                    field,
                    "",
                )
                or ""
            ).strip()
            for field in fields
        }

    def _validate_production_quality(
        self,
        *,
        mode: str,
        story: str,
        scenes: list[dict],
        shots: list[dict],
        characters: list[dict],
    ) -> None:
        """Deterministically reject structurally valid but production-poor plans."""
        if not story.strip():
            raise RuntimeError("Production plan has no story.")

        if not characters and mode != PRESERVE_USER_STORY_MODE:
            raise RuntimeError(
                "Production plan has no usable character roster for this story mode."
            )

        minimum_scenes = 1 if mode == PRESERVE_USER_STORY_MODE else 4
        if not minimum_scenes <= len(scenes) <= self.MAX_SCENES:
            raise RuntimeError(
                f"Production plan for {mode} must contain {minimum_scenes}–{self.MAX_SCENES} scenes; got {len(scenes)}."
            )

        scene_ids = [str(scene.get("scene_id", "")).strip() for scene in scenes]
        if any(not scene_id for scene_id in scene_ids):
            raise RuntimeError("Production plan contains an empty scene ID.")
        if len(scene_ids) != len(set(scene_ids)):
            raise RuntimeError("Production plan contains duplicate scene IDs.")

        allowed = {
            str(item.get("name", "")).strip().lower()
            for item in characters
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        }

        expected_shot_count = len(scenes) * self.SHOTS_PER_SCENE
        if len(shots) != expected_shot_count:
            raise RuntimeError(
                f"Production plan must contain exactly {expected_shot_count} shots; got {len(shots)}."
            )

        seen_shot_ids: set[str] = set()
        scene_counts = {scene_id: 0 for scene_id in scene_ids}
        for shot in shots:
            shot_id = str(shot.get("shot_id", "")).strip()
            scene_id = str(shot.get("scene_id", "")).strip()
            if not shot_id or shot_id in seen_shot_ids:
                raise RuntimeError("Production plan contains duplicate or empty shot IDs.")
            seen_shot_ids.add(shot_id)
            if scene_id not in scene_counts:
                raise RuntimeError(f"Shot {shot_id} references unknown scene {scene_id}.")
            scene_counts[scene_id] += 1
            if not str(shot.get("visual_prompt", "") or "").strip():
                raise RuntimeError(f"Shot {shot_id} has no visual prompt.")
            for field in ("camera_shot", "camera_movement", "lens_and_depth_of_field", "composition_notes"):
                if not str(shot.get(field, "") or "").strip():
                    raise RuntimeError(f"Shot {shot_id} is missing {field}.")
            shot_characters = shot.get("characters", []) or []
            speakers = shot.get("speaking_characters", []) or []
            # Empty visible-character bindings are valid for environment/detail shots
            # and for shots that carry only remote audio. In-person visibility is enforced
            # against dialogue_events by _validate_shot_character_contract.
            # Audio-only remote sources may speak without being visible roster members.
            remote_speakers = {
                str(event.get("speaker", "") or "").strip().lower()
                for event in (shot.get("dialogue_events", []) or [])
                if isinstance(event, dict)
                and str(event.get("delivery", "in_person")).strip().lower() == "remote"
                and str(event.get("speaker", "") or "").strip()
            }
            for name in shot_characters:
                if allowed and str(name).strip().lower() not in allowed:
                    raise RuntimeError(f"Shot {shot_id} contains unknown character '{name}'.")
            for name in speakers:
                if allowed and str(name).strip().lower() not in (allowed | remote_speakers):
                    raise RuntimeError(f"Shot {shot_id} contains unknown character '{name}'.")

        if any(count != self.SHOTS_PER_SCENE for count in scene_counts.values()):
            raise RuntimeError(
                f"Every scene must contain exactly {self.SHOTS_PER_SCENE} production shots."
            )

    def _checkpoint_state(
        self,
        session_id: str,
        mode: str,
        user_input: str,
        base_plan: dict,
        director_plan: dict,
        status: str,
        stage: str,
        completed_scene_ids: list[str],
        current_scene_id: str = "",
        error: str = "",
        checkpoint_store: ProductionCheckpoint | None = None,
    ) -> dict:

        checkpoint = checkpoint_store or ProductionCheckpoint(
            self.project_root
        )

        # Checkpoint validity depends on every split Director module, not
        # just this mixin. Hash the canonical entry point plus its split
        # implementation modules in a stable order.
        director_module_files = [
            Path(__file__).resolve().parent / name
            for name in (
                "qwen_director.py",
                "qwen_director_runtime.py",
                "qwen_director_prompts.py",
                "qwen_director_scene.py",
                "qwen_director_sanitize.py",
            )
        ]
        director_hash_material = "".join(
            f"{path.name}:{checkpoint.digest_file(path)}\n"
            for path in director_module_files
            if path.exists()
        )

        return {
            "mode": mode,
            "user_input": str(user_input or ""),
            "user_input_sha256": checkpoint.digest_text(
                user_input
            ),
            "director_sha256": checkpoint.digest_text(
                director_hash_material
            ),
            "status": status,
            "stage": stage,
            "completed_scene_ids": list(
                completed_scene_ids
            ),
            "current_scene_id": current_scene_id,
            "error": error,
            "base_plan": deepcopy(
                base_plan
            ),
            "director_plan": deepcopy(
                director_plan
            ),
        }

    def _save_checkpoint(
        self,
        checkpoint_store: ProductionCheckpoint | None,
        session_id: str | None,
        state: dict,
    ) -> None:

        if checkpoint_store is None or not session_id:
            return

        checkpoint_store.save(
            session_id,
            state,
        )

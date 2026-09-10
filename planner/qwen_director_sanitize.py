from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

from pipeline.production_checkpoint import ProductionCheckpoint

from planner.config import (
    AI_STORY_MODE,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
)


class QwenDirectorSanitizeMixin:
    def _valid_character_name(
        self,
        name: str,
    ) -> bool:

        value = str(
            name or ""
        ).strip()

        if not value:
            return False

        lowered = value.lower()

        if lowered in (
            self.VALID_GENERIC_ROLES
        ):
            return True

        if lowered in (
            self.FORBIDDEN_CHARACTER_NAMES
        ):
            return False

        if len(
            value.split()
        ) > 5:
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

            if not self._valid_character_name(
                name
            ):
                continue

            key = name.lower()

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
        # Keeping the raw sanitized scene list here lets _compress_scenes_to_budget()
        # see every narrative beat instead of silently discarding over-segmented
        # scenes before the semantic compression pass can run.
        return result

    @staticmethod
    def _normalize_shot_response(
        response: dict,
    ) -> list[dict]:

        if not isinstance(
            response,
            dict,
        ):
            return []

        shots = response.get(
            "shots"
        )

        if isinstance(
            shots,
            list,
        ):
            return [
                item
                for item
                in shots
                if isinstance(
                    item,
                    dict,
                )
            ]

        # Qwen3 sometimes returns a single shot object even when the
        # prompt requests {"shots": [...]}. That response is still useful.
        shot_fields = {
            "shot_id",
            "camera_shot",
            "camera_movement",
            "lens_and_depth_of_field",
            "composition_notes",
            "lighting",
            "color_temperature",
            "mood",
            "visual_prompt",
        }

        if (
            shot_fields
            & set(
                response.keys()
            )
        ):
            return [
                response
            ]

        return []

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
                result[scene_id] = [
                    item for item in shots if isinstance(item, dict)
                ]
                continue
            if isinstance(shots, list):
                for item in shots:
                    if not isinstance(item, dict):
                        continue
                    sid = str(item.get("scene_id", "") or "").strip()
                    if sid:
                        result.setdefault(sid, []).append(item)
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

            # Repair non-critical omissions deterministically instead of
            # discarding a useful shot. Use scene-owned values where available.
            scene_description = str(scene.get("description", "") or "").strip()
            scene_lighting = str(scene.get("lighting", "") or "").strip() or "soft natural light"
            scene_color = str(scene.get("color_temperature", "") or "").strip() or "neutral"
            scene_mood = str(scene.get("mood", "") or "").strip() or "cinematic"

            defaults = {
                "camera_shot": "medium wide",
                "camera_movement": "static",
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

            if not selected_characters:

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
                normalized_dialogue.append({
                    "speaker": speaker,
                    "text": text,
                    "continues_from_previous_shot": bool(event.get("continues_from_previous_shot", False)),
                    "continues_to_next_shot": bool(event.get("continues_to_next_shot", False)),
                })
            if not normalized_dialogue:
                legacy_speakers = candidate.get("speaking_characters", []) or []
                legacy_text = str(candidate.get("speech_text", "") or "").strip()
                if legacy_text and legacy_speakers:
                    # Keep the compatibility bridge, but never manufacture a
                    # dialogue event from obviously long narrative prose. The
                    # Director's semantic normalizer remains the final gate.
                    word_count = len(legacy_text.split())
                    quoted = re.findall(r'“([^”\n]+)”|"([^"\n]+)"|‘([^’\n]+)’|(?<!\w)\'([^\'\n]+)\'(?!\w)', legacy_text)
                    quoted_texts = [next((part for part in group if part), "").strip() for group in quoted]
                    quoted_texts = [value for value in quoted_texts if value]
                    direct_speech_like = bool(
                        re.match(
                            r"^(?:i|we|you|your|why|what|how|when|where|please|do not|don't|let's|can|could|would|will|is|are|did|have|has)\b",
                            legacy_text,
                            re.IGNORECASE,
                        )
                    )
                    if quoted_texts:
                        legacy_text = " ".join(quoted_texts)
                        normalized_dialogue = [{
                            "speaker": str(legacy_speakers[0]).strip(),
                            "text": legacy_text,
                            "continues_from_previous_shot": False,
                            "continues_to_next_shot": False,
                        }]
                    elif 0 < word_count <= 32 and direct_speech_like:
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

        return re.sub(
            r"\s+",
            " ",
            str(
                text or ""
            ).strip(),
        )

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
    def _ends_cleanly(text: str) -> bool:
        value = str(text or "").strip()
        if not value:
            return False
        return bool(re.search(r"[.!?][\"')\]]*$", value))

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

            coverage, missing_sentences = self._preservation_coverage(
                source,
                result,
                minimum_sentence_overlap=0.20,
            )
            if source and coverage < 0.5:
                detail = "; ".join(missing_sentences[:3])
                raise RuntimeError(
                    "AI Story mode did not preserve enough of the supplied "
                    f"premise (coverage={coverage:.2f}). {detail}".strip()
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

                if overlap < 0.35:

                    raise RuntimeError(
                        "Expand Story mode changed too much of the supplied story "
                        f"(meaningful-token overlap={overlap:.3f}; minimum=0.350)."
                    )

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

            if not shot_characters:
                raise RuntimeError(
                    f"Shot {shot.get('shot_id', '')} has no character binding."
                )

            for field in (
                "characters",
                "speaking_characters",
            ):
                for name in (
                    shot.get(field, [])
                    or []
                ):
                    if (
                        str(name).strip().lower()
                        not in allowed
                    ):
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
            if not speakers.issubset(
                {
                    name.lower()
                    for name in shot_characters
                }
            ):
                raise RuntimeError(
                    f"Shot {shot.get('shot_id', '')} has a speaker not present in its character bindings."
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

        if not 4 <= len(scenes) <= self.MAX_SCENES:
            raise RuntimeError(
                f"Production plan must contain 4–{self.MAX_SCENES} scenes; got {len(scenes)}."
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
            if allowed and not shot_characters:
                raise RuntimeError(
                    f"Shot {shot_id} has no character binding."
                )
            for name in [*shot_characters, *speakers]:
                if allowed and str(name).strip().lower() not in allowed:
                    raise RuntimeError(f"Shot {shot_id} contains unknown character '{name}'.")

        if any(count != self.SHOTS_PER_SCENE for count in scene_counts.values()):
            raise RuntimeError("Every scene must contain exactly two production shots.")

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
    ) -> dict:

        checkpoint = ProductionCheckpoint(
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

    def _compress_scenes_to_budget(
        self,
        mode: str,
        story: str,
        characters: list[dict],
        scenes: list[dict],
        character_names: set[str],
    ) -> list[dict]:
        """Reduce an over-segmented scene plan while preserving narrative beats.

        This is only called when metadata violates the 4-6 scene contract.
        Prefer one controlled Qwen restructuring pass over silently discarding
        most of the story. If that repair fails, fall back to deterministic
        first/middle/last sampling so production can still continue.
        """
        if len(scenes) <= self.MAX_SCENES:
            return scenes

        scene_payload = []
        for scene in scenes:
            scene_payload.append(
                {
                    "scene_id": str(
                        scene.get("scene_id", "") or ""
                    ).strip(),
                    "order": int(
                        scene.get("order", len(scene_payload) + 1) or (
                            len(scene_payload) + 1
                        )
                    ),
                    "title": str(
                        scene.get("title", "") or ""
                    ).strip(),
                    "description": self._limit_text(
                        scene.get("description", ""),
                        600,
                    ),
                    "location": str(
                        scene.get("location", "") or ""
                    ).strip(),
                    "characters": self._clean_list(
                        scene.get("characters", []),
                        limit=6,
                    ),
                    "scene_objective": self._limit_text(
                        scene.get("scene_objective", ""),
                        180,
                    ),
                    "continuity_notes": self._limit_text(
                        scene.get("continuity_notes", ""),
                        180,
                    ),
                    "scene_function": str(
                        scene.get("scene_function", "development")
                        or "development"
                    ).strip(),
                    "obligatory_moment": self._limit_text(
                        scene.get("obligatory_moment", scene.get("description", "")),
                        220,
                    ),
                }
            )

        system_prompt = """
    You are the STORYBOARD STRUCTURE EDITOR for MiniMax H3.

    The supplied scene list is over-segmented.

    Compress it into exactly 4–6 meaningful narrative scenes.

    Do NOT delete important story events merely to reduce the count.
    Instead MERGE adjacent or closely related beats into stronger scenes.

    Every source scene has a load-bearing obligatory moment.
    Treat that moment as protected narrative information.
    When merging scenes, preserve the obligatory moments of all merged beats inside the resulting scene description/objective.
    Prefer merging adjacent compatible beats rather than deleting beats.
    Never solve the budget by silently truncating the source timeline.

    Preserve:
    - chronological order;
    - protagonist goals;
    - important character introductions;
    - major discoveries;
    - major conflict/escalation;
    - climax;
    - resolution;
    - important locations and continuity.

    Each resulting scene must represent a real dramatic beat.

    Return JSON only:
    {
      "scenes": [
    {
      "scene_id": "scene_001",
      "title": "...",
      "order": 1,
      "location": "...",
      "time_of_day": "...",
      "weather": "...",
      "atmosphere": "...",
      "description": "...",
      "mood": "...",
      "lighting": "...",
      "color_temperature": "...",
      "environment_details": [],
      "key_props": [],
      "characters": [],
      "scene_objective": "...",
      "continuity_notes": "..."
    }
      ]
    }
    """.strip()

        user_payload = json.dumps(
            {
                "mode": mode,
                "story": self._limit_text(story, 4500),
                "characters": [
                    {
                        "name": str(
                            item.get("name", "") or ""
                        ).strip(),
                        "role": str(
                            item.get("role", "") or ""
                        ).strip(),
                    }
                    for item in characters
                    if isinstance(item, dict)
                    and str(
                        item.get("name", "") or ""
                    ).strip()
                ],
                "scenes": scene_payload,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

        try:
            repaired = self._chat_json(
                system_prompt,
                user_payload,
                minimum_completion=500,
                temperature=0.20,
                top_p=0.82,
                call_name="scene_budget_compression",
                max_completion=1600,
                json_mode=True,
                disable_thinking=True,
                response_schema=self._scene_compression_json_schema(),
            )

            compressed = self._sanitize_scenes(
                repaired.get("scenes", []),
                character_names,
            )

            compressed = self._annotate_scene_functions(
                compressed
            )

            if 4 <= len(compressed) <= self.MAX_SCENES:
                for order, scene in enumerate(compressed, start=1):
                    scene["order"] = order
                return compressed

        except Exception as exc:
            print(
                "[QWEN]",
                "scene_budget_compression_failed",
                str(exc),
                flush=True,
            )

        # Deterministic fallback preserves ALL source beats by merging
        # adjacent groups rather than dropping scenes.
        reduced = self._deterministic_compress_scenes(
            scenes,
            target_count=self.MAX_SCENES,
        )

        for order, scene in enumerate(
            reduced,
            start=1,
        ):
            scene["order"] = order

        return reduced

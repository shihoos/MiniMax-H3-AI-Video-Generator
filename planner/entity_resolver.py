from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class AliasDecision:
    alias: str
    canonical: str
    confidence: float
    reason: str


class EntityResolver:
    """
    Deterministic-first character entity resolver.

    Important design rule:
        the resolver NEVER invents the character roster.

    The production roster supplied by Qwen or deterministic extraction
    remains authoritative. This module only resolves references TO that
    approved roster.

    Resolution order:
        1. exact canonical
        2. unique first name
        3. unique full final-name pair
        4. one global semantic Qwen decision for unresolved references
        5. confidence gate >= 0.95
        6. unresolved references remain unresolved
    """

    PRONOUNS = {
        "i",
        "me",
        "my",
        "mine",
        "you",
        "your",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "we",
        "us",
        "our",
        "ours",
        "they",
        "them",
        "their",
        "theirs",
        "this",
        "that",
        "these",
        "those",
    }

    GENERIC_REFERENCES = {
        "the man",
        "the woman",
        "the person",
        "the doctor",
        "the scientist",
        "the captain",
        "the detective",
        "the soldier",
        "the officer",
        "the guard",
        "the stranger",
        "someone",
        "somebody",
        "everyone",
        "nobody",
    }

    HONORIFICS = {
        "dr",
        "doctor",
        "mr",
        "mrs",
        "ms",
        "miss",
        "prof",
        "professor",
        "captain",
        "commander",
        "detective",
        "agent",
    }

    def __init__(
        self,
        qwen=None,
    ) -> None:
        self.qwen = qwen

    @classmethod
    def normalize(
        cls,
        value: str,
    ) -> str:
        value = str(
            value or ""
        ).strip().lower()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value

    @classmethod
    def strip_honorific(
        cls,
        value: str,
    ) -> str:
        tokens = cls.normalize(
            value
        ).split()

        while (
            tokens
            and
            tokens[0].rstrip(".")
            in cls.HONORIFICS
        ):
            tokens.pop(0)

        return " ".join(
            tokens
        )

    @classmethod
    def build_alias_map(
        cls,
        canonical_names,
    ) -> dict[str, str]:

        names = sorted(
            {
                cls.normalize(name)
                for name
                in canonical_names
                if cls.normalize(name)
            }
        )

        aliases = {
            name: name
            for name in names
        }

        first_map: dict[
            str,
            set[str],
        ] = {}

        pair_map: dict[
            str,
            set[str],
        ] = {}

        for canonical in names:

            stripped = cls.strip_honorific(
                canonical
            )

            tokens = stripped.split()

            if not tokens:
                continue

            first_map.setdefault(
                tokens[0],
                set(),
            ).add(
                canonical
            )

            if len(tokens) >= 2:

                pair = " ".join(
                    tokens[-2:]
                )

                pair_map.setdefault(
                    pair,
                    set(),
                ).add(
                    canonical
                )

        for alias, matches in first_map.items():

            if len(matches) == 1:

                aliases[
                    alias
                ] = next(
                    iter(matches)
                )

        for alias, matches in pair_map.items():

            if len(matches) == 1:

                aliases[
                    alias
                ] = next(
                    iter(matches)
                )

        return aliases

    @classmethod
    def is_safe_semantic_reference(
        cls,
        value: str,
    ) -> bool:

        normalized = cls.normalize(
            value
        )

        if not normalized:
            return False

        if normalized in cls.PRONOUNS:
            return False

        if normalized in cls.GENERIC_REFERENCES:
            return False

        if len(normalized) > 80:
            return False

        if any(
            token in normalized
            for token in (
                "{",
                "}",
                "[",
                "]",
                ":",
                ";",
                "|",
            )
        ):
            return False

        return True

    @classmethod
    def collect_unresolved_scene_refs(
        cls,
        scenes,
        aliases: dict[str, str],
    ) -> list[dict]:

        unresolved = {}

        for index, scene in enumerate(
            scenes or [],
            start=1,
        ):

            if not isinstance(
                scene,
                dict,
            ):
                continue

            description = str(
                scene.get(
                    "description",
                    "",
                )
                or ""
            )

            title = str(
                scene.get(
                    "title",
                    "",
                )
                or ""
            )

            objective = str(
                scene.get(
                    "scene_objective",
                    "",
                )
                or ""
            )

            context = " ".join(
                [
                    title,
                    description,
                    objective,
                ]
            ).strip()

            for raw in (
                scene.get(
                    "characters",
                    [],
                )
                or []
            ):

                value = str(
                    raw
                ).strip()

                normalized = cls.normalize(
                    value
                )

                if not normalized:
                    continue

                if normalized in aliases:
                    continue

                if not cls.is_safe_semantic_reference(
                    normalized
                ):
                    continue

                existing = unresolved.get(
                    normalized
                )

                if existing is None:

                    unresolved[
                        normalized
                    ] = {
                        "alias":
                            normalized,

                        "scene_ids":
                            [
                                str(
                                    scene.get(
                                        "scene_id",
                                        f"scene_{index:03d}",
                                    )
                                    or
                                    f"scene_{index:03d}"
                                ).strip()
                            ],

                        "contexts":
                            [
                                context[:700]
                            ],
                    }

                else:

                    scene_id = str(
                        scene.get(
                            "scene_id",
                            f"scene_{index:03d}",
                        )
                        or
                        f"scene_{index:03d}"
                    ).strip()

                    if scene_id not in existing[
                        "scene_ids"
                    ]:
                        existing[
                            "scene_ids"
                        ].append(
                            scene_id
                        )

                    if context and context not in existing[
                        "contexts"
                    ]:
                        existing[
                            "contexts"
                        ].append(
                            context[:700]
                        )

        return list(
            unresolved.values()
        )

    def resolve_scene_aliases(
        self,
        scenes,
        canonical_names,
        *,
        qwen_chat=None,
    ) -> list[dict]:

        raw_scenes = [
            dict(scene)
            for scene
            in scenes or []
            if isinstance(
                scene,
                dict,
            )
        ]

        canonical = {
            self.normalize(name)
            for name
            in canonical_names
            if self.normalize(name)
        }

        aliases = self.build_alias_map(
            canonical
        )

        unresolved = (
            self.collect_unresolved_scene_refs(
                raw_scenes,
                aliases,
            )
        )

        # ALWAYS apply deterministic aliases first.
        # This is the zero-Qwen fast path when every reference
        # already has a confident deterministic mapping.
        for scene in raw_scenes:

            selected = []

            for raw in (
                scene.get(
                    "characters",
                    [],
                )
                or []
            ):

                normalized = self.normalize(
                    raw
                )

                resolved = (
                    aliases.get(normalized)
                    or aliases.get(self.strip_honorific(normalized))
                )

                selected.append(
                    resolved
                    if resolved
                    else normalized
                )

            scene[
                "characters"
            ] = selected

        # No ambiguity means ZERO Qwen calls.
        if not unresolved:
            return raw_scenes

        if qwen_chat is None:
            return raw_scenes

        schema = {
            "type": "object",
            "properties": {
                "aliases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "alias": {
                                "type": "string"
                            },
                            "canonical": {
                                "type": "string"
                            },
                            "confidence": {
                                "type": "number"
                            }
                        },
                        "required": [
                            "alias",
                            "canonical",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": [
                "aliases"
            ],
            "additionalProperties": False,
        }

        payload = {
            "canonical_characters":
                sorted(
                    canonical
                ),

            "ambiguous_references":
                unresolved,
        }

        system_prompt = """
You are a deterministic character-reference resolver.

The canonical character roster is authoritative.

Determine whether each unresolved scene reference refers to exactly
one canonical character.

Rules:
- Do not invent characters.
- Do not rename canonical characters.
- Do not resolve pronouns.
- Do not guess from a weak association.
- Only return a mapping when the evidence is strong.
- Confidence must be 0.0–1.0.
- A confidence below 0.95 is rejected by the caller.
- "Voss" must remain unresolved when multiple canonical characters
  share that surname unless the supplied context unambiguously identifies
  one person.
- Explicit contextual epithets may resolve when the story clearly
  establishes the identity.

Return JSON only.
""".strip()

        user_prompt = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        try:

            response = qwen_chat(
                system_prompt,
                user_prompt,
                minimum_completion=80,
                temperature=0.10,
                top_p=0.80,
                call_name="entity_semantic_resolution",
                max_completion=320,
                json_mode=True,
                disable_thinking=True,
                response_schema=schema,
            )

        except Exception:
            return raw_scenes

        decisions = (
            response.get(
                "aliases",
                [],
            )
            if isinstance(
                response,
                dict,
            )
            else []
        )

        canonical_set = set(
            canonical
        )

        accepted: dict[
            str,
            str,
        ] = {}

        for decision in decisions:

            if not isinstance(
                decision,
                dict,
            ):
                continue

            alias = self.normalize(
                decision.get(
                    "alias",
                    "",
                )
            )

            target = self.normalize(
                decision.get(
                    "canonical",
                    "",
                )
            )

            try:
                confidence = float(
                    decision.get(
                        "confidence",
                        0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                confidence = 0.0

            if (
                not alias
                or
                target not in canonical_set
                or
                confidence < 0.95
            ):
                continue

            accepted[
                alias
            ] = target

        if not accepted:
            return raw_scenes

        for scene in raw_scenes:

            selected = []

            for raw in (
                scene.get(
                    "characters",
                    [],
                )
                or []
            ):

                normalized = self.normalize(
                    raw
                )

                resolved = (
                    aliases.get(normalized)
                    or aliases.get(self.strip_honorific(normalized))
                    or accepted.get(normalized)
                    or accepted.get(self.strip_honorific(normalized))
                )

                selected.append(
                    resolved
                    if resolved
                    else normalized
                )

            scene[
                "characters"
            ] = selected

        return raw_scenes

    @staticmethod
    def merge_resolved_aliases(
        canonical_names,
        raw_names,
    ) -> list[str]:

        aliases = EntityResolver.build_alias_map(
            canonical_names
        )

        result = []

        for value in (
            raw_names or []
        ):

            normalized = EntityResolver.normalize(
                value
            )

            resolved = aliases.get(
                normalized
            )

            result.append(
                resolved
                if resolved
                else normalized
            )

        return result

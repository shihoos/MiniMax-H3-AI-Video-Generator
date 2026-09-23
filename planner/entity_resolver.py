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
        4. unresolved references remain unresolved for Director-level semantic handling

    The resolver deliberately has no live Qwen dependency. Semantic decisions
    are owned and telemetried by QwenDirector; this class only performs
    deterministic identity binding against an already-approved roster.
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

    GENERIC_ROLE_ALIASES = {
        "man", "woman", "boy", "girl", "child", "person",
        "doctor", "scientist", "guard", "officer", "soldier",
        "captain", "commander", "detective", "stranger", "pilot",
        "nurse", "teacher", "engineer", "driver", "officer",
    }

    RELATIONSHIP_LABELS = {
        "father", "mother", "dad", "mom", "parent", "son", "daughter",
        "child", "brother", "sister", "husband", "wife", "partner",
        "fiance", "fiancee", "uncle", "aunt", "cousin", "grandfather",
        "grandmother", "grandson", "granddaughter", "nephew", "niece",
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
    def build_character_alias_map(cls, characters) -> dict[str, str]:
        """Build a deterministic semantic alias map from approved character payloads.

        Generic labels are never canonical identities. They may be aliases only for
        an already-grounded relational character, and only when exactly one owner
        claims the alias.
        """
        payloads = []
        for item in characters or []:
            if isinstance(item, dict):
                payloads.append(item)
            elif hasattr(item, "to_dict"):
                try:
                    value = item.to_dict()
                except Exception:
                    continue
                if isinstance(value, dict):
                    payloads.append(value)

        named_names = []
        relational_names = []
        descriptive_names = []
        semantic_alias_owners: dict[str, set[str]] = {}

        for item in payloads:
            name = str(item.get("name", "") or "").strip()
            if not name:
                continue
            profile = item.get("identity_profile") if isinstance(item.get("identity_profile"), dict) else {}
            identity_type = str(
                item.get("identity_type", profile.get("identity_type", "named_character"))
                or "named_character"
            ).strip().lower()
            normalized_name = cls.normalize(name)
            is_relational = identity_type == "relational_character"
            is_descriptive = identity_type == "descriptive_character"
            if is_relational:
                relational_names.append(name)
            elif is_descriptive:
                descriptive_names.append(name)
            elif normalized_name not in cls.GENERIC_ROLE_ALIASES and normalized_name not in cls.GENERIC_REFERENCES:
                named_names.append(name)
            else:
                continue

            raw_aliases = item.get("semantic_aliases")
            if raw_aliases is None:
                raw_aliases = profile.get("semantic_aliases", [])
            if isinstance(raw_aliases, str):
                raw_aliases = [raw_aliases]
            for raw_alias in raw_aliases or []:
                alias = cls.normalize(str(raw_alias or ""))
                if not alias or alias == normalized_name:
                    continue
                if alias in cls.PRONOUNS:
                    continue
                generic_surface = cls.generic_role_surface(alias)
                if not cls.is_safe_semantic_reference(alias) and not ((is_relational or is_descriptive) and generic_surface):
                    continue
                if generic_surface and not (is_relational or is_descriptive):
                    continue
                semantic_alias_owners.setdefault(alias, set()).add(normalized_name)

        aliases = cls.build_alias_map(named_names)
        for name in [*relational_names, *descriptive_names]:
            normalized = cls.normalize(name)
            if normalized:
                aliases[normalized] = normalized

        for alias, owners in semantic_alias_owners.items():
            if len(owners) != 1:
                continue
            owner = next(iter(owners))
            existing = aliases.get(alias)
            if existing is None or existing == owner:
                aliases[alias] = owner
        return aliases

    @classmethod
    def contextual_generic_alias(
        cls,
        value: str,
        characters,
        *,
        bound_names: set[str] | None = None,
        story: str = "",
    ) -> str | None:
        """Resolve a generic surface only when one grounded relational owner is explicit.

        The method never creates an entity. It only maps an already-approved relational
        character when the current shot binding or story evidence makes ownership unique.
        """
        alias = cls.normalize(value)
        generic_surface = cls.generic_role_surface(alias)
        if not generic_surface:
            return None
        alias = generic_surface
        candidates: list[str] = []
        bound = {cls.normalize(x) for x in (bound_names or set()) if cls.normalize(x)}
        story_lower = cls.normalize(story)
        payloads = []
        for item in characters or []:
            if isinstance(item, dict):
                payloads.append(item)
            elif hasattr(item, "to_dict"):
                try:
                    item = item.to_dict()
                except Exception:
                    continue
                if isinstance(item, dict):
                    payloads.append(item)
        for item in payloads:
            name = str(item.get("name", "") or "").strip()
            profile = item.get("identity_profile") if isinstance(item.get("identity_profile"), dict) else {}
            identity_type = str(item.get("identity_type", profile.get("identity_type", "")) or "").strip().lower()
            if identity_type not in {"relational_character", "descriptive_character"}:
                continue
            norm_name = cls.normalize(name)
            if bound and norm_name not in bound:
                continue
            aliases = item.get("semantic_aliases")
            if aliases is None:
                aliases = profile.get("semantic_aliases", [])
            aliases_norm = {cls.normalize(str(x or "")) for x in (aliases or [])}
            if alias in aliases_norm:
                candidates.append(name)
                continue
            owner = cls.normalize(str(item.get("relationship_to", profile.get("relationship_to", "")) or ""))
            relation = cls.normalize(str(item.get("relationship", profile.get("relationship", "")) or ""))
            generic_pattern = r"\b(?:a|an|the|older|younger|young|old|hooded|masked|another)?\s*" + re.escape(alias) + r"\b"
            if owner and relation and story_lower:
                for owner_match in re.finditer(re.escape(owner), story_lower):
                    start = max(0, owner_match.start() - 250)
                    end = min(len(story_lower), owner_match.end() + 800)
                    window = story_lower[start:end]
                    relation_hit = bool(re.search(r"\b" + re.escape(relation) + r"\b", window))
                    generic_hit = bool(re.search(generic_pattern, window))
                    if relation_hit and generic_hit:
                        candidates.append(name)
                        break

        unique = list(dict.fromkeys(candidates))
        return unique[0] if len(unique) == 1 else None

    @classmethod
    def generic_role_surface(cls, value: str) -> str | None:
        """Return a stable bare role/relation surface for contextual identity binding.

        The returned value is never itself a canonical character. It is only an
        alias candidate that can bind to an already-approved named/relational
        character when ownership is unique.
        """
        normalized = cls.normalize(value)
        surfaces = set(cls.GENERIC_ROLE_ALIASES) | set(cls.RELATIONSHIP_LABELS)
        if normalized in surfaces:
            return normalized
        tokens = normalized.split()
        if not tokens or tokens[-1] not in surfaces:
            return None
        if tokens[-1] in cls.RELATIONSHIP_LABELS:
            allowed_prefixes = {
                "the", "a", "an", "older", "younger", "young", "old",
                "hooded", "masked", "another", "his", "her", "their",
                "my", "our", "your",
            }
        else:
            allowed_prefixes = {
                "the", "a", "an", "older", "younger", "young", "old",
                "hooded", "masked", "another",
            }
        if all(token in allowed_prefixes for token in tokens[:-1]):
            return tokens[-1]
        return None

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

        # Semantic reasoning belongs to QwenDirector. The resolver is a
        # deterministic identity-binding layer and must never hold a live model
        # handle or perform an untelemetried Qwen call. Unresolved aliases remain
        # unresolved here and are handled by the Director-level semantic path.
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

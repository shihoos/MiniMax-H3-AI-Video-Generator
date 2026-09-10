from __future__ import annotations

import re



class QwenDirectorSceneMixin:
    def _resolve_scene_character_aliases(
        self,
        scenes,
        character_names,
    ) -> list[dict]:
        return self._entity_resolver.resolve_scene_aliases(
            scenes,
            character_names,
        )

    @staticmethod
    def _classify_scene_function(
        scene: dict,
        index: int,
        total: int,
    ) -> str:

        text = " ".join(
            [
                str(
                    scene.get(
                        "title",
                        "",
                    )
                    or ""
                ),
                str(
                    scene.get(
                        "description",
                        "",
                    )
                    or ""
                ),
                str(
                    scene.get(
                        "scene_objective",
                        "",
                    )
                    or ""
                ),
            ]
        ).lower()

        # ----------------------------------------------------
        # Structural boundaries
        # ----------------------------------------------------

        if index == 0:
            return "setup"

        if index == total - 1:
            return "finale"

        # ----------------------------------------------------
        # Narrative vocabulary
        #
        # IMPORTANT:
        # Midpoint/revelation must be checked BEFORE catalyst,
        # because words such as "learns", "discovers", and
        # "signal" can appear inside a revelation.
        # ----------------------------------------------------

        climax_terms = (
            "climax",
            "final confrontation",
            "decisive choice",
            "must choose",
            "must shut",
            "collapses",
            "collapse",
            "explodes",
            "final battle",
            "last chance",
            "saves",
            "destroy",
            "destroying",
            "escape",
        )

        midpoint_terms = (
            "midpoint",
            "revelation",
            "reveals the truth",
            "reveals that",
            "reveals",
            "realizes the truth",
            "realizes that",
            "realizes",
            "understands the truth",
            "understands that",
            "understands",
            "discovers the truth",
            "learns the truth",
            "hidden truth",
            "secret is revealed",
            "truth is revealed",
            "turning point",
            "identity",
            "identity is revealed",
        )

        catalyst_terms = (
            "inciting",
            "inciting incident",
            "receives a warning",
            "receives",
            "warning",
            "arrives",
            "unexpected attack",
            "attack",
            "first discovery",
            "initial discovery",
            "discovers a clue",
            "finds a clue",
            "signal appears",
            "signal",
            "learns that",
            "disruption",
        )

        # ----------------------------------------------------
        # Highest-priority dramatic state first.
        # ----------------------------------------------------

        if any(
            term in text
            for term in climax_terms
        ):
            return "climax"

        # A revelation/turning-point interpretation has higher
        # precedence than generic discovery/catalyst language.
        if any(
            term in text
            for term in midpoint_terms
        ):
            return "midpoint"

        if any(
            term in text
            for term in catalyst_terms
        ):
            return "catalyst"

        return "development"

    @classmethod
    def _annotate_scene_functions(
        cls,
        scenes: list[dict],
    ) -> list[dict]:

        result = []

        total = len(
            scenes
        )

        for index, raw_scene in enumerate(
            scenes
        ):

            scene = dict(
                raw_scene
            )

            function = (
                cls._classify_scene_function(
                    scene,
                    index,
                    total,
                )
            )

            description = str(
                scene.get(
                    "description",
                    "",
                )
                or ""
            ).strip()

            objective = str(
                scene.get(
                    "scene_objective",
                    "",
                )
                or ""
            ).strip()

            moment = (
                objective
                or
                description
            )

            moment = re.sub(
                r"\s+",
                " ",
                moment,
            ).strip()

            if len(moment) > 220:
                moment = (
                    moment[:220]
                    .rsplit(
                        " ",
                        1,
                    )[0]
                    .strip()
                )

            scene[
                "scene_function"
            ] = function

            scene[
                "obligatory_moment"
            ] = moment

            result.append(
                scene
            )

        return result

    @classmethod
    def _scene_function_coverage(
        cls,
        scenes: list[dict],
    ) -> set[str]:

        return {
            str(
                scene.get(
                    "scene_function",
                    "",
                )
                or ""
            ).strip().lower()
            for scene in scenes or []
            if isinstance(
                scene,
                dict,
            )
        }

    @staticmethod
    def _merge_scene_group(
        group: list[dict],
        index: int,
    ) -> dict:

        first = dict(
            group[0]
        )

        descriptions = [
            str(item.get("description", "") or "").strip()
            for item in group
            if str(item.get("description", "") or "").strip()
        ]

        story_summaries = [
            str(item.get("story_summary", "") or "").strip()
            for item in group
            if str(item.get("story_summary", "") or "").strip()
        ]
        obligatory = [
            str(item.get("obligatory_moment", "") or "").strip()
            for item in group
            if str(item.get("obligatory_moment", "") or "").strip()
        ]
        narrative_beats = [
            str(item.get(key, "") or "").strip()
            for item in group
            for key in ("narrative_beat", "key_event", "event")
            if str(item.get(key, "") or "").strip()
        ]

        objectives = [
            str(
                item.get(
                    "scene_objective",
                    "",
                )
                or ""
            ).strip()
            for item in group
        ]

        objectives = [
            value
            for value in objectives
            if value
        ]

        continuity = [
            str(
                item.get(
                    "continuity_notes",
                    "",
                )
                or ""
            ).strip()
            for item in group
        ]

        continuity = [
            value
            for value in continuity
            if value
        ]

        env = []
        props = []
        chars = []

        for item in group:

            for value in (
                item.get(
                    "environment_details",
                    [],
                )
                or []
            ):
                text = str(
                    value
                ).strip()
                if text and text not in env:
                    env.append(
                        text
                    )

            for value in (
                item.get(
                    "key_props",
                    [],
                )
                or []
            ):
                text = str(
                    value
                ).strip()
                if text and text not in props:
                    props.append(
                        text
                    )

            for value in (
                item.get(
                    "characters",
                    [],
                )
                or []
            ):
                text = str(
                    value
                ).strip()
                if text and text not in chars:
                    chars.append(
                        text
                    )

        first[
            "scene_id"
        ] = str(
            first.get(
                "scene_id",
                f"scene_{index:03d}",
            )
            or
            f"scene_{index:03d}"
        ).strip()

        first[
            "order"
        ] = index

        first[
            "description"
        ] = " ".join(descriptions).strip()

        if story_summaries:
            first["story_summary"] = " ".join(story_summaries).strip()
        if obligatory:
            first["obligatory_moment"] = " ".join(obligatory).strip()
        if narrative_beats:
            first["narrative_beat"] = " ".join(narrative_beats).strip()

        first[
            "scene_objective"
        ] = " ".join(objectives).strip()

        first[
            "continuity_notes"
        ] = " ".join(
            continuity
        ).strip()

        first[
            "environment_details"
        ] = env[:12]

        first[
            "key_props"
        ] = props[:8]

        first[
            "characters"
        ] = chars[:6]

        return first

    @classmethod
    def _deterministic_compress_scenes(
        cls,
        scenes: list[dict],
        target_count: int = 6,
    ) -> list[dict]:

        if len(scenes) <= target_count:
            return cls._annotate_scene_functions(
                scenes
            )

        target_count = max(
            4,
            min(
                target_count,
                len(scenes),
            )
        )

        total = len(
            scenes
        )

        groups = []

        start = 0

        for group_index in range(
            target_count
        ):

            remaining_items = (
                total - start
            )

            remaining_groups = (
                target_count
                - group_index
            )

            size = (
                (remaining_items + remaining_groups - 1)
                // remaining_groups
            )

            end = min(
                total,
                start + size,
            )

            groups.append(
                scenes[
                    start:end
                ]
            )

            start = end

        merged = []

        for index, group in enumerate(
            groups,
            start=1,
        ):

            merged.append(
                cls._merge_scene_group(
                    group,
                    index,
                )
            )

        # Re-tag after merging because the structural positions changed.
        return cls._annotate_scene_functions(
            merged
        )

    @classmethod
    def _deterministic_expand_scenes(
        cls,
        story: str,
        existing_scenes: list[dict],
        characters: list[dict],
    ) -> list[dict]:

        story = str(
            story or ""
        ).strip()

        if not story:
            return []

        sentences = [
            sentence.strip()
            for sentence
            in re.split(
                r"(?<=[.!?])\s+",
                story,
            )
            if sentence.strip()
        ]

        if len(sentences) < 4:
            # If source prose itself is short, use the existing scenes
            # and do not fabricate events.
            return cls._annotate_scene_functions(
                existing_scenes
            )

        target = min(
            6,
            max(
                4,
                len(existing_scenes),
            ),
        )

        target = min(
            target,
            len(sentences),
        )

        groups = [
            []
            for _ in range(
                target
            )
        ]

        for index, sentence in enumerate(
            sentences
        ):

            bucket = min(
                target - 1,
                int(
                    index
                    * target
                    /
                    max(
                        1,
                        len(sentences),
                    )
                ),
            )

            groups[
                bucket
            ].append(
                sentence
            )

        outputs = []

        fallback_characters = [
            str(
                item.get(
                    "name",
                    "",
                )
                or ""
            ).strip()
            for item in characters
            if isinstance(
                item,
                dict,
            )
            and
            str(
                item.get(
                    "name",
                    "",
                )
                or ""
            ).strip()
        ]

        for index, group in enumerate(
            groups,
            start=1,
        ):

            description = " ".join(
                group
            ).strip()

            if not description:
                continue

            template = {}

            if existing_scenes:

                template = dict(
                    existing_scenes[
                        min(
                            index - 1,
                            len(existing_scenes) - 1,
                        )
                    ]
                )

            template.update(
                {
                    "scene_id":
                        f"scene_{index:03d}",

                    "title":
                        str(
                            template.get(
                                "title",
                                "",
                            )
                            or ""
                        ).strip()
                        or
                        f"Story Beat {index}",

                    "order":
                        index,

                    "description":
                        description,

                    "characters":
                        fallback_characters[
                            :6
                        ],

                    "scene_objective":
                        "Advance the source narrative while preserving its event.",

                    "continuity_notes":
                        "Preserve chronology and character continuity.",
                }
            )

            outputs.append(
                template
            )

        return cls._annotate_scene_functions(
            outputs
        )

    def _planner(
        self,
    ):

        if self._fallback_planner is None:

            from planner.production_planner import (
                ProductionPlanner,
            )

            self._fallback_planner = (
                ProductionPlanner(
                    self.project_root
                )
            )

        return self._fallback_planner

    def _build_deterministic_fallback(
        self,
        story: str,
    ) -> tuple[
        list[dict],
        list[dict],
    ]:

        story = str(
            story or ""
        ).strip()

        if not story:

            return (
                [],
                [],
            )

        planner = self._planner()

        characters = []
        scenes = []

        try:

            characters = (
                planner.create_characters(
                    story
                )
                or []
            )

        except Exception:

            characters = []

        try:

            scenes = (
                planner.create_scenes(
                    story,
                    characters,
                )
                or []
            )

        except Exception:

            scenes = []

        return (
            [
                character.to_dict()
                for character
                in characters
                if character is not None
            ],
            [
                scene.to_dict()
                for scene
                in scenes
                if scene is not None
            ],
        )

from __future__ import annotations

import re



class QwenDirectorSceneMixin:
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


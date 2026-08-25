from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from planner.config import (
    H3_AUDIO_VAE,
    H3_REF2VA_MODEL,
    H3_TEXT_ENCODER,
    H3_VIDEO_VAE,
    H3_TURBO_LORA,
    TURBO_STEPS,
)


class H3WorkflowBuilder:

    MODES = {
        "ref2v": (
            "generation",
            "H3_Ref2V_Production.json",
        ),
        "turbo_ref2v": (
            "generation",
            "H3_Turbo_Ref2V_Production.json",
        ),
        "upscale": (
            "postprocess",
            "MMH3_Ultimate_Upscale.json",
        ),
    }

    def __init__(
        self,
        project_root: Path,
        comfy_client,
    ):
        self.project_root = Path(
            project_root
        )

        self.client = comfy_client

        self.root = (
            self.project_root
            / "workflows"
            / "MiniMax-H3"
        )

    def path_for_mode(
        self,
        mode: str,
    ) -> Path:

        try:
            directory, filename = (
                self.MODES[mode]
            )
        except KeyError as error:
            raise ValueError(
                f"Unknown H3 workflow mode: {mode}"
            ) from error

        path = (
            self.root
            / directory
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Workflow missing: {path}"
            )

        return path

    def load(
        self,
        mode: str,
    ) -> dict[str, Any]:

        path = self.path_for_mode(
            mode
        )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            data,
            dict,
        ):
            raise RuntimeError(
                f"Invalid workflow: {path}"
            )

        return copy.deepcopy(data)

    @staticmethod
    def _nodes(
        workflow: dict,
    ):
        return workflow.get(
            "nodes",
            []
        )

    @staticmethod
    def _widgets(
        node: dict,
    ) -> list:

        widgets = node.setdefault(
            "widgets_values",
            []
        )

        if not isinstance(
            widgets,
            list,
        ):
            raise RuntimeError(
                f"Invalid widgets_values in "
                f"node {node.get('id')}"
            )

        return widgets

    @staticmethod
    def _set_widget(
        node: dict,
        index: int,
        value: Any,
    ) -> None:

        widgets = H3WorkflowBuilder._widgets(
            node
        )

        while len(widgets) <= index:
            widgets.append(None)

        widgets[index] = value

    @staticmethod
    def _type(
        node: dict,
    ) -> str:
        return str(
            node.get(
                "type",
                ""
            )
        )

    @staticmethod
    def _title(
        node: dict,
    ) -> str:
        return str(
            node.get(
                "title",
                ""
            )
        ).lower()

    def _find(
        self,
        workflow: dict,
        *,
        types: set[str] | None = None,
        title_contains: str | None = None,
    ) -> list[dict]:

        result = []

        for node in self._nodes(
            workflow
        ):

            if (
                types
                and self._type(node)
                not in types
            ):
                continue

            if (
                title_contains
                and title_contains.lower()
                not in self._title(node)
            ):
                continue

            result.append(node)

        return result

    def _patch_common_models(
        self,
        workflow: dict,
    ) -> None:

        found_diffusion = False
        found_text = False

        for node in self._nodes(
            workflow
        ):

            node_type = self._type(
                node
            )

            widgets = (
                self._widgets(node)
            )

            text = " ".join(
                str(value).lower()
                for value in widgets
            )

            if node_type in {
                "UNETLoader",
                "H3ModelLoaderAny",
            }:
                if widgets:
                    self._set_widget(
                        node,
                        0,
                        H3_REF2VA_MODEL,
                    )
                    found_diffusion = True

            elif node_type in {
                "CLIPLoader",
                "H3ClipLoaderAny",
            }:
                if widgets:
                    self._set_widget(
                        node,
                        0,
                        H3_TEXT_ENCODER,
                    )
                    found_text = True

            elif node_type == "VAELoader":

                if "audio" in text:
                    self._set_widget(
                        node,
                        0,
                        H3_AUDIO_VAE,
                    )

                elif "video" in text:
                    self._set_widget(
                        node,
                        0,
                        H3_VIDEO_VAE,
                    )

                elif "minimax_h3" in text:
                    # Only change if the filename itself identifies VAE type.
                    if "audio" in text:
                        self._set_widget(
                            node,
                            0,
                            H3_AUDIO_VAE,
                        )
                    else:
                        self._set_widget(
                            node,
                            0,
                            H3_VIDEO_VAE,
                        )

        if not found_diffusion:
            raise RuntimeError(
                "No diffusion model loader found "
                "in H3 workflow."
            )

        if not found_text:
            raise RuntimeError(
                "No H3 text encoder loader found."
            )

    def _patch_prompt(
        self,
        workflow: dict,
        prompt: str,
    ) -> None:

        candidates = []

        for node in self._nodes(
            workflow
        ):

            node_type = self._type(
                node
            )

            if node_type not in {
                "PrimitiveStringMultiline",
                "PrimitiveString",
            }:
                continue

            if (
                "prompt"
                in self._title(node)
            ):
                candidates.append(
                    node
                )

        if not candidates:
            raise RuntimeError(
                "No prompt widget found."
            )

        self._set_widget(
            candidates[0],
            0,
            prompt,
        )

    def _patch_seed(
        self,
        workflow: dict,
        seed: int,
    ) -> None:

        for node in self._find(
            workflow,
            types={"RandomNoise"},
        ):
            self._set_widget(
                node,
                0,
                int(seed),
            )

    def _patch_steps(
        self,
        workflow: dict,
        turbo: bool,
    ) -> None:

        steps = (
            TURBO_STEPS
            if turbo
            else 14
        )

        for node in self._find(
            workflow,
            types={"BasicScheduler"},
        ):
            widgets = self._widgets(
                node
            )

            if len(widgets) > 1:
                self._set_widget(
                    node,
                    1,
                    steps,
                )

    def _validate_turbo(
        self,
        workflow: dict,
    ) -> None:

        turbo_nodes = [
            node
            for node in self._nodes(
                workflow
            )
            if "Turbo" in self._type(node)
        ]

        if not turbo_nodes:
            raise RuntimeError(
                "Turbo workflow contains no MiniMax-H3 "
                "Turbo node. It cannot be called Turbo."
            )

    def build(
        self,
        *,
        mode: str,
        prompt: str,
        seed: int = 0,
    ) -> dict:

        workflow = self.load(
            mode
        )

        self._patch_common_models(
            workflow
        )

        self._patch_prompt(
            workflow,
            prompt
        )

        self._patch_seed(
            workflow,
            seed
        )

        turbo = (
            mode == "turbo_ref2v"
        )

        self._patch_steps(
            workflow,
            turbo
        )

        if turbo:
            self._validate_turbo(
                workflow
            )

        return self.client.convert_workflow(
            workflow
        )

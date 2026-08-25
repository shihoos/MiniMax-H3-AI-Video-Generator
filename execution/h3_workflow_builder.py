from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from planner.config import (
    H3_AUDIO_VAE,
    H3_HEIGHT,
    H3_REF2VA_MODEL,
    H3_STEPS,
    H3_TEXT_ENCODER,
    H3_VIDEO_VAE,
    H3_WIDTH,
    H3_TURBO_LORA,
    TURBO_STEPS,
)


class H3WorkflowBuilder:

    WORKFLOWS = {
        "ref2v": (
            "base",
            "H3_HardMode_R2V.json",
        ),
        "ref2v_chain": (
            "base",
            "H3_HardMode_R2V.json",
        ),
        "turbo_ref2v": (
            "turbo",
            "H3_Turbo_Ref2V.json",
        ),
    }

    REQUIRED_NODE_TYPES = {
        "H3ModelLoaderAny",
        "H3ClipLoaderAny",
        "MiniMaxH3ReferenceToVideo",
        "VAELoader",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
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
                self.WORKFLOWS[mode]
            )
        except KeyError as error:
            raise ValueError(
                f"Unsupported H3 workflow mode: {mode}"
            ) from error

        path = (
            self.root
            / directory
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        return path

    @staticmethod
    def _set_widget(
        node: dict[str, Any],
        index: int,
        value: Any,
    ) -> None:

        values = node.setdefault(
            "widgets_values",
            [],
        )

        while len(values) <= index:
            values.append(None)

        values[index] = value

    @staticmethod
    def _nodes(
        workflow: dict[str, Any],
        node_type: str,
    ):

        return [
            node
            for node in workflow.get(
                "nodes",
                [],
            )
            if node.get("type") == node_type
        ]

    def _load(
        self,
        mode: str,
    ) -> dict:

        path = self.path_for_mode(
            mode
        )

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            workflow = json.load(handle)

        if not isinstance(
            workflow,
            dict,
        ):
            raise ValueError(
                f"Invalid workflow: {path}"
            )

        return workflow

    def _validate_node_types(
        self,
        workflow: dict,
    ) -> None:

        actual = {
            str(node.get("type"))
            for node in workflow.get(
                "nodes",
                [],
            )
        }

        missing = (
            self.REQUIRED_NODE_TYPES
            - actual
        )

        if missing:
            raise RuntimeError(
                "Workflow is missing required H3 nodes: "
                + ", ".join(sorted(missing))
            )

    def _patch_models(
        self,
        workflow: dict,
        turbo: bool,
    ) -> None:

        model_nodes = (
            self._nodes(
                workflow,
                "H3ModelLoaderAny",
            )
        )

        if not model_nodes:
            raise RuntimeError(
                "No H3ModelLoaderAny node found."
            )

        for node in model_nodes:
            self._set_widget(
                node,
                0,
                H3_REF2VA_MODEL,
            )

        clip_nodes = (
            self._nodes(
                workflow,
                "H3ClipLoaderAny",
            )
        )

        if not clip_nodes:
            raise RuntimeError(
                "No H3ClipLoaderAny node found."
            )

        for node in clip_nodes:
            self._set_widget(
                node,
                0,
                H3_TEXT_ENCODER,
            )

        for node in self._nodes(
            workflow,
            "VAELoader",
        ):

            title = str(
                node.get(
                    "title",
                    "",
                )
            ).lower()

            if "audio" in title:
                self._set_widget(
                    node,
                    0,
                    H3_AUDIO_VAE,
                )

            elif "video" in title:
                self._set_widget(
                    node,
                    0,
                    H3_VIDEO_VAE,
                )

        if turbo:
            self._patch_turbo(
                workflow
            )

    def _patch_turbo(
        self,
        workflow: dict,
    ) -> None:

        candidates = []

        for node in workflow.get(
            "nodes",
            [],
        ):
            node_type = str(
                node.get(
                    "type",
                    "",
                )
            )

            if (
                "LoraLoader"
                in node_type
            ):
                candidates.append(node)

        if not candidates:
            raise RuntimeError(
                "Turbo workflow does not expose a "
                "LoRA loader node. Do not silently "
                "pretend Step600 Turbo is active."
            )

        for node in candidates:

            widgets = node.setdefault(
                "widgets_values",
                [],
            )

            if not widgets:
                widgets.append(None)

            widgets[0] = H3_TURBO_LORA

            if len(widgets) > 1:
                widgets[1] = 1.0

            if len(widgets) > 2:
                widgets[2] = 1.0

    def _patch_resolution(
        self,
        workflow: dict,
        width: int,
        height: int,
        steps: int,
    ) -> None:

        for node in self._nodes(
            workflow,
            "ResolutionSelector",
        ):
            self._set_widget(
                node,
                1,
                (width * height)
                / 1_000_000,
            )

        for node in self._nodes(
            workflow,
            "BasicScheduler",
        ):
            self._set_widget(
                node,
                1,
                steps,
            )

    def _patch_prompt(
        self,
        workflow: dict,
        prompt: str,
    ) -> None:

        candidates = [
            node
            for node in workflow.get(
                "nodes",
                [],
            )
            if node.get("type")
            in {
                "PrimitiveStringMultiline",
                "PrimitiveString",
            }
            and (
                "prompt"
                in str(
                    node.get(
                        "title",
                        "",
                    )
                ).lower()
            )
        ]

        if not candidates:
            raise RuntimeError(
                "H3 prompt input node not found."
            )

        node = candidates[0]

        self._set_widget(
            node,
            0,
            prompt,
        )

    def _patch_seed(
        self,
        workflow: dict,
        seed: int,
    ) -> None:

        for node in self._nodes(
            workflow,
            "RandomNoise",
        ):
            self._set_widget(
                node,
                0,
                int(seed),
            )

    def build(
        self,
        *,
        mode: str,
        prompt: str,
        width: int = H3_WIDTH,
        height: int = H3_HEIGHT,
        steps: int = H3_STEPS,
        seed: int = 0,
    ) -> dict:

        workflow = copy.deepcopy(
            self._load(mode)
        )

        self._validate_node_types(
            workflow
        )

        self._patch_models(
            workflow,
            turbo=(
                mode == "turbo_ref2v"
            ),
        )

        self._patch_resolution(
            workflow,
            width,
            height,
            (
                TURBO_STEPS
                if mode == "turbo_ref2v"
                else steps
            ),
        )

        self._patch_prompt(
            workflow,
            prompt,
        )

        self._patch_seed(
            workflow,
            seed,
        )

        return (
            self.client.convert_workflow(
                workflow
            )
        )

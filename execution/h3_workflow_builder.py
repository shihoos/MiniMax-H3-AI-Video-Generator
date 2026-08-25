from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class H3WorkflowBuilder:
    """
    Loads the canonical production MiniMax H3 workflows.

    Production workflows are already constructed and statically validated.
    This builder only applies runtime-safe inputs such as prompt, seed,
    reference images, and Turbo step count.

    It never mutates source/reference workflows.
    """

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
            "H3_Ref2V_UltimateUpscale_Production.json",
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
            directory, filename = self.MODES[mode]
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
                f"Canonical H3 workflow missing: {path}"
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
                f"Invalid workflow object: {path}"
            )

        if not data.get("nodes"):
            raise RuntimeError(
                f"Workflow contains no nodes: {path}"
            )

        return copy.deepcopy(data)

    @staticmethod
    def _nodes(
        workflow: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return workflow.get(
            "nodes",
            []
        )

    @staticmethod
    def _type(
        node: dict[str, Any],
    ) -> str:
        return str(
            node.get(
                "type",
                ""
            )
        )

    @staticmethod
    def _title(
        node: dict[str, Any],
    ) -> str:
        return str(
            node.get(
                "title",
                ""
            )
        ).lower()

    @staticmethod
    def _widgets(
        node: dict[str, Any],
    ) -> list[Any]:

        widgets = node.setdefault(
            "widgets_values",
            []
        )

        if not isinstance(
            widgets,
            list,
        ):
            raise RuntimeError(
                f"Invalid widgets_values in node "
                f"{node.get('id')}"
            )

        return widgets

    @classmethod
    def _set_widget(
        cls,
        node: dict[str, Any],
        index: int,
        value: Any,
    ) -> None:

        widgets = cls._widgets(
            node
        )

        while len(widgets) <= index:
            widgets.append(None)

        widgets[index] = value

    def _find(
        self,
        workflow: dict[str, Any],
        node_type: str,
    ) -> list[dict[str, Any]]:

        return [
            node
            for node in self._nodes(workflow)
            if self._type(node) == node_type
        ]

    def _find_one(
        self,
        workflow: dict[str, Any],
        node_type: str,
    ) -> dict[str, Any]:

        nodes = self._find(
            workflow,
            node_type,
        )

        if len(nodes) != 1:
            raise RuntimeError(
                f"Expected exactly one {node_type}; "
                f"found {len(nodes)}"
            )

        return nodes[0]

    def _set_prompt(
        self,
        workflow: dict[str, Any],
        prompt: str,
    ) -> None:

        candidates = [
            node
            for node in self._nodes(workflow)
            if self._type(node)
            in {
                "PrimitiveStringMultiline",
                "PrimitiveString",
            }
        ]

        if not candidates:
            raise RuntimeError(
                "No prompt string node exists "
                "in the H3 production workflow."
            )

        # Prefer a node explicitly labelled as prompt.
        labelled = [
            node
            for node in candidates
            if "prompt" in self._title(node)
        ]

        target = (
            labelled[0]
            if labelled
            else candidates[0]
        )

        self._set_widget(
            target,
            0,
            str(prompt),
        )

    def _set_seed(
        self,
        workflow: dict[str, Any],
        seed: int,
    ) -> None:

        noise_nodes = self._find(
            workflow,
            "RandomNoise",
        )

        if not noise_nodes:
            raise RuntimeError(
                "H3 workflow has no RandomNoise node."
            )

        for node in noise_nodes:
            self._set_widget(
                node,
                0,
                int(seed),
            )

    def _set_turbo_steps(
        self,
        workflow: dict[str, Any],
        steps: int,
    ) -> None:

        if int(steps) != 8:
            raise ValueError(
                "This project uses the validated "
                "8-step Turbo workflow."
            )

        scheduler = self._find_one(
            workflow,
            "BasicScheduler",
        )

        widgets = self._widgets(
            scheduler
        )

        if len(widgets) < 2:
            raise RuntimeError(
                "Turbo BasicScheduler is malformed."
            )

        # H3 scheduler:
        # [scheduler, steps, denoise]
        self._set_widget(
            scheduler,
            1,
            8,
        )

    def _validate_turbo(
        self,
        workflow: dict[str, Any],
    ) -> None:

        lora_nodes = self._find(
            workflow,
            "MiniMaxH3TurboLoRA",
        )

        sampler_nodes = self._find(
            workflow,
            "MiniMaxH3TurboSampler",
        )

        if len(lora_nodes) != 1:
            raise RuntimeError(
                "Production Turbo workflow must contain "
                "exactly one MiniMaxH3TurboLoRA."
            )

        if len(sampler_nodes) != 1:
            raise RuntimeError(
                "Production Turbo workflow must contain "
                "exactly one MiniMaxH3TurboSampler."
            )

        lora_widgets = self._widgets(
            lora_nodes[0]
        )

        if not lora_widgets:
            raise RuntimeError(
                "Turbo LoRA node has no widgets."
            )

        expected = (
            "minimax_h3_turbo_v4_step600_ema.safetensors"
        )

        if lora_widgets[0] != expected:
            raise RuntimeError(
                "Production Turbo workflow is not using "
                f"the locked Step600 LoRA: {lora_widgets[0]}"
            )

    def _validate_ref2v(
        self,
        workflow: dict[str, Any],
    ) -> None:

        self._find_one(
            workflow,
            "UNETLoader",
        )

        self._find_one(
            workflow,
            "CLIPLoader",
        )

        self._find_one(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

        self._find_one(
            workflow,
            "SamplerCustomAdvanced",
        )

    def _validate_upscale(
        self,
        workflow: dict[str, Any],
    ) -> None:

        self._find_one(
            workflow,
            "MMH3UltimateUpscale",
        )

        param = self._find_one(
            workflow,
            "MMH3LatentUpscaleWithModelParams",
        )

        widgets = self._widgets(
            param
        )

        expected = (
            "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        )

        if not widgets or widgets[0] != expected:
            raise RuntimeError(
                "Production upscale workflow is not using "
                f"the locked 3D H3 upscaler: "
                f"{widgets[0] if widgets else None}"
            )

    def build(
        self,
        *,
        mode: str,
        prompt: str = "",
        seed: int = 0,
        turbo_steps: int = 8,
    ) -> dict[str, Any]:

        workflow = self.load(
            mode
        )

        if mode == "ref2v":

            self._validate_ref2v(
                workflow
            )

            if prompt:
                self._set_prompt(
                    workflow,
                    prompt,
                )

            self._set_seed(
                workflow,
                seed,
            )

        elif mode == "turbo_ref2v":

            self._validate_ref2v(
                workflow
            )

            self._validate_turbo(
                workflow
            )

            if prompt:
                self._set_prompt(
                    workflow,
                    prompt,
                )

            self._set_seed(
                workflow,
                seed,
            )

            self._set_turbo_steps(
                workflow,
                turbo_steps,
            )

        elif mode == "upscale":

            self._validate_ref2v(
                workflow
            )

            self._validate_upscale(
                workflow
            )

        else:
            raise ValueError(
                f"Unsupported H3 workflow mode: {mode}"
            )

        return self.client.convert_workflow(
            workflow
        )

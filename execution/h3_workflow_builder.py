from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from planner.config import (
    H3_AUDIO_VAE,
    H3_LATENT_UPSCALER_3D,
    H3_REF2VA_MODEL,
    H3_TEXT_ENCODER,
    H3_TURBO_LORA,
    H3_VIDEO_VAE,
    TURBO_STEPS,
)


class H3WorkflowBuilder:

    WORKFLOWS = {
        "ref2v": (
            "generation",
            "H3_Ref2V_Production.json",
        ),
        "turbo_source": (
            "generation",
            "H3_Turbo_Reference_Source.json",
        ),
        "latent_upscaler_source": (
            "postprocess",
            "H3_LatentUpscaler_Source.json",
        ),
        "ultimate_upscale_source": (
            "postprocess",
            "MMH3_UltimateUpscale_Source.json",
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
        )

    def load(
        self,
        workflow_name: str,
    ) -> dict[str, Any]:

        try:
            directory, filename = (
                self.WORKFLOWS[workflow_name]
            )
        except KeyError as error:
            raise ValueError(
                f"Unknown workflow: {workflow_name}"
            ) from error

        path = (
            self.root
            / directory
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(path)

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
        workflow: dict[str, Any],
        node_type: str,
    ) -> list[dict]:

        return [
            node
            for node in workflow.get(
                "nodes",
                [],
            )
            if node.get("type")
            == node_type
        ]

    @staticmethod
    def _set_widget(
        node: dict,
        index: int,
        value: Any,
    ) -> None:

        widgets = node.setdefault(
            "widgets_values",
            [],
        )

        if not isinstance(
            widgets,
            list,
        ):
            raise RuntimeError(
                f"Node {node.get('id')} has "
                "invalid widgets_values."
            )

        while len(widgets) <= index:
            widgets.append(None)

        widgets[index] = value

    def patch_ref2v_models(
        self,
        workflow: dict[str, Any],
    ) -> dict[str, Any]:

        found_unet = False
        found_clip = False
        found_video_vae = False
        found_audio_vae = False

        for node in workflow.get(
            "nodes",
            [],
        ):

            node_type = node.get(
                "type"
            )

            if node_type == "UNETLoader":

                self._set_widget(
                    node,
                    0,
                    H3_REF2VA_MODEL,
                )

                found_unet = True

            elif node_type == "CLIPLoader":

                widgets = node.get(
                    "widgets_values",
                    [],
                )

                if not isinstance(
                    widgets,
                    list,
                ) or len(widgets) < 1:
                    raise RuntimeError(
                        f"Invalid CLIPLoader node "
                        f"{node.get('id')}"
                    )

                self._set_widget(
                    node,
                    0,
                    H3_TEXT_ENCODER,
                )

                found_clip = True

            elif node_type == "VAELoader":

                widgets = node.get(
                    "widgets_values",
                    [],
                )

                text = " ".join(
                    str(value).lower()
                    for value in widgets
                )

                if "audio" in text:
                    self._set_widget(
                        node,
                        0,
                        H3_AUDIO_VAE,
                    )
                    found_audio_vae = True

                elif "video" in text:
                    self._set_widget(
                        node,
                        0,
                        H3_VIDEO_VAE,
                    )
                    found_video_vae = True

        missing = []

        if not found_unet:
            missing.append("UNETLoader")

        if not found_clip:
            missing.append("CLIPLoader")

        if not found_video_vae:
            missing.append("video VAELoader")

        if not found_audio_vae:
            missing.append("audio VAELoader")

        if missing:
            raise RuntimeError(
                "Ref2V workflow is missing: "
                + ", ".join(missing)
            )

        return workflow

    def build_ref2v(
        self,
        prompt: str,
        seed: int,
    ) -> dict[str, Any]:

        workflow = self.load("ref2v")

        workflow = self.patch_ref2v_models(
            workflow
        )

        prompt_nodes = [
            node
            for node in workflow["nodes"]
            if node.get("type")
            == "PrimitiveStringMultiline"
        ]

        if not prompt_nodes:
            raise RuntimeError(
                "No PrimitiveStringMultiline "
                "prompt node found."
            )

        self._set_widget(
            prompt_nodes[0],
            0,
            prompt,
        )

        for node in self._nodes(
            workflow,
            "RandomNoise",
        ):
            self._set_widget(
                node,
                0,
                int(seed),
            )

        return self.client.convert_workflow(
            workflow
        )

    def validate_source(
        self,
        workflow_name: str,
    ) -> dict[str, Any]:

        workflow = self.load(
            workflow_name
        )

        node_types = {
            node.get("type")
            for node in workflow.get(
                "nodes",
                [],
            )
        }

        return {
            "workflow": workflow_name,
            "nodes": len(
                workflow.get(
                    "nodes",
                    [],
                )
            ),
            "node_types": sorted(
                str(node_type)
                for node_type in node_types
            ),
            "has_unet": (
                "UNETLoader" in node_types
            ),
            "has_clip": (
                "CLIPLoader" in node_types
            ),
            "has_h3_ref2v": (
                "MiniMaxH3ReferenceToVideo"
                in node_types
            ),
            "has_turbo": (
                "MiniMaxH3TurboLoRA"
                in node_types
            ),
            "has_upscaler": (
                "MinimaxH3LatentUpscaler3D"
                in node_types
            ),
            "has_ultimate_upscale": (
                "MMH3UltimateUpscale"
                in node_types
            ),
        }

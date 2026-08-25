from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any


class H3WorkflowBuilder:
    """
    Runtime builder for the canonical MiniMax H3 production workflows.

    IMPORTANT:
    - Production JSON files are never modified on disk.
    - workflows/sources is reference-only.
    - Runtime inputs are applied to an in-memory deep copy.
    """

    MODES = {
        "ref2v": (
            Path("generation"),
            "H3_Ref2V_Production.json",
        ),
        "turbo_ref2v": (
            Path("generation"),
            "H3_Turbo_Ref2V_Production.json",
        ),
        "upscale": (
            Path("postprocess"),
            "H3_Ref2V_UltimateUpscale_Production.json",
        ),
    }

    LOCKED_DIFFUSION_MODEL = (
        "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors"
    )

    LOCKED_TEXT_ENCODER = (
        "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"
    )

    LOCKED_VIDEO_VAE = (
        "minimax_h3_video_vae_fp16.safetensors"
    )

    LOCKED_AUDIO_VAE = (
        "minimax_h3_audio_vae_fp32.safetensors"
    )

    LOCKED_TURBO_LORA = (
        "minimax_h3_turbo_v4_step600_ema.safetensors"
    )

    LOCKED_UPSCALER = (
        "minimax_h3_latent_upscaler_3d_fp16.safetensors"
    )

    def __init__(
        self,
        project_root: Path,
        comfy_client,
    ):
        self.project_root = Path(project_root)
        self.client = comfy_client

        self.workflow_root = (
            self.project_root
            / "workflows"
        )

        if not self.workflow_root.is_dir():
            raise FileNotFoundError(
                "Workflow directory missing:\n"
                f"{self.workflow_root}"
            )

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def path_for_mode(
        self,
        mode: str,
    ) -> Path:

        try:
            relative_dir, filename = self.MODES[mode]
        except KeyError as error:
            raise ValueError(
                f"Unknown H3 workflow mode: {mode}"
            ) from error

        path = (
            self.workflow_root
            / relative_dir
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(
                "Canonical H3 workflow missing:\n"
                f"{path}"
            )

        return path

    def load(
        self,
        mode: str,
    ) -> dict[str, Any]:

        path = self.path_for_mode(mode)

        try:
            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"Invalid workflow JSON: {path}\n"
                f"{error}"
            ) from error

        if not isinstance(data, dict):
            raise RuntimeError(
                f"Workflow root must be an object: {path}"
            )

        nodes = data.get("nodes")

        if not isinstance(nodes, list) or not nodes:
            raise RuntimeError(
                f"Workflow contains no nodes: {path}"
            )

        return copy.deepcopy(data)

    # ------------------------------------------------------------------
    # Generic graph helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nodes(
        workflow: dict[str, Any],
    ) -> list[dict[str, Any]]:

        nodes = workflow.get("nodes", [])

        if not isinstance(nodes, list):
            raise RuntimeError(
                "Workflow nodes must be a list."
            )

        return [
            node
            for node in nodes
            if isinstance(node, dict)
        ]

    @staticmethod
    def _node_type(
        node: dict[str, Any],
    ) -> str:

        return str(
            node.get("type", "")
        )

    @classmethod
    def _find(
        cls,
        workflow: dict[str, Any],
        node_type: str,
    ) -> list[dict[str, Any]]:

        return [
            node
            for node in cls._nodes(workflow)
            if cls._node_type(node) == node_type
        ]

    @classmethod
    def _find_one(
        cls,
        workflow: dict[str, Any],
        node_type: str,
    ) -> dict[str, Any]:

        nodes = cls._find(
            workflow,
            node_type,
        )

        if len(nodes) != 1:
            raise RuntimeError(
                f"Expected exactly one {node_type}; "
                f"found {len(nodes)}."
            )

        return nodes[0]

    @staticmethod
    def _widgets(
        node: dict[str, Any],
    ) -> list[Any]:

        widgets = node.setdefault(
            "widgets_values",
            []
        )

        if not isinstance(widgets, list):
            raise RuntimeError(
                "widgets_values must be a list for node "
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

        widgets = cls._widgets(node)

        while len(widgets) <= index:
            widgets.append(None)

        widgets[index] = value

    @staticmethod
    def _node_id(node: dict[str, Any]) -> int | None:

        value = node.get("id")

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Input setters
    # ------------------------------------------------------------------

    def _set_prompt(
        self,
        workflow: dict[str, Any],
        prompt: str,
    ) -> None:

        prompt_nodes = [
            node
            for node in self._nodes(workflow)
            if self._node_type(node)
            == "PrimitiveStringMultiline"
        ]

        if not prompt_nodes:
            raise RuntimeError(
                "No PrimitiveStringMultiline prompt node "
                "exists in the H3 workflow."
            )

        labelled = [
            node
            for node in prompt_nodes
            if "prompt"
            in str(node.get("title", "")).lower()
            or "prompt"
            in str(
                node.get("properties", {})
                .get("Node name for S&R", "")
            ).lower()
        ]

        target = labelled[0] if labelled else prompt_nodes[0]

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
                "H3 workflow contains no RandomNoise node."
            )

        for node in noise_nodes:
            self._set_widget(
                node,
                0,
                int(seed),
            )

    @staticmethod
    def _duration_to_frames(
        duration_seconds: float,
    ) -> int:

        duration = max(
            0.25,
            float(duration_seconds),
        )

        raw = max(
            5,
            round(duration * 24),
        )

        # Match the workflow's frame alignment rule.
        aligned = (
            raw
            + (
                5
                - (raw % 17)
            ) % 17
        )

        return max(5, aligned)

    def _set_dimensions(
        self,
        workflow: dict[str, Any],
        width: int | None,
        height: int | None,
    ) -> None:

        if width is None and height is None:
            return

        reference_nodes = self._find(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

        if len(reference_nodes) != 1:
            raise RuntimeError(
                "Expected exactly one "
                "MiniMaxH3ReferenceToVideo node."
            )

        node = reference_nodes[0]
        widgets = self._widgets(node)

        # Current H3 ReferenceToVideo layout:
        # [prompt, width, height, length, ref_image_size]
        if width is not None:
            self._set_widget(
                node,
                1,
                int(width),
            )

        if height is not None:
            self._set_widget(
                node,
                2,
                int(height),
            )

        # Keep widgets_values_named coherent when present.
        named = node.get("widgets_values_named")

        if isinstance(named, dict):
            if width is not None:
                named["width"] = int(width)
            if height is not None:
                named["height"] = int(height)

        # ResolutionSelector is upstream of the reference node.
        # The API conversion uses the actual graph links, so the safest
        # runtime override is the reference-node widget itself.
        _ = widgets

    def _set_duration(
        self,
        workflow: dict[str, Any],
        duration_seconds: float | None,
    ) -> None:

        if duration_seconds is None:
            return

        frames = self._duration_to_frames(
            duration_seconds
        )

        reference_nodes = self._find(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

        if len(reference_nodes) != 1:
            raise RuntimeError(
                "Expected exactly one "
                "MiniMaxH3ReferenceToVideo node."
            )

        node = reference_nodes[0]

        # Current H3 ReferenceToVideo layout:
        # [prompt, width, height, length, ref_image_size]
        self._set_widget(
            node,
            3,
            frames,
        )

        named = node.get("widgets_values_named")

        if isinstance(named, dict):
            named["length"] = frames

    # ------------------------------------------------------------------
    # Reference image injection
    # ------------------------------------------------------------------

    def _reference_loaders(
        self,
        workflow: dict[str, Any],
    ) -> list[dict[str, Any]]:

        reference_nodes = self._find(
            workflow,
            "MiniMaxH3ReferenceToVideo",
        )

        if len(reference_nodes) != 1:
            raise RuntimeError(
                "Expected exactly one "
                "MiniMaxH3ReferenceToVideo node."
            )

        ref_node = reference_nodes[0]
        ref_id = self._node_id(ref_node)

        if ref_id is None:
            raise RuntimeError(
                "MiniMaxH3ReferenceToVideo node has no valid id."
            )

        links = workflow.get("links", [])

        if not isinstance(links, list):
            raise RuntimeError(
                "Workflow links must be a list."
            )

        source_ids: set[int] = set()

        # ComfyUI link format:
        # [link_id, source_node_id, source_slot,
        #  target_node_id, target_slot, type]
        for link in links:
            if not isinstance(link, list) or len(link) < 5:
                continue

            source_id = link[1]
            target_id = link[3]
            target_slot = link[4]

            try:
                source_id = int(source_id)
                target_id = int(target_id)
                target_slot = int(target_slot)
            except (TypeError, ValueError):
                continue

            if target_id != ref_id:
                continue

            # In MiniMaxH3ReferenceToVideo, ref_image_0 starts at
            # target slot 3, followed by ref_image_1, etc.
            if target_slot < 3:
                continue

            if target_slot > 11:
                continue

            source_ids.add(source_id)

        loaders = [
            node
            for node in self._find(
                workflow,
                "LoadImage",
            )
            if self._node_id(node) in source_ids
        ]

        loaders.sort(
            key=lambda node: self._node_id(node) or 0
        )

        return loaders

    def _set_reference_images(
        self,
        workflow: dict[str, Any],
        reference_images: list[str],
    ) -> None:

        references = [
            str(path)
            for path in (
                reference_images or []
            )
            if str(path).strip()
        ][:9]

        if not references:
            return

        loaders = self._reference_loaders(
            workflow
        )

        if len(loaders) < len(references):
            raise RuntimeError(
                "Workflow does not expose enough reference-image "
                "LoadImage nodes.\n"
                f"Needed: {len(references)}\n"
                f"Available: {len(loaders)}"
            )

        for index, source in enumerate(
            references
        ):

            node = loaders[index]
            widgets = self._widgets(node)

            if not widgets:
                raise RuntimeError(
                    f"LoadImage node {node.get('id')} "
                    "has no filename widget."
                )

            self._set_widget(
                node,
                0,
                source,
            )

            self._set_widget(
                node,
                1,
                "image",
            )

            named = node.get(
                "widgets_values_named"
            )

            if isinstance(named, dict):
                named["image"] = source
                named["upload"] = "image"

    # ------------------------------------------------------------------
    # Model validation
    # ------------------------------------------------------------------

    def _validate_locked_models(
        self,
        workflow: dict[str, Any],
        mode: str,
    ) -> None:

        unets = self._find(
            workflow,
            "UNETLoader",
        )

        if len(unets) != 1:
            raise RuntimeError(
                f"{mode}: expected exactly one UNETLoader."
            )

        diffusion = self._widgets(
            unets[0]
        )

        if not diffusion:
            raise RuntimeError(
                f"{mode}: UNETLoader has no model value."
            )

        if diffusion[0] != self.LOCKED_DIFFUSION_MODEL:
            raise RuntimeError(
                f"{mode}: unexpected diffusion model:\n"
                f"{diffusion[0]}"
            )

        clips = self._find(
            workflow,
            "CLIPLoader",
        )

        if len(clips) != 1:
            raise RuntimeError(
                f"{mode}: expected exactly one CLIPLoader."
            )

        clip = self._widgets(
            clips[0]
        )

        if not clip:
            raise RuntimeError(
                f"{mode}: CLIPLoader has no model value."
            )

        if clip[0] != self.LOCKED_TEXT_ENCODER:
            raise RuntimeError(
                f"{mode}: unexpected text encoder:\n"
                f"{clip[0]}"
            )

        vaes = self._find(
            workflow,
            "VAELoader",
        )

        values = [
            self._widgets(node)[0]
            for node in vaes
            if self._widgets(node)
        ]

        if self.LOCKED_VIDEO_VAE not in values:
            raise RuntimeError(
                f"{mode}: missing locked video VAE."
            )

        if self.LOCKED_AUDIO_VAE not in values:
            raise RuntimeError(
                f"{mode}: missing locked audio VAE."
            )

    def _validate_ref2v(
        self,
        workflow: dict[str, Any],
    ) -> None:

        required = [
            "UNETLoader",
            "CLIPLoader",
            "MiniMaxH3ReferenceToVideo",
            "BasicGuider",
            "BasicScheduler",
            "SamplerCustomAdvanced",
            "VAEDecode",
            "VAEDecodeAudio",
            "CreateVideo",
            "SaveVideo",
        ]

        for node_type in required:
            if not self._find(
                workflow,
                node_type,
            ):
                raise RuntimeError(
                    f"Ref2V workflow missing node: "
                    f"{node_type}"
                )

        self._validate_locked_models(
            workflow,
            "ref2v",
        )

    def _validate_turbo(
        self,
        workflow: dict[str, Any],
    ) -> None:

        self._validate_ref2v(
            workflow
        )

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
                "Turbo workflow must contain exactly one "
                "MiniMaxH3TurboLoRA."
            )

        if len(sampler_nodes) != 1:
            raise RuntimeError(
                "Turbo workflow must contain exactly one "
                "MiniMaxH3TurboSampler."
            )

        lora_widgets = self._widgets(
            lora_nodes[0]
        )

        if not lora_widgets:
            raise RuntimeError(
                "Turbo LoRA node has no widgets."
            )

        if (
            lora_widgets[0]
            != self.LOCKED_TURBO_LORA
        ):
            raise RuntimeError(
                "Turbo workflow is not using the locked "
                f"Step600 LoRA:\n{lora_widgets[0]}"
            )

        scheduler = self._find_one(
            workflow,
            "BasicScheduler",
        )

        scheduler_widgets = self._widgets(
            scheduler
        )

        if len(scheduler_widgets) < 3:
            raise RuntimeError(
                "Turbo scheduler is malformed."
            )

        scheduler_widgets[1] = 8
        scheduler_widgets[2] = 1

    def _validate_upscale(
        self,
        workflow: dict[str, Any],
    ) -> None:

        required = [
            "MMH3LatentUpscaleWithModelParams",
            "MMH3TemporalSplitParams",
            "MMH3SpatialSplitParams",
            "MMH3UltimateUpscale",
        ]

        for node_type in required:
            if not self._find(
                workflow,
                node_type,
            ):
                raise RuntimeError(
                    "Upscale workflow missing node: "
                    f"{node_type}"
                )

        param = self._find_one(
            workflow,
            "MMH3LatentUpscaleWithModelParams",
        )

        widgets = self._widgets(
            param
        )

        if (
            not widgets
            or widgets[0]
            != self.LOCKED_UPSCALER
        ):
            raise RuntimeError(
                "Upscale workflow does not use the locked "
                "3D H3 upscaler."
            )

    # ------------------------------------------------------------------
    # Public build API
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        mode: str,
        prompt: str = "",
        seed: int = 0,
        turbo_steps: int = 8,
        reference_images: list[str] | None = None,
        width: int | None = None,
        height: int | None = None,
        duration_seconds: float | None = None,
    ) -> dict[str, Any]:

        workflow = self.load(mode)

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

            self._set_reference_images(
                workflow,
                reference_images or [],
            )

            self._set_dimensions(
                workflow,
                width,
                height,
            )

            self._set_duration(
                workflow,
                duration_seconds,
            )

        elif mode == "turbo_ref2v":

            if int(turbo_steps) != 8:
                raise ValueError(
                    "This production Turbo workflow is locked "
                    "to exactly 8 steps."
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

            self._set_reference_images(
                workflow,
                reference_images or [],
            )

            self._set_dimensions(
                workflow,
                width,
                height,
            )

            self._set_duration(
                workflow,
                duration_seconds,
            )

        elif mode == "upscale":

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

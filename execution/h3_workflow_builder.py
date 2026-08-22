from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


class H3WorkflowBuilder:
    """
    Workflow-driven MiniMax-H3 builder.

    IMPORTANT:
    This class does NOT construct H3 graphs.

    The canonical ComfyUI workflow remains the source of truth.
    This class only:
      1. loads a workflow
      2. converts UI workflow -> API workflow when necessary
      3. patches known H3 inputs
      4. returns the API prompt graph
    """

    WORKFLOW_ROOT = (
        "workflows"
        / Path("MiniMax-H3")
    )

    def __init__(
        self,
        project_root: Path,
        comfy_client=None,
    ):
        self.project_root = Path(project_root)
        self.client = comfy_client

        self.workflow_root = (
            self.project_root
            / "workflows"
            / "MiniMax-H3"
        )

    # ---------------------------------------------------------
    # Workflow selection
    # ---------------------------------------------------------

    def workflow_path(
        self,
        mode: str,
    ) -> Path:

        mapping = {
            "multishot": (
                "canonical"
                / "H3_Multishot_AIO.json"
            ),
            "memory": (
                "canonical"
                / "H3_Multishot_MEMORY.json"
            ),
            "keyframes": (
                "canonical"
                / "H3_Keyframes.json"
            ),
            "hard_mode": (
                "canonical"
                / "H3_HardMode_Chained.json"
            ),
        }

        if mode not in mapping:
            raise ValueError(
                f"Unknown H3 workflow mode: {mode}"
            )

        path = self.workflow_root / mapping[mode]

        if not path.is_file():
            raise FileNotFoundError(
                f"Canonical H3 workflow missing: {path}"
            )

        return path

    # ---------------------------------------------------------
    # Loading
    # ---------------------------------------------------------

    def load(
        self,
        mode: str,
    ) -> dict[str, Any]:

        path = self.workflow_path(mode)

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)

    # ---------------------------------------------------------
    # Generic node helpers
    # ---------------------------------------------------------

    @staticmethod
    def is_api_workflow(
        workflow: dict,
    ) -> bool:

        if not workflow:
            return False

        return all(
            isinstance(value, dict)
            and "class_type" in value
            and "inputs" in value
            for value in workflow.values()
        )

    @staticmethod
    def api_nodes(
        workflow: dict,
    ):
        return workflow.items()

    @staticmethod
    def find_nodes(
        workflow: dict,
        class_type: str,
    ):
        return [
            (
                str(node_id),
                node,
            )
            for node_id, node in workflow.items()
            if (
                isinstance(node, dict)
                and node.get("class_type")
                == class_type
            )
        ]

    @staticmethod
    def first_node(
        workflow: dict,
        class_type: str,
    ):
        nodes = H3WorkflowBuilder.find_nodes(
            workflow,
            class_type,
        )

        if not nodes:
            return None

        return nodes[0]

    @staticmethod
    def set_input(
        node: dict,
        key: str,
        value: Any,
    ):
        node.setdefault(
            "inputs",
            {},
        )[key] = value

    # ---------------------------------------------------------
    # Conversion
    # ---------------------------------------------------------

    def to_api(
        self,
        workflow: dict,
    ) -> dict:

        if self.is_api_workflow(workflow):
            return copy.deepcopy(workflow)

        if self.client is None:
            raise RuntimeError(
                "ComfyClient is required to convert "
                "a UI workflow to API format."
            )

        return self.client.convert_workflow(
            workflow
        )

    # ---------------------------------------------------------
    # H3 generic controls
    # ---------------------------------------------------------

    def patch_common(
        self,
        workflow: dict,
        *,
        width: int,
        height: int,
        frames_per_shot: int,
        steps: int,
        seed: int,
    ):

        samplers = (
            self.find_nodes(
                workflow,
                "H3MultishotMemorySampler",
            )
            + self.find_nodes(
                workflow,
                "H3MultishotSampler",
            )
        )

        for _, node in samplers:

            self.set_input(
                node,
                "width",
                int(width),
            )

            self.set_input(
                node,
                "height",
                int(height),
            )

            self.set_input(
                node,
                "frames_per_shot",
                int(frames_per_shot),
            )

            self.set_input(
                node,
                "steps",
                int(steps),
            )

            self.set_input(
                node,
                "seed",
                int(seed),
            )

    # ---------------------------------------------------------
    # Multishot script
    # ---------------------------------------------------------

    def patch_script(
        self,
        workflow: dict,
        script: str,
        shot_count: int,
    ):

        samplers = (
            self.find_nodes(
                workflow,
                "H3MultishotMemorySampler",
            )
            + self.find_nodes(
                workflow,
                "H3MultishotSampler",
            )
        )

        if not samplers:
            raise RuntimeError(
                "Selected workflow does not contain "
                "an H3 multishot sampler."
            )

        for _, node in samplers:

            self.set_input(
                node,
                "script",
                script,
            )

            self.set_input(
                node,
                "shot_count",
                int(shot_count),
            )

    # ---------------------------------------------------------
    # Reference files
    # ---------------------------------------------------------

    def patch_load_images(
        self,
        workflow: dict,
        filenames: list[str],
    ):

        nodes = self.find_nodes(
            workflow,
            "LoadImage",
        )

        for index, filename in enumerate(
            filenames
        ):

            if index >= len(nodes):
                break

            _, node = nodes[index]

            self.set_input(
                node,
                "image",
                filename,
            )

    def patch_load_audio(
        self,
        workflow: dict,
        filenames: list[str],
    ):

        nodes = self.find_nodes(
            workflow,
            "LoadAudio",
        )

        for index, filename in enumerate(
            filenames
        ):

            if index >= len(nodes):
                break

            _, node = nodes[index]

            self.set_input(
                node,
                "audio",
                filename,
            )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------

    def patch_output_prefix(
        self,
        workflow: dict,
        prefix: str,
    ):

        for class_type in (
            "SaveVideo",
            "VHS_VideoCombine",
        ):

            for _, node in self.find_nodes(
                workflow,
                class_type,
            ):

                if (
                    "filename_prefix"
                    in node.get("inputs", {})
                ):
                    self.set_input(
                        node,
                        "filename_prefix",
                        prefix,
                    )

    # ---------------------------------------------------------
    # Public build
    # ---------------------------------------------------------

    def build(
        self,
        *,
        mode: str,
        script: str,
        shot_count: int,
        width: int,
        height: int,
        frames_per_shot: int,
        steps: int,
        seed: int,
        image_files: list[str] | None = None,
        audio_files: list[str] | None = None,
        output_prefix: str = "h3/output",
    ) -> dict:

        workflow = self.load(mode)

        workflow = self.to_api(
            workflow
        )

        self.patch_common(
            workflow,
            width=width,
            height=height,
            frames_per_shot=frames_per_shot,
            steps=steps,
            seed=seed,
        )

        self.patch_script(
            workflow,
            script=script,
            shot_count=shot_count,
        )

        self.patch_load_images(
            workflow,
            image_files or [],
        )

        self.patch_load_audio(
            workflow,
            audio_files or [],
        )

        self.patch_output_prefix(
            workflow,
            output_prefix,
        )

        return workflow

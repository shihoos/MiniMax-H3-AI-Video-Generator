from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]


PRODUCTION_WORKFLOWS = [
    ROOT
    / "workflows"
    / "MiniMax-H3"
    / "generation"
    / "H3_Ref2V_Production.json",

    ROOT
    / "workflows"
    / "MiniMax-H3"
    / "generation"
    / "H3_Turbo_Ref2V_Production.json",

    ROOT
    / "workflows"
    / "MiniMax-H3"
    / "postprocess"
    / "H3_Ref2V_UltimateUpscale_Production.json",
]


REQUIRED_FILES = [
    ROOT / "configs" / "model_inventory.yaml",
    ROOT / "configs" / "custom_nodes.yaml",
    ROOT / "kaggle" / "bootstrap.py",
    ROOT / "kaggle" / "preflight_h3.py",
    ROOT / "execution" / "h3_workflow_builder.py",
    ROOT / "execution" / "shot_executor.py",
    ROOT / "execution" / "production_runner.py",
    ROOT / "pipeline" / "h3_scene_continuity.py",
    ROOT / "pipeline" / "continuity_manager.py",
]


FORBIDDEN_EXECUTABLE_MODEL_TOKENS = {
    "minimax_h3_fl2va",
    "minimax_h3_fl2v",
    "qwen3-4b",
    "qwen3vl_32b_minimax_h3-Q4",
    "Q4_K_M",
    "minimax_h3_video_vae_int8_convrot",
    "minimax_h3_ref2va_pruned_int8_convrot",
}


def validate_python():
    for path in ROOT.rglob("*.py"):

        if ".git" in path.parts:
            continue

        if "ComfyUI" in path.parts:
            continue

        if "__pycache__" in path.parts:
            continue

        ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
            filename=str(path),
        )


def validate_required_files():

    for path in REQUIRED_FILES:

        if not path.is_file():
            raise RuntimeError(
                f"Missing required project file: {path}"
            )


def load_workflow(path: Path) -> dict:

    if not path.is_file():
        raise RuntimeError(
            f"Missing production workflow: {path}"
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
            f"Workflow is not a JSON object: {path}"
        )

    if not data.get("nodes"):
        raise RuntimeError(
            f"Workflow contains no nodes: {path}"
        )

    return data


def executable_model_values(
    workflow: dict,
) -> list[str]:

    values = []

    executable_types = {
        "UNETLoader",
        "CLIPLoader",
        "CLIPLoaderGGUF",
        "VAELoader",
        "LoraLoaderModelOnly",
        "MMH3LatentUpscaleWithModelParams",
    }

    for node in workflow.get(
        "nodes",
        []
    ):

        if node.get("type") not in executable_types:
            continue

        for value in node.get(
            "widgets_values",
            []
        ):

            if isinstance(
                value,
                str,
            ):
                values.append(
                    value.lower()
                )

    return values


def validate_workflows():

    workflows = [
        load_workflow(path)
        for path in PRODUCTION_WORKFLOWS
    ]

    # Ref2V
    ref2v = workflows[0]

    ref2v_types = {
        node.get("type")
        for node in ref2v["nodes"]
    }

    for required in {
        "UNETLoader",
        "CLIPLoader",
        "MiniMaxH3ReferenceToVideo",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    }:

        if required not in ref2v_types:
            raise RuntimeError(
                f"Ref2V production workflow missing node: "
                f"{required}"
            )

    # Turbo
    turbo = workflows[1]

    turbo_types = {
        node.get("type")
        for node in turbo["nodes"]
    }

    if "MiniMaxH3TurboLoRA" not in turbo_types:
        raise RuntimeError(
            "Turbo production workflow is missing "
            "MiniMaxH3TurboLoRA."
        )

    if "MiniMaxH3TurboSampler" not in turbo_types:
        raise RuntimeError(
            "Turbo production workflow is missing "
            "MiniMaxH3TurboSampler."
        )

    # Upscale
    upscale = workflows[2]

    upscale_types = {
        node.get("type")
        for node in upscale["nodes"]
    }

    if "MMH3UltimateUpscale" not in upscale_types:
        raise RuntimeError(
            "Ultimate Upscale production workflow missing "
            "MMH3UltimateUpscale."
        )

    if (
        "MMH3LatentUpscaleWithModelParams"
        not in upscale_types
    ):
        raise RuntimeError(
            "Ultimate Upscale production workflow missing "
            "H3 3D parameter node."
        )

    # No legacy executable assets.
    for index, workflow in enumerate(
        workflows
    ):

        values = executable_model_values(
            workflow
        )

        for token in (
            FORBIDDEN_EXECUTABLE_MODEL_TOKENS
        ):

            if any(
                token.lower() in value
                for value in values
            ):
                raise RuntimeError(
                    f"Legacy executable model "
                    f"reference '{token}' found in "
                    f"production workflow #{index + 1}"
                )


def main():

    validate_python()
    validate_required_files()
    validate_workflows()

    print(
        "MiniMax H3 project validation PASSED."
    )


if __name__ == "__main__":
    main()

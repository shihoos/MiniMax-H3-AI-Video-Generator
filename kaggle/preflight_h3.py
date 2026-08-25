from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

COMFY = ROOT / "ComfyUI"

MODEL_MANIFEST = (
    ROOT
    / "configs"
    / "model_inventory.yaml"
)

NODE_MANIFEST = (
    ROOT
    / "configs"
    / "custom_nodes.yaml"
)

WORKFLOW_ROOT = (
    ROOT
    / "workflows"
    / "MiniMax-H3"
)


def load_yaml(path: Path):
    return yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )


def require_file(path: Path):
    if not path.is_file():
        raise RuntimeError(
            f"Missing required file: {path}"
        )


def check_models():

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    for model in (
        manifest["models"].values()
    ):

        path = (
            COMFY
            / "models"
            / model["directory"]
            / model["filename"]
        )

        require_file(path)


def check_workflows():

    required = [
        WORKFLOW_ROOT
        / "generation"
        / "H3_Ref2V_Production.json",

        WORKFLOW_ROOT
        / "generation"
        / "H3_Turbo_Ref2V_Production.json",

        WORKFLOW_ROOT
        / "postprocess"
        / "MMH3_Ultimate_Upscale.json",
    ]

    for path in required:

        require_file(path)

        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )


def check_no_legacy_models():

    forbidden = [
        "Q4_K_M",
        "_fl2va_",
        "qwen3-4b",
        "qwen3vl_32b_minimax_h3-Q4",
    ]

    for directory in [
        COMFY / "models",
        ROOT / "planner",
        ROOT / "execution",
        ROOT / "kaggle",
        ROOT / "workflows",
    ]:

        if not directory.exists():
            continue

        for path in directory.rglob("*"):

            if not path.is_file():
                continue

            try:
                text = path.read_text(
                    encoding="utf-8"
                )
            except UnicodeDecodeError:
                continue

            for value in forbidden:

                if value in text:
                    raise RuntimeError(
                        f"Legacy model reference "
                        f"'{value}' found in {path}"
                    )


def check_custom_nodes():

    manifest = load_yaml(
        NODE_MANIFEST
    )

    custom_nodes = (
        manifest["custom_nodes"]["required"]
        + manifest["custom_nodes"]["supporting"]
    )

    for node in custom_nodes:

        path = (
            COMFY
            / "custom_nodes"
            / node["name"]
        )

        require_file(
            path / "__init__.py"
        )


def main():

    check_models()
    check_workflows()
    check_custom_nodes()
    check_no_legacy_models()

    print(
        "MiniMax H3 preflight PASSED."
    )


if __name__ == "__main__":
    main()

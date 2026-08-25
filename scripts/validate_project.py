from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LEGACY_TOKENS = {
    "Q4_K_M",
    "qwen3-4b",
    "minimax_h3_fl2va",
    "minimax_h3_ref2va_pruned-Q4",
    "H3_Ref2VA_Memory_API",
    "H3_HardMode_Chained",
    "H3_Seamless_Chain_CORE",
    "H3_Seamless_Chain_v2",
}


REQUIRED_FILES = [
    ROOT
    / "configs"
    / "model_inventory.yaml",

    ROOT
    / "configs"
    / "custom_nodes.yaml",

    ROOT
    / "kaggle"
    / "bootstrap.py",

    ROOT
    / "kaggle"
    / "preflight_h3.py",

    ROOT
    / "execution"
    / "h3_workflow_builder.py",

    ROOT
    / "execution"
    / "shot_executor.py",

    ROOT
    / "execution"
    / "production_runner.py",

    ROOT
    / "pipeline"
    / "h3_scene_continuity.py",

    ROOT
    / "pipeline"
    / "continuity_manager.py",
]


WORKFLOW_FILES = [
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
    / "MMH3_Ultimate_Upscale.json",
]


def validate_python():

    for path in ROOT.rglob(
        "*.py"
    ):

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
                f"Missing required project file: "
                f"{path}"
            )


def validate_workflows():

    for path in WORKFLOW_FILES:

        if not path.is_file():
            raise RuntimeError(
                f"Missing required workflow: "
                f"{path}"
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
                f"Workflow is not an object: "
                f"{path}"
            )

        if not data.get(
            "nodes"
        ):
            raise RuntimeError(
                f"Workflow contains no nodes: "
                f"{path}"
            )


def validate_no_legacy_text():

    for directory in [
        ROOT
        / "planner",

        ROOT
        / "pipeline",

        ROOT
        / "execution",

        ROOT
        / "scheduler",

        ROOT
        / "kaggle",

        ROOT
        / "workflows",
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

            for token in LEGACY_TOKENS:

                if token in text:
                    raise RuntimeError(
                        f"Legacy token '{token}' "
                        f"found in {path}"
                    )


def main():

    validate_python()
    validate_required_files()
    validate_workflows()
    validate_no_legacy_text()

    print(
        "MiniMax H3 project validation PASSED."
    )


if __name__ == "__main__":
    main()

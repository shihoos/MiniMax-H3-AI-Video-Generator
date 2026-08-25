from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(
    __file__
).resolve().parents[1]

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


PRODUCTION_WORKFLOWS = [
    WORKFLOW_ROOT
    / "generation"
    / "H3_Ref2V_Production.json",

    WORKFLOW_ROOT
    / "generation"
    / "H3_Turbo_Ref2V_Production.json",

    WORKFLOW_ROOT
    / "postprocess"
    / "H3_Ref2V_UltimateUpscale_Production.json",
]


def load_yaml(path: Path):

    if not path.is_file():
        raise RuntimeError(
            f"Missing manifest: {path}"
        )

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

        print(
            "MODEL OK:",
            model["filename"]
        )


def load_workflow(path: Path):

    require_file(path)

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid workflow JSON: {path}"
        ) from exc


def check_workflows():

    ref2v = load_workflow(
        PRODUCTION_WORKFLOWS[0]
    )

    turbo = load_workflow(
        PRODUCTION_WORKFLOWS[1]
    )

    upscale = load_workflow(
        PRODUCTION_WORKFLOWS[2]
    )

    ref2v_types = {
        node.get("type")
        for node in ref2v.get(
            "nodes",
            []
        )
    }

    turbo_types = {
        node.get("type")
        for node in turbo.get(
            "nodes",
            []
        )
    }

    upscale_types = {
        node.get("type")
        for node in upscale.get(
            "nodes",
            []
        )
    }

    required_ref2v = {
        "UNETLoader",
        "CLIPLoader",
        "MiniMaxH3ReferenceToVideo",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    }

    missing = (
        required_ref2v
        - ref2v_types
    )

    if missing:
        raise RuntimeError(
            "Ref2V missing nodes:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    if (
        "MiniMaxH3TurboLoRA"
        not in turbo_types
    ):
        raise RuntimeError(
            "Turbo workflow missing MiniMaxH3TurboLoRA."
        )

    if (
        "MiniMaxH3TurboSampler"
        not in turbo_types
    ):
        raise RuntimeError(
            "Turbo workflow missing MiniMaxH3TurboSampler."
        )

    if (
        "MMH3UltimateUpscale"
        not in upscale_types
    ):
        raise RuntimeError(
            "Upscale workflow missing MMH3UltimateUpscale."
        )

    if (
        "MMH3LatentUpscaleWithModelParams"
        not in upscale_types
    ):
        raise RuntimeError(
            "Upscale workflow missing H3 3D "
            "upscale parameter node."
        )

    print(
        "WORKFLOWS OK"
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

        if not path.is_dir():
            raise RuntimeError(
                f"Missing custom node: {path}"
            )

        print(
            "CUSTOM NODE OK:",
            node["name"],
        )


def check_no_legacy_executable_models():

    forbidden = [
        "Q4_K_M",
        "qwen3-4b",
        "minimax_h3_fl2va",
        "minimax_h3_fl2v",
        "minimax_h3_video_vae_int8_convrot",
        "minimax_h3_ref2va_pruned_int8_convrot",
    ]

    executable_types = {
        "UNETLoader",
        "CLIPLoader",
        "CLIPLoaderGGUF",
        "VAELoader",
        "LoraLoaderModelOnly",
        "MMH3LatentUpscaleWithModelParams",
    }

    for workflow_path in PRODUCTION_WORKFLOWS:

        data = load_workflow(
            workflow_path
        )

        for node in data.get(
            "nodes",
            []
        ):

            if node.get(
                "type"
            ) not in executable_types:
                continue

            widgets = node.get(
                "widgets_values",
                []
            )

            for value in widgets:

                if not isinstance(
                    value,
                    str,
                ):
                    continue

                value_lower = value.lower()

                for token in forbidden:

                    if token.lower() in value_lower:
                        raise RuntimeError(
                            f"Legacy executable model "
                            f"'{token}' found in "
                            f"{workflow_path}"
                        )

    print(
        "LEGACY EXECUTABLE MODEL SCAN: PASS"
    )


def main():

    check_models()
    check_workflows()
    check_custom_nodes()
    check_no_legacy_executable_models()

    print(
        "MiniMax H3 preflight PASSED."
    )


if __name__ == "__main__":
    main()

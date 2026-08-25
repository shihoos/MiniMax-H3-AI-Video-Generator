from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

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

PRODUCTION_WORKFLOWS = [
    ROOT
    / "workflows"
    / "generation"
    / "H3_Ref2V_Production.json",

    ROOT
    / "workflows"
    / "generation"
    / "H3_Turbo_Ref2V_Production.json",

    ROOT
    / "workflows"
    / "postprocess"
    / "H3_Ref2V_UltimateUpscale_Production.json",
]

SOURCE_WORKFLOWS = [
    ROOT
    / "workflows"
    / "sources"
    / "H3_Turbo_Reference_Source.json",

    ROOT
    / "workflows"
    / "sources"
    / "H3_LatentUpscaler_Source.json",
]


def load_yaml(
    path: Path,
):
    if not path.is_file():
        raise RuntimeError(
            f"Missing manifest: {path}"
        )

    return yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_json(
    path: Path,
):
    if not path.is_file():
        raise RuntimeError(
            f"Missing workflow: {path}"
        )

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data.get("nodes"),
        list,
    ):
        raise RuntimeError(
            f"Invalid workflow: {path}"
        )

    return data


def check_models():
    manifest = load_yaml(
        MODEL_MANIFEST
    )

    for model in manifest[
        "models"
    ].values():

        path = (
            COMFY
            / "models"
            / model["directory"]
            / model["filename"]
        )

        if not path.is_file():
            raise RuntimeError(
                f"Missing locked model: {path}"
            )

        print(
            "MODEL OK:",
            model["filename"],
        )


def check_workflows():

    for path in (
        PRODUCTION_WORKFLOWS
        + SOURCE_WORKFLOWS
    ):
        load_json(path)

        print(
            "WORKFLOW OK:",
            path.relative_to(ROOT),
        )

    ref2v = load_json(
        PRODUCTION_WORKFLOWS[0]
    )

    turbo = load_json(
        PRODUCTION_WORKFLOWS[1]
    )

    upscale = load_json(
        PRODUCTION_WORKFLOWS[2]
    )

    ref_types = {
        n.get("type")
        for n in ref2v["nodes"]
    }

    turbo_types = {
        n.get("type")
        for n in turbo["nodes"]
    }

    upscale_types = {
        n.get("type")
        for n in upscale["nodes"]
    }

    for node_type in {
        "UNETLoader",
        "CLIPLoader",
        "MiniMaxH3ReferenceToVideo",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    }:
        if node_type not in ref_types:
            raise RuntimeError(
                f"Ref2V missing {node_type}"
            )

    for node_type in {
        "MiniMaxH3TurboLoRA",
        "MiniMaxH3TurboSampler",
    }:
        if node_type not in turbo_types:
            raise RuntimeError(
                f"Turbo missing {node_type}"
            )

    for node_type in {
        "MMH3LatentUpscaleWithModelParams",
        "MMH3TemporalSplitParams",
        "MMH3SpatialSplitParams",
        "MMH3UltimateUpscale",
    }:
        if node_type not in upscale_types:
            raise RuntimeError(
                f"Upscale missing {node_type}"
            )


def check_custom_nodes():

    manifest = load_yaml(
        NODE_MANIFEST
    )

    nodes = (
        manifest["custom_nodes"]["required"]
        + manifest["custom_nodes"]["supporting"]
    )

    for node in nodes:

        path = (
            COMFY
            / "custom_nodes"
            / node["name"]
        )

        if not path.is_dir():
            raise RuntimeError(
                f"Missing custom node: {path}"
            )


def check_no_legacy():

    forbidden = {
        "Q4_K_M",
        "qwen3-4b",
        "minimax_h3_fl2va",
        "minimax_h3_fl2v",
        "minimax_h3_ref2va_pruned_int8_convrot",
        "minimax_h3_video_vae_int8_convrot",
    }

    executable_types = {
        "UNETLoader",
        "CLIPLoader",
        "CLIPLoaderGGUF",
        "VAELoader",
        "LoraLoaderModelOnly",
        "MMH3LatentUpscaleWithModelParams",
    }

    for path in PRODUCTION_WORKFLOWS:

        graph = load_json(path)

        for node in graph["nodes"]:

            if node.get("type") not in executable_types:
                continue

            for value in node.get(
                "widgets_values",
                [],
            ):

                if not isinstance(
                    value,
                    str,
                ):
                    continue

                lowered = value.lower()

                for token in forbidden:
                    if token.lower() in lowered:
                        raise RuntimeError(
                            f"Legacy executable asset "
                            f"{token} in {path}"
                        )


def main():

    check_models()
    check_workflows()
    check_custom_nodes()
    check_no_legacy()

    print(
        "MiniMax H3 preflight PASSED."
    )


if __name__ == "__main__":
    main()

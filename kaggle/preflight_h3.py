from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

COMFY = ROOT / "ComfyUI"

WORKFLOW_ROOT = (
    ROOT
    / "workflows"
    / "MiniMax-H3"
)


BASE_WORKFLOWS = [
    WORKFLOW_ROOT
    / "base"
    / "H3_Extend_Take.json",

    WORKFLOW_ROOT
    / "base"
    / "H3_HardMode_Chained.json",

    WORKFLOW_ROOT
    / "base"
    / "H3_HardMode_R2V.json",

    WORKFLOW_ROOT
    / "base"
    / "H3_Keyframes.json",

    WORKFLOW_ROOT
    / "base"
    / "H3_Seamless_Chain_CORE.json",

    WORKFLOW_ROOT
    / "base"
    / "H3_Seamless_Chain_v2.json",
]


TURBO_WORKFLOWS = [
    WORKFLOW_ROOT
    / "turbo"
    / "H3_Turbo_I2V.json",

    WORKFLOW_ROOT
    / "turbo"
    / "H3_Turbo_Ref2V.json",

    WORKFLOW_ROOT
    / "turbo"
    / "H3_Turbo_T2V.json",
]


def require_file(
    path,
):

    if not (
        path.is_file()
        or path.is_symlink()
    ):

        raise RuntimeError(
            f"Missing required file: {path}"
        )


def check_models():

    for path in (
        COMFY
        / "models"
        / "diffusion_models"
        / "minimax_h3_ref2va_pruned-Q4_K_M.gguf",

        COMFY
        / "models"
        / "diffusion_models"
        / "minimax_h3_fl2va_pruned-Q4_K_M.gguf",

        COMFY
        / "models"
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",

        COMFY
        / "models"
        / "vae"
        / "minimax_h3_video_vae_fp16.safetensors",

        COMFY
        / "models"
        / "vae"
        / "minimax_h3_audio_vae_fp32.safetensors",
    ):
        require_file(path)


def check_workflows():

    manual = (
        WORKFLOW_ROOT
        / "base"
        / "H3_Ref2VA_Memory_API.json"
    )

    if manual.exists():
        raise RuntimeError(
            "Manual H3_Ref2VA_Memory_API.json still exists. "
            "Delete it; it is not a production workflow."
        )

    for path in BASE_WORKFLOWS:
        require_file(path)

        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    if (
        os.getenv(
            "H3_ENABLE_TURBO",
            "0",
        )
        == "1"
    ):

        for path in TURBO_WORKFLOWS:
            require_file(path)

            json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )


def check_nodes():

    sys.path.insert(
        0,
        str(COMFY),
    )

    import nodes  # noqa: F401

    import comfy_extras.nodes_minimax_h3  # noqa: F401

    from execution.comfy_client import (
        ComfyClient,
    )

    client = ComfyClient(
        "http://127.0.0.1:8188"
    )

    if not client.health_check():
        raise RuntimeError(
            "ComfyUI worker on 8188 is not running."
        )

    info = client.get_object_info()

    required = {
        "H3ModelLoaderAny",
        "H3ClipLoaderAny",
        "MiniMaxH3ReferenceToVideo",
        "H3FreeTextEncoder",
        "H3MultishotSampler",
        "H3MultishotMemorySampler",
        "H3LastFrame",
        "H3ConcatAV",
    }

    missing = sorted(
        required
        - set(info)
    )

    if missing:
        raise RuntimeError(
            "Missing H3 nodes: "
            + ", ".join(missing)
        )


def main():

    check_models()
    check_workflows()

    print(
        "Static H3 preflight passed."
    )


if __name__ == "__main__":
    main()

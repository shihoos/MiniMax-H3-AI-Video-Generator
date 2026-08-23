from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

COMFY_MODELS = (
    ROOT
    / "ComfyUI"
    / "models"
)


def download(
    repo_id,
    filename,
    local_dir,
):

    local_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Downloading {repo_id}:{filename}"
    )

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(
            local_dir
        ),
        local_dir_use_symlinks=False,
    )


def install_q4_fl2va():

    download(
        repo_id="leejet/MiniMax-H3-GGUF",
        filename=(
            "minimax_h3_fl2va_pruned-Q4_K_M.gguf"
        ),
        local_dir=(
            COMFY_MODELS
            / "diffusion_models"
        ),
    )


def install_turbo():

    # Official/current Comfy-Org H3 model family.
    download(
        repo_id="Comfy-Org/MiniMax-H3",
        filename=(
            "diffusion_models/"
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        ),
        local_dir=(
            COMFY_MODELS
            / "diffusion_models"
        ),
    )

    download(
        repo_id="Comfy-Org/MiniMax-H3",
        filename=(
            "diffusion_models/"
            "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        ),
        local_dir=(
            COMFY_MODELS
            / "diffusion_models"
        ),
    )

    download(
        repo_id="Comfy-Org/MiniMax-H3",
        filename=(
            "text_encoders/"
            "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
        ),
        local_dir=(
            COMFY_MODELS
            / "text_encoders"
        ),
    )

    download(
        repo_id="Comfy-Org/MiniMax-H3",
        filename=(
            "loras/"
            "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
        ),
        local_dir=(
            COMFY_MODELS
            / "loras"
        ),
    )

    download(
        repo_id="Comfy-Org/MiniMax-H3",
        filename=(
            "loras/"
            "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
        ),
        local_dir=(
            COMFY_MODELS
            / "loras"
        ),
    )


def main():

    install_q4_fl2va()

    if (
        os.getenv(
            "H3_ENABLE_TURBO",
            "0",
        )
        == "1"
    ):
        install_turbo()

    print(
        "Extra H3 assets installed."
    )


if __name__ == "__main__":
    main()

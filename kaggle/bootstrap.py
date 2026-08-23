from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

COMFY = ROOT / "ComfyUI"
CUSTOM = COMFY / "custom_nodes"


def run(
    *args,
):

    print(
        "+",
        " ".join(
            str(arg)
            for arg in args
        ),
    )

    subprocess.run(
        [str(arg) for arg in args],
        check=True,
    )


def clone(
    url,
    destination,
):

    destination = Path(
        destination
    )

    if destination.exists():
        print(
            "EXISTS:",
            destination,
        )
        return

    run(
        "git",
        "clone",
        "--depth",
        "1",
        url,
        destination,
    )


def locate(
    root,
    filename,
):

    matches = [
        path
        for path in Path(root).rglob(
            filename
        )
        if path.is_file()
    ]

    if not matches:
        return None

    return matches[0]


def locate_case_insensitive(
    root,
    filename,
):

    target = filename.lower()

    for path in Path(root).rglob("*"):

        if (
            path.is_file()
            and path.name.lower()
            == target
        ):
            return path

    return None


def locate_dataset():

    root = Path(
        "/kaggle/input"
    )

    if not root.is_dir():
        raise FileNotFoundError(
            "/kaggle/input does not exist."
        )

    diffusion = locate_case_insensitive(
        root,
        "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
    )

    if diffusion is None:
        raise FileNotFoundError(
            "Q4 Ref2VA diffusion model not found."
        )

    dataset_root = (
        diffusion
        .parents[2]
    )

    required = {
        "qwen":
            locate_case_insensitive(
                dataset_root,
                "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
            ),

        "mmproj":
            locate_case_insensitive(
                dataset_root,
                "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",
            ),

        "video_vae":
            locate_case_insensitive(
                dataset_root,
                "minimax_h3_video_vae_fp16.safetensors",
            ),

        "audio_vae":
            locate_case_insensitive(
                dataset_root,
                "minimax_h3_audio_vae_fp32.safetensors",
            ),
    }

    missing = [
        name
        for name, path
        in required.items()
        if path is None
    ]

    if missing:
        raise FileNotFoundError(
            "Missing H3 Q4 dataset files: "
            + ", ".join(missing)
        )

    return {
        "root": dataset_root,
        "diffusion": diffusion,
        **required,
    }


def link(
    source,
    destination,
):

    source = Path(source)
    destination = Path(
        destination
    )

    if not source.is_file():
        raise FileNotFoundError(
            source
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        destination.exists()
        or destination.is_symlink()
    ):
        destination.unlink()

    try:

        destination.symlink_to(
            source
        )

    except OSError:

        shutil.copy2(
            source,
            destination,
        )


def install_comfy():

    if not (
        COMFY / "main.py"
    ).is_file():

        clone(
            "https://github.com/comfyanonymous/ComfyUI.git",
            COMFY,
        )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        COMFY / "requirements.txt",
    )


def install_nodes():

    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    clone(
        "https://github.com/city96/ComfyUI-GGUF.git",
        CUSTOM / "ComfyUI-GGUF",
    )

    clone(
        "https://github.com/jlucasmcrell/ComfyUI-H3-Multishot.git",
        CUSTOM / "ComfyUI-H3-Multishot",
    )

    clone(
        "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git",
        CUSTOM / "ComfyUI-VideoHelperSuite",
    )

    clone(
        "https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint.git",
        CUSTOM
        / "comfyui-workflow-to-api-converter-endpoint",
    )

    patcher = (
        CUSTOM
        / "ComfyUI-H3-Multishot"
        / "apply_gguf_arch_patch.py"
    )

    if patcher.is_file():

        run(
            sys.executable,
            patcher,
        )


def install_base_models():

    dataset = locate_dataset()

    models = (
        COMFY
        / "models"
    )

    link(
        dataset["diffusion"],
        models
        / "diffusion_models"
        / "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
    )

    link(
        dataset["qwen"],
        models
        / "text_encoders"
        / "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    )

    link(
        dataset["mmproj"],
        models
        / "text_encoders"
        / "Qwen3-VL-32B-Instruct-MiniMax-H3-L0-49-mmproj-BF16.gguf",
    )

    link(
        dataset["video_vae"],
        models
        / "vae"
        / "minimax_h3_video_vae_fp16.safetensors",
    )

    link(
        dataset["audio_vae"],
        models
        / "vae"
        / "minimax_h3_audio_vae_fp32.safetensors",
    )


def main():

    install_comfy()
    install_nodes()
    install_base_models()

    print()
    print(
        "=" * 72
    )
    print(
        "MiniMax H3 Q4 base bootstrap complete."
    )
    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()

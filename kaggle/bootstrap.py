from __future__ import annotations

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
MODELS = COMFY / "models"

KAGGLE_INPUT = Path(
    "/kaggle/input"
)


NODE_REPOSITORIES = {
    "Comfyui_Minimax_h3_latent_Upscaler": {
        "url": (
            "https://github.com/"
            "LBH-123-AI/"
            "Comfyui_Minimax_h3_latent_Upscaler.git"
        ),
        "revision": (
            "6a4b191e8af583b7c097f564690325f91d18c2e2"
        ),
    },
    "Comfyui-MMH3-UltimateUpscale": {
        "url": (
            "https://github.com/"
            "bbaudio-2025/"
            "Comfyui-MMH3-UltimateUpscale.git"
        ),
        "revision": (
            "2553ad1b66ee0956df02e9146dd78b93395f9f69"
        ),
    },
    "ComfyUI-VideoHelperSuite": {
        "url": (
            "https://github.com/"
            "Kosinkadink/"
            "ComfyUI-VideoHelperSuite.git"
        ),
        "revision": (
            "4ee72c065db22c9d96c2427954dc69e7b908444b"
        ),
    },
}


MODEL_FILES = {
    "diffusion_model": (
        "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors",
        MODELS / "diffusion_models",
    ),
    "text_encoder": (
        "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
        MODELS / "text_encoders",
    ),
    "turbo_lora": (
        "minimax_h3_turbo_v4_step600_ema.safetensors",
        MODELS / "loras",
    ),
    "video_vae": (
        "minimax_h3_video_vae_fp16.safetensors",
        MODELS / "vae",
    ),
    "audio_vae": (
        "minimax_h3_audio_vae_fp32.safetensors",
        MODELS / "vae",
    ),
    "latent_upscaler": (
        "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        MODELS / "latent_upscale_models",
    ),
}


def run(
    *args: str | Path,
) -> None:

    print(
        "+",
        " ".join(
            str(value)
            for value in args
        ),
    )

    subprocess.run(
        [str(value) for value in args],
        check=True,
    )


def git_checkout(
    url: str,
    destination: Path,
    revision: str,
) -> None:

    if not destination.exists():

        run(
            "git",
            "clone",
            url,
            destination,
        )

    run(
        "git",
        "-C",
        destination,
        "fetch",
        "--all",
        "--tags",
        "--prune",
    )

    run(
        "git",
        "-C",
        destination,
        "checkout",
        "--detach",
        revision,
    )


def find_file(
    filename: str,
) -> Path:

    if not KAGGLE_INPUT.is_dir():
        raise RuntimeError(
            "/kaggle/input does not exist."
        )

    matches = []

    for path in KAGGLE_INPUT.rglob("*"):
        if (
            path.is_file()
            and path.name.lower()
            == filename.lower()
        ):
            matches.append(path)

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Locked model not found: {filename}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple copies of locked model found: "
            f"{filename}\n"
            + "\n".join(
                str(path)
                for path in matches
            )
        )

    return matches[0]


def link_model(
    source: Path,
    destination: Path,
) -> None:

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


def install_comfy() -> None:

    if not (
        COMFY / "main.py"
    ).is_file():

        raise RuntimeError(
            "ComfyUI is not installed. "
            "Install/pin the required ComfyUI revision first."
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


def install_nodes() -> None:

    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, config in (
        NODE_REPOSITORIES.items()
    ):

        destination = (
            CUSTOM / name
        )

        git_checkout(
            url=config["url"],
            destination=destination,
            revision=config["revision"],
        )


def install_models() -> None:

    for name, (
        filename,
        destination_dir,
    ) in MODEL_FILES.items():

        source = find_file(
            filename
        )

        destination = (
            destination_dir
            / filename
        )

        link_model(
            source,
            destination,
        )

        print(
            f"[MODEL] {name}: {destination}"
        )


def verify_locked_inventory() -> None:

    expected = {
        filename.lower()
        for filename, _directory
        in MODEL_FILES.values()
    }

    found = set()

    for directory in (
        MODELS / "diffusion_models",
        MODELS / "text_encoders",
        MODELS / "loras",
        MODELS / "vae",
        MODELS / "latent_upscale_models",
    ):

        if not directory.is_dir():
            continue

        for path in directory.iterdir():

            if path.is_file():
                found.add(
                    path.name.lower()
                )

    unexpected = sorted(
        found - expected
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected production model files "
            "found in ComfyUI/models:\n"
            + "\n".join(
                unexpected
            )
        )


def main() -> None:

    install_comfy()
    install_nodes()
    install_models()
    verify_locked_inventory()

    print(
        "MiniMax H3 production bootstrap PASSED."
    )


if __name__ == "__main__":
    main()

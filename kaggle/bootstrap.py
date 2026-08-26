from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

COMFY = ROOT / "ComfyUI"
CUSTOM = COMFY / "custom_nodes"
MODELS = COMFY / "models"

KAGGLE_INPUT = Path("/kaggle/input")

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


# ============================================================
# QWEN DIRECTOR RUNTIME
# ============================================================

DIRECTOR_MODEL_FILENAME = (
    "Qwen3-14B-Q4_K_M.gguf"
)

# Qwen3-14B Q4_K_M is used by llama.cpp/llama-cpp-python,
# not as a ComfyUI production model.
#
# Keep this separate from ComfyUI/models.

LLAMA_CPP_PACKAGE = (
    "llama-cpp-python"
)

# CUDA 13 wheel index. The runtime is used only during
# planning and is completely unloaded before H3 generation.
LLAMA_CPP_CUDA_INDEX = (
    "https://abetlen.github.io/"
    "llama-cpp-python/whl/cu130"
)


# ============================================================
# GENERIC HELPERS
# ============================================================

def run(*args: str | Path) -> None:
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


def load_yaml(
    path: Path,
) -> dict:
    return yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# GIT CHECKOUT
# ============================================================

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


# ============================================================
# MODEL DISCOVERY
# ============================================================

def find_kaggle_file(
    filename: str,
) -> Path:

    matches = []

    for path in KAGGLE_INPUT.rglob("*"):

        if (
            path.is_file()
            and path.name.lower()
            == filename.lower()
        ):
            matches.append(path)

    if not matches:
        raise FileNotFoundError(
            f"Required Kaggle asset not found: {filename}\n"
            "Attach the required Kaggle dataset before "
            "starting production."
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple copies of asset found: {filename}\n"
            + "\n".join(
                str(path)
                for path in matches
            )
        )

    return matches[0]


def find_model(
    filename: str,
) -> Path:

    return find_kaggle_file(
        filename
    )


# ============================================================
# COMFY MODEL LINKING
# ============================================================

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


# ============================================================
# DIRECTOR RUNTIME
# ============================================================

def install_director_runtime() -> None:

    print(
        "=" * 80
    )
    print(
        "INSTALLING LOCAL QWEN DIRECTOR RUNTIME"
    )
    print(
        "=" * 80
    )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "--only-binary=:all:",
        LLAMA_CPP_PACKAGE,
        "--extra-index-url",
        LLAMA_CPP_CUDA_INDEX,
    )

    # Import immediately so a broken installation fails during
    # bootstrap rather than halfway through production.
    run(
        sys.executable,
        "-c",
        (
            "from llama_cpp import Llama; "
            "print('llama-cpp-python import: PASS')"
        ),
    )


def find_director_model() -> Path:

    model = find_kaggle_file(
        DIRECTOR_MODEL_FILENAME
    )

    if (
        model.name.lower()
        != DIRECTOR_MODEL_FILENAME.lower()
    ):
        raise RuntimeError(
            "Unexpected Qwen director filename: "
            f"{model}"
        )

    if model.stat().st_size <= 0:
        raise RuntimeError(
            f"Qwen director model is empty: {model}"
        )

    size_gib = (
        model.stat().st_size
        / (1024 ** 3)
    )

    print(
        f"[DIRECTOR MODEL] {model}"
    )

    print(
        f"[DIRECTOR MODEL SIZE] {size_gib:.2f} GiB"
    )

    return model


# ============================================================
# CUSTOM NODES
# ============================================================

def install_nodes() -> None:

    manifest = load_yaml(
        NODE_MANIFEST
    )

    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = (
        manifest[
            "custom_nodes"
        ][
            "required"
        ],
        manifest[
            "custom_nodes"
        ][
            "supporting"
        ],
    )

    for group in groups:

        for node in group:

            destination = (
                CUSTOM
                / node["name"]
            )

            git_checkout(
                node["repository"],
                destination,
                node["revision"],
            )

            print(
                f"[NODE] {node['name']} "
                f"@ {node['revision']}"
            )


# ============================================================
# H3 PRODUCTION MODELS
# ============================================================

def install_models() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    models = manifest[
        "models"
    ]

    for model in models.values():

        filename = model[
            "filename"
        ]

        source = find_model(
            filename
        )

        destination = (
            MODELS
            / model["directory"]
            / filename
        )

        link_model(
            source,
            destination,
        )

        print(
            f"[H3 MODEL] {filename}"
        )


# ============================================================
# H3 MODEL INVENTORY
# ============================================================

def verify_inventory() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    expected = {
        model[
            "filename"
        ].lower()
        for model
        in manifest[
            "models"
        ].values()
    }

    actual = set()

    model_dirs = {
        "diffusion_models",
        "text_encoders",
        "loras",
        "vae",
        "latent_upscale_models",
    }

    for directory_name in model_dirs:

        directory = (
            MODELS
            / directory_name
        )

        if not directory.is_dir():
            continue

        for path in directory.iterdir():

            if path.is_file():

                actual.add(
                    path.name.lower()
                )

    missing = sorted(
        expected
        - actual
    )

    unexpected = sorted(
        actual
        - expected
    )

    if missing:

        raise RuntimeError(
            "Missing locked production models:\n"
            + "\n".join(
                missing
            )
        )

    if unexpected:

        raise RuntimeError(
            "Unexpected production models:\n"
            + "\n".join(
                unexpected
            )
        )


# ============================================================
# DIRECTOR DATASET CONTRACT
# ============================================================

def verify_director_is_not_in_comfy_inventory() -> None:

    if not MODEL_MANIFEST.is_file():
        raise RuntimeError(
            f"Missing model manifest: {MODEL_MANIFEST}"
        )

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    inventory_files = {
        model[
            "filename"
        ]
        for model
        in manifest[
            "models"
        ].values()
    }

    if (
        DIRECTOR_MODEL_FILENAME
        in inventory_files
    ):

        raise RuntimeError(
            "Qwen3-14B director model must NOT "
            "be placed in the ComfyUI production "
            "model inventory."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    if not (
        COMFY / "main.py"
    ).is_file():

        raise RuntimeError(
            "ComfyUI is missing."
        )

    verify_director_is_not_in_comfy_inventory()

    install_director_runtime()

    director_model = (
        find_director_model()
    )

    install_nodes()
    install_models()
    verify_inventory()

    print(
        "=" * 80
    )

    print(
        "DIRECTOR MODEL READY:",
        director_model,
    )

    print(
        "MiniMax H3 Kaggle bootstrap PASSED."
    )


if __name__ == "__main__":
    main()

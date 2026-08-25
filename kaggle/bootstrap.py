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
    ROOT / "configs" / "model_inventory.yaml"
)

NODE_MANIFEST = (
    ROOT / "configs" / "custom_nodes.yaml"
)


def run(*args: str | Path) -> None:
    print(
        "+",
        " ".join(str(value) for value in args),
    )

    subprocess.run(
        [str(value) for value in args],
        check=True,
    )


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
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


def find_model(
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
            f"Locked model not found: {filename}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple copies of model found: {filename}\n"
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
            destination
        )


def install_nodes() -> None:

    manifest = load_yaml(
        NODE_MANIFEST
    )

    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = (
        manifest["custom_nodes"]["required"],
        manifest["custom_nodes"]["supporting"],
    )

    for group in groups:
        for node in group:

            destination = (
                CUSTOM / node["name"]
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


def install_models() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    models = manifest["models"]

    for model in models.values():

        filename = model["filename"]

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
            f"[MODEL] {filename}"
        )


def verify_inventory() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    expected = {
        model["filename"].lower()
        for model in manifest["models"].values()
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
            MODELS / directory_name
        )

        if not directory.is_dir():
            continue

        for path in directory.iterdir():

            if path.is_file():
                actual.add(
                    path.name.lower()
                )

    missing = sorted(
        expected - actual
    )

    unexpected = sorted(
        actual - expected
    )

    if missing:
        raise RuntimeError(
            "Missing locked production models:\n"
            + "\n".join(missing)
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected production models:\n"
            + "\n".join(unexpected)
        )


def main() -> None:

    if not (
        COMFY / "main.py"
    ).is_file():
        raise RuntimeError(
            "ComfyUI is missing."
        )

    install_nodes()
    install_models()
    verify_inventory()

    print(
        "MiniMax H3 Kaggle bootstrap PASSED."
    )


if __name__ == "__main__":
    main()

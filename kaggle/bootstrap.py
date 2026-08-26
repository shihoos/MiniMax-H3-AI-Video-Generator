from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml


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

RUNTIME_MANIFEST = (
    ROOT
    / "configs"
    / "runtime_versions.yaml"
)


def run(
    *args,
):
    print(
        "+",
        " ".join(
            str(value)
            for value
            in args
        ),
    )

    subprocess.run(
        [
            str(value)
            for value in args
        ],
        check=True,
    )


def load_yaml(
    path: Path,
):
    return yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )


def find_kaggle_file(
    filename: str,
) -> Path:

    matches = []

    for path in KAGGLE_INPUT.rglob(
        "*"
    ):

        if (
            path.is_file()
            and path.name.lower()
            == filename.lower()
        ):
            matches.append(
                path
            )

    if not matches:
        raise FileNotFoundError(
            f"Required Kaggle asset not found: {filename}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple copies found for {filename}:\n"
            + "\n".join(
                str(path)
                for path
                in matches
            )
        )

    return matches[0]


def link_model(
    source: Path,
    destination: Path,
):

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


def install_director_runtime(
    runtime: dict,
) -> None:

    package = runtime[
        "llama_cpp"
    ][
        "package"
    ]

    version = runtime[
        "llama_cpp"
    ][
        "version"
    ]

    index = runtime[
        "llama_cpp"
    ][
        "cuda_index"
    ]

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--only-binary=:all:",
        f"{package}=={version}",
        "--extra-index-url",
        index,
    )

    run(
        sys.executable,
        "-c",
        (
            "from llama_cpp import Llama; "
            "print('llama-cpp-python: PASS')"
        ),
    )


def install_nodes():

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

            if not destination.exists():

                run(
                    "git",
                    "clone",
                    node[
                        "repository"
                    ],
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
                node[
                    "revision"
                ],
            )

            print(
                "[NODE]",
                node[
                    "name"
                ],
            )


def install_models():

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    for model in manifest[
        "models"
    ].values():

        filename = model[
            "filename"
        ]

        source = (
            find_kaggle_file(
                filename
            )
        )

        destination = (
            MODELS
            / model[
                "directory"
            ]
            / filename
        )

        link_model(
            source,
            destination,
        )

        print(
            "[MODEL]",
            filename,
        )


def verify_inventory():

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    expected = {
        value[
            "filename"
        ].lower()
        for value
        in manifest[
            "models"
        ].values()
    }

    actual = set()

    for directory in (
        "diffusion_models",
        "text_encoders",
        "loras",
        "vae",
        "latent_upscale_models",
    ):

        path = (
            MODELS
            / directory
        )

        if not path.is_dir():
            continue

        for item in path.iterdir():

            if item.is_file():
                actual.add(
                    item.name.lower()
                )

    missing = (
        expected
        - actual
    )

    unexpected = (
        actual
        - expected
    )

    if missing:

        raise RuntimeError(
            "Missing H3 models:\n"
            + "\n".join(
                sorted(
                    missing
                )
            )
        )

    if unexpected:

        raise RuntimeError(
            "Unexpected H3 production models:\n"
            + "\n".join(
                sorted(
                    unexpected
                )
            )
        )


def main():

    runtime = load_yaml(
        RUNTIME_MANIFEST
    )

    director_filename = (
        runtime[
            "director"
        ][
            "model_filename"
        ]
    )

    director_model = (
        find_kaggle_file(
            director_filename
        )
    )

    print(
        "[DIRECTOR MODEL]",
        director_model,
    )

    install_director_runtime(
        runtime
    )

    install_nodes()
    install_models()
    verify_inventory()

    print(
        "=" * 80
    )

    print(
        "MiniMax H3 Kaggle bootstrap PASSED."
    )


if __name__ == "__main__":
    main()

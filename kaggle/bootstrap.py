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
            for value in args
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
                for path in matches
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

    llama_config = runtime[
        "llama_cpp"
    ]

    cuda_config = runtime[
        "cuda_runtime"
    ]

    package = llama_config[
        "package"
    ]

    version = llama_config[
        "version"
    ]

    index = llama_config[
        "cuda_index"
    ]

    cuda_runtime_package = cuda_config[
        "runtime_package"
    ]

    cuda_runtime_version = cuda_config[
        "runtime_version"
    ]

    cublas_package = cuda_config[
        "cublas_package"
    ]

    cublas_version = cuda_config[
        "cublas_version"
    ]

    print(
        "=" * 80
    )

    print(
        "INSTALLING QWEN DIRECTOR CUDA RUNTIME"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # CUDA runtime libraries required by the CUDA 13
    # llama.cpp wheel.
    # --------------------------------------------------------

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        f"{cuda_runtime_package}=={cuda_runtime_version}",
        f"{cublas_package}=={cublas_version}",
    )

    # --------------------------------------------------------
    # Verify the NVIDIA Python packages and locate their
    # native library directories.
    # --------------------------------------------------------

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import site; "
                "print('\\n'.join(site.getsitepackages()))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    site_roots = [
        Path(
            line.strip()
        )
        for line
        in probe.stdout.splitlines()
        if line.strip()
    ]

    library_directories = []

    for site_root in site_roots:

        nvidia_root = (
            site_root
            / "nvidia"
        )

        if not nvidia_root.is_dir():
            continue

        for library in nvidia_root.rglob(
            "libcudart.so*"
        ):

            if library.is_file():

                parent = (
                    library.parent
                )

                if parent not in library_directories:
                    library_directories.append(
                        parent
                    )

        for library in nvidia_root.rglob(
            "libcublas.so*"
        ):

            if library.is_file():

                parent = (
                    library.parent
                )

                if parent not in library_directories:
                    library_directories.append(
                        parent
                    )

    if not library_directories:
        raise RuntimeError(
            "NVIDIA CUDA Python packages were installed, "
            "but no native CUDA library directories were found."
        )

    cuda_runtime_libraries = []

    cublas_libraries = []

    for directory in library_directories:

        if any(
            path.name.startswith(
                "libcudart.so.13"
            )
            for path
            in directory.iterdir()
            if path.is_file()
        ):
            cuda_runtime_libraries.append(
                directory
            )

        if any(
            path.name.startswith(
                "libcublas.so.13"
            )
            for path
            in directory.iterdir()
            if path.is_file()
        ):
            cublas_libraries.append(
                directory
            )

    if not cuda_runtime_libraries:
        raise RuntimeError(
            "libcudart.so.13 was not found after installing "
            f"{cuda_runtime_package}=={cuda_runtime_version}."
        )

    if not cublas_libraries:
        raise RuntimeError(
            "libcublas.so.13 was not found after installing "
            f"{cublas_package}=={cublas_version}."
        )

    runtime_library_paths = []

    for directory in (
        cuda_runtime_libraries
        + cublas_libraries
    ):

        if directory not in runtime_library_paths:

            runtime_library_paths.append(
                directory
            )

    print(
        "[CUDA RUNTIME]"
    )

    for directory in runtime_library_paths:

        print(
            " ",
            directory,
        )

    # --------------------------------------------------------
    # Install llama-cpp-python CUDA wheel.
    # --------------------------------------------------------

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--only-binary=:all:",
        "--disable-pip-version-check",
        f"{package}=={version}",
        "--extra-index-url",
        index,
    )

    # --------------------------------------------------------
    # Verify the CUDA llama.cpp wheel in an environment where
    # the native NVIDIA libraries are explicitly visible.
    # --------------------------------------------------------

    environment = {
        **__import__(
            "os"
        ).environ
    }

    existing_ld = environment.get(
        "LD_LIBRARY_PATH",
        "",
    )

    ld_parts = [
        str(path)
        for path
        in runtime_library_paths
    ]

    if existing_ld:
        ld_parts.append(
            existing_ld
        )

    environment[
        "LD_LIBRARY_PATH"
    ] = ":".join(
        ld_parts
    )

    verification = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from llama_cpp import Llama; "
                "print('llama-cpp-python CUDA import: PASS')"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    if verification.stdout:
        print(
            verification.stdout
        )

    if verification.stderr:
        print(
            verification.stderr
        )

    if verification.returncode != 0:
        raise RuntimeError(
            "llama-cpp-python CUDA import failed "
            "after installing the required CUDA runtime libraries."
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

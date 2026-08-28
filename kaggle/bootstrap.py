from __future__ import annotations

import os
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

_configured_input_root = os.getenv("H3_INPUT_ROOT", "").strip()
if _configured_input_root:
    KAGGLE_INPUT = Path(_configured_input_root).expanduser().resolve()
elif Path("/kaggle/input").is_dir():
    KAGGLE_INPUT = Path("/kaggle/input").resolve()
else:
    KAGGLE_INPUT = (ROOT / "input").resolve()

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
    env=None,
) -> None:

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
        env=env,
    )


def load_yaml(
    path: Path,
) -> dict:

    value = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        value,
        dict,
    ):
        raise RuntimeError(
            f"Invalid YAML mapping: {path}"
        )

    return value


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
            "Required Kaggle asset not found: "
            f"{filename}"
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


def _site_packages() -> list[Path]:

    result = subprocess.run(
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

    return [
        Path(
            line.strip()
        )
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def _cuda_library_dirs() -> list[Path]:

    directories = []

    for site_root in _site_packages():

        nvidia_root = (
            site_root
            / "nvidia"
        )

        if not nvidia_root.is_dir():
            continue

        for pattern in (
            "libcudart.so.13*",
            "libcublas.so.13*",
        ):

            for library in nvidia_root.rglob(
                pattern
            ):

                if not library.is_file():
                    continue

                directory = (
                    library.parent
                )

                if directory not in directories:
                    directories.append(
                        directory
                    )

    return directories


def _configure_cuda_environment(
    directories: list[Path],
) -> dict[str, str]:

    environment = dict(
        os.environ
    )

    existing = environment.get(
        "LD_LIBRARY_PATH",
        "",
    )

    values = [
        str(path)
        for path in directories
    ]

    if existing:
        values.append(
            existing
        )

    environment[
        "LD_LIBRARY_PATH"
    ] = ":".join(
        values
    )

    return environment


def install_base_requirements() -> None:
    requirements = ROOT / "requirements.txt"

    if not requirements.is_file():
        raise RuntimeError(
            f"Repository dependency manifest is missing: {requirements}"
        )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        "-r",
        requirements,
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

    llama_package = llama_config[
        "package"
    ]

    llama_version = llama_config[
        "version"
    ]

    cuda_index = llama_config[
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
        "INSTALLING QWEN DIRECTOR RUNTIME"
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
        "--disable-pip-version-check",
        f"{cuda_runtime_package}=={cuda_runtime_version}",
        f"{cublas_package}=={cublas_version}",
    )

    library_dirs = (
        _cuda_library_dirs()
    )

    if not library_dirs:

        raise RuntimeError(
            "NVIDIA CUDA runtime packages installed, "
            "but no native CUDA library directories were found."
        )

    has_cudart = any(
        any(
            path.name.startswith(
                "libcudart.so.13"
            )
            for path in directory.iterdir()
            if path.is_file()
        )
        for directory in library_dirs
    )

    has_cublas = any(
        any(
            path.name.startswith(
                "libcublas.so.13"
            )
            for path in directory.iterdir()
            if path.is_file()
        )
        for directory in library_dirs
    )

    if not has_cudart:

        raise RuntimeError(
            "libcudart.so.13 was not found."
        )

    if not has_cublas:

        raise RuntimeError(
            "libcublas.so.13 was not found."
        )

    environment = (
        _configure_cuda_environment(
            library_dirs
        )
    )

    print(
        "[CUDA RUNTIME LIBRARIES]"
    )

    for directory in library_dirs:

        print(
            " ",
            directory,
        )

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--only-binary=:all:",
        "--disable-pip-version-check",
        f"{llama_package}=={llama_version}",
        "--extra-index-url",
        cuda_index,
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
            "llama-cpp-python CUDA import failed."
        )




def install_comfyui(runtime: dict) -> None:
    """Install the exact ComfyUI revision required by the project."""
    config = runtime.get("comfyui", {})
    repository = str(config.get("repository", "") or "").strip()
    revision = str(config.get("revision", "") or "").strip()
    if not repository or not revision:
        raise RuntimeError("runtime_versions.yaml must define comfyui.repository and comfyui.revision.")

    print("=" * 80)
    print("INSTALLING COMFYUI")
    print("=" * 80)

    if COMFY.exists() and not (COMFY / ".git").exists():
        raise RuntimeError(
            f"ComfyUI path exists but is not a git checkout: {COMFY}. "
            "Move it away and rerun bootstrap."
        )

    if not COMFY.exists():
        run("git", "clone", repository, COMFY)

    run("git", "-C", COMFY, "fetch", "--tags", "--prune", "origin")
    run("git", "-C", COMFY, "checkout", "--detach", revision)

    requirements = COMFY / "requirements.txt"
    if not requirements.is_file():
        raise RuntimeError(f"ComfyUI requirements.txt is missing: {requirements}")

    run(
        sys.executable, "-m", "pip", "install", "-q",
        "--disable-pip-version-check", "-r", requirements,
    )

    expected_version = str(config.get("expected_version", "") or "").strip()
    version_check = subprocess.run(
        [sys.executable, "-c", "import importlib.metadata as m; print(m.version('comfyui'))"],
        cwd=str(COMFY),
        capture_output=True,
        text=True,
        check=False,
    )
    installed_version = (version_check.stdout or "").strip()
    # The git checkout is the authority. If a package distribution is not
    # installed under the same name, verify the git revision directly below.
    rev = subprocess.run(
        ["git", "-C", str(COMFY), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    expected_rev = subprocess.run(
        ["git", "-C", str(COMFY), "rev-list", "-n", "1", revision],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if rev != expected_rev:
        raise RuntimeError(
            f"ComfyUI revision mismatch: checked out {rev}, expected {expected_rev}."
        )
    print(f"[COMFYUI] revision={rev}")
    if expected_version:
        print(f"[COMFYUI] expected release={expected_version}")
    if installed_version:
        print(f"[COMFYUI] package version={installed_version}")

def install_storyboard_runtime(
    runtime: dict,
) -> None:

    version = runtime[
        "storyboard"
    ][
        "gradio_version"
    ]

    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        f"gradio=={version}",
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
                / node[
                    "name"
                ]
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

            requirements = destination / "requirements.txt"

            if requirements.is_file():
                run(
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "--disable-pip-version-check",
                    "-r",
                    requirements,
                )

                print(
                    "[NODE DEPS]",
                    node[
                        "name"
                    ],
                )

            print(
                "[NODE]",
                node[
                    "name"
                ],
            )


def install_models() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    for model in manifest[
        "models"
    ].values():

        filename = model[
            "filename"
        ]

        source = find_kaggle_file(
            filename
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


def verify_inventory() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    expected = {
        (
            model[
                "directory"
            ],
            model[
                "filename"
            ].lower(),
        )
        for model
        in manifest[
            "models"
        ].values()
    }

    actual = set()

    production_directories = {
        "diffusion_models",
        "text_encoders",
        "loras",
        "vae",
        "latent_upscale_models",
    }

    for directory_name in (
        production_directories
    ):

        directory = (
            MODELS
            / directory_name
        )

        if not directory.is_dir():
            continue

        for item in directory.iterdir():

            if not item.is_file():
                continue

            actual.add(
                (
                    directory_name,
                    item.name.lower(),
                )
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
                f"{directory}/{filename}"
                for directory, filename
                in sorted(
                    missing
                )
            )
        )

    if unexpected:

        raise RuntimeError(
            "Unexpected H3 production models:\n"
            + "\n".join(
                f"{directory}/{filename}"
                for directory, filename
                in sorted(
                    unexpected
                )
            )
        )




def verify_runtime_files(runtime: dict) -> None:
    if not (COMFY / "main.py").is_file():
        raise RuntimeError(f"ComfyUI main.py is missing: {COMFY / 'main.py'}")
    if not CUSTOM.is_dir():
        raise RuntimeError(f"ComfyUI custom_nodes directory is missing: {CUSTOM}")
    if not MODELS.is_dir():
        raise RuntimeError(f"ComfyUI models directory is missing: {MODELS}")

    expected_version = str(runtime.get("comfyui", {}).get("expected_version", "") or "").strip()
    revision = str(runtime.get("comfyui", {}).get("revision", "") or "").strip()
    head = subprocess.run(
        ["git", "-C", str(COMFY), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    tagged = subprocess.run(
        ["git", "-C", str(COMFY), "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if revision:
        expected_head = subprocess.run(
            ["git", "-C", str(COMFY), "rev-list", "-n", "1", revision],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if head != expected_head:
            raise RuntimeError("ComfyUI checkout is not at the locked revision.")
    if expected_version and tagged not in {expected_version, f"v{expected_version}"}:
        raise RuntimeError(
            f"ComfyUI checkout is not the expected release tag. "
            f"HEAD={tagged or 'untagged'}, expected={expected_version}."
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

    install_base_requirements()

    install_comfyui(runtime)

    install_director_runtime(
        runtime
    )

    install_storyboard_runtime(
        runtime
    )

    install_nodes()

    install_models()

    verify_inventory()
    verify_runtime_files(runtime)

    print(
        "=" * 80
    )

    print(
        "MiniMax H3 Kaggle bootstrap PASSED."
    )


if __name__ == "__main__":
    main()

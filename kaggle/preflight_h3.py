from __future__ import annotations

import ctypes
import json
import os
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

PRODUCTION_WORKFLOWS = [
    (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Ref2V_Production.json"
    ),
    (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Turbo_Ref2V_Production.json"
    ),
    (
        ROOT
        / "workflows"
        / "postprocess"
        / "H3_Ref2V_UltimateUpscale_Production.json"
    ),
]

SOURCE_WORKFLOWS = [
    (
        ROOT
        / "workflows"
        / "sources"
        / "H3_Turbo_Reference_Source.json"
    ),
    (
        ROOT
        / "workflows"
        / "sources"
        / "H3_LatentUpscaler_Source.json"
    ),
]


def load_yaml(
    path: Path,
) -> dict:

    if not path.is_file():
        raise RuntimeError(
            f"Missing YAML manifest: {path}"
        )

    data = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"Invalid YAML mapping: {path}"
        )

    return data


def load_json(
    path: Path,
) -> dict:

    if not path.is_file():
        raise RuntimeError(
            f"Missing workflow: {path}"
        )

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON in {path}: {exc}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            f"Workflow root must be an object: {path}"
        )

    return data


# ============================================================
# NVIDIA CUDA NATIVE LIBRARIES
# ============================================================

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
        for line
        in result.stdout.splitlines()
        if line.strip()
    ]


def _find_cuda_libraries() -> tuple[
    Path,
    Path,
]:

    cudart_candidates = []
    cublas_candidates = []

    for site_root in _site_packages():

        nvidia_root = (
            site_root
            / "nvidia"
        )

        if not nvidia_root.is_dir():
            continue

        for path in nvidia_root.rglob(
            "libcudart.so.13*"
        ):

            if path.is_file():
                cudart_candidates.append(
                    path
                )

        for path in nvidia_root.rglob(
            "libcublas.so.13*"
        ):

            if path.is_file():
                cublas_candidates.append(
                    path
                )

    if not cudart_candidates:

        raise RuntimeError(
            "libcudart.so.13 was not found in the "
            "installed NVIDIA Python packages."
        )

    if not cublas_candidates:

        raise RuntimeError(
            "libcublas.so.13 was not found in the "
            "installed NVIDIA Python packages."
        )

    return (
        cudart_candidates[0],
        cublas_candidates[0],
    )


def _prepare_cuda_environment() -> tuple[
    Path,
    Path,
]:

    cudart, cublas = (
        _find_cuda_libraries()
    )

    library_dirs = [
        str(
            cudart.parent
        ),
        str(
            cublas.parent
        ),
    ]

    existing = os.environ.get(
        "LD_LIBRARY_PATH",
        "",
    )

    if existing:
        library_dirs.append(
            existing
        )

    os.environ[
        "LD_LIBRARY_PATH"
    ] = ":".join(
        library_dirs
    )

    try:

        ctypes.CDLL(
            str(cudart),
            mode=ctypes.RTLD_GLOBAL,
        )

        ctypes.CDLL(
            str(cublas),
            mode=ctypes.RTLD_GLOBAL,
        )

    except OSError as exc:

        raise RuntimeError(
            "Unable to load NVIDIA CUDA native libraries.\n"
            f"CUDA runtime: {cudart}\n"
            f"cuBLAS: {cublas}\n"
            f"Error: {exc}"
        ) from exc

    return (
        cudart,
        cublas,
    )


# ============================================================
# DIRECTOR
# ============================================================

def find_director_model() -> Path:

    runtime = load_yaml(
        RUNTIME_MANIFEST
    )

    filename = (
        runtime[
            "director"
        ][
            "model_filename"
        ]
    )

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

    if len(matches) != 1:

        raise RuntimeError(
            "Expected exactly one Qwen director model.\n"
            f"Filename: {filename}\n"
            f"Found: {len(matches)}\n"
            + (
                "\n".join(
                    str(path)
                    for path in matches
                )
                if matches
                else
                "No matching model was found."
            )
        )

    model = matches[0]

    if model.stat().st_size <= 0:

        raise RuntimeError(
            f"Qwen director model is empty: {model}"
        )

    return model


def check_director() -> None:

    model = (
        find_director_model()
    )

    print(
        "DIRECTOR MODEL:",
        model,
    )

    try:
        cudart, cublas = (
            _prepare_cuda_environment()
        )

        print(
            "CUDA RUNTIME:",
            cudart,
        )

        print(
            "CUBLAS:",
            cublas,
        )

        from llama_cpp import (
            Llama,
        )

    except ImportError as exc:

        raise RuntimeError(
            "llama-cpp-python is unavailable."
        ) from exc

    except OSError as exc:

        raise RuntimeError(
            "llama-cpp-python native CUDA library "
            "could not be loaded:\n"
            f"{exc}"
        ) from exc

    except RuntimeError:
        raise

    if Llama is None:
        raise RuntimeError(
            "llama_cpp.Llama is unavailable."
        )

    print(
        "DIRECTOR RUNTIME: PASS"
    )


# ============================================================
# H3 MODELS
# ============================================================

def check_models() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    for model in manifest[
        "models"
    ].values():

        filename = model[
            "filename"
        ]

        path = (
            COMFY
            / "models"
            / model[
                "directory"
            ]
            / filename
        )

        if not path.is_file():

            raise RuntimeError(
                f"Missing H3 model: {path}"
            )

        if path.stat().st_size <= 0:

            raise RuntimeError(
                f"H3 model is empty: {path}"
            )

        print(
            "MODEL OK:",
            filename,
        )


# ============================================================
# WORKFLOWS
# ============================================================

def check_workflows() -> None:

    for path in (
        PRODUCTION_WORKFLOWS
        + SOURCE_WORKFLOWS
    ):

        graph = load_json(
            path
        )

        nodes = graph.get(
            "nodes"
        )

        if not isinstance(
            nodes,
            list,
        ):

            raise RuntimeError(
                f"Workflow has invalid nodes list: {path}"
            )

        print(
            "WORKFLOW OK:",
            path.relative_to(
                ROOT
            ),
        )


# ============================================================
# UPSCALE CONTRACT
# ============================================================

def check_upscale_contract() -> None:

    graph = load_json(
        PRODUCTION_WORKFLOWS[
            2
        ]
    )

    types = {
        node.get(
            "type"
        )
        for node
        in graph.get(
            "nodes",
            [],
        )
    }

    required = {
        "MMH3LatentUpscaleWithModelParams",
        "MMH3TemporalSplitParams",
        "MMH3SpatialSplitParams",
        "MMH3UltimateUpscale",
    }

    missing = (
        required
        - types
    )

    if missing:

        raise RuntimeError(
            "Upscale workflow missing nodes: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    text = json.dumps(
        graph
    )

    if (
        "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        not in text
    ):

        raise RuntimeError(
            "Upscale workflow does not reference "
            "the locked H3 3D latent upscaler."
        )

    print(
        "UPSCALE CONTRACT: PASS"
    )


# ============================================================
# CUSTOM NODES
# ============================================================

def check_custom_nodes() -> None:

    manifest = load_yaml(
        NODE_MANIFEST
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

            path = (
                COMFY
                / "custom_nodes"
                / node[
                    "name"
                ]
            )

            if not path.is_dir():

                raise RuntimeError(
                    f"Missing custom node: {path}"
                )

            print(
                "NODE OK:",
                node[
                    "name"
                ],
            )


# ============================================================
# DELIVERY CONTRACT
# ============================================================

def check_delivery() -> None:

    runtime = load_yaml(
        RUNTIME_MANIFEST
    )

    generation = runtime[
        "generation"
    ]

    upscale = runtime[
        "upscale"
    ]

    delivery = runtime[
        "delivery"
    ]

    if (
        generation[
            "width"
        ],
        generation[
            "height"
        ],
        generation[
            "fps"
        ],
    ) != (
        1344,
        768,
        24,
    ):

        raise RuntimeError(
            "Invalid H3 generation contract."
        )

    if (
        generation[
            "turbo_steps"
        ]
    ) != 8:

        raise RuntimeError(
            "Turbo must use 8 steps."
        )

    if (
        upscale[
            "width"
        ],
        upscale[
            "height"
        ],
    ) != (
        1920,
        1088,
    ):

        raise RuntimeError(
            "Invalid H3 upscale contract."
        )

    if (
        delivery[
            "width"
        ],
        delivery[
            "height"
        ],
        delivery[
            "fps"
        ],
    ) != (
        1280,
        720,
        24,
    ):

        raise RuntimeError(
            "Invalid final delivery contract."
        )

    print(
        "DELIVERY CONTRACT: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    check_director()

    check_models()

    check_workflows()

    check_upscale_contract()

    check_custom_nodes()

    check_delivery()

    print(
        "=" * 80
    )

    print(
        "MiniMax H3 PRE-FLIGHT PASSED."
    )


if __name__ == "__main__":
    main()

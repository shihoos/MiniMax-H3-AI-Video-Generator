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

PRODUCTION_WORKFLOWS = [
    (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Ref2VA_Production.json"
    ),
    (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Turbo_Ref2VA_Production.json"
    ),
    (
        ROOT
        / "workflows"
        / "postprocess"
        / "H3_Ref2VA_UltimateUpscale_Production.json"
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

def _resolve_kaggle_asset(configured: str) -> Path:
    """Resolve a logical Kaggle input path, including nested dataset mounts."""
    configured_path = Path(str(configured or "").strip()).expanduser()
    candidates = [configured_path, KAGGLE_INPUT / configured_path.name]
    if KAGGLE_INPUT.is_dir():
        try:
            candidates.extend(p for p in KAGGLE_INPUT.rglob(configured_path.name) if p.is_dir())
        except OSError:
            pass
    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if str(candidate) in seen or not candidate.is_dir():
            continue
        seen.add(str(candidate))
        return candidate
    raise FileNotFoundError(f"Kaggle asset directory was not found: {configured}")

def find_director_model() -> Path:
    runtime = load_yaml(RUNTIME_MANIFEST)
    director = dict(runtime.get("director", {}) or {})
    if str(director.get("backend", "") or "").strip().lower() != "vllm":
        raise RuntimeError("Director backend must be vllm.")
    configured = str(director.get("model_path", "") or "").strip()
    if not configured:
        raise RuntimeError("runtime_versions.yaml director.model_path is empty.")
    try:
        candidate = _resolve_kaggle_asset(configured)
    except FileNotFoundError:
        candidate = None
    required = ("config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "tokenizer.json")
    if candidate is not None and all((candidate / name).is_file() for name in required):
        return candidate
    raise RuntimeError("Complete Qwen3-14B-AWQ checkpoint was not found at " + configured)


def find_director_speculator() -> Path:
    runtime = load_yaml(RUNTIME_MANIFEST)
    director = dict(runtime.get("director", {}) or {})
    configured = str(director.get("speculative_model_path", "") or "").strip()
    if configured != "/kaggle/input/eagle-3":
        raise RuntimeError(
            "runtime_versions.yaml director.speculative_model_path must be /kaggle/input/eagle-3 for the locked Eagle-3 Kaggle dataset."
        )

    try:
        resolved_candidate = _resolve_kaggle_asset(configured)
    except FileNotFoundError:
        resolved_candidate = None
    for candidate in ([resolved_candidate] if resolved_candidate is not None else []):
        config_path = candidate / "config.json"
        if not (candidate.is_dir() and config_path.is_file()):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid Eagle-3 config: {config_path}") from exc
        spec = config.get("speculators_config", {}) or {}
        verifier = spec.get("verifier", {}) or {}
        if str(spec.get("algorithm", "")).strip().lower() != "eagle3":
            raise RuntimeError(
                f"Eagle-3 checkpoint has unexpected algorithm: {spec.get('algorithm')!r}."
            )
        if str(verifier.get("name_or_path", "")).strip() != "Qwen/Qwen3-14B":
            raise RuntimeError(
                "Eagle-3 checkpoint is not paired with Qwen/Qwen3-14B."
            )

        index_files = tuple(candidate.glob("*.index.json"))
        for index_path in index_files:
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if isinstance(weight_map, dict) and weight_map:
                missing = sorted({str(name) for name in weight_map.values() if not (candidate / str(name)).is_file()})
                if missing:
                    raise RuntimeError(
                        "Eagle-3 checkpoint is incomplete; missing indexed weight files: "
                        + ", ".join(missing)
                    )
                return candidate

        if any(any(candidate.glob(pattern)) for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")):
            return candidate

        raise RuntimeError(
            f"Eagle-3 checkpoint has config.json but no model weight files: {candidate}"
        )

    raise RuntimeError(
        "Complete Eagle-3 checkpoint was not found at /kaggle/input/eagle-3. "
        "Attach the Kaggle dataset named 'eagle-3'."
    )


def check_director() -> None:
    model = find_director_model()
    speculator = find_director_speculator()
    print("DIRECTOR MODEL:", model)
    print("DIRECTOR SPECULATOR:", speculator)
    director = load_yaml(RUNTIME_MANIFEST).get("director", {}) or {}
    version = str(director.get("vllm_version", "") or "").strip()
    env_dir_value = str(director.get("vllm_env_dir", "") or "").strip()
    method = str(director.get("speculative_method", "") or "").strip().lower()
    tokens = int(director.get("speculative_tokens", 0) or 0)
    if not version:
        raise RuntimeError("runtime_versions.yaml director.vllm_version is required.")
    if not env_dir_value:
        raise RuntimeError("runtime_versions.yaml director.vllm_env_dir is required.")
    if method != "eagle3":
        raise RuntimeError("runtime_versions.yaml director.speculative_method must be eagle3.")
    if tokens <= 0:
        raise RuntimeError("runtime_versions.yaml director.speculative_tokens must be positive.")
    env_dir = Path(env_dir_value).expanduser()
    venv_python = env_dir / "bin" / "python"
    if not version or not venv_python.is_file():
        raise RuntimeError("Pinned vLLM environment is missing; run bootstrap first.")
    result = subprocess.run([str(venv_python), "-c", "import vllm; print(vllm.__version__)"], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("vLLM Director runtime is unavailable.\n" + (result.stderr or result.stdout))
    observed = result.stdout.strip()
    if observed != version:
        raise RuntimeError(f"Director vLLM version mismatch: observed={observed}, expected={version}")
    print("DIRECTOR RUNTIME: PASS")
    print("DIRECTOR VLLM:", observed, "expected:", version)
    print("DIRECTOR SPECULATION: PASS", f"method={method}", f"tokens={tokens}", f"speculator={speculator}")


# ============================================================
# COMFYUI RUNTIME
# ============================================================

def check_comfyui() -> None:
    runtime = load_yaml(RUNTIME_MANIFEST)
    config = runtime.get("comfyui", {})
    main_py = COMFY / "main.py"
    if not main_py.is_file():
        raise RuntimeError(f"ComfyUI is not installed: {main_py}")

    expected_version = str(config.get("expected_version", "") or "").strip()
    revision = str(config.get("revision", "") or "").strip()
    if not (COMFY / ".git").is_dir():
        raise RuntimeError(f"ComfyUI is not a git checkout: {COMFY}")

    head = subprocess.run(
        ["git", "-C", str(COMFY), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if revision:
        expected_head = subprocess.run(
            ["git", "-C", str(COMFY), "rev-list", "-n", "1", revision],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if head != expected_head:
            raise RuntimeError(
                f"ComfyUI revision mismatch: {head} != {expected_head}"
            )

    print("COMFYUI RUNTIME: PASS", expected_version or revision)


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

            expected_revision = str(
                node.get(
                    "revision",
                    "",
                )
                or ""
            ).strip()

            if expected_revision:

                result = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(path),
                        "rev-parse",
                        "HEAD",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if result.returncode != 0:

                    raise RuntimeError(
                        "Unable to determine custom-node "
                        f"revision for {node['name']}:\n"
                        f"{result.stderr.strip()}"
                    )

                actual_revision = (
                    result.stdout.strip()
                )

                if actual_revision != expected_revision:

                    raise RuntimeError(
                        "Custom node revision mismatch:\n"
                        f"Node: {node['name']}\n"
                        f"Expected: {expected_revision}\n"
                        f"Actual: {actual_revision}"
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

    check_comfyui()

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

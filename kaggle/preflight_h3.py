from __future__ import annotations

import importlib.metadata as importlib_metadata
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
# MAIN COMFYUI PYTORCH RUNTIME
# ============================================================

def check_main_pytorch() -> None:
    runtime = load_yaml(RUNTIME_MANIFEST)
    config = runtime.get("pytorch", {}) or {}
    version = str(config.get("version", "") or "").strip()
    cuda_tag = str(config.get("cuda", "") or "").strip().lower()
    if version != "2.15.0" or cuda_tag != "cu132":
        raise RuntimeError("Main ComfyUI runtime must use PyTorch 2.15.0/cu132.")
    expected_torch = f"{version}+{cuda_tag}"
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Unable to import the main ComfyUI PyTorch runtime.") from exc
    actual_torch = str(torch.__version__)
    expected_cuda = f"{cuda_tag[2:4]}.{cuda_tag[4:]}"
    if actual_torch != expected_torch or str(torch.version.cuda) != expected_cuda:
        raise RuntimeError(
            "Main ComfyUI PyTorch runtime mismatch: "
            f"expected={expected_torch}/{expected_cuda}, "
            f"actual={actual_torch}/{torch.version.cuda}."
        )
    expected_tv = str(config.get("torchvision_version", "") or "").strip()
    expected_ta = str(config.get("torchaudio_version", "") or "").strip()
    if importlib_metadata.version("torchvision") != expected_tv:
        raise RuntimeError("Main ComfyUI torchvision version does not match runtime_versions.yaml.")
    if importlib_metadata.version("torchaudio") != expected_ta:
        raise RuntimeError("Main ComfyUI torchaudio version does not match runtime_versions.yaml.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Main H3 runtime requires both configured Tesla T4 GPUs to be visible.")
    capabilities = [torch.cuda.get_device_capability(index) for index in range(torch.cuda.device_count())]
    if any(capability != (7, 5) for capability in capabilities):
        raise RuntimeError(f"Main H3 runtime expects SM75 Tesla T4 GPUs; got {capabilities}.")
    director = runtime.get("director", {}) or {}
    director_env = Path(str(director.get("vllm_env_dir", ""))).expanduser().resolve()
    main_python = Path(sys.executable).resolve()
    director_python = director_env / "bin" / "python"
    if director_python.resolve() == main_python:
        raise RuntimeError("vLLM Director must not share the ComfyUI Python environment.")
    print(f"MAIN PYTORCH: PASS torch={actual_torch} cuda={torch.version.cuda} GPUs={capabilities}")


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



def _exact_requirement_pin(package_name: str) -> str:
    """Read one exact package pin from the project requirements file."""
    requirements_path = ROOT / "requirements.txt"
    matches = []
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, version = line.split("==", 1)
        if name.strip().lower().replace("_", "-") == package_name.lower().replace("_", "-"):
            version = version.split(";", 1)[0].strip()
            if version:
                matches.append(version)
    if len(matches) != 1:
        raise RuntimeError(
            f"requirements.txt must contain exactly one exact pin for {package_name}."
        )
    return matches[0]


def check_director() -> None:
    model = find_director_model()
    speculator = find_director_speculator()
    print("DIRECTOR MODEL:", model)
    print("DIRECTOR SPECULATOR:", speculator)
    director = load_yaml(RUNTIME_MANIFEST).get("director", {}) or {}
    version = str(director.get("vllm_version", "") or "").strip()
    if version != "0.32.0":
        raise RuntimeError(f"Director vLLM must be exactly 0.32.0, got {version!r}.")
    loader_strategy = str(director.get("safetensors_load_strategy", "") or "").strip().lower()
    if loader_strategy != "eager":
        raise RuntimeError("runtime_versions.yaml Director safetensors_load_strategy must default to eager.")
    transformers_version = _exact_requirement_pin("transformers")
    tokenizers_version = _exact_requirement_pin("tokenizers")
    env_dir_value = str(director.get("vllm_env_dir", "") or "").strip()
    env_dir_value = os.getenv("H3_DIRECTOR_VLLM_ENV_DIR", "").strip() or env_dir_value
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
    torch_version = str(director.get("vllm_torch_version", "") or "").strip()
    cuda_tag = str(director.get("vllm_cuda", "") or "").strip().lower()
    expected_torch = f"{torch_version}+{cuda_tag}"
    if cuda_tag != "cu130" or not torch_version:
        raise RuntimeError("Director vLLM runtime must explicitly lock PyTorch 2.13.0/cu130.")
    if str(load_yaml(RUNTIME_MANIFEST).get("python", {}).get("kaggle", "")) != "3.12":
        raise RuntimeError("Kaggle runtime must use Python 3.12 for the pinned vLLM environment.")
    probe = (
        "import sys, inspect, vllm, torch, transformers, tokenizers; "
        "from transformers import Qwen3Config; "
        "from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM; "
        "from vllm.config.load import LoadConfig; "
        "assert sys.version_info[:2] == (3, 12), sys.version; "
        f"assert torch.__version__ == {expected_torch!r}, torch.__version__; "
        "assert torch.version.cuda == '13.0', torch.version.cuda; "
        "assert 'safetensors_load_strategy' in inspect.signature(LoadConfig).parameters; "
        "assert torch.cuda.is_available() and torch.cuda.device_count() == 2; "
        "assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(torch.cuda.device_count())); "
        "print('vLLM version:', vllm.__version__); "
        "print('Director Torch version:', torch.__version__); "
        "print('Director CUDA runtime:', torch.version.cuda); "
        "print('safetensors loader API: PASS'); "
        "print('Transformers version:', transformers.__version__); "
        "print('tokenizers version:', tokenizers.__version__); "
        "print('Qwen3 imports: PASS')"
    )
    result = subprocess.run([str(venv_python), "-c", probe], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("vLLM Director runtime is unavailable.\n" + (result.stderr or result.stdout))
    print(result.stdout.strip())
    observed_versions = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            observed_versions[key.strip()] = value.strip()
    observed = observed_versions.get("vLLM version", "")
    observed_transformers = observed_versions.get("Transformers version", "")
    observed_tokenizers = observed_versions.get("tokenizers version", "")
    if observed != version:
        raise RuntimeError(f"Director vLLM version mismatch: observed={observed}, expected={version}")
    if observed_transformers != transformers_version:
        raise RuntimeError(
            "Director Transformers version mismatch: "
            f"observed={observed_transformers}, expected={transformers_version}"
        )
    if observed_tokenizers != tokenizers_version:
        raise RuntimeError(
            "Director tokenizers version mismatch: "
            f"observed={observed_tokenizers}, expected={tokenizers_version}"
        )
    if bool(director.get("allow_transformers_vllm_constraint_override", False)):
        print("DIRECTOR COMPATIBILITY: EXPERIMENTAL override enabled")
    print("DIRECTOR RUNTIME: PASS")
    print("DIRECTOR VLLM:", observed, "expected:", version)
    print("DIRECTOR TRANSFORMERS:", observed_transformers, "expected:", transformers_version)
    print("DIRECTOR TOKENIZERS:", observed_tokenizers, "expected:", tokenizers_version)
    print("DIRECTOR SPECULATION: PASS", f"method={method}", f"tokens={tokens}", f"speculator={speculator}")


# ============================================================
# COMFYUI RUNTIME
# ============================================================

def check_comfyui() -> None:
    runtime = load_yaml(RUNTIME_MANIFEST)
    config = runtime.get("comfyui", {}) or {}
    main_py = COMFY / "main.py"
    if not main_py.is_file():
        raise RuntimeError(f"ComfyUI is not installed: {main_py}")
    expected_version = str(config.get("expected_version", "") or "").strip()
    revision = str(config.get("revision", "") or "").strip()
    if expected_version != "0.43.2" or revision != "423cbd20511becb1f16122ab518e92e58bfa03d9":
        raise RuntimeError("Runtime manifest must lock ComfyUI v0.43.2 to its exact release commit.")
    if not (COMFY / ".git").is_dir():
        raise RuntimeError(f"ComfyUI is not a git checkout: {COMFY}")
    head = subprocess.run(["git", "-C", str(COMFY), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    expected_head = subprocess.run(["git", "-C", str(COMFY), "rev-list", "-n", "1", revision], capture_output=True, text=True, check=True).stdout.strip()
    if head != expected_head:
        raise RuntimeError(f"ComfyUI revision mismatch: {head} != {expected_head}")
    tagged = subprocess.run(["git", "-C", str(COMFY), "describe", "--tags", "--exact-match", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    if tagged not in {"0.43.2", "v0.43.2"}:
        raise RuntimeError(f"ComfyUI release tag mismatch: expected=v0.43.2, actual={tagged or 'untagged'}")

    model_source = (COMFY / "comfy/ldm/minimax/model.py").read_text(encoding="utf-8")
    vae_source = (COMFY / "comfy/ldm/minimax/vae.py").read_text(encoding="utf-8")
    supported_source = (COMFY / "comfy/supported_models.py").read_text(encoding="utf-8")
    required_model = ("class MiniMaxH3Model", "def _embed_and_pack(", "preferred_attention=self.comfy_attention")
    missing = [marker for marker in required_model if marker not in model_source]
    if missing:
        raise RuntimeError("ComfyUI H3 model.py source contract mismatch: " + ", ".join(missing))
    if "v = v.clone()" in model_source:
        raise RuntimeError("Obsolete H3 value clone is present in ComfyUI 0.43.2 model.py")
    if "# H3-T4-VAE-DTYPE: decoder input matches decoder parameters" not in vae_source:
        raise RuntimeError("H3 VAE decoder dtype patch marker is missing")
    if "decoder_parameter.dtype" not in vae_source:
        raise RuntimeError("H3 VAE decoder dtype guard is incomplete")
    if "memory_usage_factor = 0.17" not in supported_source:
        raise RuntimeError("MiniMax H3 memory_usage_factor must be 0.17 for the T4 runtime")
    if "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]" not in supported_source:
        raise RuntimeError("MiniMax H3 FP16 inference dtype is not registered")
    print("COMFYUI RUNTIME: PASS", tagged, head)

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

    check_main_pytorch()

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

from __future__ import annotations

import ast
import os
import json
import re
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Version consistency: configs/runtime_versions.yaml + requirements.txt own every
# version. Code derives from them; these checks never repeat a version literal.
# ---------------------------------------------------------------------------
_LOCK_SEMVER = re.compile(r"\d+\.\d+\.\d+")
_LOCK_CUDA_TAG = re.compile(r"cu(\d{2})(\d)")
_LOCK_STRATEGIES = {"eager", "lazy", "prefetch", "torchao"}

# Code that must read versions from the manifests instead of embedding them.
_LOCK_DERIVED_SOURCES = (
    "kaggle/bootstrap.py",
    "kaggle/preflight_h3.py",
    "kaggle/verify_kaggle_layout.py",
    "kaggle/verify_live_runtime.py",
    "planner/config.py",
    "planner/qwen_director_runtime.py",
    "scripts/validate_runtime_contract.py",
    "scripts/validate_enhancements.py",
)


def _lock_load_runtime(root: Path) -> dict:
    return yaml.safe_load((Path(root) / "configs" / "runtime_versions.yaml").read_text(encoding="utf-8")) or {}


def _lock_cuda_runtime_from_tag(tag: str) -> str:
    match = _LOCK_CUDA_TAG.fullmatch(str(tag).strip().lower())
    if not match:
        raise RuntimeError(f"CUDA wheel tag must look like cu132, got {tag!r}.")
    return f"{match.group(1)}.{match.group(2)}"


def _lock_requirement_pins(root: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    text = (Path(root) / "requirements.txt").read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if "==" not in line:
            continue
        name, version = [part.strip() for part in line.split("==", 1)]
        if name and version:
            pins[name.lower().replace("_", "-")] = version.split(";", 1)[0].strip()
    return pins


def _lock_compact(version: str, parts: int | None = None) -> str:
    items = str(version).split(".")
    return "".join(items[:parts] if parts else items)


def _lock_expected_director_env_name(runtime: dict, pins: dict[str, str]) -> str:
    director = runtime.get("director", {}) or {}
    return (
        f"h3_qwen_vllm_{_lock_compact(director.get('vllm_version', ''), 2)}"
        f"_t{_lock_compact(director.get('vllm_torch_version', ''), 2)}{str(director.get('vllm_cuda', '')).lower()}"
        f"_tf{_lock_compact(pins.get('transformers', ''), 2)}"
        f"_tok{_lock_compact(pins.get('tokenizers', ''))}"
    )


def _lock_check_manifests(root: Path, runtime: dict | None = None) -> list[str]:
    """Return a list of inconsistencies between runtime_versions.yaml and requirements.txt."""
    root = Path(root)
    runtime = runtime if runtime is not None else _lock_load_runtime(root)
    pins = _lock_requirement_pins(root)
    problems: list[str] = []

    pytorch = runtime.get("pytorch", {}) or {}
    cuda = str(pytorch.get("cuda", "")).strip().lower()
    if not _LOCK_CUDA_TAG.fullmatch(cuda):
        problems.append(f"pytorch.cuda must be a cuNNN wheel tag, got {cuda!r}")
    if not _LOCK_SEMVER.fullmatch(str(pytorch.get("version", ""))):
        problems.append(f"pytorch.version must be X.Y.Z, got {pytorch.get('version')!r}")
    index = str(pytorch.get("index", "")).rstrip("/")
    if not index.endswith(f"/{cuda}"):
        problems.append(f"pytorch.index must end with /{cuda}, got {index!r}")
    for key in ("torchvision_version", "torchaudio_version"):
        value = str(pytorch.get(key, ""))
        if not re.fullmatch(rf"\d+\.\d+\.\d+\+{re.escape(cuda)}", value):
            problems.append(f"pytorch.{key} must be X.Y.Z+{cuda} (same CUDA build as torch), got {value!r}")
    driver = str(pytorch.get("minimum_driver_version", ""))
    if not re.fullmatch(r"\d+(\.\d+)*", driver):
        problems.append(f"pytorch.minimum_driver_version must be numeric, got {driver!r}")

    cuda_cfg = runtime.get("cuda_runtime", {}) or {}
    runtime_version = str(cuda_cfg.get("runtime_version", ""))
    tag_match = _LOCK_CUDA_TAG.fullmatch(cuda)
    if tag_match:
        digits = cuda[2:]
        tag_major, tag_minor = digits[:2], digits[2:]
        if not re.fullmatch(rf"{tag_major}\.{int(tag_minor)}\.\d+", runtime_version):
            problems.append(
                f"cuda_runtime.runtime_version must be a {tag_major}.{int(tag_minor)}.x release "
                f"(the CUDA library build PyTorch {cuda} depends on), got {runtime_version!r}"
            )
        cublas_version = str(cuda_cfg.get("cublas_version", ""))
        if not re.fullmatch(rf"{tag_major}\.\d+(\.\d+)+", cublas_version):
            problems.append(f"cuda_runtime.cublas_version must start with CUDA major {tag_major}, got {cublas_version!r}")

    sage = runtime.get("sage_attention", {}) or {}
    sage_cuda = str(sage.get("build_cuda", "")).strip().lower()
    if sage_cuda != cuda:
        problems.append(f"sage_attention.build_cuda must match pytorch.cuda ({cuda}), got {sage_cuda!r}")
    sage_toolkit = str(sage.get("build_toolkit_package_version", ""))
    if not _LOCK_SEMVER.fullmatch(sage_toolkit):
        problems.append(f"sage_attention.build_toolkit_package_version must be X.Y.Z, got {sage_toolkit!r}")
    sage_component = str(sage.get("build_toolkit_component_version", ""))
    if not _LOCK_SEMVER.fullmatch(sage_component):
        problems.append(f"sage_attention.build_toolkit_component_version must be X.Y.Z, got {sage_component!r}")
    if runtime_version and sage_component != runtime_version:
        problems.append(
            "sage_attention.build_toolkit_component_version must match cuda_runtime.runtime_version "
            f"({runtime_version}), got {sage_component!r}"
        )

    director = runtime.get("director", {}) or {}
    for key in ("vllm_version", "vllm_torch_version"):
        if not _LOCK_SEMVER.fullmatch(str(director.get(key, ""))):
            problems.append(f"director.{key} must be X.Y.Z, got {director.get(key)!r}")
    if not _LOCK_CUDA_TAG.fullmatch(str(director.get("vllm_cuda", "")).strip().lower()):
        problems.append(f"director.vllm_cuda must be a cuNNN wheel tag, got {director.get('vllm_cuda')!r}")
    if str(director.get("vllm_torch_version", "")) != str(pytorch.get("version", "")):
        problems.append("director.vllm_torch_version must match pytorch.version for the aligned one-Torch target")
    if str(director.get("vllm_cuda", "")).strip().lower() != cuda:
        problems.append("director.vllm_cuda must match pytorch.cuda for the aligned one-CUDA target")
    wheel_build_cuda = str(director.get("vllm_wheel_build_cuda", "")).strip().lower()
    install_mode = str(director.get("vllm_install_mode", "")).strip().lower()
    if not _LOCK_CUDA_TAG.fullmatch(wheel_build_cuda):
        problems.append("director.vllm_wheel_build_cuda must be a cuNNN wheel-build tag")
    if install_mode != "stable_abi_no_deps":
        problems.append("director.vllm_install_mode must be stable_abi_no_deps")
    strategy = str(director.get("safetensors_load_strategy", "")).strip().lower()
    if strategy not in _LOCK_STRATEGIES:
        problems.append(
            f"director.safetensors_load_strategy must be one of {sorted(_LOCK_STRATEGIES)}, got {strategy!r}"
        )

    for name in ("transformers", "tokenizers", "psutil", "pyyaml", "gradio"):
        if not _LOCK_SEMVER.fullmatch(pins.get(name, "")):
            problems.append(f"requirements.txt must contain one exact {name}==X.Y.Z pin")
    if "vllm_non_torch_requirements" in director:
        problems.append(
            "director.vllm_non_torch_requirements must not duplicate wheel metadata; "
            "the bootstrap must read Requires-Dist from the installed vLLM wheel"
        )
    env_dir = str(director.get("vllm_env_dir", "")).strip()
    if pins.get("transformers") and pins.get("tokenizers") and _LOCK_SEMVER.fullmatch(str(director.get("vllm_version", ""))):
        expected_name = _lock_expected_director_env_name(runtime, pins)
        if Path(env_dir).name != expected_name:
            problems.append(
                f"director.vllm_env_dir must be named {expected_name} (encodes vLLM/transformers/tokenizers "
                f"versions so a version change always creates a fresh environment), got {Path(env_dir).name!r}"
            )

    storyboard = runtime.get("storyboard", {}) or {}
    if str(storyboard.get("gradio_version", "")) != pins.get("gradio", ""):
        problems.append(
            f"storyboard.gradio_version={storyboard.get('gradio_version')!r} must equal "
            f"requirements.txt gradio=={pins.get('gradio')!r}"
        )
    if not _LOCK_SEMVER.fullmatch(str(storyboard.get("pillow_version", ""))):
        problems.append(f"storyboard.pillow_version must be X.Y.Z, got {storyboard.get('pillow_version')!r}")
    if "pillow" in pins and pins["pillow"] != str(storyboard.get("pillow_version", "")):
        problems.append("requirements.txt Pillow pin must equal storyboard.pillow_version (or be omitted)")

    comfy = runtime.get("comfyui", {}) or {}
    if not _LOCK_SEMVER.fullmatch(str(comfy.get("expected_version", ""))):
        problems.append(f"comfyui.expected_version must be X.Y.Z, got {comfy.get('expected_version')!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(comfy.get("revision", ""))):
        problems.append("comfyui.revision must be a full 40-character commit SHA")
    return problems


def _lock_manifest_literals(root: Path, runtime: dict | None = None) -> dict[str, str]:
    """Every version string owned by the manifests, mapped to where it is defined."""
    runtime = runtime if runtime is not None else _lock_load_runtime(root)
    pins = _lock_requirement_pins(root)
    pytorch = runtime.get("pytorch", {}) or {}
    director = runtime.get("director", {}) or {}
    storyboard = runtime.get("storyboard", {}) or {}
    sage = runtime.get("sage_attention", {}) or {}
    found = {
        str(pytorch.get("version", "")): "pytorch.version",
        str(pytorch.get("cuda", "")): "pytorch.cuda",
        str(pytorch.get("torchvision_version", "")): "pytorch.torchvision_version",
        str(pytorch.get("torchaudio_version", "")): "pytorch.torchaudio_version",
        str(director.get("vllm_version", "")): "director.vllm_version",
        str(director.get("vllm_torch_version", "")): "director.vllm_torch_version",
        str(director.get("vllm_cuda", "")): "director.vllm_cuda",
        str(director.get("vllm_wheel_build_cuda", "")): "director.vllm_wheel_build_cuda",
        str(storyboard.get("gradio_version", "")): "storyboard.gradio_version",
        str(storyboard.get("pillow_version", "")): "storyboard.pillow_version",
        str(sage.get("build_toolkit_package_version", "")): "sage_attention.build_toolkit_package_version",
        str(sage.get("build_toolkit_component_version", "")): "sage_attention.build_toolkit_component_version",
    }
    for name in ("transformers", "tokenizers", "psutil"):
        if pins.get(name):
            found[pins[name]] = f"requirements.txt {name}"
    return {value: origin for value, origin in found.items() if value}


def _lock_check_no_repinned_versions(root: Path, runtime: dict | None = None) -> list[str]:
    """Fail when code repeats a manifest version instead of reading it."""
    root = Path(root)
    literals = _lock_manifest_literals(root, runtime)
    problems: list[str] = []
    for relative in _LOCK_DERIVED_SOURCES:
        path = root / relative
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in literals:
                problems.append(
                    f"{relative}:{node.lineno} hard-codes {node.value!r}; read {literals[node.value]} instead"
                )
        for match in re.finditer(r"torch\.version\.cuda == '\d+\.\d+'", source):
            line = source.count("\n", 0, match.start()) + 1
            problems.append(f"{relative}:{line} hard-codes a CUDA runtime version; derive it from the cu tag")
    return problems


def _lock_require_consistent(root: Path, runtime: dict | None = None) -> None:
    problems = _lock_check_manifests(root, runtime) + _lock_check_no_repinned_versions(root, runtime)
    if problems:
        raise RuntimeError("Runtime version lock is inconsistent:\n- " + "\n- ".join(problems))




def _validate_custom_node_contract(manifest: dict, model_inventory: dict) -> None:
    custom = dict(manifest.get("custom_nodes", {}) or {})
    required = list(custom.get("required", []) or [])
    supporting = list(custom.get("supporting", []) or [])
    nodes = required + supporting
    if not nodes:
        raise RuntimeError("custom_nodes.yaml contains no custom nodes.")

    policy = dict(manifest.get("policy", {}) or {})
    expected_policy = {
        "vendor_source_code": False,
        "clone_at_runtime": True,
        "third_party_models": False,
        "locked_project_model_inventory": True,
        "pin_revisions": True,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) is not expected:
            raise RuntimeError(
                f"custom_nodes.yaml policy.{key} must be {expected!r}."
            )

    names = {}
    for node in nodes:
        if not isinstance(node, dict):
            raise RuntimeError("custom_nodes.yaml node entries must be mappings.")
        name = str(node.get("name", "") or "").strip()
        repository = str(node.get("repository", "") or "").strip()
        revision = str(node.get("revision", "") or "").strip()
        if not name or not repository:
            raise RuntimeError("Every custom node requires name and repository.")
        if name in names:
            raise RuntimeError(f"Duplicate custom node declaration: {name}")
        names[name] = node
        if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise RuntimeError(
                f"Custom node {name!r} must use a full 40-character revision SHA."
            )
        deps = node.get("depends_on", []) or []
        if not isinstance(deps, list) or any(not str(value).strip() for value in deps):
            raise RuntimeError(f"Custom node {name!r} depends_on must be a list of names.")
        model_directory = str(node.get("model_directory", "") or "").strip()
        if model_directory:
            declared_dirs = {
                str(item.get("directory", "") or "").strip()
                for item in (model_inventory.get("models", {}) or {}).values()
                if isinstance(item, dict)
            }
            if model_directory not in declared_dirs:
                raise RuntimeError(
                    f"Custom node {name!r} references undeclared model directory {model_directory!r}."
                )

    indegree = {name: 0 for name in names}
    dependents = {name: [] for name in names}
    for name, node in names.items():
        for dependency in (str(value).strip() for value in node.get("depends_on", []) or []):
            if dependency not in names:
                raise RuntimeError(
                    f"Custom node {name!r} depends on unknown node {dependency!r}."
                )
            indegree[name] += 1
            dependents[dependency].append(name)
    ready = sorted(name for name, degree in indegree.items() if degree == 0)
    visited = []
    while ready:
        name = ready.pop(0)
        visited.append(name)
        for dependent in sorted(dependents[name]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort()
    if len(visited) != len(names):
        cyclic = sorted(name for name, degree in indegree.items() if degree > 0)
        raise RuntimeError(
            "Custom-node dependency cycle detected: " + ", ".join(cyclic)
        )


def _validate_control_plane_requirements() -> None:
    requirements_path = ROOT / "requirements.txt"
    requirements = {}
    for raw in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "==" not in line:
            continue
        name, version = [part.strip() for part in line.split("==", 1)]
        if name and version:
            requirements[name.lower().replace("_", "-")] = version.split(";", 1)[0].strip()

    # requirements.txt owns these versions; they only have to be exact pins here.
    # _lock_require_consistent() checks them against runtime_versions.yaml.
    for name, expected in (("transformers", None), ("tokenizers", None), ("psutil", None),
                           ("pyyaml", "6.0.3"), ("requests", "2.34.2")):
        if expected is not None and requirements.get(name) != expected:
            raise RuntimeError(f"requirements.txt must pin {name}=={expected}; got {requirements.get(name)!r}.")
        if not _LOCK_SEMVER.fullmatch(requirements.get(name, "")):
            raise RuntimeError(f"requirements.txt must contain one exact {name}==X.Y.Z pin; got {requirements.get(name)!r}.")

    # The warm probe must enforce the same package pins as the cold install.
    bootstrap_source = (ROOT / "kaggle/bootstrap.py").read_text(encoding="utf-8")
    probe_start = bootstrap_source.index("def _probe_director_venv(")
    probe_end = bootstrap_source.index("def _sage_build_stamp_path(", probe_start)
    warm_probe = bootstrap_source[probe_start:probe_end]
    required = (
        '_exact_requirement_pin("transformers")',
        '_exact_requirement_pin("tokenizers")',
        "transformers.__version__",
        "tokenizers.__version__",
        "from transformers import Qwen3Config",
        "sys.version_info[:2] == (3, 12)",
        "torch.__version__",
        "torch.version.cuda ==",
        "from vllm.config.load import LoadConfig",
        "safetensors_load_strategy",
        "include-system-site-packages",
        "vllm._C_stable_libtorch",
        "silu_and_mul",
        "torchvision",
        "torchaudio",
    )
    if any(item not in warm_probe for item in required):
        raise RuntimeError("Bootstrap warm vLLM reuse must verify Python 3.12, vLLM load API, Torch/CUDA, Transformers/tokenizers and Qwen3 imports.")
    for diagnostic in (
        "def _requirements_diagnostics(",
        "def _print_probe_failure(",
        "[DIRECTOR CACHE MISS]",
        "[PYTORCH CACHE MISS]",
        "[SAGE CACHE MISS]",
    ):
        if diagnostic not in bootstrap_source:
            raise RuntimeError(f"Bootstrap cache-reuse diagnostics are missing: {diagnostic}")

def _validate_requirements_diagnostics_behavior() -> None:
    """Exercise the bootstrap dependency checker, including extras and markers."""
    import tempfile
    from pathlib import Path
    from kaggle.bootstrap import _requirements_diagnostics

    with tempfile.TemporaryDirectory() as tmp:
        manifest = Path(tmp) / "requirements.txt"
        manifest.write_text("requests[socks]==2.34.2\n", encoding="utf-8")
        issues = _requirements_diagnostics(manifest)
        if not any("optional extras [socks]" in item for item in issues):
            raise RuntimeError(
                "Requirements diagnostics must conservatively reject an unverified optional extra."
            )

        manifest.write_text("package-that-does-not-exist-h3-audit==1.0.0\n", encoding="utf-8")
        issues = _requirements_diagnostics(manifest)
        if not any("not installed" in item for item in issues):
            raise RuntimeError("Requirements diagnostics must detect a missing distribution.")

        manifest.write_text(
            "package-that-does-not-exist-h3-audit==1.0.0; python_version < '0'\n",
            encoding="utf-8",
        )
        issues = _requirements_diagnostics(manifest)
        if issues:
            raise RuntimeError(f"An inactive environment marker must not force a cache miss: {issues!r}")

        manifest.write_text("-r nested-requirements.txt\n", encoding="utf-8")
        issues = _requirements_diagnostics(manifest)
        if not any("pip option/directive" in item for item in issues):
            raise RuntimeError("Included requirement files must conservatively force a cache miss.")


def _bootstrap_main_call_names(main: ast.FunctionDef) -> list[tuple[int, str]]:
    """Return recognized main() call targets in lexical source order.

    The production bootstrap wraps stage calls in _run_timed(...), so the
    validator must inspect wrapped calls instead of requiring bare
    install_*() expression statements.
    """
    recognized: list[tuple[int, str]] = []
    for node in ast.walk(main):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue

        if node.func.id in {"_run_timed", "_timed_call"} and len(node.args) >= 2:
            target = node.args[1]
            if isinstance(target, ast.Name):
                recognized.append((node.lineno, target.id))
            continue

        recognized.append((node.lineno, node.func.id))

    return sorted(recognized, key=lambda item: item[0])


def _validate_bootstrap_order() -> None:
    bootstrap_path = ROOT / "kaggle/bootstrap.py"
    tree = ast.parse(bootstrap_path.read_text(encoding="utf-8"), filename=str(bootstrap_path))
    main = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
        None,
    )
    if main is None:
        raise RuntimeError("kaggle/bootstrap.py main() is missing.")

    calls = _bootstrap_main_call_names(main)
    call_names = [name for _, name in calls]
    try:
        nodes_index = call_names.index("install_nodes")
        base_index = call_names.index("install_base_requirements")
        torch_index = call_names.index("install_pytorch_runtime")
    except ValueError as exc:
        raise RuntimeError(
            "Bootstrap dependency/Torch install calls are missing. "
            "The validator recognizes both direct calls and _run_timed(...) wrappers."
        ) from exc

    if not nodes_index < base_index < torch_index:
        raise RuntimeError(
            "Bootstrap must run the final control-plane dependency install after custom nodes and before locked PyTorch."
        )


def _validate_model_policy_contract(model_inventory: dict) -> None:
    policy = dict(model_inventory.get("policy", {}) or {})
    expected = {
        "exact_inventory": True,
        "allow_extra_production_models": False,
        "allow_q4_models": False,
        "allow_director_qwen": True,
        "allow_additional_qwen_models": False,
    }
    for key, value in expected.items():
        if policy.get(key) is not value:
            raise RuntimeError(
                f"model_inventory.yaml policy.{key} must be {value!r}."
            )


def _validate_custom_node_ordering() -> None:
    from kaggle.bootstrap import _ordered_custom_nodes

    manifest = {
        "custom_nodes": {
            "required": [
                {"name": "consumer", "depends_on": ["provider"]},
                {"name": "independent", "depends_on": []},
            ],
            "supporting": [
                {"name": "provider", "depends_on": []},
            ],
        }
    }
    ordered = [item["name"] for item in _ordered_custom_nodes(manifest)]
    if ordered != ["independent", "provider", "consumer"]:
        raise RuntimeError(
            "Custom-node dependency ordering is not deterministic or dependency-safe."
        )

    cyclic = {
        "custom_nodes": {
            "required": [
                {"name": "a", "depends_on": ["b"]},
                {"name": "b", "depends_on": ["a"]},
            ]
        }
    }
    try:
        _ordered_custom_nodes(cyclic)
    except RuntimeError as exc:
        if "cycle" not in str(exc).lower():
            raise RuntimeError(
                "Custom-node cycle detection failed with the wrong error."
            ) from exc
    else:
        raise RuntimeError("Custom-node dependency cycles must fail closed.")


def _validate_runtime_configuration_contract(runtime: dict) -> None:
    comfy = dict(runtime.get("comfyui", {}) or {})
    director = dict(runtime.get("director", {}) or {})
    python = dict(runtime.get("python", {}) or {})
    cuda_runtime = dict(runtime.get("cuda_runtime", {}) or {})
    h3_optimization = dict(runtime.get("h3_optimization", {}) or {})
    retired_config = {
        "comfyui.h3_t4_value_clone_workaround": "h3_t4_value_clone_workaround" in comfy,
        "cuda_runtime.runtime_package": "runtime_package" in cuda_runtime,
        "cuda_runtime.cublas_package": "cublas_package" in cuda_runtime,
        "h3_optimization.sparse_attention": "sparse_attention" in h3_optimization,
    }
    active_retired = [name for name, present in retired_config.items() if present]
    if active_retired:
        raise RuntimeError(
            "Runtime manifest contains retired, unconsumed configuration fields: "
            + ", ".join(active_retired)
        )
    pytorch = dict(runtime.get("pytorch", {}) or {})
    if comfy.get("h3_vae_decoder_dtype_patch") is not True:
        raise RuntimeError("The H3 VAE dtype guard must remain enabled.")
    # Versions live ONLY in runtime_versions.yaml + requirements.txt. Check that those agree with each
    # other (torch/torchvision/torchaudio CUDA build, index URL, gradio, Pillow, env-dir name, loader
    # strategy) and that no source file re-pins them, instead of repeating literals in this validator.
    _lock_require_consistent(ROOT, runtime)
    if str(python.get("kaggle", "")) != "3.12":
        raise RuntimeError("Kaggle runtime Python must be 3.12.")
    if director.get("tensor_parallel_size") != 2 or director.get("gpu_memory_utilization") != 0.82 or director.get("max_num_seqs") != 1:
        raise RuntimeError("Director TP=2, GPU memory=0.82, max_num_seqs=1 must match the validated 2xT4 operating point.")
    if director.get("speculative_method") != "eagle3" or director.get("speculative_tokens") != 2:
        raise RuntimeError("Director EAGLE-3 speculative settings must be preserved.")
    if director.get("context") != 10240 or director.get("story_max_tokens") != 6500:
        raise RuntimeError("Director context/story global ceilings must be preserved.")
    story = dict(director.get("story_contract", {}) or {})
    writing = dict(director.get("story_sampling", {}).get("writing", {}) or {})
    if story.get("minimum_words", 0) < 1400 or story.get("accept_min_words") != story.get("minimum_words"):
        raise RuntimeError("The 1,400+ word single-pass story contract must be preserved.")
    if writing.get("max_tokens") != 3000:
        raise RuntimeError("Director story Writer token ceiling must remain 3,000.")
    if story.get("scene_count") != 12 or story.get("shots_per_scene") != 3:
        raise RuntimeError("12 scenes x 3 shots story topology must be preserved.")
    if runtime.get("generation", {}).get("frames_per_shot") != 141:
        raise RuntimeError("Current 141-frame shot topology must be preserved; do not revert to 124.")


def _validate_director_runtime_source() -> None:
    path = ROOT / "planner/qwen_director_runtime.py"
    source = path.read_text(encoding="utf-8")
    for marker in (
        "DIRECTOR_VLLM_SAFETENSORS_LOAD_STRATEGY",
        "def _h3_director_child_environment()",
        "child_env = _h3_director_child_environment()",
        "--safetensors-load-strategy",
        "DIRECTOR_VLLM_VERSION",
        '"/version"',
        "def _stage_to_local_scratch(",
        "def _parallel_copy_file(",
        "def _resolve_safetensors_strategy(",
        "def prestage_director_checkpoints(",
        "def prewarm_director_server(",
        "apply_chat_template(",
        '"stop_reason": (str(stop_reason) if stop_reason is not None else "")',
        'stop_reason = choice.get("stop_reason")',
        "vLLM 0.33 API supports top_k",
    ):
        if marker not in source:
            raise RuntimeError(f"Director runtime source is missing required vLLM Director contract: {marker}")
    # Exercise telemetry serialization: vendor-specific stop_reason must survive
    # into the per-call record instead of being discarded after finish_reason.
    from contextlib import redirect_stdout
    from io import StringIO
    from planner.qwen_director_runtime import QwenDirectorRuntimeMixin
    probe = object.__new__(QwenDirectorRuntimeMixin)
    probe._qwen_telemetry = {
        "calls": [], "total_elapsed_seconds": 0.0, "prompt_tokens": 0,
        "completion_tokens": 0, "retries": 0, "cache_hits": 0,
        "deterministic_recoveries": 0,
    }
    with redirect_stdout(StringIO()):
        probe._record_qwen_call(
            call_name="telemetry_probe", elapsed=0.1, finish_reason="stop", stop_reason=123,
        )
    record = probe._qwen_telemetry["calls"][0]
    if record.get("finish_reason") != "stop" or record.get("stop_reason") != "123":
        raise RuntimeError(f"Qwen stop reason was not preserved in telemetry: {record!r}")

    # Staging is the fast path: /kaggle/input is GCSFUSE and direct loading took 6+ minutes per shard.
    if 'os.getenv("H3_DIRECTOR_STAGE_LOCAL", "auto")' not in source:
        raise RuntimeError("Model staging must default to auto (H3_DIRECTOR_STAGE_LOCAL=0 forces direct loading).")


def _validate_single_pass_story_writer(runtime: dict) -> None:
    """Lock the requested 1,400-word, 12-scene, one-author-call contract."""
    director = dict(runtime.get("director", {}) or {})
    contract = dict(director.get("story_contract", {}) or {})
    sampling = dict(director.get("story_sampling", {}) or {})
    writing = dict(sampling.get("writing", {}) or {})
    architecture = dict(director.get("story_architecture", {}) or {})
    if int(director.get("story_max_tokens", 0) or 0) != 6500:
        raise RuntimeError("Global Director story ceiling must remain 6,500 tokens.")
    if int(architecture.get("max_tokens", 0) or 0) != 5400:
        raise RuntimeError("Story Architect ceiling must remain 5,400 tokens.")
    if int(writing.get("max_tokens", 0) or 0) != 3000:
        raise RuntimeError("Single-pass Story Author ceiling must be 3,000 tokens.")
    if int(writing.get("minimum_output_tokens", 0) or 0) != 0:
        raise RuntimeError("Story Author minimum_output_tokens must remain 0 so EOS is not artificially delayed.")
    if int(contract.get("minimum_words", 0) or 0) < 1400 or int(contract.get("accept_min_words", 0) or 0) != int(contract.get("minimum_words", 0) or 0):
        raise RuntimeError("Story acceptance must enforce the 1,400-word minimum with no lower secondary floor.")
    if int(contract.get("scene_count", 0) or 0) != 12 or int(contract.get("shots_per_scene", 0) or 0) != 3:
        raise RuntimeError("Story topology must remain exactly 12 scenes x 3 shots.")
    if int(contract.get("target_words", 0) or 0) != int(contract.get("scene_count", 0) or 0) * int(contract.get("words_per_scene", 0) or 0):
        raise RuntimeError("Story target must equal scene count multiplied by per-scene guidance.")

    director_path = ROOT / "planner/qwen_director.py"
    director_source = director_path.read_text(encoding="utf-8")
    tree = ast.parse(director_source, filename=str(director_path))
    author = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_generate_story_once_impl"), None)
    wrapper = next((node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_generate_story_once"), None)
    if author is None or wrapper is None:
        raise RuntimeError("Single-pass story author implementation or rejection wrapper is missing.")
    if not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "_generate_story_once_impl" for node in ast.walk(wrapper)):
        raise RuntimeError("Author wrapper must call exactly one single-pass implementation.")
    author_calls = [
        node for node in ast.walk(author)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_chat_text"
    ]
    if len(author_calls) != 1:
        raise RuntimeError(f"Story author must make exactly one _chat_text call; found {len(author_calls)}.")
    if any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(author)):
        raise RuntimeError("Story author must not loop to retry or repair the story.")

    runtime_path = ROOT / "planner/qwen_director_runtime.py"
    runtime_source = runtime_path.read_text(encoding="utf-8")
    runtime_tree = ast.parse(runtime_source, filename=str(runtime_path))
    post = next((node for node in ast.walk(runtime_tree) if isinstance(node, ast.FunctionDef) and node.name == "_post_chat"), None)
    if post is None:
        raise RuntimeError("Qwen runtime _post_chat method is missing.")
    requests = [
        node for node in ast.walk(post)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "post"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_vllm_session"
    ]
    if len(requests) != 1:
        raise RuntimeError(f"One model completion must issue exactly one HTTP POST; found {len(requests)}.")
    if "sampling_extra_fallback" in runtime_source:
        raise RuntimeError("Qwen runtime still retries with weakened sampling parameters.")
    print("PASS single-pass story contract: 1,400-word floor, 12x3 topology, 3,000-token Author, one HTTP request")


def _validate_sage_setup_patch_behavior() -> None:
    """Verify the pinned-fork CUDA bypass is removed and C++20 flags are applied."""
    import tempfile
    from pathlib import Path
    from kaggle.bootstrap import _patch_sage_cxx_standard

    sample = (
        "import torch\n"
        "torch.utils.cpp_extension._check_cuda_version = lambda *a, **kw: None\n"
        "CXX_FLAGS = [\"-std=c++17\"]\n"
        "NVCC_FLAGS = [\"-std=c++17\"]\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        setup_py = Path(tmp) / "setup.py"
        setup_py.write_text(sample, encoding="utf-8")
        _patch_sage_cxx_standard(Path(tmp))
        patched = setup_py.read_text(encoding="utf-8")
        if "_check_cuda_version = lambda" in patched:
            raise RuntimeError("Sage setup patch left the CUDA-version-check bypass in place.")
        if patched.count("-std=c++20") != 2 or "-std=c++17" in patched:
            raise RuntimeError("Sage setup patch did not consistently apply the C++20 compiler flags.")


def _validate_sage_toolchain_probe_behavior() -> None:
    """Exercise warm/cold CUDA pip-toolkit selection and mismatched-config rejection."""
    import importlib.util
    import sys
    import tempfile
    import types
    from pathlib import Path
    from unittest.mock import patch

    runtime_cfg = _lock_load_runtime(ROOT)
    pytorch_cuda_tag = str(runtime_cfg["pytorch"]["cuda"])
    sage_cfg = runtime_cfg["sage_attention"]
    toolkit_package_version = str(sage_cfg["build_toolkit_package_version"])
    component_version = str(sage_cfg["build_toolkit_component_version"])
    cuda_release = _lock_cuda_runtime_from_tag(pytorch_cuda_tag)
    toolkit_digits = component_version.split(".")

    spec = importlib.util.spec_from_file_location("h3_bootstrap_sage_toolchain_test", ROOT / "kaggle" / "bootstrap.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load bootstrap for CUDA toolchain tests.")
    bootstrap = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bootstrap
    spec.loader.exec_module(bootstrap)

    expected = {
        package: component_version
        for package in (
            "nvidia-cuda-nvcc",
            "nvidia-cuda-crt",
            "nvidia-cuda-cccl",
            "nvidia-cuda-runtime",
            "nvidia-nvvm",
        )
    }
    runtime = {
        "pytorch": {"cuda": pytorch_cuda_tag},
        "sage_attention": {
            "build_cuda": pytorch_cuda_tag,
            "build_toolkit_package_version": toolkit_package_version,
            "build_toolkit_component_version": component_version,
        },
    }

    with tempfile.TemporaryDirectory() as tmp:
        nvidia_root = Path(tmp) / "nvidia"
        cuda_home = nvidia_root / "cu13"
        (cuda_home / "bin").mkdir(parents=True)
        (cuda_home / "include").mkdir()
        (cuda_home / "lib").mkdir()
        (cuda_home / "include" / "cuda.h").write_text("// test header", encoding="utf-8")
        (cuda_home / "include" / "cuda_runtime.h").write_text("// test header", encoding="utf-8")
        (cuda_home / "lib" / f"libcudart.so.{component_version}").write_bytes(b"test library")
        nvcc = cuda_home / "bin" / "nvcc"
        nvcc.write_text(
            "#!/bin/sh\necho 'Cuda compilation tools, release "
            + cuda_release
            + ", V"
            + component_version
            + "'\n",
            encoding="utf-8",
        )
        nvcc.chmod(0o755)

        fake_nvidia = types.ModuleType("nvidia")
        fake_nvidia.__path__ = [str(nvidia_root)]
        old_nvidia = sys.modules.get("nvidia")
        sys.modules["nvidia"] = fake_nvidia
        try:
            with patch.object(bootstrap.importlib_metadata, "version", side_effect=lambda name: expected[name]):
                env = {}
                selected = bootstrap._prepare_sage_cuda_build_toolchain(runtime, env)
                bootstrap._describe_cuda_toolchain(runtime, env)
                if selected != cuda_home or env.get("CUDA_HOME") != str(cuda_home):
                    raise RuntimeError("Sage build did not select the CUDA pip toolkit root.")
                if not (cuda_home / "lib" / "libcudart.so").is_symlink():
                    raise RuntimeError("Sage build did not create the CUDA cudart linker name.")

            # The cold-cache branch must request the manifest-pinned toolkit package.
            call_counts = {}
            commands = []

            def package_version(name: str):
                call_counts[name] = call_counts.get(name, 0) + 1
                return None if call_counts[name] == 1 else expected[name]

            with patch.object(bootstrap.importlib_metadata, "version", side_effect=package_version):
                with patch.object(bootstrap, "run", side_effect=lambda *args, **kwargs: commands.append(args)):
                    env = {}
                    bootstrap._prepare_sage_cuda_build_toolchain(runtime, env)
            wanted_requirement = (
                f"cuda-toolkit[nvcc,crt,cccl,cudart,nvvm]=={toolkit_package_version}"
            )
            if not commands or not any(wanted_requirement in str(part) for part in commands[0]):
                raise RuntimeError("Cold Sage toolkit probe did not install the manifest-pinned CUDA toolkit extras.")
        finally:
            if old_nvidia is None:
                sys.modules.pop("nvidia", None)
            else:
                sys.modules["nvidia"] = old_nvidia
            sys.modules.pop(spec.name, None)

    bad = {
        "pytorch": {"cuda": pytorch_cuda_tag},
        "sage_attention": {"build_cuda": pytorch_cuda_tag + "-mismatch"},
    }
    try:
        bootstrap._prepare_sage_cuda_build_toolchain(bad, {})
    except RuntimeError as exc:
        if "must match main PyTorch CUDA" not in str(exc):
            raise
    else:
        raise RuntimeError("Sage toolkit probe accepted a CUDA build tag that differs from main Torch.")


def _validate_vllm_wheel_metadata_reader(bootstrap_module) -> None:
    """Exercise wheel Requires-Dist parsing and the no-Torch/no-extra filter."""
    metadata_text = """Metadata-Version: 2.1
Name: vllm
Version: 0.33.0
Requires-Dist: torch ==2.14.0
Requires-Dist: torchvision ==0.29.0
Requires-Dist: torchaudio ==2.11.0
Requires-Dist: requests >=2.28
Requires-Dist: fastapi[standard] >=0.133; python_version >= \"3.10\"
Requires-Dist: win-only-dependency; sys_platform == \"win32\"
Requires-Dist: extra-only-dependency; extra == \"testing\"
"""
    with tempfile.TemporaryDirectory(prefix="h3-vllm-metadata-test-") as temporary:
        site = Path(temporary)
        dist_info = site / "vllm-0.33.0.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(metadata_text, encoding="utf-8")
        test_env = dict(os.environ)
        inherited = test_env.get("PYTHONPATH", "")
        test_env["PYTHONPATH"] = str(site) + (os.pathsep + inherited if inherited else "")
        requirements = bootstrap_module._read_installed_vllm_requirements(Path(sys.executable), test_env)
    normalized = [item.lower().replace("_", "-") for item in requirements]
    if not any(item.startswith("requests") for item in normalized):
        raise RuntimeError("vLLM metadata parser dropped a base Requires-Dist dependency.")
    if not any(item.startswith("fastapi[standard]") for item in normalized):
        raise RuntimeError("vLLM metadata parser dropped an active PEP 508 dependency.")
    forbidden_fragments = ("torch", "torchvision", "torchaudio", "vllm", "win-only-dependency", "extra-only-dependency")
    if any(any(fragment in item for fragment in forbidden_fragments) for item in normalized):
        raise RuntimeError("vLLM metadata parser retained a separately managed or inactive dependency.")
    print("[PASS] vLLM wheel metadata dependency filtering")


def _validate_bootstrap_upgrade_source() -> None:
    path = ROOT / "kaggle/bootstrap.py"
    source = path.read_text(encoding="utf-8")
    for forbidden in (
        "condition_proj(text_states.to(torch.float32))",
        "low_precision_attention=False",
        "# H3-T4-WORKAROUND: removed redundant V clone for SM75",
        "MINIMAX_H3_PROJECT_T4_MEMORY_COMPAT_V3",
    ):
        if forbidden in source:
            raise RuntimeError(f"Bootstrap contains obsolete H3 compatibility logic: {forbidden}")
    if "import re\n" not in source:
        raise RuntimeError("Bootstrap uses regular expressions and must import re explicitly.")
    if "def _validate_h3_vae_decoder_source" not in source:
        raise RuntimeError("Bootstrap is missing the AST-level H3 VAE decoder postcondition.")
    for marker in (
        "vllm_install_mode",
        "vllm_wheel_build_cuda",
        "_read_installed_vllm_requirements",
        "importlib.metadata as metadata",
        "pip._vendor.packaging.requirements",
        "pip._vendor.packaging.markers",
        "metadata.requires(\"vllm\")",
        'environment["extra"] = ""',
        'blocked = {"vllm", "torch", "torchvision", "torchaudio"}',
        "--no-deps",
        "--constraints",
        "--index-url",
        "--extra-index-url",
        "--index-strategy",
        "_director_isolated_environment",
        "vllm._C_stable_libtorch",
        "silu_and_mul",
        "_patch_sage_cxx_standard(install_dir)",
        "_prepare_sage_cuda_build_toolchain(runtime, build_env)",
        "cuda-toolkit[nvcc,crt,cccl,cudart,nvvm]",
        "nvidia-cuda-nvcc",
        "nvidia-cuda-crt",
        "nvidia-cuda-cccl",
        "nvidia-nvvm",
        "CUDA_HOME",
        "Refusing to fall back to the system CUDA compiler",
        '"-std=c++20"',
        "vllm=={vllm_version}",
        "from vllm.config.load import LoadConfig",
        "safetensors_load_strategy",
        "include-system-site-packages",
        "vllm._C_stable_libtorch",
        "silu_and_mul",
        "torchvision",
        "torchaudio",
        "423cbd20511becb1f16122ab518e92e58bfa03d9",
        "memory_usage_factor = 0.17",
        "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]",
        "# H3-T4-VAE-DTYPE: decoder input matches decoder parameters",
        "h3_embedding._validate_upstream_forward(MiniMaxH3Model._forward)",
        "from sageattention import _qattn_sm75 as _sage_sm75",
        "from sageattention import core as _sage_core",
        'getattr(_sage_core, "SM75_ENABLED", False)',
        'getattr(_sage_core, "FUSED_ENABLED", False)',
        "qk_int8_sv_f16_accum_f32_attn_sm75",
        "per_warp_int8_cuda",
    ):
        if marker not in source:
            raise RuntimeError(f"Bootstrap is missing required runtime upgrade contract: {marker}")

    preflight_path = ROOT / "kaggle/preflight_h3.py"
    preflight_source = preflight_path.read_text(encoding="utf-8")
    if '"from vllm.config import SpeculativeConfig; "' not in preflight_source:
        raise RuntimeError("Kaggle Director preflight references SpeculativeConfig without importing it.")
    if "assert 'eagle3' in str(inspect.signature(SpeculativeConfig)); " not in preflight_source:
        raise RuntimeError("Kaggle Director preflight must verify the EAGLE3 SpeculativeConfig API.")
    if "torchvision, torchaudio" in preflight_source:
        raise RuntimeError("Kaggle Director preflight must check TorchVision/TorchAudio package metadata without requiring untested imports.")

    installer_start = source.index("def install_director_runtime(")
    installer_end = source.index("def install_storyboard_runtime(", installer_start)
    installer = source[installer_start:installer_end]
    ordered_steps = (
        "torch_stack_install = subprocess.run(",
        "vllm_install = subprocess.run(",
        "vllm_dependency_requirements = _read_installed_vllm_requirements(",
        "TemporaryDirectory(prefix=\"h3-director-torch-constraints-\")",
        "dependency_install = subprocess.run(",
        "if not _probe_director_venv(runtime, env_dir):",
    )
    offsets = [installer.find(step) for step in ordered_steps]
    if any(offset < 0 for offset in offsets) or offsets != sorted(offsets):
        raise RuntimeError("Director install must install the final Torch stack once, install the vLLM wheel with --no-deps, read its dependency metadata, install dependencies under Torch constraints, then validate Stable ABI.")
    for obsolete in ("--torch-backend", "wheel_torch_version", "wheel_cuda_tag", "Stage 1:", "Stage 2:", "torch_upgrade = subprocess.run(", "vllm_non_torch_requirements"):
        if obsolete in installer:
            raise RuntimeError(f"Director installer still contains the redundant two-stage Torch installation: {obsolete}")

    from kaggle import bootstrap as bootstrap_module
    test_env = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "/contaminating/project/path",
        "PYTHONHOME": "/contaminating/python",
        "VIRTUAL_ENV": "/other/venv",
        "CONDA_PREFIX": "/other/conda",
        "CUDA_HOME": "/opt/cuda-13.2",
        "CUDA_PATH": "/opt/cuda-13.2",
        "CUDAHOSTCXX": "/host/compiler",
        "CC": "/host/cc",
        "CXX": "/host/cxx",
        "CMAKE_ARGS": "-DHOST=1",
        "VLLM_CACHE_ROOT": "/cache/vllm",
    }
    clean_env = bootstrap_module._director_isolated_environment(test_env)
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX"):
        if key in clean_env:
            raise RuntimeError(f"Director environment isolation leaked {key}.")
    for key in ("PATH", "CUDA_HOME", "CUDA_PATH", "CUDAHOSTCXX", "CC", "CXX", "CMAKE_ARGS", "VLLM_CACHE_ROOT"):
        if clean_env.get(key) != test_env[key]:
            raise RuntimeError(f"Director environment isolation unexpectedly changed {key}.")


def main() -> int:
    runtime_path = ROOT / "configs" / "runtime_versions.yaml"
    manifest_path = ROOT / "configs" / "custom_nodes.yaml"
    model_path = ROOT / "configs" / "model_inventory.yaml"

    for path in (runtime_path, manifest_path, model_path):
        if not path.is_file():
            raise RuntimeError(f"Missing manifest: {path}")
        yaml.safe_load(path.read_text(encoding="utf-8"))

    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    model_inventory = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    _validate_custom_node_contract(manifest, model_inventory)
    _validate_model_policy_contract(model_inventory)
    _validate_custom_node_ordering()
    _validate_control_plane_requirements()
    _validate_sage_setup_patch_behavior()
    _validate_sage_toolchain_probe_behavior()
    _validate_bootstrap_order()
    _validate_runtime_configuration_contract(runtime)
    _validate_single_pass_story_writer(runtime)
    _validate_director_runtime_source()
    _validate_bootstrap_upgrade_source()
    from kaggle import bootstrap as bootstrap_module
    _validate_vllm_wheel_metadata_reader(bootstrap_module)
    _validate_requirements_diagnostics_behavior()
    director = dict(runtime.get("director", {}) or {})
    if str(director.get("backend", "") or "").strip().lower() != "vllm":
        raise RuntimeError("Director backend must be vllm.")
    if not str(director.get("model_path", "") or "").strip():
        raise RuntimeError("Director model_path is required in runtime_versions.yaml.")
    if not str(director.get("vllm_version", "") or "").strip():
        raise RuntimeError("Director vllm_version is required in runtime_versions.yaml.")
    if int(director.get("tensor_parallel_size", 0) or 0) <= 0:
        raise RuntimeError("Director tensor_parallel_size must be positive in runtime_versions.yaml.")
    if not str(director.get("vllm_env_dir", "") or "").strip():
        raise RuntimeError("Director vllm_env_dir is required in runtime_versions.yaml.")
    speculative_method = str(director.get("speculative_method", "") or "").strip().lower()
    if speculative_method != "eagle3":
        raise RuntimeError("Director speculative_method must be eagle3 in runtime_versions.yaml.")
    speculative_model_path = str(director.get("speculative_model_path", "") or "").strip()
    if speculative_model_path != "/kaggle/input/eagle-3":
        raise RuntimeError("Director speculative_model_path must be /kaggle/input/eagle-3 in runtime_versions.yaml.")
    if int(director.get("speculative_tokens", 0) or 0) <= 0:
        raise RuntimeError("Director speculative_tokens must be positive in runtime_versions.yaml.")
    if not str(director.get("generation_config", "") or "").strip():
        raise RuntimeError("Director generation_config is required in runtime_versions.yaml.")
    comfy = runtime.get("comfyui", {})
    required = ("repository", "revision", "expected_version")
    missing = [key for key in required if not str(comfy.get(key, "") or "").strip()]
    if missing:
        raise RuntimeError("Missing ComfyUI runtime fields: " + ", ".join(missing))

    workflows = [
        ROOT / "workflows/generation/H3_Ref2VA_Production.json",
        ROOT / "workflows/generation/H3_Turbo_Ref2VA_Production.json",
        ROOT / "workflows/postprocess/H3_Ref2VA_UltimateUpscale_Production.json",
    ]
    for workflow in workflows:
        data = json.loads(workflow.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not data.get("nodes"):
            raise RuntimeError(f"Invalid workflow root: {workflow}")

    qwen_runtime_path = ROOT / "planner/qwen_director_runtime.py"
    qwen_runtime_source = qwen_runtime_path.read_text(encoding="utf-8")

    bootstrap_path = ROOT / "kaggle/bootstrap.py"
    bootstrap_source = bootstrap_path.read_text(encoding="utf-8")
    bootstrap_tree = ast.parse(bootstrap_source, filename=str(bootstrap_path))
    for forbidden in ("--disable-log-requests", "--no-enable-log-requests"):
        if forbidden in qwen_runtime_source:
            raise RuntimeError(f"Pinned vLLM runtime must not pass obsolete flag {forbidden}.")
    bootstrap_defs = {
        node.name
        for node in ast.walk(bootstrap_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for required_function in ("install_comfyui", "verify_runtime_files"):
        if required_function not in bootstrap_defs:
            raise RuntimeError(
                f"bootstrap contract missing function definition: {required_function}"
            )
    for required_token in ("git", "checkout"):
        if required_token not in bootstrap_source:
            raise RuntimeError(f"bootstrap contract missing: {required_token}")

    print("Runtime install contract PASSED.")
    print(f"ComfyUI lock: {comfy['revision']} ({comfy['expected_version']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

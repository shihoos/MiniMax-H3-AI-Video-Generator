from __future__ import annotations

import ast
import json
import re
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

    expected_pins = {"transformers": "5.17.0", "tokenizers": "0.23.3", "psutil": "7.0.0"}
    for name, expected in expected_pins.items():
        if requirements.get(name) != expected:
            raise RuntimeError(
                f"requirements.txt must pin {name}=={expected}; got {requirements.get(name)!r}."
            )

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
        "torch.version.cuda == '13.0'",
        "from vllm.config.load import LoadConfig",
        "safetensors_load_strategy",
    )
    if any(item not in warm_probe for item in required):
        raise RuntimeError("Bootstrap warm vLLM reuse must verify Python 3.12, vLLM load API, Torch/CUDA, Transformers/tokenizers and Qwen3 imports.")

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

        if node.func.id == "_run_timed" and len(node.args) >= 2:
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


def _validate_runtime_upgrade_contract(runtime: dict) -> None:
    comfy = dict(runtime.get("comfyui", {}) or {})
    director = dict(runtime.get("director", {}) or {})
    python = dict(runtime.get("python", {}) or {})
    pytorch = dict(runtime.get("pytorch", {}) or {})
    if comfy.get("expected_version") != "0.43.2":
        raise RuntimeError("ComfyUI must be pinned to v0.43.2.")
    if comfy.get("revision") != "423cbd20511becb1f16122ab518e92e58bfa03d9":
        raise RuntimeError("ComfyUI must use the exact 0.43.2 release commit.")
    if comfy.get("h3_t4_value_clone_workaround") is not False:
        raise RuntimeError("The obsolete H3 V-clone workaround must be disabled.")
    if comfy.get("h3_vae_decoder_dtype_patch") is not True:
        raise RuntimeError("The H3 VAE dtype guard must remain enabled.")
    if director.get("vllm_version") != "0.32.0":
        raise RuntimeError("vLLM must be pinned to 0.32.0.")
    if director.get("vllm_torch_version") != "2.13.0" or director.get("vllm_cuda") != "cu130":
        raise RuntimeError("Isolated vLLM environment must use Torch 2.13.0/cu130.")
    if director.get("safetensors_load_strategy") != "eager":
        raise RuntimeError("Eager safetensors loading must be the default strategy.")
    if director.get("vllm_env_dir") != "/tmp/h3_qwen_vllm_032_tf517_tok0233":
        raise RuntimeError("vLLM 0.32 must use a fresh, version-specific isolated environment.")
    if str(python.get("kaggle", "")) != "3.12":
        raise RuntimeError("Kaggle runtime Python must be 3.12.")
    if pytorch.get("version") != "2.10.0" or pytorch.get("cuda") != "cu130":
        raise RuntimeError("ComfyUI main process Torch must remain 2.10.0/cu130.")
    if director.get("tensor_parallel_size") != 2 or director.get("gpu_memory_utilization") != 0.9 or director.get("max_num_seqs") != 1:
        raise RuntimeError("Director TP=2, GPU memory=0.9, max_num_seqs=1 must be preserved.")
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
        "--safetensors-load-strategy",
        "DIRECTOR_VLLM_VERSION",
        '"/version"',
        "def _use_direct_checkpoint(",
        "def _begin_startup_metrics(",
        "def _finish_startup_metrics(",
        "apply_chat_template(",
    ):
        if marker not in source:
            raise RuntimeError(f"Director runtime source is missing required v0.32 contract: {marker}")
    forbidden = (
        "_stage_to_local_scratch",
        "H3_DIRECTOR_STAGE_LOCAL",
        "/tmp/staged_models",
        "ThreadPoolExecutor",
        "shutil.copytree",
        "shutil.copy2",
    )
    for marker in forbidden:
        if marker in source:
            raise RuntimeError(f"Director checkpoint staging must be removed: {marker}")
    if "model_staging\": False" not in source and '"model_staging": False' not in source:
        raise RuntimeError("Director startup telemetry must record model_staging=false.")


def _validate_bootstrap_upgrade_source() -> None:
    path = ROOT / "kaggle/bootstrap.py"
    source = path.read_text(encoding="utf-8")
    for marker in (
        "--torch-backend",
        "vllm=={vllm_version}",
        "from vllm.config.load import LoadConfig",
        "safetensors_load_strategy",
        "423cbd20511becb1f16122ab518e92e58bfa03d9",
        "memory_usage_factor = 0.17",
        "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]",
        "# H3-T4-VAE-DTYPE: decoder input matches decoder parameters",
        "h3_embedding._validate_upstream_forward(MiniMaxH3Model._forward)",
    ):
        if marker not in source:
            raise RuntimeError(f"Bootstrap is missing required runtime upgrade contract: {marker}")
    for marker in (
        "condition_proj(text_states.to(torch.float32))",
        "low_precision_attention=False",
        "# H3-T4-WORKAROUND: removed redundant V clone for SM75",
        "MINIMAX_H3_PROJECT_" + "T4_MEMORY_COMPAT_V3",
    ):
        if marker in source:
            raise RuntimeError(f"Obsolete ComfyUI 0.34 H3 compatibility logic remains: {marker}")


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
    _validate_bootstrap_order()
    _validate_runtime_upgrade_contract(runtime)
    _validate_director_runtime_source()
    _validate_bootstrap_upgrade_source()
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
    if "--disable-log-requests" in qwen_runtime_source:
        raise RuntimeError(
            "Obsolete vLLM --disable-log-requests flag is forbidden in the Director command."
        )
    if "--no-enable-log-requests" in qwen_runtime_source:
        raise RuntimeError(
            "The pinned Director runtime must not depend on the optional "
            "--no-enable-log-requests flag; request logging is already disabled by default."
        )
    bootstrap_tree = ast.parse(bootstrap_source, filename=str(bootstrap_path))
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

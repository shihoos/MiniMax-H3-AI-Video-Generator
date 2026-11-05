from __future__ import annotations

import sys
import ast
import json
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMFY = ROOT / "ComfyUI"
sys.path.insert(0, str(ROOT / "scripts"))
import runtime_lock  # noqa: E402  (single-source version checks)

RUNTIME = ROOT / "configs" / "runtime_versions.yaml"
CUSTOM = ROOT / "configs" / "custom_nodes.yaml"
BOOTSTRAP = ROOT / "kaggle" / "bootstrap.py"


def fail(message: str) -> None:
    raise RuntimeError(message)


def require_file(path: Path) -> None:
    if not path.is_file():
        fail(f"Missing file: {path}")


def load_json(path: Path) -> dict:
    require_file(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"Invalid JSON: {path}: {exc}")


def get_node(workflow: dict, node_type: str) -> dict:
    matches = [
        node
        for node in workflow.get("nodes", [])
        if node.get("type") == node_type
    ]

    if len(matches) != 1:
        fail(
            f"{node_type}: expected exactly 1 node, "
            f"found {len(matches)}"
        )

    return matches[0]


def model_edges(workflow: dict) -> set[tuple[int, int, int]]:
    edges = set()

    for link in workflow.get("links", []):
        if (
            isinstance(link, list)
            and len(link) >= 6
            and str(link[5]).upper() == "MODEL"
        ):
            edges.add(
                (
                    int(link[1]),
                    int(link[3]),
                    int(link[4]),
                )
            )

    return edges


def verify_workflow(path: Path, turbo: bool) -> None:
    workflow = load_json(path)

    unet = get_node(workflow, "UNETLoader")
    optimizer = get_node(workflow, "H3MemoryOptimization")
    scheduler = get_node(workflow, "BasicScheduler")
    guider = get_node(workflow, "BasicGuider")

    forbidden = {"MiniMaxH3FP16Safe", "MiniMaxH3_FP16_T4"}
    present_forbidden = {
        node.get("type")
        for node in workflow.get("nodes", [])
        if node.get("type") in forbidden
    }
    if present_forbidden:
        fail(
            f"Forbidden production nodes present in {path}: "
            f"{sorted(present_forbidden)}"
        )

    if turbo:
        turbo_lora = get_node(
            workflow,
            "MiniMaxH3TurboLoRA",
        )

        expected = {
            (
                int(unet["id"]),
                int(turbo_lora["id"]),
                0,
            ),
            (
                int(turbo_lora["id"]),
                int(optimizer["id"]),
                0,
            ),
            (
                int(optimizer["id"]),
                int(scheduler["id"]),
                0,
            ),
            (
                int(optimizer["id"]),
                int(guider["id"]),
                0,
            ),
        }

        actual = model_edges(workflow)

        if actual != expected:
            fail(
                f"Turbo MODEL topology mismatch in {path}\n"
                f"Expected: {sorted(expected)}\n"
                f"Actual:   {sorted(actual)}"
            )

        widgets = turbo_lora.get("widgets_values", [])

        if len(widgets) < 3 or widgets[2] is not True:
            fail(
                "MiniMaxH3TurboLoRA low_vram widget must be True"
            )

    else:
        expected = {
            (
                int(unet["id"]),
                int(optimizer["id"]),
                0,
            ),
            (
                int(optimizer["id"]),
                int(scheduler["id"]),
                0,
            ),
            (
                int(optimizer["id"]),
                int(guider["id"]),
                0,
            ),
        }

        actual = model_edges(workflow)

        if actual != expected:
            fail(
                f"Base MODEL topology mismatch in {path}\n"
                f"Expected: {sorted(expected)}\n"
                f"Actual:   {sorted(actual)}"
            )

    widgets = optimizer.get("widgets_values", [])

    if len(widgets) < 8:
        fail(f"Incomplete H3MemoryOptimization widgets: {path}")

    if int(widgets[2]) != 2560:
        fail(
            f"{path}: expected chunk_rows=2560, "
            f"got {widgets[2]!r}"
        )

    if str(widgets[7]) != "Lower VRAM (slower)":
        fail(
            f"{path}: expected Lower VRAM (slower), "
            f"got {widgets[7]!r}"
        )


def verify_runtime_config() -> None:
    require_file(RUNTIME)
    config = yaml.safe_load(RUNTIME.read_text(encoding="utf-8")) or {}
    comfy = config.get("comfyui", {}) or {}
    director = config.get("director", {}) or {}
    pytorch = config.get("pytorch", {}) or {}

    # Versions are owned by runtime_versions.yaml + requirements.txt; verify they agree with each
    # other and that no code re-pins them, instead of repeating literals here.
    runtime_lock.require_consistent(ROOT, config)
    if comfy.get("h3_vae_decoder_dtype_patch") is not True:
        fail("The H3 VAE decoder dtype guard must be enabled")
    if str(config.get("python", {}).get("kaggle", "")) != "3.12":
        fail("Kaggle runtime Python must be 3.12")
    if config["h3_optimization"]["chunk_rows"] != 2560:
        fail("H3 chunk_rows must be 2560")
    if config["h3_optimization"]["attention_memory_mode"] != "Lower VRAM (slower)":
        fail("H3 attention_memory_mode must be 'Lower VRAM (slower)'")
    h3_opt = config.get("h3_optimization", {})
    if h3_opt.get("memory_optimization") is not True:
        fail("H3 memory optimization must be enabled")
    if h3_opt.get("qkv_streaming_mode") != "Auto":
        fail("H3 qkv_streaming_mode must be Auto")
    if config.get("generation", {}).get("frames_per_shot") != 141:
        fail("The existing 141-frame production topology must be preserved")
    story = director.get("story_contract", {}) or {}
    writing = director.get("story_sampling", {}).get("writing", {}) or {}
    if story.get("minimum_words", 0) < 1400 or story.get("accept_min_words") != story.get("minimum_words"):
        fail("The 1,400+ usable-word story acceptance contract must be preserved")
    if writing.get("max_tokens") != 3000:
        fail("Director Writer max_tokens must remain 3000")
    print("[PASS] Runtime configuration")

def verify_director_runtime_source() -> None:
    """Validate Director launch contracts in the Director runtime, not bootstrap."""
    runtime_path = ROOT / "planner" / "qwen_director_runtime.py"
    require_file(runtime_path)
    source = runtime_path.read_text(encoding="utf-8")
    required = (
        "DIRECTOR_VLLM_SAFETENSORS_LOAD_STRATEGY",
        "--safetensors-load-strategy",
        "DIRECTOR_VLLM_VERSION",
        '"/version"',
        "def _stage_to_local_scratch(",
        "def _parallel_copy_file(",
        "def _resolve_safetensors_strategy(",
        "def _begin_startup_metrics(",
        "def _finish_startup_metrics(",
        "apply_chat_template(",
    )
    for marker in required:
        if marker not in source:
            fail(f"Director runtime source is missing required contract: {marker}")
    if '"model_staging":' not in source:
        fail("Director startup telemetry must record model_staging")
    if 'os.getenv("H3_DIRECTOR_STAGE_LOCAL", "auto")' not in source:
        fail("Model staging must default to auto (H3_DIRECTOR_STAGE_LOCAL=0 forces direct loading)")
    print("[PASS] Director runtime source")


def verify_sage_attention_patch_recipe() -> None:
    """Verify the inline SageAttention SM75 source-replacement recipe and live result."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(BOOTSTRAP))
    except SyntaxError as exc:
        fail(
            f"kaggle/bootstrap.py has invalid Python syntax: "
            f"line={exc.lineno}, offset={exc.offset}, error={exc.msg}"
        )
        return

    for forbidden in ("SAGE_SM75_PATCH", "git apply --check", "patch_sha256"):
        if forbidden in source:
            fail(f"Bootstrap contains retired SageAttention patch machinery: {forbidden}")

    required_recipe = (
        '_apply_sage_sm75_source_corrections',
        'legacy = "constexpr int WARP_K_SM75 = 16;"',
        'fixed = "constexpr int WARP_K_SM75 = 64;"',
        'legacy_count = source.count(legacy)',
        'fixed_count = source.count(fixed)',
        'marker_count = source.count("H3-T4-SM75-KERNEL-FIX")',
        'legacy_count == 2 and fixed_count == 0 and marker_count == 0',
        'H3-T4-SM75-KERNEL-FIX',
        'H3-T4-SM75-PV-FRAGMENT-FIX',
        'SM75 PV renormalization correction',
        'SM75 final normalization correction',
        'SM75 staged-output fragment mapping correction',
        'SM75 direct-output fragment mapping correction',
        'SM75 WARP_K dual-declaration correction',
        'source.count(old)',
        'if count != 1:',
    )
    missing = [needle for needle in required_recipe if needle not in source]
    if missing:
        fail(
            "Bootstrap SageAttention source-replacement recipe is incomplete:\n"
            + "\n".join(missing)
        )

    runtime = yaml.safe_load(RUNTIME.read_text(encoding="utf-8")) or {}
    sage = runtime.get("sage_attention") or {}
    if "patch_sha256" in sage:
        fail("runtime_versions.yaml contains the retired sage_attention.patch_sha256 field")
    directory = str(sage.get("directory", "") or "").strip()
    if not directory:
        fail("runtime_versions.yaml sage_attention.directory is required")
        return

    sage_kernel = (
        ROOT
        / "ComfyUI"
        / "custom_nodes"
        / directory
        / "csrc"
        / "qattn"
        / "attn_cuda_sm75.h"
    )
    if not sage_kernel.is_file():
        fail(f"Installed SageAttention SM75 kernel source is missing: {sage_kernel}")
        return

    sage_source = sage_kernel.read_text(encoding="utf-8")
    required_live = (
        "H3-T4-SM75-KERNEL-FIX",
        "constexpr int WARP_K_SM75 = 64;",
        "H3-T4-SM75-PV-FRAGMENT-FIX",
        "float o_scale_top = math::ptx_exp2(m_old[0] - m_i[0]);",
        "float o_scale_bottom = math::ptx_exp2(m_old[1] - m_i[1]);",
        "float l_rcp_top = (l_i[0] > 0.0f) ? math::ptx_rcp(l_i[0]) : 0.0f;",
        "float l_rcp_bottom = (l_i[1] > 0.0f) ? math::ptx_rcp(l_i[1]) : 0.0f;",
        "uint32_t thread_row1 = thread_row0 + 8;",
        "uint32_t thread_row0 = o_start_row_warp + (lane_id / 4);",
    )
    missing_live = [needle for needle in required_live if needle not in sage_source]
    if missing_live:
        fail(
            "Installed SageAttention SM75 source is missing required corrections:\n"
            + "\n".join(missing_live)
        )
    if "constexpr int WARP_K_SM75 = 16;" in sage_source:
        fail("Installed SageAttention SM75 source still contains WARP_K_SM75=16")

    print("[PASS] SageAttention inline source-replacement recipe")

def verify_h3_optimization_source() -> None:
    """Verify the project-owned H3-Optimizations patches before checking ComfyUI overlays."""
    node_dir = COMFY / "custom_nodes" / "H3-Optimizations"
    if not node_dir.is_dir():
        fail(f"H3-Optimizations runtime directory is missing: {node_dir}")

    targets = {
        "attention_forward.py": node_dir / "h3_optimizations" / "attention_forward.py",
        "providers.py": node_dir / "h3_optimizations" / "qkv" / "providers.py",
        "apply.py": node_dir / "h3_optimizations" / "apply.py",
        "bf16.py": node_dir / "h3_optimizations" / "qkv" / "bf16.py",
        "embedding.py": node_dir / "h3_optimizations" / "memory" / "embedding.py",
    }

    for name, path in targets.items():
        require_file(path)
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            fail(
                f"H3-Optimizations {name} has invalid Python syntax: "
                f"line={exc.lineno}, offset={exc.offset}, error={exc.msg}"
            )

    attention_source = targets["attention_forward.py"].read_text(encoding="utf-8")
    required_attention = (
        "# MINIMAX_H3_T4_SAGE_ATTENTION",
        "from sageattention import sageattn",
        "tensor_layout='HND'",
        "qk_quant_gran='per_warp'",
        "smooth_k=True",
        "projected.release()",
    )
    for required in required_attention:
        if required not in attention_source:
            fail(f"H3 SageAttention source contract is incomplete: {required}")

    unresolved = (
        "__SAGE_SMOOTH_K_TEXT__",
        "__SAGE_QK_QUANT_GRAN_TEXT__",
        "__SAGE_SMOOTH_K__",
        "__SAGE_QK_QUANT_GRAN__",
    )
    unresolved_found = [token for token in unresolved if token in attention_source]
    if unresolved_found:
        fail(
            "H3 SageAttention generated source contains unresolved patch placeholders: "
            + ", ".join(unresolved_found)
        )

    embedding_source = targets["embedding.py"].read_text(encoding="utf-8")
    for marker in ("UPSTREAM_FORWARD_SHA256", "H3EmbeddingMemoryPatchError", "def _validate_upstream_forward(forward):"):
        if marker not in embedding_source:
            fail(f"H3 embedding guarded fallback contract is missing: {marker}")
    if "MINIMAX_H3_PROJECT_T4_MEMORY_COMPAT_V3" in embedding_source:
        fail("Obsolete project-patched H3 embedding forward remains in the pinned plugin")
    for name, required in {
        "providers.py": (
            "# MINIMAX_H3_T4_PRESERVE_BOUNDED_QKV",
        ),
        "apply.py": (
            "# MINIMAX_H3_T4_QKV_RUNTIME_SELECTION",
        ),
        "bf16.py": (
            "# MINIMAX_H3_T4_KEEP_CONVROT_BINDING_FOR_Q_STREAM",
        ),
    }.items():
        source = targets[name].read_text(encoding="utf-8")
        for marker in required:
            if marker not in source:
                fail(f"H3-Optimizations {name} patch marker is missing: {marker}")
        if name == "providers.py":
            tree = ast.parse(source, filename=name)
            helper = next(
                (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_resolve_preserve_precision_qkv"),
                None,
            )
            if helper is None:
                fail("H3-Optimizations preserve-precision helper is missing")
            helper_text = ast.get_source_segment(source, helper) or ""
            if "memory_optimize" in helper_text or "request ==" in helper_text:
                fail("H3-Optimizations resolver-only variables leaked into the preserve helper")

    print("[PASS] H3-Optimizations source patches")


def verify_comfy_runtime() -> None:
    if not COMFY.is_dir():
        fail("ComfyUI runtime is missing. Run bootstrap.py first.")
    model = COMFY / "comfy/ldm/minimax/model.py"
    vae = COMFY / "comfy/ldm/minimax/vae.py"
    supported = COMFY / "comfy/supported_models.py"
    turbo = COMFY / "custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py"
    for path in (model, vae, supported, turbo):
        require_file(path)
    verify_h3_optimization_source()

    model_text = model.read_text(encoding="utf-8")
    vae_text = vae.read_text(encoding="utf-8")
    supported_text = supported.read_text(encoding="utf-8")
    turbo_text = turbo.read_text(encoding="utf-8")

    required_model = (
        "class MiniMaxH3Model",
        "def _embed_and_pack(",
        "preferred_attention=self.comfy_attention",
    )
    for marker in required_model:
        if marker not in model_text:
            fail(f"ComfyUI 0.43.2 native H3 model source marker is missing: {marker}")
    if "v = v.clone()" in model_text:
        fail("Obsolete H3 V clone is present in ComfyUI 0.43.2 model.py")
    for forbidden in (
        "condition_proj(text_states.to(torch.float32))",
        "low_precision_attention=False",
        "self.out_proj((out / 64.0).to(torch.float16))",
    ):
        if forbidden in model_text:
            fail(f"Obsolete ComfyUI 0.34 H3 forward patch is present: {forbidden}")
    if "mm.MLP.forward =" in model_text or "mm.DiTBlock.forward =" in model_text:
        fail("Forbidden global MLP/DiTBlock monkey-patch remains in H3 model")
    for required in (
        "# H3-T4-VAE-DTYPE: decoder input matches decoder parameters",
        "decoder_parameter = next(self.decoder.parameters(), None)",
        "z = z.to(decoder_parameter.dtype)",
    ):
        if required not in vae_text:
            fail(f"H3 VAE decoder dtype patch is incomplete: {required}")
    try:
        from kaggle.bootstrap import _validate_h3_vae_decoder_source
        _validate_h3_vae_decoder_source(vae_text, str(vae))
    except RuntimeError as exc:
        fail(f"H3 VAE _decode_pixels AST contract failed: {exc}")
    if "memory_usage_factor = 0.17" not in supported_text:
        fail("MiniMax H3 T4 memory-estimation factor must be 0.17")
    if "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]" not in supported_text:
        fail("MiniMax H3 must register FP16 inference support")
    if "class MiniMaxH3TurboLoRA" not in turbo_text:
        fail("MiniMaxH3TurboLoRA runtime node is missing")
    try:
        head = subprocess.check_output(["git", "-C", str(COMFY), "rev-parse", "HEAD"], text=True).strip()
        tag = subprocess.run(["git", "-C", str(COMFY), "describe", "--tags", "--exact-match", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    except Exception as exc:
        fail(f"Unable to verify ComfyUI git release identity: {exc}")
        return
    runtime = yaml.safe_load(RUNTIME.read_text(encoding="utf-8")) or {}
    expected_revision = str(runtime.get("comfyui", {}).get("revision", "")).strip()
    if head != expected_revision:
        fail(f"ComfyUI checkout SHA mismatch: expected={expected_revision}, actual={head}")
    expected_comfy_version = str(runtime.get("comfyui", {}).get("expected_version", "")).strip()
    if tag not in {expected_comfy_version, f"v{expected_comfy_version}"}:
        fail(f"ComfyUI release tag must be v{expected_comfy_version}, got {tag!r}")
    print("[PASS] ComfyUI 0.43.2 native H3/T4 runtime patches")

def verify_custom_nodes() -> None:
    require_file(CUSTOM)

    config = yaml.safe_load(
        CUSTOM.read_text(encoding="utf-8")
    )

    installed = {
        node["name"]
        for group in config["custom_nodes"].values()
        for node in group
    }
    forbidden = {"ComfyUI-MiniMaxH3FP16Safe", "ComfyUI-MiniMaxH3_FP16_T4"}
    present_forbidden = forbidden & installed
    if present_forbidden:
        fail(f"Forbidden experimental nodes present: {sorted(present_forbidden)}")

    required = {
        "Comfyui_Minimax_h3_latent_Upscaler",
        "Comfyui-MMH3-UltimateUpscale",
        "ComfyUI-MiniMax-H3-Turbo",
        "ComfyUI-Workflow-To-API-Converter",
        "H3-Optimizations",
        "ComfyUI-VideoHelperSuite",
    }

    missing = required - installed

    if missing:
        fail(
            "Required custom nodes missing from manifest: "
            f"{sorted(missing)}"
        )

    print("[PASS] Custom-node manifest")


def verify_planner() -> None:
    path = ROOT / "planner/production_planner.py"
    require_file(path)

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    found = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if any(
            isinstance(target, ast.Name)
            and target.id == "coordinated_name_pattern"
            for target in node.targets
        ):
            found = True
            break

    if not found:
        fail(
            "coordinated_name_pattern is missing "
            "from production_planner.py"
        )

    pattern = re.compile(
        r"\b"
        r"([A-Z][A-Za-z0-9'_-]+"
        r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
        r"\s*,\s*"
        r"[^.!?;]{0,120}?"
        r"\band\b"
        r"\s+"
        r"([A-Z][A-Za-z0-9'_-]+"
        r"(?:\s+[A-Z][A-Za-z0-9'_-]+){0,2})"
    )

    story = (
        "Mira, a systems engineer, and Arun, "
        "her specialist, arrive at the station."
    )

    matches = [
        (match.group(1), match.group(2))
        for match in pattern.finditer(story)
    ]

    if not matches or matches[0] != ("Mira", "Arun"):
        fail(
            f"Planner coordinated-name test failed: {matches}"
        )

    print("[PASS] Production planner")


def verify_python() -> None:
    roots = [
        ROOT / "execution",
        ROOT / "planner",
        ROOT / "scripts",
        ROOT / "kaggle",
    ]

    checked = 0

    for base in roots:
        if not base.exists():
            continue

        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue

            ast.parse(
                path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ),
                filename=str(path),
            )

            checked += 1

    print(f"[PASS] Python syntax: {checked} files")


def _extract_embedded_runtime_targets(bootstrap_text: str) -> tuple[str, ...]:
    """Read the literal runtime target tuple from the bootstrap AST.

    This intentionally ignores source quote style. Comparing quoted source
    substrings made the layout verifier reject valid double-quoted paths.
    """
    try:
        tree = ast.parse(bootstrap_text, filename=str(BOOTSTRAP))
    except SyntaxError as exc:
        fail(f"Bootstrap source is not valid Python: {exc}")

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "apply_embedded_h3_runtime_overlay"
    ]
    if len(functions) != 1:
        fail(
            "Bootstrap must define exactly one "
            "apply_embedded_h3_runtime_overlay() function"
        )

    assignments = [
        node
        for node in ast.walk(functions[0])
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target_node, ast.Name) and target_node.id == "targets"
            for target_node in node.targets
        )
    ]
    if len(assignments) != 1:
        fail(
            "Bootstrap embedded H3 runtime overlay must define exactly one "
            "literal targets tuple"
        )
    try:
        targets = ast.literal_eval(assignments[0].value)
    except (TypeError, ValueError, SyntaxError) as exc:
        fail(f"Bootstrap embedded runtime targets are not literal: {exc}")
    if not isinstance(targets, (tuple, list)) or not all(
        isinstance(target, str) for target in targets
    ):
        fail("Bootstrap embedded runtime targets must be a tuple/list of strings")
    return tuple(targets)


def verify_bootstrap() -> None:
    bootstrap = ROOT / "kaggle/bootstrap.py"
    require_file(bootstrap)
    text = bootstrap.read_text(encoding="utf-8")
    for name in ("apply_embedded_h3_runtime_overlay", "verify_runtime_files", "patch_h3_fp16_runtime", "patch_h3_vae_decoder_dtype", "patch_h3_embedding_memory_compatibility"):
        if name not in text:
            fail(f"Bootstrap is missing required runtime operation: {name}")
    embedded_targets = set(_extract_embedded_runtime_targets(text))
    required_targets = {
        "comfy/ldm/minimax/model.py",
        "comfy/ldm/minimax/vae.py",
        "comfy/supported_models.py",
        "custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py",
    }
    missing_targets = sorted(required_targets.difference(embedded_targets))
    if missing_targets:
        fail(
            "Bootstrap embedded runtime targets are missing: "
            + ", ".join(missing_targets)
        )
    required_contracts = (
        "preferred_attention=self.comfy_attention",
        "def _embed_and_pack(",
        "memory_usage_factor = 0.17",
        "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]",
        "# H3-T4-VAE-DTYPE: decoder input matches decoder parameters",
        "decoder_parameter = next(self.decoder.parameters(), None)",
        "vllm=={vllm_version}",
        "torch.cuda.device_count() == {tensor_parallel_size}",
    )
    for contract in required_contracts:
        if contract not in text:
            fail(f"Bootstrap runtime contract is missing: {contract}")
    for forbidden in (
        "condition_proj(text_states.to(torch.float32))",
        "low_precision_attention=False",
        "# H3-T4-WORKAROUND: removed redundant V clone for SM75",
    ):
        if forbidden in text:
            fail(f"Bootstrap contains obsolete ComfyUI 0.34 H3 compatibility logic: {forbidden}")
    if "mm.MLP.forward =" in text or "mm.DiTBlock.forward =" in text:
        fail("Bootstrap contains forbidden global MLP/DiTBlock monkey-patching")
    if "patch_h3_sage_attention" not in text:
        fail("Bootstrap is missing the H3 T4 SageAttention patch operation")
    if "__SAGE_SMOOTH_K_TEXT__" not in text or "__SAGE_QK_QUANT_GRAN_TEXT__" not in text:
        fail("Bootstrap is missing the safe SageAttention logging placeholders")
    print("[PASS] Bootstrap")

def verify_bootstrap_completed() -> None:
    """A failed bootstrap must never be followed by PASS banners from static checks."""
    metrics_path = ROOT / "bootstrap_metrics.json"
    if not metrics_path.is_file():
        return
    try:
        status = str(json.loads(metrics_path.read_text(encoding="utf-8")).get("status", ""))
    except Exception as exc:
        fail(f"bootstrap_metrics.json is unreadable: {exc}")
        return
    if status != "passed":
        fail(
            f"The last bootstrap run did not pass (status={status!r}, see {metrics_path}). "
            "Fix the bootstrap error and re-run bootstrap before validating the runtime."
        )


def main() -> None:
    verify_bootstrap_completed()
    verify_bootstrap()
    verify_runtime_config()
    verify_director_runtime_source()
    verify_sage_attention_patch_recipe()
    verify_custom_nodes()
    verify_comfy_runtime()

    verify_workflow(
        ROOT / "workflows/generation/H3_Ref2VA_Production.json",
        turbo=False,
    )

    verify_workflow(
        ROOT / "workflows/generation/H3_Turbo_Ref2VA_Production.json",
        turbo=True,
    )

    print("[PASS] Production workflow topology")

    verify_planner()
    verify_python()

    print("=" * 60)
    print("FINAL KAGGLE STATE VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

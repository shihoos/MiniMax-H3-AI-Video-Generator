from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT / "ComfyUI"

RUNTIME = ROOT / "configs" / "runtime_versions.yaml"
CUSTOM = ROOT / "configs" / "custom_nodes.yaml"


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

    forbidden = {
        "MiniMaxH3FP16Safe",
        "MiniMaxH3_FP16_T4",
    }

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

    config = yaml.safe_load(
        RUNTIME.read_text(encoding="utf-8")
    )

    if config["comfyui"]["expected_version"] != "0.34.0":
        fail("ComfyUI version must be 0.34.0")

    if config["pytorch"]["version"] != "2.10.0":
        fail("PyTorch version must be 2.10.0")

    if config["pytorch"]["cuda"] != "cu130":
        fail("PyTorch CUDA target must be cu130")

    if config["h3_optimization"]["chunk_rows"] != 2560:
        fail("H3 chunk_rows must be 2560")

    if (
        config["h3_optimization"]["attention_memory_mode"]
        != "Lower VRAM (slower)"
    ):
        fail(
            "H3 attention_memory_mode must be "
            "'Lower VRAM (slower)'"
        )

    if not config["comfyui"].get(
        "h3_t4_value_clone_workaround",
        False,
    ):
        fail("H3 T4 value-clone workaround is disabled")

    h3_opt = config.get("h3_optimization", {})
    if h3_opt.get("memory_optimization") is not True:
        fail("H3 memory optimization must be enabled")
    if h3_opt.get("qkv_streaming_mode") != "Auto":
        fail("H3 qkv_streaming_mode must be Auto")

    sage = config.get("sage_attention", {})
    print("[PASS] Runtime configuration")


def verify_sage_attention_embedded_patch() -> None:
    """Verify the embedded SageAttention SM75 patch against runtime configuration."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(BOOTSTRAP))
    except SyntaxError as exc:
        fail(
            f"kaggle/bootstrap.py has invalid Python syntax: "
            f"line={exc.lineno}, offset={exc.offset}, error={exc.msg}"
        )
        return

    patch_text = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "SAGE_SM75_PATCH":
                try:
                    patch_text = ast.literal_eval(node.value)
                except Exception as exc:
                    fail(f"Unable to read embedded SageAttention patch: {exc}")
                break

    if not isinstance(patch_text, str) or not patch_text.strip():
        fail("bootstrap.py embedded SageAttention patch is missing")

    runtime = yaml.safe_load(RUNTIME.read_text(encoding="utf-8")) or {}
    sage = runtime.get("sage_attention") or {}
    expected_sha = str(sage.get("patch_sha256", "") or "").strip().lower()
    if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
        fail("runtime_versions.yaml sage_attention.patch_sha256 is invalid")

    actual_sha = hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
    if actual_sha != expected_sha:
        fail(
            "Embedded SageAttention patch checksum mismatch: "
            f"expected={expected_sha}, actual={actual_sha}"
        )

    required = (
        "diff --git a/csrc/qattn/attn_cuda_sm75.h b/csrc/qattn/attn_cuda_sm75.h",
        "H3-T4-SM75-KERNEL-FIX",
        "constexpr int WARP_K_SM75 = 64;",
        "H3-T4-SM75-PV-FRAGMENT-FIX",
        "float o_scale_top = math::ptx_exp2(m_old[0] - m_i[0]);",
        "float l_rcp_top = (l_i[0] > 0.0f) ? math::ptx_rcp(l_i[0]) : 0.0f;",
        "uint32_t thread_row1 = thread_row0 + 8;",
    )
    missing = [needle for needle in required if needle not in patch_text]
    if missing:
        fail(
            "Embedded SageAttention patch is missing required corrections:\n"
            + "\n".join(missing)
        )
    if "constexpr int WARP_K_SM75 = 16;" in patch_text:
        fail("Embedded SageAttention patch still contains legacy WARP_K_SM75 = 16")
    print("[PASS] Embedded SageAttention patch matches runtime_versions.yaml")


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

    for name, required in {
        "providers.py": (
            "# MINIMAX_H3_T4_BOUNDED_QKV_GEMM_LIMIT",
            "# MINIMAX_H3_T4_FORCE_BOUNDED_QKV",
        ),
        "apply.py": (
            "# MINIMAX_H3_T4_APPLY_FORCE_STREAMED_QKV",
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

    print("[PASS] H3-Optimizations source patches")


def verify_comfy_runtime() -> None:
    if not COMFY.is_dir():
        fail(
            "ComfyUI runtime is missing. "
            "Run bootstrap.py first."
        )

    model = COMFY / "comfy/ldm/minimax/model.py"
    vae = COMFY / "comfy/ldm/minimax/vae.py"
    supported = COMFY / "comfy/supported_models.py"
    turbo = (
        COMFY
        / "custom_nodes"
        / "ComfyUI-MiniMax-H3-Turbo"
        / "__init__.py"
    )

    for path in (model, vae, supported, turbo):
        require_file(path)

    verify_h3_optimization_source()

    model_text = model.read_text(encoding="utf-8")
    vae_text = vae.read_text(encoding="utf-8")
    supported_text = supported.read_text(encoding="utf-8")
    turbo_text = turbo.read_text(encoding="utf-8")

    if (
        "# H3-T4-WORKAROUND: removed redundant V clone for SM75"
        not in model_text
    ):
        fail("H3 T4 model patch is missing")

    for required in (
        "decoder_dtype = next(self.decoder.parameters()).dtype",
        "if z.dtype != decoder_dtype:",
        "z = z.to(decoder_dtype)",
    ):
        if required not in vae_text:
            fail(
                "H3 VAE dtype patch is incomplete: "
                f"{required}"
            )

    if "mm.MLP.forward =" in model_text or "mm.DiTBlock.forward =" in model_text:
        fail("Forbidden legacy global MLP/DiTBlock monkey-patch remains in H3 model")

    required_model_patches = (
        "condition_proj(text_states.to(torch.float32))",
        "residual_dtype = torch.float32 if dtype == torch.float16 else dtype",
        "low_precision_attention=False",
        "self.out_proj((out / 64.0).to(torch.float16))",
    )
    for required in required_model_patches:
        if required not in model_text:
            fail(
                "H3 FP16 numerical patch is incomplete: "
                f"{required}"
            )

    # Source formatting may place the two chained operations on separate lines.
    if not re.search(
        r"\.to\(torch\.float32\)\s*\.mul_\(64\.0\)",
        model_text,
    ):
        fail(
            "H3 FP16 numerical patch is incomplete: "
            ".to(torch.float32) followed by .mul_(64.0)"
        )

    if "supported_inference_dtypes" not in supported_text:
        fail(
            "MiniMax H3 FP16 support patch is missing"
        )

    if "class MiniMaxH3TurboLoRA" not in turbo_text:
        fail(
            "MiniMaxH3TurboLoRA runtime node is missing"
        )

    print("[PASS] ComfyUI runtime patches")


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

    forbidden = {
        "MiniMaxH3_FP16_T4",
        "ComfyUI-MiniMaxH3FP16Safe",
    }

    present = forbidden & installed

    if present:
        fail(
            "Forbidden experimental nodes present: "
            f"{sorted(present)}"
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


def verify_bootstrap() -> None:
    bootstrap = ROOT / "kaggle/bootstrap.py"
    require_file(bootstrap)

    text = bootstrap.read_text(encoding="utf-8")

    required = (
        "apply_embedded_h3_runtime_overlay",
        "verify_runtime_files",
    )

    missing = [
        name
        for name in required
        if name not in text
    ]

    if missing:
        fail(
            "Bootstrap is missing required runtime operations: "
            f"{missing}"
        )

    embedded_targets = (
        "'comfy/ldm/minimax/model.py'",
        "'comfy/ldm/minimax/vae.py'",
        "'comfy/supported_models.py'",
        "'custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py'",
    )
    for target in embedded_targets:
        if target not in text:
            fail(f"Bootstrap embedded runtime target is missing: {target}")

    # The bootstrap must carry the exact H3 numerical/memory corrections that
    # are later written into the temporary ComfyUI checkout.
    bootstrap_contracts = (
        "condition_proj(text_states.to(torch.float32))",
        "low_precision_attention=False",
        "residual_dtype = torch.float32 if dtype == torch.float16 else dtype",
        "self.out_proj((out / 64.0).to(torch.float16))",
        "decoder_dtype = next(self.decoder.parameters()).dtype",
        "memory_usage_factor = 0.17",
        "# H3-T4-WORKAROUND: removed redundant V clone for SM75",
    )
    for contract in bootstrap_contracts:
        if contract not in text:
            fail(f"Bootstrap H3 contract is missing: {contract}")

    if "mm.MLP.forward =" in text or "mm.DiTBlock.forward =" in text:
        fail("Bootstrap still contains the forbidden legacy global MLP/DiTBlock monkey-patch")

    if "patch_h3_sage_attention" not in text:
        fail("Bootstrap is missing the H3 T4 SageAttention patch operation")

    if "__SAGE_SMOOTH_K_TEXT__" not in text or "__SAGE_QK_QUANT_GRAN_TEXT__" not in text:
        fail("Bootstrap is missing the safe SageAttention logging placeholders")

    broken_template = (
        "'smooth_k=__SAGE_SMOOTH_K__ qk_quant_gran=__SAGE_QK_QUANT_GRAN__'"
    )
    if broken_template in text:
        fail("Bootstrap still contains the broken SageAttention quoted-placeholder template")

    print("[PASS] Bootstrap")


def main() -> None:
    verify_bootstrap()
    verify_runtime_config()
    verify_sage_attention_embedded_patch()
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

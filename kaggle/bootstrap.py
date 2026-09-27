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
WRAPT_VERSION = "2.4.1"


def ensure_kaggle_startup_wrapt() -> None:
    """Provide Kaggle's sitecustomize dependency without changing the H3 stack."""
    probe = subprocess.run(
        [sys.executable, "-c", "import wrapt; print(wrapt.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    installed_version = (probe.stdout or "").strip() if probe.returncode == 0 else ""
    if installed_version != WRAPT_VERSION:
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-deps",
            f"wrapt=={WRAPT_VERSION}",
        )

    verify = subprocess.run(
        [sys.executable, "-c", "import wrapt; print(wrapt.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.returncode != 0:
        raise RuntimeError(
            "Kaggle wrapt verification failed.\n"
            + (verify.stdout or "")
            + (verify.stderr or "")
        )
    if "Error in sitecustomize" in (verify.stderr or ""):
        raise RuntimeError(
            "Kaggle sitecustomize still failed after installing wrapt.\n"
            + (verify.stderr or "")
        )
    verified_version = (verify.stdout or "").strip()
    if verified_version != WRAPT_VERSION:
        raise RuntimeError(
            f"Kaggle wrapt version mismatch: expected={WRAPT_VERSION}, "
            f"actual={verified_version!r}"
        )
    print(f"[KAGGLE STARTUP] wrapt={verified_version}: PASS")
    print("[KAGGLE STARTUP] sitecustomize: PASS")


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
def install_comfyui(runtime: dict) -> None:
    """Install the locked ComfyUI checkout and its dependencies."""
    config = dict(runtime.get("comfyui", {}) or {})
    repository = str(config.get("repository", "") or "").strip()
    revision = str(config.get("revision", "") or "").strip()
    expected_version = str(config.get("expected_version", "") or "").strip()
    if not repository or not revision:
        raise RuntimeError("runtime_versions.yaml comfyui.repository/revision are required.")

    COMFY.parent.mkdir(parents=True, exist_ok=True)
    if COMFY.exists() and not (COMFY / ".git").is_dir():
        raise RuntimeError(f"ComfyUI path exists but is not a git checkout: {COMFY}")
    if not COMFY.exists():
        run("git", "clone", repository, COMFY)

    run("git", "-C", COMFY, "fetch", "--all", "--tags", "--prune")
    run("git", "-C", COMFY, "checkout", "--detach", revision)
    run(
        sys.executable, "-m", "pip", "install", "-q",
        "--disable-pip-version-check", "-r", COMFY / "requirements.txt",
    )

    head = subprocess.check_output(
        ["git", "-C", str(COMFY), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_head = subprocess.check_output(
        ["git", "-C", str(COMFY), "rev-list", "-n", "1", revision], text=True
    ).strip()
    if head != expected_head:
        raise RuntimeError(f"ComfyUI checkout mismatch: expected={expected_head}, actual={head}")
    tagged = subprocess.run(
        ["git", "-C", str(COMFY), "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if expected_version and tagged not in {expected_version, f"v{expected_version}"}:
        raise RuntimeError(
            f"ComfyUI release mismatch: expected={expected_version}, actual={tagged or 'untagged'}"
        )
    print(f"[COMFYUI] revision={head} version={tagged or 'untagged'}")


def apply_embedded_h3_runtime_overlay() -> None:
    """Apply the project-owned H3 runtime patches without embedding full upstream source."""
    targets = (
        'comfy/ldm/minimax/model.py',
        'comfy/ldm/minimax/vae.py',
        'comfy/supported_models.py',
        'custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py',
    )
    required = [COMFY / relative for relative in targets]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("H3 runtime overlay targets are missing:\n" + "\n".join(missing))

    runtime = load_yaml(RUNTIME_MANIFEST)
    patch_h3_fp16_runtime(runtime)
    patch_h3_vae_decoder_dtype(runtime)
    patch_t4_h3_value_clone(runtime)

    model_text = (COMFY / "comfy/ldm/minimax/model.py").read_text(encoding="utf-8")
    vae_text = (COMFY / "comfy/ldm/minimax/vae.py").read_text(encoding="utf-8")
    supported_text = (COMFY / "comfy/supported_models.py").read_text(encoding="utf-8")
    turbo_text = (COMFY / "custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py").read_text(encoding="utf-8")
    required_signatures = {
        "model.py": (
            "# H3-T4-WORKAROUND: removed redundant V clone for SM75",
            "condition_proj(text_states.to(torch.float32))",
            "residual_dtype = torch.float32 if dtype == torch.float16 else dtype",
            "low_precision_attention=False",
            "self.out_proj((out / 64.0).to(torch.float16))",
        ),
        "vae.py": (
            "decoder_dtype = next(self.decoder.parameters()).dtype",
            "if z.dtype != decoder_dtype:",
            "z = z.to(decoder_dtype)",
        ),
        "supported_models.py": (
            "memory_usage_factor = 0.17",
            "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]",
        ),
        "ComfyUI-MiniMax-H3-Turbo/__init__.py": (
            "class MiniMaxH3TurboLoRA",
        ),
    }
    actual_sources = {
        "model.py": model_text,
        "vae.py": vae_text,
        "supported_models.py": supported_text,
        "ComfyUI-MiniMax-H3-Turbo/__init__.py": turbo_text,
    }
    missing = [
        f"{name}: {signature}"
        for name, signatures in required_signatures.items()
        for signature in signatures
        if signature not in actual_sources[name]
    ]
    if missing:
        raise RuntimeError("H3 runtime overlay verification failed:\n" + "\n".join(missing))
    print("[H3 COMFY PATCH] embedded runtime overlay applied: 4 files")
    print("[H3 COMFY PATCH] post-overlay source verification: PASS")


def verify_inventory() -> None:
    manifest = load_yaml(MODEL_MANIFEST)
    expected = {
        (model["directory"], model["filename"].lower())
        for model in manifest["models"].values()
    }
    placeholders = {
        "put_diffusion_model_files_here",
        "put_latent_upscale_models_here",
        "put_loras_here",
        "put_text_encoder_files_here",
        "put_vae_here",
    }
    actual = set()
    for directory_name in {"diffusion_models", "text_encoders", "loras", "vae", "latent_upscale_models"}:
        directory = MODELS / directory_name
        if not directory.is_dir():
            continue
        for item in directory.iterdir():
            if item.is_file() and item.name.lower() not in placeholders:
                actual.add((directory_name, item.name.lower()))
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise RuntimeError("Missing H3 models:\n" + "\n".join(f"{d}/{f}" for d, f in sorted(missing)))
    if unexpected:
        raise RuntimeError("Unexpected H3 production models:\n" + "\n".join(f"{d}/{f}" for d, f in sorted(unexpected)))


def verify_runtime_files(runtime: dict) -> None:
    if not (COMFY / "main.py").is_file():
        raise RuntimeError(f"ComfyUI main.py is missing: {COMFY / 'main.py'}")
    if not CUSTOM.is_dir():
        raise RuntimeError(f"ComfyUI custom_nodes directory is missing: {CUSTOM}")
    if not MODELS.is_dir():
        raise RuntimeError(f"ComfyUI models directory is missing: {MODELS}")
    revision = str(runtime.get("comfyui", {}).get("revision", "") or "").strip()
    expected_version = str(runtime.get("comfyui", {}).get("expected_version", "") or "").strip()
    head = subprocess.check_output(["git", "-C", str(COMFY), "rev-parse", "HEAD"], text=True).strip()
    if revision:
        expected_head = subprocess.check_output(
            ["git", "-C", str(COMFY), "rev-list", "-n", "1", revision], text=True
        ).strip()
        if head != expected_head:
            raise RuntimeError("ComfyUI checkout is not at the locked revision.")
    tagged = subprocess.run(
        ["git", "-C", str(COMFY), "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if expected_version and tagged not in {expected_version, f"v{expected_version}"}:
        raise RuntimeError(
            f"ComfyUI checkout is not the expected release tag: expected={expected_version}, actual={tagged or 'untagged'}"
        )


def install_pytorch_runtime(runtime: dict) -> None:
    """Install and verify the project-locked PyTorch CUDA build last.
    ComfyUI and custom-node requirements are allowed to install their own
    compatible dependencies first. PyTorch is re-asserted only after all of
    those dependency installs so a transitive requirement cannot silently
    leave the worker on a different CUDA build.
    """
    config = dict(runtime.get("pytorch", {}) or {})
    version = str(config.get("version", "") or "").strip()
    cuda = str(config.get("cuda", "") or "").strip().lower()
    index = str(config.get("index", "") or "").strip()
    torchvision_version = str(config.get("torchvision_version", "") or "").strip()
    torchaudio_version = str(config.get("torchaudio_version", "") or "").strip()
    if not all((version, cuda, index, torchvision_version, torchaudio_version)):
        raise RuntimeError("runtime_versions.yaml pytorch configuration is incomplete.")
    if cuda != "cu130":
        raise RuntimeError(f"This Ref2VA project is locked to cu130, got {cuda!r}.")
    print("=" * 80)
    print("INSTALLING LOCKED PYTORCH RUNTIME")
    print("=" * 80)
    run(
        sys.executable,
        "-m", "pip", "install", "-q",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--force-reinstall",
        "--index-url", index,
        f"torch=={version}",
        f"torchvision=={torchvision_version}",
        f"torchaudio=={torchaudio_version}",
    )
    verify = subprocess.run(
        [
            sys.executable, "-c",
            (
                "import torch; "
                f"assert torch.__version__ == '2.10.0+{cuda}'; "
                "assert torch.version.cuda == '13.0'; "
                "print(torch.__version__); "
                "print(torch.version.cuda)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.returncode != 0:
        raise RuntimeError(
            "Locked PyTorch runtime verification failed.\n"
            + (verify.stdout or "")
            + (verify.stderr or "")
        )
    print("[PYTORCH]", (verify.stdout or "").strip().replace("\n", " | "))
def _enforce_no_restart() -> None:
    """Require execution inside the live Kaggle IPython kernel."""
    try:
        from IPython import get_ipython
        in_kernel = get_ipython() is not None
    except Exception:
        in_kernel = False
    if os.getenv("JPY_PARENT_PID") and not in_kernel:
        raise RuntimeError("NO-RESTART bootstrap must run in the current Kaggle kernel; use `%run kaggle/bootstrap.py`, not `!python kaggle/bootstrap.py`.")
def _repair_loaded_pillow(expected_version: str) -> None:
    """Repair Pillow modules already loaded before pip replaced the files."""
    expected_ink = float | tuple[int, ...] | str
    loaded_typing = sys.modules.get("PIL._typing")
    loaded_pkg = sys.modules.get("PIL")
    if loaded_typing is not None:
        loaded_typing._Ink = expected_ink
        print("[PILLOW CACHE] repaired loaded PIL._typing._Ink")
    if loaded_pkg is not None:
        loaded_pkg.__version__ = expected_version
        print("[PILLOW CACHE] reconciled loaded PIL.__version__", expected_version)
    import importlib
    importlib.invalidate_caches()
    import PIL
    from PIL._typing import _Ink
    if str(PIL.__version__) != expected_version or _Ink != expected_ink:
        raise RuntimeError(f"Live-process Pillow repair failed: version={PIL.__version__!r}, _Ink={_Ink!r}")
    print("[PILLOW CACHE] live-process Pillow cache: PASS")
def patch_h3_fp16_runtime(runtime: dict) -> None:
    """Apply narrow H3 FP16/T4 runtime corrections; never replace full ComfyUI files."""
    model = COMFY / "comfy/ldm/minimax/model.py"
    supported = COMFY / "comfy/supported_models.py"
    if not model.is_file() or not supported.is_file():
        raise RuntimeError("Required ComfyUI H3 source files are missing.")
    text = model.read_text(encoding="utf-8")
    old = "out = optimized_attention(q, k, v, self.heads, mask=None, skip_reshape=True, transformer_options=transformer_options)"
    if "low_precision_attention=False" not in text:
        if old not in text: raise RuntimeError("H3 attention source pattern changed.")
        text = text.replace(old, old[:-1] + ", low_precision_attention=False)", 1)
    if "condition_proj(text_states.to(torch.float32))" not in text:
        old = (
            "            text_states = self.token_refiner(self.condition_proj(text_states),\n"
            "                                             transformer_options=transformer_options)"
        )
        new = (
            "            # CRITICAL H3 T4 boundary: condition_proj must receive FP32 input.\n"
            "            text_states = self.condition_proj(text_states.to(torch.float32))\n"
            "            text_states = self.token_refiner(\n"
            "                text_states,\n"
            "                transformer_options=transformer_options,\n"
            "            )"
        )
        if old not in text: raise RuntimeError("H3 text-conditioning source pattern changed for ComfyUI v0.34.0.")
        text = text.replace(old, new, 1)
    if "residual_dtype = torch.float32 if dtype == torch.float16 else dtype" not in text:
        old = "h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)"
        if old not in text: raise RuntimeError("H3 residual source pattern changed.")
        text = text.replace(old, "residual_dtype = torch.float32 if dtype == torch.float16 else dtype\n        " + old.replace("dtype=dtype", "dtype=residual_dtype"), 1)
    if "video_embed = self.video_patch_proj(all_video_rows).to(embed_dtype)" not in text:
        old = "video_embed = self.video_patch_proj(all_video_rows).to(dtype)\n        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)"
        if old not in text: raise RuntimeError("H3 embedding source pattern changed.")
        text = text.replace(old, "embed_dtype = torch.float32 if dtype == torch.float16 else dtype\n        video_embed = self.video_patch_proj(all_video_rows).to(embed_dtype)\n        audio_embed = self.audio_patch_proj(all_audio_rows).to(embed_dtype)", 1)
    if "self.out_proj((out / 64.0).to(torch.float16))" not in text:
        old = "        return self.out_proj(out.squeeze(0))"
        if old not in text: raise RuntimeError("H3 attention projection source pattern changed for ComfyUI v0.34.0.")
        new = (
            "        out = out.squeeze(0)\n"
            "\n"
            "        # H3 T4 FP16 numerical boundary: keep the residual stream FP32, but\n"
            "        # scale the dangerous attention output projection into FP16 range.\n"
            "        proj_weight = getattr(self.out_proj, \"weight\", None)\n"
            "        proj_dtype = getattr(proj_weight, \"dtype\", None)\n"
            "        if x.dtype == torch.float32 and proj_dtype == torch.float16:\n"
            "            return (\n"
            "                self.out_proj((out / 64.0).to(torch.float16))\n"
            "                .to(torch.float32).mul_(64.0)\n"
            "            )\n"
            "\n"
            "        return self.out_proj(out)"
        )
        text = text.replace(old, new, 1)
    model.write_text(text, encoding="utf-8")
    s = supported.read_text(encoding="utf-8")
    if "memory_usage_factor = 0.17" not in s:
        if "memory_usage_factor = 0.114" not in s: raise RuntimeError("MiniMaxH3 memory factor source pattern changed.")
        s = s.replace("memory_usage_factor = 0.114", "memory_usage_factor = 0.17", 1)
    if "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]" not in s:
        old = "supported_inference_dtypes = [torch.bfloat16, torch.float32]"
        if old not in s: raise RuntimeError("MiniMaxH3 dtype source pattern changed.")
        s = s.replace(old, "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]", 1)
    supported.write_text(s, encoding="utf-8")
    print("[H3 COMFY PATCH] FP16/T4 numerical patches applied")
def patch_t4_h3_value_clone(runtime: dict) -> None:
    """Apply the narrowly-scoped T4 H3 v-clone workaround to the locked ComfyUI.
    ComfyUI 0.34.0's MiniMax H3 Attention copies the large V tensor before
    wrapping it in AttentionTensorContainer. On 16-GB-class GPUs that can
    add about a gigabyte of peak memory and severely degrade throughput.
    The workaround is applied only when the exact upstream 0.34.0 source
    pattern is present and only on SM75 GPUs. If the source changes, fail
    loudly instead of silently patching the wrong code.
    """
    enabled = bool(
        runtime.get("comfyui", {}).get("h3_t4_value_clone_workaround", True)
    )
    if not enabled:
        print("[H3 T4 PATCH] disabled by runtime configuration")
        return
    try:
        import torch
        if not torch.cuda.is_available():
            print("[H3 T4 PATCH] skipped: CUDA unavailable")
            return
        major, minor = torch.cuda.get_device_capability(0)
        if (major, minor) != (7, 5):
            print(f"[H3 T4 PATCH] skipped: GPU SM{major}{minor} is not SM75")
            return
    except Exception as exc:
        raise RuntimeError(f"Cannot determine GPU capability for H3 T4 patch: {exc}") from exc
    target = COMFY / "comfy" / "ldm" / "minimax" / "model.py"
    if not target.is_file():
        raise RuntimeError(f"H3 model source not found: {target}")
    text = target.read_text(encoding="utf-8")
    marker = "# H3-T4-WORKAROUND: removed redundant V clone for SM75"
    if marker in text:
        print("[H3 T4 PATCH] already applied")
        return
    exact = "        v = v.clone()\n        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))"
    replacement = "        " + marker + "\n        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))"
    if exact not in text:
        raise RuntimeError(
            "Refusing to apply the H3 T4 workaround because ComfyUI's expected "
            "0.34.0 Attention pattern was not found."
        )
    target.write_text(text.replace(exact, replacement, 1), encoding="utf-8")
    print("[H3 T4 PATCH] applied to", target)
def patch_h3_vae_decoder_dtype(runtime: dict) -> None:
    """Keep MiniMax H3 video VAE decoder input on the decoder's dtype.
    This is deliberately scoped to the locked H3 VAE implementation. It is
    idempotent and fails closed if the expected 0.34.0 source pattern changes.
    """
    enabled = bool(runtime.get("comfyui", {}).get("h3_vae_decoder_dtype_patch", True))
    if not enabled:
        print("[H3 VAE PATCH] disabled by runtime configuration")
        return
    target = COMFY / "comfy" / "ldm" / "minimax" / "vae.py"
    if not target.is_file():
        raise RuntimeError(f"H3 VAE source not found: {target}")
    text = target.read_text(encoding="utf-8")
    marker = "# H3-T4-VAE-DTYPE: decoder input matches decoder parameters"
    existing_patch = (
        "        z = self.post_quant_conv(z)\n"
        "        decoder_dtype = next(self.decoder.parameters()).dtype\n"
        "        if z.dtype != decoder_dtype:\n"
        "            z = z.to(decoder_dtype)\n"
        "        return self.decoder(z)"
    )
    if marker in text or existing_patch in text:
        print("[H3 VAE PATCH] already applied")
        return
    exact = "        return self.decoder(self.post_quant_conv(z))"
    replacement = (
        "        z = self.post_quant_conv(z)\n"
        f"        {marker}\n"
        "        decoder_dtype = next(self.decoder.parameters()).dtype\n"
        "        if z.dtype != decoder_dtype:\n"
        "            z = z.to(decoder_dtype)\n"
        "        return self.decoder(z)"
    )
    if exact not in text:
        raise RuntimeError(
            "Refusing to apply the H3 VAE dtype patch because the expected "
            "locked ComfyUI 0.34.0 source pattern was not found."
        )
    target.write_text(text.replace(exact, replacement, 1), encoding="utf-8")
    print("[H3 VAE PATCH] applied to", target)
def install_director_runtime(
    runtime: dict,
) -> None:
    """Install isolated vLLM + EAGLE-3 Director runtime without mutating H3 Torch."""
    director = runtime.get("director", {}) or {}
    if str(director.get("backend", "") or "").strip().lower() != "vllm":
        raise RuntimeError("runtime_versions.yaml director.backend must be 'vllm'.")
    def resolve_checkpoint(configured: str, required: tuple[str, ...], label: str, *, require_weights: bool = False) -> Path:
        configured_path = Path(configured).expanduser()
        candidates = [configured_path]
        if Path("/kaggle/input").is_dir():
            candidates.append(Path("/kaggle/input") / configured_path.name)
        for root in (Path("/kaggle/input"),):
            if root.is_dir():
                try:
                    candidates.extend(
                        p for p in root.rglob(configured_path.name) if p.is_dir()
                    )
                except OSError:
                    pass
        def has_weights(path: Path) -> bool:
            index_files = tuple(path.glob("*.index.json"))
            for index_path in index_files:
                try:
                    index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                if not isinstance(index, dict):
                    continue
                weight_map = index.get("weight_map")
                if isinstance(weight_map, dict) and weight_map:
                    referenced = {str(name) for name in weight_map.values()}
                    if referenced and all((path / name).is_file() for name in referenced):
                        return True
            return any(
                any(path.glob(pattern))
                for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")
            )
        seen = set()
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
            except OSError:
                continue
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if not candidate.is_dir():
                continue
            if not all((candidate / name).is_file() for name in required):
                continue
            if require_weights and not has_weights(candidate):
                continue
            return candidate
        raise RuntimeError(f"Complete {label} checkpoint was not found: {configured}")
    model_path = resolve_checkpoint(
        os.getenv("H3_DIRECTOR_MODEL_PATH", str(director.get("model_path", "")).strip()),
        ("config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "tokenizer.json"),
        "Qwen3-14B-AWQ",
    )
    spec_path = resolve_checkpoint(
        os.getenv("H3_DIRECTOR_VLLM_SPECULATIVE_MODEL_PATH", str(director.get("speculative_model_path", "")).strip()),
        ("config.json",),
        "Qwen3-14B EAGLE-3 speculator",
        require_weights=True,
    )
    configured_speculator = str(director.get("speculative_model_path", "")).strip()
    if configured_speculator != "/kaggle/input/eagle-3":
        raise RuntimeError(
            "runtime_versions.yaml director.speculative_model_path must be /kaggle/input/eagle-3 for the locked Eagle-3 Kaggle dataset."
        )
    try:
        spec_config = yaml.safe_load((spec_path / "config.json").read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise RuntimeError(f"Unable to read EAGLE-3 speculator config: {spec_path / 'config.json'}") from exc
    spec_meta = spec_config.get("speculators_config", {}) or {}
    verifier = spec_meta.get("verifier", {}) or {}
    speculative_method = str(director.get("speculative_method", "") or "").strip().lower()
    if speculative_method != "eagle3":
        raise RuntimeError("runtime_versions.yaml director.speculative_method must be eagle3.")
    if str(spec_meta.get("algorithm", "")).strip().lower() != speculative_method:
        raise RuntimeError(
            "Configured speculator algorithm does not match runtime_versions.yaml "
            f"director.speculative_method={speculative_method!r}."
        )
    if str(verifier.get("name_or_path", "")).strip() != "Qwen/Qwen3-14B":
        raise RuntimeError("Configured EAGLE-3 speculator is not paired with Qwen/Qwen3-14B.")
    vllm_version = str(director.get("vllm_version", "") or "").strip()
    env_dir_value = str(director.get("vllm_env_dir", "") or "").strip()
    tensor_parallel_size = int(director.get("tensor_parallel_size", 0) or 0)
    if not vllm_version:
        raise RuntimeError("runtime_versions.yaml director.vllm_version is required.")
    if not env_dir_value:
        raise RuntimeError("runtime_versions.yaml director.vllm_env_dir is required.")
    if tensor_parallel_size <= 0:
        raise RuntimeError("runtime_versions.yaml director.tensor_parallel_size must be positive.")
    env_dir = Path(
        os.getenv("H3_DIRECTOR_VLLM_ENV_DIR", env_dir_value)
    ).expanduser().resolve()
    print("=" * 80)
    print("INSTALLING QWEN DIRECTOR RUNTIME")
    print("=" * 80)
    print("[DIRECTOR]", f"model={model_path}")
    print("[DIRECTOR]", f"speculator={spec_path}")
    print("[DIRECTOR]", f"backend=vllm version={vllm_version}")
    print("[DIRECTOR]", f"isolated_env={env_dir}")
    uv = shutil.which("uv")
    if uv is None:
        run(sys.executable, "-m", "pip", "install", "-q", "uv")
        uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to create the isolated Director environment.")
    if not (env_dir / "bin" / "python").is_file():
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        run(uv, "venv", str(env_dir), "--python", sys.executable, "--seed", "--link-mode", "copy")
    venv_python = env_dir / "bin" / "python"
    if not venv_python.is_file() or not os.access(venv_python, os.X_OK):
        raise RuntimeError(f"Invalid executable Director Python: {venv_python}")
    env = os.environ.copy()
    env["UV_LINK_MODE"] = "copy"
    install = subprocess.run(
        [uv, "pip", "install", "--python", str(venv_python), "--link-mode", "copy", f"vllm=={vllm_version}", f"wrapt=={WRAPT_VERSION}"],
        env=env,
        check=False,
        text=True,
    )
    if install.returncode != 0:
        raise RuntimeError("Failed to install isolated vLLM runtime.")
    verification = subprocess.run(
        [
            str(venv_python), "-c",
            (
                "import vllm, torch, inspect, wrapt; "
                "from vllm.config import SpeculativeConfig; "
                "print('vLLM import: PASS'); "
                "print('wrapt version:', wrapt.__version__); "
                "print('vLLM version:', vllm.__version__); "
                "print('EAGLE-3 supported:', 'eagle3' in str(inspect.signature(SpeculativeConfig))); "
                "print('Torch CUDA:', torch.cuda.is_available()); "
                "print('GPU count:', torch.cuda.device_count()); "
                "print('GPU capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)"
            ),
        ],
        capture_output=True, text=True, check=False, env=env,
    )
    print(verification.stdout)
    if verification.stderr:
        print(verification.stderr)
    if verification.returncode != 0:
        raise RuntimeError("vLLM isolated runtime verification failed.")
    observed_wrapt_version = None
    for line in verification.stdout.splitlines():
        if line.startswith("wrapt version:"):
            observed_wrapt_version = line.split(":", 1)[1].strip()
            break
    if observed_wrapt_version != WRAPT_VERSION:
        raise RuntimeError(
            "Director isolated runtime wrapt mismatch: "
            f"observed={observed_wrapt_version!r}, expected={WRAPT_VERSION!r}."
        )
    observed_version = None
    for line in verification.stdout.splitlines():
        if line.startswith("vLLM version:"):
            observed_version = line.split(":", 1)[1].strip()
            break
    if observed_version != vllm_version:
        raise RuntimeError(
            "Director runtime verification version mismatch: "
            f"observed={observed_version!r}, configured={vllm_version!r}."
        )
    if f"EAGLE-3 supported: True" not in verification.stdout:
        raise RuntimeError(
            f"Director runtime verification did not confirm speculative method {speculative_method!r}."
        )
    observed_gpu_count = None
    for line in verification.stdout.splitlines():
        if line.startswith("GPU count:"):
            try:
                observed_gpu_count = int(line.split(":", 1)[1].strip())
            except ValueError:
                observed_gpu_count = None
            break
    if observed_gpu_count != tensor_parallel_size:
        raise RuntimeError(
            "Qwen Director GPU topology mismatch: "
            f"observed={observed_gpu_count}, configured tensor_parallel_size={tensor_parallel_size}."
        )
    print("[DIRECTOR] EAGLE-3 speculator ready:", spec_path)
def install_storyboard_runtime(
    runtime: dict,
) -> None:
    """Install storyboard UI dependencies from the project runtime lock.
    Pillow is intentionally NOT verified here because later ComfyUI/custom-node
    dependency installation may mutate the shared Kaggle Python environment.
    Final Pillow enforcement/verification happens immediately before the final
    runtime checks, after every package installer has completed.
    """
    storyboard = runtime["storyboard"]
    gradio_version = str(
        storyboard.get("gradio_version", "")
    ).strip()
    if gradio_version:
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            f"gradio=={gradio_version}",
        )
def install_and_verify_pillow_runtime(runtime: dict) -> None:
    pillow_version = str(runtime.get("storyboard", {}).get("pillow_version", "")).strip()
    if not pillow_version: raise RuntimeError("runtime_versions.yaml storyboard.pillow_version is missing.")
    run(sys.executable, "-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "-q", "--disable-pip-version-check", f"Pillow=={pillow_version}")
    _repair_loaded_pillow(pillow_version)
    verify = subprocess.run([sys.executable, "-c", "from PIL import Image; from PIL._typing import _Ink; print(Image.__version__); print(_Ink)"], capture_output=True, text=True, check=False)
    if verify.returncode != 0 or pillow_version not in (verify.stdout or ""):
        raise RuntimeError("Fresh-process Pillow verification failed.\n" + (verify.stdout or "") + (verify.stderr or ""))
    print("[PILLOW LIVE]", pillow_version, "_Ink=PASS")
def remove_legacy_context_ir_node() -> None:
    """Remove the retired external Context-IR bridge from runtime custom_nodes."""
    destination = CUSTOM / "ComfyUI-MiniMax-H3-ContextIR"
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    for key in (
        "H3_CONTEXT_IR_OFFICIAL_ENABLED",
        "H3_CONTEXT_IR_OFFICIAL_PREFERRED",
        "H3_CONTEXT_IR_OFFICIAL_REQUIRED",
        "H3_CONTEXT_IR_TIMEOUT_SECONDS",
        "H3_CONTEXT_IR_POLL_INTERVAL_SECONDS",
        "H3_CONTEXT_IR_MAX_POLLS",
        "H3_CONTEXT_IR_MAX_IMAGE_MB",
        "H3_CONTEXT_IR_MAX_VIDEO_MB",
        "H3_CONTEXT_IR_MAX_AUDIO_MB",
    ):
        os.environ.pop(key, None)
    print("[NODE] retired external MiniMax H3 Context-IR bridge removed")

def patch_h3_bounded_qkv(runtime: dict) -> None:
    """Apply the tested T4 bounded native QKV fix to locked H3-Optimizations.

    H3-Optimizations is cloned at runtime, so the project-owned QKV safety fix
    is applied here rather than vendoring or changing the locked upstream
    repository. The patch is deliberately exact, idempotent, and fails closed
    if the pinned H3-Optimizations source contract changes.
    """
    node_dir = CUSTOM / "H3-Optimizations"
    providers = node_dir / "h3_optimizations" / "qkv" / "providers.py"
    apply_source = node_dir / "h3_optimizations" / "apply.py"
    expected_revision = "379f9c7922b3d7831dd93ae069ba0cb82cb4cf36"
    configured_revision = str(
        runtime.get("h3_optimization", {}).get("revision", "")
    ).strip()

    if configured_revision != expected_revision:
        raise RuntimeError(
            "Refusing the H3 T4 bounded-QKV patch because runtime_versions.yaml "
            "does not pin the source contract this patch targets: "
            f"expected={expected_revision}, configured={configured_revision or 'missing'}."
        )
    if not node_dir.is_dir():
        raise RuntimeError(f"H3-Optimizations runtime directory is missing: {node_dir}")

    actual_revision = subprocess.check_output(
        ["git", "-C", str(node_dir), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            "Refusing the H3 T4 bounded-QKV patch because the installed "
            f"H3-Optimizations revision is {actual_revision}, expected {expected_revision}."
        )
    for target in (providers, apply_source):
        if not target.is_file():
            raise RuntimeError(f"H3-Optimizations source is missing: {target}")

    provider_source = providers.read_text(encoding="utf-8")
    provider_markers = (
        "# MINIMAX_H3_T4_BOUNDED_QKV_GEMM_LIMIT",
        "# MINIMAX_H3_T4_FORCE_BOUNDED_QKV",
    )
    provider_contract = (
        "QKV_BF16_CHUNKED",
        "inventory.qkv_convrot_int8_256",
        "backend_kind == 'existing'",
        "memory_optimize",
        "[H3T4_BOUNDED_QKV_SELECTED]",
    )
    provider_count = sum(marker in provider_source for marker in provider_markers)
    if provider_count == len(provider_markers):
        missing = [needle for needle in provider_contract if needle not in provider_source]
        if missing:
            raise RuntimeError(
                "H3 T4 bounded-QKV provider markers exist but the source contract "
                "is incomplete; refusing to continue:\n" + "\n".join(missing)
            )
        compile(provider_source, str(providers), "exec")
        print("[H3 T4 QKV PATCH] providers.py already patched: VERIFIED")
    elif provider_count:
        raise RuntimeError(
            "H3 T4 bounded-QKV provider patch is only partially present; refusing "
            "to guess or merge an unknown source state."
        )
    else:
        preserve_anchor = """        return QKVProviderResolution(
            (
                QKV_DENSE_KITCHEN_CHUNKED
                if backend_kind == 'comfy_kitchen_int8'
                else QKV_STREAMED_BF16_KITCHEN
            ),
            backend_kind != 'comfy_kitchen_int8',
            'checkpoint-native ConvRot-256 INT8 QKV streams BF16 projection chunks into the Kitchen INT8 carrier',
        )
    if not _qkv_is_native_bf16(inventory):"""
        preserve_replacement = """        return QKVProviderResolution(
            (
                QKV_DENSE_KITCHEN_CHUNKED
                if backend_kind == 'comfy_kitchen_int8'
                else QKV_STREAMED_BF16_KITCHEN
            ),
            backend_kind != 'comfy_kitchen_int8',
            'checkpoint-native ConvRot-256 INT8 QKV streams BF16 projection chunks into the Kitchen INT8 carrier',
        )
    # MINIMAX_H3_T4_BOUNDED_QKV_GEMM_LIMIT
    #
    # The ref2va pruned ConvRot-256 INT8 H3 checkpoint has a 21,504-wide
    # packed QKV projection. For the T4/native existing-attention path,
    # projecting the complete packed sequence in one INT8 GEMM can exceed
    # the kernel's 32-bit indexing domain:
    #
    #     sequence * qkv_width >= 2**31
    #
    # Preserve native semantics, but force the existing bounded QKV
    # projector so the native ConvRot weights are consumed in token chunks.
    if (
        backend_kind == 'existing'
        and memory_optimize
        and inventory.qkv_convrot_int8_256
        and request == FUSED_QKV_PRESERVE_BF16
    ):
        return QKVProviderResolution(
            QKV_BF16_CHUNKED,
            False,
            (
                'T4 H3 safety: bounded native ConvRot-256 INT8 QKV '
                'projection selected for existing dense attention to avoid '
                'full-sequence INT8 GEMM 32-bit indexing overflow'
            ),
        )

    if not _qkv_is_native_bf16(inventory):"""
        if provider_source.count(preserve_anchor) != 1:
            raise RuntimeError(
                "H3-Optimizations 0.2.41 providers.py no longer matches the locked "
                "QKV preserve-precision source contract; refusing to patch."
            )
        provider_source = provider_source.replace(
            preserve_anchor, preserve_replacement, 1
        )

        resolver_anchor = """    fp8_available=False,
):
    if request == FUSED_QKV_OFF:"""
        resolver_replacement = """    fp8_available=False,
):
    # MINIMAX_H3_T4_FORCE_BOUNDED_QKV
    #
    # MiniMax H3 Ref2VA pruned ConvRot-256 INT8 has a very wide packed
    # QKV projection (56 heads * 128 head_dim * 3 = 21504 output width).
    # On Tesla T4, the full-sequence INT8 GEMM can exceed the kernel's
    # 32-bit indexing domain for long H3 sequences.
    #
    # For existing ComfyUI dense attention, always route this native
    # ConvRot-256 INT8 checkpoint through the bounded native QKV projector.
    if (
        backend_kind == 'existing'
        and memory_optimize
        and inventory.qkv_convrot_int8_256
    ):
        print(
            '[H3T4_BOUNDED_QKV_SELECTED] '
            'backend=existing '
            'memory_optimize=True '
            'qkv=ConvRot-256-INT8 '
            'provider=chunked_bf16_qkv '
            'reason=force_t4_bounded_native_projection',
            flush=True,
        )
        return QKVProviderResolution(
            QKV_BF16_CHUNKED,
            False,
            'T4 H3 safety: force bounded native ConvRot-256 INT8 QKV '
            'for existing dense attention',
        )

    if request == FUSED_QKV_OFF:"""
        if provider_source.count(resolver_anchor) != 1:
            raise RuntimeError(
                "H3-Optimizations 0.2.41 providers.py no longer matches the locked "
                "QKV resolver source contract; refusing to patch."
            )
        provider_source = provider_source.replace(
            resolver_anchor, resolver_replacement, 1
        )
        compile(provider_source, str(providers), "exec")
        providers.write_text(provider_source, encoding="utf-8")
        print("[H3 T4 QKV PATCH] providers.py patched: PASS")

    apply_text = apply_source.read_text(encoding="utf-8")
    apply_markers = (
        "# MINIMAX_H3_T4_APPLY_FORCE_STREAMED_QKV",
        "# MINIMAX_H3_T4_QKV_RUNTIME_SELECTION",
    )
    apply_contract = (
        "qkv.provider_id == QKV_STANDARD",
        "tuple(getattr(environment, 'capability', ()) or ()) == (7, 5)",
        "qkv = QKVProviderResolution(",
        "StreamedDenseBF16QKVProjector(",
        "QKV runtime selection: provider=%s projector=%s",
    )
    apply_count = sum(marker in apply_text for marker in apply_markers)
    if apply_count == len(apply_markers):
        missing = [needle for needle in apply_contract if needle not in apply_text]
        if missing:
            raise RuntimeError(
                "H3 T4 bounded-QKV apply markers exist but the source contract is "
                "incomplete; refusing to continue:\n" + "\n".join(missing)
            )
        compile(apply_text, str(apply_source), "exec")
        print("[H3 T4 QKV PATCH] apply.py already patched: VERIFIED")
    elif apply_count:
        raise RuntimeError(
            "H3 T4 bounded-QKV apply patch is only partially present; refusing "
            "to guess or merge an unknown source state."
        )
    else:
        qkv_anchor = """    qkv = resolve_qkv_provider(
        inventory,
        request=(FUSED_QKV_OFF if memory is None else _qkv_request(plan)),
        backend_kind=dense.backend_kind,
        kitchen_producer_available=producer_api_available(
            device=getattr(environment, 'device_index', None)
        ),
        triton_available=dense_carrier_available,
        memory_optimize=memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    backend = None
    projector = None"""
        qkv_replacement = """    qkv = resolve_qkv_provider(
        inventory,
        request=(FUSED_QKV_OFF if memory is None else _qkv_request(plan)),
        backend_kind=dense.backend_kind,
        kitchen_producer_available=producer_api_available(
            device=getattr(environment, 'device_index', None)
        ),
        triton_available=dense_carrier_available,
        memory_optimize=memory is not None,
        fp8_available=_fp8_execution_available(environment),
    )
    # MINIMAX_H3_T4_APPLY_FORCE_STREAMED_QKV
    #
    # T4/SM75 safety rule:
    # MiniMax H3 long-sequence Ref2VA cannot safely use the ordinary full-QKV
    # projection at this sequence length. Force the package-owned streamed
    # native QKV projector at the integration point where it is actually
    # attached to H3 attention.
    #
    # QKV_BF16_CHUNKED does NOT mean "convert the model to BF16".
    # StreamedDenseBF16QKVProjector uses projection_mode="native", which selects
    # the native checkpoint-aware binding (ConvRot INT8 / W4A8 / FP8 / BF16).
    #
    # Only replace STANDARD fallback. If a compatible specialized provider
    # was already selected, preserve that provider.
    if (
        memory is not None
        and dense.backend_kind == ATTENTION_EXISTING
        and tuple(getattr(environment, 'capability', ()) or ()) == (7, 5)
        and qkv.provider_id == QKV_STANDARD
    ):
        qkv = QKVProviderResolution(
            QKV_BF16_CHUNKED,
            False,
            'T4 safety: forced streamed native H3 QKV at integration layer',
        )
        logging.warning(
            '%s T4 BOUNDED QKV FORCED: provider=%s backend=%s',
            LOG_PREFIX,
            qkv.provider_id,
            dense.backend_kind,
        )

    backend = None
    projector = None"""
        if apply_text.count(qkv_anchor) != 1:
            raise RuntimeError(
                "H3-Optimizations 0.2.41 apply.py no longer matches the locked "
                "QKV integration source contract; refusing to patch."
            )
        apply_text = apply_text.replace(qkv_anchor, qkv_replacement, 1)

        log_anchor = """        environment.device_name,
    )
    if phase == 'prepare':"""
        log_replacement = """        environment.device_name,
    )
    # MINIMAX_H3_T4_QKV_RUNTIME_SELECTION
    logging.warning(
        '%s QKV runtime selection: provider=%s projector=%s '
        'backend_kind=%s chunk_rows=%s reason=%s',
        LOG_PREFIX,
        qkv.provider_id,
        getattr(attention.projector, 'name', None),
        attention.backend_kind,
        getattr(attention.projector, 'chunk_rows', None),
        qkv.reason,
    )
    if phase == 'prepare':"""
        if apply_text.count(log_anchor) != 1:
            raise RuntimeError(
                "H3-Optimizations 0.2.41 apply.py no longer matches the locked "
                "QKV reconciliation source contract; refusing to patch."
            )
        apply_text = apply_text.replace(log_anchor, log_replacement, 1)
        compile(apply_text, str(apply_source), "exec")
        apply_source.write_text(apply_text, encoding="utf-8")
        print("[H3 T4 QKV PATCH] apply.py patched: PASS")

    # Final source-level verification after both files are handled.
    provider_source = providers.read_text(encoding="utf-8")
    apply_text = apply_source.read_text(encoding="utf-8")
    final_required = {
        str(providers): provider_markers + provider_contract + (
            "request == FUSED_QKV_PRESERVE_BF16",
            "reason=force_t4_bounded_native_projection",
        ),
        str(apply_source): apply_markers + apply_contract + (
            "projection_mode=_streamed_projection_mode(qkv, inventory)",
            "getattr(attention.projector, 'chunk_rows', None)",
        ),
    }
    final_sources = {
        str(providers): provider_source,
        str(apply_source): apply_text,
    }
    missing = [
        f"{path}: {needle}"
        for path, needles in final_required.items()
        for needle in needles
        if needle not in final_sources[path]
    ]
    if missing:
        raise RuntimeError(
            "H3 T4 bounded-QKV patch verification failed:\n" + "\n".join(missing)
        )
    compile(provider_source, str(providers), "exec")
    compile(apply_text, str(apply_source), "exec")
    print(
        "[H3 T4 QKV PATCH] PASS: providers.py + apply.py patched/verified "
        f"at H3-Optimizations {actual_revision}"
    )

def patch_h3_t4_gpu_parallel_attention(runtime: dict) -> None:
    """Persist the latest T4 dual-GPU streamed-attention fixes.

    The project repository intentionally does not vendor H3-Optimizations;
    install_nodes() checks out the pinned revision and this function applies
    the runtime overlay to that exact checkout.

    Fixes persisted here:
      1. keep the native ConvRot binding alive across all streamed Q slabs;
      2. run streamed dense attention on cuda:1 with PyTorch efficient
         attention, while preserving the H3 HND output contract.

    Only the current efficient-attention implementation is installed.
    """
    del runtime

    node_dir = CUSTOM / "H3-Optimizations"
    package_dir = node_dir / "h3_optimizations"
    bf16 = package_dir / "qkv" / "bf16.py"
    attention = package_dir / "attention_forward.py"
    expected_revision = "379f9c7922b3d7831dd93ae069ba0cb82cb4cf36"

    if not node_dir.is_dir():
        raise RuntimeError(
            f"H3-Optimizations runtime directory is missing: {node_dir}"
        )

    actual_revision = subprocess.check_output(
        ["git", "-C", str(node_dir), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            "Refusing H3 T4 GPU-parallel attention patch because the installed "
            f"H3-Optimizations revision is {actual_revision}, expected {expected_revision}."
        )

    if not bf16.is_file():
        raise RuntimeError(f"H3 QKV source is missing: {bf16}")
    if not attention.is_file():
        raise RuntimeError(f"H3 attention source is missing: {attention}")

    # ------------------------------------------------------------------
    # Fix 1: retain the native ConvRot binding for streamed Q slabs.
    # ------------------------------------------------------------------
    bf16_text = bf16.read_text(encoding="utf-8")
    bf16_marker = "# MINIMAX_H3_T4_QKV_BINDING_LIFETIME"

    if bf16_marker not in bf16_text:
        release_block = """            if _held_cast_handle(held) is not None:

                held.__exit__(
                    None,
                    None,
                    None,
                )

                held = None

"""
        compact_release_block = """            if _held_cast_handle(held) is not None:
                held.__exit__(None, None, None)
                held = None

"""
        replacement = (
            "            " + bf16_marker + "\n"
            "            # Keep the same native ConvRot session alive so every\n"
            "            # bounded Q slab reuses the binding acquired during K/V projection.\n\n"
        )

        if bf16_text.count(release_block) == 1:
            bf16_text = bf16_text.replace(release_block, replacement, 1)
        elif bf16_text.count(compact_release_block) == 1:
            bf16_text = bf16_text.replace(compact_release_block, replacement, 1)
        else:
            raise RuntimeError(
                "H3 qkv/bf16.py no longer matches the locked native-binding "
                "lifetime source contract; refusing to guess or patch an unknown state."
            )

        compile(bf16_text, str(bf16), "exec")
        bf16.write_text(bf16_text, encoding="utf-8")
        print("[H3 T4 QKV FIX] bf16.py patched: PASS")
    else:
        print("[H3 T4 QKV FIX] bf16.py already patched: VERIFIED")

    bf16_text = bf16.read_text(encoding="utf-8")
    if bf16_marker not in bf16_text:
        raise RuntimeError("H3 T4 QKV binding-lifetime marker missing after patch.")

    marker_pos = bf16_text.index(bf16_marker)
    return_pos = bf16_text.index(
        "return PreparedStreamedDenseBF16QKV(",
        marker_pos,
    )
    marker_region = bf16_text[marker_pos:return_pos]
    if "held = None" in marker_region:
        raise RuntimeError(
            "H3 T4 QKV binding-lifetime patch still releases the native handle before streamed Q."
        )
    if "held=held" not in bf16_text:
        raise RuntimeError(
            "H3 T4 QKV binding-lifetime patch no longer preserves held=held."
        )

    compile(bf16_text, str(bf16), "exec")
    print("[H3 T4 QKV FIX] final verification: PASS")

    # ------------------------------------------------------------------
    # Fix 2: replace streamed dense attention with the newest dual-GPU
    # efficient-attention implementation.
    # ------------------------------------------------------------------
    attention_text = attention.read_text(encoding="utf-8")
    helper_marker = "# MINIMAX_H3_T4_EFFICIENT_ATTENTION"
    new_streamed_function = r'''def _minimax_h3_t4_efficient_attention(
    module,
    q_slab,
    k_gpu1,
    v_gpu1,
):
    """Run one bounded H3 attention slab on the second T4.

    Q is produced on cuda:0 by the native streamed projector. K/V are retained
    on cuda:1 and the attention kernel runs there in FP16 using PyTorch's
    memory-efficient SDPA backend. The returned tensor is converted back to the
    original Q device/dtype and normalized to H3 HND [B,H,N,D].
    """
    # MINIMAX_H3_T4_EFFICIENT_ATTENTION
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError(
            "H3 T4 efficient attention requires at least two CUDA devices"
        )

    if q_slab.ndim != 4:
        raise RuntimeError(
            "H3 T4 efficient attention expected Q rank 4 HND, got rank-%d"
            % q_slab.ndim
        )

    heads = int(module.heads)
    head_dim = int(module.head_dim)
    expected_width = heads * head_dim

    if q_slab.shape[1] != heads or q_slab.shape[-1] != head_dim:
        raise RuntimeError(
            "H3 T4 efficient attention received invalid Q shape %s; "
            "expected [B,%d,N,%d]"
            % (tuple(q_slab.shape), heads, head_dim)
        )

    device0 = q_slab.device
    original_dtype = q_slab.dtype
    device1 = torch.device("cuda:1")

    q_gpu1 = q_slab.to(
        device=device1,
        dtype=torch.float16,
        non_blocking=True,
    )

    # These are normally already FP16 and on cuda:1. Keep the guards so the
    # helper remains explicit about the attention precision contract.
    if k_gpu1.device != device1 or k_gpu1.dtype != torch.float16:
        k_gpu1 = k_gpu1.to(
            device=device1,
            dtype=torch.float16,
            non_blocking=True,
        )
    if v_gpu1.device != device1 or v_gpu1.dtype != torch.float16:
        v_gpu1 = v_gpu1.to(
            device=device1,
            dtype=torch.float16,
            non_blocking=True,
        )

    with torch.nn.attention.sdpa_kernel(
        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION
    ):
        out = F.scaled_dot_product_attention(
            q_gpu1,
            k_gpu1,
            v_gpu1,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

    # Latest shape-contract fix: flatten_attention_output() expects HND rank 4.
    # Preserve the validated rank-3 -> HND conversion for a backend returning
    # [B,N,H*D].
    if out.ndim == 3:
        width = int(out.shape[-1])
        if width != expected_width:
            raise RuntimeError(
                "H3 T4 efficient attention rank-3 output width %d does not "
                "match heads*head_dim=%d"
                % (width, expected_width)
            )
        out = out.reshape(
            int(out.shape[0]),
            int(out.shape[1]),
            heads,
            head_dim,
        ).transpose(1, 2).contiguous()
    elif out.ndim != 4:
        raise RuntimeError(
            "H3 T4 efficient attention returned rank-%d output; expected HND rank 4"
            % out.ndim
        )

    if (
        out.shape[1] != heads
        or out.shape[-1] != head_dim
        or out.shape[-2] != q_slab.shape[-2]
    ):
        raise RuntimeError(
            "H3 T4 efficient attention returned invalid HND output shape %s; "
            "expected [B,%d,%d,%d]"
            % (
                tuple(out.shape),
                heads,
                int(q_slab.shape[-2]),
                head_dim,
            )
        )

    return out.to(
        device=device0,
        dtype=original_dtype,
        non_blocking=True,
    )


def _finish_streamed_dense_bf16_projected(
    module,
    projected,
    *,
    layer_index,
    transformer_options,
    out_projection=None,
):
    """Consume Q slabs against complete BF16 K/V and project each output slab."""
    output = attention_output_buffer(projected.x)
    external_consumer = get_streamed_h3_qkv_consumer(transformer_options)
    k_gpu1 = None
    v_gpu1 = None
    try:
        if external_consumer is None:
            import torch

            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                raise RuntimeError(
                    "H3 T4 dual-GPU attention requires at least two CUDA devices"
                )

            # Keep complete K/V on cuda:1. Q slabs remain source-native on cuda:0
            # until each bounded slab is transferred for efficient attention.
            k_gpu1 = projected.k.to(
                device="cuda:1",
                dtype=torch.float16,
                non_blocking=True,
            )
            v_gpu1 = projected.v.to(
                device="cuda:1",
                dtype=torch.float16,
                non_blocking=True,
            )

        for start, end, q in projected.stream_q():
            if external_consumer is None:
                raw = _minimax_h3_t4_efficient_attention(
                    module,
                    q,
                    k_gpu1,
                    v_gpu1,
                )
            else:
                raw = consume_streamed_h3_qkv(
                    external_consumer,
                    q,
                    projected.k,
                    projected.v,
                    q_start=start,
                    q_total=projected.sequence,
                    layer_index=layer_index,
                    transformer_options=transformer_options,
                )
            out = flatten_attention_output(
                module,
                raw,
                "streamed_dense_bf16",
            )
            del raw
            with diagnostics.stage("attention_out"):
                output[start:end].copy_(
                    _project_attention_output(
                        module,
                        out.squeeze(0),
                        out_projection,
                    )
                )
            del q, out
        return output
    finally:
        k_gpu1 = None
        v_gpu1 = None
        projected.release()
'''

    import ast

    try:
        tree = ast.parse(attention_text, filename=str(attention))
    except SyntaxError as exc:
        raise RuntimeError(
            f"Cannot parse H3 attention source before patching: {exc}"
        ) from exc

    target_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_finish_streamed_dense_bf16_projected"
    ]
    if len(target_nodes) != 1:
        raise RuntimeError(
            "Expected exactly one _finish_streamed_dense_bf16_projected() "
            "definition in the locked H3 attention source."
        )
    target_node = target_nodes[0]

    lines = attention_text.splitlines(keepends=True)
    start_index = target_node.lineno - 1
    end_index = target_node.end_lineno
    before = "".join(lines[:start_index])
    after = "".join(lines[end_index:])

    # The helper/function replacement is intentionally source-level and scoped
    # to the locked function boundary. This removes any old helper injected in
    # a live notebook and installs exactly one current implementation.
    attention_text = before + new_streamed_function + "\n" + after

    compile(attention_text, str(attention), "exec")
    attention.write_text(attention_text, encoding="utf-8")
    print("[H3 T4 GPU PATCH] attention_forward.py patched: PASS")

    # Final verification: the new efficient attention is present and the
    # streamed function calls it.
    attention_text = attention.read_text(encoding="utf-8")
    required = (
        helper_marker,
        "def _minimax_h3_t4_efficient_attention(",
        "SDPBackend.EFFICIENT_ATTENTION",
        "dtype=torch.float16",
        "rank-3 -> HND",
        "def _finish_streamed_dense_bf16_projected(",
        "_minimax_h3_t4_efficient_attention(",
    )
    missing = [needle for needle in required if needle not in attention_text]
    if missing:
        raise RuntimeError(
            "H3 T4 GPU attention patch verification failed:\n"
            + "\n".join(missing)
        )

    compile(attention_text, str(attention), "exec")
    print("[H3 T4 GPU PATCH] final verification: PASS")
    print("[H3 T4 GPU PATCH] GPU1 efficient attention + HND shape fix persisted")


def patch_h3_embedding_memory_compatibility() -> None:
    # Enable H3-Optimizations 0.2.41 embedding release with approved T4 _forward changes.
    node_dir = CUSTOM / "H3-Optimizations"
    target = node_dir / "h3_optimizations" / "memory" / "embedding.py"
    expected_revision = "379f9c7922b3d7831dd93ae069ba0cb82cb4cf36"
    expected_comfy_revision = "12d5279438bfefc058a269eae805ceab6047777f"
    expected_upstream_hash = "14bdfccd6860f252005b8d43ab446aa9a938a13dc819061724b8f914218f5fd1"
    marker = "# MINIMAX_H3_PROJECT_T4_MEMORY_COMPAT_V3"

    if not node_dir.is_dir():
        raise RuntimeError(f"H3-Optimizations runtime directory is missing: {node_dir}")
    actual_revision = subprocess.check_output(
        ["git", "-C", str(node_dir), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            "Refusing H3 embedding-memory compatibility patch because the installed "
            f"H3-Optimizations revision is {actual_revision}, expected {expected_revision}."
        )
    if not target.is_file():
        raise RuntimeError(f"H3 embedding optimizer source is missing: {target}")

    actual_comfy_revision = subprocess.check_output(
        ["git", "-C", str(COMFY), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_comfy_revision != expected_comfy_revision:
        raise RuntimeError(
            "Refusing H3 embedding-memory compatibility patch because the installed "
            f"ComfyUI revision is {actual_comfy_revision}, expected {expected_comfy_revision}."
        )

    source = target.read_text(encoding="utf-8")
    if marker in source:
        # Idempotent path: a marker alone is not trusted. Validate the complete
        # generated contract before accepting an already-patched installation.
        existing_contract = (
            marker,
            "def make_forward(model, original_forward, project_t4=False):",
            "project_t4 = _validate_upstream_forward(original)",
            "make_forward(model, original, project_t4=project_t4)",
            "options['h3_optimizations_preserved_embedding_patch'] = project_t4",
            "if project_t4:",
            "embed_dtype = torch.float32 if dtype == torch.float16 else dtype",
            "text_states = model.condition_proj(text_states.to(torch.float32))",
            "residual_dtype = torch.float32 if dtype == torch.float16 else dtype",
        )
        missing = [item for item in existing_contract if item not in source]
        if missing:
            raise RuntimeError(
                "H3 embedding-memory marker exists but the patched source contract "
                "is incomplete; refusing to continue:\n" + "\n".join(missing)
            )
        compile(source, str(target), "exec")
        print("[H3 OPT MEMORY] existing T4 embedding-memory compatibility: VERIFIED")
        return

    required = (
        f"UPSTREAM_FORWARD_SHA256 = '{expected_upstream_hash}'",
        "def _validate_upstream_forward(forward):",
        "def make_forward(model, original_forward):",
        "        video_embed = model.video_patch_proj(all_video_rows).to(dtype)\n        audio_embed = model.audio_patch_proj(all_audio_rows).to(dtype)",
        "        h = torch.empty(layout.seq_len, model.hidden_size, dtype=dtype, device=device)",
        "        if text_states.shape[-1] != model.hidden_size:\n            text_states = model.token_refiner(model.condition_proj(text_states),\n                                              transformer_options=transformer_options)",
        "    model_patcher.add_object_patch(FORWARD_KEY, make_forward(model, original))",
    )
    missing = [needle for needle in required if needle not in source]
    if missing:
        raise RuntimeError(
            "H3-Optimizations 0.2.41 embedding.py no longer matches the locked "
            "source contract; refusing to patch:\n" + "\n".join(missing)
        )

    old_validate = """def _validate_upstream_forward(forward):
    try:
        digest = _source_digest(forward)
    except (OSError, TypeError) as exc:
        raise H3EmbeddingMemoryPatchError(
            'cannot inspect MiniMax H3 _forward for embedding-memory compatibility'
        ) from exc
    if digest != UPSTREAM_FORWARD_SHA256:
        raise H3EmbeddingMemoryPatchError(
            'MiniMax H3 _forward changed; refusing the experimental embedding-memory patch'
        )
"""
    new_validate = f"""def _validate_upstream_forward(forward):
    try:
        source = inspect.getsource(forward)
        digest = hashlib.sha256(source.encode()).hexdigest()
    except (OSError, TypeError) as exc:
        raise H3EmbeddingMemoryPatchError(
            'cannot inspect MiniMax H3 _forward for embedding-memory compatibility'
        ) from exc
    if digest == UPSTREAM_FORWARD_SHA256:
        return False

    # The project applies exactly three T4-only source changes to the locked
    # ComfyUI 0.34 _forward. Reverse only those changes, then require the
    # resulting source to hash to the exact upstream implementation.
    normalized = source
    reverse_pairs = (
        (
            "            # CRITICAL H3 T4 boundary: condition_proj must receive FP32 input.\\n"
            "            text_states = self.condition_proj(text_states.to(torch.float32))\\n"
            "            text_states = self.token_refiner(\\n"
            "                text_states,\\n"
            "                transformer_options=transformer_options,\\n"
            "            )",
            "            text_states = self.token_refiner(self.condition_proj(text_states),\\n"
            "                                             transformer_options=transformer_options)",
        ),
        (
            "        embed_dtype = torch.float32 if dtype == torch.float16 else dtype\\n"
            "        video_embed = self.video_patch_proj(all_video_rows).to(embed_dtype)\\n"
            "        audio_embed = self.audio_patch_proj(all_audio_rows).to(embed_dtype)",
            "        video_embed = self.video_patch_proj(all_video_rows).to(dtype)\\n"
            "        audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)",
        ),
        (
            "        residual_dtype = torch.float32 if dtype == torch.float16 else dtype\\n"
            "        h = torch.empty(layout.seq_len, self.hidden_size, dtype=residual_dtype, device=device)",
            "        h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)",
        ),
    )
    replacements = 0
    for project_text, upstream_text in reverse_pairs:
        count = normalized.count(project_text)
        if count != 1:
            raise H3EmbeddingMemoryPatchError(
                'MiniMax H3 _forward did not contain exactly one instance of each approved T4 change'
            )
        normalized = normalized.replace(project_text, upstream_text, 1)
        replacements += 1
    if replacements != 3:
        raise H3EmbeddingMemoryPatchError(
            'MiniMax H3 _forward did not contain exactly the three approved project T4 changes'
        )

    normalized_hash = hashlib.sha256(normalized.encode()).hexdigest()
    if normalized_hash != UPSTREAM_FORWARD_SHA256:
        raise H3EmbeddingMemoryPatchError(
            'MiniMax H3 _forward differs from the locked upstream implementation '
            'beyond the three approved T4 changes'
        )
    return True
"""

    def replace_once(text: str, old: str, new: str, label: str) -> str:
        count = text.count(old)
        if count != 1:
            raise RuntimeError(
                f"H3 embedding-memory source contract for {label} matched {count} times; expected exactly once."
            )
        return text.replace(old, new, 1)

    patched = replace_once(source, old_validate, new_validate, "_validate_upstream_forward")
    patched = replace_once(
        patched,
        "def make_forward(model, original_forward):",
        "def make_forward(model, original_forward, project_t4=False):",
        "make_forward signature",
    )
    patched = replace_once(
        patched,
        "        video_embed = model.video_patch_proj(all_video_rows).to(dtype)\n"
        "        audio_embed = model.audio_patch_proj(all_audio_rows).to(dtype)",
        "        if project_t4:\n"
        "            embed_dtype = torch.float32 if dtype == torch.float16 else dtype\n"
        "        else:\n"
        "            embed_dtype = dtype\n"
        "        video_embed = model.video_patch_proj(all_video_rows).to(embed_dtype)\n"
        "        audio_embed = model.audio_patch_proj(all_audio_rows).to(embed_dtype)",
        "embedding dtype boundary",
    )
    patched = replace_once(
        patched,
        "        text_states = context[0]\n"
        "        if text_states.shape[-1] != model.hidden_size:\n"
        "            text_states = model.token_refiner(model.condition_proj(text_states),\n"
        "                                              transformer_options=transformer_options)",
        "        text_states = context[0]\n"
        "        if text_states.shape[-1] != model.hidden_size:\n"
        "            if project_t4:\n"
        "                text_states = model.condition_proj(text_states.to(torch.float32))\n"
        "                text_states = model.token_refiner(\n"
        "                    text_states,\n"
        "                    transformer_options=transformer_options,\n"
        "                )\n"
        "            else:\n"
        "                text_states = model.token_refiner(model.condition_proj(text_states),\n"
        "                                                  transformer_options=transformer_options)",
        "text conditioning boundary",
    )
    patched = replace_once(
        patched,
        "        h = torch.empty(layout.seq_len, model.hidden_size, dtype=dtype, device=device)",
        "        if project_t4:\n"
        "            residual_dtype = torch.float32 if dtype == torch.float16 else dtype\n"
        "        else:\n"
        "            residual_dtype = dtype\n"
        "        h = torch.empty(layout.seq_len, model.hidden_size, dtype=residual_dtype, device=device)",
        "residual dtype boundary",
    )
    patched = replace_once(
        patched,
        "        _validate_upstream_forward(original)\n",
        "        project_t4 = _validate_upstream_forward(original)\n",
        "validation result",
    )
    patched = replace_once(
        patched,
        "    model_patcher.add_object_patch(FORWARD_KEY, make_forward(model, original))",
        "    model_patcher.add_object_patch(FORWARD_KEY, make_forward(model, original, project_t4=project_t4))",
        "optimized forward installation",
    )
    patched = replace_once(
        patched,
        "    options.pop(FALLBACK_REASON_KEY, None)\n    options['h3_optimizations_preserved_embedding_patch'] = False\n    return True",
        "    options.pop(FALLBACK_REASON_KEY, None)\n"
        "    options['h3_optimizations_preserved_embedding_patch'] = project_t4\n"
        "    logging.info(\n"
        "        '[H3 Optimizations] embedding-memory compatibility: %s',\n"
        "        'project_t4' if project_t4 else 'upstream_0.34',\n"
        "    )\n"
        "    return True",
        "compatibility logging",
    )
    patched = replace_once(
        patched,
        "FALLBACK_REASON_KEY = 'h3_optimizations_embedding_memory_fallback'\n",
        "FALLBACK_REASON_KEY = 'h3_optimizations_embedding_memory_fallback'\n" + marker + "\n",
        "compatibility marker",
    )

    compile(patched, str(target), "exec")
    target.write_text(patched, encoding="utf-8")

    final_source = target.read_text(encoding="utf-8")
    final_checks = (
        marker,
        "def make_forward(model, original_forward, project_t4=False):",
        "project_t4 = _validate_upstream_forward(original)",
        "make_forward(model, original, project_t4=project_t4)",
        "options['h3_optimizations_preserved_embedding_patch'] = project_t4",
    )
    missing = [item for item in final_checks if item not in final_source]
    if missing:
        raise RuntimeError(
            "H3 embedding-memory compatibility post-write verification failed:\n"
            + "\n".join(missing)
        )
    print("[H3 OPT MEMORY] project T4 embedding-memory compatibility: PASS")


def verify_h3_optimization_runtime(runtime: dict) -> None:
    """Validate H3 optimization in a fresh Python process.
    Kaggle notebooks may already have imported a different Torch/CUDA build in
    the long-lived kernel. Native CUDA Python modules must not be hot-reloaded
    after pip replaces their shared objects, so the capability import is done
    in a clean child interpreter.
    """
    cfg = dict(runtime.get("h3_optimization", {}) or {})
    expected_revision = str(cfg.get("revision", "")).strip()
    expected_version = str(cfg.get("version", "")).strip()
    node_dir = CUSTOM / "H3-Optimizations"
    if not node_dir.is_dir():
        raise RuntimeError(f"H3-Optimizations runtime directory is missing: {node_dir}")
    if len(expected_revision) != 40:
        raise RuntimeError("runtime_versions.yaml contains no valid H3-Optimizations SHA")
    actual_revision = subprocess.check_output(
        ["git", "-C", str(node_dir), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            "Installed H3-Optimizations revision does not match the runtime lock: "
            f"expected={expected_revision}, actual={actual_revision}"
        )
    library_dirs = _cuda_library_dirs()
    if not library_dirs:
        raise RuntimeError(
            "Fresh H3 verification could not locate the installed CUDA runtime libraries."
        )
    environment = _configure_cuda_environment(library_dirs)
    environment.update({
        "H3_EXPECTED_TORCH": "2.10.0+cu130",
        "H3_EXPECTED_CUDA": "13.0",
        "H3_EXPECTED_H3_REVISION": expected_revision,
        "H3_EXPECTED_H3_VERSION": expected_version,
        "H3_COMFY_ROOT": str(COMFY),
        "H3_NODE_ROOT": str(node_dir),
    })
    verify_script = """
import os
import subprocess
import sys
from pathlib import Path
expected_torch = os.environ["H3_EXPECTED_TORCH"]
expected_cuda = os.environ["H3_EXPECTED_CUDA"]
expected_revision = os.environ["H3_EXPECTED_H3_REVISION"]
expected_version = os.environ["H3_EXPECTED_H3_VERSION"]
comfy = Path(os.environ["H3_COMFY_ROOT"]).resolve()
node_dir = Path(os.environ["H3_NODE_ROOT"]).resolve()
import torch
actual_torch = str(torch.__version__)
actual_cuda = str(torch.version.cuda)
if actual_torch != expected_torch:
    raise RuntimeError(f"Torch mismatch: expected={expected_torch}, actual={actual_torch}")
if actual_cuda != expected_cuda:
    raise RuntimeError(f"Torch CUDA mismatch: expected={expected_cuda}, actual={actual_cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("Torch CUDA is unavailable in the fresh verification process")
major, minor = torch.cuda.get_device_capability(0)
if (major, minor) != (7, 5):
    raise RuntimeError(f"Unexpected GPU capability: {(major, minor)}; expected Tesla T4 SM75")
sys.path.insert(0, str(node_dir))
sys.path.insert(0, str(comfy))
import h3_optimizations
from h3_optimizations.memory import forward as h3_forward
from h3_optimizations.memory import linear as h3_linear
from h3_optimizations.qkv import providers as h3_providers
package_file = Path(getattr(h3_optimizations, "__file__", "")).resolve()
expected_package_root = (node_dir / "h3_optimizations").resolve()
if not package_file.is_relative_to(expected_package_root):
    raise RuntimeError(
        "H3-Optimizations import resolved outside the pinned custom-node directory: "
        f"{package_file}"
    )
package_version = str(getattr(h3_optimizations, "__version__", "")).strip()
if expected_version and package_version != expected_version:
    raise RuntimeError(
        "Installed H3-Optimizations version does not match the runtime lock: "
        f"expected={expected_version}, actual={package_version}"
    )
required_symbols = (
    (h3_linear, "ConvRotTwoSliceMLP"),
    (h3_linear, "HeldMLP"),
    (h3_linear, "acquire_linear"),
    (h3_linear, "bind_convrot_mlp"),
    (h3_forward, "make_forward"),
    (h3_forward, "iter_mod_chunks"),
    (h3_providers, "MLP_CONVROT_INT8_TWO_SLICE"),
    (h3_providers, "resolve_mlp_provider"),
)
missing = [name for module, name in required_symbols if not hasattr(module, name)]
if missing:
    raise RuntimeError(
        "H3-Optimizations bounded MLP capability check failed; missing symbols: "
        f"{missing}"
    )
provider_id = str(h3_providers.MLP_CONVROT_INT8_TWO_SLICE)
if provider_id != "convrot_int8_two_slice":
    raise RuntimeError(f"Unexpected H3 ConvRot MLP provider identifier: {provider_id!r}")
actual_revision = subprocess.check_output(
    ["git", "-C", str(node_dir), "rev-parse", "HEAD"],
    text=True,
).strip()
if actual_revision != expected_revision:
    raise RuntimeError(
        "Fresh-process H3 revision mismatch: "
        f"expected={expected_revision}, actual={actual_revision}"
    )
print(f"[FRESH RUNTIME] torch={actual_torch} cuda={actual_cuda} gpu=SM{major}{minor}")
print(
    "[H3 OPT] revision={} version={} bounded_mlp=PASS ConvRotTwoSliceMLP=PASS provider={}".format(
        actual_revision, package_version, provider_id
    )
)
print("[H3 OPT] fresh-process runtime capability check passed; no H3 model generation was run.")
"""
    verification = subprocess.run(
        [sys.executable, "-c", verify_script],
        env=environment,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.stdout:
        print(verification.stdout, end="")
    if verification.stderr:
        print(verification.stderr, end="")
    if verification.returncode != 0:
        raise RuntimeError(
            "Fresh-process H3 runtime verification failed.\n"
            + (verification.stdout or "")
            + (verification.stderr or "")
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
def _warn_if_torch_already_imported() -> None:
    """Warn when bootstrap is running in a process that already imported Torch."""
    if "torch" not in sys.modules:
        return
    print("=" * 80)
    print("[BOOTSTRAP WARNING] torch is already imported in this process.")
    print("Reinstalling PyTorch changes files on disk, not the native Torch runtime")
    print("already loaded into this Python process.")
    print("GPU-dependent validation/work must run in a FRESH subprocess after bootstrap.")
    print("Do NOT import or reload torch directly in this same kernel after reinstall.")
    print("=" * 80)
def main():
    _enforce_no_restart()
    _warn_if_torch_already_imported()
    ensure_kaggle_startup_wrapt()
    legacy_runtime = ROOT / ".h3_runtime_cu130"
    if legacy_runtime.exists(): shutil.rmtree(legacy_runtime, ignore_errors=True)
    runtime = load_yaml(RUNTIME_MANIFEST)
    director_model = Path(
        os.getenv(
            "H3_DIRECTOR_MODEL_PATH",
            str(runtime["director"]["model_path"]),
        )
    ).expanduser().resolve()
    print(
        "[DIRECTOR MODEL]",
        director_model,
    )
    install_base_requirements()
    install_comfyui(runtime)
    install_pytorch_runtime(runtime)
    install_director_runtime(
        runtime
    )
    install_storyboard_runtime(
        runtime
    )
    install_nodes()
    remove_legacy_context_ir_node()
    install_and_verify_pillow_runtime(runtime)
    patch_h3_bounded_qkv(runtime)
    patch_h3_t4_gpu_parallel_attention(runtime)
    apply_embedded_h3_runtime_overlay()
    patch_h3_embedding_memory_compatibility()
    verify_h3_optimization_runtime(runtime)
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

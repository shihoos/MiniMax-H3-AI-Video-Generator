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
def _cuda_library_dirs(runtime: dict) -> list[Path]:
    cuda_cfg = runtime["cuda_runtime"]
    runtime_version = str(cuda_cfg["runtime_version"]).strip()
    cublas_version = str(cuda_cfg["cublas_version"]).strip()
    if not runtime_version or not cublas_version:
        raise RuntimeError(
            "runtime_versions.yaml cuda_runtime.runtime_version and "
            "cuda_runtime.cublas_version are required."
        )
    runtime_major = runtime_version.split(".", 1)[0]
    cublas_major = cublas_version.split(".", 1)[0]
    if not runtime_major.isdigit() or not cublas_major.isdigit():
        raise RuntimeError(
            "runtime_versions.yaml cuda_runtime versions must begin with numeric major versions: "
            f"runtime_version={runtime_version!r}, cublas_version={cublas_version!r}."
        )
    directories = []
    for site_root in _site_packages():
        nvidia_root = (
            site_root
            / "nvidia"
        )
        if not nvidia_root.is_dir():
            continue
        for pattern in (
            f"libcudart.so.{runtime_major}*",
            f"libcublas.so.{cublas_major}*",
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
def ensure_kaggle_startup_wrapt(runtime: dict) -> None:
    """Provide Kaggle's sitecustomize dependency from the runtime lock."""
    wrapt_version = str(runtime["python"]["wrapt_version"]).strip()
    probe = subprocess.run(
        [sys.executable, "-c", "import wrapt; print(wrapt.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    installed_version = (probe.stdout or "").strip() if probe.returncode == 0 else ""
    if installed_version != wrapt_version:
        run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-deps",
            f"wrapt=={wrapt_version}",
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
    if verified_version != wrapt_version:
        raise RuntimeError(
            f"Kaggle wrapt version mismatch: expected={wrapt_version}, "
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
    if not cuda.startswith("cu") or not cuda[2:].isdigit():
        raise RuntimeError(
            "runtime_versions.yaml pytorch.cuda must use a cuNNN wheel tag; "
            f"got {cuda!r}."
        )
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
    cuda_digits = cuda[2:]
    if len(cuda_digits) != 3:
        raise RuntimeError(f"Unsupported PyTorch CUDA wheel tag: {cuda!r}")
    expected_torch = f"{version}+{cuda}"
    expected_cuda = f"{cuda_digits[:2]}.{cuda_digits[2:]}"
    verify = subprocess.run(
        [
            sys.executable, "-c",
            (
                "import torch; "
                f"assert torch.__version__ == {expected_torch!r}; "
                f"assert torch.version.cuda == {expected_cuda!r}; "
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
    wrapt_version = str(runtime["python"]["wrapt_version"]).strip()
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
        [uv, "pip", "install", "--python", str(venv_python), "--link-mode", "copy", f"vllm=={vllm_version}", f"wrapt=={wrapt_version}"],
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
    if observed_wrapt_version != wrapt_version:
        raise RuntimeError(
            "Director isolated runtime wrapt mismatch: "
            f"observed={observed_wrapt_version!r}, expected={wrapt_version!r}."
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

def prepare_locked_h3_optimization_source(runtime: dict) -> None:
    """Return the locked H3-Optimizations checkout to pristine source before patching."""
    node_dir = CUSTOM / "H3-Optimizations"
    expected_revision = str(runtime.get("h3_optimization", {}).get("revision", "")).strip()
    if not node_dir.is_dir():
        raise RuntimeError(f"H3-Optimizations runtime directory is missing: {node_dir}")
    if len(expected_revision) != 40:
        raise RuntimeError("runtime_versions.yaml contains no valid H3-Optimizations SHA")
    run("git", "-C", node_dir, "reset", "--hard", expected_revision)
    actual_revision = subprocess.check_output(
        ["git", "-C", str(node_dir), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            "H3-Optimizations source reset did not reach the locked revision: "
            f"expected={expected_revision}, actual={actual_revision}"
        )
    print("[H3 OPT] locked source reset: PASS")


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

def patch_h3_qkv_binding_lifetime(runtime: dict) -> None:
    """Keep the native ConvRot QKV binding alive while streamed Q is consumed."""
    node_dir = CUSTOM / "H3-Optimizations"
    target = node_dir / "h3_optimizations" / "qkv" / "bf16.py"
    expected_revision = "379f9c7922b3d7831dd93ae069ba0cb82cb4cf36"
    configured_revision = str(runtime.get("h3_optimization", {}).get("revision", "") or "").strip()
    if configured_revision != expected_revision:
        raise RuntimeError(
            "Refusing the H3 T4 QKV binding-lifetime patch because runtime_versions.yaml "
            "does not pin the source contract this patch targets: "
            f"expected={expected_revision}, configured={configured_revision or 'missing'}."
        )
    marker = "# MINIMAX_H3_T4_KEEP_CONVROT_BINDING_FOR_Q_STREAM"

    if not target.is_file():
        raise RuntimeError(f"H3 QKV BF16 source is missing: {target}")
    actual_revision = subprocess.check_output(
        ["git", "-C", str(node_dir), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            "Refusing the H3 T4 QKV binding-lifetime patch because the installed "
            f"H3-Optimizations revision is {actual_revision}, expected {expected_revision}."
        )

    source = target.read_text(encoding="utf-8")
    if marker in source:
        required = (marker, "held=held,", "binding_factory=None,")
        missing = [needle for needle in required if needle not in source]
        if missing:
            raise RuntimeError(
                "H3 T4 QKV binding-lifetime marker exists but the complete source "
                "contract is missing:\n" + "\n".join(missing)
            )
        compile(source, str(target), "exec")
        print("[H3 T4 QKV FIX] bf16.py already patched: VERIFIED")
        return

    old = (
        "            if _held_cast_handle(held) is not None:\n"
        "                held.__exit__(None, None, None)\n"
        "                held = None\n"
        "            return PreparedStreamedDenseBF16QKV(\n"
        "                x=x,\n"
        "                k=k_full,\n"
        "                v=v_full,\n"
        "                rope_freqs=rope_freqs,\n"
        "                held=held,\n"
        "                binding_factory=(binding_factory if held is None else None),\n"
        "                chunk_rows=self.chunk_rows,\n"
        "                projection_mode=self.projection_mode,\n"
        "            )\n"
    )
    replacement = (
        "            # MINIMAX_H3_T4_KEEP_CONVROT_BINDING_FOR_Q_STREAM\n"
        "            # Keep the native ConvRot-256 binding alive after K/V projection.\n"
        "            # PreparedStreamedDenseBF16QKV.stream_q() reuses it for every Q slab.\n"
        "            return PreparedStreamedDenseBF16QKV(\n"
        "                x=x,\n"
        "                k=k_full,\n"
        "                v=v_full,\n"
        "                rope_freqs=rope_freqs,\n"
        "                held=held,\n"
        "                binding_factory=None,\n"
        "                chunk_rows=self.chunk_rows,\n"
        "                projection_mode=self.projection_mode,\n"
        "            )\n"
    )
    if source.count(old) != 1:
        raise RuntimeError(
            "H3-Optimizations 0.2.41 qkv/bf16.py no longer matches the locked "
            "binding-lifetime source contract; refusing to patch."
        )
    patched = source.replace(old, replacement, 1)
    compile(patched, str(target), "exec")
    target.write_text(patched, encoding="utf-8")
    print("[H3 T4 QKV FIX] bf16.py patched: PASS")


def patch_h3_sage_attention(runtime: dict) -> None:
    """Route the existing bounded H3 Q slabs into the pinned SM75 Sage kernel.

    This preserves the project-owned QKV lifetime/chunking contract and the
    existing output-projection contract. Sage is used as the local per-worker
    attention engine; no cross-GPU Q/K/V transfers are introduced.
    """
    node_dir = CUSTOM / "H3-Optimizations"
    target = node_dir / "h3_optimizations" / "attention_forward.py"
    expected_revision = "379f9c7922b3d7831dd93ae069ba0cb82cb4cf36"
    configured_revision = str(runtime.get("h3_optimization", {}).get("revision", "") or "").strip()
    if configured_revision != expected_revision:
        raise RuntimeError(
            "Refusing the H3 T4 Sage-attention patch because runtime_versions.yaml "
            "does not pin the H3 source contract this patch targets: "
            f"expected={expected_revision}, configured={configured_revision or 'missing'}."
        )
    sage_cfg = dict(runtime.get("sage_attention", {}) or {})
    sage_version = str(sage_cfg.get("version", "") or "").strip()
    if "smooth_k" not in sage_cfg:
        raise RuntimeError("runtime_versions.yaml sage_attention.smooth_k is required.")
    smooth_k = sage_cfg["smooth_k"]
    if not isinstance(smooth_k, bool):
        raise RuntimeError("runtime_versions.yaml sage_attention.smooth_k must be boolean.")
    if "qk_quant_gran" not in sage_cfg:
        raise RuntimeError("runtime_versions.yaml sage_attention.qk_quant_gran is required.")
    qk_quant_gran = str(sage_cfg["qk_quant_gran"] or "").strip().lower()
    if qk_quant_gran != "per_warp":
        raise RuntimeError(
            "Production H3 T4 SageAttention requires qk_quant_gran='per_warp'; "
            f"got {qk_quant_gran!r}."
        )
    marker = "# MINIMAX_H3_T4_SAGE_ATTENTION"
    chunk_rows = int(runtime.get("h3_optimization", {}).get("chunk_rows", 0) or 0)

    if chunk_rows != 2560:
        raise RuntimeError(
            f"Production H3 Sage-attention patch requires chunk_rows=2560; got {chunk_rows}."
        )
    if not sage_version:
        raise RuntimeError("runtime_versions.yaml sage_attention.version is missing.")
    if not target.is_file():
        raise RuntimeError(f"H3 attention source is missing: {target}")

    actual_revision = subprocess.check_output(
        ["git", "-C", str(node_dir), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_revision != expected_revision:
        raise RuntimeError(
            "Refusing the H3 T4 Sage-attention patch because the installed "
            f"H3-Optimizations revision is {actual_revision}, expected {expected_revision}."
        )

    source = target.read_text(encoding="utf-8")
    if marker in source:
        required = (
            marker,
            "from sageattention import sageattn",
            "sageattn(",
            "tensor_layout='HND'",
            f"qk_quant_gran={qk_quant_gran!r}",
            f"smooth_k={smooth_k!r}",
            "SageAttention-SM75",
            "projected.release()",
        )
        missing = [needle for needle in required if needle not in source]
        if missing:
            raise RuntimeError(
                "H3 Sage-attention marker exists but the complete source contract is missing:\n"
                + "\n".join(missing)
            )
        compile(source, str(target), "exec")
        print("[H3 T4 SAGE ATTENTION] attention_forward.py already patched: VERIFIED")
        return

    marker_start = "def _finish_streamed_dense_bf16_projected("
    next_marker = "def make_forward("
    start = source.find(marker_start)
    end = source.find(next_marker, start)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(
            "H3-Optimizations 0.2.41 attention_forward.py no longer exposes the "
            "locked streamed-BF16 attention function boundary; refusing to patch."
        )

    new_function = '''def _finish_streamed_dense_bf16_projected(
    module,
    projected,
    *,
    layer_index,
    transformer_options,
    out_projection=None,
):
    """Consume bounded Q slabs with the pinned SM75 SageAttention kernel."""
    del layer_index, transformer_options
    import torch
    from sageattention import sageattn

    output = attention_output_buffer(projected.x)
    attention_dtype = torch.float16
    local_device = projected.k.device
    if local_device.type != 'cuda':
        raise RuntimeError(
            'H3 T4 SageAttention requires CUDA K/V tensors; got %s' % local_device
        )
    if int(module.head_dim) not in (64, 128):
        raise RuntimeError(
            'H3 T4 SageAttention SM75 supports head_dim 64/128; got %d'
            % int(module.head_dim)
        )
    if int(module.heads) != 56:
        raise RuntimeError(
            'MiniMax H3 T4 SageAttention integration expects 56 heads; got %d'
            % int(module.heads)
        )

    k_attention = projected.k.to(
        device=local_device,
        dtype=attention_dtype,
    ).contiguous()
    v_attention = projected.v.to(
        device=local_device,
        dtype=attention_dtype,
    ).contiguous()

    try:
        # MINIMAX_H3_T4_SAGE_ATTENTION
        if not getattr(_finish_streamed_dense_bf16_projected, '_h3_t4_logged', False):
            print(
                '[H3 T4 SAGE ATTENTION] '
                'backend=SageAttention-SM75 '
                'device=%s q_rows=%d heads=%d head_dim=%d dtype=%s '
                'smooth_k=__SAGE_SMOOTH_K_TEXT__ qk_quant_gran=__SAGE_QK_QUANT_GRAN_TEXT__'
                % (
                    local_device,
                    int(projected.chunk_rows),
                    int(module.heads),
                    int(module.head_dim),
                    attention_dtype,
                ),
                flush=True,
            )
            _finish_streamed_dense_bf16_projected._h3_t4_logged = True

        for start, end, q in projected.stream_q():
            local_rows = int(end - start)
            target_dtype = q.dtype
            q_attention = q.to(
                device=local_device,
                dtype=attention_dtype,
            ).contiguous()
            if q_attention.ndim != 4:
                raise RuntimeError(
                    'H3 T4 SageAttention expected rank-4 HND Q; got rank-%d'
                    % q_attention.ndim
                )
            if (
                int(q_attention.shape[0]) != 1
                or int(q_attention.shape[1]) != int(module.heads)
                or int(q_attention.shape[2]) != local_rows
                or int(q_attention.shape[3]) != int(module.head_dim)
            ):
                raise RuntimeError(
                    'H3 T4 SageAttention received invalid Q shape %s; '
                    'expected [1,%d,%d,%d]'
                    % (
                        tuple(q_attention.shape),
                        int(module.heads),
                        local_rows,
                        int(module.head_dim),
                    )
                )

            raw = sageattn(
                q_attention,
                k_attention,
                v_attention,
                tensor_layout='HND',
                is_causal=False,
                smooth_k=__SAGE_SMOOTH_K__,
                qk_quant_gran=__SAGE_QK_QUANT_GRAN__,
            )
            if raw.ndim != 4:
                raise RuntimeError(
                    'H3 T4 SageAttention returned rank-%d output; expected rank-4 HND'
                    % raw.ndim
                )
            if (
                int(raw.shape[0]) != 1
                or int(raw.shape[1]) != int(module.heads)
                or int(raw.shape[2]) != local_rows
                or int(raw.shape[3]) != int(module.head_dim)
            ):
                raise RuntimeError(
                    'H3 T4 SageAttention returned invalid HND shape %s; '
                    'expected [1,%d,%d,%d]'
                    % (
                        tuple(raw.shape),
                        int(module.heads),
                        local_rows,
                        int(module.head_dim),
                    )
                )

            raw = raw.to(dtype=target_dtype)
            out = flatten_attention_output(
                module,
                raw,
                'h3_t4_sage_attention',
            )
            del raw, q_attention
            with diagnostics.stage('attention_out'):
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
        del k_attention, v_attention
        projected.release()

'''
    new_function = new_function.replace("__SAGE_SMOOTH_K__", repr(smooth_k))
    new_function = new_function.replace("__SAGE_QK_QUANT_GRAN__", repr(qk_quant_gran))
    new_function = new_function.replace("__SAGE_SMOOTH_K_TEXT__", str(smooth_k))
    new_function = new_function.replace("__SAGE_QK_QUANT_GRAN_TEXT__", qk_quant_gran)
    patched = source[:start] + new_function + source[end:]
    compile(patched, str(target), "exec")
    target.write_text(patched, encoding="utf-8")
    print("[H3 T4 SAGE ATTENTION] attention_forward.py patched: PASS")


def patch_h3_embedding_memory_compatibility(runtime: dict) -> None:
    # Enable H3-Optimizations 0.2.41 embedding release with approved T4 _forward changes.
    node_dir = CUSTOM / "H3-Optimizations"
    target = node_dir / "h3_optimizations" / "memory" / "embedding.py"
    expected_revision = "379f9c7922b3d7831dd93ae069ba0cb82cb4cf36"
    expected_comfy_revision = "12d5279438bfefc058a269eae805ceab6047777f"
    configured_revision = str(runtime.get("h3_optimization", {}).get("revision", "") or "").strip()
    if configured_revision != expected_revision:
        raise RuntimeError(
            "Refusing H3 embedding-memory compatibility patch because runtime_versions.yaml "
            "does not pin the H3 source contract this patch targets: "
            f"expected={expected_revision}, configured={configured_revision or 'missing'}."
        )
    configured_comfy_revision = str(runtime.get("comfyui", {}).get("revision", "") or "").strip()
    if configured_comfy_revision != expected_comfy_revision:
        raise RuntimeError(
            "Refusing H3 embedding-memory compatibility patch because runtime_versions.yaml "
            "does not pin the ComfyUI source contract this patch targets: "
            f"expected={expected_comfy_revision}, configured={configured_comfy_revision or 'missing'}."
        )
    # Patch-contract fingerprint for the locked ComfyUI _forward after the approved H3 source transforms; this is not a runtime version pin.
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
    pytorch_cfg = dict(runtime.get("pytorch", {}) or {})
    pytorch_version = str(pytorch_cfg.get("version", "") or "").strip()
    pytorch_cuda = str(pytorch_cfg.get("cuda", "") or "").strip().lower()
    if not pytorch_version or not pytorch_cuda.startswith("cu") or not pytorch_cuda[2:].isdigit():
        raise RuntimeError("runtime_versions.yaml pytorch version/cuda configuration is incomplete.")
    cuda_digits = pytorch_cuda[2:]
    if len(cuda_digits) != 3:
        raise RuntimeError(f"Unsupported PyTorch CUDA wheel tag: {pytorch_cuda!r}")
    expected_torch = f"{pytorch_version}+{pytorch_cuda}"
    expected_cuda = f"{cuda_digits[:2]}.{cuda_digits[2:]}"
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
    library_dirs = _cuda_library_dirs(runtime)
    if not library_dirs:
        raise RuntimeError(
            "Fresh H3 verification could not locate the installed CUDA runtime libraries."
        )
    environment = _configure_cuda_environment(library_dirs)
    environment.update({
        "H3_EXPECTED_TORCH": expected_torch,
        "H3_EXPECTED_CUDA": expected_cuda,
        "H3_EXPECTED_H3_REVISION": expected_revision,
        "H3_EXPECTED_H3_VERSION": expected_version,
        "H3_EXPECTED_SAGE_VERSION": str(runtime["sage_attention"]["version"]).strip(),
        "H3_EXPECTED_SAGE_SMOOTH_K": str(bool(runtime["sage_attention"]["smooth_k"])).lower(),
        "H3_EXPECTED_SAGE_QK_GRAN": str(runtime["sage_attention"]["qk_quant_gran"]).strip().lower(),
        "H3_COMFY_ROOT": str(COMFY),
        "H3_NODE_ROOT": str(node_dir),
        "H3_SAGE_ROOT": str(CUSTOM / str(runtime["sage_attention"]["directory"])),
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
import sageattention
from sageattention import sageattn
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
from h3_optimizations import attention_forward as h3_attention
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
    (h3_attention, "make_forward"),
    (h3_attention, "_finish_streamed_dense_bf16_projected"),
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
attention_source = (node_dir / "h3_optimizations" / "attention_forward.py").read_text(encoding="utf-8")
expected_sage_smooth = os.environ["H3_EXPECTED_SAGE_SMOOTH_K"].strip().lower()
expected_sage_qk = os.environ["H3_EXPECTED_SAGE_QK_GRAN"].strip().lower()
required_attention_markers = (
    "# MINIMAX_H3_T4_SAGE_ATTENTION",
    "from sageattention import sageattn",
    f"qk_quant_gran={expected_sage_qk!r}",
    f"smooth_k={expected_sage_smooth == 'true'!r}",
)
missing_attention_markers = [m for m in required_attention_markers if m not in attention_source]
if missing_attention_markers:
    raise RuntimeError("H3 Sage attention source marker verification failed: " + ", ".join(missing_attention_markers))
# Pinned 9f1c9a29... package export contract is checked below:
# SM75_CUDA_ENABLED at package level + direct _fused extension import.
expected_sage_version = str(os.environ.get("H3_EXPECTED_SAGE_VERSION", "")).strip()
if expected_sage_version and str(getattr(sageattention, "__version__", "")).strip() != expected_sage_version:
    raise RuntimeError(
        f"SageAttention package mismatch: expected={expected_sage_version}, actual={getattr(sageattention, '__version__', '')}"
    )
sage_package = Path(getattr(sageattention, "__file__", "")).resolve()
expected_sage_root = Path(os.environ["H3_SAGE_ROOT"]).resolve()
if not sage_package.is_relative_to(expected_sage_root / "sageattention"):
    raise RuntimeError(f"SageAttention imported outside pinned checkout: {sage_package}")
# Match install_sageattention_sm75(): the pinned package exports
# SM75_CUDA_ENABLED; the fused extension is verified by direct import.
if not bool(getattr(sageattention, "SM75_CUDA_ENABLED", False)):
    raise RuntimeError("SageAttention SM75 CUDA extension is not enabled in fresh H3 verification")
try:
    from sageattention import _fused as _sage_fused
except Exception as exc:
    raise RuntimeError(
        "SageAttention fused CUDA extension is not importable in fresh H3 verification"
    ) from exc
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
    "[H3 OPT] revision={} version={} bounded_mlp=PASS ConvRotTwoSliceMLP=PASS provider={} sage=PASS source_marker=PASS".format(
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
            if node["name"] == "H3-Optimizations":
                run("git", "-C", destination, "reset", "--hard")
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
def install_sageattention_sm75(runtime: dict) -> None:
    """Clone, apply the proven SM75 fragment-mapping fix, build, and verify the pinned fork."""
    cfg = dict(runtime.get("sage_attention", {}) or {})
    repository = str(cfg.get("repository", "") or "").strip()
    revision = str(cfg.get("revision", "") or "").strip()
    expected_version = str(cfg.get("version", "") or "").strip()
    directory = str(cfg.get("directory", "") or "").strip()
    if not directory:
        raise RuntimeError("runtime_versions.yaml sage_attention.directory is required.")
    install_dir = CUSTOM / directory
    if not repository or len(revision) != 40:
        raise RuntimeError("runtime_versions.yaml sage_attention.repository/revision are required and must be SHA-pinned.")
    if not expected_version:
        raise RuntimeError("runtime_versions.yaml sage_attention.version is required.")

    CUSTOM.mkdir(parents=True, exist_ok=True)
    if install_dir.exists() and not (install_dir / ".git").is_dir():
        raise RuntimeError(f"SageAttention path exists but is not a git checkout: {install_dir}")
    if not install_dir.exists():
        run("git", "clone", repository, install_dir)

    run("git", "-C", install_dir, "fetch", "--all", "--tags", "--prune")
    run("git", "-C", install_dir, "reset", "--hard", revision)
    actual_revision = subprocess.check_output(
        ["git", "-C", str(install_dir), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()
    if actual_revision != revision:
        raise RuntimeError(
            "SageAttention checkout mismatch: "
            f"expected={revision}, actual={actual_revision}"
        )

    # The pinned SM75 fork currently launches CTA_K=64 with WARP_K=16. That
    # creates four independent K warps for each Q warp, but the kernel keeps
    # softmax state and final O ownership at Q-warp scope. There is no
    # cross-K-warp reduction, and the direct output address does not include
    # warp_idx_k, so the four K warps race on the same output rows. The result
    # is numerically invalid even when Q/K quantization and scale contracts are
    # correct. Pair each Q warp with the full CTA_K tile so softmax state and
    # output ownership are warp-local and unique.
    sage_kernel = install_dir / "csrc" / "qattn" / "attn_cuda_sm75.h"
    if not sage_kernel.is_file():
        raise RuntimeError(f"Pinned SageAttention SM75 kernel source is missing: {sage_kernel}")
    sage_source = sage_kernel.read_text(encoding="utf-8")
    legacy = "constexpr int WARP_K_SM75 = 16;"
    fixed = "constexpr int WARP_K_SM75 = 64;"
    patch_marker = "// H3-T4-SM75-KERNEL-FIX: pair each Q warp with the full CTA_K tile to eliminate cross-K-warp softmax/output races.\n"
    fixed_count = sage_source.count(fixed)
    marker_count = sage_source.count("H3-T4-SM75-KERNEL-FIX")
    legacy_count = sage_source.count(legacy)
    if fixed_count == 2 and marker_count == 2 and legacy_count == 0:
        print("[SAGE SM75 PATCH] WARP_K_SM75=64 runtime kernel correction: ALREADY_APPLIED")
    elif fixed_count == 0 and marker_count == 0 and legacy_count == 2:
        sage_source = sage_source.replace(
            legacy,
            patch_marker + fixed,
        )
        sage_kernel.write_text(sage_source, encoding="utf-8")
        print("[SAGE SM75 PATCH] WARP_K_SM75=64 runtime kernel correction: PASS")
    else:
        raise RuntimeError(
            "Pinned SageAttention SM75 kernel source contract changed: expected exactly two "
            "WARP_K_SM75=16 declarations before applying the H3 T4 runtime fix, or exactly two "
            "already-marked WARP_K_SM75=64 declarations after the fix."
        )
    if fixed not in sage_kernel.read_text(encoding="utf-8"):
        raise RuntimeError("SageAttention SM75 WARP_K runtime correction did not persist.")

    sage_source = sage_kernel.read_text(encoding="utf-8")

    # H3-T4-SM75-PV-FRAGMENT-FIX:
    # NVIDIA's SM75 m16n8k8 FP32 accumulator fragment maps c0/c1 to
    # row=groupID and c2/c3 to row=groupID+8. The pinned fork treated the
    # two row pairs as adjacent rows (2*r, 2*r+1) during online-softmax
    # renormalization, final normalization, and output storage. That makes
    # the PV result and its normalization belong to different Q rows.
    # Correct the mapping without changing the tensor-core kernels or their
    # tiling; this is an index/ownership fix and does not add a slower path.
    pv_marker = "H3-T4-SM75-PV-FRAGMENT-FIX"
    if pv_marker not in sage_source:
        old_renorm = """            // CRITICAL FIX: Renormalize RO AFTER computing both mq but BEFORE PV MMA
            // This ensures: O_new = exp(m_old - m_new) * O_old + P @ V
            // Each thread writes 2 output rows (2*(lane_id/4) and 2*(lane_id/4)+1).
            // Use __shfl_sync to fetch m_old/m_i from the lane that computed
            // the matching Q row (groupOf=4: src_lane = (row % 8) * 4).
            {
                uint32_t thread_row0 = (lane_id / 4) * 2;
                uint32_t thread_row1 = thread_row0 + 1;

                uint32_t src_lane0 = (thread_row0 % 8) * 4;
                uint32_t src_lane1 = (thread_row1 % 8) * 4;

                float m_old_0 = (thread_row0 < 8) ? __shfl_sync(0xffffffff, m_old[0], src_lane0) : __shfl_sync(0xffffffff, m_old[1], src_lane0);
                float m_i_0   = (thread_row0 < 8) ? __shfl_sync(0xffffffff, m_i[0],   src_lane0) : __shfl_sync(0xffffffff, m_i[1],   src_lane0);
                float m_old_1 = (thread_row1 < 8) ? __shfl_sync(0xffffffff, m_old[0], src_lane1) : __shfl_sync(0xffffffff, m_old[1], src_lane1);
                float m_i_1   = (thread_row1 < 8) ? __shfl_sync(0xffffffff, m_i[0],   src_lane1) : __shfl_sync(0xffffffff, m_i[1],   src_lane1);

                float o_scale_0 = math::ptx_exp2(m_old_0 - m_i_0);
                float o_scale_1 = math::ptx_exp2(m_old_1 - m_i_1);

                #pragma unroll
                for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
                    RO_accum[fk * 4 + 0] *= o_scale_0;
                    RO_accum[fk * 4 + 1] *= o_scale_0;
                    RO_accum[fk * 4 + 2] *= o_scale_1;
                    RO_accum[fk * 4 + 3] *= o_scale_1;
                }
            }
"""
        new_renorm = """            // H3-T4-SM75-PV-FRAGMENT-FIX
            // m16n8k8 accumulator layout: c0/c1 -> row=groupID,
            // c2/c3 -> row=groupID+8. m_old[0]/m_i[0] already belong to
            // the top 8 Q rows and m_old[1]/m_i[1] to the bottom 8 rows.
            {
                float o_scale_top = math::ptx_exp2(m_old[0] - m_i[0]);
                float o_scale_bottom = math::ptx_exp2(m_old[1] - m_i[1]);

                #pragma unroll
                for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
                    RO_accum[fk * 4 + 0] *= o_scale_top;
                    RO_accum[fk * 4 + 1] *= o_scale_top;
                    RO_accum[fk * 4 + 2] *= o_scale_bottom;
                    RO_accum[fk * 4 + 3] *= o_scale_bottom;
                }
            }
"""
        if old_renorm not in sage_source:
            raise RuntimeError("SM75 PV renormalization source contract changed; refusing to patch.")
        sage_source=sage_source.replace(old_renorm,new_renorm,1)

        old_final = """    // --- Final Normalization ---
    // Each thread writes 2 output rows (2*(lane_id/4) and 2*(lane_id/4)+1).
    // Use __shfl_sync to fetch l_i from the lane group that computed
    // the matching Q row. src_lane = (row % 8) * 4 (groupOf=4 layout).
    {
        uint32_t thread_row0 = (lane_id / 4) * 2;
        uint32_t thread_row1 = thread_row0 + 1;

        uint32_t src_lane0 = (thread_row0 % 8) * 4;
        float l_i_0 = (thread_row0 < 8) ? __shfl_sync(0xffffffff, l_i[0], src_lane0) : __shfl_sync(0xffffffff, l_i[1], src_lane0);

        uint32_t src_lane1 = (thread_row1 % 8) * 4;
        float l_i_1 = (thread_row1 < 8) ? __shfl_sync(0xffffffff, l_i[0], src_lane1) : __shfl_sync(0xffffffff, l_i[1], src_lane1);

        float l_rcp_0 = (l_i_0 > 0.0f) ? math::ptx_rcp(l_i_0) : 0.0f;
        float l_rcp_1 = (l_i_1 > 0.0f) ? math::ptx_rcp(l_i_1) : 0.0f;

        #pragma unroll
        for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
            RO_accum[fk * 4 + 0] *= l_rcp_0;
            RO_accum[fk * 4 + 1] *= l_rcp_0;
            RO_accum[fk * 4 + 2] *= l_rcp_1;
            RO_accum[fk * 4 + 3] *= l_rcp_1;
        }
    }
"""
        new_final = """    // --- Final Normalization ---
    // m16n8k8 maps c0/c1 to row=groupID and c2/c3 to row=groupID+8.
    {
        float l_rcp_top = (l_i[0] > 0.0f) ? math::ptx_rcp(l_i[0]) : 0.0f;
        float l_rcp_bottom = (l_i[1] > 0.0f) ? math::ptx_rcp(l_i[1]) : 0.0f;

        #pragma unroll
        for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
            RO_accum[fk * 4 + 0] *= l_rcp_top;
            RO_accum[fk * 4 + 1] *= l_rcp_top;
            RO_accum[fk * 4 + 2] *= l_rcp_bottom;
            RO_accum[fk * 4 + 3] *= l_rcp_bottom;
        }
    }
"""
        if old_final not in sage_source:
            raise RuntimeError("SM75 PV final-normalization source contract changed; refusing to patch.")
        sage_source=sage_source.replace(old_final,new_final,1)

        old_smem = """        // For m16n8k8, each thread covers a 2×2 block of the 16×8 output.
        // Thread i = lane_id: rows = 2*(i/4) and 2*(i/4)+1, cols = 2*(i%4) and 2*(i%4)+1
        uint32_t thread_row0 = (lane_id / 4) * 2;
        uint32_t thread_row1 = thread_row0 + 1;
        uint32_t thread_col0 = (lane_id % 4) * 2;
        uint32_t thread_col1 = thread_col0 + 1;
"""
        new_smem = """        // H3-T4-SM75-PV-FRAGMENT-FIX: c0/c1 are row=r, c2/c3 are row=r+8.
        uint32_t thread_row0 = lane_id / 4;
        uint32_t thread_row1 = thread_row0 + 8;
        uint32_t thread_col0 = (lane_id % 4) * 2;
        uint32_t thread_col1 = thread_col0 + 1;
"""
        if old_smem not in sage_source:
            raise RuntimeError("SM75 PV shared-memory output mapping source contract changed; refusing to patch.")
        sage_source=sage_source.replace(old_smem,new_smem,1)

        old_direct = """        // Path B: Direct scattered write using the 2×2 per-thread mapping
        #pragma unroll
        for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
            uint32_t col_base = fk * MMA_SV_N_SM75;
            uint32_t thread_row0 = o_start_row_warp + (lane_id / 4) * 2;
            uint32_t thread_row1 = thread_row0 + 1;
            uint32_t thread_col0 = col_base + (lane_id % 4) * 2;
            uint32_t thread_col1 = thread_col0 + 1;

            // Check bounds and write (only write if global row < qo_len)
            if (thread_row0 < qo_len + o_start_row_warp) {
                uint32_t global_row0 = q_start_row_block + thread_row0;
                if (global_row0 < qo_len) {
                    uint32_t o_offset = batch_id * stride_bz_o + head_id * stride_h_o + global_row0 * stride_seq_o;
                    O[o_offset + thread_col0] = __float2half_rn(RO_accum[fk * 4 + 0]);
                    O[o_offset + thread_col1] = __float2half_rn(RO_accum[fk * 4 + 1]);
                }
            }
            if (thread_row1 < qo_len + o_start_row_warp) {
                uint32_t global_row1 = q_start_row_block + thread_row1;
                if (global_row1 < qo_len) {
                    uint32_t o_offset = batch_id * stride_bz_o + head_id * stride_h_o + global_row1 * stride_seq_o;
                    O[o_offset + thread_col0] = __float2half_rn(RO_accum[fk * 4 + 2]);
                    O[o_offset + thread_col1] = __float2half_rn(RO_accum[fk * 4 + 3]);
                }
            }
        }
"""
        new_direct = """            // Path B: Direct scattered write using the SM75 fragment row mapping.
        #pragma unroll
        for(int fk = 0; fk < NUM_N_V_TILES; ++fk) {
            uint32_t col_base = fk * MMA_SV_N_SM75;
            uint32_t thread_row0 = o_start_row_warp + (lane_id / 4);
            uint32_t thread_row1 = thread_row0 + 8;
            uint32_t thread_col0 = col_base + (lane_id % 4) * 2;
            uint32_t thread_col1 = thread_col0 + 1;

            uint32_t global_row0 = q_start_row_block + thread_row0;
            if (global_row0 < qo_len) {
                uint32_t o_offset = batch_id * stride_bz_o + head_id * stride_h_o + global_row0 * stride_seq_o;
                O[o_offset + thread_col0] = __float2half_rn(RO_accum[fk * 4 + 0]);
                O[o_offset + thread_col1] = __float2half_rn(RO_accum[fk * 4 + 1]);
            }

            uint32_t global_row1 = q_start_row_block + thread_row1;
            if (global_row1 < qo_len) {
                uint32_t o_offset = batch_id * stride_bz_o + head_id * stride_h_o + global_row1 * stride_seq_o;
                O[o_offset + thread_col0] = __float2half_rn(RO_accum[fk * 4 + 2]);
                O[o_offset + thread_col1] = __float2half_rn(RO_accum[fk * 4 + 3]);
            }
        }
"""
        if old_direct not in sage_source:
            raise RuntimeError("SM75 PV direct-output mapping source contract changed; refusing to patch.")
        sage_source=sage_source.replace(old_direct,new_direct,1)

        # A final source fingerprint ensures all four row-ownership fixes were applied.
        required_pv_contract=(
            pv_marker,
            "float o_scale_top = math::ptx_exp2(m_old[0] - m_i[0]);",
            "float o_scale_bottom = math::ptx_exp2(m_old[1] - m_i[1]);",
            "float l_rcp_top = (l_i[0] > 0.0f) ? math::ptx_rcp(l_i[0]) : 0.0f;",
            "uint32_t thread_row1 = thread_row0 + 8;",
            "uint32_t thread_row0 = o_start_row_warp + (lane_id / 4);",
        )
        missing=[needle for needle in required_pv_contract if needle not in sage_source]
        if missing:
            raise RuntimeError("SM75 PV fragment fix incomplete:\n"+"\n".join(missing))
        sage_kernel.write_text(sage_source,encoding="utf-8")
        print("[SAGE SM75 PATCH] m16n8k8 PV fragment row mapping/normalization correction: PASS")
    else:
        required_pv_contract=(pv_marker,
            "float o_scale_top = math::ptx_exp2(m_old[0] - m_i[0]);",
            "float l_rcp_top = (l_i[0] > 0.0f) ? math::ptx_rcp(l_i[0]) : 0.0f;",
            "uint32_t thread_row1 = thread_row0 + 8;",
        )
        missing=[needle for needle in required_pv_contract if needle not in sage_source]
        if missing:
            raise RuntimeError("SM75 PV fragment fix marker exists but source contract is incomplete:\n"+"\n".join(missing))
        print("[SAGE SM75 PATCH] m16n8k8 PV fragment row mapping/normalization correction: ALREADY_APPLIED")

    for relative in ("build", "dist", "sageattention.egg-info"):
        path = install_dir / relative
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()
    sage_pkg_dir = install_dir / "sageattention"
    if sage_pkg_dir.is_dir():
        for artifact in sage_pkg_dir.glob("*.so"):
            artifact.unlink()

    build_env = dict(os.environ)
    build_env["TORCH_CUDA_ARCH_LIST"] = "7.5"
    build_env["MAX_JOBS"] = "1"
    build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"
    run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        "--no-deps",
        "--no-build-isolation",
        "-e",
        install_dir,
        env=build_env,
    )

    library_dirs = _cuda_library_dirs(runtime)
    if not library_dirs:
        raise RuntimeError("SageAttention verification could not locate CUDA runtime libraries.")
    for gpu_id in (0, 1):
        child_env = _configure_cuda_environment(library_dirs)
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        child_env["H3_EXPECTED_SAGE_VERSION"] = expected_version
        child_env["H3_SAGE_ROOT"] = str(install_dir)
        probe = r'''import inspect
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import sageattention
from sageattention import sageattn

expected = os.environ["H3_EXPECTED_SAGE_VERSION"]
root = Path(os.environ["H3_SAGE_ROOT"]).resolve()

# Pinned 9f1c9a29... package export contract: __init__.py defines
# SM75_CUDA_ENABLED and does not define SM75_ENABLED/FUSED_ENABLED.
if str(getattr(sageattention, "__version__", "")).strip() != expected:
    raise RuntimeError(
        f"SageAttention version mismatch: expected={expected}, actual={getattr(sageattention, '__version__', '')}"
    )

package_file = Path(getattr(sageattention, "__file__", "")).resolve()
if not package_file.is_relative_to(root / "sageattention"):
    raise RuntimeError(f"SageAttention imported outside pinned checkout: {package_file}")
# The pinned SageAttention package exports SM75_CUDA_ENABLED from
# sageattention/__init__.py. FUSED_ENABLED is internal to core.py and is
# intentionally verified by importing the compiled _fused extension itself.
# Keep these checks identical to the fresh-process H3 verifier below.
if not bool(getattr(sageattention, "SM75_CUDA_ENABLED", False)):
    raise RuntimeError("SageAttention SM75 CUDA extension is not enabled")
try:
    from sageattention import _fused as _sage_fused
except Exception as exc:
    raise RuntimeError(
        "SageAttention fused CUDA extension is not importable"
    ) from exc
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable during SageAttention verification")
capability = torch.cuda.get_device_capability(0)
if capability != (7, 5):
    raise RuntimeError(f"Expected SM75/T4, got {capability}")
torch.cuda.set_device(0)

# This verifier is intentionally coupled to the SHA-pinned SM75 implementation.
# A change to the internal quantizer contract must fail closed rather than
# silently falling back to a weaker FP16-vs-INT8 statistical test.
from sageattention.quant import per_warp_int8
quant_params = set(inspect.signature(per_warp_int8).parameters)
required_quant_params = {
    "q", "k", "km", "BLKQ", "WARPQ", "BLKK", "tensor_layout",
}
missing = required_quant_params - quant_params
if missing:
    raise RuntimeError(
        "Pinned SageAttention per_warp_int8 contract changed; refusing to weaken verification. "
        f"Missing parameters: {sorted(missing)}"
    )

# These are part of the pinned SM75 implementation in core.py/attn_cuda_sm75.h:
# Q is quantized with BLKQ=64 and WARPQ=16; K is quantized per 64-key block.
BLKQ = 64
WARPQ = 16
BLKK = 64

# The Q scale indexing below is defined per WARPQ rows inside each BLKQ
# quantization tile. The pinned SM75 implementation requires BLKQ to be an
# exact multiple of WARPQ; fail closed if that geometry ever changes.
if BLKQ % WARPQ != 0:
    raise RuntimeError(
        f"Invalid SM75 quantization geometry: BLKQ={BLKQ} must be divisible by WARPQ={WARPQ}"
    )

EXPECTED_HEAD_DIMS = (64, 128)
MAX_KERNEL_RELATIVE_L2 = 0.02
MIN_KERNEL_COSINE = 0.999


def _reference_from_exact_quantized_inputs(q, v, q_int8, q_scale, k_int8, k_scale, sm_scale):
    """Reproduce the mathematical contract of the SM75 INT8-QK/FP16-PV kernel."""
    q_len = q.shape[-2]
    k_len = k_int8.shape[-2]
    expected_q_scales = ((q_len + BLKQ - 1) // BLKQ) * (BLKQ // WARPQ)
    expected_k_scales = (k_len + BLKK - 1) // BLKK

    if q_scale.ndim != 3 or q_scale.shape[-1] != expected_q_scales:
        raise RuntimeError(
            "SM75 Q scale layout mismatch: "
            f"shape={tuple(q_scale.shape)}, expected last dim={expected_q_scales} "
            f"for BLKQ={BLKQ}, WARPQ={WARPQ}, q_len={q_len}"
        )
    if k_scale.ndim != 3 or k_scale.shape[-1] != expected_k_scales:
        raise RuntimeError(
            "SM75 K scale layout mismatch: "
            f"shape={tuple(k_scale.shape)}, expected last dim={expected_k_scales} "
            f"for BLKK={BLKK}, k_len={k_len}"
        )

    q_row_scales = torch.arange(q_len, device=q.device) // WARPQ
    k_row_scales = torch.arange(k_len, device=q.device) // BLKK
    q_scale_expanded = q_scale.index_select(2, q_row_scales).unsqueeze(-1)
    k_scale_expanded = k_scale.index_select(2, k_row_scales).unsqueeze(-1)

    q_dequant = q_int8.float() * q_scale_expanded
    k_dequant = k_int8.float() * k_scale_expanded

    scores = torch.matmul(q_dequant, k_dequant.transpose(-2, -1)) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    # The CUDA kernel returns FP16, so round the dense reference to FP16 before
    # computing error metrics. This removes pure output-cast noise from the gate.
    return torch.matmul(probs, v.float()).to(torch.float16)


def _verify_case(head_dim, smooth_k, qo_len=1024, kv_len=1024, num_heads=56, seed=1729):
    torch.manual_seed(seed + head_dim + int(smooth_k) + qo_len + kv_len)
    dtype = torch.float16
    q = torch.randn((1, num_heads, qo_len, head_dim), device="cuda", dtype=dtype).contiguous()
    k = torch.randn((1, num_heads, kv_len, head_dim), device="cuda", dtype=dtype).contiguous()
    v = torch.randn((1, num_heads, kv_len, head_dim), device="cuda", dtype=dtype).contiguous()

    sm_scale = head_dim ** -0.5
    out = sageattn(
        q,
        k,
        v,
        tensor_layout="HND",
        is_causal=False,
        sm_scale=sm_scale,
        smooth_k=smooth_k,
        qk_quant_gran="per_warp",
    )
    torch.cuda.synchronize()

    if tuple(out.shape) != tuple(q.shape):
        raise RuntimeError(
            f"SageAttention output shape mismatch: {tuple(out.shape)} != {tuple(q.shape)}"
        )
    if out.dtype != torch.float16:
        raise RuntimeError(f"SageAttention SM75 output dtype mismatch: {out.dtype} != torch.float16")
    if not torch.isfinite(out).all().item():
        raise RuntimeError("SageAttention SM75 output contains non-finite values")

    # Pinned SM75 smooth-K contract: when enabled, SageAttention computes the
    # sequence-wise K mean (km) and applies the same mean-centered K contract
    # used by the fused kernel. The verifier passes that exact km to the same
    # pinned quantizer; it intentionally fails closed if the fork changes this
    # semantic contract rather than silently accepting a weaker reference.
    km = k.mean(dim=2, keepdim=True) if smooth_k else None
    q_int8, q_scale, k_int8, k_scale = per_warp_int8(
        q,
        k,
        km=km,
        tensor_layout="HND",
        BLKQ=BLKQ,
        WARPQ=WARPQ,
        BLKK=BLKK,
    )

    for name, tensor, expected_dtype in (
        ("q_int8", q_int8, torch.int8),
        ("k_int8", k_int8, torch.int8),
        ("q_scale", q_scale, torch.float32),
        ("k_scale", k_scale, torch.float32),
    ):
        if tensor.dtype != expected_dtype:
            raise RuntimeError(f"{name} dtype mismatch: {tensor.dtype} != {expected_dtype}")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} must be contiguous")
        if not torch.isfinite(tensor.float()).all().item():
            raise RuntimeError(f"{name} contains non-finite values")

    quant_reference = _reference_from_exact_quantized_inputs(
        q, v, q_int8, q_scale, k_int8, k_scale, sm_scale
    )

    diff = out.float() - quant_reference.float()
    ref_norm = float(torch.linalg.vector_norm(quant_reference.float()).item())
    err_norm = float(torch.linalg.vector_norm(diff).item())
    relative_l2 = err_norm / max(ref_norm, 1e-12)
    cosine = float(
        F.cosine_similarity(
            out.float().reshape(1, -1),
            quant_reference.float().reshape(1, -1),
            dim=1,
        ).item()
    )
    max_abs = float(diff.abs().max().item())
    mean_abs = float(diff.abs().mean().item())

    # Diagnostic only: compare the INT8 kernel to true FP16 SDPA. This must NOT
    # be used as a kernel-correctness gate because INT8 Q/K quantization is expected.
    k_reference = k - km if smooth_k else k
    fp16_reference = F.scaled_dot_product_attention(
        q, k_reference, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=sm_scale,
    )
    fp16_diff = out.float() - fp16_reference.float()
    fp16_max_abs = float(fp16_diff.abs().max().item())
    fp16_mean_abs = float(fp16_diff.abs().mean().item())

    print(
        f"[SAGE SM75 CASE] hd={head_dim} smooth_k={smooth_k} "
        f"shape={tuple(out.shape)} finite=PASS "
        f"kernel_relative_l2={relative_l2:.6g} "
        f"kernel_cosine={cosine:.8f} "
        f"kernel_max_abs={max_abs:.6g} "
        f"kernel_mean_abs={mean_abs:.6g} "
        f"fp16_max_abs={fp16_max_abs:.6g} "
        f"fp16_mean_abs={fp16_mean_abs:.6g}"
    )

    if relative_l2 > MAX_KERNEL_RELATIVE_L2 or cosine < MIN_KERNEL_COSINE:
        raise RuntimeError(
            "SageAttention SM75 kernel correctness gate failed: "
            f"hd={head_dim}, smooth_k={smooth_k}, relative_l2={relative_l2:.6g}, "
            f"cosine={cosine:.8f}, max_abs={max_abs:.6g}, mean_abs={mean_abs:.6g}; "
            f"required relative_l2<={MAX_KERNEL_RELATIVE_L2} and cosine>={MIN_KERNEL_COSINE}"
        )


# H3 production-shape validation: 56 heads, non-causal HND, per-warp Q/K, FP32
# accumulation, and a 1024x1024 multi-tile attention matrix. The head_dim=64 and
# smooth-K-off cases isolate architecture/quantization behavior without changing production behavior.
for _head_dim, _smooth_k in (
    (64, False),
    (64, True),
    (128, False),
    (128, True),
):
    if _head_dim not in EXPECTED_HEAD_DIMS:
        raise RuntimeError(f"Unexpected SageAttention verification head_dim={_head_dim}")
    _verify_case(
        head_dim=_head_dim,
        smooth_k=_smooth_k,
        qo_len=1024,
        kv_len=1024,
        num_heads=56,
    )

print(
    f"[SAGE SM75] version={sageattention.__version__} "
    f"gpu={torch.cuda.get_device_name(0)} capability=sm75 "
    f"all_kernel_correctness_gates=PASS"
)'''

        verification = subprocess.run(
            [sys.executable, "-c", probe],
            env=child_env,
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
                f"SageAttention SM75 verification failed on physical GPU {gpu_id}."
            )

    print(
        f"[SAGE SM75] repository={repository} revision={actual_revision} "
        f"version={expected_version} build=PASS gpu0=PASS gpu1=PASS"
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
    runtime = load_yaml(RUNTIME_MANIFEST)
    pytorch_cuda = str(runtime["pytorch"]["cuda"])
    legacy_runtime = ROOT / f".h3_runtime_{pytorch_cuda}"
    if legacy_runtime.exists(): shutil.rmtree(legacy_runtime, ignore_errors=True)
    ensure_kaggle_startup_wrapt(runtime)
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
    install_sageattention_sm75(runtime)
    prepare_locked_h3_optimization_source(runtime)
    patch_h3_bounded_qkv(runtime)
    patch_h3_qkv_binding_lifetime(runtime)
    patch_h3_sage_attention(runtime)
    apply_embedded_h3_runtime_overlay()
    patch_h3_embedding_memory_compatibility(runtime)
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

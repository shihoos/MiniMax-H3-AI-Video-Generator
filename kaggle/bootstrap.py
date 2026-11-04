from __future__ import annotations

import ast
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
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
def find_kaggle_files(
    filenames: tuple[str, ...],
) -> dict[str, Path]:
    """Resolve all requested Kaggle assets with one recursive traversal."""
    requested = {name.lower(): name for name in filenames}
    matches: dict[str, list[Path]] = {name: [] for name in requested}
    for path in KAGGLE_INPUT.rglob("*"):
        if not path.is_file():
            continue
        key = path.name.lower()
        if key in matches:
            matches[key].append(path)

    resolved: dict[str, Path] = {}
    for key, original_name in requested.items():
        candidates = matches[key]
        if not candidates:
            raise FileNotFoundError(
                "Required Kaggle asset not found: "
                f"{original_name}"
            )
        if len(candidates) > 1:
            raise RuntimeError(
                f"Multiple copies found for {original_name}:\n"
                + "\n".join(str(path) for path in candidates)
            )
        resolved[key] = candidates[0]
    return resolved


def _requirements_satisfied(requirements: Path) -> bool:
    """Return True when every normal requirement is already satisfied.

    Exact pins, unpinned packages, and standard version specifiers are checked
    against the live environment. Direct URLs/editable/path requirements are
    treated conservatively and force the normal pip install path.
    """
    if not requirements.is_file():
        return False
    try:
        from packaging.requirements import Requirement
    except Exception:
        return False

    for raw_line in requirements.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            return False
        try:
            requirement = Requirement(line)
        except Exception:
            return False
        if requirement.marker is not None:
            try:
                if not requirement.marker.evaluate():
                    continue
            except Exception:
                return False
        if requirement.url:
            return False
        name = requirement.name
        try:
            installed = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            return False
        if requirement.specifier and installed not in requirement.specifier:
            return False
    return True


def _package_version_satisfied(package: str, expected: str) -> bool:
    expected = str(expected or "").strip()
    if not package or not expected:
        return False
    try:
        return importlib_metadata.version(package) == expected
    except importlib_metadata.PackageNotFoundError:
        return False


def _probe_pillow(expected_version: str) -> bool:
    expected_version = str(expected_version or "").strip()
    if not expected_version:
        return False
    probe = (
        "import io, importlib.metadata as md; import PIL; from PIL import Image; "
        "from PIL._typing import _Ink; import gradio as gr; "
        f"assert PIL.__version__ == {expected_version!r}; "
        f"assert md.version('Pillow') == {expected_version!r}; "
        "assert Image.__version__ == PIL.__version__; "
        "assert _Ink is not None; "
        "image = Image.new('RGB', (32, 24), (40, 80, 120)); "
        "image = image.resize((16, 12)); buffer = io.BytesIO(); "
        "image.save(buffer, format='PNG'); buffer.seek(0); "
        "decoded = Image.open(buffer); decoded.load(); "
        "assert decoded.size == (16, 12); "
        "assert hasattr(gr, 'Image'); "
        "print(PIL.__version__, gr.__version__, 'Pillow/Gradio image smoke: PASS')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )
    return result.returncode == 0 and expected_version in result.stdout


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def _git_has_revision(path: Path, revision: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _ensure_git_revision(path: Path, revision: str) -> None:
    """Fetch only the pinned commit first; use a full fetch only as a fallback."""
    if _git_has_revision(path, revision):
        return
    targeted = subprocess.run(
        ["git", "-C", str(path), "fetch", "--no-tags", "origin", revision],
        capture_output=True,
        text=True,
        check=False,
    )
    if targeted.returncode == 0 and _git_has_revision(path, revision):
        print(f"[GIT] fetched missing pinned commit directly: {revision}")
        return
    print(
        "[GIT] targeted SHA fetch unavailable; falling back to full remote fetch for pinned commit",
        revision,
    )
    run("git", "-C", path, "fetch", "--all", "--tags", "--prune")
    if not _git_has_revision(path, revision):
        detail = (targeted.stderr or targeted.stdout or "").strip()
        raise RuntimeError(
            f"Pinned Git revision is still unavailable after fetch: {revision}. {detail}"
        )


def _git_exact_tag(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return (result.stdout or "").strip()


def _comfyui_checkout_is_locked(
    revision: str,
    expected_version: str,
) -> bool:
    if not COMFY.is_dir() or not (COMFY / ".git").is_dir():
        return False
    try:
        head = _git_head(COMFY)
        expected_head = subprocess.check_output(
            ["git", "-C", str(COMFY), "rev-list", "-n", "1", revision],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except subprocess.CalledProcessError:
        return False
    if head != expected_head:
        return False
    if expected_version:
        tagged = _git_exact_tag(COMFY)
        if tagged not in {expected_version, f"v{expected_version}"}:
            return False
    return True


def _normalize_torch_version(version: str, cuda: str) -> tuple[str, str]:
    base = version.split("+", 1)[0].strip()
    if not base:
        raise RuntimeError("runtime_versions.yaml pytorch.version is required.")
    return base, f"{base}+{cuda}"


def _pytorch_runtime_probe_code(runtime: dict) -> str:
    config = dict(runtime.get("pytorch", {}) or {})
    _, expected_torch = _normalize_torch_version(
        str(config.get("version", "") or "").strip(),
        str(config.get("cuda", "") or "").strip().lower(),
    )
    cuda_tag = str(config.get("cuda", "") or "").strip().lower()
    cuda_digits = cuda_tag[2:]
    expected_cuda = f"{cuda_digits[:2]}.{cuda_digits[2:]}"
    expected_tv = str(config.get("torchvision_version", "") or "").strip()
    expected_ta = str(config.get("torchaudio_version", "") or "").strip()
    return (
        "import importlib.metadata as md; import torch, torchvision, torchaudio; "
        f"assert torch.__version__ == {expected_torch!r}, torch.__version__; "
        f"assert torch.version.cuda == {expected_cuda!r}, torch.version.cuda; "
        f"assert md.version('torchvision') == {expected_tv!r}, md.version('torchvision'); "
        f"assert md.version('torchaudio') == {expected_ta!r}, md.version('torchaudio'); "
        "assert torchvision.__version__ == md.version('torchvision'); "
        "assert torchaudio.__version__ == md.version('torchaudio'); "
        "assert torch.cuda.is_available(), 'CUDA unavailable'; "
        "assert torch.cuda.device_count() >= 1, 'no CUDA devices'; "
        "assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(torch.cuda.device_count())), 'non-SM75 GPU'; "
        "from torchvision.ops import nms; "
        "boxes = torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]], device='cuda'); "
        "scores = torch.tensor([0.9,0.8], device='cuda'); "
        "keep = nms(boxes, scores, 0.5); assert keep.numel() == 1; "
        "audio = torch.zeros(64, dtype=torch.float32); "
        "resampled = torchaudio.functional.resample(audio, 16000, 8000); "
        "assert resampled.numel() > 0; "
        "print('PyTorch:', torch.__version__); "
        "print('CUDA:', torch.version.cuda); "
        "print('torchvision:', torchvision.__version__); "
        "print('torchaudio:', torchaudio.__version__); "
        "print('GPU count:', torch.cuda.device_count()); "
        "print('GPU capability:', [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]); "
        "print('TorchVision CUDA NMS: PASS'); print('TorchAudio resample: PASS'); "
        "print('PyTorch runtime lock: PASS')"
    )


def _pytorch_runtime_is_locked(runtime: dict) -> bool:
    config = dict(runtime.get("pytorch", {}) or {})
    cuda_tag = str(config.get("cuda", "") or "").strip().lower()
    expected_tv = str(config.get("torchvision_version", "") or "").strip()
    expected_ta = str(config.get("torchaudio_version", "") or "").strip()
    try:
        _, expected_torch = _normalize_torch_version(
            str(config.get("version", "") or "").strip(), cuda_tag
        )
    except RuntimeError:
        return False
    if (
        not cuda_tag.startswith("cu")
        or not cuda_tag[2:].isdigit()
        or len(cuda_tag[2:]) != 3
        or not expected_tv
        or not expected_ta
    ):
        return False

    # A live, already-imported Torch module cannot be replaced by pip in-place.
    # If this kernel still holds an older build, force the installer to repair
    # disk state and rely on fresh-process validation for native extensions.
    if "torch" in sys.modules:
        try:
            if str(getattr(sys.modules["torch"], "__version__", "")) != expected_torch:
                return False
        except Exception:
            return False

    result = subprocess.run(
        [sys.executable, "-c", _pytorch_runtime_probe_code(runtime)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 and os.getenv("H3_BOOTSTRAP_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        print("[PYTORCH PROBE]", result.stdout, result.stderr)
    return result.returncode == 0


def _probe_director_venv(runtime: dict, env_dir: Path) -> bool:
    """Return True only when the isolated vLLM environment matches every lock."""
    director = dict(runtime.get("director", {}) or {})
    vllm_version = str(director.get("vllm_version", "") or "").strip()
    wrapt_version = str(runtime.get("python", {}).get("wrapt_version", "") or "").strip()
    transformers_version = _exact_requirement_pin("transformers")
    tokenizers_version = _exact_requirement_pin("tokenizers")
    torch_version = str(director.get("vllm_torch_version", "") or "").strip()
    cuda_tag = str(director.get("vllm_cuda", "") or "").strip().lower()
    tp_size = int(director.get("tensor_parallel_size", 0) or 0)
    python = env_dir / "bin" / "python"
    if not all((python.is_file(), vllm_version, wrapt_version, torch_version, cuda_tag == "cu130", tp_size > 0)):
        return False
    expected_torch = f"{torch_version}+{cuda_tag}"
    code = (
        "import sys, inspect, vllm, torch, wrapt, transformers, tokenizers; "
        "from transformers import Qwen3Config; "
        "from vllm.config.load import LoadConfig; "
        "from vllm.config import SpeculativeConfig; "
        f"assert sys.version_info[:2] == (3, 12), sys.version; "
        f"assert vllm.__version__ == {vllm_version!r}; "
        f"assert torch.__version__ == {expected_torch!r}, torch.__version__; "
        "assert torch.version.cuda == '13.0', torch.version.cuda; "
        f"assert wrapt.__version__ == {wrapt_version!r}; "
        f"assert transformers.__version__ == {transformers_version!r}; "
        f"assert tokenizers.__version__ == {tokenizers_version!r}; "
        "assert 'safetensors_load_strategy' in inspect.signature(LoadConfig).parameters; "
        "assert Qwen3Config.__name__ == 'Qwen3Config'; "
        "assert 'eagle3' in str(inspect.signature(SpeculativeConfig)); "
        "assert torch.cuda.is_available() and torch.cuda.device_count() == " + str(tp_size) + "; "
        "assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(torch.cuda.device_count())); "
        "print('Director runtime probe: PASS')"
    )
    result = subprocess.run([str(python), "-c", code], capture_output=True, text=True, check=False)
    if result.returncode != 0 and os.getenv("H3_DIRECTOR_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        print("[DIRECTOR WARM PROBE]", result.stdout, result.stderr)
    return result.returncode == 0




def _sage_build_stamp_path(install_dir: Path) -> Path:
    return install_dir / ".h3_t4_build_stamp.json"


def _tool_version(command: str) -> str:
    try:
        result = subprocess.run(
            [command, "--version"], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if result.returncode != 0:
        return "unavailable"
    lines = [line.strip() for line in (result.stdout or result.stderr).splitlines() if line.strip()]
    return " | ".join(lines[-2:]) if lines else "unknown"


def _sage_build_stamp(runtime: dict, install_dir: Path | None = None) -> dict:
    pytorch = dict(runtime.get("pytorch", {}) or {})
    sage = dict(runtime.get("sage_attention", {}) or {})
    cuda_runtime = dict(runtime.get("cuda_runtime", {}) or {})
    kernel = (install_dir / "csrc" / "qattn" / "attn_cuda_sm75.h") if install_dir else None
    kernel_sha256 = None
    if kernel is not None and kernel.is_file():
        kernel_sha256 = hashlib.sha256(kernel.read_bytes()).hexdigest()
    return {
        "schema": 2,
        "sage_revision": str(sage.get("revision", "") or "").strip(),
        "sage_version": str(sage.get("version", "") or "").strip(),
        "torch_version": str(pytorch.get("version", "") or "").strip(),
        "torchvision_version": str(pytorch.get("torchvision_version", "") or "").strip(),
        "torchaudio_version": str(pytorch.get("torchaudio_version", "") or "").strip(),
        "torch_cuda": str(pytorch.get("cuda", "") or "").strip().lower(),
        "cuda_runtime": str(cuda_runtime.get("runtime_version", "") or "").strip(),
        "cublas": str(cuda_runtime.get("cublas_version", "") or "").strip(),
        "python": sys.version,
        "platform": sys.platform,
        "machine": __import__("platform").machine(),
        "gpu_arch": "sm75",
        "cxx_compiler": _tool_version("c++"),
        "nvcc_compiler": _tool_version("nvcc"),
        "patched_kernel_sha256": kernel_sha256,
    }


def _sage_install_is_reusable(runtime: dict, install_dir: Path) -> bool:
    if not install_dir.is_dir() or not (install_dir / ".git").is_dir():
        return False
    stamp_path = _sage_build_stamp_path(install_dir)
    kernel = install_dir / "csrc" / "qattn" / "attn_cuda_sm75.h"
    package_dir = install_dir / "sageattention"
    if not stamp_path.is_file() or not kernel.is_file() or not package_dir.is_dir():
        return False
    try:
        stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
        if stamp != _sage_build_stamp(runtime, install_dir):
            return False
        if _git_head(install_dir) != str(runtime.get("sage_attention", {}).get("revision", "")).strip():
            return False
        source = kernel.read_text(encoding="utf-8")
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False
    required = (
        "H3-T4-SM75-KERNEL-FIX",
        "H3-T4-SM75-PV-FRAGMENT-FIX",
        "constexpr int WARP_K_SM75 = 64;",
        "uint32_t thread_row1 = thread_row0 + 8;",
    )
    if any(needle not in source for needle in required):
        return False
    if "constexpr int WARP_K_SM75 = 16;" in source:
        return False
    if not any(package_dir.rglob("*.so")):
        return False
    expected_version = str(runtime.get("sage_attention", {}).get("version", "") or "").strip()
    library_dirs = _cuda_library_dirs(runtime)
    if not library_dirs:
        return False
    env = _configure_cuda_environment(library_dirs)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sageattention; assert sageattention.__version__ == {expected_version!r}; print(sageattention.__version__)",
        ],
        env=env,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0

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
    if _requirements_satisfied(requirements):
        print("[BOOTSTRAP WARM] project requirements already satisfy requirements.txt")
        return
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
    if expected_version != "0.43.2" or revision != "423cbd20511becb1f16122ab518e92e58bfa03d9":
        raise RuntimeError(
            "This runtime upgrade is locked to ComfyUI 0.43.2 at "
            "423cbd20511becb1f16122ab518e92e58bfa03d9."
        )
    if not repository or not revision:
        raise RuntimeError("runtime_versions.yaml comfyui.repository/revision are required.")

    COMFY.parent.mkdir(parents=True, exist_ok=True)
    if COMFY.exists() and not (COMFY / ".git").is_dir():
        raise RuntimeError(f"ComfyUI path exists but is not a git checkout: {COMFY}")
    if not COMFY.exists():
        run("git", "clone", repository, COMFY)

    if _comfyui_checkout_is_locked(revision, expected_version):
        print("[BOOTSTRAP WARM] reusing locked ComfyUI checkout")
    else:
        _ensure_git_revision(COMFY, revision)
        run("git", "-C", COMFY, "checkout", "--detach", revision)
    comfy_requirements = COMFY / "requirements.txt"
    if _requirements_satisfied(comfy_requirements):
        print("[BOOTSTRAP WARM] ComfyUI requirements already satisfied")
    else:
        run(
            sys.executable, "-m", "pip", "install", "-q",
            "--disable-pip-version-check", "-r", comfy_requirements,
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
    """Apply and verify only the ComfyUI 0.43.2 H3 changes still required on T4."""
    targets = (
        "comfy/ldm/minimax/model.py",
        "comfy/ldm/minimax/vae.py",
        "comfy/supported_models.py",
        "custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py",
    )
    missing_files = [str(COMFY / item) for item in targets if not (COMFY / item).is_file()]
    if missing_files:
        raise RuntimeError("H3 runtime overlay targets are missing:\n" + "\n".join(missing_files))
    runtime = load_yaml(RUNTIME_MANIFEST)
    patch_h3_fp16_runtime(runtime)
    patch_h3_vae_decoder_dtype(runtime)
    model_text = (COMFY / "comfy/ldm/minimax/model.py").read_text(encoding="utf-8")
    vae_text = (COMFY / "comfy/ldm/minimax/vae.py").read_text(encoding="utf-8")
    supported_text = (COMFY / "comfy/supported_models.py").read_text(encoding="utf-8")
    turbo_text = (COMFY / "custom_nodes/ComfyUI-MiniMax-H3-Turbo/__init__.py").read_text(encoding="utf-8")
    checks = {
        "model.py": ("class MiniMaxH3Model", "def _embed_and_pack(", "preferred_attention=self.comfy_attention"),
        "vae.py": ("# H3-T4-VAE-DTYPE: decoder input matches decoder parameters", "decoder_parameter.dtype"),
        "supported_models.py": ("memory_usage_factor = 0.17", "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]"),
        "ComfyUI-MiniMax-H3-Turbo/__init__.py": ("class MiniMaxH3TurboLoRA",),
    }
    sources = {
        "model.py": model_text,
        "vae.py": vae_text,
        "supported_models.py": supported_text,
        "ComfyUI-MiniMax-H3-Turbo/__init__.py": turbo_text,
    }
    missing = [f"{name}: {item}" for name, markers in checks.items() for item in markers if item not in sources[name]]
    if missing:
        raise RuntimeError("H3 v0.43.2 source verification failed:\n" + "\n".join(missing))
    print("[H3 COMFY PATCH] v0.43.2 native forward + T4 dtype/memory/VAE overlay verified")




def verify_inventory() -> None:
    manifest = load_yaml(MODEL_MANIFEST)
    policy = dict(manifest.get("policy", {}) or {})
    exact_inventory = policy.get("exact_inventory") is True
    allow_extra = policy.get("allow_extra_production_models") is True
    allow_q4 = policy.get("allow_q4_models") is True
    allow_additional_qwen = policy.get("allow_additional_qwen_models") is True
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
    if unexpected and (exact_inventory or not allow_extra):
        raise RuntimeError("Unexpected H3 production models:\n" + "\n".join(f"{d}/{f}" for d, f in sorted(unexpected)))
    if not allow_q4:
        q4 = [
            (directory, filename)
            for directory, filename in actual
            if filename.endswith((".gguf", ".gguf2")) or "q4" in filename
        ]
        if q4:
            raise RuntimeError(
                "Q4/GGUF production models are forbidden by model_inventory.yaml policy:\n"
                + "\n".join(f"{d}/{f}" for d, f in sorted(q4))
            )
    if not allow_additional_qwen:
        expected_qwen = {
            filename
            for directory, filename in expected
            if directory == "text_encoders" and "qwen" in filename
        }
        unexpected_qwen = {
            (directory, filename)
            for directory, filename in actual
            if directory == "text_encoders"
            and "qwen" in filename
            and filename not in expected_qwen
        }
        if unexpected_qwen:
            raise RuntimeError(
                "Additional Qwen production models are forbidden by model_inventory.yaml policy:\n"
                + "\n".join(f"{d}/{f}" for d, f in sorted(unexpected_qwen))
            )


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
    """Install the locked SM75 PyTorch stack and validate native extensions in a child process."""
    config = dict(runtime.get("pytorch", {}) or {})
    version = str(config.get("version", "") or "").strip()
    cuda = str(config.get("cuda", "") or "").strip().lower()
    index = str(config.get("index", "") or "").strip()
    torchvision_version = str(config.get("torchvision_version", "") or "").strip()
    torchaudio_version = str(config.get("torchaudio_version", "") or "").strip()
    if not all((version, cuda, index, torchvision_version, torchaudio_version)):
        raise RuntimeError("runtime_versions.yaml pytorch configuration is incomplete.")
    if not cuda.startswith("cu") or not cuda[2:].isdigit() or len(cuda[2:]) != 3:
        raise RuntimeError(
            "runtime_versions.yaml pytorch.cuda must use a cuNNN wheel tag; "
            f"got {cuda!r}."
        )
    base_version, expected_torch = _normalize_torch_version(version, cuda)
    if _pytorch_runtime_is_locked(runtime):
        print(f"[BOOTSTRAP WARM] reusing verified PyTorch={expected_torch}")
        return

    # The long-lived Kaggle notebook may still hold an older `torch` module in
    # sys.modules even after a previous bootstrap upgraded files on disk. Check
    # a clean interpreter before forcing another large wheel reinstall.
    disk_probe = subprocess.run(
        [sys.executable, "-c", _pytorch_runtime_probe_code(runtime)],
        capture_output=True, text=True, check=False,
    )
    if disk_probe.returncode == 0:
        live_torch = sys.modules.get("torch")
        if live_torch is not None and str(getattr(live_torch, "__version__", "")) != expected_torch:
            print(
                "[PYTORCH CACHE WARNING] the installed disk runtime is correct, but this notebook "
                f"still has torch {getattr(live_torch, '__version__', 'unknown')} loaded in memory. "
                "Fresh subprocesses use the locked build; do not reload torch in-place."
            )
        else:
            print(f"[BOOTSTRAP WARM] disk PyTorch runtime verified={expected_torch}")
        if disk_probe.stdout:
            print(disk_probe.stdout, end="")
        return

    expected_metadata = {
        "torch": expected_torch,
        "torchvision": torchvision_version,
        "torchaudio": torchaudio_version,
    }
    metadata_matches = all(
        _package_version_satisfied(package, expected)
        for package, expected in expected_metadata.items()
    )
    if os.getenv("H3_BOOTSTRAP_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        print("[PYTORCH PROBE BEFORE INSTALL]", disk_probe.stdout, disk_probe.stderr)
    print("=" * 80)
    print("INSTALLING LOCKED PYTORCH RUNTIME")
    print("=" * 80)
    args = [
        sys.executable, "-m", "pip", "install", "-q",
        "--disable-pip-version-check", "--upgrade", "--index-url", index,
        f"torch=={base_version}",
        f"torchvision=={torchvision_version}",
        f"torchaudio=={torchaudio_version}",
    ]
    # Only force-reinstall when metadata already matches but the clean native
    # extension/CUDA probe failed. Normal version upgrades remain cacheable.
    if metadata_matches:
        args.insert(args.index("--upgrade"), "--force-reinstall")
    run(*args)

    verify = subprocess.run(
        [sys.executable, "-c", _pytorch_runtime_probe_code(runtime)],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.stdout:
        print(verify.stdout, end="")
    if verify.stderr:
        print(verify.stderr, end="")
    if verify.returncode != 0:
        raise RuntimeError(
            "Locked PyTorch/TorchVision/TorchAudio SM75 verification failed. "
            "See the fresh-process probe output above; refusing to continue with an unverified GPU stack."
        )
    print("[PYTORCH] exact CUDA wheel trio + TorchVision CUDA op + TorchAudio import/resample: PASS")

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
    """Report the live-module boundary; never spoof PIL.__version__ after a wheel replacement."""
    loaded_pkg = sys.modules.get("PIL")
    if loaded_pkg is not None:
        loaded_version = str(getattr(loaded_pkg, "__version__", "unknown"))
        if loaded_version != expected_version:
            print(
                "[PILLOW CACHE WARNING] this notebook kernel already imported Pillow "
                f"{loaded_version}; disk installation is {expected_version}. "
                "Python cannot safely replace the loaded native Pillow extension in-place. "
                "Fresh subprocesses use the upgraded wheel; restart the Kaggle kernel "
                "before using Pillow in this already-running process if the patched "
                "security version must be active here."
            )
            return
    import importlib
    importlib.invalidate_caches()
    import PIL
    if str(PIL.__version__) != expected_version:
        raise RuntimeError(
            f"Live Pillow version mismatch: expected={expected_version}, actual={PIL.__version__}."
        )
    print("[PILLOW CACHE] live Pillow module already matches disk version:", expected_version)

def patch_h3_fp16_runtime(runtime: dict) -> None:
    """Enable the native ComfyUI 0.43.2 H3 FP16 model path on T4 without stale forward rewrites."""
    model = COMFY / "comfy/ldm/minimax/model.py"
    supported = COMFY / "comfy/supported_models.py"
    if not model.is_file() or not supported.is_file():
        raise RuntimeError("Required ComfyUI H3 source files are missing.")
    model_text = model.read_text(encoding="utf-8")
    required_native = (
        "class MiniMaxH3Model",
        "def _embed_and_pack(",
        "preferred_attention=self.comfy_attention",
    )
    missing = [item for item in required_native if item not in model_text]
    if missing:
        raise RuntimeError(
            "ComfyUI H3 model.py does not match the v0.43.2 source contract; "
            "refusing to apply an old T4 forward patch: " + ", ".join(missing)
        )
    source = supported.read_text(encoding="utf-8")
    if "memory_usage_factor = 0.17" not in source:
        old_factor = "memory_usage_factor = 0.114"
        if source.count(old_factor) != 1:
            raise RuntimeError("MiniMaxH3 memory factor source contract changed; expected one 0.114 declaration.")
        source = source.replace(old_factor, "memory_usage_factor = 0.17", 1)
    expected_dtypes = "supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]"
    if expected_dtypes not in source:
        old_dtypes = "supported_inference_dtypes = [torch.bfloat16, torch.float32]"
        if source.count(old_dtypes) != 1:
            raise RuntimeError("MiniMaxH3 supported dtype declaration changed; refusing broad source modification.")
        source = source.replace(old_dtypes, expected_dtypes, 1)
    compile(source, str(supported), "exec")
    supported.write_text(source, encoding="utf-8")
    print("[H3 COMFY PATCH] native v0.43.2 H3 forward retained; T4 FP16 dtype and memory estimate enabled")


def _validate_h3_vae_decoder_source(source: str, filename: str) -> None:
    """Fail closed unless the pinned VAE class retains a valid guarded decoder."""
    marker = "# H3-T4-VAE-DTYPE: decoder input matches decoder parameters"
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise RuntimeError(
            f"Patched H3 VAE source is not valid Python: {filename}:{exc.lineno}: {exc.msg}"
        ) from exc

    vae_classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3VideoVAE"
    ]
    if len(vae_classes) != 1:
        raise RuntimeError(
            "Refusing to accept H3 VAE patch: expected exactly one top-level "
            "MiniMaxH3VideoVAE class."
        )
    methods = [
        node for node in vae_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_decode_pixels"
    ]
    if len(methods) != 1:
        raise RuntimeError(
            "H3 VAE decoder patch is invalid: MiniMaxH3VideoVAE must retain "
            "exactly one _decode_pixels(self, z) method."
        )

    method = methods[0]
    if len(method.args.args) < 2 or [arg.arg for arg in method.args.args[:2]] != ["self", "z"]:
        raise RuntimeError(
            "H3 VAE decoder patch is invalid: _decode_pixels must accept self and z."
        )
    segment = ast.get_source_segment(source, method) or ""
    required = (
        marker,
        "z = self.post_quant_conv(z)",
        "decoder_parameter = next(self.decoder.parameters(), None)",
        "if decoder_parameter is not None and z.dtype != decoder_parameter.dtype:",
        "z = z.to(decoder_parameter.dtype)",
        "return self.decoder(z)",
    )
    missing = [item for item in required if item not in segment]
    if missing:
        raise RuntimeError(
            "H3 VAE decoder patch is missing required method statements: "
            + ", ".join(missing)
        )
    if len(method.body) != 4 or not isinstance(method.body[2], ast.If):
        raise RuntimeError(
            "H3 VAE decoder patch has an unexpected _decode_pixels control-flow shape."
        )


def patch_h3_vae_decoder_dtype(runtime: dict) -> None:
    """Protect the v0.43.2 MiniMaxH3VideoVAE decoder input dtype."""
    enabled = bool(runtime.get("comfyui", {}).get("h3_vae_decoder_dtype_patch", True))
    if not enabled:
        print("[H3 VAE PATCH] disabled by runtime configuration")
        return
    target = COMFY / "comfy" / "ldm" / "minimax" / "vae.py"
    if not target.is_file():
        raise RuntimeError(f"H3 VAE source not found: {target}")

    source = target.read_text(encoding="utf-8")
    marker = "# H3-T4-VAE-DTYPE: decoder input matches decoder parameters"
    if marker in source:
        # A marker alone is not evidence of a valid patch. In particular, a
        # previous broken patch could leave the marker while deleting the def.
        _validate_h3_vae_decoder_source(source, str(target))
        print("[H3 VAE PATCH] already applied and AST-verified")
        return

    pattern = re.compile(
        r"(?m)^(?P<defline>[ \t]*def _decode_pixels\(self, z\):)[ \t]*\n"
        r"(?P<bodyindent>[ \t]+)return self\.decoder\(self\.post_quant_conv\(z\)\)[ \t]*$"
    )
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise RuntimeError(
            "Refusing to patch H3 VAE: expected exactly one pinned v0.43.2 "
            "MiniMaxH3VideoVAE._decode_pixels(self, z) return expression."
        )

    match = matches[0]
    indent = match.group("bodyindent")
    replacement = "\n".join(
        (
            match.group("defline"),
            f"{indent}z = self.post_quant_conv(z)",
            f"{indent}{marker}",
            f"{indent}decoder_parameter = next(self.decoder.parameters(), None)",
            f"{indent}if decoder_parameter is not None and z.dtype != decoder_parameter.dtype:",
            f"{indent}    z = z.to(decoder_parameter.dtype)",
            f"{indent}return self.decoder(z)",
        )
    )
    patched = source[:match.start()] + replacement + source[match.end():]
    compile(patched, str(target), "exec")
    _validate_h3_vae_decoder_source(patched, str(target))
    target.write_text(patched, encoding="utf-8")
    print("[H3 VAE PATCH] v0.43.2 _decode_pixels dtype guard applied and AST-verified")


def _exact_requirement_pin(package_name: str) -> str:
    """Return the sole exact pin for a package from requirements.txt."""
    requirements_path = ROOT / "requirements.txt"
    if not requirements_path.is_file():
        raise RuntimeError(f"Project dependency manifest is missing: {requirements_path}")
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


def install_director_runtime(
    runtime: dict,
) -> None:
    """Install isolated vLLM + EAGLE-3 Director runtime without mutating H3 Torch."""
    inventory = load_yaml(MODEL_MANIFEST)
    inventory_policy = dict(inventory.get("policy", {}) or {})
    if inventory_policy.get("allow_director_qwen") is not True:
        raise RuntimeError(
            "model_inventory.yaml policy.allow_director_qwen must be true for the locked Director runtime."
        )
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
    director_python = tuple(int(value) for value in str(runtime.get("python", {}).get("kaggle", "3.12")).split(".")[:2])
    if director_python != (3, 12) or sys.version_info[:2] != director_python:
        raise RuntimeError(
            "The isolated vLLM Director environment must be created from Python 3.12; "
            f"configured={director_python}, current={sys.version_info[:2]}."
        )
    torch_version = str(director.get("vllm_torch_version", "") or "").strip()
    cuda_tag = str(director.get("vllm_cuda", "") or "").strip().lower()
    loader_strategy = str(director.get("safetensors_load_strategy", "eager") or "eager").strip().lower()
    if vllm_version != "0.32.0":
        raise RuntimeError(f"Director vLLM must be pinned to 0.32.0, got {vllm_version!r}.")
    if not torch_version or cuda_tag != "cu130":
        raise RuntimeError("runtime_versions.yaml Director vLLM PyTorch/CUDA lock must specify torch 2.13.0 and cu130.")
    if loader_strategy not in {"eager", "lazy", "prefetch", "torchao"}:
        raise RuntimeError(f"Unsupported vLLM safetensors loader strategy: {loader_strategy!r}")
    if loader_strategy != "eager":
        raise RuntimeError("runtime_versions.yaml must default Director safetensors_load_strategy to eager.")
    expected_director_torch = f"{torch_version}+{cuda_tag}"
    transformers_version = _exact_requirement_pin("transformers")
    tokenizers_version = _exact_requirement_pin("tokenizers")
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
    # A failed/partial 0.32 install must not be reused. The versioned env path
    # is deliberately separate from the historical 0.31 path; only this venv
    # is removed, never either model checkpoint directory.
    if env_dir.is_symlink():
        raise RuntimeError(f"Refusing to rebuild a symlinked Director environment: {env_dir}")
    if env_dir.exists():
        shutil.rmtree(env_dir)
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    run(uv, "venv", str(env_dir), "--python", sys.executable, "--seed", "--link-mode", "copy")
    venv_python = env_dir / "bin" / "python"
    if not venv_python.is_file() or not os.access(venv_python, os.X_OK):
        raise RuntimeError(f"Invalid executable Director Python: {venv_python}")
    env = os.environ.copy()
    env["UV_LINK_MODE"] = "copy"
    env["UV_TORCH_BACKEND"] = cuda_tag
    install = subprocess.run(
        [
            uv, "pip", "install", "--python", str(venv_python), "--link-mode", "copy",
            "--torch-backend", cuda_tag,
            f"vllm=={vllm_version}", f"wrapt=={wrapt_version}",
            f"transformers=={transformers_version}", f"tokenizers=={tokenizers_version}",
        ],
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
                "import sys, vllm, torch, inspect, wrapt, transformers, tokenizers; "
                "from transformers import Qwen3Config; "
                "from vllm.config import SpeculativeConfig; "
                "from vllm.config.load import LoadConfig; "
                "assert sys.version_info[:2] == (3, 12), sys.version; "
                "print('vLLM import: PASS'); "
                "print('wrapt version:', wrapt.__version__); "
                "print('vLLM version:', vllm.__version__); "
                "print('Director Torch version:', torch.__version__); "
                "print('Director CUDA runtime:', torch.version.cuda); "
                "print('safetensors_load_strategy supported:', 'safetensors_load_strategy' in inspect.signature(LoadConfig).parameters); "
                "print('Transformers version:', transformers.__version__); "
                "print('tokenizers version:', tokenizers.__version__); "
                "print('Qwen3 config import: PASS', Qwen3Config.__name__); "
                "print('EAGLE-3 supported:', 'eagle3' in str(inspect.signature(SpeculativeConfig))); "
                f"assert torch.__version__ == {expected_director_torch!r}; "
                "assert torch.version.cuda == '13.0'; "
                "assert 'safetensors_load_strategy' in inspect.signature(LoadConfig).parameters; "
                f"assert torch.cuda.is_available() and torch.cuda.device_count() == {tensor_parallel_size}; "
                "assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(torch.cuda.device_count())); "
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
    observed_versions = {}
    for line in verification.stdout.splitlines():
        if ":" in line:
            name, value = line.split(":", 1)
            observed_versions[name.strip()] = value.strip()
    observed_version = observed_versions.get("vLLM version", "")
    if observed_version != vllm_version:
        raise RuntimeError(
            "Director runtime verification version mismatch: "
            f"observed={observed_version!r}, configured={vllm_version!r}."
        )
    observed_transformers = observed_versions.get("Transformers version", "")
    if observed_transformers != transformers_version:
        raise RuntimeError(
            "Director Transformers version mismatch: "
            f"observed={observed_transformers!r}, expected={transformers_version!r}."
        )
    observed_tokenizers = observed_versions.get("tokenizers version", "")
    if observed_tokenizers != tokenizers_version:
        raise RuntimeError(
            "Director tokenizers version mismatch: "
            f"observed={observed_tokenizers!r}, expected={tokenizers_version!r}."
        )
    if "Qwen3 config import: PASS" not in verification.stdout:
        raise RuntimeError("Director runtime did not import the pinned Transformers Qwen3 config.")
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
    if gradio_version and _package_version_satisfied("gradio", gradio_version):
        print("[BOOTSTRAP WARM] Gradio already locked", gradio_version)
    elif gradio_version:
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
    """Install Pillow last, intentionally bypassing Gradio 6.x's stale <13 metadata cap."""
    pillow_version = str(runtime.get("storyboard", {}).get("pillow_version", "")).strip()
    if not pillow_version:
        raise RuntimeError("runtime_versions.yaml storyboard.pillow_version is missing.")
    if _probe_pillow(pillow_version):
        print("[BOOTSTRAP WARM] Pillow + Gradio image smoke already verified", pillow_version)
    else:
        print(
            "[PILLOW] installing the runtime-pinned Pillow without dependency resolution; "
            "Gradio 6.26 declares Pillow<13 but Pillow 13 is required for the current security fix."
        )
        run(
            sys.executable, "-m", "pip", "install", "-q",
            "--disable-pip-version-check", "--no-deps", "--only-binary=:all:",
            "--upgrade", f"Pillow=={pillow_version}",
        )
    _repair_loaded_pillow(pillow_version)
    verify = subprocess.run(
        [sys.executable, "-c", (
            "import io, importlib.metadata as md, PIL, gradio; "
            "from PIL import Image; from PIL._typing import _Ink; "
            f"assert PIL.__version__ == {pillow_version!r}; "
            f"assert md.version('Pillow') == {pillow_version!r}; "
            "assert Image.__version__ == PIL.__version__; assert _Ink is not None; "
            "b=io.BytesIO(); Image.new('RGB',(8,8)).save(b,format='PNG'); b.seek(0); "
            "im=Image.open(b); im.load(); assert im.size==(8,8); "
            "assert hasattr(gradio,'Image'); "
            "print('Pillow:', PIL.__version__); print('Gradio:', gradio.__version__); "
            "print('Pillow/Gradio image compatibility: PASS')"
        )],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.stdout:
        print(verify.stdout, end="")
    if verify.stderr:
        print(verify.stderr, end="")
    if verify.returncode != 0:
        raise RuntimeError(
            "Fresh-process Pillow/Gradio compatibility verification failed. "
            "The bootstrap will not continue with an unverified Pillow override."
        )

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
    """Patch the locked H3-Optimizations resolver for the H3 T4 QKV edge case.

    The rule belongs inside resolve_qkv_provider's Preserve-precision branch:
    the resolver owns request semantics, while _resolve_preserve_precision_qkv
    does not receive request or memory_optimize. The apply layer only adds
    selected-provider telemetry and never overrides an explicit user request.
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
    provider_marker = "# MINIMAX_H3_T4_PRESERVE_BOUNDED_QKV"
    provider_contract = (
        "request == FUSED_QKV_PRESERVE_BF16",
        "backend_kind == 'existing'",
        "memory_optimize",
        "inventory.qkv_convrot_int8_256",
        "QKV_BF16_CHUNKED",
        "reason=force_t4_bounded_native_projection",
    )
    if provider_marker in provider_source:
        missing = [needle for needle in provider_contract if needle not in provider_source]
        if missing:
            raise RuntimeError(
                "H3 T4 bounded-QKV provider marker exists but the resolver contract "
                "is incomplete; refusing to continue:\n" + "\n".join(missing)
            )
        compile(provider_source, str(providers), "exec")
        print("[H3 T4 QKV PATCH] providers.py already patched: AST/contract verified")
    else:
        resolver_anchor = """    if request == FUSED_QKV_PRESERVE_BF16:
        return _resolve_preserve_precision_qkv(
            inventory, backend_kind, kitchen_producer_available
        )"""
        resolver_replacement = """    if request == FUSED_QKV_PRESERVE_BF16:
        # MINIMAX_H3_T4_PRESERVE_BOUNDED_QKV
        # Keep the special case in this resolver: request and memory_optimize
        # are parameters here, not locals of _resolve_preserve_precision_qkv.
        # The native ConvRot-256 INT8 projection is processed in bounded token
        # chunks for the existing dense-attention Preserve-precision path.
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
                'T4 H3 safety: bounded native ConvRot-256 INT8 QKV '
                'projection selected for existing dense attention',
            )
        return _resolve_preserve_precision_qkv(
            inventory, backend_kind, kitchen_producer_available
        )"""
        if provider_source.count(resolver_anchor) != 1:
            raise RuntimeError(
                "H3-Optimizations 0.2.41 providers.py no longer matches the locked "
                "Preserve-precision resolver contract; refusing to patch."
            )
        provider_source = provider_source.replace(resolver_anchor, resolver_replacement, 1)
        compile(provider_source, str(providers), "exec")
        providers.write_text(provider_source, encoding="utf-8")
        print("[H3 T4 QKV PATCH] providers.py patched: PASS")

    apply_text = apply_source.read_text(encoding="utf-8")
    apply_marker = "# MINIMAX_H3_T4_QKV_RUNTIME_SELECTION"
    apply_contract = (
        "QKV runtime selection: provider=%s projector=%s",
        "getattr(attention.projector, 'chunk_rows', None)",
    )
    if apply_marker in apply_text:
        missing = [needle for needle in apply_contract if needle not in apply_text]
        if missing:
            raise RuntimeError(
                "H3 T4 QKV apply telemetry marker exists but its source contract "
                "is incomplete; refusing to continue:\n" + "\n".join(missing)
            )
        compile(apply_text, str(apply_source), "exec")
        print("[H3 T4 QKV PATCH] apply.py telemetry already present: VERIFIED")
    else:
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
                "QKV runtime telemetry contract; refusing to patch."
            )
        apply_text = apply_text.replace(log_anchor, log_replacement, 1)
        compile(apply_text, str(apply_source), "exec")
        apply_source.write_text(apply_text, encoding="utf-8")
        print("[H3 T4 QKV PATCH] apply.py telemetry patched: PASS")

    # Confirm the branch is in the right function, after explicit Off handling,
    # and that its helper no longer references out-of-scope resolver variables.
    provider_source = providers.read_text(encoding="utf-8")
    apply_text = apply_source.read_text(encoding="utf-8")
    resolver = provider_source.split("def resolve_qkv_provider(", 1)
    if len(resolver) != 2:
        raise RuntimeError("H3 provider resolver definition is missing after patch.")
    resolver_tail = resolver[1]
    preserve_position = resolver_tail.find("if request == FUSED_QKV_PRESERVE_BF16:")
    off_position = resolver_tail.find("if request == FUSED_QKV_OFF:")
    if off_position < 0 or preserve_position < 0 or off_position > preserve_position:
        raise RuntimeError(
            "H3 T4 QKV resolver patch must preserve the explicit QKV Off guard before Preserve-precision handling."
        )
    helper = provider_source.split("def _resolve_preserve_precision_qkv(", 1)[1].split(
        "def resolve_qkv_provider(", 1
    )[0]
    if "memory_optimize" in helper or "request ==" in helper:
        raise RuntimeError(
            "H3 T4 QKV patch left resolver-only variables inside the preserve helper."
        )
    required = (
        provider_marker,
        "request == FUSED_QKV_PRESERVE_BF16",
        "backend_kind == 'existing'",
        "memory_optimize",
        "inventory.qkv_convrot_int8_256",
        "QKV_BF16_CHUNKED",
        apply_marker,
        *apply_contract,
    )
    missing = [needle for needle in required if needle not in provider_source + apply_text]
    if missing:
        raise RuntimeError(
            "H3 T4 bounded-QKV patch verification failed:\n" + "\n".join(missing)
        )
    compile(provider_source, str(providers), "exec")
    compile(apply_text, str(apply_source), "exec")
    print(
        "[H3 T4 QKV PATCH] PASS: locked providers.py + apply.py patched/verified "
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

    # The streamed projector may own a live Comfy weight binding. Enter the
    # release guard before imports, allocations, and casts so any setup or
    # execution failure releases that binding deterministically.
    k_attention = None
    v_attention = None
    try:
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
    """Verify H3-Optimizations' guarded fallback; never rewrite its versioned plugin source."""
    node_dir = CUSTOM / "H3-Optimizations"
    target = node_dir / "h3_optimizations" / "memory" / "embedding.py"
    expected_revision = str(runtime.get("h3_optimization", {}).get("revision", "") or "").strip()
    expected_comfy_revision = str(runtime.get("comfyui", {}).get("revision", "") or "").strip()
    for label, revision, checkout in (("H3-Optimizations", expected_revision, node_dir), ("ComfyUI", expected_comfy_revision, COMFY)):
        if not checkout.is_dir():
            raise RuntimeError(f"{label} checkout is missing: {checkout}")
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, stderr=subprocess.STDOUT).strip()
        if actual != revision:
            raise RuntimeError(f"{label} revision mismatch: expected={revision}, actual={actual}")
    if not target.is_file():
        raise RuntimeError(f"H3-Optimizations embedding source is missing: {target}")
    source = target.read_text(encoding="utf-8")
    try:
        compile(source, str(target), "exec")
    except SyntaxError as exc:
        raise RuntimeError(f"H3-Optimizations embedding.py has invalid syntax: {exc}") from exc
    required = ("UPSTREAM_FORWARD_SHA256", "H3EmbeddingMemoryPatchError", "def _validate_upstream_forward(forward):")
    missing = [item for item in required if item not in source]
    if missing:
        raise RuntimeError("Pinned H3-Optimizations embedding guard is incomplete: " + ", ".join(missing))
    print("[H3 OPT MEMORY] pinned embedding implementation retained; incompatible ComfyUI _forward must fall back to native path")




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
import time
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
from sageattention import _qattn_sm75 as _sage_sm75
from sageattention import core as _sage_core
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
from comfy.ldm.minimax.model import MiniMaxH3Model
from h3_optimizations.memory import embedding as h3_embedding
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
if not hasattr(MiniMaxH3Model, "_forward"):
    raise RuntimeError("ComfyUI MiniMaxH3Model._forward is unavailable for the plugin fallback test")
try:
    h3_embedding._validate_upstream_forward(MiniMaxH3Model._forward)
except h3_embedding.H3EmbeddingMemoryPatchError as exc:
    print("[H3 OPT MEMORY] ComfyUI 0.43.2 upstream forward rejected by pinned hash guard; native fallback expected:", str(exc)[:300])
else:
    raise RuntimeError("H3-Optimizations unexpectedly accepted a different ComfyUI H3 _forward implementation")
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
# Package wrapper flags do not prove the underlying SM75 binary was loaded.
expected_sage_version = str(os.environ.get("H3_EXPECTED_SAGE_VERSION", "")).strip()
if expected_sage_version and str(getattr(sageattention, "__version__", "")).strip() != expected_sage_version:
    raise RuntimeError(
        f"SageAttention package mismatch: expected={expected_sage_version}, actual={getattr(sageattention, '__version__', '')}"
    )
sage_package = Path(getattr(sageattention, "__file__", "")).resolve()
expected_sage_root = Path(os.environ["H3_SAGE_ROOT"]).resolve()
if not sage_package.is_relative_to(expected_sage_root / "sageattention"):
    raise RuntimeError(f"SageAttention imported outside pinned checkout: {sage_package}")
if not bool(getattr(sageattention, "SM75_CUDA_ENABLED", False)):
    raise RuntimeError("SageAttention SM75 CUDA wrapper is not enabled in fresh H3 verification")
if not bool(getattr(_sage_core, "SM75_ENABLED", False)):
    raise RuntimeError("Fresh H3 verification could not load SageAttention _qattn_sm75")
if not bool(getattr(_sage_core, "FUSED_ENABLED", False)):
    raise RuntimeError("Fresh H3 verification could not load SageAttention _fused")
if not callable(getattr(_sage_sm75, "qk_int8_sv_f16_accum_f32_attn_sm75", None)):
    raise RuntimeError("Fresh H3 verification found no SM75 CUDA attention entrypoint")
if not callable(getattr(_sage_core, "per_warp_int8_cuda", None)):
    raise RuntimeError("Fresh H3 verification found no fused per-warp quantizer")
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


def _ordered_custom_nodes(manifest: dict) -> list[dict]:
    """Return custom nodes in deterministic dependency order.

    The manifest's ``depends_on`` declarations are part of the runtime contract.
    Install order must not rely on the incidental YAML ordering of the current
    file, so perform a stable topological sort and fail closed on unknown or
    cyclic dependencies.
    """
    custom_nodes = manifest.get("custom_nodes", {}) or {}
    nodes = list(custom_nodes.get("required", []) or []) + list(
        custom_nodes.get("supporting", []) or []
    )
    by_name: dict[str, dict] = {}
    order_index: dict[str, int] = {}
    for index, node in enumerate(nodes):
        name = str(node.get("name", "") or "").strip()
        if not name:
            raise RuntimeError("Custom-node manifest contains a node with no name.")
        if name in by_name:
            raise RuntimeError(f"Custom-node manifest contains duplicate node: {name}")
        by_name[name] = node
        order_index[name] = index

    indegree = {name: 0 for name in by_name}
    dependents: dict[str, list[str]] = {name: [] for name in by_name}
    for name, node in by_name.items():
        dependencies = [
            str(value).strip()
            for value in (node.get("depends_on", []) or [])
            if str(value).strip()
        ]
        for dependency in dependencies:
            if dependency not in by_name:
                raise RuntimeError(
                    f"Custom-node {name!r} depends on unknown node {dependency!r}."
                )
            indegree[name] += 1
            dependents[dependency].append(name)

    ready = sorted(
        [name for name, degree in indegree.items() if degree == 0],
        key=order_index.__getitem__,
    )
    ordered: list[dict] = []
    while ready:
        name = ready.pop(0)
        ordered.append(by_name[name])
        for dependent in sorted(dependents[name], key=order_index.__getitem__):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
                ready.sort(key=order_index.__getitem__)

    if len(ordered) != len(by_name):
        cyclic = sorted(
            [name for name, degree in indegree.items() if degree > 0],
            key=order_index.__getitem__,
        )
        raise RuntimeError(
            "Custom-node dependency cycle detected: " + ", ".join(cyclic)
        )
    return ordered


def install_nodes() -> None:
    manifest = load_yaml(
        NODE_MANIFEST
    )
    policy = dict(manifest.get("policy", {}) or {})
    if policy.get("clone_at_runtime") is not True:
        raise RuntimeError(
            "custom_nodes.yaml policy.clone_at_runtime must be true for the locked Kaggle bootstrap."
        )
    if policy.get("pin_revisions") is not True:
        raise RuntimeError(
            "custom_nodes.yaml policy.pin_revisions must be true for the locked Kaggle bootstrap."
        )
    if policy.get("vendor_source_code") is not False:
        raise RuntimeError(
            "custom_nodes.yaml policy.vendor_source_code must be false for the locked bootstrap."
        )
    if policy.get("third_party_models") is not False:
        raise RuntimeError(
            "custom_nodes.yaml policy.third_party_models must be false for the locked bootstrap."
        )
    CUSTOM.mkdir(
        parents=True,
        exist_ok=True,
    )
    for node in _ordered_custom_nodes(manifest):
        name = str(node.get("name", "") or "").strip()
        repository = str(node.get("repository", "") or "").strip()
        revision = str(node.get("revision", "") or "").strip()
        if not repository or not revision:
            raise RuntimeError(
                f"Custom-node {name!r} must declare repository and pinned revision."
            )
        if len(revision) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in revision):
            raise RuntimeError(
                f"Custom-node {name!r} revision must be a full 40-character SHA: {revision!r}"
            )
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
            checkout_locked = False
        elif not (destination / ".git").exists():
            raise RuntimeError(
                f"Custom-node path exists but is not a git checkout: {destination}"
            )
        else:
            try:
                checkout_locked = _git_head(destination) == revision
            except (OSError, subprocess.CalledProcessError):
                checkout_locked = False
        if checkout_locked:
            print("[BOOTSTRAP WARM] reusing locked node", node["name"])
        else:
            _ensure_git_revision(destination, revision)
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
            if _requirements_satisfied(requirements):
                print("[BOOTSTRAP WARM] node requirements already satisfied", node["name"])
            else:
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

# Project-owned SageAttention SM75 correction recipe.
# This is intentionally embedded here so the bootstrap is self-contained:
# no external patch file is required at runtime.
# The pinned upstream checkout is reset first; these exact source contracts are
# then applied in-place with fail-closed replacement counts. No external patch
# artifact or checksum bookkeeping is required.
def _apply_sage_sm75_source_corrections(source: str) -> str:
    old = r"""            // CRITICAL FIX: Renormalize RO AFTER computing both mq but BEFORE PV MMA
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

            // --- PV Computation (m16n8k8 FP16 MMA): RO += P × V ---
"""
    new = r"""            // H3-T4-SM75-PV-FRAGMENT-FIX
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

            // --- PV Computation (m16n8k8 FP16 MMA): RO += P × V ---
"""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"SageAttention SM75 PV renormalization correction: expected exactly one source match, found {count}"
        )
    source = source.replace(old, new, 1)
    print("[SAGE SM75 PATCH] SM75 PV renormalization correction: PASS")

    old = r"""    // --- Final Normalization ---
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

    // --- Output ---
"""
    new = r"""    // --- Final Normalization ---
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

    // --- Output ---
"""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"SageAttention SM75 final normalization correction: expected exactly one source match, found {count}"
        )
    source = source.replace(old, new, 1)
    print("[SAGE SM75 PATCH] SM75 final normalization correction: PASS")

    old = r"""        // For m16n8k8, each thread covers a 2×2 block of the 16×8 output.
        // Thread i = lane_id: rows = 2*(i/4) and 2*(i/4)+1, cols = 2*(i%4) and 2*(i%4)+1
        uint32_t thread_row0 = (lane_id / 4) * 2;
        uint32_t thread_row1 = thread_row0 + 1;
        uint32_t thread_col0 = (lane_id % 4) * 2;
        uint32_t thread_col1 = thread_col0 + 1;
        uint32_t smem_row_base = o_start_row_warp;
"""
    new = r"""        // H3-T4-SM75-PV-FRAGMENT-FIX: c0/c1 are row=r, c2/c3 are row=r+8.
        uint32_t thread_row0 = lane_id / 4;
        uint32_t thread_row1 = thread_row0 + 8;
        uint32_t thread_col0 = (lane_id % 4) * 2;
        uint32_t thread_col1 = thread_col0 + 1;
        uint32_t smem_row_base = o_start_row_warp;
"""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"SageAttention SM75 staged-output fragment mapping correction: expected exactly one source match, found {count}"
        )
    source = source.replace(old, new, 1)
    print("[SAGE SM75 PATCH] SM75 staged-output fragment mapping correction: PASS")

    old = r"""        // Path B: Direct scattered write using the 2×2 per-thread mapping
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
    }
"""
    new = r"""        // Path B: Direct scattered write using the SM75 fragment row mapping.
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
    }
"""
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"SageAttention SM75 direct-output fragment mapping correction: expected exactly one source match, found {count}"
        )
    source = source.replace(old, new, 1)
    print("[SAGE SM75 PATCH] SM75 direct-output fragment mapping correction: PASS")

    legacy = "constexpr int WARP_K_SM75 = 16;"
    fixed = "constexpr int WARP_K_SM75 = 64;"
    patch_marker = (
        "// H3-T4-SM75-KERNEL-FIX: pair each Q warp with the full CTA_K tile "
        "to eliminate cross-K-warp softmax/output races.\n"
    )
    legacy_count = source.count(legacy)
    fixed_count = source.count(fixed)
    marker_count = source.count("H3-T4-SM75-KERNEL-FIX")
    if not (legacy_count == 2 and fixed_count == 0 and marker_count == 0):
        raise RuntimeError(
            "SageAttention SM75 WARP_K source contract changed: expected exactly "
            "two unmodified WARP_K_SM75=16 declarations before applying the fix "
            f"(legacy={legacy_count}, fixed={fixed_count}, markers={marker_count})"
        )
    source = source.replace(legacy, patch_marker + fixed)
    print("[SAGE SM75 PATCH] SM75 WARP_K dual-declaration correction: PASS")

    return source

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

    _ensure_git_revision(install_dir, revision)
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

    sage_kernel = install_dir / "csrc" / "qattn" / "attn_cuda_sm75.h"
    if not sage_kernel.is_file():
        raise RuntimeError(f"Pinned SageAttention SM75 kernel source is missing: {sage_kernel}")

    source = sage_kernel.read_text(encoding="utf-8")
    source = _apply_sage_sm75_source_corrections(source)
    sage_kernel.write_text(source, encoding="utf-8")
    print("[SAGE SM75 PATCH] embedded source corrections applied: PASS")

    sage_source = sage_kernel.read_text(encoding="utf-8")
    required_patch_contract = (
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
    missing = [needle for needle in required_patch_contract if needle not in sage_source]
    if missing:
        raise RuntimeError(
            "SageAttention SM75 embedded patch application verification failed:\n"
            + "\n".join(missing)
        )
    if "constexpr int WARP_K_SM75 = 16;" in sage_source:
        raise RuntimeError("SageAttention SM75 patch application left a legacy WARP_K_SM75=16 declaration.")
    print("[SAGE SM75 PATCH] embedded bootstrap patch applied: PASS")

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
        child_env["H3_SAGE_PHYSICAL_GPU_ID"] = str(gpu_id)
        probe = r'''import inspect
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import sageattention
from sageattention import sageattn
from sageattention import _qattn_sm75 as _sage_sm75
from sageattention import core as _sage_core

expected = os.environ["H3_EXPECTED_SAGE_VERSION"]
root = Path(os.environ["H3_SAGE_ROOT"]).resolve()
physical_gpu_id = os.environ["H3_SAGE_PHYSICAL_GPU_ID"]

# Package wrapper flags do not prove the underlying SM75 binary was loaded.
if str(getattr(sageattention, "__version__", "")).strip() != expected:
    raise RuntimeError(
        f"SageAttention version mismatch: expected={expected}, actual={getattr(sageattention, '__version__', '')}"
    )

package_file = Path(getattr(sageattention, "__file__", "")).resolve()
if not package_file.is_relative_to(root / "sageattention"):
    raise RuntimeError(f"SageAttention imported outside pinned checkout: {package_file}")
if not bool(getattr(sageattention, "SM75_CUDA_ENABLED", False)):
    raise RuntimeError("SageAttention SM75 CUDA wrapper is not enabled")
if not bool(getattr(_sage_core, "SM75_ENABLED", False)):
    raise RuntimeError("SageAttention core could not load the compiled _qattn_sm75 extension")
if not bool(getattr(_sage_core, "FUSED_ENABLED", False)):
    raise RuntimeError("SageAttention core could not load the compiled _fused extension")
if not callable(getattr(_sage_sm75, "qk_int8_sv_f16_accum_f32_attn_sm75", None)):
    raise RuntimeError("SageAttention _qattn_sm75 extension lacks its SM75 attention entrypoint")
if not callable(getattr(_sage_core, "per_warp_int8_cuda", None)):
    raise RuntimeError("SageAttention fused per-warp quantizer is unavailable")
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
    cosine_tensor = F.cosine_similarity(
        out.float().reshape(1, -1),
        quant_reference.float().reshape(1, -1),
        dim=1,
    )
    if not torch.isfinite(cosine_tensor).all().item():
        raise RuntimeError("SageAttention SM75 cosine similarity is non-finite")
    # Cosine similarity is mathematically bounded to [-1, 1], but the
    # floating-point reduction can produce tiny round-off overshoots such as
    # 1.00000012. Clamp only the reported/gated metric; do not alter outputs.
    cosine = float(torch.clamp(cosine_tensor, -1.0, 1.0).item())
    if not math.isfinite(cosine):
        raise RuntimeError("SageAttention SM75 cosine similarity is non-finite after clamping")
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
        f"[SAGE SM75 CASE] gpu={physical_gpu_id} hd={head_dim} smooth_k={smooth_k} "
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
            f"gpu={physical_gpu_id}, hd={head_dim}, smooth_k={smooth_k}, relative_l2={relative_l2:.6g}, "
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

    stamp = _sage_build_stamp(runtime, install_dir)
    _sage_build_stamp_path(install_dir).write_text(
        json.dumps(stamp, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[SAGE SM75] repository={repository} revision={actual_revision} "
        f"version={expected_version} build=PASS gpu0=PASS gpu1=PASS"
    )


def install_models() -> None:
    manifest = load_yaml(MODEL_MANIFEST)
    models = tuple(manifest["models"].values())
    filenames = tuple(str(model["filename"]) for model in models)
    normalized_filenames = tuple(name.lower() for name in filenames)
    if len(set(normalized_filenames)) != len(normalized_filenames):
        raise RuntimeError(
            "Model manifest contains duplicate filenames (case-insensitive): "
            + ", ".join(filenames)
        )
    sources = find_kaggle_files(filenames)
    for model in models:
        filename = str(model["filename"])
        source = sources[filename.lower()]
        destination = MODELS / model["directory"] / filename
        link_model(source, destination)
        print("[MODEL]", filename)
_BOOTSTRAP_PHASES: list[dict] = []


def _timed_call(label: str, function, *args, **kwargs):
    started = time.perf_counter()
    print(f"[BOOTSTRAP TIMING] {label}: started")
    status = "passed"
    try:
        return function(*args, **kwargs)
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = round(time.perf_counter() - started, 3)
        _BOOTSTRAP_PHASES.append({"phase": label, "seconds": elapsed, "status": status})
        print(f"[BOOTSTRAP TIMING] {label}: {elapsed:.3f}s ({status})")


def _write_bootstrap_metrics(started: float, status: str) -> None:
    metrics = {
        "schema": 1,
        "status": status,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "python": sys.version,
        "phases": list(_BOOTSTRAP_PHASES),
    }
    destination = ROOT / "bootstrap_metrics.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
        print(f"[BOOTSTRAP METRICS] {destination}")
    except OSError as exc:
        print(f"[BOOTSTRAP METRICS WARNING] unable to write metrics: {exc}")


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
    bootstrap_started = time.perf_counter()
    bootstrap_status = "failed"
    _BOOTSTRAP_PHASES.clear()
    _enforce_no_restart()
    _warn_if_torch_already_imported()
    runtime = load_yaml(RUNTIME_MANIFEST)
    pytorch_cuda = str(runtime["pytorch"]["cuda"])
    legacy_runtime = ROOT / f".h3_runtime_{pytorch_cuda}"
    if legacy_runtime.exists():
        shutil.rmtree(legacy_runtime, ignore_errors=True)
    _timed_call("ensure_kaggle_startup_wrapt", ensure_kaggle_startup_wrapt, runtime)
    director_config = dict(runtime.get("director", {}) or {})
    configured_model_path = str(director_config.get("model_path", "") or "").strip()
    director_model_path = os.getenv("H3_DIRECTOR_MODEL_PATH", "").strip() or configured_model_path
    if not director_model_path:
        raise RuntimeError(
            "runtime_versions.yaml director.model_path is required unless "
            "H3_DIRECTOR_MODEL_PATH is set."
        )
    director_model = Path(director_model_path).expanduser().resolve()
    print("[DIRECTOR MODEL]", director_model)

    director_executor = None
    director_future = None
    director_env = Path(
        os.getenv(
            "H3_DIRECTOR_VLLM_ENV_DIR",
            str(runtime.get("director", {}).get("vllm_env_dir", "")),
        )
    ).expanduser().resolve()
    director_ready = _probe_director_venv(runtime, director_env)
    if director_ready:
        _BOOTSTRAP_PHASES.append({"phase": "reuse_director_vllm_environment", "seconds": 0.0, "status": "reused"})
        print(f"[BOOTSTRAP WARM] reusing Director vLLM environment={director_env}")
    elif shutil.which("uv") is not None:
        from concurrent.futures import ThreadPoolExecutor
        director_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-qwen-bootstrap")
        director_future = director_executor.submit(
            _timed_call, "install_director_runtime", install_director_runtime, runtime
        )
        print("[BOOTSTRAP PARALLEL] Director vLLM install started")

    sage_executor = None
    sage_future = None
    sage_dir = CUSTOM / str(runtime.get("sage_attention", {}).get("directory", "") or "")

    try:
        _timed_call("install_comfyui", install_comfyui, runtime)
        _timed_call("install_custom_nodes", install_nodes)
        _timed_call("remove_legacy_context_ir_node", remove_legacy_context_ir_node)
        _timed_call("install_base_requirements", install_base_requirements)
        _timed_call("install_storyboard_runtime", install_storyboard_runtime, runtime)
        _timed_call("install_pytorch_runtime", install_pytorch_runtime, runtime)
        _timed_call("install_and_verify_pillow_runtime", install_and_verify_pillow_runtime, runtime)

        if _sage_install_is_reusable(runtime, sage_dir):
            _BOOTSTRAP_PHASES.append({"phase": "reuse_sageattention_sm75_build", "seconds": 0.0, "status": "reused"})
            print(f"[BOOTSTRAP WARM] reusing SageAttention SM75 build={sage_dir}")
        else:
            from concurrent.futures import ThreadPoolExecutor
            sage_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-sage-bootstrap")
            sage_future = sage_executor.submit(
                _timed_call, "install_sageattention_sm75", install_sageattention_sm75, runtime
            )
            print("[BOOTSTRAP PARALLEL] SageAttention build started")

        _timed_call("prepare_h3_optimization_source", prepare_locked_h3_optimization_source, runtime)
        _timed_call("patch_h3_bounded_qkv", patch_h3_bounded_qkv, runtime)
        _timed_call("patch_h3_qkv_binding_lifetime", patch_h3_qkv_binding_lifetime, runtime)
        _timed_call("patch_h3_sage_attention", patch_h3_sage_attention, runtime)
        _timed_call("apply_embedded_h3_runtime_overlay", apply_embedded_h3_runtime_overlay)
        _timed_call("patch_h3_embedding_memory_compatibility", patch_h3_embedding_memory_compatibility, runtime)

        if sage_future is not None:
            sage_future.result()

        if director_future is not None:
            director_future.result()
        elif not director_ready:
            _timed_call("install_director_runtime", install_director_runtime, runtime)

        _timed_call("verify_h3_optimization_runtime", verify_h3_optimization_runtime, runtime)
        _timed_call("install_models", install_models)
        _timed_call("verify_inventory", verify_inventory)
        _timed_call("verify_runtime_files", verify_runtime_files, runtime)
        print("=" * 80)
        print("MiniMax H3 Kaggle bootstrap PASSED.")
        bootstrap_status = "passed"
    finally:
        if director_executor is not None:
            director_executor.shutdown(wait=True, cancel_futures=False)
        if sage_executor is not None:
            sage_executor.shutdown(wait=True, cancel_futures=False)
        _write_bootstrap_metrics(bootstrap_started, bootstrap_status)

if __name__ == "__main__":
    main()

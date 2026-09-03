from __future__ import annotations

import gc
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from pipeline.vram_profile import VRAMProfile, resolve_vram_profile
from planner.config import RUNTIME


class H3Runtime:
    """Manage one isolated ComfyUI worker per physical GPU."""

    @staticmethod
    def clear_cuda():
        try:
            import torch
            gc.collect()
            if not torch.cuda.is_available():
                return
            for device_id in range(torch.cuda.device_count()):
                with torch.cuda.device(device_id):
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
        except Exception:
            pass

    @staticmethod
    def worker_environment(gpu_id: int, *, extra_env: dict[str, str] | None = None) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTORCH_CUDA_ALLOC_CONF"] = env.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True",
        )
        env["PYTHONUNBUFFERED"] = "1"
        env["H3_LOCKED_TORCH_RUNTIME"] = "2.10.0+cu130"
        if extra_env:
            env.update({str(k): str(v) for k, v in extra_env.items()})
        return env

    @staticmethod
    def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((host, int(port)))
            except OSError:
                return False
            return True

    @staticmethod
    def _resolve_bool(value: Any, default: bool) -> bool:
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def vram_handoff(clients=None, *, unload_models: bool = True, min_free_vram_gib: float = 1.0) -> None:
        """Flush remote ComfyUI models and verify worker health before H3 rendering."""
        for client in (clients or {}).values():
            free_and_wait = getattr(client, "free_and_wait", None)
            if callable(free_and_wait):
                if not free_and_wait(unload_models=unload_models, free_memory=True):
                    raise RuntimeError("ComfyUI refused the VRAM handoff request.")
            else:
                free = getattr(client, "free_memory", None)
                if callable(free) and not free(unload_models=unload_models, free_memory=True):
                    raise RuntimeError("ComfyUI refused the VRAM free request.")
        H3Runtime.clear_cuda()
        # Controller-side CUDA memory is only a secondary signal. Worker-side
        # /system_stats must also report healthy availability after /free.
        for client in (clients or {}).values():
            stats_fn = getattr(client, "get_system_stats", None)
            if not callable(stats_fn):
                continue
            stats = stats_fn()
            devices = stats.get("devices", []) if isinstance(stats, dict) else []
            if not devices:
                continue
            for device in devices:
                free = device.get("vram_free")
                if free is None:
                    continue
                try:
                    free_gib = float(free) / (1024 ** 3)
                except (TypeError, ValueError):
                    continue
                if free_gib < float(min_free_vram_gib):
                    raise RuntimeError(f"Insufficient worker free VRAM after handoff: {free_gib:.2f} GiB < {min_free_vram_gib:.2f} GiB.")

    @classmethod
    def launch_worker(
        cls,
        comfy_root: Path,
        gpu_id: int,
        port: int,
        log_path: Path,
        *,
        lowvram: bool = True,
        cpu_vae: bool = True,
        extra_args: list[str] | None = None,
        vram_profile: VRAMProfile | None = None,
    ):
        comfy_root = Path(comfy_root).resolve()
        main_py = comfy_root / "main.py"
        if not main_py.is_file():
            raise FileNotFoundError(
                f"ComfyUI main.py not found at {main_py}. Run kaggle/bootstrap.py first."
            )
        if not cls.port_is_free(port):
            raise RuntimeError(f"ComfyUI port {port} is already in use.")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        # ComfyUI 0.34+ uses DynamicVRAM by default on Nvidia. Explicitly keep
        # asynchronous weight offload enabled; it is a core H3 performance path.
        if vram_profile is None:
            try:
                from planner.config import RUNTIME
                runtime_vram = dict(RUNTIME.get("runtime", {}).get("vram", {}) or {})
                configured_cpu_vae = RUNTIME.get("runtime", {}).get("cpu_vae", cpu_vae)
                runtime_vram["cpu_vae"] = configured_cpu_vae
            except Exception:
                runtime_vram = {"cpu_vae": cpu_vae}
            profile = resolve_vram_profile(runtime_vram, gpu_id=gpu_id)
        else:
            profile = vram_profile
        if lowvram:
            command.append("--lowvram")
        if profile.async_offload_streams > 0:
            command.extend(["--async-offload", str(profile.async_offload_streams)])
        if profile.reserve_vram_gb > 0:
            command.extend(["--reserve-vram", str(profile.reserve_vram_gb)])
        if profile.disable_pinned_memory:
            command.append("--disable-pinned-memory")
        if profile.fast_disk:
            command.append("--fast-disk")
        if profile.cpu_vae:
            command.append("--cpu-vae")
        if extra_args:
            command.extend(str(v) for v in extra_args)

        try:
            process = subprocess.Popen(
                command,
                cwd=str(comfy_root),
                env=cls.worker_environment(gpu_id),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:
            try:
                handle.close()
            finally:
                raise
        return process, handle

    @staticmethod
    def wait_http(url: str, timeout: float = 300.0, *, process=None) -> None:
        import urllib.request

        started = time.monotonic()
        last_error = None
        while time.monotonic() - started < timeout:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited before readiness check completed (code {process.returncode})."
                )
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        return
            except Exception as error:
                last_error = error
            time.sleep(1.5)
        detail = f" Last error: {last_error}" if last_error else ""
        raise TimeoutError(f"ComfyUI did not start: {url}.{detail}")

    @classmethod
    def launch_workers(
        cls,
        project_root: Path,
        gpu_ids: list[int],
        base_port: int = 8188,
        *,
        lowvram: bool = True,
        cpu_vae: bool = True,
        startup_timeout: float | None = None,
    ) -> dict[int, dict[str, Any]]:
        if startup_timeout is None:
            runtime_cfg = dict(RUNTIME.get("runtime", {}) or {})
            startup_timeout = float(
                os.getenv(
                    "H3_COMFY_STARTUP_TIMEOUT",
                    str(runtime_cfg.get("comfyui_startup_timeout_seconds", 300)),
                )
            )
        lowvram = H3Runtime._resolve_bool(
            os.getenv("H3_COMFY_LOWVRAM"), lowvram
        )
        cpu_vae_env = os.getenv("H3_COMFY_CPU_VAE")
        if cpu_vae_env is not None:
            cpu_vae = H3Runtime._resolve_bool(cpu_vae_env, cpu_vae)
        import torch

        # Fail closed before launching any worker if the live interpreter does
        # not match the project-locked CUDA build. This prevents another long
        # H3 run on a silently downgraded cu128 environment.
        try:
            pytorch_cfg = dict(RUNTIME.get("pytorch", {}) or {})
            expected_version = str(pytorch_cfg.get("version", "2.10.0") or "2.10.0").strip()
            expected_cuda = str(pytorch_cfg.get("cuda", "cu130") or "cu130").strip().lower()
            expected_torch = f"{expected_version}+{expected_cuda}"
            if torch.__version__ != expected_torch or torch.version.cuda != "13.0":
                raise RuntimeError(
                    "H3 worker refused to start because the live PyTorch runtime "
                    f"does not match the locked project runtime: got {torch.__version__} / "
                    f"CUDA {torch.version.cuda}, expected {expected_torch} / CUDA 13.0."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to validate locked PyTorch runtime: {exc}") from exc

        try:
            runtime_cfg = dict(RUNTIME.get("runtime", {}) or {})
            vram_cfg = dict(runtime_cfg.get("vram", {}) or {})
            vram_cfg["cpu_vae"] = runtime_cfg.get("cpu_vae", cpu_vae)
        except Exception:
            vram_cfg = {"cpu_vae": cpu_vae}

        if not torch.cuda.is_available():
            raise RuntimeError("NVIDIA CUDA is required for the production H3 runtime.")
        available = torch.cuda.device_count()
        normalized_gpu_ids = sorted({int(g) for g in gpu_ids})
        if not normalized_gpu_ids:
            raise ValueError("At least one GPU is required.")
        for gpu_id in normalized_gpu_ids:
            if gpu_id < 0 or gpu_id >= available:
                raise RuntimeError(
                    f"GPU {gpu_id} is unavailable. Detected {available} GPUs."
                )

        project_root = Path(project_root).resolve()
        comfy_root = project_root / "ComfyUI"
        if not (comfy_root / "main.py").is_file():
            raise RuntimeError(
                f"ComfyUI is not installed at {comfy_root}. Run kaggle/bootstrap.py first."
            )

        log_root = project_root / "data" / "workers"
        processes: dict[int, dict[str, Any]] = {}
        used_ports: set[int] = set()
        try:
            for offset, gpu_id in enumerate(normalized_gpu_ids):
                port = int(base_port) + offset
                # Resolve VRAM policy against the actual worker GPU. This avoids
                # accidentally using the largest GPU's profile on a smaller GPU
                # in heterogeneous multi-GPU hosts.
                profile = resolve_vram_profile(vram_cfg, gpu_id=gpu_id)
                if port in used_ports or not cls.port_is_free(port):
                    raise RuntimeError(f"Cannot allocate free ComfyUI port {port} for GPU {gpu_id}.")
                used_ports.add(port)
                process, handle = cls.launch_worker(
                    comfy_root,
                    gpu_id,
                    port,
                    log_root / f"gpu_{gpu_id}.log",
                    lowvram=lowvram,
                    cpu_vae=profile.cpu_vae,
                    vram_profile=profile,
                )
                try:
                    cls.wait_http(
                        f"http://127.0.0.1:{port}/system_stats",
                        timeout=startup_timeout,
                        process=process,
                    )
                    cls.wait_http(
                        f"http://127.0.0.1:{port}/object_info",
                        timeout=60,
                        process=process,
                    )
                except Exception:
                    cls._terminate_one(process, handle)
                    raise
                processes[gpu_id] = {
                    "process": process,
                    "handle": handle,
                    "url": f"http://127.0.0.1:{port}",
                    "port": port,
                }
        except Exception:
            cls.stop_workers(processes)
            raise
        return processes

    @staticmethod
    def _terminate_one(process, handle) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            try:
                handle.close()
            except Exception:
                pass

    @classmethod
    def stop_workers(cls, workers):
        for item in (workers or {}).values():
            process = item.get("process")
            handle = item.get("handle")
            if process is not None:
                cls._terminate_one(process, handle)

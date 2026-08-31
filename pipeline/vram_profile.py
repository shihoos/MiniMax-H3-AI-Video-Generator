from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VRAMProfile:
    name: str
    async_offload_streams: int
    cpu_vae: bool
    disable_pinned_memory: bool
    fast_disk: bool
    reserve_vram_gib: float
    reason: str


def _env_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _system_memory_gib() -> tuple[float, float]:
    total = available = 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            values = {}
            for line in handle:
                key, _, raw = line.partition(":")
                if key in {"MemTotal", "MemAvailable"}:
                    values[key] = float(raw.strip().split()[0]) / (1024 ** 2)
            total = values.get("MemTotal", 0.0)
            available = values.get("MemAvailable", total)
    except Exception:
        try:
            import psutil  # type: ignore
            mem = psutil.virtual_memory()
            total = float(mem.total) / (1024 ** 3)
            available = float(mem.available) / (1024 ** 3)
        except Exception:
            pass
    return total, available


def _gpu_vram_gib() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        return max(
            float(torch.cuda.get_device_properties(i).total_memory) / (1024 ** 3)
            for i in range(torch.cuda.device_count())
        )
    except Exception:
        return 0.0


def resolve_vram_profile(config: dict[str, Any] | None = None) -> VRAMProfile:
    cfg = dict(config or {})
    requested = str(os.getenv("H3_VRAM_PROFILE", cfg.get("profile", "auto")) or "auto").strip().lower()
    async_streams = max(1, int(os.getenv("H3_COMFY_ASYNC_OFFLOAD_STREAMS", cfg.get("async_offload_streams", 2))))
    reserve = max(0.0, float(os.getenv("H3_COMFY_RESERVE_VRAM_GIB", cfg.get("reserve_vram_gib", 0.0))))
    total_ram, available_ram = _system_memory_gib()
    gpu_vram = _gpu_vram_gib()
    ram_ratio = (available_ram / total_ram) if total_ram > 0 else 1.0

    if requested not in {"auto", "low_vram", "balanced", "throughput"}:
        raise ValueError(f"Unknown H3_VRAM_PROFILE={requested!r}")

    if requested == "auto":
        if gpu_vram and gpu_vram < 20.0:
            requested = "low_vram"
        elif gpu_vram and gpu_vram >= 40.0:
            requested = "throughput"
        else:
            requested = "balanced"

    configured_cpu_vae = cfg.get("cpu_vae", True)
    if str(configured_cpu_vae).strip().lower() == "auto":
        cpu_vae = requested == "low_vram" or gpu_vram < 32.0
    else:
        cpu_vae = _env_bool(os.getenv("H3_COMFY_CPU_VAE"), _env_bool(configured_cpu_vae, True))

    disable_pinned_raw = os.getenv("H3_COMFY_DISABLE_PINNED_MEMORY", str(cfg.get("disable_pinned_memory", "auto")))
    if disable_pinned_raw.strip().lower() == "auto":
        # Pinned host memory is normally useful for throughput. Disable it only
        # when system RAM is under real pressure; this avoids the common H3
        # failure mode where offload consumes most of host RAM.
        disable_pinned = total_ram > 0 and ram_ratio < float(cfg.get("ram_available_floor", 0.18))
    else:
        disable_pinned = _env_bool(disable_pinned_raw, False)

    fast_disk = _env_bool(os.getenv("H3_COMFY_FAST_DISK"), _env_bool(cfg.get("fast_disk"), False))
    if requested == "low_vram" and total_ram and available_ram < 8.0:
        fast_disk = True

    reasons = {
        "low_vram": "Prioritize fit and stability on smaller GPUs.",
        "balanced": "Keep ComfyUI DynamicVRAM and async offload active without aggressive experimental flags.",
        "throughput": "Use GPU VAE where practical while retaining DynamicVRAM and async offload.",
    }
    return VRAMProfile(
        name=requested,
        async_offload_streams=async_streams,
        cpu_vae=cpu_vae,
        disable_pinned_memory=disable_pinned,
        fast_disk=fast_disk,
        reserve_vram_gib=reserve,
        reason=reasons[requested],
    )

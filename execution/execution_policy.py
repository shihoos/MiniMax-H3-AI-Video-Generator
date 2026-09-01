from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pipeline.vram_profile import VRAMProfile, resolve_vram_profile

ExecutionMode = Literal["production", "preview", "retake", "diagnostic"]


def _feature(config: dict, name: str, default: bool) -> bool:
    value = config.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ExecutionPolicy:
    """Single immutable execution contract for every H3 execution mode."""

    mode: ExecutionMode = "production"
    turbo: bool = False
    upscale: bool = False
    live_preview: bool = True
    require_context_ir: bool = True
    run_visual_qa: bool = True
    auto_retake: bool = True
    max_auto_retries: int = 1
    vram_profile: VRAMProfile | None = None
    allow_preview_mode: bool = True
    allow_diagnostic_mode: bool = True
    allow_retake_mode: bool = True

    @classmethod
    def from_runtime(
        cls,
        *,
        mode: ExecutionMode = "production",
        turbo: bool = False,
        upscale: bool = False,
        gpu_id: int | None = None,
    ) -> "ExecutionPolicy":
        from planner.config import RUNTIME

        features = dict(RUNTIME.get("features", {}) or {})
        max_retries = max(0, int(features.get("max_auto_retakes_per_shot", 1) or 0))
        auto_retake = (
            _feature(features, "auto_retake", True)
            and _feature(features, "selective_retake", True)
            and _feature(features, "qa_enabled", True)
        )
        runtime_vram = dict(RUNTIME.get("runtime", {}).get("vram", {}) or {})
        runtime_vram["cpu_vae"] = RUNTIME.get("runtime", {}).get("cpu_vae", "auto")
        vram_profile = resolve_vram_profile(runtime_vram, gpu_id=gpu_id)
        return cls(
            mode=mode,
            turbo=bool(turbo),
            upscale=bool(upscale),
            live_preview=_feature(features, "live_preview", True),
            require_context_ir=_feature(features, "context_ir_required", True),
            run_visual_qa=_feature(features, "vlm_visual_qa", True) and _feature(features, "qa_enabled", True),
            auto_retake=auto_retake,
            max_auto_retries=max_retries if auto_retake else 0,
            vram_profile=vram_profile,
            allow_preview_mode=_feature(features, "preview_execution_mode", True),
            allow_diagnostic_mode=_feature(features, "diagnostic_execution_mode", True),
            allow_retake_mode=_feature(features, "retake_execution_mode", True),
        ).for_mode(mode)

    def for_mode(self, mode: ExecutionMode) -> "ExecutionPolicy":
        requested = str(mode)
        if requested not in {"production", "preview", "retake", "diagnostic"}:
            raise ValueError(f"Unsupported execution mode: {requested!r}")
        if requested == "preview" and not self.allow_preview_mode:
            raise RuntimeError("Preview execution mode is disabled by runtime configuration.")
        if requested == "diagnostic" and not self.allow_diagnostic_mode:
            raise RuntimeError("Diagnostic execution mode is disabled by runtime configuration.")
        if requested == "retake" and not self.allow_retake_mode:
            raise RuntimeError("Retake execution mode is disabled by runtime configuration.")
        if requested == "preview":
            return replace(self, mode="preview", upscale=False, auto_retake=False, max_auto_retries=0)
        if requested == "diagnostic":
            return replace(self, mode="diagnostic", upscale=False, live_preview=False, run_visual_qa=False, auto_retake=False, max_auto_retries=0)
        if requested == "retake":
            return replace(self, mode="retake", auto_retake=False, max_auto_retries=0)
        return replace(self, mode="production")

    def as_dict(self) -> dict:
        """Return the resolved immutable execution contract for diagnostics/manifests."""
        profile = self.vram_profile
        return {
            "mode": self.mode,
            "turbo": self.turbo,
            "upscale": self.upscale,
            "live_preview": self.live_preview,
            "require_context_ir": self.require_context_ir,
            "run_visual_qa": self.run_visual_qa,
            "auto_retake": self.auto_retake,
            "max_auto_retries": self.max_auto_retries,
            "allow_preview_mode": self.allow_preview_mode,
            "allow_diagnostic_mode": self.allow_diagnostic_mode,
            "allow_retake_mode": self.allow_retake_mode,
            "vram_profile": ({
                "name": profile.name,
                "cpu_vae": profile.cpu_vae,
                "async_offload_streams": profile.async_offload_streams,
                "disable_pinned_memory": profile.disable_pinned_memory,
                "fast_disk": profile.fast_disk,
            } if profile is not None else None),
        }

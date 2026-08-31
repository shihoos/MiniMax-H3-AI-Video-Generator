from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

ExecutionMode = Literal["production", "preview", "retake", "diagnostic"]


@dataclass(frozen=True)
class ExecutionPolicy:
    """Single execution contract shared by normal, preview, retake and diagnostic runs."""

    mode: ExecutionMode = "production"
    turbo: bool = False
    upscale: bool = False
    live_preview: bool = True
    require_context_ir: bool = True
    run_visual_qa: bool = True
    auto_retake: bool = False
    max_auto_retries: int = 1

    def for_mode(self, mode: ExecutionMode) -> "ExecutionPolicy":
        mode = str(mode)  # type: ignore[assignment]
        if mode not in {"production", "preview", "retake", "diagnostic"}:
            raise ValueError(f"Unsupported execution mode: {mode!r}")
        if mode == "preview":
            return replace(self, mode=mode, upscale=False, auto_retake=False)
        if mode == "diagnostic":
            return replace(self, mode=mode, upscale=False, live_preview=False, run_visual_qa=False, auto_retake=False)
        if mode == "retake":
            return replace(self, mode=mode, auto_retake=False, max_auto_retries=0)
        return replace(self, mode=mode)

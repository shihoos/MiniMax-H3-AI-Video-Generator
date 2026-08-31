from __future__ import annotations

from typing import Any, Callable


class VisualFeedbackEngine:
    """Convert rendered-frame observations into structured continuity evidence."""

    def __init__(self, observer, vision_analyzer: Callable[[str], dict[str, Any]] | None = None):
        self.observer = observer
        self.vision_analyzer = vision_analyzer

    def analyze(self, video_path, frame_path, expected_state: dict[str, Any] | None = None) -> dict[str, Any]:
        observed = self.observer.observe_video_tail(video_path, frame_path)
        feedback: dict[str, Any] = {
            "observed_state": observed,
            "expected_state": dict(expected_state or {}),
            "deterministic_observation": True,
            "vision_escalated": False,
        }
        if self.vision_analyzer is not None:
            try:
                visual = self.vision_analyzer(str(frame_path))
                if isinstance(visual, dict):
                    feedback["vision_state"] = visual
                    feedback["vision_escalated"] = True
            except Exception as exc:
                feedback["vision_warning"] = str(exc)
        return feedback

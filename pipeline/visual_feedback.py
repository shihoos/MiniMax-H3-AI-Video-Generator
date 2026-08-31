from __future__ import annotations

from typing import Any

from pipeline.quality_gate import ProductionQualityGate
from pipeline.vlm_analyzer import VLMAnalyzer
from planner.config import H3_QA_ENABLED, H3_VLM_VISUAL_QA


class VisualFeedbackEngine:
    """Turn deterministic frame observations and optional VLM evidence into QA data."""

    def __init__(self, observer, vision_analyzer: VLMAnalyzer | None = None):
        self.observer = observer
        self.vision_analyzer = vision_analyzer or VLMAnalyzer()
        self.quality_gate = ProductionQualityGate()

    def analyze(
        self,
        video_path,
        frame_path,
        expected_state: dict[str, Any] | None = None,
        review_frames: list[Any] | None = None,
    ) -> dict[str, Any]:
        observed = self.observer.observe_video_tail(video_path, frame_path)
        feedback: dict[str, Any] = {
            "observed_state": observed,
            "expected_state": dict(expected_state or {}),
            "deterministic_observation": True,
            "vision_escalated": False,
        }
        if H3_QA_ENABLED and H3_VLM_VISUAL_QA and self.vision_analyzer.available:
            try:
                candidates = [frame_path] + list(review_frames or [])
                unique = []
                seen = set()
                for candidate in candidates:
                    key = str(candidate)
                    if key not in seen:
                        unique.append(candidate)
                        seen.add(key)
                visual = self.vision_analyzer.score_frames(
                    unique[:3],
                    expected_state or {},
                )
                feedback["vision_state"] = visual
                feedback["vision_escalated"] = True
            except Exception as exc:
                feedback["vision_warning"] = str(exc)
        feedback["quality_gate"] = self.quality_gate.evaluate(feedback, technical_ok=True)
        return feedback

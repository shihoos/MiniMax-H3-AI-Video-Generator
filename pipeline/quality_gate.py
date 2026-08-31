from __future__ import annotations

from typing import Any


class ProductionQualityGate:
    """Combine technical, semantic and continuity evidence without owning identity."""

    def __init__(self, *, accept_score: float = 90.0, review_score: float = 75.0):
        self.accept_score = float(accept_score)
        self.review_score = float(review_score)

    @staticmethod
    def _score(value: Any, default: float = 100.0) -> float:
        try:
            return max(0.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            return default

    def evaluate(self, feedback: dict[str, Any], *, technical_ok: bool = True) -> dict[str, Any]:
        vlm = feedback.get("vision_state") if isinstance(feedback.get("vision_state"), dict) else {}
        observed = feedback.get("observed_state") if isinstance(feedback.get("observed_state"), dict) else {}
        technical = 100.0 if technical_ok and not observed.get("observer_warning") else 70.0
        identity = self._score(vlm.get("identity_score"), 100.0)
        continuity = self._score(vlm.get("continuity_score"), 100.0)
        prompt = self._score(vlm.get("prompt_compliance_score"), 100.0)
        overall = round((technical * 0.25) + (identity * 0.30) + (continuity * 0.25) + (prompt * 0.20), 2)
        status = "accept" if overall >= self.accept_score else "review" if overall >= self.review_score else "retake"
        findings = list(vlm.get("findings", []) or []) if isinstance(vlm.get("findings"), list) else []
        action = str(vlm.get("recommended_action", "") or "").strip().lower()
        if action == "retake" and status == "accept":
            status = "review"
        return {
            "evidence_level": "semantic_vlm" if vlm else "technical_only",
            "status": status,
            "overall_score": overall,
            "technical_score": round(technical, 2),
            "identity_score": round(identity, 2),
            "continuity_score": round(continuity, 2),
            "prompt_compliance_score": round(prompt, 2),
            "findings": findings,
            "recommended_action": "retake" if status == "retake" else "review" if status == "review" else "accept",
        }

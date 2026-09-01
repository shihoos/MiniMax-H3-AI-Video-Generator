from __future__ import annotations

from typing import Any


class ProductionQualityGate:
    """Deterministic decision layer combining technical and semantic evidence."""

    def __init__(self, *, accept_score: float = 90.0, review_score: float = 75.0, semantic_required: bool = False):
        self.accept_score = float(accept_score)
        self.review_score = float(review_score)
        self.semantic_required = bool(semantic_required)

    @staticmethod
    def _score(value: Any, default: float | None = None) -> float | None:
        try:
            return max(0.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _reason_class(findings: list[str], action: str) -> str:
        text = " ".join(findings).lower()
        if any(k in text for k in ("identity", "face", "character", "wardrobe", "appearance")):
            return "identity"
        if any(k in text for k in ("continuity", "boundary", "previous", "next", "state")):
            return "continuity"
        if any(k in text for k in ("audio", "dialogue", "sound")):
            return "audio"
        if any(k in text for k in ("technical", "codec", "frame", "resolution", "duration", "corrupt")):
            return "technical"
        if action == "retake":
            return "semantic"
        return "quality"

    @staticmethod
    def _severity(vlm: dict[str, Any], findings: list[str]) -> str:
        raw = str(vlm.get("severity", "") or "").strip().lower()
        if raw in {"critical", "high", "medium", "low", "none"}:
            return raw
        text = " ".join(findings).lower()
        if any(k in text for k in ("critical", "severe", "missing character", "wrong character", "identity failure")):
            return "critical"
        if any(k in text for k in ("major", "strong mismatch", "continuity failure", "wrong wardrobe")):
            return "high"
        if findings:
            return "medium"
        return "none"

    def evaluate(self, feedback: dict[str, Any], *, technical_ok: bool = True) -> dict[str, Any]:
        feedback = dict(feedback or {})
        vlm = feedback.get("vision_state") if isinstance(feedback.get("vision_state"), dict) else {}
        observed = feedback.get("observed_state") if isinstance(feedback.get("observed_state"), dict) else {}
        semantic_available = bool(vlm)
        technical = 100.0 if technical_ok and not observed.get("observer_warning") else 70.0
        findings = list(vlm.get("findings", []) or []) if isinstance(vlm.get("findings"), list) else []
        requested_action = str(vlm.get("recommended_action", "") or "").strip().lower()
        severity = self._severity(vlm, findings)
        reason_class = self._reason_class(findings, requested_action)

        if self.semantic_required and not semantic_available:
            return {"evidence_level": "technical_only", "status": "review", "overall_score": round(technical * 0.5, 2), "technical_score": round(technical, 2), "identity_score": None, "continuity_score": None, "prompt_compliance_score": None, "semantic_score": None, "findings": findings + ["Semantic VLM QA is required but unavailable."], "recommended_action": "review", "decision_reason": "vlm_unavailable", "retake_class": "semantic", "severity": "high"}

        identity = self._score(vlm.get("identity_score"), None if self.semantic_required else 100.0)
        continuity = self._score(vlm.get("continuity_score"), None if self.semantic_required else 100.0)
        prompt = self._score(vlm.get("prompt_compliance_score"), None if self.semantic_required else 100.0)
        semantic_values = [v for v in (identity, continuity, prompt) if v is not None]
        semantic_score = (sum(semantic_values) / len(semantic_values)) if semantic_values else None
        overall = round((technical * 0.25) + (float(identity or 0.0) * 0.30) + (float(continuity or 0.0) * 0.25) + (float(prompt or 0.0) * 0.20), 2)
        status = "accept" if overall >= self.accept_score else "review" if overall >= self.review_score else "retake"

        # Reason/severity-aware overrides. Identity and continuity are production-critical.
        if identity is not None and identity < 80:
            status, reason_class = "retake", "identity"
        elif continuity is not None and continuity < 75:
            status, reason_class = "retake", "continuity"
        elif requested_action == "retake" and severity in {"critical", "high"}:
            status = "retake"
        elif severity == "critical":
            status = "retake"
        elif severity in {"high", "medium"} and status == "accept":
            status = "review"

        if not technical_ok:
            status, reason_class = "retake", "technical"
        if observed.get("observer_warning") and status == "accept":
            status = "review"

        return {"evidence_level": "semantic_vlm" if semantic_available else "technical_only", "status": status, "overall_score": overall, "technical_score": round(technical, 2), "identity_score": None if identity is None else round(identity, 2), "continuity_score": None if continuity is None else round(continuity, 2), "prompt_compliance_score": None if prompt is None else round(prompt, 2), "semantic_score": None if semantic_score is None else round(semantic_score, 2), "findings": findings, "recommended_action": status, "decision_reason": requested_action or reason_class, "retake_class": reason_class, "severity": severity}

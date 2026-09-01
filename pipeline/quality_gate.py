from __future__ import annotations

from typing import Any


class ProductionQualityGate:
    """Deterministic quality/retake decision layer.

    Semantic observations are evidence; this class owns the production decision.
    """

    RETAKE_THRESHOLDS = {"identity": 70.0, "continuity": 65.0, "technical": 80.0}
    SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

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
    def _classify_finding(text: str) -> str:
        value = str(text or "").lower()
        if any(k in value for k in ("identity", "face", "character", "wardrobe", "appearance")):
            return "identity"
        if any(k in value for k in ("continuity", "boundary", "previous", "next", "state")):
            return "continuity"
        if any(k in value for k in ("audio", "dialogue", "sound")):
            return "audio"
        if any(k in value for k in ("technical", "decode", "codec", "frame", "resolution", "corrupt")):
            return "technical"
        return "semantic"

    @classmethod
    def _severity(cls, value: Any, finding: str) -> str:
        explicit = str(value or "").strip().lower()
        if explicit in cls.SEVERITY_RANK:
            return explicit
        text = str(finding or "").lower()
        if any(k in text for k in ("critical", "missing character", "wrong person", "completely different")):
            return "critical"
        if any(k in text for k in ("severe", "major", "identity drift", "continuity break")):
            return "high"
        if any(k in text for k in ("mismatch", "inconsistent", "drift", "missing")):
            return "medium"
        return "low"

    def evaluate(self, feedback: dict[str, Any], *, technical_ok: bool = True) -> dict[str, Any]:
        feedback = dict(feedback or {})
        vlm = feedback.get("vision_state") if isinstance(feedback.get("vision_state"), dict) else {}
        observed = feedback.get("observed_state") if isinstance(feedback.get("observed_state"), dict) else {}
        semantic_available = bool(vlm)
        technical_score = 100.0 if technical_ok and not observed.get("observer_warning") else 70.0

        findings = list(vlm.get("findings", []) or []) if isinstance(vlm.get("findings"), list) else []
        requested_action = str(vlm.get("recommended_action", "") or "").strip().lower()

        if self.semantic_required and not semantic_available:
            return {
                "evidence_level": "technical_only",
                "status": "review",
                "overall_score": round(technical_score * 0.5, 2),
                "technical_score": round(technical_score, 2),
                "identity_score": None,
                "continuity_score": None,
                "prompt_compliance_score": None,
                "semantic_score": None,
                "findings": findings + ["Semantic VLM QA is required but unavailable."],
                "recommended_action": "review",
                "decision_reason": "vlm_unavailable",
                "retake_class": "semantic",
                "retake_severity": "high",
            }

        identity = self._score(vlm.get("identity_score"), None if self.semantic_required else 100.0)
        continuity = self._score(vlm.get("continuity_score"), None if self.semantic_required else 100.0)
        prompt = self._score(vlm.get("prompt_compliance_score"), None if self.semantic_required else 100.0)
        semantic_values = [v for v in (identity, continuity, prompt) if v is not None]
        semantic_score = sum(semantic_values) / len(semantic_values) if semantic_values else None
        overall = round(
            technical_score * 0.25 + float(identity or 0.0) * 0.30 +
            float(continuity or 0.0) * 0.25 + float(prompt or 0.0) * 0.20, 2
        )

        classes = []
        severities = []
        for finding in findings:
            classes.append(self._classify_finding(finding))
            severities.append(self._severity(None, finding))

        reason_class = max(classes, key=lambda c: {"identity": 4, "continuity": 3, "technical": 2, "audio": 2, "semantic": 1}.get(c, 0)) if classes else "none"
        max_severity = max(severities, key=lambda s: self.SEVERITY_RANK[s]) if severities else "none"

        status = "accept" if overall >= self.accept_score else "review" if overall >= self.review_score else "retake"
        decision_reason = requested_action or reason_class or "score_threshold"

        # Hard safety rules override aggregate score.
        if not technical_ok or technical_score < self.RETAKE_THRESHOLDS["technical"]:
            status, reason_class, max_severity = "retake", "technical", "high"
            decision_reason = "technical_failure"
        elif identity is not None and identity < self.RETAKE_THRESHOLDS["identity"]:
            status, reason_class, max_severity = "retake", "identity", "high"
            decision_reason = "identity_failure"
        elif continuity is not None and continuity < self.RETAKE_THRESHOLDS["continuity"]:
            status, reason_class, max_severity = "retake", "continuity", "high"
            decision_reason = "continuity_failure"
        elif max_severity == "critical":
            status = "retake"
            decision_reason = "critical_semantic_failure"
        elif max_severity == "high" and reason_class in {"identity", "continuity", "technical"}:
            status = "retake"
            decision_reason = f"{reason_class}_high_severity"
        elif requested_action == "retake" and status == "accept":
            status = "review"
            decision_reason = "vlm_requested_retake_but_score_is_accept"

        return {
            "evidence_level": "semantic_vlm" if semantic_available else "technical_only",
            "status": status,
            "overall_score": overall,
            "technical_score": round(technical_score, 2),
            "identity_score": None if identity is None else round(identity, 2),
            "continuity_score": None if continuity is None else round(continuity, 2),
            "prompt_compliance_score": None if prompt is None else round(prompt, 2),
            "semantic_score": None if semantic_score is None else round(semantic_score, 2),
            "findings": findings,
            "finding_classes": classes,
            "retake_severity": max_severity,
            "recommended_action": "retake" if status == "retake" else "review" if status == "review" else "accept",
            "decision_reason": decision_reason,
            "retake_class": reason_class if reason_class != "none" else "semantic",
        }

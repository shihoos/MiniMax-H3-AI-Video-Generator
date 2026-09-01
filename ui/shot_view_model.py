from __future__ import annotations

from typing import Any
from pathlib import Path


def shot_choices(plan: dict[str, Any]) -> list[str]:
    return [str(s.get("shot_id", "")).strip() for s in (plan.get("shots", []) or []) if isinstance(s, dict) and str(s.get("shot_id", "")).strip()]


def render_shot_card(plan: dict[str, Any], shot_id: str | None) -> str:
    wanted = str(shot_id or "").strip()
    shot = next((s for s in (plan.get("shots", []) or []) if isinstance(s, dict) and str(s.get("shot_id", "")).strip() == wanted), None)
    if not shot:
        return "### Select a shot\nChoose a shot from the timeline to inspect its prompt, references, continuity, QA and retake state."
    q = shot.get("quality_gate", {}) or {}
    vf = shot.get("visual_feedback", {}) or {}
    critique = shot.get("director_critique", {}) or plan.get("director_critique", {}) or {}
    findings = list(q.get("findings", []) or [])
    return (
        f"## {wanted}\n"
        f"**Scene:** {shot.get('scene_id','')} · **Duration:** {shot.get('duration_seconds',0)}s · **Continuity:** {shot.get('continuity_mode','chained')}\n\n"
        f"### Prompt\n{shot.get('h3_prompt') or shot.get('visual_prompt') or 'No prompt available.'}\n\n"
        f"### References\n{', '.join(str(x) for x in (shot.get('reference_bindings', []) or [])) or 'No references.'}\n\n"
        f"### Continuity\n{json.dumps(shot.get('continuity_start_state', {}) or {}, ensure_ascii=False)}\n\n"
        f"### Critic\n{json.dumps(critique, ensure_ascii=False) if critique else 'No critic findings.'}\n\n"
        f"### VLM / QA\n**Status:** {q.get('status','unknown')} · **Score:** {q.get('overall_score','—')} · **Evidence:** {q.get('evidence_level','unknown')}\n\n"
        f"**Findings:** {'; '.join(findings) if findings else 'None.'}\n\n"
        f"**VLM warning:** {vf.get('vision_warning','None.')}\n"
    )


def shot_preview_path(plan: dict[str, Any], shot_id: str | None) -> str | None:
    wanted = str(shot_id or "").strip()
    shot = next((s for s in (plan.get("shots", []) or []) if isinstance(s, dict) and str(s.get("shot_id", "")).strip() == wanted), None)
    if not shot:
        return None
    candidates = [shot.get("output"), shot.get("final_video"), shot.get("retake_execution", {}).get("output") if isinstance(shot.get("retake_execution"), dict) else None]
    for raw in candidates:
        if raw:
            path = Path(str(raw))
            if path.is_file():
                return str(path.resolve())
    return None

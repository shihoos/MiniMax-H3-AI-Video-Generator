from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.context_ir import H3ContextIRCompiler
from pipeline.quality_gate import ProductionQualityGate
from pipeline.timeline import MAX_SEGMENT_SECONDS, MIN_SEGMENT_SECONDS, ProductionTimeline
from pipeline.vlm_analyzer import VLMAnalyzer


def main() -> None:
    plan = {
        "production_id": "validation_enhancements",
        "story": "A scientist crosses a dark archive.",
        "shots": [
            {"shot_id": "shot_001", "scene_id": "scene_001", "order": 1, "duration_seconds": 5.2, "continuity_mode": "scene_reset"},
            {"shot_id": "shot_002", "scene_id": "scene_001", "order": 2, "duration_seconds": 4.5, "continuity_mode": "chained"},
            {"shot_id": "shot_003", "scene_id": "scene_002", "order": 3, "duration_seconds": 4.2, "continuity_mode": "hard_cut"},
        ],
    }
    ProductionTimeline(plan).build()
    ProductionTimeline.validate(plan)
    assert all(
        MIN_SEGMENT_SECONDS <= float(row[4]) <= MAX_SEGMENT_SECONDS
        for row in ProductionTimeline(plan).table()
    )
    assert plan["shots"][0]["is_scene_boundary"] is True
    ctx = H3ContextIRCompiler().compile(plan, plan["shots"][0])
    assert ctx["version"] == 1
    assert ctx["shot"]["shot_id"] == "shot_001"
    gate = ProductionQualityGate().evaluate({"observed_state": {"sha256": "x"}}, technical_ok=True)
    assert gate["status"] == "accept"
    assert gate["evidence_level"] == "technical_only"
    assert VLMAnalyzer().available is True
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "Pillow==12.3.0" in requirements
    assert "websocket-client==1.9.1" in requirements
    runtime = (ROOT / "configs" / "runtime_versions.yaml").read_text(encoding="utf-8")
    assert "cuda_wheel: cu130" in runtime
    assert "cuda_index: https://abetlen.github.io/llama-cpp-python/whl/cu130" in runtime
    assert "director_critic: True" in runtime
    assert "timeline_version: 1" in runtime
    print("Enhancement validation PASSED.")


if __name__ == "__main__":
    main()

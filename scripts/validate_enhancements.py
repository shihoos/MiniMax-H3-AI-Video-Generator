from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.context_ir import H3ContextIRCompiler
from execution.h3_workflow_builder import H3WorkflowBuilder
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
    plan["shots"][0]["h3_prompt"] = "A scientist crosses a dark archive."
    ctx = H3ContextIRCompiler().compile(plan, plan["shots"][0])
    assert ctx["version"] == 1
    assert ctx["shot"]["shot_id"] == "shot_001"
    gate = ProductionQualityGate().evaluate({"observed_state": {"sha256": "x"}}, technical_ok=True)
    assert gate["status"] == "accept"
    assert gate["evidence_level"] == "technical_only"
    vlm = VLMAnalyzer()
    assert vlm.enabled is True
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "Pillow==12.3.0" in requirements
    assert "websocket-client==1.9.1" in requirements
    runtime = (ROOT / "configs" / "runtime_versions.yaml").read_text(encoding="utf-8")
    assert "cuda_wheel: cu130" in runtime
    assert "cuda_index: https://abetlen.github.io/llama-cpp-python/whl/cu130" in runtime
    assert "director_critic: true" in runtime
    assert "timeline_version: 1" in runtime
    # Context-IR must be executable by the H3 builder, not merely stored on the shot.
    class FakeClient:
        @staticmethod
        def convert_workflow(workflow):
            return workflow
    builder = H3WorkflowBuilder(ROOT, FakeClient())
    built = builder.build(
        mode="ref2v",
        prompt="this must be replaced by Context-IR",
        seed=1,
        reference_images=[],
        reference_videos=[],
        reference_audio=[],
        width=1344,
        height=768,
        duration_seconds=5.2,
        ref_image_size="match",
        context_ir=ctx,
    )
    prompt_nodes = [node for node in built.get("nodes", []) if node.get("type") == "PrimitiveStringMultiline" and "prompt" in str(node.get("title", "")).lower()]
    assert prompt_nodes
    widgets = prompt_nodes[0].get("widgets_values", [])
    assert widgets and str(widgets[0]) == ctx["h3_prompt"]
    assert H3ContextIRCompiler.prompt(ctx) == ctx["h3_prompt"]
    print("PASS: context_ir builder consumption")
    print("Enhancement validation PASSED.")


if __name__ == "__main__":
    main()

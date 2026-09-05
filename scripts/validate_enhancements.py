from __future__ import annotations

import json
import re
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


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"Expected {exc_type.__name__} to be raised.")


def main() -> None:
    plan = {
        "production_id": "validation_enhancements",
        "story": "A scientist crosses a dark archive.",
        "characters": [
            {
                "name": "Alice",
                "role": "protagonist",
                "description": "adult scientist",
                "appearance": {"hair": "black hair"},
                "clothing": {"wardrobe": "dark field jacket"},
                "distinctive_features": ["small silver pendant"],
                "continuity_rules": ["preserve facial identity"],
            },
            {
                "name": "Bob",
                "role": "archivist",
                "description": "older archivist",
                "appearance": {"hair": "grey hair"},
                "clothing": {"wardrobe": "brown cardigan"},
            },
        ],
        "scenes": [
            {"scene_id": "scene_001", "location": "archive", "mood": "tense", "atmosphere": "quiet"}
        ],
        "shots": [
            {"shot_id": "shot_001", "scene_id": "scene_001", "order": 1, "duration_seconds": 5.2, "continuity_mode": "scene_reset", "characters": ["Alice"], "location": "archive", "time_of_day": "night", "action": "Alice crosses the archive."},
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

    # ------------------------------------------------------------------
    # Baseline Context-IR contract.
    # ------------------------------------------------------------------
    baseline_shot = plan["shots"][0]
    baseline_shot["h3_prompt"] = "A scientist crosses a dark archive."
    ctx = H3ContextIRCompiler().compile(plan, baseline_shot)
    assert ctx["version"] == 2
    expected_sections = [
        "subject_definitions:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ]
    prompt_text = ctx["h3_prompt"]
    assert all(section in prompt_text for section in expected_sections)
    positions = [prompt_text.index(section) for section in expected_sections]
    assert positions == sorted(positions)
    assert ctx["shot"]["shot_id"] == "shot_001"

    # ------------------------------------------------------------------
    # Semantic reference regression: image + video + voice reference.
    # ------------------------------------------------------------------
    reference_shot = {
        "shot_id": "shot_ref_001",
        "scene_id": "scene_001",
        "duration_seconds": 5.2,
        "characters": ["Alice", "Bob"],
        "location": "archive",
        "time_of_day": "night",
        "action": "Alice enters the archive while Bob watches from the desk.",
        "camera_shot": "medium wide shot",
        "camera_movement": "slow dolly forward",
        "lens_and_depth_of_field": "50mm shallow depth of field",
        "composition_notes": "Alice centered in the midground, Bob in the right background.",
        "reference_images": ["/refs/alice.png", "/refs/archive_storyboard.png"],
        "reference_videos": ["/refs/alice_motion.mp4"],
        "reference_audio_paths": ["/refs/alice_voice.wav"],
        "reference_audio_by_character": {"Alice": ["/refs/alice_voice.wav"]},
        "reference_video_by_character": {"Alice": ["/refs/alice_motion.mp4"]},
        "reference_roles": [
            {
                "path": "/refs/alice.png",
                "role": "character_identity",
                "character_name": "Alice",
                "character_id": "char_alice",
                "description": "Canonical visual identity for Alice: face, hair, body structure and stable identity.",
                "priority": 80,
                "media_type": "picture",
            },
            {
                "path": "/refs/archive_storyboard.png",
                "role": "storyboard",
                "description": "Concrete storyboard frame for composition, blocking and environment.",
                "priority": 70,
                "media_type": "picture",
            },
        ],
        "dialogue_events": [
            {"speaker_id": "char_alice", "speaker_name": "Alice", "speaker": "Alice", "text": "We are late.", "start_seconds": 0.5, "end_seconds": 1.2},
            {"speaker_id": "char_bob", "speaker_name": "Bob", "speaker": "Bob", "text": "Then move.", "start_seconds": 1.4, "end_seconds": 1.9},
            {"speaker_id": "char_alice", "speaker_name": "Alice", "speaker": "Alice", "text": "I am moving.", "start_seconds": 2.0, "end_seconds": 2.8},
        ],
        "continuity_start_state": {"location": "archive", "lighting": "cool moonlight", "state_description": "quiet tense entry"},
        "overall_soundscape": "Soft room tone, paper movement and restrained footsteps.",
        "non_diegetic_music": "A low restrained pulse enters under the dialogue.",
        "continuity_mode": "chained",
    }
    reference_ctx = H3ContextIRCompiler().compile(plan, reference_shot)
    H3ContextIRCompiler.validate(reference_ctx)
    refs = {item["label"]: item for item in reference_ctx["references"]}
    assert refs["<Picture 1>"]["relationship"] == "fully_preserved"
    assert refs["<Picture 2>"]["relationship"] == "partially_preserved"
    assert refs["<Video 1>"]["relationship"] == "weak_reference"
    assert refs["<Audio 1>"]["relationship"] == "reference"
    assert "<Picture 1> =" not in reference_ctx["h3_prompt"]
    assert "<Picture 2> =" not in reference_ctx["h3_prompt"]
    assert "/refs/alice.png" not in reference_ctx["h3_prompt"]
    assert "(S1)" in reference_ctx["sections"]["detailed_description"]
    assert "(S2)" in reference_ctx["sections"]["detailed_description"]
    assert reference_ctx["sections"]["detailed_description"].count("(S1)") >= 2
    assert "<d>[English] We are late.</d>" in reference_ctx["sections"]["detailed_description"]
    assert "<d>[English] Then move.</d>" in reference_ctx["sections"]["detailed_description"]
    print("PASS: Context-IR six-section structure")
    print("PASS: Context-IR canonical reference/entity semantics")
    print("PASS: Context-IR stable speaker mapping and language markers")

    # Audio cannot be the only reference modality.
    audio_only = dict(reference_ctx)
    audio_only["references"] = [dict(refs["<Audio 1>"])]
    audio_only["h3_prompt"] = "\n\n".join(
        [f"{name}:\n{reference_ctx['sections'][name]}" for name in H3ContextIRCompiler.REQUIRED_SECTIONS]
    ).replace("<Picture 1>", "<Audio 1>")
    audio_only["sections"] = dict(reference_ctx["sections"])
    _assert_raises(ValueError, lambda: H3ContextIRCompiler.validate(audio_only))
    print("PASS: Context-IR audio-only guard")

    # ------------------------------------------------------------------
    # Existing H3 workflow must consume Context-IR and keep the official API bridge.
    # ------------------------------------------------------------------
    class FakeClient:
        @staticmethod
        def convert_workflow(workflow):
            return workflow

    builder = H3WorkflowBuilder(ROOT, FakeClient())
    built = builder.build(
        mode="ref2va",
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
    prompt_nodes = [
        node for node in built.get("nodes", [])
        if node.get("type") == "PrimitiveStringMultiline"
        and "prompt" in str(node.get("title", "")).lower()
    ]
    assert prompt_nodes
    widgets = prompt_nodes[0].get("widgets_values", [])
    assert widgets and str(widgets[0]) == ctx["h3_prompt"]
    assert H3ContextIRCompiler.prompt(ctx) == ctx["h3_prompt"]
    ctx_nodes = [node for node in built.get("nodes", []) if node.get("type") == "MiniMaxH3ContextIR"]
    assert len(ctx_nodes) == 1
    ctx_node = ctx_nodes[0]
    ref_node = next(node for node in built.get("nodes", []) if node.get("type") == "MiniMaxH3ReferenceToVideo")
    ref_prompt_slot = next(i for i, item in enumerate(ref_node.get("inputs", [])) if item.get("name") == "prompt")
    ctx_prompt_link = [
        row for row in built.get("links", [])
        if isinstance(row, list)
        and len(row) >= 6
        and str(row[5]).upper() == "STRING"
        and int(row[1]) == int(ctx_node["id"])
        and int(row[3]) == int(ref_node["id"])
        and int(row[4]) == ref_prompt_slot
    ]
    assert len(ctx_prompt_link) == 1
    props = ctx_node.get("properties", {})
    assert props.get("requires_official_api") is True
    assert props.get("official_api_base_env") == "MINIMAX_API_BASE"
    assert props.get("official_api_token_env") == "MINIMAX_API_TOKEN"
    assert props.get("reference_order") == {"images": [], "videos": [], "audios": []}
    print("PASS: existing official Context-IR API bridge remains wired")

    # Standard Ref2VA scheduler is deliberately tuned; Turbo stays simple.
    standard_scheduler = next(node for node in built["nodes"] if node.get("type") == "BasicScheduler")
    assert standard_scheduler.get("widgets_values", [None])[0] == "beta"
    with open(ROOT / "workflows" / "generation" / "H3_Turbo_Ref2VA_Production.json", encoding="utf-8") as handle:
        turbo_workflow = json.load(handle)
    turbo_scheduler = next(node for node in turbo_workflow["nodes"] if node.get("type") == "BasicScheduler")
    assert turbo_scheduler.get("widgets_values", [None])[0] == "simple"
    print("PASS: Ref2VA sampler scheduler tuning (standard=beta, Turbo=simple)")

    # ------------------------------------------------------------------
    # Runtime/config contract.
    # ------------------------------------------------------------------
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "Pillow==12.3.0" in requirements
    assert "websocket-client==1.9.1" in requirements
    runtime = (ROOT / "configs" / "runtime_versions.yaml").read_text(encoding="utf-8")
    assert "cuda_wheel: cu130" in runtime
    assert "cuda_index: https://abetlen.github.io/llama-cpp-python/whl/cu130" in runtime
    assert "director_critic: true" in runtime
    assert "timeline_version: 1" in runtime
    assert "ref2va_scheduler: beta" in runtime
    assert "turbo_scheduler: simple" in runtime
    print("PASS: Kaggle/runtime configuration contract")

    import yaml
    runtime_features = yaml.safe_load(runtime)["features"]
    assert runtime_features["context_ir_official_api"] is True
    assert runtime_features["context_ir_official_preferred"] is True
    assert runtime_features["context_ir_official_required"] is True
    assert int(runtime_features["context_ir_official_timeout_seconds"]) == 180
    assert int(runtime_features["context_ir_official_poll_interval_seconds"]) == 5
    assert int(runtime_features["context_ir_official_max_polls"]) == 36

    bootstrap = (ROOT / "kaggle" / "bootstrap.py").read_text(encoding="utf-8")
    assert 'CREATE_PATH = "/v2/h3_context_ir"' in bootstrap
    assert 'QUERY_PATH = "/v2/query/video_generation/{task_id}"' in bootstrap
    assert 'UPLOAD_PATH = "/v1/files/upload"' in bootstrap
    live = (ROOT / "kaggle" / "verify_live_runtime.py").read_text(encoding="utf-8")
    assert '"MiniMaxH3ContextIR"' in live

    production_policy = __import__("execution.execution_policy", fromlist=["ExecutionPolicy"]).ExecutionPolicy.from_runtime(mode="production", gpu_id=None)
    assert production_policy.require_context_ir is True
    assert production_policy.auto_retake is True
    assert production_policy.max_auto_retries >= 1
    assert production_policy.vram_profile is not None
    print("PASS: execution policy runtime alignment")

    gate = ProductionQualityGate().evaluate({"observed_state": {"sha256": "x"}}, technical_ok=True)
    assert gate["status"] == "accept"
    assert gate["evidence_level"] == "technical_only"
    vlm = VLMAnalyzer()
    assert vlm.enabled is True

    unavailable_gate = ProductionQualityGate(semantic_required=True).evaluate(
        {"observed_state": {"sha256": "x"}, "vision_warning": "VLM unavailable"},
        technical_ok=True,
    )
    assert unavailable_gate["status"] == "review"
    assert unavailable_gate["evidence_level"] == "technical_only"
    assert unavailable_gate["identity_score"] is None
    print("PASS: quality/VLM degradation policy")

    ui_source = (ROOT / "ui" / "storyboard_gradio.py").read_text(encoding="utf-8")
    assert "outputs=[result_status, final_video, session_plan_path]" in ui_source
    print("PASS: Gradio refresh callback arity")

    print("Enhancement validation PASSED.")


if __name__ == "__main__":
    main()

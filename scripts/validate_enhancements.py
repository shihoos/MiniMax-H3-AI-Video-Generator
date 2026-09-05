from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.execution_policy import ExecutionPolicy
from execution.h3_workflow_builder import H3WorkflowBuilder
from pipeline.context_ir import H3ContextIRCompiler
from pipeline.quality_gate import ProductionQualityGate
from pipeline.timeline import MAX_SEGMENT_SECONDS, MIN_SEGMENT_SECONDS, ProductionTimeline
from pipeline.vlm_analyzer import VLMAnalyzer


def _base_plan() -> dict:
    return {
        "production_id": "validation_enhancements",
        "story": "A scientist crosses a dark archive.",
        "language": "English",
        "characters": [
            {
                "name": "Alex",
                "character_id": "char_001",
                "role": "protagonist",
                "description": "adult scientist",
                "appearance": {"hair": "short black hair"},
                "clothing": {"upper": "dark coat"},
                "continuity_rules": ["preserve identity"],
            },
            {
                "name": "Mina",
                "character_id": "char_002",
                "role": "archivist",
                "description": "adult archivist",
                "appearance": {"hair": "brown hair"},
                "clothing": {"upper": "gray cardigan"},
                "continuity_rules": ["preserve identity"],
            },
        ],
        "scenes": [
            {
                "scene_id": "scene_001",
                "location": "archive",
                "mood": "tense",
                "atmosphere": "quiet rows of shelves",
                "environment_details": ["dust motes", "old lamps"],
                "key_props": ["wooden desk"],
            }
        ],
        "shots": [
            {
                "shot_id": "shot_001",
                "scene_id": "scene_001",
                "order": 1,
                "duration_seconds": 5.2,
                "continuity_mode": "scene_reset",
            }
        ],
    }


def _reference_shot() -> dict:
    return {
        "shot_id": "shot_001",
        "scene_id": "scene_001",
        "duration_seconds": 5.2,
        "characters": ["Alex", "Mina"],
        "location": "archive",
        "time_of_day": "night",
        "action": "Alex enters the archive while Mina watches from the desk.",
        "camera_shot": "medium wide shot",
        "camera_movement": "slow dolly forward",
        "lens_and_depth_of_field": "50mm shallow depth of field",
        "composition_notes": "Alex foreground left, Mina background right.",
        "lighting": "cool practical lamps with soft shadows",
        "color_temperature": "cool",
        "mood": "tense",
        "reference_images": ["alex.png", "mina.png", "board.png"],
        "reference_videos": ["alex_motion.mp4"],
        "reference_audio_paths": ["alex_voice.wav"],
        "reference_roles": [
            {
                "path": "alex.png",
                "media_type": "picture",
                "role": "character_identity",
                "character_name": "Alex",
                "character_id": "char_001",
                "description": "Canonical visual identity reference for Alex; preserve face, hair, body structure and stable identity only.",
                "relationship": "fully_preserved",
                "priority": 80,
            },
            {
                "path": "mina.png",
                "media_type": "picture",
                "role": "character_identity",
                "character_name": "Mina",
                "character_id": "char_002",
                "description": "Canonical visual identity reference for Mina; preserve stable identity only.",
                "relationship": "fully_preserved",
                "priority": 80,
            },
            {
                "path": "board.png",
                "media_type": "picture",
                "role": "storyboard",
                "description": "Unified storyboard for sequencing, composition and blocking; not the canonical character identity source.",
                "relationship": "partially_preserved",
                "priority": 90,
            },
        ],
        "reference_video_by_character": {"Alex": ["alex_motion.mp4"]},
        "reference_audio_by_character": {"Alex": ["alex_voice.wav"]},
        "continuity_start_state": {
            "location": "archive",
            "lighting": "cool practical lamps",
            "environment": "same archive interior",
            "state_description": "Alex enters while Mina remains at the desk.",
        },
        "dialogue_events": [
            {"speaker_id": "S1", "speaker_name": "Alex", "text": "I found the ledger.", "start_seconds": 0.8, "end_seconds": 2.1, "language": "English"},
            {"speaker_id": "S2", "speaker_name": "Mina", "text": "Then bring it here.", "start_seconds": 2.4, "end_seconds": 3.5, "language": "English"},
            {"speaker_id": "S1", "speaker_name": "Alex", "text": "Right away.", "start_seconds": 3.8, "end_seconds": 4.4, "language": "English"},
        ],
        "overall_soundscape": "Quiet room tone, soft footsteps, paper movement and distant lamp hum.",
        "non_diegetic_music": "Low restrained suspense bed for the audience.",
    }


def main() -> None:
    plan = _base_plan()
    ProductionTimeline(plan).build()
    ProductionTimeline.validate(plan)
    assert all(MIN_SEGMENT_SECONDS <= float(row[4]) <= MAX_SEGMENT_SECONDS for row in ProductionTimeline(plan).table())

    # Basic no-reference shot: no invented Subject labels and no fake reference registry.
    plain_shot = dict(plan["shots"][0])
    plain_shot.update({"characters": [], "action": "A dark archive corridor sits empty."})
    plain_ctx = H3ContextIRCompiler().compile(plan, plain_shot)
    H3ContextIRCompiler.validate(plain_ctx)
    assert not plain_ctx["references"]
    assert "<Subject 1>" not in plain_ctx["h3_prompt"]

    # Full multimodal reference contract.
    shot = _reference_shot()
    ctx = H3ContextIRCompiler().compile(plan, shot)
    H3ContextIRCompiler.validate(ctx)
    prompt = ctx["h3_prompt"]
    assert tuple(ctx["sections"]) == H3ContextIRCompiler.REQUIRED_SECTIONS
    assert prompt.index("subject_definitions:") < prompt.index("summary:") < prompt.index("retention_analysis:") < prompt.index("detailed_description:") < prompt.index("overall_soundscape:") < prompt.index("non_diegetic_music:")

    refs = {item["label"]: item for item in ctx["references"]}
    assert refs["<Picture 1>"]["character_name"] == "Alex"
    assert refs["<Picture 2>"]["character_name"] == "Mina"
    assert refs["<Picture 3>"]["role"] == "storyboard"
    assert refs["<Picture 3>"]["relationship"] == "partially_preserved"
    assert refs["<Video 1>"]["relationship"] == "weak_reference"
    assert refs["<Audio 1>"]["relationship"] == "reference"
    assert "/" not in ctx["sections"]["subject_definitions"].split("<Picture 1>")[-1].split("\n")[0]  # no filesystem paths
    assert "<Picture 1> is the character_identity reference for Alex" in ctx["sections"]["subject_definitions"]
    assert "It defines <Subject 1>." in ctx["sections"]["subject_definitions"]
    assert "It defines <Subject 2>." in ctx["sections"]["subject_definitions"]
    assert "(S1)" in ctx["sections"]["detailed_description"]
    assert "(S2)" in ctx["sections"]["detailed_description"]
    assert "[English]" in ctx["sections"]["detailed_description"]
    assert ctx["sections"]["detailed_description"].count("(S1)") == 2
    assert ctx["sections"]["detailed_description"].count("(S2)") == 1

    # Audio-only Ref2VA is forbidden by the current official input contract.
    try:
        H3ContextIRCompiler().compile(
            plan,
            {
                "shot_id": "audio_only",
                "scene_id": "scene_001",
                "duration_seconds": 5.0,
                "reference_audio_paths": ["voice.wav"],
            },
        )
    except ValueError as exc:
        assert "visual reference" in str(exc).lower()
    else:
        raise AssertionError("Audio-only Ref2VA reference was not rejected.")

    # Builder still consumes the local Context-IR exactly as before; the official
    # Context-IR node remains in the existing workflow graph.
    class FakeClient:
        @staticmethod
        def convert_workflow(workflow):
            return workflow

    builder = H3WorkflowBuilder(ROOT, FakeClient())
    built = builder.build(
        mode="ref2va",
        prompt="ignored because Context-IR is authoritative",
        seed=1,
        reference_images=shot["reference_images"],
        reference_videos=shot["reference_videos"],
        reference_audio=shot["reference_audio_paths"],
        width=1344,
        height=768,
        duration_seconds=5.2,
        ref_image_size="match",
        context_ir=ctx,
    )
    ctx_nodes = [node for node in built.get("nodes", []) if node.get("type") == "MiniMaxH3ContextIR"]
    assert len(ctx_nodes) == 1
    ctx_node = ctx_nodes[0]
    assert ctx_node.get("properties", {}).get("requires_official_api") is True
    assert ctx_node.get("properties", {}).get("official_api_base_env") == "MINIMAX_API_BASE"
    assert ctx_node.get("properties", {}).get("official_api_token_env") == "MINIMAX_API_TOKEN"
    values = ctx_node.get("widgets_values", [])
    assert values[3] == "\n".join(shot["reference_images"])
    assert values[4] == "\n".join(shot["reference_videos"])
    assert values[5] == "\n".join(shot["reference_audio_paths"])

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "Pillow==12.3.0" in requirements
    runtime = (ROOT / "configs" / "runtime_versions.yaml").read_text(encoding="utf-8")
    assert "context_ir_official_api: true" in runtime
    assert "context_ir_official_required: true" in runtime
    bootstrap = (ROOT / "kaggle" / "bootstrap.py").read_text(encoding="utf-8")
    assert 'CREATE_PATH = "/v2/h3_context_ir"' in bootstrap
    assert 'QUERY_PATH = "/v2/query/video_generation/{task_id}"' in bootstrap

    policy = ExecutionPolicy.from_runtime(mode="production", gpu_id=None)
    assert policy.require_context_ir is True
    gate = ProductionQualityGate().evaluate({"observed_state": {"sha256": "x"}}, technical_ok=True)
    assert gate["status"] == "accept"
    assert VLMAnalyzer().enabled is True

    print("PASS: canonical Ref2VA Context-IR semantics")
    print("PASS: multimedia reference registry alignment")
    print("PASS: stable speaker IDs and dialogue language markers")
    print("PASS: official Ref2VA media constraints")
    print("PASS: existing workflow + official Context-IR API bridge unchanged")
    print("Enhancement validation PASSED.")


if __name__ == "__main__":
    main()

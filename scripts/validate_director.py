from pathlib import Path
import json
import os
import sys
import tempfile
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _planner():
    from planner.production_planner import ProductionPlanner
    return ProductionPlanner(ROOT)


def test_deterministic_character_regressions():
    planner = _planner()
    for story, expected in planner.character_detection_regression_cases():
        actual = set(planner.detect_character_descriptors(story))
        _assert(actual == expected, f"detector regression mismatch: {story!r}: {actual} != {expected}")

    single = planner.detect_character_descriptors("Eli entered the vault and Eli looked back.")
    _assert("Eli" in single, f"one-word deterministic character was lost: {single}")


def test_semantic_empty_fallback():
    planner = _planner()
    story = "Elias Kade entered the vault and Lin Mei followed him. Elias looked at Lin."

    calls = []
    def extractor(*_args):
        calls.append("extract")
        return {"candidates": []}

    def adjudicator(*_args):
        calls.append("adjudicate")
        return {"candidates": []}

    names = [c.name for c in planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=adjudicator,
    )]
    _assert(names == ["Elias Kade", "Lin Mei"], f"empty semantic responses must fall back deterministically: {names}")
    _assert(calls == ["extract", "adjudicate"], f"empty extraction must trigger adjudication: {calls}")


def test_semantic_partial_positive_is_adjudicated():
    planner = _planner()
    story = "Elias Kade entered the vault and Lin Mei followed him. Elias looked at Lin."
    calls = []

    def extractor(*_args):
        calls.append("extract")
        return {"candidates": [{
            "name": "Elias Kade",
            "entity_type": "PERSON",
            "is_character": True,
            "aliases": [],
            "identity_type": "named_character",
        }]}

    def adjudicator(*_args):
        calls.append("adjudicate")
        return {"candidates": [
            {
                "name": "Elias Kade",
                "entity_type": "PERSON",
                "is_character": True,
                "aliases": [],
                "identity_type": "named_character",
            },
            {
                "name": "Lin Mei",
                "entity_type": "PERSON",
                "is_character": True,
                "aliases": [],
                "identity_type": "named_character",
            },
        ]}

    names = [c.name for c in planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=adjudicator,
    )]
    _assert(names == ["Elias Kade", "Lin Mei"], f"partial extractor result lost a deterministic character: {names}")
    _assert(calls == ["extract", "adjudicate"], f"partial extractor result must be adjudicated: {calls}")


def test_semantic_negative_does_not_destroy_strong_deterministic_roster():
    planner = _planner()
    story = "Elias Kade entered the vault and Lin Mei followed him. Elias looked at Lin."

    def extractor(*_args):
        return {"candidates": [{
            "name": "Elias Kade",
            "entity_type": "PERSON",
            "is_character": False,
            "aliases": [],
            "identity_type": "named_character",
        }]}

    def empty_adjudicator(*_args):
        return {"candidates": []}

    names = [c.name for c in planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=empty_adjudicator,
    )]
    _assert("Elias Kade" in names and "Lin Mei" in names, f"strong deterministic identities were erased: {names}")


def test_explicit_adjudicated_negative_is_respected_when_other_characters_remain():
    planner = _planner()
    story = "Elias Kade entered the vault and Lin Mei followed him."

    def extractor(*_args):
        return {"candidates": [{
            "name": "Elias Kade",
            "entity_type": "PERSON",
            "is_character": False,
            "aliases": [],
            "identity_type": "named_character",
        }]}

    def adjudicator(*_args):
        return {"candidates": [
            {
                "name": "Elias Kade",
                "entity_type": "PERSON",
                "is_character": False,
                "aliases": [],
                "identity_type": "named_character",
            },
            {
                "name": "Lin Mei",
                "entity_type": "PERSON",
                "is_character": True,
                "aliases": [],
                "identity_type": "named_character",
            },
        ]}

    names = [c.name for c in planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=adjudicator,
    )]
    _assert(names == ["Lin Mei"], f"complete adjudicator negative was not respected: {names}")


def test_sanitizer_identity_contract():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    payload = [{
        "name": "Eli's father",
        "identity_type": "relational_character",
        "relationship_to": "Eli",
        "relationship": "father",
        "semantic_aliases": ["man"],
        "identity_profile": {
            "identity_type": "relational_character",
            "relationship_to": "Eli",
            "relationship": "father",
            "semantic_aliases": ["man"],
        },
    }]
    result = director._sanitize_characters(payload)
    _assert(len(result) == 1, f"sanitizer unexpectedly removed valid relational character: {result}")
    char = result[0]
    _assert(char["identity_type"] == char["identity_profile"]["identity_type"], "identity_type mismatch")
    _assert(char["relationship_to"] == char["identity_profile"]["relationship_to"], "relationship_to mismatch")
    _assert(char["relationship"] == char["identity_profile"]["relationship"], "relationship mismatch")
    _assert(char["semantic_aliases"] == char["identity_profile"]["semantic_aliases"], "semantic_aliases mismatch")


def test_qwen_cache_generation_contract():
    # Static behavioral contract: the cache key must change when material
    # generation configuration changes.
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    director._model_path = "/model/a"
    director._cache_namespace = "test"
    base = dict(
        call_name="characters",
        system_prompt="system",
        user_prompt="story",
        response_schema={"type": "object"},
        json_mode=True,
        disable_thinking=True,
        temperature=0.1,
        top_p=0.9,
        max_tokens=256,
    )
    first = director._cache_key(**base)
    second = director._cache_key(**{**base, "temperature": 0.2})
    third = director._cache_key(**{**base, "max_tokens": 512})
    _assert(first != second, "temperature must participate in Qwen cache key")
    _assert(first != third, "max_tokens must participate in Qwen cache key")


def test_disabled_director_path():
    from planner.qwen_director import QwenDirector

    previous = os.environ.get("H3_DIRECTOR_ENABLED")
    os.environ["H3_DIRECTOR_ENABLED"] = "0"
    try:
        director = QwenDirector(ROOT)
        plan = {
            "story": "Eli enters a station.",
            "characters": [{"name": "Eli"}],
            "scenes": [{
                "scene_id": "scene_001",
                "characters": ["Eli"],
                "description": "An abandoned station.",
                "lighting": "cool natural light",
                "color_temperature": "cool",
                "mood": "tense",
                "location": "abandoned station",
            }],
            "shots": [{
                "shot_id": "scene_001_shot_001",
                "scene_id": "scene_001",
                "characters": ["Eli"],
                "action": "Eli enters the station.",
                "duration_seconds": 5.0,
            }],
        }
        result = director.generate(
            mode="preserve_user_story",
            user_input=plan["story"],
            base_plan=plan,
        )
        _assert(result["enabled"] is False, "disabled Director must remain disabled")
        _assert(result["plan"]["shots"], "disabled Director must produce deterministic shots")
        shot = result["plan"]["shots"][0]
        for field in ("camera_shot", "camera_movement", "lens_and_depth_of_field", "composition_notes", "lighting", "color_temperature", "mood", "visual_prompt"):
            _assert(str(shot.get(field, "")).strip(), f"disabled Director path left required shot field empty: {field}")
    finally:
        if previous is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = previous


def test_context_ir_capture_root():
    from pipeline.context_ir import H3ContextIRCompiler
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ctx"
        plan = {
            "production_id": "p1",
            "context_ir_capture_root": str(root),
            "story": "Eli enters the station.",
        }
        shot = {
            "shot_id": "shot_001",
            "scene_id": "scene_001",
            "location": "station",
            "duration_seconds": 5.0,
            "camera_shot": "medium",
            "camera_movement": "static",
            "lens_and_depth_of_field": "deep focus",
            "composition_notes": "centered",
            "lighting": "soft light",
            "mood": "tense",
            "action": "Eli enters.",
            "visual_prompt": "Eli enters the station.",
            "characters": ["Eli"],
            "dialogue_events": [],
            "reference_images": [],
            "reference_videos": [],
            "reference_audio_paths": [],
            "reference_roles": [],
        }
        result = H3ContextIRCompiler().compile(plan, shot)
        path = Path(result["official_context_ir"]["capture_path"])
        _assert(path.parent == root.resolve(), f"Context-IR ignored requested capture root: {path}")


def test_checkpoint_digest_excludes_runtime_outputs():
    from pipeline.production_checkpoint import ProductionCheckpoint
    base = {"story": "Eli enters.", "shots": [{"shot_id": "s1", "action": "enter"}]}
    digest = ProductionCheckpoint.plan_digest(base)
    mutated = dict(base)
    mutated.update({
        "final_video": "/tmp/final.mp4",
        "shot_outputs": ["/tmp/s1.mp4"],
        "job_result": {"final_video": "/tmp/final.mp4"},
        "completed_shot_ids": ["s1"],
    })
    _assert(digest == ProductionCheckpoint.plan_digest(mutated), "output/runtime metadata changed plan digest")


def test_job_state_clears_stale_completion():
    from pipeline.production_plan_store import ProductionPlanStore
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plan.json"
        ProductionPlanStore.atomic_save(path, {
            "job_id": "old",
            "job_status": "completed",
            "final_video": "/old/final.mp4",
            "job_result": {"final_video": "/old/final.mp4"},
        })
        updated = ProductionPlanStore.set_job_state_unlocked(
            path,
            job_id="new",
            status="queued",
            result={},
        )
        _assert("final_video" not in updated and "job_result" not in updated, "stale completion state survived queue transition")


def test_source_contracts():
    from pathlib import Path
    orchestrator = (ROOT / "pipeline/production_orchestrator.py").read_text(encoding="utf-8")
    runner = (ROOT / "execution/production_runner.py").read_text(encoding="utf-8")
    _assert("require_context_ir_results=False" in orchestrator, "planning manifest must be non-strict before rendering")
    _assert('"h3_context_ir"' in runner and '"h3_effective_prompt"' in runner, "completed shot records must persist official Context-IR provenance")
    _assert("production_plan[\"final_video\"] = str(final_video)" in runner, "final video must be bound before final manifest")
    _assert(runner.find('require_context_ir_results=True') > runner.find('production_plan["final_video"]'), "strict final manifest must be written after assembly")


def main():
    tests = [
        test_deterministic_character_regressions,
        test_semantic_empty_fallback,
        test_semantic_partial_positive_is_adjudicated,
        test_semantic_negative_does_not_destroy_strong_deterministic_roster,
        test_explicit_adjudicated_negative_is_respected_when_other_characters_remain,
        test_sanitizer_identity_contract,
        test_qwen_cache_generation_contract,
        test_disabled_director_path,
        test_context_ir_capture_root,
        test_checkpoint_digest_excludes_runtime_outputs,
        test_job_state_clears_stale_completion,
        test_source_contracts,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("Director and integration validation PASSED.")


if __name__ == "__main__":
    main()

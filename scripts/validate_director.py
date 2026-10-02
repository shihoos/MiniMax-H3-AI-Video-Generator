from pathlib import Path
import os
import sys
import tempfile

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


def test_story_prompt_restores_successful_compact_narrative_contract():
    source = Path(ROOT, "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert("The user provides a premise." in source, "AI story prompt must use the proven premise framing")
    _assert("Write a complete cinematic short-film story with a clear beginning" in source, "AI story prompt must use the proven compact narrative contract")
    _assert("SUBVERT THE OBVIOUS" in source, "AI story prompt must require one earned reversal")
    _assert("Include at least one short line of spoken dialogue by a named" in source, "AI story prompt must require named-character dialogue")
    _assert("End with a complete aftermath paragraph showing what happened" in source, "AI story prompt must require an explicit aftermath")
    _assert("Aim for 400-650 words" in source, "AI/Expand story target must remain 400-650 words")
    _assert("One protagonist whose goal is stated in the first two sentences" in source, "AI story opening contract is missing")
    _assert("Prioritize finishing the full narrative" in source, "story user prompt must prioritize completion over padding")
    _assert("The cast is a narrative decision, not a production rule." in source, "cast selection must remain flexible")
    _assert("Do not force a minimum or maximum cast" in source, "cast size must remain flexible")
    _assert("active causal or emotional counterpart" in source, "recurring supporting characters must be causally active")
    _assert("first major action must be caused by that goal" in source, "opening must be goal-driven")
    _assert("meaningful resistance, opposition, or consequence" in source, "protagonist must face meaningful early resistance")
    _assert("When the premise naturally" in source and "supports another recurring character" in source, "early interaction should prefer a supported recurring character")
    _assert("change the protagonist's choice, belief, goal, or relationship" in source, "supporting-character interaction must change the protagonist meaningfully")
    _assert("Before the midpoint, create at least one meaningful interaction" in source, "story must contain an early interactive beat")
    _assert("PERSONAL CAUSALITY" in source, "story should support protagonist-linked central conflict when the premise allows")
    _assert("prior choice, relationship, mistake, promise, desire, or responsibility" in source, "personal causality mechanism is missing")
    _assert("Do not manufacture backstory or a" in source and "personal connection when the premise does not support one" in source, "personal causality must remain premise-grounded")
    _assert("PAYOFF DETAIL" in source, "story must plant and pay off a concrete detail")
    _assert("one dominant causal reversal" in source, "story must avoid stacked unrelated twists")
    _assert("final image or behavior that echoes an earlier detail" in source, "resolution should echo an earlier planted detail")
    _assert("Use one protagonist or" not in source, "single-protagonist bias must remain absent")
    _assert("when the story contains a speaking character" not in source, "dialogue requirement must not be weakened")
    _assert("final character must end" not in source.lower(), "obsolete final-character wording remains")


def test_expand_source_fallback_removed():
    source = Path(ROOT, "planner", "qwen_director.py").read_text(encoding="utf-8")
    _assert("expand_story_source_fallback" not in source, "Expand Story must not silently fall back to preserve-story mode")
    _assert("Expand Story generation failed validation after the controlled retry" in source, "Expand retry must fail closed after its bounded retry")


def test_story_wiring_redundancy_removed():
    compiler = Path(ROOT, "planner", "cinematic_compiler.py").read_text(encoding="utf-8")
    _assert('if not shot.get("speaking_characters")' in compiler, "absent/empty speaking_characters must compile to []")
    orchestrator = Path(ROOT, "pipeline", "production_orchestrator.py").read_text(encoding="utf-8")
    _assert(orchestrator.count("        self._rebind_shots(") == 1, "production plan should have one canonical _rebind_shots() call")
    _assert(orchestrator.count("ProductionTimeline(plan).build()") == 1, "production plan should have one canonical timeline build")


def test_logged_story_rosters_are_not_poisoned_by_prose_surfaces():
    from planner.production_planner import ProductionPlanner
    planner = ProductionPlanner(ROOT)
    ai_story = (
        "Elara Voss clawed through the snowdrifts. Dr. Kess's final transmission had warned her. Dr. Kess entered the vault. "
        "Below, the air was thick with ozone. A voice announced, \"Unauthorized access detected. Initiation of Protocol Epsilon.\" "
        "Elara sprinted to the terminal."
    )
    ai_names = planner.detect_character_descriptors(ai_story)
    _assert(set(ai_names) == {"Elara Voss", "Kess"}, f"AI Story prose poisoned the deterministic roster: {ai_names}")

    expand_story = (
        "Eli reached the vault. His sister's voice warned him. His father had vanished years ago. "
        "A photograph of a child with his sister's eyes lay inside the crate. Eli sprinted to the exit."
    )
    expand_names = planner.detect_character_descriptors(expand_story)
    _assert(expand_names == ["Eli"], f"Expand Story visual/relational prose poisoned the deterministic roster: {expand_names}")
    relations = planner._extract_relational_character_hints(expand_story, expand_names)
    _assert(
        [item["name"] for item in relations] == ["Eli's sister", "Eli's father"],
        f"Expand Story relational identities were not grounded correctly: {relations}",
    )


def test_semantic_named_surface_safety_boundary():
    from planner.production_planner import ProductionPlanner
    _assert(ProductionPlanner._semantic_named_surface_is_safe("Elara Voss"), "valid proper name was rejected")
    for value in ("child with his sister's eyes", "father was", "her instead"):
        _assert(
            not ProductionPlanner._semantic_named_surface_is_safe(value),
            f"non-name semantic surface was accepted: {value!r}",
        )
    protocol_context = "Protocol Epsilon was a self-destruct sequence. Unauthorized access detected."
    _assert(
        "Protocol Epsilon" not in ProductionPlanner(".").detect_character_descriptors(protocol_context),
        "contextually explicit protocol entity leaked into the deterministic roster",
    )


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


def test_relational_character_survives_generic_negative_and_resolves_dialogue():
    from planner.production_planner import ProductionPlanner
    from planner.qwen_director import QwenDirector

    story = (
        "Eli entered the vault. His father had disappeared years ago. "
        "A voice said, \"You should not be here.\" Eli turned and saw an older man. "
        "The man stepped forward, revealing his father's face. "
        "\"I was trying to protect you.\""
    )
    planner = ProductionPlanner(ROOT)

    def extractor(*_args):
        return {"candidates": [{
            "name": "Eli", "entity_type": "PERSON", "is_character": True,
            "aliases": [], "identity_type": "named_character",
            "relationship_to": "", "relationship": "",
        }, {
            "name": "Eli's father", "entity_type": "CHARACTER", "is_character": False,
            "aliases": ["his father", "man"], "identity_type": "relational_character",
            "relationship_to": "Eli", "relationship": "father",
        }]}

    def adjudicator(*_args):
        return {"candidates": [{
            "name": "Eli", "entity_type": "PERSON", "is_character": True,
            "aliases": [], "identity_type": "named_character",
            "relationship_to": "", "relationship": "",
        }, {
            "name": "Eli's father", "entity_type": "CHARACTER", "is_character": False,
            "aliases": ["his father", "man"], "identity_type": "relational_character",
            "relationship_to": "Eli", "relationship": "father",
        }]}

    characters = planner.create_characters(
        story, qwen_character_extractor=extractor, qwen_character_adjudicator=adjudicator
    )
    names = [c.name for c in characters]
    _assert("Eli's father" in names and "man" not in names and "All" not in names,
            f"generic role leaked into canonical roster or relational identity was lost: {names}")

    payload = [c.to_dict() for c in characters]
    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    scenes = [{"scene_id": "scene_001", "characters": ["Eli"]}]
    shots = [{
        "shot_id": "scene_001_shot_001",
        "scene_id": "scene_001",
        "characters": ["Eli"],
        "dialogue_events": [{
            "speaker": "man",
            "text": "You should not be here.",
            "continues_from_previous_shot": False,
            "continues_to_next_shot": False,
        }],
        "speaking_characters": ["man"],
        "speech_text": "You should not be here.",
    }]
    director._normalize_dialogue_speakers(story, scenes, shots, payload)
    event = shots[0]["dialogue_events"][0]
    _assert(event["speaker"] == "Eli's father", f"generic dialogue speaker did not resolve: {event}")
    _assert("Eli's father" in shots[0]["characters"], f"shot binding missing relational speaker: {shots[0]['characters']}")
    _assert("Eli's father" in scenes[0]["characters"], f"scene binding missing relational speaker: {scenes[0]['characters']}")


def test_generic_relationship_speakers_resolve_for_uncle_and_father():
    from planner.entity_resolver import EntityResolver
    characters = [{
        "name": "Eli's uncle",
        "identity_type": "relational_character",
        "relationship_to": "Eli",
        "relationship": "uncle",
        "semantic_aliases": [],
    }]
    story = "Eli's uncle entered the room. The uncle spoke to Eli while the older man watched."
    _assert(EntityResolver.contextual_generic_alias("uncle", characters, story=story) == "Eli's uncle",
            "bare relationship speaker 'uncle' did not resolve")
    _assert(EntityResolver.contextual_generic_alias("father", [{
        "name": "Eli's father",
        "identity_type": "relational_character",
        "relationship_to": "Eli",
        "relationship": "father",
        "semantic_aliases": [],
    }], story="Eli met his father. The father spoke.") == "Eli's father",
            "bare relationship speaker 'father' did not resolve")


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



def test_reference_visual_context_setter_is_live_instance_method():
    previous = os.environ.get("H3_DIRECTOR_ENABLED")
    os.environ["H3_DIRECTOR_ENABLED"] = "0"
    try:
        from planner.qwen_director import QwenDirector

        director = QwenDirector(ROOT)
        director.set_reference_visual_context({
            "scene_001": {"image_path": "/tmp/reference.png"},
            "ignored": "not-a-dict",
        })
        _assert(
            director._reference_visual_context == {
                "scene_001": {"image_path": "/tmp/reference.png"},
            },
            "reference visual context setter did not bind as an instance method",
        )
    finally:
        if previous is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = previous


def test_sampling_for_mode_is_live_instance_method():
    previous = os.environ.get("H3_DIRECTOR_ENABLED")
    os.environ["H3_DIRECTOR_ENABLED"] = "0"
    try:
        from planner.qwen_director import AI_STORY_MODE, QwenDirector

        director = QwenDirector(ROOT)
        temperature, top_p = director._sampling_for_mode(AI_STORY_MODE)
        _assert(
            (temperature, top_p) == (0.78, 0.90),
            f"unexpected AI Story sampling values: {(temperature, top_p)}",
        )
    finally:
        if previous is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = previous


def test_story_token_budget_not_reduced():
    source = Path(ROOT, "planner", "qwen_director.py").read_text(encoding="utf-8")
    _assert(source.count('max_completion=1800') >= 2, "story generation budget must be 1800 for primary pass and retry")
    _assert(source.count('disable_thinking=True') >= 2, "story generation must keep the proven no-think mode")
    _assert('max_completion=2200' not in source, "old 2200 story budget must not regress back in")
    _assert('expand_story_text_retry' in source, "Expand Story controlled retry must remain available")

def test_named_only_roster_skips_semantic_character_calls_safely():
    from planner.production_planner import ProductionPlanner

    planner = ProductionPlanner(ROOT)
    story = (
        "Elias Kade entered the Arctic station. "
        "Dr. Lin Mei stood in the doorway and warned him about the vault."
    )
    characters = planner.create_characters(
        story,
        qwen_character_extractor=None,
        qwen_character_adjudicator=None,
    )
    names = [character.name for character in characters]
    _assert(names == ["Elias Kade", "Lin Mei"], f"bare honorific leaked into roster: {names}")

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
        path = Path(result["context_ir_provenance"]["capture_path"])
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



def test_unresolved_explicit_dialogue_fails_closed():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    director._qwen_telemetry = {"deterministic_recoveries": 0}
    story = 'Eli entered the vault. A stranger said, "Run."'
    characters = [{
        "name": "Eli",
        "identity_type": "named_character",
        "semantic_aliases": [],
    }]
    scenes = [{"scene_id": "scene_001", "characters": ["Eli"]}]
    shots = [{
        "shot_id": "scene_001_shot_001",
        "scene_id": "scene_001",
        "characters": ["Eli"],
        "dialogue_events": [{
            "speaker": "stranger",
            "text": "Run.",
        }],
    }]
    try:
        director._normalize_dialogue_speakers(story, scenes, shots, characters)
    except RuntimeError as exc:
        _assert("speaker" in str(exc).lower() and "stranger" in str(exc),
                f"unexpected unresolved-dialogue error: {exc}")
    else:
        raise AssertionError("unresolved explicit dialogue was silently accepted/dropped")


def test_canonical_dialogue_speaker_is_rebound_into_shot_and_scene():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    director._qwen_telemetry = {"deterministic_recoveries": 0}
    story = 'Eli met his father. "I was trying to protect you," his father said.'
    characters = [
        {
            "name": "Eli",
            "identity_type": "named_character",
            "semantic_aliases": [],
        },
        {
            "name": "Eli's father",
            "identity_type": "relational_character",
            "relationship_to": "Eli",
            "relationship": "father",
            "semantic_aliases": ["his father", "father", "man"],
        },
    ]
    scenes = [{"scene_id": "scene_001", "characters": ["Eli"]}]
    shots = [{
        "shot_id": "scene_001_shot_001",
        "scene_id": "scene_001",
        "characters": ["Eli"],
        "dialogue_events": [{
            "speaker": "Eli's father",
            "text": "I was trying to protect you,",
        }],
    }]
    director._normalize_dialogue_speakers(story, scenes, shots, characters)
    event = shots[0]["dialogue_events"][0]
    _assert(event["speaker"] == "Eli's father", f"speaker was not canonicalized: {event}")
    _assert("Eli's father" in shots[0]["characters"], "canonical speaker missing from shot binding")
    _assert("Eli's father" in scenes[0]["characters"], "canonical speaker missing from scene binding")


def test_downstream_production_preserves_dialogue_multiset():
    from pipeline.production_orchestrator import ProductionOrchestrator

    before = ProductionOrchestrator._snapshot_dialogue_contract({
        "shots": [{
            "shot_id": "scene_001_shot_001",
            "dialogue_events": [{
                "speaker": "Eli",
                "text": "I was trying to protect you.",
            }],
        }]
    })
    ProductionOrchestrator._assert_dialogue_contract_preserved(before, {
        "shots": [{
            "shot_id": "scene_001_shot_001",
            "dialogue_events": [{
                "speaker_name": "Eli",
                "speaker_id": "char_eli",
                "text": "I was trying to protect you.",
            }],
        }]
    })

    try:
        ProductionOrchestrator._assert_dialogue_contract_preserved(before, {
            "shots": [{
                "shot_id": "scene_001_shot_001",
                "dialogue_events": [],
            }]
        })
    except RuntimeError:
        pass
    else:
        raise AssertionError("downstream dialogue deletion was not detected")

def test_dialogue_scene_boundary_clears_stale_continuation():
    from pipeline.dialogue_timeline import DialogueTimeline

    characters = [{"name": "Eli", "character_id": "eli"}]
    plan = {
        "shots": [
            {
                "shot_id": "scene_001_shot_001",
                "scene_id": "scene_001",
                "is_scene_boundary": True,
                "duration_seconds": 4.0,
                "characters": ["Eli"],
                "dialogue_events": [
                    {
                        "speaker": "Eli",
                        "text": "A fresh scene begins here.",
                        "continues_from_previous_shot": True,
                        "continues_to_next_shot": False,
                    }
                ],
            }
        ]
    }
    DialogueTimeline(characters).apply_to_plan(plan)
    event = plan["shots"][0]["dialogue_events"][0]
    _assert(
        event["continues_from_previous_shot"] is False,
        "scene-boundary normalization did not persist the cleared continuation flag",
    )


def test_dialogue_h3_feasibility_rebalances_generated_shots():
    from copy import deepcopy
    from planner.qwen_director import QwenDirector
    from pipeline.dialogue_timeline import DialogueTimeline

    previous = os.environ.get("H3_DIRECTOR_ENABLED")
    os.environ["H3_DIRECTOR_ENABLED"] = "0"
    try:
        director = QwenDirector(ROOT)
    finally:
        if previous is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = previous

    characters = [{"name": "Eli", "character_id": "eli"}]

    def event(text, *, continues_from=False, continues_to=False):
        return {
            "speaker": "Eli",
            "text": text,
            "continues_from_previous_shot": continues_from,
            "continues_to_next_shot": continues_to,
        }

    shots = [
        {
            "shot_id": "scene_005_shot_001",
            "scene_id": "scene_005",
            "duration_seconds": 13.667,
            "characters": ["Eli"],
            "dialogue_events": [
                event("one two three four five six seven eight nine ten"),
                event("one two three four five six seven eight nine ten"),
                event("one two three four five six seven eight nine ten"),
                event("one two three four five six seven", continues_to=True),
            ],
        },
        {
            "shot_id": "scene_005_shot_002",
            "scene_id": "scene_005",
            "duration_seconds": 5.2,
            "characters": ["Eli"],
            "dialogue_events": [
                event("the sentence continues here", continues_from=True),
            ],
        },
    ]
    scenes = [{"scene_id": "scene_005", "characters": ["Eli"]}]

    source_text = [
        event_data["text"]
        for shot in shots
        for event_data in shot["dialogue_events"]
    ]

    try:
        DialogueTimeline(characters).apply_to_plan({"shots": deepcopy(shots)})
    except ValueError:
        pass
    else:
        raise AssertionError("regression fixture unexpectedly fit before Director H3 rebalancing")

    director._normalize_dialogue_h3_feasibility(scenes, shots, characters)

    final_text = [
        event_data["text"]
        for shot in shots
        for event_data in shot["dialogue_events"]
    ]
    _assert(final_text == source_text, "H3 dialogue rebalancing changed spoken text or order")

    try:
        DialogueTimeline(characters).apply_to_plan({"shots": deepcopy(shots)})
    except Exception as exc:
        raise AssertionError(f"rebalanced dialogue still violates H3 scheduling: {exc}") from exc

    _assert(
        len(shots[0]["dialogue_events"]) < 4 or len(shots[1]["dialogue_events"]) > 1,
        "H3 rebalancing did not redistribute the overfull shot",
    )


def test_dialogue_h3_feasibility_searches_the_whole_scene():
    import planner.qwen_director as qwen_director_module
    from planner.qwen_director import QwenDirector

    previous = os.environ.get("H3_DIRECTOR_ENABLED")
    os.environ["H3_DIRECTOR_ENABLED"] = "0"
    try:
        director = QwenDirector(ROOT)
    finally:
        if previous is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = previous

    characters = [{"name": "Eli", "character_id": "eli"}]

    def event(number):
        return {
            "speaker": "Eli",
            "text": f"line {number}",
            "continues_from_previous_shot": False,
            "continues_to_next_shot": False,
        }

    shots = [
        {
            "shot_id": "scene_global_shot_001",
            "scene_id": "scene_global",
            "characters": ["Eli"],
            "dialogue_events": [event(1), event(2), event(3), event(4)],
        },
        {
            "shot_id": "scene_global_shot_002",
            "scene_id": "scene_global",
            "characters": ["Eli"],
            "dialogue_events": [],
        },
        {
            "shot_id": "scene_global_shot_003",
            "scene_id": "scene_global",
            "characters": ["Eli"],
            "dialogue_events": [],
        },
    ]
    scenes = [{"scene_id": "scene_global", "characters": ["Eli"]}]

    original_fit = QwenDirector._dialogue_scene_fits_h3

    def fake_fit(scene_shots, _characters):
        counts = tuple(len(shot.get("dialogue_events", []) or []) for shot in scene_shots)
        # Force the only legal scene-global arrangement to be [2, 1, 1].
        if counts == (2, 1, 1):
            return True, ""
        return False, "synthetic H3 timing overflow"

    qwen_director_module.QwenDirector._dialogue_scene_fits_h3 = staticmethod(fake_fit)
    try:
        director._normalize_dialogue_h3_feasibility(scenes, shots, characters)
    finally:
        qwen_director_module.QwenDirector._dialogue_scene_fits_h3 = staticmethod(original_fit)

    counts = [len(shot.get("dialogue_events", []) or []) for shot in shots]
    _assert(
        counts == [2, 1, 1],
        f"scene-global feasibility search selected the wrong partition: {counts}",
    )


def test_dialogue_h3_feasibility_propagates_non_timing_errors():
    import planner.qwen_director as qwen_director_module
    from planner.qwen_director import QwenDirector

    class DefectTimeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def apply_to_plan(self, _plan):
            raise ValueError("continued dialogue must use the previous shot's speaker.")

    original = qwen_director_module.DialogueTimeline
    qwen_director_module.DialogueTimeline = DefectTimeline
    try:
        try:
            QwenDirector._dialogue_scene_fits_h3([], [])
        except ValueError as exc:
            _assert(
                "continued dialogue must use the previous shot's speaker" in str(exc),
                "non-H3 scheduler errors must propagate from feasibility probes",
            )
        else:
            raise AssertionError(
                "non-H3 scheduler ValueError was swallowed by the feasibility probe"
            )
    finally:
        qwen_director_module.DialogueTimeline = original


def test_source_contracts():
    from pathlib import Path
    orchestrator = (ROOT / "pipeline/production_orchestrator.py").read_text(encoding="utf-8")
    runner = (ROOT / "execution/production_runner.py").read_text(encoding="utf-8")
    _assert("require_context_ir_results=False" in orchestrator, "planning manifest must be non-strict before rendering")
    _assert('"h3_context_ir"' in runner and '"h3_effective_prompt"' in runner, "completed shot records must persist local Context-IR provenance")
    _assert("production_plan[\"final_video\"] = str(final_video)" in runner, "final video must be bound before final manifest")
    _assert(runner.find('require_context_ir_results=True') > runner.find('production_plan["final_video"]'), "strict final manifest must be written after assembly")


def main():
    tests = [
        test_deterministic_character_regressions,
        test_story_prompt_restores_successful_compact_narrative_contract,
        test_expand_source_fallback_removed,
        test_story_wiring_redundancy_removed,
        test_semantic_named_surface_safety_boundary,
        test_logged_story_rosters_are_not_poisoned_by_prose_surfaces,
        test_semantic_empty_fallback,
        test_semantic_partial_positive_is_adjudicated,
        test_semantic_negative_does_not_destroy_strong_deterministic_roster,
        test_explicit_adjudicated_negative_is_respected_when_other_characters_remain,
        test_relational_character_survives_generic_negative_and_resolves_dialogue,
        test_generic_relationship_speakers_resolve_for_uncle_and_father,
        test_sanitizer_identity_contract,
        test_qwen_cache_generation_contract,
        test_disabled_director_path,
        test_story_token_budget_not_reduced,
        test_reference_visual_context_setter_is_live_instance_method,
        test_sampling_for_mode_is_live_instance_method,
        test_named_only_roster_skips_semantic_character_calls_safely,
        test_context_ir_capture_root,
        test_checkpoint_digest_excludes_runtime_outputs,
        test_job_state_clears_stale_completion,
        test_unresolved_explicit_dialogue_fails_closed,
        test_canonical_dialogue_speaker_is_rebound_into_shot_and_scene,
        test_downstream_production_preserves_dialogue_multiset,
        test_dialogue_scene_boundary_clears_stale_continuation,
        test_dialogue_h3_feasibility_rebalances_generated_shots,
        test_dialogue_h3_feasibility_searches_the_whole_scene,
        test_dialogue_h3_feasibility_propagates_non_timing_errors,
        test_source_contracts,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("Director and integration validation PASSED.")


if __name__ == "__main__":
    main()

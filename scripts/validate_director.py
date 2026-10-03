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


def _has_normalized(source, phrase):
    """Check prompt contracts without depending on source line wrapping."""
    import re
    normalized_source = re.sub(r"\s+", " ", source)
    normalized_phrase = re.sub(r"\s+", " ", phrase)
    return normalized_phrase in normalized_source


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
    ai = source[source.index("You are the narrative writer for MiniMax H3"):source.index("You are the narrative expansion writer")]
    expand = source[source.index("You are the narrative expansion writer"):source.index("Preserve Story does not use a story-text pass.")]
    for label, text in (("AI story", ai), ("Expand story", expand)):
        _assert("exactly six paragraphs" in text, f"{label} prompt must produce six scene-sized paragraphs")
        _assert("420 to 560 words" in text, f"{label} prompt must state its word target")
        _assert("stable canonical name" in text, f"{label} prompt must require stable character identity")
        for beat in ("SETUP", "CATALYST", "COMPLICATION", "REVERSAL", "CHOICE", "AFTERMATH"):
            _assert(beat in text, f"{label} prompt is missing the {beat} beat")
        _assert(
            "materially affect the choice" in text or "materially affects the final choice" in text,
            f"{label} prompt must link the planted detail to the choice",
        )
        _assert(
            "personal stake" in text and "matter to the final choice" in text,
            f"{label} prompt must connect personal stake to the choice",
        )
        _assert("never quoted as speech" in text, f"{label} prompt must keep machine voices out of dialogue")
        _assert("no would/will/could/might" in text, f"{label} prompt must require a completed final action")
        _assert("organism" not in text.lower(), f"{label} prompt must not prime hook imagery")
    _assert("1 to 3 short lines" in ai, "AI story prompt must keep dialogue compact")
    _assert("Preserve source character names exactly" in expand, "Expand prompt must preserve source names")
    _assert("relational character" in expand and "Do not invent unrelated people" in expand, "Expand prompt must allow only source-grounded relational additions")
    _assert("Begin with concrete physical action" in ai, "AI story prompt must prioritize immediate filmable action")
    _assert("source_dialogue" in source, "shot prompt must receive an explicit source-dialogue whitelist")
    _assert("If `is_character=true`, `entity_type` MUST be PERSON, CHARACTER, or SENTIENT" in source, "character extraction must keep identity type and entity type consistent")
    _assert("Reject pronouns, contractions, sentence fragments" in source, "character extraction must reject prose fragments as identities")
    _assert("final character must end" not in source.lower(), "obsolete final-character wording remains")
    for leaked_name in ("Eli's father", "Sara's sister", "Mira's commander", "Eli", "Lin Mei"):
        _assert(leaked_name not in source, f"prompt must not seed a hard-coded character name: {leaked_name}")


def test_expand_source_fallback_removed():
    source = Path(ROOT, "planner", "qwen_director.py").read_text(encoding="utf-8")
    _assert("expand_story_source_fallback" not in source, "Expand Story must not silently fall back to preserve-story mode")
    _assert("Expand Story generation failed validation after the controlled retry" in source, "Expand retry must fail closed after its bounded retry")


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


def test_expand_long_distance_relational_character_recovery():
    """Regression for Expand Story discourse-distance recovery (e.g. `Eli ... his father`)."""
    planner = _planner()
    story = (
        "Eli stepped cautiously through the rusted gates of the abandoned train station, "
        "the air thick with dust and the scent of decay. The flickering light from his "
        "flashlight cast long shadows on the cracked tiles, each step echoing in the "
        "cavernous silence. He had come here for a reason—whispers of a sealed vault "
        "buried beneath the station, a relic from the city's forgotten past. His fingers "
        "brushed against the cold metal of a door marked with faded engravings. "
        "The door was sealed, but not locked. A faint hum of power pulsed through the air "
        "as Eli pressed his palm against the surface. A memory surfaced—his father, "
        "standing over a similar case, his face pale and drawn."
    )

    descriptors = planner.detect_character_descriptors(story)
    _assert(
        descriptors == ["Eli"],
        f"long Expand Story fixture polluted the deterministic named roster: {descriptors}",
    )

    hints = planner._extract_relational_character_hints(story, descriptors)
    names = [item.get("name") for item in hints]
    _assert(
        names == ["Eli's father"],
        f"long-distance possessive relation was not recovered: {hints}",
    )

    # Verify the hint reaches the bounded semantic adjudication contract rather than
    # being merely detectable by the helper.
    calls = []

    def extractor(*_args):
        calls.append("extract")
        return {"candidates": [{
            "name": "Eli",
            "entity_type": "PERSON",
            "is_character": True,
            "aliases": [],
            "identity_type": "named_character",
            "relationship_to": "",
            "relationship": "",
        }]}

    def adjudicator(_story, hints_payload, _semantic):
        calls.append("adjudicate")
        _assert(
            "Eli's father" in hints_payload,
            f"long-distance relational hint was not supplied to adjudication: {hints_payload}",
        )
        return {"candidates": [
            {
                "name": "Eli",
                "entity_type": "PERSON",
                "is_character": True,
                "aliases": [],
                "identity_type": "named_character",
                "relationship_to": "",
                "relationship": "",
            },
            {
                "name": "Eli's father",
                "entity_type": "CHARACTER",
                "is_character": False,
                "aliases": ["his father", "father", "man"],
                "identity_type": "relational_character",
                "relationship_to": "Eli",
                "relationship": "father",
            },
        ]}

    characters = planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=adjudicator,
    )
    names = [character.name for character in characters]
    _assert(
        "Eli" in names and "Eli's father" in names,
        f"Expand Story relational character was lost during canonicalization: {names}",
    )
    _assert(
        calls == ["extract", "adjudicate"],
        f"long-distance relational recovery took an unexpected semantic path: {calls}",
    )


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



def test_story_token_budget_not_reduced():
    source = Path(ROOT, "planner", "qwen_director.py").read_text(encoding="utf-8")
    # Primary creative pass: thinking ON with room for plan + story.
    _assert("max_completion=3200,\n                    disable_thinking=False" in source, "primary story pass must think with a 3200-token budget")
    # Retry / repair passes: proven no-think mode, original budget.
    _assert(source.count("max_completion=1800") >= 1, "story retry budget must stay 1800")
    _assert(source.count("disable_thinking=True") >= 2, "story retry and non-creative calls must stay no-think")
    _assert("max_completion=2200" not in source, "old 2200 story budget must not regress back in")
    _assert("expand_story_text_retry" in source, "Expand Story controlled retry must remain available")


def test_quoted_dialogue_speaker_comes_from_speech_tag():
    from planner.qwen_director import QwenDirector
    cases = (
        ('"Nobody goes up," Ines said. She held the chain.', {"ines"}),
        ('Tomas turned. "Move aside," said Tomas.', {"tomas"}),
        ('Ines whispered, "It is already open."', {"ines"}),
        ('"Access granted," the terminal said.', set()),
        ('"Run," she said.', set()),
    )
    for text, expected in cases:
        segments = QwenDirector._extract_story_spoken_segments(text)
        got = set().union(*[s.get("tag_speakers", set()) for s in segments]) if segments else set()
        _assert(got == expected, f"speech-tag attribution wrong for {text!r}: {got} != {expected}")

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


def test_wrong_shot_speaker_is_corrected_by_speech_tag():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    director._qwen_telemetry = {"deterministic_recoveries": 0}
    story = 'Tomas pushed the door. "Nobody goes up," Ines said. "Then stop me," Tomas said.'
    characters = [
        {"name": "Tomas", "identity_type": "named_character", "semantic_aliases": []},
        {"name": "Ines", "identity_type": "named_character", "semantic_aliases": []},
    ]
    scenes = [{"scene_id": "scene_001", "characters": ["Tomas", "Ines"]}]
    shots = [{
        "shot_id": "scene_001_shot_001",
        "scene_id": "scene_001",
        "characters": ["Tomas", "Ines"],
        "dialogue_events": [
            {"speaker": "Tomas", "text": "Nobody goes up,"},
            {"speaker": "Tomas", "text": "Then stop me,"},
        ],
    }]
    director._normalize_dialogue_speakers(story, scenes, shots, characters)
    speakers = [e["speaker"] for e in shots[0]["dialogue_events"]]
    _assert(speakers == ["Ines", "Tomas"], f"speech tag must correct a wrong shot speaker: {speakers}")
    _assert(shots[0]["speaking_characters"] == ["Ines", "Tomas"], "speaking_characters must follow corrected speakers")


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


def test_shot_prompt_requires_same_speaker_continuation_rule():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert(
        "A continuation edge is SAME-SPEAKER ONLY" in source,
        "shot prompt must require same-speaker continuation",
    )
    _assert(
        "Never set `continues_to_next_shot=true` when the next shot begins with a different speaker." in source,
        "shot prompt must forbid cross-speaker continuation",
    )
    _assert(
        "source_dialogue` whitelist" in source,
        "shot prompt must identify the exact source-dialogue whitelist",
    )
    _assert(
        "must never invent a new line from narrative prose" in source,
        "shot prompt must forbid narrative-to-dialogue invention",
    )


def test_dialogue_continuation_requires_same_canonical_speaker():
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

    characters = [
        {"name": "Elias Kade", "character_id": "elias_kade"},
        {"name": "Lin Mei", "character_id": "lin_mei"},
    ]
    scenes = [{"scene_id": "scene_001"}]

    # A safe alias for the SAME character must preserve continuation.
    alias_same = [
        {
            "shot_id": "scene_001_shot_001",
            "scene_id": "scene_001",
            "characters": ["Elias Kade"],
            "duration_seconds": 6.0,
            "dialogue_events": [{
                "speaker": "Elias",
                "text": "We have to go now.",
                "continues_from_previous_shot": False,
                "continues_to_next_shot": True,
            }],
        },
        {
            "shot_id": "scene_001_shot_002",
            "scene_id": "scene_001",
            "characters": ["Elias Kade"],
            "duration_seconds": 6.0,
            "dialogue_events": [{
                "speaker": "Elias Kade",
                "text": "Before they find us.",
                "continues_from_previous_shot": False,
                "continues_to_next_shot": False,
            }],
        },
    ]
    director._normalize_dialogue_continuations(scenes, alias_same, characters)
    _assert(
        alias_same[0]["dialogue_events"][0]["continues_to_next_shot"] is True
        and alias_same[1]["dialogue_events"][0]["continues_from_previous_shot"] is True,
        "same-character aliases were incorrectly broken at the continuation boundary",
    )

    # A real speaker change must break continuation deterministically.
    speaker_change = [
        {
            "shot_id": "scene_001_shot_001",
            "scene_id": "scene_001",
            "characters": ["Elias Kade"],
            "duration_seconds": 6.0,
            "dialogue_events": [{
                "speaker": "Elias Kade",
                "text": "We have to go now.",
                "continues_from_previous_shot": False,
                "continues_to_next_shot": True,
            }],
        },
        {
            "shot_id": "scene_001_shot_002",
            "scene_id": "scene_001",
            "characters": ["Lin Mei"],
            "duration_seconds": 6.0,
            "dialogue_events": [{
                "speaker": "Lin Mei",
                "text": "No. We stay.",
                "continues_from_previous_shot": False,
                "continues_to_next_shot": False,
            }],
        },
    ]
    director._normalize_dialogue_continuations(scenes, speaker_change, characters)
    _assert(
        speaker_change[0]["dialogue_events"][0]["continues_to_next_shot"] is False
        and speaker_change[1]["dialogue_events"][0]["continues_from_previous_shot"] is False,
        "speaker-change boundary was incorrectly preserved as a continuation",
    )
    try:
        DialogueTimeline(characters).apply_to_plan({"shots": deepcopy(speaker_change)})
    except Exception as exc:
        raise AssertionError(
            f"speaker-change continuation repair still violates the dialogue scheduler: {exc}"
        ) from exc


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



def test_story_completion_contracts():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    sentence = "Eli crossed the abandoned station with measured steps, checked the sealed vault, and listened to the cooling machinery."
    body = " ".join([sentence] * 4)
    paragraphs = [body for _ in range(6)]
    paragraphs[0] = 'Eli entered the abandoned station and found a sealed vault. "I should have left it closed," he whispered. ' + body
    open_story = "\n\n".join(paragraphs[:-1] + [
        body + " Eli escaped after the station shook. He knew he had set something in motion that could never be undone."
    ])
    try:
        director._validate_story_completion_contract("expand_user_story", open_story)
    except RuntimeError as exc:
        _assert("unresolved future hook" in str(exc), f"wrong completion failure: {exc}")
    else:
        raise AssertionError("open-ended story ending was not rejected")

    closed_paragraphs = [body for _ in range(6)]
    closed_paragraphs[0] = 'Eli entered the abandoned station and found a sealed vault. "I should have left it closed," he whispered. ' + body
    closed_paragraphs[-1] = (
        "Eli shut the system down before dawn, sealed the journal away, and secured the station again. "
        "He finally stopped searching for the answers his father had taken to his grave. " + body
    )
    closed_story = "\n\n".join(closed_paragraphs)
    director._validate_story_completion_contract("expand_user_story", closed_story)


def test_story_normalization_preserves_scene_paragraphs():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    normalized = director._normalize_story(
        " First  paragraph.\n\n Second paragraph.\n\n\nThird   paragraph. "
    )
    _assert(
        normalized == "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        f"story normalization did not preserve paragraph topology: {normalized!r}",
    )


def test_story_topology_fallback_is_last_resort_and_prose_preserving():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    seven = "\n\n".join(
        f"Beat {index} happened with concrete action and a settled consequence."
        for index in range(1, 8)
    )
    repaired = director._coerce_story_to_six_paragraphs(seven)
    paragraphs = [part for part in repaired.split("\n\n") if part.strip()]
    _assert(len(paragraphs) == 6, f"seven-paragraph fallback did not reach six: {len(paragraphs)}")
    _assert(
        "".join(repaired.split()) == "".join(seven.split()),
        "topology fallback changed story prose tokens",
    )

    five = "\n\n".join(
        f"Beat {index} opened the door and found the same concrete evidence. The character acted."
        for index in range(1, 6)
    )
    repaired_five = director._coerce_story_to_six_paragraphs(five)
    five_paragraphs = [part for part in repaired_five.split("\n\n") if part.strip()]
    _assert(len(five_paragraphs) == 6, f"five-paragraph fallback did not reach six: {len(five_paragraphs)}")
    _assert(
        "".join(repaired_five.split()) == "".join(five.split()),
        "five-to-six topology fallback changed story prose tokens",
    )


def test_story_validation_precedes_topology_fallback():
    source = Path(ROOT, "planner", "qwen_director.py").read_text(encoding="utf-8")
    primary = source[source.index("story = self._chat_text("):source.index("except RuntimeError as first_error:")]
    _assert(
        primary.index("self._validate_story_output_contracts") < primary.index("generated_story = True"),
        "primary story flow must validate through the contract helper before completion",
    )
    _assert(
        "story = self._coerce_story_to_six_paragraphs(story)" not in primary,
        "primary story flow must not silently coerce topology before validation",
    )


def test_six_story_paragraphs_become_six_production_units():
    from planner.production_planner import ProductionPlanner

    planner = ProductionPlanner(ROOT)
    story = "\n\n".join(
        f"Paragraph {index} establishes a complete narrative beat in one physical setting."
        for index in range(1, 7)
    )
    units = planner._split_story(story)
    _assert(len(units) == 6, f"six story paragraphs became {len(units)} units")
    _assert(
        [unit.text for unit in units] == story.split("\n\n"),
        "production units changed six-paragraph story order/content",
    )


def test_four_to_six_scene_functions_are_structural():
    from planner.qwen_director_scene import QwenDirectorSceneMixin

    expected = {
        4: ["setup", "catalyst", "climax", "finale"],
        5: ["setup", "catalyst", "development", "climax", "finale"],
        6: ["setup", "catalyst", "development", "midpoint", "climax", "finale"],
    }
    for count, target in expected.items():
        scenes = [{"description": "generic prose with no structural keywords"} for _ in range(count)]
        actual = [
            scene["scene_function"]
            for scene in QwenDirectorSceneMixin._annotate_scene_functions(scenes)
        ]
        _assert(actual == target, f"{count}-scene structural function mapping drifted: {actual}")


def test_story_completion_contract_rejects_wrong_structure_and_word_budget():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    sentence = "Eli moved through the station and checked the sealed vault before dawn."
    short_body = " ".join([sentence] * 8)
    five_paragraph_story = "\n\n".join([short_body] * 5)
    try:
        director._validate_story_completion_contract("expand_user_story", five_paragraph_story)
    except RuntimeError as exc:
        _assert("exactly six paragraphs" in str(exc), f"wrong structural error: {exc}")
    else:
        raise AssertionError("five-paragraph story passed the six-paragraph contract")

    six_paragraph_short_story = "\n\n".join([short_body] * 6)
    try:
        director._validate_story_completion_contract("expand_user_story", six_paragraph_short_story)
    except RuntimeError as exc:
        _assert("420 to 560 words" in str(exc), f"wrong word-budget error: {exc}")
    else:
        raise AssertionError("under-length story passed the word-budget contract")


def test_text_generation_length_reason_is_fail_closed():
    source = Path(ROOT, "planner/qwen_director_runtime.py").read_text(encoding="utf-8")
    _assert('if finish_reason == "length":' in source, "text length finish reason is not handled")
    _assert("completion limit" in source, "length failure does not identify the completion limit")


def test_critic_roster_is_bidirectional_and_patch_consumer_remains_wired():
    director_source = Path(ROOT, "planner/qwen_director.py").read_text(encoding="utf-8")
    orchestrator_source = Path(ROOT, "pipeline/production_orchestrator.py").read_text(encoding="utf-8")
    _assert("enforce the roster in both directions" in director_source, "critic roster check is not bidirectional")
    _assert('critique.get("shot_patches", [])' in orchestrator_source, "critic patch consumer was removed despite being wired")


def test_dialogue_source_occurrence_budget():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    director._qwen_telemetry = {"deterministic_recoveries": 0}
    characters = [{"name": "Eli", "identity_type": "named_character", "semantic_aliases": []}]
    scenes = [{"scene_id": "scene_001", "characters": ["Eli"]}]

    def run(story, texts):
        shots = []
        for index, text in enumerate(texts, start=1):
            shots.append({
                "shot_id": f"scene_001_shot_{index:03d}",
                "scene_id": "scene_001",
                "characters": ["Eli"],
                "dialogue_events": [{
                    "speaker": "Eli",
                    "text": text,
                    "continues_from_previous_shot": False,
                    "continues_to_next_shot": False,
                }],
            })
        director._normalize_dialogue_speakers(story, scenes, shots, characters)
        return shots

    duplicate = run('Eli said, "Stay here."', ["Stay here.", "Stay here."])
    _assert(len(duplicate[0]["dialogue_events"]) == 1, "first source line was lost")
    _assert(not duplicate[1]["dialogue_events"], "same source line was duplicated")

    split = run('Eli said, "We need to leave before dawn."', ["We need to leave", "before dawn."])
    _assert(all(len(shot["dialogue_events"]) == 1 for shot in split), "split source dialogue was lost")

    repeated = run(
        'Eli said, "Stay here." Later he said, "Stay here."',
        ["Stay here.", "Stay here."],
    )
    _assert(all(len(shot["dialogue_events"]) == 1 for shot in repeated), "distinct repeated source lines were collapsed")


def test_source_contracts():
    from pathlib import Path
    orchestrator = (ROOT / "pipeline/production_orchestrator.py").read_text(encoding="utf-8")
    runner = (ROOT / "execution/production_runner.py").read_text(encoding="utf-8")
    _assert("require_context_ir_results=False" in orchestrator, "planning manifest must be non-strict before rendering")
    _assert('"h3_context_ir"' in runner and '"h3_effective_prompt"' in runner, "completed shot records must persist local Context-IR provenance")
    _assert("production_plan[\"final_video\"] = str(final_video)" in runner, "final video must be bound before final manifest")
    _assert(runner.find('require_context_ir_results=True') > runner.find('production_plan["final_video"]'), "strict final manifest must be written after assembly")


def test_sentence_initial_common_words_never_skip_semantic_roster():
    from planner.production_planner import ProductionPlanner

    story = (
        'Elias trudged on. Dust swirled in the light. '
        '\u201cAccess granted,\u201d the terminal said. Anika Marlowe stared at Elias and waited.'
    )
    for word in ("Dust", "Access"):
        assert not ProductionPlanner._high_confidence_deterministic_character(story, word), word
    assert ProductionPlanner._high_confidence_deterministic_character(story, "Elias")
    assert ProductionPlanner._high_confidence_deterministic_character(
        "Eli, a scientist, enters the station.", "Eli"
    )


def main():
    tests = [
        test_deterministic_character_regressions,
        test_story_prompt_restores_successful_compact_narrative_contract,
        test_expand_source_fallback_removed,
        test_semantic_named_surface_safety_boundary,
        test_logged_story_rosters_are_not_poisoned_by_prose_surfaces,
        test_semantic_empty_fallback,
        test_semantic_partial_positive_is_adjudicated,
        test_semantic_negative_does_not_destroy_strong_deterministic_roster,
        test_explicit_adjudicated_negative_is_respected_when_other_characters_remain,
        test_expand_long_distance_relational_character_recovery,
        test_relational_character_survives_generic_negative_and_resolves_dialogue,
        test_sanitizer_identity_contract,
        test_qwen_cache_generation_contract,
        test_disabled_director_path,
        test_story_token_budget_not_reduced,
        test_quoted_dialogue_speaker_comes_from_speech_tag,
        test_named_only_roster_skips_semantic_character_calls_safely,
        test_context_ir_capture_root,
        test_checkpoint_digest_excludes_runtime_outputs,
        test_job_state_clears_stale_completion,
        test_unresolved_explicit_dialogue_fails_closed,
        test_wrong_shot_speaker_is_corrected_by_speech_tag,
        test_canonical_dialogue_speaker_is_rebound_into_shot_and_scene,
        test_downstream_production_preserves_dialogue_multiset,
        test_dialogue_scene_boundary_clears_stale_continuation,
        test_shot_prompt_requires_same_speaker_continuation_rule,
        test_dialogue_continuation_requires_same_canonical_speaker,
        test_dialogue_h3_feasibility_rebalances_generated_shots,
        test_dialogue_h3_feasibility_searches_the_whole_scene,
        test_dialogue_h3_feasibility_propagates_non_timing_errors,
        test_story_completion_contracts,
        test_story_normalization_preserves_scene_paragraphs,
        test_story_topology_fallback_is_last_resort_and_prose_preserving,
        test_story_validation_precedes_topology_fallback,
        test_six_story_paragraphs_become_six_production_units,
        test_four_to_six_scene_functions_are_structural,
        test_story_completion_contract_rejects_wrong_structure_and_word_budget,
        test_text_generation_length_reason_is_fail_closed,
        test_critic_roster_is_bidirectional_and_patch_consumer_remains_wired,
        test_dialogue_source_occurrence_budget,
        test_source_contracts,
        test_sentence_initial_common_words_never_skip_semantic_roster,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("Director and integration validation PASSED.")


if __name__ == "__main__":
    main()

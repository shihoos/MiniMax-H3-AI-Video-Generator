from pathlib import Path
import os
import ast
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _planner():
    from planner.production_planner import ProductionPlanner
    return ProductionPlanner(ROOT)


# ── Schema contracts (structural) ──────────────────────────────────────────

def test_shot_schema_tracks_instance_topology():
    from planner.qwen_director import QwenDirector
    class ProbeDirector(QwenDirector):
        SHOTS_PER_SCENE = 3
    probe = ProbeDirector.__new__(ProbeDirector)
    shot_schema = probe._shot_json_schema()
    batch_schema = probe._shot_batch_json_schema()
    nested = batch_schema["properties"]["scene_shots"]["items"]["properties"]["shots"]
    _assert(shot_schema["properties"]["shots"]["minItems"] == 3, "shot schema ignored instance SHOTS_PER_SCENE")
    _assert(nested["minItems"] == 3 and nested["maxItems"] == 3, "batch schema ignored instance SHOTS_PER_SCENE")
    _assert(shot_schema["properties"]["shots"]["items"]["properties"]["location"].get("minLength") == 1, "location schema must require a value")
    _assert(shot_schema["properties"]["shots"]["items"]["properties"]["camera_shot"].get("enum"),
            "camera_shot must be constrained to the canonical framing vocabulary")
    _assert(shot_schema["properties"]["shots"]["items"]["properties"]["camera_movement"].get("enum"),
            "camera_movement must be constrained to the canonical movement vocabulary")
    scene_entry = batch_schema["properties"]["scene_shots"]["items"]
    metadata = scene_entry["properties"].get("scene_metadata", {})
    _assert("scene_metadata" in scene_entry["required"], "Qwen batch must author canonical scene metadata")
    _assert(set(("location", "time_of_day", "weather", "lighting", "mood")) <= set(metadata.get("properties", {})),
            "scene metadata schema must include semantic setting and lighting fields")
    _assert("dramatic_purpose" in metadata.get("required", []),
            "Qwen must supply story-grounded dramatic purpose in the existing shot batch")
    from planner.qwen_director_scene import QwenDirectorSceneMixin
    annotated = QwenDirectorSceneMixin._annotate_scene_functions([
        {"description": "The witness refuses to identify the person in the photograph."},
        {"description": "The protagonist quietly returns the key after learning the cost.",
         "dramatic_purpose": "The protagonist accepts the cost of the choice."},
    ])
    _assert(all("scene_function" not in item for item in annotated),
            "deterministic scene compiler must not force dramatic function by index")
    _assert(annotated[1].get("dramatic_purpose") == "The protagonist accepts the cost of the choice.",
            "scene context preparation must preserve story-dependent dramatic purpose")
    sample_metadata = {"location": "coastal archive", "time_of_day": "blue hour", "weather": "steady rain",
                       "atmosphere": "hushed and damp", "mood": "apprehensive", "lighting": "cold window light",
                       "color_temperature": "cool", "environment_details": ["rain against glass"], "key_props": ["sealed ledger"],
                       "dramatic_purpose": "The protagonist discovers the archive has been quietly altered."}
    scene = {"scene_id": "scene_001"}
    probe._apply_scene_metadata(scene, sample_metadata)
    _assert(scene.get("time_of_day") == "blue hour" and scene.get("weather") == "steady rain",
            "Qwen-authored scene time and weather must survive scene metadata normalization")
    _assert(scene.get("dramatic_purpose") == sample_metadata["dramatic_purpose"],
            "Qwen-authored dramatic purpose must survive scene metadata normalization")
    extracted = probe._normalize_batch_scene_metadata_response({
        "scene_shots": [{"scene_id": "scene_001", "scene_metadata": sample_metadata, "shots": []}]
    })
    _assert(extracted.get("scene_001", {}).get("weather") == "steady rain",
            "scene metadata must be retained separately from shot records")
    shots = [
        {"shot_id": "scene_001_shot_001", "scene_id": "scene_001", "location": "coastal archive",
         "lighting": "cold window light", "color_temperature": "cool", "mood": "apprehensive", "characters": []},
        {"shot_id": "scene_001_shot_002", "scene_id": "scene_001", "location": "archive room",
         "lighting": "blue practical light", "color_temperature": "cool", "mood": "tense", "characters": []},
    ]
    probe._enforce_scene_coherence([scene], shots, [])
    _assert(all(item.get("time_of_day") == "blue hour" and item.get("weather") == "steady rain" for item in shots),
            "scene time/weather must propagate to every shot in the final plan")
    _assert(all(item.get("location") == "coastal archive" for item in shots),
            "scene-level canonical location must win over inconsistent shot wording")
    from schemas.shot import Shot
    compiled_shot = Shot(shot_id="s1", scene_id="scene_001", order=1, duration_seconds=5.2,
                         time_of_day="blue hour", weather="steady rain")
    _assert(compiled_shot.to_dict().get("weather") == "steady rain" and "steady rain" in compiled_shot.h3_prompt(),
            "scene time/weather must survive Shot serialization into H3's final prompt")
    empty_scene = _planner().create_scenes("A person enters a place.\n\nThey find a key.\n\nThey make a choice.\n\nThey leave.", [])
    _assert(all(not item.time_of_day and not item.weather and not item.mood and not item.lighting for item in empty_scene),
            "ProductionPlanner must leave semantic scene fields to Qwen rather than keyword-filling them")


def test_compact_shot_batch_schema_omits_derived_and_empty_fields():
    from planner.qwen_director import QwenDirector
    probe = QwenDirector.__new__(QwenDirector)
    batch = probe._shot_batch_json_schema(scene_count=2)
    shot = batch["properties"]["scene_shots"]["items"]["properties"]["shots"]["items"]
    properties = shot["properties"]
    for field in (
        "shot_id", "scene_id", "location", "lighting", "color_temperature",
        "speaking_characters", "speech_text", "is_scene_boundary",
        "character_spatial_bboxes", "character_spatial_regions",
        "character_spatial_bboxes_start", "character_spatial_bboxes_end",
        "character_spatial_regions_start", "character_spatial_regions_end",
        "continuity_start_state", "continuity_end_state",
    ):
        _assert(field not in properties and field not in shot["required"],
                f"compact batch schema should derive {field} after generation")
    for field in ("action", "visual_prompt", "dialogue_events"):
        _assert(field in properties and field in shot["required"],
                f"compact batch schema must retain creative field {field}")
    dialogue_schema = properties["dialogue_events"]
    dialogue_item = dialogue_schema["items"]
    _assert(set(dialogue_item["properties"]) == {"source_dialogue_id"},
            "batch dialogue should carry a source ID, not duplicate speaker/text payloads")
    constrained = probe._shot_batch_json_schema(
        scene_count=2,
        dialogue_ids=["scene_001_d001", "scene_002_d001"],
    )
    ref_property = constrained["properties"]["scene_shots"]["items"]["properties"]["shots"]["items"]["properties"]["dialogue_events"]["items"]["properties"]["source_dialogue_id"]
    _assert(ref_property.get("enum") == ["scene_001_d001", "scene_002_d001"],
            "batch schema must constrain source-dialogue references to the supplied whitelist")
    full = probe._shot_json_schema()["properties"]["shots"]["items"]["properties"]
    _assert("location" in full and "character_spatial_bboxes" in full,
            "compact optimization must not remove fields from the canonical standalone shot schema")


def test_compact_shot_prompt_example_is_valid_json():
    import json
    import re
    from planner.qwen_director_prompts import SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT
    match = re.search(
        r"COMPACT OUTPUT CONTRACT:.*?\n\n(\{.*?\})\n\nThere must be exactly",
        SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT,
        flags=re.DOTALL,
    )
    _assert(match is not None, "compact shot prompt example block is missing")
    sample = json.loads(match.group(1))
    _assert(isinstance(sample.get("scene_shots"), list) and sample["scene_shots"],
            "compact shot example must show the scene_shots envelope")
    scene = sample["scene_shots"][0]
    _assert("scene_id" in scene and all(key in scene.get("scene_metadata", {}) for key in (
        "location", "lighting", "color_temperature", "time_of_day", "weather", "dramatic_purpose")),
        "compact example must show all required parent-scene metadata")
    _assert(len(scene.get("shots", [])) == 3,
            "compact prompt example should match the configured three-shot scene topology")
    shot = scene["shots"][0]
    for field in ("shot_id", "scene_id", "location", "lighting", "color_temperature",
                  "speaking_characters", "speech_text", "is_scene_boundary",
                  "character_spatial_bboxes", "character_spatial_regions"):
        _assert(field not in shot, f"compact prompt example must not teach output of derived field {field}")
    _assert("dialogue_events" not in shot and "continuity_start_state" not in shot and "continuity_end_state" not in shot,
            "shot continuity is derived downstream; the prompt example must not spend tokens on empty state maps")
    _assert("dialogue_assignments" in scene and scene["dialogue_assignments"] == {},
            "compact example must show the parent-level dialogue assignment object")


def test_character_schema_does_not_silently_cap_qwen_cast_at_32():
    from planner.qwen_director import QwenDirector
    schema = QwenDirector._character_extraction_json_schema()
    candidates = schema["properties"]["candidates"]
    _assert("maxItems" not in candidates, "character extraction schema must not impose a cast-size cap")
    _assert(schema["properties"]["quote_classifications"].get("maxItems") == 256,
            "quote classification must support long dialogue-heavy sources without a 128-line cap")
    dialogue_item = schema["properties"]["quote_classifications"]["items"]
    _assert(set(dialogue_item["required"]) == {"quote_id", "classification", "speaker"},
            "every quote must have an explicit ID, category, and speaker field")


def test_character_semantic_schema_includes_spoken_dialogue():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    schema = QwenDirectorPromptMixin._character_extraction_json_schema()
    _assert("quote_classifications" in schema["properties"], "semantic character extraction schema lacks quote_classifications")
    _assert("quote_classifications" in schema["required"], "every source quote must be classified in the single semantic extraction contract")
    _assert(schema["properties"]["quote_classifications"]["maxItems"] >= 128, "quote classification schema is artificially capped")


# ── Qwen as sole cast authority (no silent add-back) ──────────────────────

def test_qwen_can_choose_any_cast_size():
    planner = _planner()
    story = "Elias Kade entered the vault. Lin Mei followed. Mara waited by the console. Tomas locked the door."
    def extractor(_story, _hints):
        return {"candidates": [
            {"name": "Elias Kade", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Lin Mei", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Mara", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Tomas", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
        ]}
    names = [c.name for c in planner.create_characters(story, qwen_character_extractor=extractor)]
    _assert(names == ["Elias Kade", "Lin Mei", "Mara", "Tomas"], f"Qwen-selected cast was constrained: {names}")


def test_qwen_negative_is_authoritative():
    planner = _planner()
    story = "Elias Kade entered the vault and Lin Mei followed him."
    def extractor(_story, _hints):
        return {"candidates": [{
            "name": "Elias Kade", "entity_type": "PERSON", "is_character": False,
            "aliases": [], "identity_type": "named_character",
        }]}
    try:
        planner.create_characters(story, qwen_character_extractor=extractor)
        raise AssertionError("Qwen negative must not be overridden by deterministic detection")
    except RuntimeError as exc:
        _assert("no valid production characters" in str(exc), f"wrong fail-closed error: {exc}")


def test_deterministic_detection_never_adds_missing_qwen_character():
    planner = _planner()
    story = "Elias Kade entered the vault and Lin Mei followed him."
    def extractor(_story, _hints):
        return {"candidates": [{
            "name": "Elias Kade", "entity_type": "PERSON", "is_character": True,
            "aliases": [], "identity_type": "named_character",
        }]}
    names = [c.name for c in planner.create_characters(story, qwen_character_extractor=extractor)]
    _assert(names == ["Elias Kade"], f"deterministic detector added a character Qwen omitted: {names}")


def test_qwen_extraction_does_not_force_candidates_from_hints_plan_or_dialogue():
    import json
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    mixin = QwenDirectorPromptMixin()
    mixin._character_semantic_calls = 0
    mixin._semantic_spoken_dialogue = []
    mixin._anchor_semantic_dialogue_to_story = lambda _story, items: list(items)
    observed = {}
    story = "Elara opened the vault. The radio carried Kael Mercer's voice: \"I can walk.\" Elara helped her teammate reach the exit."
    def fake_chat_json(system_prompt, user_prompt, *args, **kwargs):
        observed["schema"] = kwargs["response_schema"]
        observed["system"] = system_prompt
        observed["payload"] = json.loads(user_prompt)
        return {
            "candidates": [
                {"name": "Elara", "entity_type": "PERSON", "is_character": True,
                 "identity_type": "named_character"},
                {"name": "Kael Mercer", "entity_type": "PERSON", "is_character": True,
                 "identity_type": "named_character"},
            ],
            "quote_classifications": [{"quote_id": 1, "classification": "remote_or_offscreen_spoken", "speaker": "Kael Mercer"}],
        }
    mixin._chat_json = fake_chat_json
    result = mixin.extract_character_entities(
        story, deterministic_candidates=["Elara", "Kael Mercer", "Station Echo-7"]
    )
    _assert(set(observed["payload"]) == {"story_evidence", "quotes", "name_surface_index", "name_surface_evidence"},
            "the extractor must receive bounded story evidence, a complete numbered quote inventory, and source-grounded recall indexes")
    _assert(observed["payload"]["quotes"] == [{"quote_id": 1, "text": "I can walk.", "context": observed["payload"]["quotes"][0]["context"]}],
            "every quote must retain its exact text and stable quote ID")
    _assert(set(observed["schema"]["properties"]) == {"candidates", "quote_classifications"},
            "Qwen should decide the cast directly and classify each numbered source quote")
    _assert("physically participate" in observed["system"] and "no cast-count quota" in observed["system"].lower() and "every supplied `quote_id`" in observed["system"].lower(),
            "extractor must request all current-action participants and classify every quote ID")
    _assert([item["name"] for item in result["candidates"]] == ["Elara", "Kael Mercer"],
            "the extractor must use Qwen's returned roster and must not programmatically add a missing character")
    _assert(len(mixin._semantic_spoken_dialogue) == 1
            and mixin._semantic_spoken_dialogue[0].get("delivery") == "remote"
            and mixin._semantic_spoken_dialogue[0].get("speaker") == "Kael Mercer",
            "explicit attributed remote dialogue must be retained as remote audio, not reclassified as in-person")


def test_mention_only_qwen_candidates_are_filtered_without_restoring_them():
    planner = _planner()
    story = (
        "Lena Voss opened the vault. A name was etched into the metal: Kovac. "
        "Kovac had vanished during the thaw. A message was from Dr. Renn. Access granted."
    )
    def extractor(_story, _hints):
        return {"candidates": [
            {"name": "Lena Voss", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Kovac", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Renn", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Access", "entity_type": "OTHER", "is_character": True, "aliases": [], "identity_type": "named_character"},
        ]}
    names = [c.name for c in planner.create_characters(story, qwen_character_extractor=extractor)]
    _assert(names == ["Lena Voss"], f"mention-only Qwen identities leaked into production: {names}")


def test_non_person_qwen_candidates_are_filtered():
    planner = _planner()
    story = "Lena Voss entered the station. Protocol Epsilon was active. Access granted."
    def extractor(_story, _hints):
        return {"candidates": [
            {"name": "Lena Voss", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Station", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Access", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Protocol Epsilon", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
        ]}
    names = [c.name for c in planner.create_characters(story, qwen_character_extractor=extractor)]
    _assert(names == ["Lena Voss"], f"non-person Qwen entities leaked into production: {names}")


def test_relational_owner_must_be_qwen_approved():
    planner = _planner()
    story = "Eli entered the station. Kovac had vanished. Eli's father was never mentioned again."
    def extractor(_story, _hints):
        return {"candidates": [
            {"name": "Eli", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Kovac", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Eli's father", "entity_type": "CHARACTER", "is_character": True, "aliases": ["his father"], "identity_type": "relational_character", "relationship_to": "Kovac", "relationship": "father"},
        ]}
    names = [c.name for c in planner.create_characters(story, qwen_character_extractor=extractor)]
    _assert(names == ["Eli"], f"relational identity attached to filtered owner: {names}")


# ── Dialogue ID/text integrity (exact source restoration) ─────────────────

def test_source_dialogue_ids_are_stable_and_scene_scoped():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    payload = [
        {"scene_id": "scene_001", "source_dialogue": [
            {"speaker": "Mara", "text": "Open the door."},
            {"speaker": "Eli", "text": "Not yet."},
        ]},
        {"scene_id": "scene_002", "source_dialogue": [
            {"speaker": "Mara", "text": "Now!"},
        ]},
    ]
    QwenDirectorPromptMixin._assign_source_dialogue_ids(payload)
    ids = [[item["dialogue_id"] for item in scene["source_dialogue"]] for scene in payload]
    _assert(ids == [["scene_001_d001", "scene_001_d002"], ["scene_002_d001"]],
            f"dialogue references must be stable and globally unique by scene: {ids}")
    _assert(payload[0]["source_dialogue"][0]["text"] == "Open the door.",
            "dialogue ID assignment must preserve original source text")


def test_shot_batch_dialogue_references_restore_exact_source_text_and_deduplicate():
    import json
    from planner.qwen_director import QwenDirector
    payload = json.dumps({"scenes": [
        {"scene_id": "scene_001", "source_dialogue": [
            {"dialogue_id": "scene_001_d001", "speaker": "Mara Venn", "text": "Open the door… now!"},
        ]},
        {"scene_id": "scene_002", "source_dialogue": [
            {"dialogue_id": "scene_002_d001", "speaker": "Eli Rhee", "text": "Wait."},
        ]},
    ]}, ensure_ascii=False)
    index, ids = QwenDirector._shot_batch_dialogue_index(payload)
    _assert(ids == ["scene_001_d001", "scene_002_d001"], f"wrong indexed dialogue IDs: {ids}")
    director = QwenDirector.__new__(QwenDirector)
    recovered_events = []
    director._record_recovery = lambda *args: recovered_events.append(args)
    shots = [
        {"shot_id": "scene_001_shot_001", "dialogue_events": [
            {"source_dialogue_id": "scene_001_d001"},
        ]},
        {"shot_id": "scene_001_shot_002", "dialogue_events": [
            {"source_dialogue_id": "scene_001_d001"},
            {"source_dialogue_id": "scene_002_d001"},
        ]},
    ]
    expanded = director._expand_shot_batch_dialogue_references(
        shots, index["scene_001"]
    )
    _assert(expanded[0]["dialogue_events"] == [{
        "speaker": "Mara Venn", "text": "Open the door… now!",
        "continues_from_previous_shot": False, "continues_to_next_shot": False,
    }], "reference expansion must restore exact source speaker/text/punctuation")
    _assert(expanded[1]["dialogue_events"] == [],
            "duplicate and cross-scene IDs must be rejected before source reconciliation")
    _assert(len(recovered_events) == 2,
            f"duplicate/foreign dialogue references should be observable: {recovered_events}")


def test_scene_local_dialogue_balancer_spreads_lines_and_preserves_source_constraints():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    source = {
        "scene_007_d001": {"speaker": "Ana", "text": "I found the hidden signal."},
        "scene_007_d002": {"speaker": "Ben", "text": "We need to leave right now."},
    }
    shots = [
        {"scene_id": "scene_007", "shot_id": f"scene_007_shot_{i:03d}",
         "characters": ["Ana", "Ben"],
         "dialogue_events": ([{"source_dialogue_id": "scene_007_d001"},
                              {"source_dialogue_id": "scene_007_d002"}] if i == 1 else [])}
        for i in range(1, 4)
    ]
    director._record_recovery = lambda *args: None
    expanded = director._expand_shot_batch_dialogue_references(shots, source)
    per_shot = [shot["dialogue_events"] for shot in expanded]
    _assert(all(len(events) <= 1 for events in per_shot),
            f"eligible source lines must be spread to separate shots: {per_shot!r}")
    spoken = [event for events in per_shot for event in events]
    _assert(len(spoken) == 2 and [event["text"] for event in spoken] == [source["scene_007_d001"]["text"], source["scene_007_d002"]["text"]],
            f"balancing must preserve both exact source lines and their order: {spoken!r}")
    _assert([event["speaker"] for event in spoken] == ["Ana", "Ben"],
            f"balancing must preserve canonical speakers: {spoken!r}")

    # A line must not be reassigned to a shot where its speaker is not visible.
    hidden = [dict(shot, characters=["Ana"]) for shot in shots]
    before = [[e["source_dialogue_id"] for e in shot["dialogue_events"]] for shot in hidden]
    after = director._h3_balance_scene_dialogue(hidden, source)
    after_ids = [[e["source_dialogue_id"] for e in shot["dialogue_events"]] for shot in after]
    _assert(before == after_ids,
            f"when no speaker-valid distribution exists, assignments must remain unchanged: {after_ids!r}")

    # Do not let a foreign scene ID migrate across scene boundaries.
    mixed_index = {**source, "scene_008_d001": {"speaker": "Mara", "text": "Wait."}}
    mixed = [dict(shot) for shot in shots]
    mixed[1]["dialogue_events"] = [{"source_dialogue_id": "scene_008_d001"}]
    before_mixed = [[e["source_dialogue_id"] for e in shot["dialogue_events"]] for shot in mixed]
    after_mixed = director._h3_balance_scene_dialogue(mixed, mixed_index)
    after_mixed_ids = [[e["source_dialogue_id"] for e in shot["dialogue_events"]] for shot in after_mixed]
    _assert(before_mixed == after_mixed_ids,
            f"balancer must leave mixed-scene assignments untouched: {after_mixed_ids!r}")


def test_scene_sentence_counts_are_flexible_but_dialogue_timing_is_advisory():
    from planner.qwen_director import QwenDirector
    from planner.config import DIRECTOR_STORY_SCENES
    director = QwenDirector.__new__(QwenDirector)
    # One sentence per scene is intentionally allowed by the advisory diagnostic.
    story = "\n\n".join(f"Scene {i} ends." for i in range(1, DIRECTOR_STORY_SCENES + 1))
    issues = director._story_sentence_contract_issues(story)
    _assert(not any("sentence count" in issue.lower() for issue in issues),
            f"sentence-count quotas must not be diagnosed as violations: {issues!r}")
    _assert(not any("topology" in issue.lower() for issue in issues),
            f"correct twelve-scene topology should pass: {issues!r}")
    dialogue_story = [f"Scene {i} is complete." for i in range(1, DIRECTOR_STORY_SCENES + 1)]
    dialogue_story[0] = 'Ana said, "One two three four five six seven eight nine ten." Then Ben left.'
    dialogue_issues = director._story_sentence_contract_issues("\n\n".join(dialogue_story))
    _assert(any("exceeds 9 words" in issue for issue in dialogue_issues),
            f"overlong quoted dialogue should be flagged as advisory: {dialogue_issues!r}")


def test_scene_local_dialogue_assignments_are_unique_and_constrained():
    from planner.qwen_director import QwenDirector
    probe = QwenDirector.__new__(QwenDirector)
    schema = probe._shot_batch_json_schema(
        scene_count=2,
        dialogue_ids_by_scene={
            "scene_001": ["scene_001_d001", "scene_001_d002"],
            "scene_002": [],
        },
    )
    variants = schema["properties"]["scene_shots"]["items"]["oneOf"]
    by_scene = {
        variant["properties"]["scene_id"]["enum"][0]: variant
        for variant in variants
    }
    first = by_scene["scene_001"]["properties"]["dialogue_assignments"]
    _assert(first["required"] == ["scene_001_d001", "scene_001_d002"],
            "each allowed source ID must be represented exactly once as a parent object key")
    for dialogue_id, property_schema in first["properties"].items():
        _assert(property_schema["enum"] == [1, 2, 3],
                f"dialogue assignment {dialogue_id} must select exactly one legal 1-based shot")
        _assert(property_schema["type"] == "integer",
                f"dialogue assignment {dialogue_id} must use a legal integer shot index")
    second = by_scene["scene_002"]["properties"]["dialogue_assignments"]
    _assert(second["properties"] == {} and second["required"] == [] and second["additionalProperties"] is False,
            "a scene with no source dialogue must allow only an empty assignment object")
    _assert("dialogue_assignments" in by_scene["scene_001"]["required"],
            "the keyed assignment map must be required on every scene response")
    shot_properties = by_scene["scene_001"]["properties"]["shots"]["items"]["properties"]
    _assert("dialogue_events" not in shot_properties,
            "live shot schema must not permit per-shot duplicated source IDs")


def test_batch_normalizer_expands_scene_level_dialogue_assignments():
    from planner.qwen_director_sanitize import QwenDirectorSanitizeMixin
    normalized = QwenDirectorSanitizeMixin._normalize_batch_shot_response({
        "scene_shots": [{
            "scene_id": "scene_004",
            "dialogue_assignments": {"scene_004_d001": 2, "scene_004_d002": 3},
            "shots": [{"action": "A"}, {"action": "B"}, {"action": "C"}],
        }]
    })
    shots = normalized["scene_004"]
    _assert(shots[0].get("dialogue_events", []) == [],
            "unassigned shot must not receive an invented dialogue reference")
    _assert(shots[1].get("dialogue_events", []) == [{"source_dialogue_id": "scene_004_d001"}],
            "parent assignment must expand to the selected 1-based shot only")
    _assert(shots[2].get("dialogue_events", []) == [{"source_dialogue_id": "scene_004_d002"}],
            "each source ID must expand only to its own assigned shot")
    _assert("source_dialogue_id" not in {event.get("source_dialogue_id") for event in shots[1].get("dialogue_events", [])},
            "a source ID assigned to another shot must not leak into this shot")


def test_repeated_dialogue_text_maps_to_the_exact_source_paragraph():
    import json
    from planner.qwen_director import QwenDirector

    story = (
        'Mira stood by the entrance and said, “Stay here.” She watched the door close.'
        '\n\n'
        'Tarek crossed the room and said, “Stay here.” He then guarded the corridor.'
    )
    director = QwenDirector.__new__(QwenDirector)
    director._reference_visual_context = {}
    director._semantic_spoken_dialogue = [
        {"quote_id": 1, "speaker": "Mira", "text": "Stay here."},
        {"quote_id": 2, "speaker": "Tarek", "text": "Stay here."},
    ]
    scenes = [
        {"scene_id": "scene_001", "description": "Mira watches the entrance."},
        {"scene_id": "scene_002", "description": "Tarek guards a corridor."},
    ]
    payload = json.loads(director._shot_director_batch_user(
        story, [{"name": "Mira"}, {"name": "Tarek"}], scenes, visual_language={}
    ))
    sources = {
        scene["scene_id"]: [(entry["speaker"], entry["text"]) for entry in scene["source_dialogue"]]
        for scene in payload["scenes"]
    }
    _assert(sources["scene_001"] == [("Mira", "Stay here.")],
            f"first repeated quote must bind to its first source scene: {sources}")
    _assert(sources["scene_002"] == [("Tarek", "Stay here.")],
            f"second repeated quote must bind to its own source scene: {sources}")


def test_source_dialogue_omitted_by_shot_model_is_restored_exactly_once():
    from planner.qwen_director import QwenDirector
    director = object.__new__(QwenDirector)
    director._qwen_telemetry = {"deterministic_recoveries": 0, "calls": [], "total_elapsed_seconds": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0, "cache_hits": 0}
    director._semantic_spoken_dialogue = [{"quote_id": 1, "text": "I found the key…", "speaker": "Mara"}]
    story = 'Mara entered and said, “I found the key…”'
    scenes = [{"scene_id": "scene_001", "description": story, "story_summary": story, "characters": ["Mara"]}]
    shots = [{"shot_id": f"scene_001_shot_{i:03d}", "scene_id": "scene_001", "characters": ["Mara"], "dialogue_events": [], "speaking_characters": [], "speech_text": ""} for i in range(1, 4)]
    director._normalize_dialogue_speakers(story, scenes, shots, [{"name": "Mara"}])
    events = [event for shot in shots for event in shot["dialogue_events"]]
    _assert(len(events) == 1, f"omitted source dialogue was not restored exactly once: {events!r}")
    _assert(events[0]["text"] == "I found the key…", f"source punctuation was not preserved: {events!r}")
    _assert(events[0]["speaker"] == "Mara", f"source speaker was not preserved: {events!r}")


def test_dialogue_cannot_drift_to_another_scene_and_is_restored_in_source_scene():
    from planner.qwen_director import QwenDirector
    director = object.__new__(QwenDirector)
    director._qwen_telemetry = {"deterministic_recoveries": 0, "calls": [], "total_elapsed_seconds": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0, "cache_hits": 0}
    story = 'Mara said, “The gate is open.”\n\nIvo said, “The bridge is gone.”'
    director._semantic_spoken_dialogue = [
        {"quote_id": 1, "text": "The gate is open.", "speaker": "Mara"},
        {"quote_id": 2, "text": "The bridge is gone.", "speaker": "Ivo"},
    ]
    scenes = [
        {"scene_id": "scene_001", "description": "Mara said, “The gate is open.”", "story_summary": "Mara said, “The gate is open.”", "characters": ["Mara"]},
        {"scene_id": "scene_002", "description": "Ivo said, “The bridge is gone.”", "story_summary": "Ivo said, “The bridge is gone.”", "characters": ["Ivo"]},
    ]
    shots = []
    for scene_id in ("scene_001", "scene_002"):
        for i in range(1, 4):
            shots.append({"shot_id": f"{scene_id}_shot_{i:03d}", "scene_id": scene_id, "characters": [], "dialogue_events": [], "speaking_characters": [], "speech_text": ""})
    shots[0]["dialogue_events"] = [{"speaker": "Ivo", "text": "The bridge is gone."}]
    director._normalize_dialogue_speakers(story, scenes, shots, [{"name": "Mara"}, {"name": "Ivo"}])
    by_scene = {scene_id: [event["text"] for shot in shots if shot["scene_id"] == scene_id for event in shot["dialogue_events"]] for scene_id in ("scene_001", "scene_002")}
    _assert(by_scene["scene_001"] == ["The gate is open."], f"scene 1 dialogue drifted or source line was lost: {by_scene}")
    _assert(by_scene["scene_002"] == ["The bridge is gone."], f"scene 2 dialogue was not restored: {by_scene}")


def test_long_preserve_dialogue_maps_through_rebalanced_source_scenes():
    from planner.qwen_director import QwenDirector
    director = object.__new__(QwenDirector)
    director._qwen_telemetry = {"deterministic_recoveries": 0, "calls": [], "total_elapsed_seconds": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0, "cache_hits": 0}
    paragraphs = [f'Mara said, “Line {i} remains source-anchored.”' for i in range(14)]
    story = "\n\n".join(paragraphs)
    director._semantic_spoken_dialogue = [
        {"quote_id": i + 1, "text": f"Line {i} remains source-anchored.", "speaker": "Mara"}
        for i in range(14)
    ]
    scene_texts = [" ".join(paragraphs[:3])] + paragraphs[3:]
    scenes = [{"scene_id": f"scene_{i + 1:03d}", "description": text, "story_summary": text, "characters": ["Mara"]} for i, text in enumerate(scene_texts)]
    shots = [{"shot_id": f"{scene['scene_id']}_shot_{j:03d}", "scene_id": scene["scene_id"], "characters": ["Mara"], "dialogue_events": [], "speaking_characters": [], "speech_text": "", "action": "Mara studies the evidence in the room.", "visual_prompt": "Mara studies the evidence in a grounded cinematic composition with the source setting visible."} for scene in scenes for j in range(1, 4)]
    director._normalize_dialogue_speakers(story, scenes, shots, [{"name": "Mara"}])
    emitted = [event["text"] for shot in shots for event in shot["dialogue_events"]]
    _assert(len(emitted) == 14, f"long source lost utterances while scenes were rebalanced: {emitted!r}")
    _assert(set(emitted) == {f"Line {i} remains source-anchored." for i in range(14)}, f"long source utterances changed: {emitted!r}")
    for i, paragraph in enumerate(paragraphs):
        target_scene = next(scene for scene in scenes if f"Line {i} remains source-anchored." in scene["description"])
        target_events = [event["text"] for shot in shots if shot["scene_id"] == target_scene["scene_id"] for event in shot["dialogue_events"]]
        _assert(f"Line {i} remains source-anchored." in target_events, f"source quote {i} drifted to a different scene")


# ── Speaker safety (in-person vs remote) ──────────────────────────────────

def test_in_person_dialogue_speaker_must_exist_in_qwen_cast():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    mixin = object.__new__(QwenDirectorPromptMixin)
    mixin._character_semantic_calls = 0
    mixin._semantic_spoken_dialogue = []
    mixin._qwen_telemetry = {"deterministic_recoveries": 0, "calls": [], "total_elapsed_seconds": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0, "cache_hits": 0}
    mixin._anchor_semantic_dialogue_to_story = lambda _story, items: list(items)
    mixin._chat_json = lambda *args, **kwargs: {
        "candidates": [{"name": "Elara", "entity_type": "PERSON", "is_character": True, "identity_type": "named_character"}],
        "quote_classifications": [{"quote_id": 1, "classification": "in_person_spoken", "speaker": "Kael Mercer"}],
    }
    result = mixin.extract_character_entities('Kael said, “I can walk.”')
    _assert(result["spoken_dialogue"] == [], "unrostered in-person dialogue must be dropped, not assigned to a guessed character")
    _assert(mixin._semantic_spoken_dialogue == [], "dropped dialogue must not leak into the later shot whitelist")
    _assert(mixin._qwen_telemetry["deterministic_recoveries"] == 1, "the deterministic drop must be observable in telemetry")


def test_source_attribution_overrides_mislabelled_ui_quote_speaker():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    story = "The display flashed “AEGIS: Access Denied.” Elara shouted, “I'm not here to play games!”"
    mixin = object.__new__(QwenDirectorPromptMixin)
    mixin._character_semantic_calls = 0
    mixin._semantic_spoken_dialogue = []
    mixin._qwen_telemetry = {"deterministic_recoveries": 0, "calls": [], "total_elapsed_seconds": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0, "cache_hits": 0}
    mixin._anchor_semantic_dialogue_to_story = lambda _story, items: list(items)
    mixin._chat_json = lambda *args, **kwargs: {
        "candidates": [
            {"name": "Elara", "entity_type": "PERSON", "is_character": True, "identity_type": "named_character"},
            {"name": "AEGIS", "entity_type": "SENTIENT", "is_character": True, "identity_type": "named_character"},
        ],
        "quote_classifications": [
            {"quote_id": 1, "classification": "written_or_displayed", "speaker": "AEGIS"},
            {"quote_id": 2, "classification": "remote_or_offscreen_spoken", "speaker": "AEGIS"},
        ],
    }
    result = mixin.extract_character_entities(story)
    spoken = result["spoken_dialogue"]
    _assert(len(spoken) == 1, f"only the directly spoken Elara line should survive: {spoken!r}")
    _assert(spoken[0]["text"] == "I'm not here to play games!" and spoken[0]["speaker"] == "Elara",
            f"source-named speaker attribution must override Qwen's mistaken AEGIS label: {spoken!r}")
    _assert(spoken[0].get("delivery") == "in_person", f"a nearby UI message must not make Elara's line remote: {spoken!r}")


def test_unrostered_remote_speaker_without_source_grounding_is_dropped():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    mixin = object.__new__(QwenDirectorPromptMixin)
    mixin._character_semantic_calls = 0
    mixin._semantic_spoken_dialogue = []
    mixin._qwen_telemetry = {"deterministic_recoveries": 0, "calls": [], "total_elapsed_seconds": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0, "cache_hits": 0}
    mixin._anchor_semantic_dialogue_to_story = lambda _story, items: list(items)
    mixin._chat_json = lambda *args, **kwargs: {
        "candidates": [{"name": "Eli", "entity_type": "PERSON", "is_character": True, "identity_type": "named_character"}],
        "quote_classifications": [{"quote_id": 1, "classification": "remote_or_offscreen_spoken", "speaker": "Ghost"}],
    }
    result = mixin.extract_character_entities('Eli entered the empty control room. “Open the door.” The corridor remained silent.')
    _assert(result["spoken_dialogue"] == [],
            f"Qwen's remote label alone must not invent an ungrounded source: {result['spoken_dialogue']!r}")
    _assert(mixin._qwen_telemetry["deterministic_recoveries"] == 1,
            "dropping a source-ungrounded remote speaker must be visible in telemetry")


def test_unrostered_hologram_speaker_is_audio_only_not_a_crash():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    story = "Eli opened the vault. Eli's sister (hologram) said, “Do not touch the core.”"
    mixin = object.__new__(QwenDirectorPromptMixin)
    mixin._character_semantic_calls = 0
    mixin._semantic_spoken_dialogue = []
    mixin._qwen_telemetry = {"deterministic_recoveries": 0, "calls": [], "total_elapsed_seconds": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0, "cache_hits": 0}
    mixin._anchor_semantic_dialogue_to_story = lambda _story, items: list(items)
    mixin._chat_json = lambda *args, **kwargs: {
        "candidates": [{"name": "Eli", "entity_type": "PERSON", "is_character": True, "identity_type": "named_character"}],
        "quote_classifications": [{"quote_id": 1, "classification": "in_person_spoken", "speaker": "Eli's sister (hologram)"}],
    }
    result = mixin.extract_character_entities(story)
    _assert(len(result["spoken_dialogue"]) == 1, f"an explicit hologram voice should survive as dialogue-only audio: {result!r}")
    _assert(result["spoken_dialogue"][0].get("delivery") == "remote", f"explicit hologram delivery must be remote: {result['spoken_dialogue']!r}")
    _assert([item["name"] for item in result["candidates"]] == ["Eli"], "hologram recovery must never add a cast member")


def test_remote_only_shot_can_have_no_visible_source_body():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    director._validate_shot_character_contract([{
        "shot_id": "scene_001_shot_002", "characters": [],
        "speaking_characters": ["AURORA"],
        "dialogue_events": [{"speaker": "AURORA", "text": "Vault integrity compromised.", "delivery": "remote"}],
    }], [{"name": "AURORA"}])
    try:
        director._validate_shot_character_contract([{
            "shot_id": "scene_001_shot_003", "characters": [],
            "speaking_characters": ["AURORA"],
            "dialogue_events": [{"speaker": "AURORA", "text": "Vault integrity compromised.", "delivery": "in_person"}],
        }], [{"name": "AURORA"}])
    except RuntimeError as exc:
        _assert("in-person speaker" in str(exc), f"wrong rejection for unbound in-person speaker: {exc}")
    else:
        raise AssertionError("an unbound in-person speaker must still fail closed")


def test_remote_delivery_survives_timeline_and_context_ir_without_visible_source():
    from pipeline.dialogue_timeline import DialogueTimeline
    from pipeline.context_ir import H3ContextIRCompiler
    timeline = DialogueTimeline(characters=[
        {"character_id": "character-aurora", "name": "AURORA"},
        {"character_id": "character-elara", "name": "Elara Voss"},
    ])
    shot = {
        "shot_id": "scene_001_shot_001", "scene_id": "scene_001",
        "duration_seconds": 5.2, "characters": ["Elara Voss"],
        "dialogue_events": [{"speaker": "AURORA", "text": "Opening the vault will trigger the storm.", "delivery": "remote"}],
    }
    events = timeline.schedule_shot(shot)
    _assert(len(events) == 1 and events[0]["delivery"] == "remote",
            f"timeline did not retain remote delivery or incorrectly require visible binding: {events!r}")
    _assert(events[0]["speaker_id"] == "character-aurora",
            f"remote speaker must resolve through canonical roster: {events!r}")
    context_lines = H3ContextIRCompiler._dialogue_lines(
        {"language": "English"},
        {"characters": ["Elara Voss"], "dialogue_events": events},
    )
    _assert(len(context_lines) == 1 and "remote channel" in context_lines[0].lower(),
            f"Context IR lost remote channel semantics: {context_lines!r}")
    _assert("do not show a physical source body" in context_lines[0].lower(),
            f"Context IR must distinguish audio-only source from a story-established projection: {context_lines!r}")


def test_dialogue_reconciliation_preserves_remote_audio_without_visible_binding():
    from planner.qwen_director import QwenDirector
    story = (
        'Lena’s radio crackled. “Voss, you still there?” It was Dr. Renn, his voice taut. '
        '“Don’t open that vault,” he said.\n\n'
        'At the main entrance, Renn appeared, his face lit by the emergency lights. '
        '“You’ve got ten minutes before the storm breaches the outer shell,” he said, gripping her arm.\n\n'
        'Renn yanked the ignition key from the dashboard. “No one leaves with that data,” he said, tossing the key into the wind.'
    )
    director = QwenDirector.__new__(QwenDirector)
    director._reference_visual_context = {}
    director._semantic_spoken_dialogue = [
        {"text": "Voss, you still there?", "speaker": "Dr. Renn"},
        {"text": "Don’t open that vault", "speaker": "Dr. Renn"},
        {"text": "You’ve got ten minutes before the storm breaches the outer shell", "speaker": "Dr. Renn"},
        {"text": "No one leaves with that data", "speaker": "Dr. Renn"},
    ]
    director._qwen_telemetry = {}
    director._recovery_events = []
    scenes = [
        {"scene_id": "scene_001", "characters": ["Lena"]},
        {"scene_id": "scene_002", "characters": ["Lena"]},
        {"scene_id": "scene_003", "characters": ["Lena"]},
    ]
    shots = [
        {"shot_id": "scene_001_shot_001", "scene_id": "scene_001", "characters": ["Lena"],
         "dialogue_events": [{"speaker": "Dr. Renn", "text": "Voss, you still there?"}]},
        {"shot_id": "scene_002_shot_001", "scene_id": "scene_002", "characters": ["Lena"],
         "dialogue_events": [{"speaker": "Dr. Renn", "text": "You’ve got ten minutes before the storm breaches the outer shell"}]},
        {"shot_id": "scene_003_shot_001", "scene_id": "scene_003", "characters": ["Lena"],
         "dialogue_events": [{"speaker": "Renn", "text": "No one leaves with that data"}]},
    ]
    characters = [
        {"name": "Lena", "identity_type": "named_character"},
        {"name": "Renn", "identity_type": "named_character"},
    ]
    director._normalize_dialogue_speakers(story, scenes, shots, characters)
    _assert(len(shots[0]["dialogue_events"]) == 2
            and all(event.get("delivery") == "remote" and event.get("speaker") == "Renn"
                    for event in shots[0]["dialogue_events"]),
            f"both remote radio turns must remain in the audio track with remote delivery: {shots[0]}")
    _assert("Renn" not in shots[0]["characters"] and "Renn" not in scenes[0]["characters"],
            "remote speaker must not be added to visual shot/scene bindings")
    _assert(len(shots[1]["dialogue_events"]) == 1 and shots[1]["dialogue_events"][0]["speaker"] == "Renn",
            f"honorific-bearing in-person speaker was lost or not canonicalized: {shots[1]}")
    _assert(len(shots[2]["dialogue_events"]) == 1 and shots[2]["dialogue_events"][0]["speaker"] == "Renn",
            f"canonical in-person speaker was lost: {shots[2]}")
    _assert("Renn" in shots[1]["characters"] and "Renn" in scenes[1]["characters"],
            "canonical speaker omitted from shot/scene bindings should be restored deterministically")


# ── Story pipeline hard gates (single-call, one-shot) ─────────────────────

def test_story_prompt_is_open_cast_and_fully_filled():
    from planner.qwen_director import QwenDirector
    from planner.config import (
        AI_STORY_MODE, EXPAND_USER_STORY_MODE, DIRECTOR_STORY_SCENES,
        DIRECTOR_STORY_CAST_SIZE_POLICY, RUNTIME, STORY_TARGET_WORDS,
        STORY_WRITER_WORDS_PER_SCENE,
    )
    q = QwenDirector.__new__(QwenDirector)
    floor = max(1300, int(STORY_TARGET_WORDS) - 20)
    sample_blueprint = "CAST: Ada | Bea | Cy\n" + "\n".join(
        f"SCENE {index:02d}: beat" for index in range(1, int(DIRECTOR_STORY_SCENES) + 1)
    )
    _assert(int(RUNTIME["director"]["story_sampling"]["writing"]["max_tokens"]) == 3000,
            "Author token ceiling must remain 3000")
    _assert(int(RUNTIME["director"]["story_sampling"]["writing"].get("minimum_output_tokens", 0)) == 0,
            "minimum_output_tokens must remain zero")
    _assert(int(RUNTIME["director"]["story_max_tokens"]) == 6500,
            "global story ceiling must remain unchanged")
    _assert(int(RUNTIME["director"]["story_architecture"]["max_tokens"]) == 3600,
            "Architect ceiling must remain unchanged")
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = q._story_text_system(mode)
        low = " ".join(text.lower().split())
        _assert("__" not in text, f"{mode}: unfilled prompt placeholder")
        _assert(f"exactly {DIRECTOR_STORY_SCENES} explicitly marked scene blocks" in low
                and "[[scene_01]]" in low and f"[[scene_{DIRECTOR_STORY_SCENES:02d}]]" in low,
                f"{mode}: explicit scene-boundary contract missing")
        _assert("names" in low and "relationships" in low,
                f"{mode}: Qwen must own names and relationships")
        _assert(DIRECTOR_STORY_CAST_SIZE_POLICY == "plot_driven"
                and RUNTIME["director"]["story_contract"].get("cast_size_policy") == "plot_driven",
                f"{mode}: general runtime cast policy must remain plot-driven")
        _assert(f"finished story must reach at least {floor}" in low,
                f"{mode}: prompt floor must state the 1300+ target")
        _assert("110-130" in low and "not flexible guidance" in low,
                f"{mode}: each scene must have a consistent 110-130 word contract")
        _assert("countdown values must only decrease" in low or "countdown values only decrease" in low,
                f"{mode}: countdown continuity contract missing")
        _assert("quotation marks only for audible" in low and "journal" in low,
                f"{mode}: written text must not be misclassified as spoken dialogue")
        _assert("varied sentence openings" in low and "never begin three consecutive" in low,
                f"{mode}: repetition guard missing")
        if mode == AI_STORY_MODE:
            _assert("at least three distinct named, living human characters" in low
                    and "physically appear" in low and "at least two scenes" in low,
                    "AI Story Writer must require three causally active humans")
        else:
            _assert("no fixed minimum or maximum cast size" in low,
                    "Expand Story must preserve its source-driven cast policy")

        writer = q._story_text_user(mode, "premise", story_blueprint=sample_blueprint)
        writer_low = " ".join(writer.lower().split())
        _assert(f"must reach at least {floor}" in writer_low,
                f"{mode}: Writer must state the hard 1300+ target")
        _assert("110-130 usable story words" in writer_low,
                f"{mode}: Writer must require 110-130 words in each scene")
        _assert(str(STORY_WRITER_WORDS_PER_SCENE) in writer and "continuity lock" in writer_low,
                f"{mode}: Writer must state configured per-scene guidance and continuity contract")
        _assert("begin now with [[scene_01]]" in writer_low,
                f"{mode}: Writer start marker missing")
        if mode == AI_STORY_MODE:
            _assert("ai story cast execution contract" in writer_low,
                    "AI Story cast contract must be explicit in the Writer prompt")
        blueprint = " ".join(q._story_blueprint_system(mode).lower().split())
        _assert("continuity ledger" in blueprint and "irreversible action" in blueprint,
                f"{mode}: Architect continuity ledger missing")
        _assert("85-105 words" in blueprint and "six labelled fields" in blueprint,
                f"{mode}: Architect beat length/shape contract missing")
        if mode == AI_STORY_MODE:
            _assert("at least three distinct named, living human characters" in blueprint
                    and "physically appears" in blueprint and "materially change" in blueprint,
                    "AI Story Architect must plan three active human characters")
        else:
            _assert("no fixed minimum or maximum cast quota" in blueprint,
                    "Expand Story Architect must preserve source-driven cast size")


def test_story_prompt_values_and_dialogue_exchange_contract_are_consistent():
    """Compatibility alias for repo environments whose main() references this name."""
    return test_story_prompt_is_open_cast_and_fully_filled()


def test_generated_story_cast_contract_is_plot_driven_without_a_numeric_quota():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE, DIRECTOR_STORY_CAST_SIZE_POLICY, RUNTIME

    contract = RUNTIME["director"]["story_contract"]
    _assert(DIRECTOR_STORY_CAST_SIZE_POLICY == "plot_driven", "cast policy must be plot-driven")
    _assert("minimum_active_characters" not in contract and "target_active_characters" not in contract,
            "runtime contract must not define a fixed cast minimum or target")

    for size in (1, 2, 3, 4, 7, 12, 24):
        cast = [{"name": f"Character {index}"} for index in range(1, size + 1)]
        for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
            QwenDirector._validate_generated_story_cast_size(mode, cast)

    try:
        QwenDirector._validate_generated_story_cast_size(AI_STORY_MODE, [])
    except RuntimeError as exc:
        _assert("non-empty cast" in str(exc).lower(), f"wrong empty-cast error: {exc}")
    else:
        raise AssertionError("an empty cast must fail because production cannot plan with no characters")


def test_story_blueprint_normalization_requires_ordered_scene_beats():
    from planner.qwen_director import QwenDirector
    from planner.config import DIRECTOR_STORY_SCENES
    director = QwenDirector.__new__(QwenDirector)
    cast = "CAST: Ines Vale (protagonist; save the archive; precise) | Tomas Reed (ally; expose theft; blunt) | Mara Quinn (opponent; protect family; guarded)"
    beats = "\n".join(
        f"SCENE {i:02d}: Beat {i} changes the story and carries the state forward."
        for i in range(1, DIRECTOR_STORY_SCENES + 1)
    )
    source = cast + "\n" + beats
    normalized = director._normalize_story_blueprint(source, DIRECTOR_STORY_SCENES)
    lines = normalized.splitlines()
    _assert(len(lines) == DIRECTOR_STORY_SCENES + 1, "blueprint must have a CAST line plus one line per scene")
    _assert(lines[0].startswith("CAST:") and lines[1].startswith("SCENE 01:")
            and lines[-1].startswith(f"SCENE {DIRECTOR_STORY_SCENES:02d}:"),
            "CAST must lead and scene numbering must be normalized and ordered")
    try:
        director._normalize_story_blueprint(cast + "\nSCENE 01: only one beat", DIRECTOR_STORY_SCENES)
    except RuntimeError as exc:
        _assert("missing" in str(exc).lower(), f"unexpected invalid-blueprint error: {exc}")
    else:
        raise AssertionError("incomplete story blueprint must fail closed without a retry")
    try:
        director._normalize_story_blueprint(beats, DIRECTOR_STORY_SCENES)
    except RuntimeError as exc:
        _assert("cast" in str(exc).lower(), f"missing CAST line must fail closed: {exc}")
    else:
        raise AssertionError("a blueprint without the CAST line must fail closed without a retry")
    duplicate = source + "\nSCENE 12: duplicate final beat that must not be silently merged."
    try:
        director._normalize_story_blueprint(duplicate, DIRECTOR_STORY_SCENES)
    except RuntimeError as exc:
        _assert("repeated scene label" in str(exc).lower(), f"unexpected duplicate-beat error: {exc}")
    else:
        raise AssertionError("duplicate scene labels must fail closed without a retry")


def test_story_architect_uses_native_bounded_thinking_and_no_retry():
    from planner.qwen_director import QwenDirector
    from planner.config import (
        AI_STORY_MODE, DIRECTOR_STORY_PLAN_MAX_TOKENS, DIRECTOR_STORY_PLAN_THINKING_BUDGET,
        DIRECTOR_STORY_THINKING_TEMPERATURE, DIRECTOR_STORY_THINKING_TOP_P, DIRECTOR_STORY_SCENES,
    )
    director = QwenDirector.__new__(QwenDirector)
    calls = []
    cast = "CAST: Ines Vale (protagonist; save the archive; precise) | Tomas Reed (ally; expose the theft; blunt) | Mara Quinn (opponent; protect her family; guarded)"
    raw = cast + "\n" + "\n".join(
        f"SCENE {i:02d}: Ines acts with Tomas and Mara; a visible choice changes the evidence and causes the next beat."
        for i in range(1, DIRECTOR_STORY_SCENES + 1)
    )
    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return raw
    director._chat_text = fake_chat
    blueprint = director._generate_story_blueprint(AI_STORY_MODE, "A premise.")
    _assert(len(calls) == 1, "architect pass must be single-call with no retry")
    _assert(calls[0].get("disable_thinking") is False, "architect must use thinking mode")
    _assert(calls[0].get("thinking_token_budget") == DIRECTOR_STORY_PLAN_THINKING_BUDGET == 1280,
            f"architect must use its configured reasoning budget: {calls[0]}")
    _assert(calls[0].get("max_completion") == DIRECTOR_STORY_PLAN_MAX_TOKENS,
            f"architect must use its configured hard output ceiling: {calls[0]}")
    _assert(calls[0].get("temperature") == DIRECTOR_STORY_THINKING_TEMPERATURE == 0.6
            and calls[0].get("top_p") == DIRECTOR_STORY_THINKING_TOP_P == 0.95,
            f"architect must use the official Qwen3 thinking sampler: {calls[0]}")
    _assert(blueprint.count("SCENE ") == DIRECTOR_STORY_SCENES,
            "normalized blueprint should include all configured scenes")


def test_story_minimum_is_a_hard_floor_without_retry():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES, STORY_ACCEPT_MIN_WORDS, STORY_MIN_WORDS
    from contextlib import redirect_stdout
    from io import StringIO

    temperature, top_p = QwenDirector.__new__(QwenDirector)._sampling_for_mode(AI_STORY_MODE)

    def run(words_per_scene):
        paragraphs = [
            " ".join([f"Ines{k}"] + [f"w{k}x{i}" for i in range(words_per_scene - 1)]) + "."
            for k in range(1, DIRECTOR_STORY_SCENES + 1)
        ]
        director = QwenDirector.__new__(QwenDirector)
        director._story_min_words = STORY_ACCEPT_MIN_WORDS
        calls = []

        def fake_chat(*args, **kwargs):
            calls.append(kwargs)
            return "\n\n".join(paragraphs)

        director._chat_text = fake_chat
        director._validate_story_output_contracts = lambda _mode, _input, candidate, **_kwargs: candidate
        try:
            with redirect_stdout(StringIO()):
                result, generated = director._generate_story_once(
                    AI_STORY_MODE, "premise", "system", "user",
                    temperature=temperature, top_p=top_p, source_character_names=[],
                )
            return result, generated, calls, None
        except RuntimeError as exc:
            return None, False, calls, str(exc)

    floor_words = (STORY_MIN_WORDS + DIRECTOR_STORY_SCENES - 1) // DIRECTOR_STORY_SCENES
    accepted_result, accepted_generated, accepted_calls, accepted_error = run(floor_words)
    _assert(STORY_MIN_WORDS == STORY_ACCEPT_MIN_WORDS == 816,
            f"minimum and acceptance floor must be 816: {STORY_MIN_WORDS}, {STORY_ACCEPT_MIN_WORDS}")
    _assert(accepted_generated and accepted_error is None and len(accepted_calls) == 1,
            f"story meeting the lower boundary must pass with one writer call: {accepted_error}")
    _assert(len(re.findall(r"\b[\w'’-]+\b", accepted_result)) >= STORY_MIN_WORDS,
            "accepted story must meet the minimum word boundary")

    short_result, short_generated, short_calls, short_error = run(65)
    _assert(not short_generated and short_result is None and len(short_calls) == 1,
            "sub-816-word story must fail after exactly one writer call, without retry")
    _assert("below the configured minimum" in (short_error or "").lower(),
            f"sub-floor story must report the minimum-length failure: {short_error}")

    _assert(len([x for x in re.split(r"\n\s*\n+", accepted_result) if x.strip()]) == DIRECTOR_STORY_SCENES,
            "accepted story must preserve the configured scene count")


def test_story_uses_one_architect_and_one_author_without_retry_or_repair():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    generate_start = source.index("story_blueprint = self._generate_story_blueprint(", source.index("def generate("))
    normal_path_end = source.index("# PASS 1B: deterministic production foundation", generate_start)
    normal_path = source[generate_start:normal_path_end]
    _assert(normal_path.count("self._generate_story_blueprint(") == 1,
            "normal path must make one Architect request")
    _assert(normal_path.count("self._generate_story_once(") == 1,
            "normal path must make one Author request")
    _assert("story_blueprint=story_blueprint" in normal_path,
            "Author must receive the validated Architect blueprint and CAST line")
    _assert("_story_single_call_system" not in source and "single_call=True" not in source,
            "combined thinking/write call must not remain")
    _assert("_rescue_short_scenes" not in source, "no rescue calls may exist")
    _assert("_repair_story_if_needed(" not in normal_path and "story_repair_pass" not in normal_path,
            "normal path must not retry or automatically repair the story")
    _assert("_resample_for_audit(" not in normal_path and "_pairwise_story_judgment(" not in normal_path,
            "normal path must not add challenger or judge calls")


def test_story_writer_is_one_qwen3_nonthinking_call_with_configured_ceiling():
    from planner.qwen_director import QwenDirector
    from planner.config import (
        AI_STORY_MODE, DIRECTOR_STORY_SCENES, DIRECTOR_STORY_MAX_TOKENS,
        DIRECTOR_STORY_WRITER_MAX_TOKENS, DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
        DIRECTOR_STORY_WRITER_TEMPERATURE,
        DIRECTOR_STORY_WRITER_TOP_P, STORY_MIN_WORDS,
    )
    paragraphs = [
        f"[[SCENE_{i:02d}]] " + " ".join(
            [f"Scene{i}", "Ines", "carried", "the", "lantern", "down", "the", "cellar", "stairs", "and", "counted", "each", "step."]
            + ["She", "listened", "for", "the", "door", "above", "her", "head."] * 8
        )
        for i in range(1, DIRECTOR_STORY_SCENES + 1)
    ]
    story = "\n\n".join(paragraphs)
    director = QwenDirector.__new__(QwenDirector)
    director._story_min_words = STORY_MIN_WORDS
    calls = []
    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return story
    director._chat_text = fake_chat
    director._validate_story_output_contracts = lambda _mode, _input, candidate, **_kwargs: candidate
    from contextlib import redirect_stdout
    from io import StringIO
    temperature, top_p = director._sampling_for_mode(AI_STORY_MODE)
    with redirect_stdout(StringIO()):
        result, generated = director._generate_story_once(
            AI_STORY_MODE, "premise", "system", "user",
            temperature=temperature, top_p=top_p, source_character_names=[],
        )
    count = len([p for p in re.split(r"\n\s*\n+", result) if p.strip()])
    _assert(generated and len(calls) == 1, f"story must have exactly one prose-writing call: {len(calls)}")
    _assert(calls[0].get("disable_thinking") is True, "prose writer must explicitly disable Qwen3 thinking")
    _assert(calls[0].get("creative") is True, "prose writer must use the configured creative-writing penalty")
    _assert(calls[0].get("temperature") == DIRECTOR_STORY_WRITER_TEMPERATURE == 0.7
            and calls[0].get("top_p") == DIRECTOR_STORY_WRITER_TOP_P == 0.8,
            f"prose writer must use the Qwen3 non-thinking sampler: {calls[0]}")
    expected_ceiling = min(DIRECTOR_STORY_MAX_TOKENS, DIRECTOR_STORY_WRITER_MAX_TOKENS)
    _assert(calls[0].get("max_completion") == expected_ceiling,
            f"prose writer must use its runtime-configured ceiling ({expected_ceiling}): {calls[0]}")
    _assert(calls[0].get("minimum_output_tokens") == DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
            f"prose writer must use the runtime-configured minimum output floor: {calls[0]}")
    _assert(STORY_MIN_WORDS == 816,
            f"minimum accepted story length must remain 816 words: {STORY_MIN_WORDS}")
    _assert(count == DIRECTOR_STORY_SCENES, f"paragraphs must be normalized to {DIRECTOR_STORY_SCENES}, got {count}")


def test_scene_markers_preserve_one_to_one_scene_boundaries():
    from planner.qwen_director import QwenDirector
    from planner.config import DIRECTOR_STORY_SCENES
    marked = "\n\n".join(
        f"[[SCENE_{index:02d}]] Scene {index} starts and ends in its own location."
        for index in range(1, DIRECTOR_STORY_SCENES + 1)
    )
    normalized = QwenDirector._coerce_story_to_scene_paragraphs(marked)
    paragraphs = [part for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    _assert(len(paragraphs) == DIRECTOR_STORY_SCENES,
            f"marked scenes should parse one-to-one: got {len(paragraphs)}")
    _assert("[[SCENE_" not in normalized, "structural markers must not leak into finished story prose")
    _assert(paragraphs[0].startswith("Scene 1") and paragraphs[-1].startswith(f"Scene {DIRECTOR_STORY_SCENES}"),
            "scene marker parsing changed story order")

    eleven = "\n\n".join(f"Existing beat {index}." for index in range(1, DIRECTOR_STORY_SCENES))
    untouched = QwenDirector._coerce_story_to_scene_paragraphs(eleven)
    _assert(len([p for p in untouched.split("\n\n") if p.strip()]) == DIRECTOR_STORY_SCENES - 1,
            "an 11-scene story must not be split into 12 artificial scenes")

    malformed = marked.replace("[[SCENE_06]]", "[[SCENE_05]]", 1)
    try:
        QwenDirector._coerce_story_to_scene_paragraphs(malformed)
    except ValueError as exc:
        _assert("must appear exactly once and in order" in str(exc),
                f"malformed marker failure should explain its invariant: {exc}")
    else:
        raise AssertionError("duplicate/out-of-order scene markers must fail closed")


# ── Context compaction + camera contract ─────────────────────────────────

def test_context_compaction_never_loses_closing_quote_or_splits_attribution():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    story = (
        "The station corridor remained cold and blue, while machinery rattled behind thick walls "
        "and distant ventilation fans filled the air with a steady metallic roar. "
        "The display flashed “AEGIS: Access Denied.” Elara shouted, “I'm not here to play games!” "
        "The old control panel glowed under a layer of frost and dust, its narrow buttons almost "
        "hidden below the cracked glass. The warning light remained red above the sealed door, "
        "while a weak amber lamp flickered in the far corner of the room. The wall panels vibrated "
        "whenever the generators changed pitch, making the thin shadows jump across the damp metal "
        "floor. The ventilation fans struggled against the cold, and a thin trail of vapor drifted "
        "through the broken service grille. Behind her, distant alarms filled the station as the "
        "storm pressed against reinforced outer walls, rattling every loose fixture in the corridor."
    )
    _assert(len(story) > 400, "quote-preservation test must exercise the compaction path")
    compact = QwenDirectorPromptMixin._compact_story_context(story, 400)
    _assert(len(compact) <= 400, f"compacted context must respect its budget: {len(compact)}")
    _assert('“AEGIS: Access Denied.”' in compact, f"closing punctuation for displayed text was lost: {compact!r}")
    _assert("Elara shouted, “I'm not here to play games!”" in compact,
            f"the named attribution and its quoted line must stay together: {compact!r}")


def test_story_context_compactor_keeps_causal_middle_action():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    story = "\n\n".join(
        f"The diver reached chamber {i} after a long descent through a narrow access tunnel lined with old warning labels and corroded support ribs. "
        f"The station had been silent for decades, and the pressure readings moved unpredictably as the diver crossed the damaged threshold. "
        f"A pressure seal failed, so Lena grabbed the valve and stopped the leak in chamber {i}. "
        f"The diver left chamber {i} with the valve locked, marking the repair with a strip of orange emergency tape for the next crew to find."
        for i in range(1, 7)
    )
    compact = QwenDirectorPromptMixin._compact_story_context(story, 1800)
    _assert("Lena grabbed the valve" in compact, "shot context dropped the causal action in the middle of a paragraph")
    _assert("P1:" in compact and "P6:" in compact, "shot context must retain all six paragraph anchors")


def test_camera_vocabulary_is_enforced_and_aliases_are_normalized():
    from planner.qwen_director import QwenDirector
    from planner.qwen_director_prompts import CAMERA_SHOT_OPTIONS, CAMERA_MOVEMENT_OPTIONS
    director = QwenDirector.__new__(QwenDirector)
    scene = {
        "scene_id": "scene_001", "description": "Lena watches the sealed vault under cold emergency lights.",
        "lighting": "cold emergency lights", "color_temperature": "cool", "mood": "tense",
        "location": "vault chamber", "characters": ["Lena"],
    }
    shots = [
        {"scene_id": "scene_001", "shot_id": "scene_001_shot_001", "location": "vault chamber",
         "action": "Lena examines the sealed vault door.", "visual_prompt": "Lena examines the sealed vault door under cold emergency light.",
         "characters": ["Lena"], "camera_shot": "wide shot", "camera_movement": "slow pan", "dialogue_events": []},
        {"scene_id": "scene_001", "shot_id": "scene_001_shot_002", "location": "vault chamber",
         "action": "Lena turns toward the rattling door.", "visual_prompt": "Lena turns toward the rattling vault door under cold emergency light.",
         "characters": ["Lena"], "camera_shot": "tracking shot", "camera_movement": "dolly in", "dialogue_events": []},
        {"scene_id": "scene_001", "shot_id": "scene_001_shot_003", "location": "vault chamber",
         "action": "Lena studies the control panel.", "visual_prompt": "Lena studies the control panel beneath cold emergency lights.",
         "characters": ["Lena"], "camera_shot": "nonsense framing", "camera_movement": "slow drift", "dialogue_events": []},
    ]
    sanitized = director._sanitize_shots(shots, scene, {"lena"})
    _assert([item["camera_shot"] for item in sanitized] == ["wide", "medium wide", "medium wide"],
            f"camera framing was not normalized/fail-closed: {[item['camera_shot'] for item in sanitized]!r}")
    _assert([item["camera_movement"] for item in sanitized] == ["pan", "push-in", "static"],
            f"camera movement was not normalized/fail-closed: {[item['camera_movement'] for item in sanitized]!r}")
    _assert(all(item["camera_shot"] in CAMERA_SHOT_OPTIONS for item in sanitized), "sanitizer emitted illegal camera framing")
    _assert(all(item["camera_movement"] in CAMERA_MOVEMENT_OPTIONS for item in sanitized), "sanitizer emitted illegal camera movement")


# ── Runner ────────────────────────────────────────────────────────────────

def main():
    tests = [
        # Schema contracts
        test_shot_schema_tracks_instance_topology,
        test_compact_shot_batch_schema_omits_derived_and_empty_fields,
        test_compact_shot_prompt_example_is_valid_json,
        test_character_schema_does_not_silently_cap_qwen_cast_at_32,
        test_character_semantic_schema_includes_spoken_dialogue,
        # Qwen as sole cast authority
        test_qwen_can_choose_any_cast_size,
        test_qwen_negative_is_authoritative,
        test_deterministic_detection_never_adds_missing_qwen_character,
        test_qwen_extraction_does_not_force_candidates_from_hints_plan_or_dialogue,
        test_mention_only_qwen_candidates_are_filtered_without_restoring_them,
        test_non_person_qwen_candidates_are_filtered,
        test_relational_owner_must_be_qwen_approved,
        # Dialogue ID/text integrity
        test_source_dialogue_ids_are_stable_and_scene_scoped,
        test_shot_batch_dialogue_references_restore_exact_source_text_and_deduplicate,
        test_scene_local_dialogue_balancer_spreads_lines_and_preserves_source_constraints,
        test_scene_sentence_counts_are_flexible_but_dialogue_timing_is_advisory,
        test_scene_local_dialogue_assignments_are_unique_and_constrained,
        test_batch_normalizer_expands_scene_level_dialogue_assignments,
        test_repeated_dialogue_text_maps_to_the_exact_source_paragraph,
        test_source_dialogue_omitted_by_shot_model_is_restored_exactly_once,
        test_dialogue_cannot_drift_to_another_scene_and_is_restored_in_source_scene,
        test_long_preserve_dialogue_maps_through_rebalanced_source_scenes,
        # Speaker safety (in-person vs remote)
        test_in_person_dialogue_speaker_must_exist_in_qwen_cast,
        test_source_attribution_overrides_mislabelled_ui_quote_speaker,
        test_unrostered_remote_speaker_without_source_grounding_is_dropped,
        test_unrostered_hologram_speaker_is_audio_only_not_a_crash,
        test_remote_only_shot_can_have_no_visible_source_body,
        test_remote_delivery_survives_timeline_and_context_ir_without_visible_source,
        test_dialogue_reconciliation_preserves_remote_audio_without_visible_binding,
        # Story pipeline hard gates
        test_story_prompt_values_and_dialogue_exchange_contract_are_consistent,
        test_generated_story_cast_contract_is_plot_driven_without_a_numeric_quota,
        test_story_blueprint_normalization_requires_ordered_scene_beats,
        test_story_architect_uses_native_bounded_thinking_and_no_retry,
        test_story_minimum_is_a_hard_floor_without_retry,
        test_story_uses_one_architect_and_one_author_without_retry_or_repair,
        test_story_writer_is_one_qwen3_nonthinking_call_with_configured_ceiling,
        test_scene_markers_preserve_one_to_one_scene_boundaries,
        # Context compaction + camera contract
        test_context_compaction_never_loses_closing_quote_or_splits_attribution,
        test_story_context_compactor_keeps_causal_middle_action,
        test_camera_vocabulary_is_enforced_and_aliases_are_normalized,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"Director validation PASSED ({len(tests)} checks).")


if __name__ == "__main__":
    main()

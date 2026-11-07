from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)



def _story_blueprint_fixture(scene_count=12):
    """A structurally complete plan with an explicit scene-state continuity ledger."""
    scenes = ", ".join(f"{i:02d}" for i in range(1, scene_count + 1))
    cast = (
        "CAST: Ines Vale (protagonist; wants to recover the original shipping ledger before the flood; "
        "pressure fears exposing her own signature; voice precise and restrained; aliases: none; "
        "on-screen scenes: " + scenes + "; fate: survives)"
    )
    fields = [
        "LOGLINE: A harbor surveyor must recover a falsified shipping ledger before a flood destroys the evidence and her reputation.",
        "PROTAGONIST_GOAL: Ines needs to recover the original ledger and verify which vessel carried the missing relief cargo.",
        "OPPOSITION: The harbor clerk wants the duplicate accepted and uses the delivery schedule to keep Ines away from the original.",
        "PRESSURE: The tide will close the lower archive and the council votes before dawn, making the evidence inaccessible.",
        "STORY_ENGINE: The harbor's physical tide clock and paper cargo chain expose how a small correction changed the rescue route.",
        "REVERSAL: A matching pressure mark proves Ines herself approved the altered route, changing the investigation into an admission.",
        "CLIMAX_CHOICE: Ines can preserve her career by hiding her signature or release the complete ledger and accept responsibility.",
        "RESOLUTION: She releases the original record, the rescue route is corrected, and her public admission costs her appointment.",
        "RELATIONSHIPS: none",
        "NAMED_ENTITIES: none",
        "KEY_OBJECTS: original shipping ledger, tide chart, pressure seal and cargo receipt",
        "CONTINUITY_LEDGER: Ines remains physically present in the harbor archive throughout this plan. She knows the ledger was duplicated, then verifies the pressure mark. The original ledger and tide chart remain under her control. The tide rises toward dawn. Ines survives and accepts public responsibility.",
    ]
    beats=[]
    for i in range(1, scene_count + 1):
        cause = "none" if i == scene_count else f"Because Ines records transfer clue {i}, she must verify the next cargo handoff before the rising tide blocks the archive route."
        beats.append(
            f"SCENE {i:02d}: ENTRY_LOCATION=harbor archive; ENTRY_PRESENT=Ines Vale; "
            "ENTRY_KNOWS=Ines knows the duplicate ledger conceals the missing relief cargo route; "
            "ENTRY_OBJECTS=original ledger and tide chart remain under Ines's control; "
            f"EVENT=Ines examines transfer stage {i}, compares the physical pressure mark with the approved cargo seal, "
            "and records the discrepancy before the clerk can remove the evidence; "
            "EXIT_LOCATION=harbor archive; EXIT_PRESENT=Ines Vale; "
            "EXIT_KNOWS=Ines knows the duplicate ledger conceals the missing relief cargo route; "
            "EXIT_OBJECTS=original ledger and tide chart remain under Ines's control; "
            f"NEXT_CAUSE={cause}"
        )
    return "\n".join([cast, *fields, *beats])


def _story_test_paragraphs(words_per_scene):
    starts = ["At dawn", "By noon", "After sunset", "Before the tide", "Across the quay", "Inside the archive",
              "Beyond the gate", "Under the crane", "Near the ferry", "Along the seawall", "Before first light", "At the final bell"]
    paragraphs=[]
    for scene in range(1, 13):
        words = [starts[scene-1], f"Ines{scene}", "moves", "the", "evidence", "through", "a", "specific", "physical", "choice"]
        while len(words) < words_per_scene - 1:
            words.append(f"scene{scene}detail{len(words)}")
        paragraphs.append(" ".join(words) + ".")
    return paragraphs


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
    annotated = QwenDirectorSceneMixin._annotate_scene_context([
        {"description": "The witness refuses to identify the person in the photograph."},
        {"description": "The protagonist quietly returns the key after learning the cost.",
         "dramatic_purpose": "The protagonist accepts the cost of the choice."},
    ])
    _assert(all("scene_function" not in item for item in annotated),
            "deterministic scene context must not assign fixed dramatic roles by scene position")
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
        "speaker": "Mara Venn", "text": "Open the door… now!", "delivery": "in_person",
        "continues_from_previous_shot": False, "continues_to_next_shot": False,
    }], "reference expansion must restore exact source speaker/text/punctuation/delivery")
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

    # If the model omits a physically speaking person from a shot, restore that
    # canonical visible binding rather than dropping/rejecting the source utterance.
    hidden = [dict(shot, characters=["Ana"]) for shot in shots]
    after = director._h3_balance_scene_dialogue(hidden, source)
    after_ids = [[e["source_dialogue_id"] for e in shot["dialogue_events"]] for shot in after]
    _assert({did for group in after_ids for did in group} == set(source),
            f"all source lines must be restored exactly once: {after_ids!r}")
    ben_shots = [shot for shot in after if any(e["source_dialogue_id"] == "scene_007_d002" for e in shot["dialogue_events"])]
    _assert(len(ben_shots) == 1 and "Ben" in ben_shots[0]["characters"],
            "an in-person source speaker must be visible in the shot receiving their line")

    # Foreign scene IDs are dropped, but missing valid source IDs are restored.
    mixed_index = {**source, "scene_008_d001": {"speaker": "Mara", "text": "Wait."}}
    mixed = [dict(shot) for shot in shots]
    mixed[1]["dialogue_events"] = [{"source_dialogue_id": "scene_008_d001"}]
    after_mixed = director._h3_balance_scene_dialogue(mixed, mixed_index)
    after_mixed_ids = [[e["source_dialogue_id"] for e in shot["dialogue_events"]] for shot in after_mixed]
    _assert("scene_008_d001" not in {did for group in after_mixed_ids for did in group}
            and {did for group in after_mixed_ids for did in group} == set(source),
            f"foreign IDs must be dropped and local source IDs restored: {after_mixed_ids!r}")


def test_scene_008_duplicate_dialogue_ids_are_redistributed_exactly_once():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    recoveries = []
    director._record_recovery = lambda *args: recoveries.append(args)
    source = {
        "scene_008_d001": {"speaker": "Mira", "text": "The core is in the reactor.", "delivery": "in_person"},
        "scene_008_d002": {"speaker": "Eli", "text": "Then shut the route.", "delivery": "in_person"},
        "scene_008_d003": {"speaker": "Rourke", "text": "You have four minutes.", "delivery": "remote"},
    }
    shots = [
        {"scene_id": "scene_008", "shot_id": f"scene_008_shot_{i:03d}",
         "characters": ["Mira", "Eli", "Rourke"],
         "dialogue_events": ([{"source_dialogue_id": "scene_008_d001"},
                              {"source_dialogue_id": "scene_008_d003"}] if i == 1 else
                             [{"source_dialogue_id": "scene_008_d002"}] if i == 2 else [])}
        for i in range(1, 4)
    ]
    expanded = director._expand_shot_batch_dialogue_references(shots, source)
    all_events = [event for shot in expanded for event in shot["dialogue_events"]]
    _assert([event["text"] for event in all_events] == [source[k]["text"] for k in source],
            "source dialogue must retain stable order and exact text")
    _assert(sum(len(shot["dialogue_events"]) for shot in expanded) == 3,
            "each of the three source IDs must be assigned exactly once")
    remote_shots = [shot for shot in expanded if any(e.get("delivery") == "remote" for e in shot["dialogue_events"])]
    _assert(len(remote_shots) == 1 and "Rourke" not in remote_shots[0]["characters"],
            "remote-only Rourke must remain audio-only even if Qwen listed him as visible")
    _assert(all("Rourke" not in shot["characters"] for shot in expanded),
            "remote-only speaker must be removed from every visible shot in the scene")


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


def test_remote_channel_state_ends_when_a_local_speaker_takes_over():
    from planner.qwen_director import QwenDirector
    story = "Rourke's voice came through comms. “Hold the line.” Mira hissed, “He's lying.”"
    semantic = [
        {"text": "Hold the line.", "speaker": "Rourke", "delivery": "remote"},
        {"text": "He's lying.", "speaker": "Mira", "delivery": "in_person"},
    ]
    segments = QwenDirector._extract_story_spoken_segments(
        story, semantic_spoken_dialogue=semantic,
        allowed_speakers=["Rourke", "Mira"], speaker_aliases={},
    )
    _assert(len(segments) == 2, f"both source-grounded speech turns must survive: {segments!r}")
    _assert(segments[0]["source_speakers"] == {"rourke"} and segments[0]["delivery"] == "remote",
            f"the comms line must remain remote: {segments[0]!r}")
    _assert(segments[1]["source_speakers"] == {"mira"} and segments[1]["delivery"] == "in_person",
            f"a new local speaker attribution must terminate remote-channel carry: {segments[1]!r}")


def test_remote_only_voice_is_removed_from_visible_scene_and_shot_bindings():
    from planner.qwen_director import QwenDirector
    scenes = [{
        "scene_id": "scene_001",
        "description": "Rourke's voice crackled through comms. Mira stood beside the console.",
        "characters": ["Rourke", "Mira"],
    }]
    shots = [{"scene_id": "scene_001", "characters": ["Rourke", "Mira"]}]
    characters = [{"name": "Rourke"}, {"name": "Mira"}]
    QwenDirector._filter_quote_only_character_bindings(scenes, shots, characters)
    _assert(scenes[0]["characters"] == ["Mira"],
            f"remote-only Rourke must not be a visible scene entity: {scenes[0]!r}")
    _assert(shots[0]["characters"] == ["Mira"],
            f"remote-only Rourke must not be a visible shot entity: {shots[0]!r}")


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
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE, DIRECTOR_STORY_SCENES
    from planner.config import DIRECTOR_STORY_CAST_SIZE_POLICY, RUNTIME, STORY_TARGET_WORDS, STORY_WRITER_WORDS_PER_SCENE
    q = QwenDirector.__new__(QwenDirector)
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = q._story_text_system(mode)
        low = text.lower()
        _assert("__" not in text, f"{mode}: unfilled length placeholder in prompt")
        _assert(f"return only the {DIRECTOR_STORY_SCENES} blocks" in low and "[[scene_01]]" in low and "[[scene_12]]" in low,
                f"{mode}: explicit 12-scene boundary contract missing")
        _assert(DIRECTOR_STORY_CAST_SIZE_POLICY == "plot_driven"
                and RUNTIME["director"]["story_contract"].get("cast_size_policy") == "plot_driven",
                f"{mode}: cast-size policy must be plot-driven")
        _assert("no numeric cast quota" in low or "only the cast established" in low,
                f"{mode}: prose prompt must not impose a cast quota")
        _assert("not physically present" in low and "remote" in low,
                f"{mode}: remote speakers must not be rendered as visible characters")
        _assert("do not use a fixed sentence count" in low and "vary scene lengths" in low,
                f"{mode}: fixed 9-sentence template must be removed")
        _assert("no more than two" in low and "nine words or fewer" in low,
                f"{mode}: spoken dialogue limits must match 3-shot timing")
        writer_prompt = q._story_text_user(
            mode, "premise", story_blueprint=_story_blueprint_fixture()
        ).lower()
        _assert(str(STORY_TARGET_WORDS) in writer_prompt
                and str(DIRECTOR_STORY_SCENES * STORY_WRITER_WORDS_PER_SCENE) in low
                and "validated plot architecture" in writer_prompt
                and "source-preservation lock" in writer_prompt if mode == EXPAND_USER_STORY_MODE else "validated plot architecture" in writer_prompt,
                f"{mode}: prose request must bind the plan and length target")
        blueprint = q._story_blueprint_system(mode).lower()
        _assert("plot-driven" in blueprint and "do not target four people" in blueprint,
                f"{mode}: Architect must not impose a numeric cast quota")
        _assert("continuity ledger" in blueprint and "costly climax choice" in blueprint,
                f"{mode}: Architect must plan continuity and meaningful climax cost")
        _assert("create exactly 12 linked beats" in blueprint and "do not assign the same fixed function" in blueprint,
                f"{mode}: Architect must avoid fixed scene-function and sentence templates")
        _assert("relationships" in blueprint and "named_entities" in blueprint and "entry_present" in blueprint
                and "fate: survives or dies in scene nn" in blueprint,
                f"{mode}: Architect must declare relationship/entity/fate/state contracts")
        _assert("at least five scene beats" in blueprint,
                f"{mode}: Architect must explicitly link beat causality")

def test_story_prompt_values_and_dialogue_exchange_contract_are_consistent():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES, STORY_MIN_WORDS, STORY_TARGET_WORDS, STORY_WRITER_WORDS_PER_SCENE
    director = QwenDirector.__new__(QwenDirector)
    system = director._story_text_system(AI_STORY_MODE)
    blueprint = director._story_blueprint_system(AI_STORY_MODE).lower()
    writer = director._story_text_user(AI_STORY_MODE, "premise", story_blueprint=_story_blueprint_fixture()).lower()
    stretch = DIRECTOR_STORY_SCENES * STORY_WRITER_WORDS_PER_SCENE
    for rendered in (system, blueprint, writer):
        _assert(not re.search(r"__[A-Z0-9_]+__", rendered), "rendered prompt has unresolved placeholders")
    _assert(str(STORY_MIN_WORDS) in system and str(STORY_MIN_WORDS) in writer and STORY_MIN_WORDS >= 1400,
            "1,400-word minimum must be explicit to both system and user prompts")
    _assert(str(STORY_TARGET_WORDS) in system and str(STORY_TARGET_WORDS) in writer,
            "system and user prompts must agree on the configured word target")
    _assert(str(stretch) in system and str(stretch) in writer,
            "system and user prompts must agree on stretch guidance")
    _assert("target is guidance, not permission to pad" in system.lower(),
            "target length must be guidance, not padding quota")
    _assert("do not use a fixed sentence count" in system.lower()
            and "action-obstacle-choice" in system.lower(),
            "writer must be free of rigid nine-sentence scene shape")
    _assert("no more than two" in system.lower() and "spoken lines per scene" in system.lower(),
            "writer must respect the two-line dialogue maximum")
    _assert("at least five scene beats" in blueprint and "each scene changes" in blueprint,
            "Architect must define causal scene progression")

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


def test_story_blueprint_accepts_wrapped_cast_formats():
    from planner.qwen_director import QwenDirector
    from planner.config import DIRECTOR_STORY_SCENES
    director = QwenDirector.__new__(QwenDirector)
    lines = _story_blueprint_fixture(DIRECTOR_STORY_SCENES).splitlines()
    entry = lines[0].split(":", 1)[1].strip()
    rest = lines[1:]
    split_at = entry.index("; pressure")
    variants = {
        "header alone, entry on next line": ["CAST:", entry] + rest,
        "bold header with bullet entry": ["**CAST**:", "- " + entry] + rest,
        "entry wrapped inside parentheses": ["CAST: " + entry[:split_at] + ";", entry[split_at + 2:]] + rest,
        "heading marker and qualifier": ["## CAST (core people): " + entry] + rest,
        "bare CAST header, entry per line with trailing pipe": ["CAST", entry + " |"] + rest,
        "honorific in the name": ["CAST", entry.replace("Ines Vale (", "Dr. Ines Vale (", 1) + " |"] + rest,
        "non-death fate wording (remains active)": ["CAST", re.sub(r"fate:\s*survives", "fate: remains active", entry) + " |"] + rest,
        "generic dead fate infers the last on-screen scene": ["CAST", re.sub(r"fate:\s*survives", "fate: dead", entry) + " |"] + rest,
        "bare bold CAST header": ["**CAST**", "- " + entry] + rest,
        "fate worded as survives in NN": ["CAST", re.sub(r"fate:\s*survives", "fate: survives in 12", entry) + " |"] + rest,
        "fate worded as survives to the end": ["CAST", re.sub(r"fate:\s*survives", "fate: survives to the end", entry) + " |"] + rest,
        "fate worded as survives in scene NN": ["CAST", re.sub(r"fate:\s*survives", "fate: survives in scene 12", entry) + " |"] + rest,
    }
    for label, text in variants.items():
        normalized = director._normalize_story_blueprint("\n".join(text), DIRECTOR_STORY_SCENES).splitlines()
        _assert(normalized[0].startswith("CAST: Ines Vale (") and normalized[0].count("(") == normalized[0].count(")"),
                f"CAST variant not recovered: {label}: {normalized[0][:120]!r}")
        _assert(len(normalized) == DIRECTOR_STORY_SCENES + 13, f"variant changed blueprint shape: {label}")
    try:
        director._normalize_story_blueprint("\n".join(rest), DIRECTOR_STORY_SCENES)
    except RuntimeError as exc:
        _assert("omitted the required cast line" in str(exc).lower() and "output started with" in str(exc).lower(),
                f"a missing CAST must still fail closed and show the output head: {exc}")
    else:
        raise AssertionError("a blueprint with no CAST must fail closed")


def test_story_blueprint_normalization_requires_ordered_scene_beats():
    from planner.qwen_director import QwenDirector
    from planner.config import DIRECTOR_STORY_SCENES
    director = QwenDirector.__new__(QwenDirector)
    source = _story_blueprint_fixture(DIRECTOR_STORY_SCENES)
    normalized = director._normalize_story_blueprint(source, DIRECTOR_STORY_SCENES)
    lines = normalized.splitlines()
    _assert(len(lines) == DIRECTOR_STORY_SCENES + 13,
            "blueprint must have CAST, twelve plot fields, and twelve scene beats")
    _assert(lines[0].startswith("CAST:") and lines[1].startswith("LOGLINE:")
            and lines[13].startswith("SCENE 01:") and lines[-1].startswith(f"SCENE {DIRECTOR_STORY_SCENES:02d}:"),
            "required plan fields and scene numbering must be normalized and ordered")
    try:
        director._normalize_story_blueprint("CAST: Ines Vale (protagonist; wants evidence; pressure fear; voice clipped; on-screen scenes: 01)\nSCENE 01: short", DIRECTOR_STORY_SCENES)
    except RuntimeError as exc:
        _assert("required plot fields" in str(exc).lower() or "missing" in str(exc).lower(),
                f"unexpected invalid-blueprint error: {exc}")
    else:
        raise AssertionError("incomplete blueprint must fail closed without a retry")
    duplicate = source + "\nSCENE 12: duplicate final beat that must not be silently merged."
    try:
        director._normalize_story_blueprint(duplicate, DIRECTOR_STORY_SCENES)
    except RuntimeError as exc:
        _assert("repeated scene label" in str(exc).lower(), f"unexpected duplicate-beat error: {exc}")
    else:
        raise AssertionError("duplicate scene labels must fail closed without a retry")

def test_story_architect_uses_native_bounded_thinking_and_no_retry():
    from planner.qwen_director import QwenDirector
    from planner.config import (AI_STORY_MODE, DIRECTOR_STORY_PLAN_MAX_TOKENS, DIRECTOR_STORY_PLAN_THINKING_BUDGET,
                                DIRECTOR_STORY_THINKING_TEMPERATURE, DIRECTOR_STORY_THINKING_TOP_P, DIRECTOR_STORY_SCENES)
    director = QwenDirector.__new__(QwenDirector)
    calls=[]
    raw=_story_blueprint_fixture(DIRECTOR_STORY_SCENES)
    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return raw
    director._chat_text=fake_chat
    blueprint=director._generate_story_blueprint(AI_STORY_MODE,"A harbor-surveyor premise.")
    _assert(len(calls)==1,"Architect must be a single bounded reasoning call with no retry")
    _assert(calls[0].get("disable_thinking") is False,"Architect must use thinking mode")
    _assert(calls[0].get("thinking_token_budget")==DIRECTOR_STORY_PLAN_THINKING_BUDGET==1280,
            f"Architect reasoning budget changed unexpectedly: {calls[0]}")
    _assert(calls[0].get("max_completion")==DIRECTOR_STORY_PLAN_MAX_TOKENS,
            f"Architect completion ceiling changed unexpectedly: {calls[0]}")
    _assert(calls[0].get("temperature")==DIRECTOR_STORY_THINKING_TEMPERATURE==0.6
            and calls[0].get("top_p")==DIRECTOR_STORY_THINKING_TOP_P==0.95,
            f"Architect thinking sampler changed unexpectedly: {calls[0]}")
    director._validate_story_blueprint_contract(AI_STORY_MODE,blueprint,"A premise.")
    _assert(blueprint.count("SCENE ")==DIRECTOR_STORY_SCENES,"normalized blueprint must include every unique scene")

def test_story_minimum_is_a_hard_floor_without_retry():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES, STORY_ACCEPT_MIN_WORDS, STORY_MIN_WORDS
    from contextlib import redirect_stdout
    from io import StringIO
    temperature, top_p = QwenDirector.__new__(QwenDirector)._sampling_for_mode(AI_STORY_MODE)
    def run(words_per_scene):
        paragraphs=_story_test_paragraphs(words_per_scene)
        director=QwenDirector.__new__(QwenDirector)
        director._story_min_words=STORY_ACCEPT_MIN_WORDS
        calls=[]
        def fake_chat(*args,**kwargs):
            calls.append(kwargs)
            return "\n\n".join(paragraphs)
        director._chat_text=fake_chat
        director._validate_story_output_contracts=lambda _mode,_input,candidate,**_kwargs:candidate
        try:
            with redirect_stdout(StringIO()):
                result,generated=director._generate_story_once(AI_STORY_MODE,"premise","system","user",
                    temperature=temperature,top_p=top_p,source_character_names=[])
            return result,generated,calls,None
        except RuntimeError as exc:
            return None,False,calls,str(exc)
    floor_words=(STORY_MIN_WORDS+DIRECTOR_STORY_SCENES-1)//DIRECTOR_STORY_SCENES
    accepted_result,accepted_generated,accepted_calls,accepted_error=run(floor_words)
    _assert(STORY_MIN_WORDS==STORY_ACCEPT_MIN_WORDS==1400,
            f"story floor and acceptance minimum must both be 1,400: {STORY_MIN_WORDS}, {STORY_ACCEPT_MIN_WORDS}")
    _assert(accepted_generated and accepted_error is None and len(accepted_calls)==1,
            f"story at or above the minimum must pass with one Author call: {accepted_error}")
    _assert(len(re.findall(r"\b[\w'’-]+\b",accepted_result))>=STORY_MIN_WORDS,
            "accepted story must meet the 1,400-word boundary")
    short_result,short_generated,short_calls,short_error=run(65)
    _assert(not short_generated and short_result is None and len(short_calls)==1,
            "sub-1,400-word story must fail after the single Author call, without retry")
    _assert("below the configured minimum" in (short_error or "").lower(),
            f"sub-floor story should report the hard minimum: {short_error}")
    completion_probe=QwenDirector.__new__(QwenDirector)
    completion_probe._story_min_words=STORY_MIN_WORDS
    try:
        completion_probe._validate_story_completion_contract(AI_STORY_MODE,"\n\n".join(_story_test_paragraphs(65)))
    except RuntimeError as exc:
        _assert("below the minimum accepted length" in str(exc).lower(),f"completion contract floor was not enforced: {exc}")
    else:
        raise AssertionError("completion contract must reject a story below 1,400 words")
    _assert(len([x for x in re.split(r"\n\s*\n+",accepted_result) if x.strip()])==DIRECTOR_STORY_SCENES,
            "accepted story must preserve twelve separate scene paragraphs")

def test_story_hard_quality_gate_rejects_repeated_stock_phrases_and_overlong_dialogue():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    paragraphs = [
        f"{word} the surveyor changed the route, checked the instrument, and committed to a different course through the station."
        for word in ["At", "Beyond", "Inside", "Before", "Across", "Under", "Near", "After", "Outside", "Between", "Past", "When"]
    ]
    clean = "\n\n".join(paragraphs)
    _assert(director._story_hard_quality_errors(clean, None) == [],
            "varied, non-repetitive twelve-scene prose should pass the local hard gate")
    defective = paragraphs.copy()
    defective[0] += ' Breath fogging covered the glass. Breath fogging covered the lens. “This line contains far too many words for the available shot duration.” “Second line.” “Third line.”'
    errors = director._story_hard_quality_errors("\n\n".join(defective), None)
    _assert(any("repeated stock phrases" in error for error in errors),
            f"repeated stock phrasing must be a hard failure: {errors!r}")
    _assert(any("spoken quote pairs" in error for error in errors),
            f"more than two dialogue quote pairs per scene must fail: {errors!r}")
    _assert(any("longer than 9 words" in error for error in errors),
            f"overlong dialogue must fail: {errors!r}")


def _story_contract_fixture_prose():
    """Twelve concise paragraphs that realize the fixture blueprint without extra entities."""
    paragraphs = []
    for i in range(1, 13):
        paragraph = (
            f"Ines examines transfer stage {i}, compares the physical pressure mark with the approved cargo seal, "
            "and records the discrepancy before the clerk removes evidence. The original ledger and tide chart "
            "remain under her control while the water rises toward the archive steps."
        )
        if i == 6:
            paragraph += " The matching pressure mark proves Ines approved the altered route, reversing the investigation."
        if i >= 10:
            paragraph += " Ines releases the original ledger, admits her signature, and accepts responsibility; the cargo route is corrected and her appointment is withdrawn."
        paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def test_blueprint_state_ledger_rejects_unrostered_people_location_jumps_and_post_death_presence():
    from planner.story_contracts import validate_blueprint_state_contract
    blueprint = _story_blueprint_fixture()
    _assert(not validate_blueprint_state_contract(blueprint), "valid fixture state ledger should pass")

    unrostered = blueprint.replace(
        "ENTRY_PRESENT=Ines Vale;", "ENTRY_PRESENT=Ines Vale, Rourke;", 1
    )
    issues = validate_blueprint_state_contract(unrostered)
    _assert(any("unrostered identity" in issue for issue in issues),
            f"state ledger must reject a person outside CAST: {issues!r}")

    moved = []
    for line in blueprint.splitlines():
        if line.startswith("SCENE 02:"):
            line = line.replace("ENTRY_LOCATION=harbor archive", "ENTRY_LOCATION=ferry office", 1)
        moved.append(line)
    issues = validate_blueprint_state_contract("\n".join(moved))
    _assert(any("changes location without a route/time transition" in issue for issue in issues),
            f"location discontinuity must fail before prose generation: {issues!r}")

    post_death = blueprint.replace("fate: survives", "fate: dies in scene 06", 1)
    issues = validate_blueprint_state_contract(post_death)
    _assert(any("physical appearances after their planned death" in issue for issue in issues),
            f"a character cannot be scheduled physically after death: {issues!r}")


def test_story_entity_relationship_and_person_grounding_is_a_hard_gate():
    from planner.story_contracts import audit_story_against_blueprint
    blueprint = _story_blueprint_fixture()
    story = _story_contract_fixture_prose()
    _assert(not audit_story_against_blueprint(story, blueprint),
            "grounded story fixture should pass the entity and continuity audit")

    contaminated = story.replace(
        "Ines examines transfer stage 1,",
        "Ines's sister Rourke entered Project Echo inside Eli's Lab. Ines examines transfer stage 1,", 1
    )
    issues = audit_story_against_blueprint(contaminated, blueprint)
    for expected in ("unseeded named project/facility/organization", "unseeded relationship/relative term", "unseeded named person"):
        _assert(any(expected in issue for issue in issues),
                f"hard entity/relationship gate missed {expected!r}: {issues!r}")

    seeded = blueprint.replace(
        "RELATIONSHIPS: none", "RELATIONSHIPS: Ines's sister Mara Vale is a source-confirmed family relationship."
    ).replace("NAMED_ENTITIES: none", "NAMED_ENTITIES: Project Echo, Eli's Lab")
    seeded_story = story.replace(
        "Ines examines transfer stage 1,",
        "Ines's sister was mentioned in the letter from Project Echo at Eli's Lab. Ines examines transfer stage 1,", 1
    )
    _assert(not audit_story_against_blueprint(seeded_story, seeded),
            "explicitly seeded relationship and named entities should pass the grounding gate")


def test_story_cinematic_coverage_enforces_unique_framing_without_camera_motion_quota():
    from planner.story_contracts import ensure_cinematic_coverage, validate_cinematic_coverage
    shots = [
        {"camera_shot": "close-up", "camera_movement": "static", "action": "Ines checks the tide chart beside the ledger.", "characters": ["Ines"]},
        {"camera_shot": "close-up", "camera_movement": "static", "action": "The clerk blocks the archive door with one hand.", "characters": ["Ines", "Clerk"]},
        {"camera_shot": "close-up", "camera_movement": "static", "action": "Ines notices the pressure seal is broken.", "characters": ["Ines"]},
    ]
    before_actions = [shot["action"] for shot in shots]
    result, adjustments = ensure_cinematic_coverage(shots, "scene_008")
    _assert(len({shot["camera_shot"] for shot in result}) == 3,
            f"three-shot scene must receive distinct framing: {[s['camera_shot'] for s in result]!r}")
    _assert(all(shot["camera_movement"] == "static" for shot in result),
            "cinematic variety must not force artificial camera movement")
    _assert([shot["action"] for shot in result] == before_actions,
            "framing normalization must never invent or rewrite story actions")
    _assert(not any("fewer than three distinct framings" in item for item in validate_cinematic_coverage(result, "scene_008")),
            "enforced cinematic coverage must pass its hard framing validator")
    _assert(adjustments, "duplicate framing should be explicitly recorded as a deterministic normalization")

    neutral = [
        {"camera_shot": "medium", "camera_movement": "static", "action": "Ines waits beside the closed doorway.", "characters": ["Ines"]},
        {"camera_shot": "medium", "camera_movement": "static", "action": "Ines listens to the rain against the roof.", "characters": ["Ines"]},
        {"camera_shot": "medium", "camera_movement": "static", "action": "Ines studies the empty corridor.", "characters": ["Ines"]},
    ]
    neutral_result, _ = ensure_cinematic_coverage(neutral, "scene_009")
    neutral_frames = {shot["camera_shot"] for shot in neutral_result}
    _assert(len(neutral_frames) == 3, "neutral single-person coverage must still produce three framings")
    _assert(not ({"insert", "pov", "two-shot", "over-the-shoulder"} & neutral_frames),
            "coverage normalization must not invent an object, viewpoint, or second participant")


def test_story_completion_contract_rejects_unresolved_final_hook():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES, STORY_MIN_WORDS
    director=QwenDirector.__new__(QwenDirector)
    director._story_min_words=STORY_MIN_WORDS
    paragraphs=_story_test_paragraphs(120)
    paragraphs[-1]=paragraphs[-1].replace(". ", ". ", 1)+" I’ll find it. The harbor gate closes and the rescue crew reaches safe ground."
    story="\n\n".join(paragraphs)
    _assert(len(re.findall(r"\b[\w'’-]+\b",story))>=1400,"hook fixture must meet the length floor")
    try:
        director._validate_story_completion_contract(AI_STORY_MODE,story)
    except RuntimeError as exc:
        _assert("unresolved future hook" in str(exc).lower(),f"completion gate did not catch an earlier final-scene hook: {exc}")
    else:
        raise AssertionError("a future promise anywhere in the final scene must fail")

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
    runtime_source = (ROOT / "planner/qwen_director_runtime.py").read_text(encoding="utf-8")
    _assert("_h3_n = 1" in runtime_source and "H3_STORY_WRITER_SAMPLES" not in runtime_source
            and "_h3_pick_best_story" not in runtime_source,
            "Writer must return one completion, not fan out and rank three full stories")
    _assert("_revise_story_once" not in source and "story_revision_pass" not in source,
            "revision model call and dead revision code must be removed")


def test_story_writer_is_one_qwen3_nonthinking_call_with_configured_ceiling():
    from planner.qwen_director import QwenDirector
    from planner.config import (AI_STORY_MODE,DIRECTOR_STORY_SCENES,DIRECTOR_STORY_MAX_TOKENS,
        DIRECTOR_STORY_WRITER_MAX_TOKENS,DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
        DIRECTOR_STORY_WRITER_TEMPERATURE,DIRECTOR_STORY_WRITER_TOP_P,STORY_MIN_WORDS)
    story="\n\n".join(_story_test_paragraphs(125))
    director=QwenDirector.__new__(QwenDirector)
    director._story_min_words=STORY_MIN_WORDS
    calls=[]
    def fake_chat(*args,**kwargs):
        calls.append(kwargs.copy())
        return story
    director._chat_text=fake_chat
    director._validate_story_output_contracts=lambda _mode,_input,candidate,**_kwargs:candidate
    from contextlib import redirect_stdout
    from io import StringIO
    temperature,top_p=director._sampling_for_mode(AI_STORY_MODE)
    with redirect_stdout(StringIO()):
        result,generated=director._generate_story_once(AI_STORY_MODE,"premise","system","user",
            temperature=temperature,top_p=top_p,source_character_names=[])
    count=len([p for p in re.split(r"\n\s*\n+",result) if p.strip()])
    _assert(generated and len(calls)==1,"story must have exactly one prose-writing call")
    _assert(calls[0].get("disable_thinking") is True,"prose writer must disable Qwen3 thinking")
    _assert(calls[0].get("creative") is True,"prose writer must use configured creative sampler")
    _assert(calls[0].get("temperature")==DIRECTOR_STORY_WRITER_TEMPERATURE==0.7
            and calls[0].get("top_p")==DIRECTOR_STORY_WRITER_TOP_P==0.8,
            f"Writer sampler changed unexpectedly: {calls[0]}")
    expected=min(DIRECTOR_STORY_MAX_TOKENS,DIRECTOR_STORY_WRITER_MAX_TOKENS)
    _assert(calls[0].get("max_completion")==expected and expected==3000,
            f"Writer must use the configured 3,000-token completion ceiling: {calls[0]}")
    _assert(calls[0].get("minimum_output_tokens")==DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
            f"Writer min-output setting changed: {calls[0]}")
    _assert(STORY_MIN_WORDS==1400,"minimum accepted story length must be 1,400 words")
    _assert(count==DIRECTOR_STORY_SCENES,"story must preserve twelve scene paragraphs")

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
        test_scene_008_duplicate_dialogue_ids_are_redistributed_exactly_once,
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
        test_remote_channel_state_ends_when_a_local_speaker_takes_over,
        test_remote_only_voice_is_removed_from_visible_scene_and_shot_bindings,
        # Story pipeline hard gates
        test_story_prompt_is_open_cast_and_fully_filled,
        test_story_prompt_values_and_dialogue_exchange_contract_are_consistent,
        test_generated_story_cast_contract_is_plot_driven_without_a_numeric_quota,
        test_story_blueprint_normalization_requires_ordered_scene_beats,
        test_story_blueprint_accepts_wrapped_cast_formats,
        test_story_architect_uses_native_bounded_thinking_and_no_retry,
        test_story_minimum_is_a_hard_floor_without_retry,
        test_story_hard_quality_gate_rejects_repeated_stock_phrases_and_overlong_dialogue,
        test_blueprint_state_ledger_rejects_unrostered_people_location_jumps_and_post_death_presence,
        test_story_entity_relationship_and_person_grounding_is_a_hard_gate,
        test_story_cinematic_coverage_enforces_unique_framing_without_camera_motion_quota,
        test_story_completion_contract_rejects_unresolved_final_hook,
        test_story_uses_one_architect_and_one_author_without_retry_or_repair,
        test_story_writer_is_one_qwen3_nonthinking_call_with_configured_ceiling,
        test_scene_markers_preserve_one_to_one_scene_boundaries,
        # Context compaction + camera contract
        test_context_compaction_never_loses_closing_quote_or_splits_attribution,
        test_story_context_compactor_keeps_causal_middle_action,
        test_camera_vocabulary_is_enforced_and_aliases_are_normalized,
    ]
    import contextlib
    import io
    import os
    verbose = os.getenv("H3_VALIDATE_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}
    for test in tests:
        if verbose:
            test()
            print(f"PASS {test.__name__}")
            continue
        captured = io.StringIO()
        try:
            with contextlib.redirect_stdout(captured):
                test()
        except BaseException:
            print(captured.getvalue(), end="")
            print(f"FAIL {test.__name__}")
            raise
    print(f"Director validation PASSED ({len(tests)} checks).")


if __name__ == "__main__":
    main()

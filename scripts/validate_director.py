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
    """Structurally complete plan without MID_SCENE_* fields (current Architect contract)."""
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
        "SETUP_PAYOFF_MAP: PLANT Scene 02: Ines finds the duplicate manifest's pressure seal with a shifted notch; PAYOFF Scene 08: Ines uses that same shifted pressure seal notch to prove her correction changed the rescue route.",
        "CONTINUITY_LEDGER: Ines remains physically present in the harbor archive throughout this plan. She knows the ledger was duplicated, then verifies the pressure mark. The original ledger and tide chart remain under her control. The tide rises toward dawn. Ines survives and accepts public responsibility.",
    ]
    beats = []
    for i in range(1, scene_count + 1):
        cause = (
            "none" if i == scene_count else
            f"Because Ines records transfer clue {i}, she must verify the next cargo handoff before the rising tide blocks the archive route."
        )
        beats.append(
            f"SCENE {i:02d}: PURPOSE=The surveyor loses another viable way to verify cargo transfer stage {i}; ENTRY_LOCATION=harbor archive; ENTRY_PRESENT=Ines Vale; ON_SCREEN_PRESENT=Ines Vale; "
            "ENTRY_KNOWS=Ines knows the duplicate ledger conceals the missing relief cargo route; "
            "ENTRY_OBJECTS=original ledger and tide chart remain under Ines's control; "
            f"EVENT=Ines examines transfer stage {i}, compares the physical pressure mark with the approved cargo seal, "
            + ("finds the duplicate manifest's pressure seal with a shifted notch, and records it; " if i == 2 else "")
            + ("uses that same shifted pressure seal notch to prove her correction changed the rescue route; " if i == 8 else "")
            + ("and records the discrepancy before the clerk can remove the evidence; " if i not in (2, 8) else "")
            + "EXIT_LOCATION=harbor archive; EXIT_PRESENT=Ines Vale; "
            "EXIT_KNOWS=Ines knows the duplicate ledger conceals the missing relief cargo route; "
            "EXIT_OBJECTS=original ledger and tide chart remain under Ines's control; "
            f"NEXT_CAUSE={cause}"
        )
    return "\n".join([cast, *fields, *beats])


def _story_test_paragraphs(words_per_scene):
    starts = ["At dawn", "By noon", "After sunset", "Before the tide", "Across the quay", "Inside the archive",
              "Beyond the gate", "Under the crane", "Near the ferry", "Along the seawall", "Before first light", "At the final bell"]
    paragraphs = []
    for scene in range(1, 13):
        words = [starts[scene - 1], f"Ines{scene}", "moves", "the", "evidence", "through", "a", "specific", "physical", "choice"]
        while len(words) < words_per_scene - 1:
            words.append(f"scene{scene}detail{len(words)}")
        paragraphs.append(" ".join(words) + ".")
    return paragraphs


def _planner():
    from planner.production_planner import ProductionPlanner
    return ProductionPlanner(ROOT)


def test_cast_absent_from_one_scene_is_advisory_but_never_appearing_is_fatal():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    blueprint = _story_blueprint_fixture()
    paragraphs = _story_contract_fixture_prose().split("\n\n")
    one_scene = paragraphs.copy()
    one_scene[11] = one_scene[11].replace("Ines", "She")
    errors = director._story_hard_quality_errors("\n\n".join(one_scene), blueprint, "")
    _assert(not any("CAST character" in error for error in errors),
            f"a cast member named elsewhere but not in one declared scene must not discard the draft: {errors!r}")
    _assert(any("not named in declared on-screen scene 12" in w for w in director._last_story_contract_warnings),
            f"the missing scene must still be reported: {director._last_story_contract_warnings!r}")
    nowhere = [p.replace("Ines", "She").replace("Vale", "") for p in paragraphs]
    errors = director._story_hard_quality_errors("\n\n".join(nowhere), blueprint, "")
    _assert(any("never appears in the story" in error for error in errors),
            f"a planned character who never appears must stay a hard failure: {errors!r}")


def _story_contract_fixture_prose():
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


def _two_person_blueprint(mara_scenes="02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12", mara_fate="survives", arrival="meets"):
    lines = _story_blueprint_fixture().splitlines()
    lines[0] = lines[0] + (
        " | Mara Quill (harbor clerk; wants to keep the duplicate ledger accepted; pressure fears the audit; "
        f"voice curt; aliases: none; on-screen scenes: {mara_scenes}; fate: {mara_fate})"
    )
    out = []
    for line in lines:
        if line.startswith("SCENE "):
            number = int(line[6:8])
            if number >= 2:
                line = line.replace("ENTRY_PRESENT=Ines Vale;", "ENTRY_PRESENT=Ines Vale, Mara Quill;")
                line = line.replace("ON_SCREEN_PRESENT=Ines Vale;", "ON_SCREEN_PRESENT=Ines Vale, Mara Quill;")
                line = line.replace("EXIT_PRESENT=Ines Vale;", "EXIT_PRESENT=Ines Vale, Mara Quill;")
            if number == 2 and arrival:
                line = line.replace(
                    "EVENT=Ines examines",
                    f"EVENT=Mara Quill {arrival} Ines at the archive door while Ines examines",
                )
        out.append(line)
    return "\n".join(out)


# ── Schema contracts ────────────────────────────────────────────────────────

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


# ── Qwen as sole cast authority ────────────────────────────────────────────

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


def test_trace_sentient_ai_candidate_survives_normalization_and_reaches_roster():
    """Regression for 2026-11-14: SENTIENT 'AI' was deleted as a generic placeholder."""
    planner = _planner()
    story = (
        "Dr. Elara Voss reached the vault. The station's AI denied access over the terminal. "
        "The AI triggered a lockdown, then spoke through the station speakers. "
        "A holographic interface from the station's AI materialized before Elara."
    )
    extracted = {
        "candidates": [
            {"name": "Dr. Elara Voss", "entity_type": "PERSON", "is_character": True,
             "aliases": ["Dr. Voss"], "identity_type": "named_character"},
            {"name": "AI", "entity_type": "SENTIENT", "is_character": True,
             "aliases": [], "identity_type": "named_character"},
        ],
        "quote_classifications": [],
    }
    characters = planner.create_characters(story, qwen_character_extractor=lambda *_: extracted)
    names = [character.name for character in characters]
    _assert(names == ["Elara Voss", "Station AI"],
            f"Qwen's two grounded candidates must both survive into the canonical roster: {names!r}")
    station_ai = characters[1]
    _assert("AI" in station_ai.semantic_aliases,
            f"the model's original spoken surface must remain a resolvable alias: {station_ai.semantic_aliases!r}")


def test_sentient_ai_label_is_not_promoted_when_qwen_does_not_mark_it_sentient():
    planner = _planner()
    story = "Elara read the AI label on a damaged terminal."
    extracted = {"candidates": [{
        "name": "AI", "entity_type": "CHARACTER", "is_character": True,
        "aliases": [], "identity_type": "named_character",
    }]}
    try:
        planner.create_characters(story, qwen_character_extractor=lambda *_: extracted)
    except RuntimeError as exc:
        _assert("no valid production characters" in str(exc), f"unexpected rejection: {exc}")
    else:
        raise AssertionError("a generic AI label without Qwen's SENTIENT decision must not become a character")


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


# ── Dialogue ID/text integrity ─────────────────────────────────────────────

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
    hidden = [dict(shot, characters=["Ana"]) for shot in shots]
    after = director._h3_balance_scene_dialogue(hidden, source)
    after_ids = [[e["source_dialogue_id"] for e in shot["dialogue_events"]] for shot in after]
    _assert({did for group in after_ids for did in group} == set(source),
            f"all source lines must be restored exactly once: {after_ids!r}")
    ben_shots = [shot for shot in after if any(e["source_dialogue_id"] == "scene_007_d002" for e in shot["dialogue_events"])]
    _assert(len(ben_shots) == 1 and "Ben" in ben_shots[0]["characters"],
            "an in-person source speaker must be visible in the shot receiving their line")
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


# ── Speaker safety (in-person vs remote) ───────────────────────────────────

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


# ── Story pipeline hard gates ──────────────────────────────────────────────

def test_story_prompt_is_open_cast_and_fully_filled():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE, DIRECTOR_STORY_SCENES
    from planner.config import DIRECTOR_STORY_CAST_SIZE_POLICY, RUNTIME, STORY_TARGET_WORDS, STORY_WRITER_WORDS_PER_SCENE
    q = QwenDirector.__new__(QwenDirector)
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = q._story_text_system(mode)
        low = " ".join(text.lower().split())
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
        _assert("full canonical name" in low and "never depict it as a human" in low
                and "visible nonhuman" in low,
                f"{mode}: the Author must preserve canonical sentient-agent identity without humanizing it")
        _assert("three distinct dramatic movements" in low and "2-3 paragraphs" in low
                and "natural sentence lengths" in low,
                f"{mode}: each scene must have three distinct dramatized movements without a rigid sentence quota")
        _assert("no more than two" in low and "nine words or fewer" in low,
                f"{mode}: spoken dialogue limits must match 3-shot timing")
        writer_prompt = q._story_text_user(
            mode, "premise", story_blueprint=_story_blueprint_fixture()
        ).lower()
        _assert(str(STORY_TARGET_WORDS) in writer_prompt
                and str(STORY_TARGET_WORDS) in low
                and f"{STORY_WRITER_WORDS_PER_SCENE} words" in writer_prompt
                and f"{STORY_WRITER_WORDS_PER_SCENE} words" in low
                and "validated plot architecture" in writer_prompt
                and "source-preservation lock" in writer_prompt if mode == EXPAND_USER_STORY_MODE else "validated plot architecture" in writer_prompt,
                f"{mode}: prose request must bind the plan and length target")
        blueprint = " ".join(q._story_blueprint_system(mode).lower().split())
        for structured in (False, True):
            rendered_architect = " ".join(q._story_blueprint_system(mode, structured=structured).lower().split())
            design_heading = ("cast design / ensemble decision" if mode == AI_STORY_MODE
                              else "cast design / expansion ensemble")
            _assert(design_heading in rendered_architect,
                    f"{mode}: structured={structured} Architect must include the mode-appropriate cast-design instructions")
            _assert("no fixed minimum, maximum, or target" in rendered_architect,
                    f"{mode}: structured={structured} Architect must remain free of cast-count quotas")
            if mode == AI_STORY_MODE:
                _assert("3-5 person active ensemble" in rendered_architect
                        and "compare it against 1, 2, 6, or more" in rendered_architect
                        and "do not default to a one-person story" in rendered_architect,
                        f"{mode}: structured={structured} Architect must compare an ensemble without forcing a cast quota")
                _assert("independent goals" in rendered_architect and "consequential decision" in rendered_architect,
                        f"{mode}: structured={structured} supporting characters must have independent agency")
        _assert("plot-driven" in blueprint and "no fixed minimum, maximum, or target" in blueprint,
                f"{mode}: Architect must not impose a numeric cast quota")
        if mode == EXPAND_USER_STORY_MODE:
            _assert("source's original cast size is not a limit" in blueprint
                    and "3-5 person active ensemble" in blueprint
                    and "do not create a role merely to reach a number" in blueprint,
                    "Expand mode must preserve source anchors while considering purposeful ensemble additions")
            _assert("architect's approved cast may include new premise-grounded people" in low,
                    "Expand Author must honor new characters approved by the Architect")
        if mode == AI_STORY_MODE:
            _assert("3-5 person active ensemble" in blueprint
                    and "compare it against 1, 2, 6, or more" in blueprint
                    and "do not default to a one-person story" in blueprint,
                    "AI Story Architect must actively evaluate an ensemble without imposing a cast count")
            _assert("independent goals" in blueprint and "consequential decision" in blueprint,
                    "additional cast members must have distinct agency, not filler roles")
        _assert("continuity ledger" in blueprint and "costly climax choice" in blueprint,
                f"{mode}: Architect must plan continuity and meaningful climax cost")
        _assert("create exactly 12 linked beats" in blueprint and "do not assign the same fixed function" in blueprint,
                f"{mode}: Architect must avoid fixed scene-function and sentence templates")
        _assert("relationships" in blueprint and "named_entities" in blueprint and "entry_present" in blueprint
                and "on_screen_present" in blueprint and "fate: survives or dies in scene nn" in blueprint,
                f"{mode}: Architect must declare relationship/entity/fate/full-scene visibility contracts")
        blueprint_compact = " ".join(blueprint.split())
        _assert("kind: person|animal|embodied agent" in blueprint_compact and "weather" in blueprint_compact
                and "remote-only voices" in blueprint_compact
                and "visible nonhuman" in blueprint_compact,
                f"{mode}: the Architect must distinguish physical cast, visible agents and remote opposition")
        _assert("each of the eleven transitions" in blueprint,
                f"{mode}: Architect must explicitly link every scene transition")


def test_story_prompt_values_and_dialogue_exchange_contract_are_consistent():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES, STORY_MIN_WORDS, STORY_TARGET_WORDS, STORY_WORDS_PER_SCENE, STORY_WRITER_WORDS_PER_SCENE
    director = QwenDirector.__new__(QwenDirector)
    system = director._story_text_system(AI_STORY_MODE)
    blueprint = director._story_blueprint_system(AI_STORY_MODE).lower()
    writer = director._story_text_user(AI_STORY_MODE, "premise", story_blueprint=_story_blueprint_fixture()).lower()
    scene_cap = STORY_WRITER_WORDS_PER_SCENE
    for rendered in (system, blueprint, writer):
        _assert(not re.search(r"__[A-Z0-9_]+__", rendered), "rendered prompt has unresolved placeholders")
    _assert(str(STORY_MIN_WORDS) in system and str(STORY_MIN_WORDS) in writer and STORY_MIN_WORDS >= 1400,
            "1,400-word minimum must be explicit to both system and user prompts")
    # The 1,560-word overall target and 190-word per-scene soft maximum are separate controls.
    _assert(str(STORY_TARGET_WORDS) in system and str(STORY_TARGET_WORDS) in writer
            and STORY_TARGET_WORDS == DIRECTOR_STORY_SCENES * STORY_WORDS_PER_SCENE == 1560,
            "system and user prompts must agree on the configured 1,560-word overall target")
    _assert(f"per-scene soft maximum is {scene_cap} words" in system.lower()
            and f"soft maximum of {scene_cap} words" in writer.lower() and scene_cap == 190,
            "190 must be available as the per-scene soft maximum, not multiplied into a 2,280-word quota")
    _assert("2280" not in system and "2280" not in writer,
            "the per-scene maximum must not become an overall 2,280-word drafting quota")
    _assert("only the rejection floor" in " ".join(system.split()) and "only the rejection floor" in " ".join(writer.split()),
            "both prompts must call the minimum a rejection floor, not a goal")
    system_flat = " ".join(system.lower().split())
    _assert("not filler, repeated description, or extra twists" in system_flat,
            "length must come from events, not padding")
    _assert("three distinct dramatic movements" in system_flat and "2-3 paragraphs" in system_flat
            and "same causal and dramatic care as scenes 01-04" in system_flat,
            "writer must give every scene three distinct movements and the final third full depth")
    _assert("no more than two" in system.lower() and "spoken lines per scene" in system.lower(),
            "writer must respect the two-line dialogue maximum")
    _assert("each of the eleven transitions" in blueprint and "each scene changes" in blueprint,
            "Architect must define causal progression across all eleven transitions")
    _assert("opposition variety" in blueprint and "clock and timeline integrity" in blueprint
            and "family/location shortcuts" in blueprint,
            "Architect must avoid repetitive puzzle traps, arbitrary countdowns, family shortcuts, and unmotivated geography")
    _assert("exact units and arithmetic" in system_flat,
            "Author must preserve any approved countdown/timestamp arithmetic across scenes")
    allocation_match = re.search(
        r"per-scene quota \(sums to \d+\):\s*(.*?)\. every scene must contain", system_flat, re.S
    )
    allocations = [int(value) for value in re.findall(r"scene \d{2}: (\d+) words", allocation_match.group(1))] if allocation_match else []
    _assert(len(allocations) == 12 and sum(allocations) == STORY_TARGET_WORDS,
            "Author must receive one explicit twelve-scene allocation that sums to the overall drafting target")
    _assert("a scene must be complete, not a fragment" in system_flat
            and "scenes 08-12 receive the same causal and dramatic care" in system_flat,
            "Author must be told the per-scene completeness and final-third depth guidance")
    # A non-thinking writer cannot "silently count"; the model is given a budget and an expansion retry instead.
    _assert("silently check" not in system_flat and "silently verify" not in " ".join(writer.split()),
            "the Author must not be asked to count words silently")
    _assert("premise-specific mechanism" in blueprint and "same plot would work unchanged" in blueprint,
            "Architect must anchor the plot's causal engine to the supplied premise")
    _assert("explicitly refer back to the planted evidence" in system_flat and "climax" in system_flat
            and "architect payoff map" in system_flat,
            "Author must preserve the mapped reversal plants and seeded climax")


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
        _assert(len(normalized) == DIRECTOR_STORY_SCENES + 14, f"variant changed blueprint shape: {label}")
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
    _assert(len(lines) == DIRECTOR_STORY_SCENES + 14,
            "blueprint must have CAST, thirteen plot fields, and twelve scene beats")
    _assert(lines[0].startswith("CAST:") and lines[1].startswith("LOGLINE:")
            and "SETUP_PAYOFF_MAP:" in lines[12] and lines[14].startswith("SCENE 01:")
            and lines[-1].startswith(f"SCENE {DIRECTOR_STORY_SCENES:02d}:"),
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


def test_cast_wants_label_spellings_match_the_prompt_example_and_reach_downstream_parsers():
    """Regression: the prompt example says 'wants: X' (colon) but the gate only accepted 'wants X'."""
    from planner.config import DIRECTOR_STORY_SCENES
    from planner.qwen_director import QwenDirector
    from planner.qwen_director_prompts import canonicalize_cast_metadata
    base = _story_blueprint_fixture()
    scenes = ", ".join(f"{i:02d}" for i in range(1, DIRECTOR_STORY_SCENES + 1))
    tail = f"aliases: none; on-screen scenes: {scenes}; fate: survives"
    variants = {
        "colon": "wants: recover the original shipping ledger before the flood",
        "colon_caps": "Wants: recover the original shipping ledger before the flood",
        "plain": "wants to recover the original shipping ledger before the flood",
        "bracket": "wants [recover the original shipping ledger before the flood]",
        "goal": "goal: recover the original shipping ledger before the flood",
        "needs": "needs to recover the original shipping ledger before the flood",
    }
    for label, wants in variants.items():
        cast = (f"CAST: Elara Voss (kind: person; role: lead engineer; {wants}; pressure [her signature is on the forgery]; "
                f"voice [clipped, exact]; {tail})")
        raw = base.replace(base.splitlines()[0], cast, 1)
        out = QwenDirector._normalize_story_blueprint(raw, DIRECTOR_STORY_SCENES)
        entry = out.splitlines()[0]
        _assert("wants:" in entry or "wants " in entry, f"[{label}] wants lost: {entry}")
        _assert("pressure: her signature" in entry and "voice: clipped" in entry,
                f"[{label}] pressure/voice must be canonical 'label: value': {entry}")
        _assert("[" not in entry, f"[{label}] brackets must be stripped: {entry}")
    # A genuinely missing goal still fails, and the error shows the offending entry.
    bad = base.replace(base.splitlines()[0],
                       f"CAST: Elara Voss (kind: person; role: lead engineer; pressure: dread; voice: dry; {tail})", 1)
    try:
        QwenDirector._normalize_story_blueprint(bad, DIRECTOR_STORY_SCENES)
    except RuntimeError as exc:
        _assert("Elara Voss" in str(exc) and "Entry as returned" in str(exc), f"unhelpful error: {exc}")
    else:
        raise AssertionError("a CAST entry with no goal must still be rejected")
    _assert(canonicalize_cast_metadata("voice actor; wants: x y z") == "voice actor; wants: x y z",
            "a leading role that merely starts with 'voice' must not be rewritten")


def test_scene_purpose_survives_scene_sanitizer_and_scene_floor_is_enforced():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE
    director = QwenDirector.__new__(QwenDirector)
    scenes = director._sanitize_scenes([{"scene_id": "scene_001", "description": "Ines examines the ledger.",
                                         "dramatic_purpose": "The surveyor loses a viable verification route."}], {"ines vale"})
    _assert(scenes[0]["dramatic_purpose"] == "The surveyor loses a viable verification route.",
            "sanitizer must keep the Architect purpose")
    director._story_min_words = 1400
    starved = _story_test_paragraphs(140)
    starved[3] = " ".join(starved[3].split()[:60]) + "."
    try:
        director._validate_story_completion_contract(AI_STORY_MODE, "\n\n".join(starved))
    except RuntimeError as exc:
        _assert("below the 70-word hard floor" in str(exc), f"thin scene not named: {exc}")
    else:
        raise AssertionError("a 60-word scene is starved and must fail the per-scene hard floor")
    # A near miss (95 words against a 100-word guideline) is not a defect and must not reject the story.
    near_miss = _story_test_paragraphs(140)
    near_miss[3] = " ".join(near_miss[3].split()[:95]) + "."
    director._validate_story_completion_contract(AI_STORY_MODE, "\n\n".join(near_miss))


def test_story_prompts_encode_escalation_opposition_setup_payoff_and_scene_purpose():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS, STORY_TARGET_WORDS, STORY_WRITER_WORDS_PER_SCENE
    from planner.qwen_director_prompts import SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT

    director = QwenDirector.__new__(QwenDirector)
    architect = " ".join(director._story_blueprint_system(AI_STORY_MODE).split())  # prompt is hard-wrapped
    author = director._story_text_system_raw(AI_STORY_MODE)
    _assert("raises the stakes, narrows the protagonist's viable options" in architect,
            "Architect RULE 2 must explicitly demand escalation")
    _assert("at least two concrete tactical decisions/actions" in architect,
            "Architect must require consequential opposition actions")
    _assert("SETUP_PAYOFF_MAP" in architect and "silent recurrence" in architect,
            "Architect must explicitly map plants to visible payoffs")
    _assert("CLIMAX CHOICE" in architect and "physical, procedural, or moral constraint" in architect,
            "Architect must define both constraints in the climax dilemma")
    for banned in ("simulation/dream/memory", "protagonist-was-dead", "becomes sentient and turns against its creators",
                   "one-last-job", "chosen-one prophecy", "only a symbol for grief or trauma"):
        _assert(banned in architect, f"Architect anti-cliche list missing {banned!r}")
    _assert("PURPOSE=[" in architect and "PURPOSE" in author,
            "Architect scene PURPOSE must be passed as a required Author contract")
    _assert("Architect-authored `dramatic_purpose`" in SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT
            and "copy it verbatim" in SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT,
            "Shot Director must consume, not reinvent, Architect dramatic purpose")
    allocation_match = re.search(r"Per-scene quota \(sums to \d+\):\s*(.*?)\. Every scene must contain", " ".join(author.split()), re.S)
    _assert(allocation_match is not None, "Author needs one explicit per-scene allocation schedule")
    allocated = [int(value) for value in re.findall(r"SCENE \d{2}: (\d+) words", allocation_match.group(1))]
    _assert(len(allocated) == 12 and sum(allocated) == STORY_TARGET_WORDS == 1560,
            f"Author allocation must cover twelve scenes and total the overall target: {allocated!r}")
    _assert(f"per-scene soft maximum is {STORY_WRITER_WORDS_PER_SCENE} words" in author.lower(),
            "Author system prompt must expose the 190-word per-scene soft maximum")
    _assert("minimum_output_tokens" not in author and "min_tokens" not in author,
            "Prompt improvement must not change the runtime min_tokens setting")
    _assert(DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS == 0,
            "the Author minimum output token setting must remain exactly zero")


def test_architect_scene_purpose_is_preserved_through_shot_director_metadata():
    from planner.qwen_director import QwenDirector
    import json
    from planner.story_contracts import parse_scene_state

    director = QwenDirector.__new__(QwenDirector)
    blueprint = _story_blueprint_fixture()
    beats = director._blueprint_scene_beats(blueprint)
    _assert(len(beats) == 12, "all twelve Architect scene beats must be mapped")
    first_purpose = parse_scene_state(beats[1]).get("PURPOSE")
    _assert(first_purpose and len(first_purpose.split()) >= 5,
            f"Architect purpose did not parse: {first_purpose!r}")
    scenes = director._sanitize_scenes(
        [{"scene_id": "scene_001", "description": "Ines examines the ledger.",
          "dramatic_purpose": first_purpose}],
        {"ines vale"},
    )
    _assert(scenes and scenes[0].get("dramatic_purpose") == first_purpose,
            "scene sanitizer must retain Architect-authored dramatic purpose")
    QwenDirector._apply_scene_metadata(
        scenes[0],
        {"dramatic_purpose": "A conflicting purpose invented by the Shot Director."},
    )
    _assert(scenes[0].get("dramatic_purpose") == first_purpose,
            "Shot Director metadata must not overwrite Architect canonical purpose")
    director._reference_visual_context = {}
    director._semantic_spoken_dialogue = []
    shot_payload = json.loads(director._shot_director_batch_user(
        story="", characters=[{"name": "Ines Vale"}], scenes=scenes,
    ))
    _assert(shot_payload["scenes"][0].get("dramatic_purpose") == first_purpose,
            "the actual Shot Director request payload must carry Architect PURPOSE")


def test_successful_architect_plan_uses_one_native_bounded_thinking_call():
    from planner.qwen_director import QwenDirector
    from planner.config import (AI_STORY_MODE, DIRECTOR_STORY_PLAN_MAX_TOKENS, DIRECTOR_STORY_PLAN_THINKING_BUDGET,
                                DIRECTOR_STORY_THINKING_TEMPERATURE, DIRECTOR_STORY_THINKING_TOP_P, DIRECTOR_STORY_SCENES)
    director = QwenDirector.__new__(QwenDirector)
    calls = []
    raw = _story_blueprint_fixture(DIRECTOR_STORY_SCENES)
    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return raw
    director._chat_text = fake_chat
    blueprint = director._generate_story_blueprint(AI_STORY_MODE, "A harbor-surveyor premise.")
    _assert(len(calls) == 1, "a valid first Architect plan must cost exactly one bounded reasoning call")
    _assert(calls[0].get("disable_thinking") is False, "Architect must use thinking mode")
    _assert(calls[0].get("thinking_token_budget") == DIRECTOR_STORY_PLAN_THINKING_BUDGET == 1280,
            f"Architect reasoning budget changed unexpectedly: {calls[0]}")
    _assert(calls[0].get("max_completion") == DIRECTOR_STORY_PLAN_MAX_TOKENS,
            f"Architect completion ceiling changed unexpectedly: {calls[0]}")
    _assert(calls[0].get("temperature") == DIRECTOR_STORY_THINKING_TEMPERATURE == 0.6
            and calls[0].get("top_p") == DIRECTOR_STORY_THINKING_TOP_P == 0.95,
            f"Architect thinking sampler changed unexpectedly: {calls[0]}")
    director._validate_story_blueprint_contract(AI_STORY_MODE, blueprint, "A premise.")
    _assert(blueprint.count("SCENE ") == DIRECTOR_STORY_SCENES, "normalized blueprint must include every unique scene")



def _blueprint_json_from_fixture():
    """The structured Architect payload equivalent to _story_blueprint_fixture()."""
    from planner.config import DIRECTOR_STORY_SCENES
    from planner.qwen_director import QwenDirector
    from planner.story_contracts import parse_scene_state
    from planner.story_blueprint_schema import AUTHORED_SCENE_FIELDS
    text = _story_blueprint_fixture()
    director = QwenDirector.__new__(QwenDirector)
    scenes = []
    opening = {}
    for number, beat in sorted(director._blueprint_scene_beats(text).items()):
        state = {k.lower(): v for k, v in parse_scene_state(beat).items()}
        for key in ("entry_present", "on_screen_present", "exit_present"):
            state[key] = [v.strip() for v in state[key].split(",") if v.strip()]
        if number == 1:
            opening = {"location": state["entry_location"], "present": state["entry_present"],
                       "knows": state["entry_knows"], "objects": state["entry_objects"]}
        scenes.append({k: state[k] for k in AUTHORED_SCENE_FIELDS})
    fields = {k: v for k, v in (line.split(": ", 1) for line in text.splitlines()[1:14])}
    data = {
        "cast": [{"name": "Ines Vale", "kind": "person", "role": "harbor surveyor",
                  "wants": "recover the original shipping ledger before the flood",
                  "pressure": "her own signature is on the forgery", "voice": "precise and restrained",
                  "aliases": [], "fate": "survives", "death_scene": 0}],
        "setup_payoff_map": [{"plant_scene": 2, "plant": "the duplicate manifest's pressure seal has a shifted notch",
                              "payoff_scene": 8, "payoff": "Ines uses that same shifted notch to prove her correction changed the route"}],
        "opening": opening,
        "scenes": scenes,
    }
    for key in ("logline", "protagonist_goal", "opposition", "pressure", "story_engine", "reversal",
                "climax_choice", "resolution", "relationships", "named_entities", "key_objects", "continuity_ledger"):
        data[key] = fields[key.upper()]
    _assert(len(scenes) == DIRECTOR_STORY_SCENES, "fixture must yield every scene")
    return data


def test_structured_architect_json_round_trips_through_every_contract():
    import json
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES
    from planner.qwen_director import QwenDirector
    from planner.story_blueprint_schema import blueprint_json_schema, render_blueprint_text
    data = _blueprint_json_from_fixture()
    # Mechanically derived: the cast list never has to be authored twice, and cannot disagree with the scenes.
    text = render_blueprint_text(data, DIRECTOR_STORY_SCENES)
    _assert("on-screen scenes: 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12" in text.splitlines()[0],
            f"on-screen scenes must be derived from on_screen_present: {text.splitlines()[0]}")
    director = QwenDirector.__new__(QwenDirector)
    normalized = QwenDirector._normalize_story_blueprint(text, DIRECTOR_STORY_SCENES)
    director._validate_story_blueprint_contract(AI_STORY_MODE, normalized, "A premise.")
    # The same payload through the real Architect entry point, including a fenced reply.
    json_blueprint = director._blueprint_json_to_text("```json\n" + json.dumps(data) + "\n```")
    _assert(json_blueprint == text, "JSON conversion must be deterministic")
    schema = blueprint_json_schema(DIRECTOR_STORY_SCENES)
    _assert(schema["properties"]["scenes"]["minItems"] == schema["properties"]["scenes"]["maxItems"] == DIRECTOR_STORY_SCENES,
            "the schema must force exactly one object per scene")
    _assert("maxItems" not in schema["properties"]["cast"],
            "the structured Architect schema must not impose an upper cast-size limit")
    _assert(set(schema["required"]) == set(schema["properties"]), "every plan field must be required by the schema")
    member = schema["properties"]["cast"]["items"]
    _assert({"wants", "pressure", "voice"} <= set(member["required"]),
            "wants/pressure/voice are schema keys, so a goal can no longer be mis-labelled or omitted")
    # Regression (Kaggle run): the model listed "The Storm" as cast; it is never on screen, so it must be
    # removed (and reported) rather than failing the whole plan with an empty on-screen scene list.
    storm = json.loads(json.dumps(data))
    storm["cast"].append({"name": "The Storm", "kind": "embodied agent", "role": "relentless antagonist",
                          "wants": "drown the harbor", "pressure": "none", "voice": "roaring",
                          "aliases": [], "fate": "survives", "death_scene": 0})
    dropped = []
    storm_text = render_blueprint_text(storm, DIRECTOR_STORY_SCENES, dropped)
    _assert(dropped == ["The Storm"] and "The Storm" not in storm_text.splitlines()[0], "a never-on-screen cast entry must be removed")
    QwenDirector._normalize_story_blueprint(storm_text, DIRECTOR_STORY_SCENES)
    # Two cast members sharing a surname are not conflated by name-part matching.
    twin = json.loads(json.dumps(data))
    twin["cast"].append(dict(twin["cast"][0], name="Tomas Vale"))
    twin_dropped = []
    twin_line = render_blueprint_text(twin, DIRECTOR_STORY_SCENES, twin_dropped).splitlines()[0]
    _assert(twin_dropped == ["Tomas Vale"] and "Tomas Vale" not in twin_line and "Ines Vale (kind: person" in twin_line,
            f"a shared surname must not put one person in the other's scenes (Tomas is on screen nowhere): {twin_line}")
    twin["scenes"][5]["on_screen_present"] = ["Tomas Vale"]
    twin_line = render_blueprint_text(twin, DIRECTOR_STORY_SCENES).splitlines()[0]
    ines, tomas = twin_line.split(" | ")[:2]
    _assert("on-screen scenes: 06;" in tomas and "on-screen scenes: 01, 02, 03, 04, 05, 06, 07" in ines,
            f"Tomas gets only the scene that names him; Ines stays at both boundaries throughout: {twin_line}")


def test_architect_and_author_get_only_informed_hard_gate_revisions():
    import json
    from contextlib import redirect_stdout
    from io import StringIO
    from tempfile import TemporaryDirectory
    from planner.config import (
        AI_STORY_MODE, STORY_ARCHITECT_REVISION_RETRIES, STORY_AUTHOR_REVISION_RETRIES,
    )
    from planner.qwen_director import QwenDirector
    from planner.qwen_director_runtime import StoryTruncated

    good = json.dumps(_blueprint_json_from_fixture())
    _assert(STORY_ARCHITECT_REVISION_RETRIES == 1,
            "production policy must retain exactly one conditional Architect revision")
    _assert(STORY_AUTHOR_REVISION_RETRIES == 1,
            "production policy must retain exactly one conditional Author revision")

    with TemporaryDirectory() as tmp:
        def architect(replies):
            director = QwenDirector.__new__(QwenDirector)
            director.project_root = Path(tmp)
            users = []
            def fake_chat(system, user, **kwargs):
                users.append((user, kwargs.get("response_format")))
                if len(users) > len(replies):
                    raise AssertionError("Architect exceeded its one-revision limit")
                return replies[len(users) - 1]
            director._chat_text = fake_chat
            with redirect_stdout(StringIO()):
                try:
                    return director._generate_story_blueprint(AI_STORY_MODE, "A premise."), users
                except RuntimeError as exc:
                    return exc, users

        blueprint, users = architect([good])
        _assert(isinstance(blueprint, str) and len(users) == 1,
                "a valid first plan must cost exactly one Architect call")
        _assert(users[0][1] and users[0][1]["json_schema"]["name"] == "story_blueprint",
                "the Architect request must carry the JSON schema")

        broken = json.loads(good)
        broken["scenes"][10]["next_cause"] = ""
        rejected_then_fixed, users = architect([json.dumps(broken), good])
        _assert(isinstance(rejected_then_fixed, str) and len(users) == 2,
                "a returned plan rejected by a hard gate must receive exactly one Architect revision")
        _assert("REVISION:" in users[1][0] and "NEXT_CAUSE" in users[1][0],
                "the Architect revision must be informed by the exact failed handoff contract")

        presence_feedback = QwenDirector._revision_feedback(
            [
                "scene 08 mid-scene entry for 'Marcus Hale' has no character-specific EVENT anchor",
                "scene 08 mid-scene exit for 'Marcus Hale' has no character-specific EVENT anchor",
            ],
            "story plan",
        )
        _assert(
            "Marcus Hale" in presence_feedback
            and "Scene 07 exit_present" in presence_feedback
            and "this scene's exit_present" in presence_feedback
            and "BOTH arrival and later departure" in presence_feedback
            and "Do not invent an ENTRY_PRESENT JSON field" in presence_feedback
            and "Do not remove a valid CAST member" in presence_feedback,
            "mid-scene presence rejection must produce character-specific boundary reconciliation instructions",
        )

        still_rejected, users = architect([json.dumps(broken), json.dumps(broken)])
        _assert(isinstance(still_rejected, RuntimeError) and len(users) == 2,
                "a second rejected plan must fail closed without a third Architect call")

        # An Author draft rejected by a content contract may receive one informed revision.
        author = QwenDirector.__new__(QwenDirector)
        author.project_root = Path(tmp)
        author._story_min_words = 1400
        author._qwen_telemetry = {"calls": []}
        author_calls = []
        def reject_then_accept(mode, user_input, story_system, story_user, **kwargs):
            author_calls.append(story_user)
            author._qwen_telemetry["calls"].append({"call_name": "ai_story_text_pass"})
            if len(author_calls) == 1:
                author._last_story_raw_output = "First completion"
                author._last_story_draft = "Returned draft rejected by a hard quality gate."
                author._last_story_quality_errors = ["scene 11 has no required causal payoff"]
                raise RuntimeError("hard story-contract rejection")
            return "Revised story accepted", True
        author._generate_story_once_impl = reject_then_accept
        with redirect_stdout(StringIO()):
            accepted, generated = author._run_author_with_one_revision(
                AI_STORY_MODE, "premise", "system", "original prompt", temperature=0.7, top_p=0.8,
            )
        _assert(generated and accepted == "Revised story accepted" and len(author_calls) == 2,
                "one returned rejected Author draft must receive exactly one conditional revision")
        _assert("REVISION:" in author_calls[1] and "causal payoff" in author_calls[1],
                "the Author revision must include the exact hard-gate feedback")

        # Infrastructure truncation is not a content revision opportunity.
        truncated = QwenDirector.__new__(QwenDirector)
        truncated._last_story_draft = "partial draft"
        truncation_calls = []
        def fail_truncated(*args, **kwargs):
            truncation_calls.append(1)
            raise StoryTruncated("output ceiling", partial="partial draft")
        truncated._generate_story_once_impl = fail_truncated
        with redirect_stdout(StringIO()):
            try:
                truncated._run_author_with_one_revision(
                    AI_STORY_MODE, "premise", "system", "prompt", temperature=0.7, top_p=0.8,
                )
            except StoryTruncated:
                pass
            else:
                raise AssertionError("Author truncation must be propagated")
        _assert(len(truncation_calls) == 1,
                "a token-ceiling failure must not trigger a content revision")


def test_scene_boundaries_are_derived_and_wording_only_findings_are_advisory():
    import json
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES as N
    from planner.qwen_director import QwenDirector
    from planner.story_blueprint_schema import blueprint_json_schema, render_blueprint_text, resolve_scenes
    from planner.story_contracts import parse_scene_state, split_contract_findings, validate_blueprint_state_contract
    data = _blueprint_json_from_fixture()
    schema_scene = blueprint_json_schema(N)["properties"]["scenes"]["items"]["properties"]
    _assert(not any(key.startswith("entry_") for key in schema_scene),
            "the model must not be asked to write entry state; it is the previous scene's exit")
    # A messy plan: every later scene's exit wording differs and the handoffs share no words with the next event.
    messy = json.loads(json.dumps(data))
    for scene in messy["scenes"][:-1]:
        scene["next_cause"] = "an unrelated consequence nobody mentions again"
        scene["exit_knows"] = "the situation has changed in ways that are phrased differently every time"
    text = render_blueprint_text(messy, N)
    beats = {int(line[6:8]): parse_scene_state(line) for line in text.splitlines() if line.startswith("SCENE ")}
    for number in range(2, N + 1):
        for entry, exit_ in (("ENTRY_LOCATION", "EXIT_LOCATION"), ("ENTRY_PRESENT", "EXIT_PRESENT"),
                             ("ENTRY_KNOWS", "EXIT_KNOWS"), ("ENTRY_OBJECTS", "EXIT_OBJECTS")):
            _assert(beats[number][entry].casefold() == beats[number - 1][exit_].casefold(),
                    f"scene {number:02d} {entry} must equal scene {number - 1:02d} {exit_}")
    normalized = QwenDirector._normalize_story_blueprint(text, N)
    fatal, soft = split_contract_findings(validate_blueprint_state_contract(normalized, N))
    _assert(not any(t in item for item in fatal for t in ("knowledge", "location", "object inventory", "NEXT_CAUSE")),
            f"paraphrase must never be a fatal knowledge/location/object/handoff finding: {fatal!r}")
    _assert(any("NEXT_CAUSE" in item for item in soft), f"the wording mismatch is still reported as a warning: {soft!r}")
    # Anyone present at either boundary is on screen by definition.
    forgetful = json.loads(json.dumps(data))
    forgetful["scenes"][3]["on_screen_present"] = []
    _assert(resolve_scenes(forgetful)[3]["on_screen_present"], "boundary presence must imply on-screen presence")


def test_a_cut_off_draft_is_rewritten_with_pacing_not_expanded_as_if_complete():
    from planner.qwen_director import QwenDirector
    from planner.config import DIRECTOR_STORY_SCENES
    # Three paragraphs per scene: four cut-off scenes look like twelve paragraphs, but they are not a short full story.
    cut = "\n\n".join(
        f"[[SCENE_{i:02d}]] " + "\n\n".join(" ".join(["word"] * 200) for _ in range(3)) for i in range(1, 5)
    )
    _assert(QwenDirector._story_expansion_user(cut) == "", "a draft missing scene markers must not be expanded")
    system = " ".join(QwenDirector.__new__(QwenDirector)._story_text_system("ai_story").split()).lower()
    _assert("soft scene cap" in system and "natural sentence lengths" in system
            and "120-140 words per scene" in system and "190 words" in system,
            "Author must be given a realistic average and 190-word per-scene soft cap so it cannot front-load the story")


def test_story_minimum_is_a_hard_floor_with_one_conditional_revision():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES, STORY_ACCEPT_MIN_WORDS, STORY_MIN_WORDS
    from contextlib import redirect_stdout
    from io import StringIO
    temperature, top_p = QwenDirector.__new__(QwenDirector)._sampling_for_mode(AI_STORY_MODE)
    def run(words_per_scene):
        paragraphs = _story_test_paragraphs(words_per_scene)
        director = QwenDirector.__new__(QwenDirector)
        director._story_min_words = STORY_ACCEPT_MIN_WORDS
        calls = []
        def fake_chat(*args, **kwargs):
            calls.append(dict(kwargs, _user=args[1] if len(args) > 1 else ""))
            return "\n\n".join(paragraphs)
        director._chat_text = fake_chat
        director._validate_story_output_contracts = lambda _mode, _input, candidate, **_kwargs: candidate
        try:
            with redirect_stdout(StringIO()):
                result, generated = director._generate_story_once(
                    AI_STORY_MODE, "premise", "system",
                    "VALIDATED PLOT ARCHITECTURE:\n" + _story_blueprint_fixture(),
                    temperature=temperature, top_p=top_p, source_character_names=[]
                )
            return result, generated, calls, None
        except RuntimeError as exc:
            return None, False, calls, str(exc)
    floor_words = (STORY_MIN_WORDS + DIRECTOR_STORY_SCENES - 1) // DIRECTOR_STORY_SCENES
    accepted_result, accepted_generated, accepted_calls, accepted_error = run(120)
    _assert(STORY_MIN_WORDS == STORY_ACCEPT_MIN_WORDS == 1400,
            f"story floor and acceptance minimum must both be 1,400: {STORY_MIN_WORDS}, {STORY_ACCEPT_MIN_WORDS}")
    _assert(accepted_generated and accepted_error is None and len(accepted_calls) == 1,
            f"story at or above the minimum must pass with one Author call: {accepted_error}")
    _assert(len(re.findall(r"\b[\w'’-]+\b", accepted_result)) >= STORY_MIN_WORDS,
            "accepted story must meet the 1,400-word boundary")
    short_result, short_generated, short_calls, short_error = run(65)
    _assert(not short_generated and short_result is None and len(short_calls) == 2,
            "a returned sub-1,400-word draft must receive at most one conditional Author revision")
    _assert("AUTHOR REVISION" in short_calls[1].get("_user", "")
            and "VALIDATED PLOT ARCHITECTURE" in short_calls[1].get("_user", "")
            and "01=65" in short_calls[1].get("_user", "")
            and "DRAFT:" not in short_calls[1].get("_user", ""),
            "a too-short draft must be rebuilt from the validated plan with targeted scene diagnostics")
    _assert("below the configured minimum" in (short_error or "").lower(),
            f"sub-floor story should report the hard minimum: {short_error}")
    completion_probe = QwenDirector.__new__(QwenDirector)
    completion_probe._story_min_words = STORY_MIN_WORDS
    try:
        completion_probe._validate_story_completion_contract(AI_STORY_MODE, "\n\n".join(_story_test_paragraphs(65)))
    except RuntimeError as exc:
        _assert("below the minimum accepted length" in str(exc).lower(), f"completion contract floor was not enforced: {exc}")
    else:
        raise AssertionError("completion contract must reject a story below 1,400 words")
    _assert(len([x for x in re.split(r"\n\s*\n+", accepted_result) if x.strip()]) == DIRECTOR_STORY_SCENES,
            "accepted story must preserve twelve separate scene paragraphs")
    distribution_probe = QwenDirector.__new__(QwenDirector)
    distribution_probe._story_min_words = STORY_MIN_WORDS
    uneven = _story_test_paragraphs(117)
    # 117 words per scene is a sound 1,404-word story: per-scene lengths above the starvation floor are
    # guidance, so it is accepted (the 1,400 total and the 70-word starvation floor are the hard limits).
    distribution_probe._validate_story_completion_contract(AI_STORY_MODE, "\n\n".join(uneven))


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
    _assert(not any("stock phrase" in error or "stock-phrase" in error for error in errors),
            f"stock phrasing is advisory and must never discard the single Author draft: {errors!r}")
    _assert(any("repeated stock phrases" in warning for warning in director._last_story_contract_warnings),
            f"repeated stock phrasing must still be reported as an advisory: {director._last_story_contract_warnings!r}")
    _assert(any("spoken quote pairs" in error for error in errors),
            f"more than two dialogue quote pairs per scene must fail: {errors!r}")
    _assert(any("longer than 9 words" in error for error in errors),
            f"overlong dialogue must fail: {errors!r}")


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


def test_a_prop_the_plan_declares_is_seeded_but_an_invented_organization_is_not():
    """Regression (Kaggle run): 'Containment Protocol Manual' was declared in KEY_OBJECTS yet flagged as unseeded."""
    from planner.story_contracts import audit_story_against_blueprint
    story = _story_contract_fixture_prose()
    with_prop = story.replace("Ines examines transfer stage 1,",
                              "Ines opens the Containment Protocol Manual. Ines examines transfer stage 1,", 1)
    _assert(with_prop != story, "fixture sentence to extend was not found")
    blueprint = _story_blueprint_fixture()
    undeclared = [i for i in audit_story_against_blueprint(with_prop, blueprint) if "unseeded named project" in i]
    _assert(undeclared, "a protocol manual nobody planned is still an unseeded name")
    declared = re.sub(r"(?m)^(KEY_OBJECTS: .*)$", r"\1, Containment Protocol Manual", blueprint, count=1)
    _assert(declared != blueprint, "fixture must have a KEY_OBJECTS line")
    issues = [i for i in audit_story_against_blueprint(with_prop, declared) if "unseeded named project" in i]
    _assert(not issues, f"a prop declared in KEY_OBJECTS is seeded by the plan: {issues!r}")
    invented = story.replace("Ines examines transfer stage 1,", "Ines reads about Project Hydra. Ines examines transfer stage 1,", 1)
    _assert(any("Project Hydra" in i for i in audit_story_against_blueprint(invented, declared)),
            "an organization the plan never mentions must still be rejected")


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


def test_story_completion_contract_reports_an_unresolved_final_hook_as_a_warning():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, STORY_MIN_WORDS
    director = QwenDirector.__new__(QwenDirector)
    director._story_min_words = STORY_MIN_WORDS
    paragraphs = _story_test_paragraphs(120)
    paragraphs[-1] = paragraphs[-1].replace(". ", ". ", 1) + " I\u2019ll find it. The harbor gate closes and the rescue crew reaches safe ground."
    story = "\n\n".join(paragraphs)
    _assert(len(re.findall(r"\b[\w'\u2019-]+\b", story)) >= 1400, "hook fixture must meet the length floor")
    # A phrase list cannot separate a sequel hook from a resolved ending that mentions the future: warn, do not discard.
    director._validate_story_completion_contract(AI_STORY_MODE, story)
    _assert(any("open future hook" in w for w in director._last_story_quality_warnings),
            f"the ending phrase check must still be reported: {getattr(director, '_last_story_quality_warnings', None)!r}")


def test_a_person_the_architect_named_but_never_put_on_screen_is_seeded():
    """Regression (Kaggle run): 'Dr. Marcus Hale' was dropped from CAST (never on screen), then flagged as unseeded in prose."""
    from planner.story_contracts import audit_story_against_blueprint
    from planner.story_blueprint_schema import render_blueprint_text
    from planner.config import DIRECTOR_STORY_SCENES
    data = _blueprint_json_from_fixture()
    data["cast"].append(dict(data["cast"][0], name="Dr. Marcus Hale", aliases=[]))
    dropped = []
    text = render_blueprint_text(data, DIRECTOR_STORY_SCENES, dropped)
    _assert(dropped == ["Dr. Marcus Hale"], "a never-on-screen person is still removed from CAST")
    _assert("Named but never on screen: Dr. Marcus Hale" in text, "the removed person must stay named in the plan")
    story = _story_contract_fixture_prose().replace(
        "Ines examines transfer stage 1,", "Dr. Hale's journal lay open beside the ledger. Ines examines transfer stage 1,", 1)
    blueprint = _story_blueprint_fixture()
    _assert(any("unseeded named person" in i for i in audit_story_against_blueprint(story, blueprint)),
            "a person nobody planned is still an unseeded name")
    declared = re.sub(r"(?m)^RELATIONSHIPS: .*$", "RELATIONSHIPS: Named but never on screen: Dr. Marcus Hale", blueprint, count=1)
    _assert(declared != blueprint, "fixture must have a RELATIONSHIPS line")
    issues = [i for i in audit_story_against_blueprint(story, declared) if "unseeded named person" in i]
    _assert(not issues, f"a person the Architect named is seeded: {issues!r}")


def test_story_pipeline_uses_two_bounded_stages_and_only_conditional_revisions():
    from planner.config import STORY_ARCHITECT_REVISION_RETRIES, STORY_AUTHOR_REVISION_RETRIES
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    generate_start = source.index("story_blueprint = self._generate_story_blueprint(", source.index("def generate("))
    normal_path_end = source.index("# PASS 1B: deterministic production foundation", generate_start)
    normal_path = source[generate_start:normal_path_end]
    _assert(normal_path.count("self._generate_story_blueprint(") == 1,
            "the top-level path must enter the bounded Architect stage exactly once")
    _assert(normal_path.count("self._generate_story_once(") == 1,
            "the top-level path must enter the bounded Author stage exactly once")
    _assert("story_blueprint=story_blueprint" in normal_path,
            "Author must receive the validated Architect blueprint and CAST line")
    _assert(STORY_ARCHITECT_REVISION_RETRIES == 1 and STORY_AUTHOR_REVISION_RETRIES == 1,
            "production policy must allow at most one conditional revision for each hard-gated stage")
    architect_wrapper = source[source.index("def _generate_story_blueprint("):source.index("def _generate_story_blueprint_attempt(")]
    author_wrapper = source[source.index("def _run_author_with_one_revision("):source.index("@staticmethod\n    def _story_expansion_user")]
    _assert("range(int(STORY_ARCHITECT_REVISION_RETRIES) + 1)" in architect_wrapper
            and "except self._BlueprintRejected" in architect_wrapper,
            "Architect revision must be conditional on a returned blueprint being rejected by its hard gate")
    _assert("STORY_AUTHOR_REVISION_RETRIES" in author_wrapper and "_last_story_draft" in author_wrapper
            and "StoryTruncated" in author_wrapper and "vLLM request failed" in author_wrapper,
            "Author revision must require a returned draft and exclude truncation/transport failures")
    _assert("_story_single_call_system" not in source and "single_call=True" not in source,
            "combined thinking/write call must not remain")
    _assert("_rescue_short_scenes" not in source, "no rescue calls may exist")
    _assert("_repair_story_if_needed(" not in normal_path and "story_repair_pass" not in normal_path,
            "normal path must not perform a separate repair stage")
    _assert("_resample_for_audit(" not in normal_path and "_pairwise_story_judgment(" not in normal_path,
            "normal path must not add challenger or judge calls")
    runtime_source = (ROOT / "planner/qwen_director_runtime.py").read_text(encoding="utf-8")
    _assert("_h3_n = 1" in runtime_source and "H3_STORY_WRITER_SAMPLES" not in runtime_source
            and "_h3_pick_best_story" not in runtime_source,
            "each Author attempt must return one completion, not fan out and rank multiple stories")
    _assert("_revise_story_once" not in source and "story_revision_pass" not in source,
            "legacy unbounded revision paths must remain absent")


def test_story_writer_is_one_qwen3_nonthinking_call_with_configured_ceiling():
    from planner.qwen_director import QwenDirector
    from planner.config import (AI_STORY_MODE, DIRECTOR_STORY_SCENES, DIRECTOR_STORY_MAX_TOKENS,
        DIRECTOR_STORY_WRITER_MAX_TOKENS, DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
        DIRECTOR_STORY_WRITER_TEMPERATURE, DIRECTOR_STORY_WRITER_TOP_P, STORY_MIN_WORDS)
    story = "\n\n".join(_story_test_paragraphs(125))
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
        result, generated = director._generate_story_once(AI_STORY_MODE, "premise", "system", "user",
            temperature=temperature, top_p=top_p, source_character_names=[])
    count = len([p for p in re.split(r"\n\s*\n+", result) if p.strip()])
    _assert(generated and len(calls) == 1, "story must have exactly one prose-writing call")
    _assert(calls[0].get("disable_thinking") is True, "prose writer must disable Qwen3 thinking")
    _assert(calls[0].get("creative") is True, "prose writer must use configured creative sampler")
    _assert(calls[0].get("temperature") == DIRECTOR_STORY_WRITER_TEMPERATURE == 0.7
            and calls[0].get("top_p") == DIRECTOR_STORY_WRITER_TOP_P == 0.8,
            f"Writer sampler changed unexpectedly: {calls[0]}")
    expected = min(DIRECTOR_STORY_MAX_TOKENS, DIRECTOR_STORY_WRITER_MAX_TOKENS)
    _assert(calls[0].get("max_completion") == expected and expected == 3000,
            f"Writer must use the configured 3,000-token completion ceiling: {calls[0]}")
    _assert(DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS == 0,
            "Author must preserve natural EOS; minimum output tokens must remain 0")
    _assert(calls[0].get("minimum_output_tokens") == 0,
            f"Writer must preserve min_tokens=0: {calls[0]}")
    _assert(STORY_MIN_WORDS == 1400, "minimum accepted story length must be 1,400 words")
    _assert(count == DIRECTOR_STORY_SCENES, "story must preserve twelve scene paragraphs")


def test_author_min_tokens_zero_is_omitted_from_vllm_payload():
    """Natural EOS means the Author request must not send a min_tokens field."""
    from planner.qwen_director import QwenDirector

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2, 3]

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "A short fixture."}}]}

    class FakeSession:
        def __init__(self):
            self.payload = None

        def post(self, url, *, json, timeout):
            self.payload = dict(json)
            return FakeResponse()

    director = QwenDirector.__new__(QwenDirector)
    director._vllm_session = FakeSession()
    director._tokenizer = FakeTokenizer()
    director._model_path = None
    director._post_chat(
        messages=[{"role": "user", "content": "Write the story."}],
        temperature=0.7, top_p=0.8, max_tokens=3000,
        min_tokens=0, creative=True, enable_thinking=False, n_samples=1,
    )
    payload = director._vllm_session.payload
    _assert(payload is not None, "the one Author request must be sent")
    _assert(payload.get("max_tokens") == 3000, f"Author max_tokens changed: {payload!r}")
    _assert("min_tokens" not in payload,
            f"min_tokens=0 must be omitted from the vLLM payload, not converted into a positive floor: {payload!r}")


def test_rejected_author_draft_is_saved_for_diagnosis():
    from tempfile import TemporaryDirectory
    from planner.qwen_director import QwenDirector
    with TemporaryDirectory() as tmp:
        director = QwenDirector.__new__(QwenDirector)
        director.project_root = Path(tmp)
        director._story_min_words = 1400
        draft = "Scene 01: A complete but short generated scene."
        raw_draft = "<think>hidden reasoning must not be invented</think>\n" + draft
        processed = "Cleaned candidate: " + draft
        director._save_failed_story_draft(
            raw_draft,
            raw_story=raw_draft,
            processed_story=processed,
            stage="unit_test_rejection",
            reason="Final story has 1251 words, below the configured minimum of 1400.",
            word_count=1251,
        )
        saved = Path(tmp) / "story_author_last_failed.txt"
        _assert(saved.is_file(), "a rejected primary Author draft must be preserved")
        contents = saved.read_text(encoding="utf-8")
        _assert("usable_words: 1251" in contents and raw_draft in contents and processed in contents,
                "the diagnostic artifact must preserve both raw output and the processed candidate")
        _assert("stage: unit_test_rejection" in contents,
                "the diagnostic artifact must identify the rejection stage")
        _assert("architect_revision_limit=1" in contents and "author_revision_limit=1" in contents
                and "minimum_output_tokens=0" in contents and "repair_calls=0" in contents,
                "draft preservation must report the configured conditional-revision policy and min_tokens=0")


def test_author_wrapper_preserves_draft_when_pipeline_rejects_it():
    from tempfile import TemporaryDirectory
    from contextlib import redirect_stdout
    from io import StringIO
    from planner.qwen_director import QwenDirector
    with TemporaryDirectory() as tmp:
        director = QwenDirector.__new__(QwenDirector)
        director.project_root = Path(tmp)
        director._story_min_words = 1400
        director._qwen_telemetry = {"calls": []}
        raw = "Raw Qwen completion with an unfinished tail"
        calls = 0
        def reject(*args, **kwargs):
            nonlocal calls
            calls += 1
            director._qwen_telemetry["calls"].append({"call_name": "ai_story_text_pass"})
            director._last_story_raw_output = raw
            director._last_story_draft = "Processed candidate with complete sentences."
            raise RuntimeError("post-generation quality gate rejected the draft")
        director._generate_story_once_impl = reject
        with redirect_stdout(StringIO()):
            try:
                director._generate_story_once(
                    "ai_story", "premise", "system", "user",
                    temperature=0.7, top_p=0.8, source_character_names=[],
                )
            except RuntimeError as exc:
                _assert("quality gate rejected" in str(exc), "wrapper must re-raise the original rejection")
            else:
                raise AssertionError("author wrapper must not swallow a post-generation rejection")
        saved = Path(tmp) / "story_author_last_failed.txt"
        _assert(saved.is_file(), "outer Author wrapper must preserve rejected output")
        contents = saved.read_text(encoding="utf-8")
        _assert(raw in contents and "Processed candidate with complete sentences." in contents,
                "outer wrapper must preserve raw and processed drafts separately")
        _assert("author_calls=2" in contents and "author_revisions=1" in contents
                and "author_revision_limit=1" in contents and "repair_calls=0" in contents,
                "draft preservation must report the failed primary call plus its single conditional revision")
        _assert(calls == 2, "a returned rejected draft may receive one revision, never a third Author call")


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


# ── Context compaction + camera contract ──────────────────────────────────

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



def test_story_quote_cleanup_preserves_all_dialogue_words():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    paragraphs = [f"Scene {i:02d} proceeds through a distinct decision and visible consequence." for i in range(1, 13)]
    paragraphs[0] = (
        'Mara said, “First short line.” Eli replied, “Second short line.” '
        'Kael answered, “Third line remains in the story even when it exceeds the shot quote allowance.” '
        'The log displayed “ACCESS DENIED”. Mara continued, “Fourth short line remains too.”'
    )
    source = "\n\n".join(paragraphs)
    cleaned = director._normalize_story_quotes(source)
    tokens = lambda value: re.findall(r"[\w'’-]+", value.casefold())
    _assert(tokens(source) == tokens(cleaned),
            "quote cleanup may normalize punctuation but must never delete dialogue or attribution words")
    quote_pairs = re.findall(r'"([^"\n]+)"|“([^”\n]+)”', cleaned.split("\n\n")[0])
    _assert(len(quote_pairs) <= 2,
            f"only two concise lines should remain quoted for shot timing; overflow must be unquoted, not deleted: {quote_pairs!r}")
    _assert("Third line remains in the story" in cleaned and "Fourth short line remains too" in cleaned,
            "overflow dialogue must remain present as text")


def test_duplicate_scene_blocks_are_hard_and_author_revision_rebuilds_from_architecture():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    paragraphs = _story_test_paragraphs(120)
    paragraphs[9] = paragraphs[8]
    paragraphs[10] = paragraphs[8]
    paragraphs[11] = paragraphs[8]
    story = "\n\n".join(paragraphs)
    errors = director._story_hard_quality_errors(story, None, "")
    _assert(any("near-duplicate scene blocks" in error for error in errors),
            f"duplicating complete scene blocks must be a hard Author rejection: {errors!r}")
    revision = QwenDirector._story_expansion_user(
        story, base_user="SOURCE STORY\nVALIDATED PLOT ARCHITECTURE: preserve the approved twelve-scene plan"
    )
    _assert("AUTHOR REVISION" in revision and "near-duplicates" in revision
            and "VALIDATED PLOT ARCHITECTURE" in revision and "Target 1560 words" in revision
            and "DRAFT:" not in revision,
            "duplicate/short story revision must target the validated architecture rather than copy defective prose")


def test_blueprint_story_structure_failures_are_hard_and_feedback_is_actionable():
    from planner.story_contracts import split_contract_findings
    from planner.qwen_director import QwenDirector
    findings = [
        "SETUP_PAYOFF_MAP plant in scene 03 is not grounded in that scene's EVENT",
        "SETUP_PAYOFF_MAP payoff in scene 08 is not grounded in that scene's EVENT",
        "scene 01 changes its physical location without an explicit transition ('main hall' to 'vault chamber')",
        "scene 04 object state change is not anchored to the named object and a physical action",
    ]
    fatal, warnings = split_contract_findings(findings)
    _assert(len(fatal) == 4 and not warnings,
            f"ungrounded plants/payoffs, unexplained movement and object-state changes must reject the blueprint: {fatal!r}")
    feedback = QwenDirector._revision_feedback(findings, "story plan")
    for phrase in ("SETUP/PAYOFF PLANT", "SETUP/PAYOFF PAYOFF", "LOCATION CONTINUITY", "OBJECT STATE"):
        _assert(phrase in feedback, f"Architect revision must provide field-level repair instructions for {phrase}")


def _architect_repair_fixture():
    cast = (
        "CAST: Elena Voss (kind: person; role: polar systems engineer; wants: recover reactor data; "
        "pressure: storm damage; voice: precise; aliases: Elena; "
        "on-screen scenes: 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12; fate: survives)"
    )
    lines = [
        cast,
        "LOGLINE: An engineer must retrieve reactor evidence from an abandoned Arctic station.",
        "CLIMAX_CHOICE: Elena uses the manual override although it risks destroying the station.",
        "RESOLUTION: Elena stabilizes the reactor data and sacrifices the station to prevent a larger disaster.",
        "SETUP_PAYOFF_MAP: PLANT Scene 02: The sealed vault access panel has a pressure lock and manual override; PAYOFF Scene 09: Elena uses the manual override code to depressurize the vault. | PLANT Scene 04: The control-room panel contains the security officer's manual override code; PAYOFF Scene 10: Elena enters the officer's PIN into the control panel to trigger the failsafe.",
        "SCENE 02: PURPOSE=Elena discovers the override; ENTRY_OBJECTS=none; EVENT=Elena finds a control panel with a manual override function, but it requires a security officer's PIN to activate.; EXIT_OBJECTS=Control Panel (activated); NEXT_CAUSE=Elena must find the security officer's log.",
        "SCENE 03: PURPOSE=Elena finds the log; ENTRY_LOCATION=Abandoned Arctic Research Station, control room; ENTRY_OBJECTS=Control Panel (activated); EVENT=Elena navigates through a hidden maintenance corridor to the archive room, where she finds the security officer's final log.; EXIT_LOCATION=Abandoned Arctic Research Station, archive room; EXIT_OBJECTS=Security Officer's Log (discovered); NEXT_CAUSE=Elena must return to the control room.",
        "SCENE 04: PURPOSE=Elena activates the override; ENTRY_LOCATION=Abandoned Arctic Research Station, archive room; EVENT=Elena returns to the control room, enters the officer's PIN into the control panel, and activates the manual override.; EXIT_LOCATION=Abandoned Arctic Research Station, control room; NEXT_CAUSE=Elena must reach the hidden terminal.",
        "SCENE 08: PURPOSE=Elena stabilizes the reactor; ENTRY_OBJECTS=Geothermal Reactor Data Core (retrieved); EVENT=Elena uses the geothermal reactor data core to stabilize the reactor.; EXIT_OBJECTS=Geothermal Reactor Data Core (stabilized); NEXT_CAUSE=The station collapses and Elena must survive the storm.",
        "SCENE 09: PURPOSE=Elena survives; EVENT=Elena shelters from the storm and waits for rescue.; NEXT_CAUSE=The rescue team must locate Elena.",
        "SCENE 10: PURPOSE=Elena is rescued; EVENT=Elena is rescued by a search team.; NEXT_CAUSE=none",
        "SCENE 11: PURPOSE=Elena reaches safety; ENTRY_LOCATION=Abandoned Arctic Research Station, exterior; EVENT=Elena is transported to a recovery facility.; EXIT_LOCATION=Recovery Facility, Arctic; NEXT_CAUSE=none",
        "SCENE 12: PURPOSE=The story resolves; EVENT=Elena receives confirmation that the reactor data prevented a larger disaster.; NEXT_CAUSE=none",
    ]
    return "\n".join(lines)


def test_architect_repair_recognizes_logged_location_and_object_verbs():
    from planner.story_contracts import _scene_local_transition_is_explained, OBJECT_CHANGE_RE
    cast = [{"name": "Elena Voss", "aliases": ["Elena"]}]
    _assert(_scene_local_transition_is_explained(
        "Abandoned Arctic Research Station, control room",
        "Abandoned Arctic Research Station, archive room",
        "Elena navigates through a hidden maintenance corridor to the archive room.", cast,
    ), "navigates + named destination should ground the logged control-room to archive-room move")
    _assert(_scene_local_transition_is_explained(
        "Abandoned Arctic Research Station, archive room",
        "Abandoned Arctic Research Station, control room",
        "Elena returns to the control room and enters the officer's PIN.", cast,
    ), "returns + control-room anchor should ground the logged return transition")
    _assert(_scene_local_transition_is_explained(
        "Abandoned Arctic Research Station, exterior", "Recovery Facility, Arctic",
        "Elena is transported to a recovery facility.", cast,
    ), "transported + recovery-facility anchor should ground the logged transport")
    _assert(not _scene_local_transition_is_explained(
        "control room", "archive room", "Elena repairs the control panel and reads the log.", cast,
    ), "an unrelated event must not satisfy a declared location transition")
    _assert(OBJECT_CHANGE_RE.search("Elena finds the security officer's final log."),
            "finding a named object must count as an observable acquisition/discovery action")
    _assert(OBJECT_CHANGE_RE.search("Elena uses the geothermal reactor data core to stabilize the reactor."),
            "using a named data core to stabilize the reactor must count as a physical state action")
    _assert(not OBJECT_CHANGE_RE.search("Elena considers the consequences of the decision."),
            "an event with no object-action verb must not validate an object-state change")


def test_architect_revision_context_preserves_the_failed_plan_and_compacts_findings():
    from planner.qwen_director import QwenDirector
    blueprint = _architect_repair_fixture()
    problems = [
        "SETUP_PAYOFF_MAP payoff scene 09 does not explicitly refer back to the planted evidence from scene 02",
        "SETUP_PAYOFF_MAP payoff in scene 10 is not grounded in that scene's EVENT",
        "SETUP_PAYOFF_MAP payoff scene 10 does not explicitly refer back to the planted evidence from scene 04",
        "scene 03 changes its physical location without an explicit transition ('control room' to 'archive room')",
        "scene 04 changes its physical location without an explicit transition ('archive room' to 'control room')",
        "scene 03 object state change is not anchored to the named object and a physical action",
        "scene 08 object state change is not anchored to the named object and a physical action",
        "scene 10 has no NEXT_CAUSE handoff",
        "scene 11 has no NEXT_CAUSE handoff",
    ]
    context = QwenDirector._story_blueprint_revision_context(blueprint, problems)
    original_cast = next(line for line in blueprint.splitlines() if line.startswith("CAST:"))
    _assert("IDENTITY LOCK:" in context and original_cast in context,
            "revision context must preserve the exact canonical CAST line")
    _assert("FULL ORIGINAL BLUEPRINT" in context and "SCENE 02:" in context and "SCENE 10:" in context,
            "revision context must contain the full rejected plan, not only selected excerpts")
    _assert(context.index("PLANT Scene 02") < context.index("PAYOFF Scene 09"),
            "the plant scene must not be reversed with the payoff scene in revision context")
    _assert(context.index("PLANT Scene 04") < context.index("PAYOFF Scene 10"),
            "the second plant/payoff pair must preserve its scene order")
    _assert("Control Panel (activated)" in context and "navigates through a hidden maintenance corridor" in context,
            "revision context must preserve object states and the exact movement event")
    _assert("NEXT_CAUSE=none" in context,
            "revision context must retain the missing handoff location")
    _assert(len(context) < 10000,
            f"full-plan revision context must remain bounded: {len(context)} chars")
    warnings = [f"scene {n} event is not anchored to the previous scene's NEXT_CAUSE" for n in (10, 11, 12)]
    feedback = QwenDirector._revision_feedback(
        problems, "story plan", previous_plan_context=context, advisory_findings=warnings,
    )
    for phrase in ("SETUP/PAYOFF", "LOCATION CONTINUITY", "OBJECT STATE", "NEXT_CAUSE", "FULL ORIGINAL BLUEPRINT"):
        _assert(phrase in feedback, f"revised Architect feedback must preserve {phrase} repair guidance")
    _assert(len(feedback) < 13000,
            f"revision feedback must stay bounded despite carrying the full original plan: {len(feedback)} chars")


def test_architect_revision_retry_receives_the_actual_rejected_blueprint():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    director._last_story_quality_errors = [
        "scene 10 has no NEXT_CAUSE handoff",
        "scene 11 has no NEXT_CAUSE handoff",
    ]
    director._last_story_blueprint_warnings = []
    candidate = _architect_repair_fixture()
    calls = []
    def fake_attempt(mode, user_input, *, source_character_names=None, feedback="", revision_base=None,
                     revision_problems=None, revision_advisories=None):
        calls.append({
            "feedback": feedback,
            "revision_base": revision_base,
            "revision_problems": revision_problems,
            "revision_advisories": revision_advisories,
        })
        if len(calls) == 1:
            raise QwenDirector._BlueprintRejected("test rejection", blueprint=candidate)
        return "repaired blueprint"
    director._generate_story_blueprint_attempt = fake_attempt
    result = QwenDirector._generate_story_blueprint(director, "ai_story", "A polar engineer finds a vault.")
    _assert(result == "repaired blueprint" and len(calls) == 2,
            "the Architect should retain one conditional revision and no extra call")
    _assert("FULL ORIGINAL BLUEPRINT" in calls[1]["feedback"] and "Elena Voss" in calls[1]["feedback"]
            and "SCENE 10:" in calls[1]["feedback"] and "SCENE 11:" in calls[1]["feedback"],
            "the revision request must include the complete exact rejected blueprint")
    _assert(calls[1]["revision_base"] == candidate and calls[1]["revision_problems"] == director._last_story_quality_errors,
            "the retry must also pass the rejected blueprint and its scoped failures to the deterministic merger")


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


def test_environmental_opposition_is_not_cast_and_boundary_presence_is_reconciled():
    from planner.qwen_director import QwenDirector
    from planner.story_contracts import (
        _event_explicitly_assigns_death, is_non_person_cast_entry, parse_cast,
        parse_scene_state, split_contract_findings, validate_blueprint_state_contract,
    )

    director = QwenDirector.__new__(QwenDirector)
    lines = _story_blueprint_fixture().splitlines()
    lines[0] += (
        " | Dr. Kael (kind: person; role: station biologist; wants to escape the storm; pressure guilt over abandoning his team; "
        "voice terse; aliases: none; on-screen scenes: 02, 04, 08; fate: dies in scene 08)"
        " | Arctic Storm (opposition; wants to destroy all life; pressure none; voice: none; "
        "aliases: none; on-screen scenes: all; fate: none)"
    )
    for index, line in enumerate(lines):
        if line.startswith("OPPOSITION:"):
            lines[index] = "OPPOSITION: Arctic Storm and failing station infrastructure threaten the sealed vault."
        elif line.startswith("SCENE 02:"):
            line = line.replace(
                "EVENT=Ines examines",
                "EVENT=Kael arrives at the harbor archive and meets Ines while Ines examines",
                1,
            )
            line = line.replace("EXIT_PRESENT=Ines Vale;", "EXIT_PRESENT=Ines Vale, Kael;", 1)
            lines[index] = line
        elif line.startswith("SCENE 03:"):
            line = line.replace("ENTRY_PRESENT=Ines Vale;", "ENTRY_PRESENT=Ines Vale, Kael;", 1)
            line = line.replace("EVENT=Ines examines", "EVENT=Kael leaves the harbor archive while Ines examines", 1)
            lines[index] = line
        elif line.startswith("SCENE 08:"):
            line = line.replace(
                "EVENT=Ines examines",
                "EVENT=Kael emerges from the lower archive, helps Ines secure the ledger, then dies when the collapsing ceiling strikes him; Ines examines",
                1,
            )
            lines[index] = line

    normalized = director._normalize_story_blueprint("\n".join(lines), 12)
    cast = {item["name"]: item for item in parse_cast(normalized)}
    _assert(set(cast) == {"Ines Vale", "Kael"}, f"environmental opposition must not be a physical cast member: {sorted(cast)!r}")
    _assert("Arctic Storm" not in next(line for line in normalized.splitlines() if line.startswith("CAST:")),
            "the exact legacy generic-opposition Arctic Storm entry must be removed from canonical CAST")
    _assert(is_non_person_cast_entry("Arctic Storm", "kind: environmental force; role: opposition; voice: none", {"OPPOSITION": "Arctic Storm approaches"}),
            "explicit environmental-force types must be excluded from physical CAST")
    _assert(not is_non_person_cast_entry("Arctic Storm", "kind: person; role: opposition; voice: none", {"OPPOSITION": "Arctic Storm approaches"}),
            "explicit person type must override a generic opposition role")
    _assert(not _event_explicitly_assigns_death("Elara Voss kills Kael, then escapes.", {"name": "Elara Voss", "aliases": []}),
            "the killer must not be misclassified as the victim")
    _assert(_event_explicitly_assigns_death("Elara Voss kills Kael, then escapes.", {"name": "Kael", "aliases": []}),
            "the named victim of a transitive death predicate must be identified")
    scene02 = parse_scene_state(next(line for line in normalized.splitlines() if line.startswith("SCENE 02:")))
    scene03 = parse_scene_state(next(line for line in normalized.splitlines() if line.startswith("SCENE 03:")))
    scene08 = parse_scene_state(next(line for line in normalized.splitlines() if line.startswith("SCENE 08:")))
    _assert("Kael" in scene02["ON_SCREEN_PRESENT"], "an EXIT_PRESENT person must be included in full-scene visibility")
    _assert("Kael" in scene03["ON_SCREEN_PRESENT"], "an ENTRY_PRESENT person must be included in full-scene visibility")
    _assert("Kael" in scene08["ON_SCREEN_PRESENT"], "a character declared dead in this scene must appear in its full-scene visibility")
    _assert(cast["Kael"]["scenes"] == [2, 3, 8], f"CAST scene inventory must follow reconciled physical visibility: {cast['Kael']!r}")
    findings = validate_blueprint_state_contract(normalized, 12)
    fatal, _warnings = split_contract_findings(findings)
    _assert(not fatal, f"reconciled weather opposition and physical boundary state must not be fatal: {fatal!r}")



def test_prose_presence_audit_accepts_common_departure_and_behavior_cues():
    from planner.story_contracts import TRAVEL_RE, ACTION_VERBS
    _assert(TRAVEL_RE.search("Kael disappeared down the corridor toward the power grid room.") is not None,
            "common prose departure 'disappeared' must be recognized")
    _assert(re.search(r"\b(?:" + ACTION_VERBS + r")\b", "Elara hesitated, then nodded.", re.I) is not None,
            "observable hesitation must count as a character behavior cue")

def test_duplicate_adjacent_scene_block_is_cleaned_without_retry_and_isolated_repetition_is_advisory():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    director._last_story_contract_warnings = []
    director._last_story_quality_warnings = []
    paragraphs = _story_test_paragraphs(125)
    copied = (
        "The emergency exit remained blocked by frozen debris across the station. "
        "Elara turned back toward the broken access corridor and checked the vault. "
        "The sealed chamber held the pathogen behind its final pressure lock. "
        "The storm had passed, leaving the station silent beneath the snow. "
        "She understood that the choice had cost her any chance of returning home."
    )
    paragraphs[9] += " " + copied
    paragraphs[10] = "Elara reached the emergency exit. " + copied + " The control lights faded beyond the sealed door."
    isolated = "The white station lights flickered above the empty harbor archive."
    paragraphs[0] += " " + isolated
    paragraphs[1] += " " + isolated
    source = "\n\n".join(paragraphs)
    cleaned, events = director._remove_repeated_adjacent_scene_block(source, minimum_words=1400)
    count = len(re.findall(r"\b[\w'’-]+\b", cleaned))
    scenes = [part for part in re.split(r"\n\s*\n+", cleaned) if part.strip()]
    _assert(len(scenes) == 12, "duplicate-block cleanup must preserve all twelve scene paragraphs")
    _assert(count >= 1400, f"duplicate-block cleanup must preserve the hard story floor, got {count}")
    _assert(any("5 sentences" in event and "scene 11" in event for event in events),
            f"the exact copied five-sentence block should be removed deterministically: {events!r}")
    _assert(copied not in scenes[10], "the later scene must not retain the copied passage")
    errors = director._story_hard_quality_errors(cleaned, None, "")
    _assert(not errors, f"an isolated exact sentence repetition must not fail the one-pass draft: {errors!r}")
    _assert(any("exact sentence repeated" in item for item in director._last_story_quality_warnings),
            f"an isolated repeat should remain visible as an advisory: {director._last_story_quality_warnings!r}")


def test_story_run_diagnostics_and_rejected_draft_are_in_trace_bundle():
    from tempfile import TemporaryDirectory
    import json
    from planner.qwen_director import QwenDirector

    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "project"
        trace = Path(tmp) / "qwen_director_traces"
        root.mkdir()
        trace.mkdir()
        director = QwenDirector.__new__(QwenDirector)
        director.project_root = root
        director._trace_dir = trace
        director._story_min_words = 1400
        director._last_story_word_counts = {"after_raw_prose_normalization": 1531, "after_cleanup": 1479}
        director._last_story_scene_word_counts = {
            "scene_01": 192, "scene_02": 146, "scene_03": 128, "scene_04": 112,
            "scene_05": 101, "scene_06": 109, "scene_07": 110, "scene_08": 95,
            "scene_09": 90, "scene_10": 75, "scene_11": 76, "scene_12": 76,
        }
        director._last_story_blueprint_warnings = [f"scene {index:02d} knowledge transition needs review" for index in range(1, 23)]
        director._last_story_contract_warnings = ["cast member not named in one planned scene"]
        director._last_story_quality_warnings = ["exact sentence repeated in scenes 10 and 11"]
        director._last_story_quality_errors = ["synthetic hard-gate failure"]
        director._last_story_cleanup_events = ["removed exact copied block of 5 sentences/64 words from scene 11"]
        director._qwen_telemetry = {"calls": [
            {"call_name": "ai_story_architecture_pass"},
            {"call_name": "ai_story_architecture_pass"},
            {"call_name": "ai_story_text_pass"},
            {"call_name": "character_entity_extraction"},
            {"call_name": "shot_batch:scene_001_scene_002"},
        ]}
        author_trace = trace / "ai_story_text_pass_test.json"
        author_trace.write_text(json.dumps({"call_name": "ai_story_text_pass", "raw_content": "draft"}), encoding="utf-8")
        director._save_failed_story_draft(
            "raw model draft", raw_story="raw model draft", processed_story="cleaned model draft",
            reason="synthetic hard-gate failure", word_count=1479, stage="author_post_generation_rejection",
        )
        director._save_story_run_diagnostics(
            status="failed", stage="ai_story_author_validation", reason="synthetic hard-gate failure", word_count=1479,
        )
        report_path = trace / "story_run_diagnostics.json"
        text_path = trace / "story_run_diagnostics.txt"
        failed_draft = trace / "story_author_last_failed.txt"
        _assert(report_path.is_file() and text_path.is_file() and failed_draft.is_file(),
                "trace bundle must include the full diagnostics in JSON/TXT and the rejected draft")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        attached = json.loads(author_trace.read_text(encoding="utf-8"))["post_generation_validation"]
        _assert(report["required_scene_minimum_words"] == 70
                and report["scene_guidance_range_words"] == [120, 140]
                and report["scenes_at_least_120_words"] == 3
                and "required_scenes_at_least_120_words" not in report,
                "diagnostics must distinguish the actual 70-word hard floor from the 120-140-word guidance and report observed counts")
        _assert(report["blueprint_warnings"] == director._last_story_blueprint_warnings,
                "diagnostics must preserve every blueprint warning")
        _assert(report["prose_audit_warnings"] == director._last_story_contract_warnings,
                "diagnostics must preserve every prose-audit warning")
        _assert(report["quality_warnings"] == director._last_story_quality_warnings,
                "diagnostics must preserve all quality warnings, even on rejection")
        _assert(len(report["blueprint_warnings"]) == 22,
                "diagnostics must preserve the complete warning list, not only console previews")
        policy = report["generation_policy"]
        _assert(policy["architect_calls"] == 2 and policy["author_calls"] == 1,
                f"diagnostics must count actual story model calls, not hard-code one each: {policy!r}")
        _assert(policy["architect_revisions"] == 1 and policy["author_revisions"] == 0
                and policy["architect_revision_limit"] == 1 and policy["author_revision_limit"] == 1,
                f"diagnostics must report actual revisions separately from configured limits: {policy!r}")
        _assert(policy["architect_max_tokens"] == 5400 and policy["author_max_tokens"] == 3000
                and policy["author_minimum_output_tokens"] == 0
                and policy["total_story_revisions"] == 1 and policy["runtime_retry_calls"] == 0
                and policy["repair_calls"] == 0,
                f"diagnostics must retain the token contract and distinguish revisions from infrastructure retries: {policy!r}")
        _assert(policy["architect_call_sequence"] == ["ai_story_architecture_pass"] * 2
                and policy["author_call_sequence"] == ["ai_story_text_pass"],
                f"diagnostics must retain the exact story generation call sequence: {policy!r}")
        _assert("scene 22 knowledge transition needs review" in text_path.read_text(encoding="utf-8"),
                "human-readable trace log must retain the final warning, not truncate the list")
        diagnostic_text = text_path.read_text(encoding="utf-8")
        _assert("architect_calls=2; author_calls=1; architect_revisions=1; author_revisions=0; architect_revision_limit=1; author_revision_limit=1; architect_max_tokens=5400; author_max_tokens=3000; author_minimum_output_tokens=0; total_story_revisions=1; runtime_retry_calls=0; repair_calls=0" in diagnostic_text,
                "diagnostic text must accurately state actual calls, revisions, limits, and token policy")
        _assert(report["rejected_draft_artifact"] == "story_author_last_failed.txt",
                "diagnostics must point to the preserved rejected first draft")
        _assert(attached["cleanup_events"] == director._last_story_cleanup_events,
                "the Author trace JSON must carry deterministic cleanup and validation details")
        _assert("[BLUEPRINT WARNINGS]" in json.loads(author_trace.read_text(encoding="utf-8"))["post_generation_validation_text"],
                "the Author trace JSON must embed the readable complete warning log for JSON-only trace ZIPs")
        _assert("cleaned model draft" in json.loads(author_trace.read_text(encoding="utf-8"))["rejected_story_artifact"],
                "the Author trace JSON must embed the processed rejected draft for JSON-only trace ZIPs")
        _assert("cleaned model draft" in failed_draft.read_text(encoding="utf-8"),
                "the rejected processed story must be available as a separate trace artifact too")
        director._save_story_run_diagnostics(
            status="passed", stage="ai_story_author_validation", word_count=1624,
        )
        recovered = json.loads(report_path.read_text(encoding="utf-8"))
        _assert(recovered["recovered_after_architect_revision"] is True
                and recovered["recovered_after_author_revision"] is False,
                "final diagnostics must explicitly show the Architect recovered from its first rejected plan")
        director._qwen_telemetry["calls"].append({"call_name": "ai_story_text_pass"})
        director._save_story_run_diagnostics(
            status="passed", stage="ai_story_author_validation", word_count=1624,
        )
        author_recovered = json.loads(report_path.read_text(encoding="utf-8"))
        _assert(author_recovered["recovered_after_author_revision"] is True,
                "a passing Author run after one rejected draft must explicitly record its recovery")

        # A subsequent fresh Architect run must not attach the previous Author failure to its log.
        from planner.config import AI_STORY_MODE
        (root / "story_author_last_failed.txt").write_text("STALE AUTHOR FAILURE", encoding="utf-8")
        (root / "story_architect_last_failure.txt").write_text("STALE ARCHITECT FAILURE", encoding="utf-8")
        (trace / "story_author_last_failed.txt").write_text("STALE TRACE FAILURE", encoding="utf-8")
        director._last_story_blueprint_warnings = ["stale blueprint warning"]
        director._last_story_contract_warnings = ["stale prose warning"]
        director._last_story_quality_warnings = ["stale quality warning"]
        director._last_story_quality_errors = ["stale quality error"]
        director._last_story_cleanup_events = ["stale cleanup"]
        director._last_story_word_counts = {"stale": 1}
        director._qwen_telemetry = {"calls": []}  # a new generate() session resets per-run telemetry
        director._validate_story_blueprint_contract(AI_STORY_MODE, _story_blueprint_fixture(), "new premise")
        fresh = json.loads(report_path.read_text(encoding="utf-8"))
        _assert(not (root / "story_author_last_failed.txt").exists()
                and not (root / "story_architect_last_failure.txt").exists()
                and not (trace / "story_author_last_failed.txt").exists(),
                "a fresh Architect pass must remove stale failure artifacts")
        _assert(fresh["generation_policy"]["architect_calls"] == 0
                and fresh["generation_policy"]["author_calls"] == 0,
                "a fresh Architect-stage report must count only this new telemetry session")
        _assert(not fresh["prose_audit_warnings"] and not fresh["quality_errors"],
                "a new story run must not inherit previous run's warnings or errors")


# ── Runner ────────────────────────────────────────────────────────────────

def test_director_checkpoint_is_accepted_by_orchestrator_resume():
    import tempfile
    from planner.qwen_director import QwenDirector
    from pipeline.production_checkpoint import ProductionCheckpoint
    from pipeline.production_orchestrator import ProductionOrchestrator
    with tempfile.TemporaryDirectory() as tmp:
        store = ProductionCheckpoint(tmp)
        director = QwenDirector.__new__(QwenDirector)
        director.project_root = ROOT
        plan = {"shots": [{"shot_id": "S01_01", "prompt": "a wide harbor"}]}
        state = director._checkpoint_state(
            "resume-test", "ai_story", "a short story", {"scenes": []}, plan,
            "running", "shots", ["S01"], "S01", checkpoint_store=store,
        )
        _assert(state.get("plan_sha256") and state["plan_sha256"] == store.plan_digest(plan),
                "Director checkpoints must carry the plan digest")
        _assert(state["director_sha256"] == ProductionOrchestrator._director_fingerprint(ROOT, store),
                "Director and Orchestrator must use the same director fingerprint")
        _assert("planner/story_contracts.py" in ProductionCheckpoint.DIRECTOR_FINGERPRINT_FILES,
                "the fingerprint must cover planner/story_contracts.py")
        store.save("resume-test", state)
        orchestrator = ProductionOrchestrator.__new__(ProductionOrchestrator)
        orchestrator.project_root = ROOT
        orchestrator._checkpoint_store = lambda: store
        loaded = orchestrator._load_resume_state("resume-test", "ai_story", "a short story", "auto", "base")
        _assert(loaded.get("completed_scene_ids") == ["S01"], "resume must return the Director checkpoint state")


def test_critic_cannot_break_cinematic_coverage():
    from pipeline.production_orchestrator import ProductionOrchestrator
    plan = {"shots": [
        {"shot_id": "S01_01", "scene_id": "S01", "camera_shot": "wide", "action": "a"},
        {"shot_id": "S01_02", "scene_id": "S01", "camera_shot": "medium", "action": "b"},
        {"shot_id": "S01_03", "scene_id": "S01", "camera_shot": "close-up", "action": "c"},
        {"shot_id": "S02_01", "scene_id": "S02", "camera_shot": "wide", "action": "d"},
        {"shot_id": "S02_02", "scene_id": "S02", "camera_shot": "medium", "action": "e"},
        {"shot_id": "S02_03", "scene_id": "S02", "camera_shot": "close-up", "action": "f"},
    ]}
    critique = {"shot_patches": [
        {"shot_id": "S01_03", "patch": {"camera_shot": "medium", "mood": "tense"}},
        {"shot_id": "S02_03", "patch": {"camera_shot": "extreme close-up"}},
    ]}
    result = ProductionOrchestrator._apply_director_critic_patches(plan, critique)
    by_id = {s["shot_id"]: s for s in result["shots"]}
    _assert(by_id["S01_03"]["camera_shot"] == "close-up", "a critic patch that leaves a scene with two framings must be reverted")
    _assert(by_id["S01_03"].get("mood") == "tense", "non-framing critic patches must still apply")
    _assert(by_id["S02_03"]["camera_shot"] == "extreme close-up", "a framing patch that keeps three distinct framings is allowed")
    applied = {(i["shot_id"], i["field"]) for i in result["director_critique_applied"]}
    _assert(("S01_03", "camera_shot") not in applied and ("S02_03", "camera_shot") in applied, "applied-patch ledger must match what was kept")


def test_story_contract_regressions_found_by_audit():
    from planner.story_contracts import (
        ACTION_VERBS, audit_story_against_blueprint, validate_blueprint_state_contract,
        parse_cast, reconcile_cast_scenes, split_contract_findings,
    )

    ranked = _story_blueprint_fixture().replace(
        "CAST: Ines Vale (", "CAST: Lieutenant Ines Vale (", 1
    )
    _assert(parse_cast(ranked)[0]["name"] == "Ines Vale",
            "military ranks must normalize out of canonical character names")

    # An empty visibility ledger must NOT silently zero out the CAST scene list;
    # the CAST/ledger mismatch surfaces per scene in validate_blueprint_state_contract.
    no_presence = _story_blueprint_fixture().replace(
        "ON_SCREEN_PRESENT=Ines Vale", "ON_SCREEN_PRESENT=none"
    ).replace(
        "ENTRY_PRESENT=Ines Vale", "ENTRY_PRESENT=none"
    ).replace(
        "EXIT_PRESENT=Ines Vale", "EXIT_PRESENT=none"
    )
    reconciled_cast = parse_cast(reconcile_cast_scenes(no_presence))[0]
    _assert(reconciled_cast["scenes"] == list(range(1, 13)),
            "CAST reconciliation must preserve declared scenes when the ledger has no appearances")
    issues = validate_blueprint_state_contract(no_presence)
    _assert(any("CAST appearance for 'Ines Vale' is absent from ON_SCREEN_PRESENT" in i for i in issues),
            f"an empty ledger with declared CAST scenes must fail the state contract: {issues!r}")

    # Cross-scene: a new person in ENTRY_PRESENT must be named in that scene's EVENT.
    issues = validate_blueprint_state_contract(_two_person_blueprint(arrival=None))
    _assert(any("without a character-specific arrival EVENT anchor" in i for i in issues),
            f"an unexplained new person must still be rejected: {issues!r}")
    fatal, _warnings = split_contract_findings(issues)
    _assert(any("Mara Quill" in i for i in fatal),
            f"a missing character-specific arrival must be a blocking plan-contract finding: {fatal!r}")

    _assert(not validate_blueprint_state_contract(_two_person_blueprint()),
            "a new character named in the arriving scene's EVENT must satisfy the arrival anchor")
    for verb in ("joins", "is introduced to", "greets"):
        text = _two_person_blueprint(arrival=verb)
        issues = [i for i in validate_blueprint_state_contract(text) if "arrival EVENT anchor" in i]
        _assert(not issues, f"'{verb}' with the person named in EVENT must satisfy the anchor: {issues!r}")

    # Mid-scene: a person on screen but absent at the entry boundary must be named in EVENT.
    def _mid_scene_blueprint(event: str) -> str:
        lines = _story_blueprint_fixture().splitlines()
        lines[0] += (
            " | Mara Quill (harbor clerk; wants to preserve the duplicate ledger; pressure fears the audit; "
            "voice curt; aliases: none; on-screen scenes: 02; fate: survives)"
        )
        scene_two = next(i for i, line in enumerate(lines) if line.startswith("SCENE 02:"))
        lines[scene_two] = lines[scene_two].replace(
            "ON_SCREEN_PRESENT=Ines Vale;", "ON_SCREEN_PRESENT=Ines Vale, Mara Quill;"
        )
        lines[scene_two] = re.sub(
            r"EVENT=.*?;\s*EXIT_LOCATION=",
            f"EVENT={event}; EXIT_LOCATION=",
            lines[scene_two],
            count=1,
        )
        return "\n".join(lines)

    valid_mid = _mid_scene_blueprint(
        "Mara Quill checks the corridor and then slips through the side passage after joining Ines at the ledger table"
    )
    issues = validate_blueprint_state_contract(valid_mid)
    _assert(not any("mid-scene entry" in i or "mid-scene exit" in i for i in issues),
            f"a named mid-scene entrant should pass: {issues!r}")
    unexplained_entry = _mid_scene_blueprint(
        "Ines examines the transfer stage and records the discrepancy before the clerk can remove the evidence"
    )
    issues = validate_blueprint_state_contract(unexplained_entry)
    _assert(any("mid-scene entry for 'Mara Quill'" in i for i in issues),
            f"an unnamed mid-scene entrant must fail: {issues!r}")

    # CAST presence is enforced in both directions (blueprint state ledger and prose).
    issues = validate_blueprint_state_contract(_two_person_blueprint(mara_scenes="02"))
    _assert(any("on-screen scene list omits it" in i for i in issues),
            f"state ledger presence outside the CAST scene list must fail: {issues!r}")

    paragraphs = _story_contract_fixture_prose().split("\n\n")
    for index in (1, 3):
        paragraphs[index] += " Mara Quill waits by the door."
    issues = audit_story_against_blueprint("\n\n".join(paragraphs), _two_person_blueprint(mara_scenes="02"))
    _assert(any("Mara Quill" in i and "scene 04" in i and "omits" in i for i in issues),
            f"prose activity outside the CAST scene list must fail: {issues!r}")

    # Possessive physical activity after death is caught; a possessive object is not.
    dead = _two_person_blueprint(mara_scenes="02, 03, 04, 05, 06", mara_fate="dies in scene 06")
    paragraphs = _story_contract_fixture_prose().split("\n\n")
    for index in range(1, 6):
        paragraphs[index] += " Mara Quill waits by the door."
    paragraphs[7] += " Mara Quill's fingers close around the brass latch."
    issues = audit_story_against_blueprint("\n\n".join(paragraphs), dead)
    _assert(any("dead character 'Mara Quill'" in i for i in issues),
            f"possessive post-death activity missed: {issues!r}")
    paragraphs[7] = paragraphs[7].replace(
        "Mara Quill's fingers close around the brass latch.",
        "Mara Quill's ledger stays on the shelf.",
    )
    issues = audit_story_against_blueprint("\n\n".join(paragraphs), dead)
    _assert(not any("dead character" in i for i in issues),
            f"a possessive object must not read as activity: {issues!r}")

    # Ordinary sentence-initial capitalized words are not people; unrostered names still are.
    base = _story_contract_fixture_prose()
    winter = base.replace(
        "Ines examines transfer stage 1,",
        "Winter followed the surveyor through the harbor archive. Ines examines transfer stage 1,", 1,
    )
    _assert(not audit_story_against_blueprint(winter, _story_blueprint_fixture()),
            "'Winter followed...' must not be read as an unseeded person")
    # A single capitalized word that opens one sentence, never speaks and is never repeated ("Permafrost followed ...")
    # cannot be told from an ordinary noun: advisory only. A name that recurs or speaks is still an unseeded person.
    once = base.replace(
        "Ines examines transfer stage 1,",
        "Permafrost followed the surveyor through the harbor archive. Ines examines transfer stage 1,", 1,
    )
    once_issues = audit_story_against_blueprint(once, _story_blueprint_fixture())
    _assert(not any("unseeded named person" in i for i in once_issues)
            and any("possible unrostered name opening a sentence" in i for i in once_issues),
            f"a lone sentence-opening noun must be an advisory, not an unseeded person: {once_issues!r}")
    stranger = base.replace(
        "Ines examines transfer stage 1,",
        "Rourke followed the surveyor through the harbor archive. Rourke waited by the door. Ines examines transfer stage 1,", 1,
    )
    _assert(any("unseeded named person" in i for i in audit_story_against_blueprint(stranger, _story_blueprint_fixture())),
            "an unrostered name that recurs must still be rejected")

    # Generic 'dead' is not a precise fate; present-tense inflections count as actions.
    issues = validate_blueprint_state_contract(_story_blueprint_fixture().replace("fate: survives", "fate: dead", 1))
    _assert(any("unsupported fate" in i for i in issues),
            f"a fate without a death scene must fail: {issues!r}")
    for word in ("stands", "runs", "holds", "sits", "leaves", "standing", "holding", "sitting", "running"):
        _assert(re.search(r"\b(?:" + ACTION_VERBS + r")\b", "Ines " + word + " there"),
                f"{word!r} must be an action verb")


def test_architect_beat_length_gate_is_tolerant_but_not_empty():
    from planner.qwen_director import QwenDirector
    from planner.config import DIRECTOR_STORY_SCENES as N, STORY_PLAN_BEAT_MIN_WORDS
    director = QwenDirector.__new__(QwenDirector)
    lines = _story_blueprint_fixture(N).splitlines()
    short = ("SCENE 01: ENTRY_LOCATION=archive; ENTRY_PRESENT=Ines; ON_SCREEN_PRESENT=Ines; ENTRY_KNOWS=ledger duplicated; "
             "ENTRY_OBJECTS=ledger; EVENT=Ines checks the pressure mark and records a discrepancy; EXIT_LOCATION=archive; "
             "EXIT_PRESENT=Ines; EXIT_KNOWS=route changed; EXIT_OBJECTS=ledger; "
             "NEXT_CAUSE=Because Ines records clue one, she must verify the next handoff before the tide closes the route")
    words = len(re.findall(r"\b[\w'’-]+\b", short))
    _assert(words < STORY_PLAN_BEAT_MIN_WORDS, "fixture beat must be below the prompted minimum to test tolerance")
    lines = [short if l.startswith("SCENE 01:") else l for l in lines]
    director._normalize_story_blueprint("\n".join(lines), N)
    tiny = [("SCENE 01: ENTRY_LOCATION=archive; EVENT=Ines looks." if l.startswith("SCENE 01:") else l) for l in lines]
    try:
        director._normalize_story_blueprint("\n".join(tiny), N)
    except RuntimeError as exc:
        _assert("scene 01 beat has" in str(exc), f"unexpected error {exc!r}")
    else:
        raise AssertionError("a beat too thin to hold the state fields must still fail")


def test_blueprint_size_floors_are_shared_and_tolerant():
    from planner.config import DIRECTOR_STORY_SCENES as N
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    blueprint = _story_blueprint_fixture(N)
    lines = []
    for line in blueprint.splitlines():
        if line.startswith("CONTINUITY_LEDGER:"):
            line = ("CONTINUITY_LEDGER: Ines remains in the harbor archive and knows the ledger was duplicated, then verifies "
                    "the pressure mark. The original ledger and tide chart stay with her as the tide rises. Ines survives.")
        lines.append(line)
    text = "\n".join(lines)
    words = len(re.findall(r"\b[\w'’-]+\b", text.split("CONTINUITY_LEDGER:")[1].splitlines()[0]))
    _assert(24 <= words < 35, f"test ledger must sit between the floor and the prompted minimum, has {words}")
    normalized = director._normalize_story_blueprint(text, N)
    director._validate_story_blueprint_contract("ai_story", normalized, "a harbor story", source_character_names=[])


def test_optional_plot_fields_accept_every_way_of_saying_none():
    from planner.config import DIRECTOR_STORY_SCENES as N
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    base = _story_blueprint_fixture(N)
    for wording in ("None.", "none", "N/A", "None (no material relationships)", "No named entities.", "**None**", "Nil", "No relationships"):
        lines = [("RELATIONSHIPS: " + wording) if l.startswith("RELATIONSHIPS:") else l for l in base.splitlines()]
        normalized = director._normalize_story_blueprint("\n".join(lines), N)
        _assert("\nRELATIONSHIPS: none\n" in normalized, f"{wording!r} must normalize to 'none'")
        director._validate_story_blueprint_contract("ai_story", normalized, "a harbor story", source_character_names=[])
    lines = [("RELATIONSHIPS: Ines trusts nobody") if l.startswith("RELATIONSHIPS:") else l for l in base.splitlines()]
    try:
        director._normalize_story_blueprint("\n".join(lines), N)
    except RuntimeError as exc:
        _assert("RELATIONSHIPS" in str(exc), f"unexpected error {exc!r}")
    else:
        raise AssertionError("a thin but real relationships value must still be rejected")


def test_cast_entry_variants_are_repaired_not_rejected():
    from planner.config import DIRECTOR_STORY_SCENES as N
    from planner.qwen_director import QwenDirector
    from planner.story_contracts import parse_cast, validate_blueprint_state_contract
    director = QwenDirector.__new__(QwenDirector)
    base = _story_blueprint_fixture(N)
    entry = base.splitlines()[0].split(":", 1)[1].strip()
    scenes_12 = ", ".join(f"{i:02d}" for i in range(1, N + 1))
    variants = {
        "titled one-word name": entry.replace("Ines Vale (", "Captain Renn (", 1),
        "scene range": entry.replace(f"on-screen scenes: {scenes_12}", "on-screen scenes: 01-12", 1),
        "scene range with en dash": entry.replace(f"on-screen scenes: {scenes_12}", "on-screen scenes: 1–12", 1),
        "all scenes": entry.replace(f"on-screen scenes: {scenes_12}", "on-screen scenes: all", 1),
        "duplicate scene numbers": entry.replace(f"on-screen scenes: {scenes_12}", f"on-screen scenes: {scenes_12}, 12", 1),
        "needs instead of wants": entry.replace("wants to recover", "needs to recover", 1),
    }
    for label, changed in variants.items():
        text = "\n".join(["CAST: " + changed] + base.splitlines()[1:])
        if label == "titled one-word name":
            text = text.replace("Ines Vale", "Renn").replace("Ines", "Renn")
        normalized = director._normalize_story_blueprint(text, N)
        cast = parse_cast(normalized)
        _assert(len(cast) == 1 and cast[0]["scenes"] == list(range(1, N + 1)),
                f"{label}: scenes must resolve to 1..{N}, got {cast[0]['scenes'] if cast else cast!r}")
        issues = validate_blueprint_state_contract(normalized, N)
        _assert(not issues, f"{label}: normalized blueprint must pass the state contract: {issues!r}")
    _assert("addressed as Captain" in director._normalize_story_blueprint(
        "\n".join(["CAST: " + variants["titled one-word name"]] + base.splitlines()[1:]).replace("Ines", "Renn"), N),
        "the honorific must move into the role")


def test_cast_and_state_ledger_are_reconciled_like_the_real_failure():
    from planner.config import DIRECTOR_STORY_SCENES as N
    from planner.qwen_director import QwenDirector
    from planner.story_contracts import parse_cast, validate_blueprint_state_contract
    director = QwenDirector.__new__(QwenDirector)
    lines = _story_blueprint_fixture(N).splitlines()
    lines[0] = (
        "CAST: Dr. Elara Voss (protagonist; wants to recover the original shipping ledger before the flood; pressure fears exposing "
        "her signature; voice precise; aliases: none; on-screen scenes: 01, 03, 06; fate: survives)"
        " | Renn Tolk (harbor clerk; wants to keep the duplicate accepted; pressure fears the audit; voice curt; aliases: none; "
        "on-screen scenes: 02, 03; fate: survives)"
        " | The Vault (location; wants to stay sealed; pressure none; voice none; aliases: none; on-screen scenes: 04; fate: survives)"
    )
    out = []
    for line in lines:
        if line.startswith("SCENE "):
            number = int(line[6:8])
            line = line.replace("Ines Vale", "Dr. Elara Voss").replace("Ines", "Elara")
            if number >= 3:
                line = line.replace("ENTRY_PRESENT=Dr. Elara Voss;", "ENTRY_PRESENT=Dr. Elara Voss, Renn Tolk;")
                line = line.replace("ON_SCREEN_PRESENT=Dr. Elara Voss;", "ON_SCREEN_PRESENT=Dr. Elara Voss, Renn Tolk;")
                line = line.replace("EXIT_PRESENT=Dr. Elara Voss;", "EXIT_PRESENT=Dr. Elara Voss, Renn Tolk;")
            if number == 3:
                line = line.replace(
                    "EVENT=Elara examines",
                    "EVENT=Renn Tolk meets Elara at the archive door while Elara examines",
                )
        else:
            line = line.replace("Ines Vale", "Dr. Elara Voss").replace("Ines", "Elara") if not line.startswith("CAST:") else line
        out.append(line)
    normalized = director._normalize_story_blueprint("\n".join(out), N)
    cast = {c["name"]: c for c in parse_cast(normalized)}
    _assert(set(cast) == {"Elara Voss", "Renn Tolk"}, f"a place led by an article must not be a cast member: {sorted(cast)!r}")
    _assert(cast["Elara Voss"]["scenes"] == list(range(1, N + 1)), f"CAST scenes must follow the ledger: {cast['Elara Voss']['scenes']!r}")
    _assert(cast["Renn Tolk"]["scenes"] == list(range(3, N + 1)), f"CAST scenes must follow the ledger: {cast['Renn Tolk']['scenes']!r}")
    issues = validate_blueprint_state_contract(normalized, N)
    _assert(not issues, f"reconciled blueprint must pass the state contract: {issues!r}")


def test_second_real_failure_surnames_call_signs_final_cause_and_connectors():
    from planner.config import DIRECTOR_STORY_SCENES as N
    from planner.qwen_director import QwenDirector
    from planner.story_contracts import parse_cast, validate_blueprint_state_contract
    director = QwenDirector.__new__(QwenDirector)
    lines = _story_blueprint_fixture(N).splitlines()
    lines[0] = (
        "CAST: Elara Voss (lead engineer; wants to recover the original shipping ledger before the flood; pressure fears exposing her own "
        "signature; voice precise; aliases: none; on-screen scenes: 01, 03, 12; fate: survives)"
        " | K-9 Unit \"Rook\" (search robot; wants to follow the scent of the ledger; pressure low battery; voice synthetic; aliases: none; "
        "on-screen scenes: 02, 05, 08; fate: survives)"
        " | Sergei Petrov (harbor clerk; wants to keep the duplicate accepted; pressure fears the audit; voice curt; aliases: none; "
        "on-screen scenes: 04, 07, 10; fate: survives)"
    )
    out = []
    for line in lines:
        if line.startswith("SCENE "):
            number = int(line[6:8])
            line = line.replace("Ines Vale", "Voss").replace("Ines", "Voss")
            if number >= 3:
                line = line.replace("ENTRY_PRESENT=Voss;", "ENTRY_PRESENT=Voss, Rook, Petrov;")
                line = line.replace("ON_SCREEN_PRESENT=Voss;", "ON_SCREEN_PRESENT=Voss, Rook, Petrov;")
                line = line.replace("EXIT_PRESENT=Voss;", "EXIT_PRESENT=Voss, Rook, Petrov;")
            if number == 3:
                line = line.replace(
                    "EVENT=Voss examines",
                    "EVENT=Rook meets Petrov and Voss at the archive door while Voss examines",
                )
            line = re.sub(
                r"NEXT_CAUSE=Because [^;]*?clue (\d+), she must verify the next cargo handoff before the rising tide blocks the archive route\.",
                r"NEXT_CAUSE=Voss's record of transfer clue \1 reveals the next cargo handoff before the tide closes the archive route.",
                line,
            )
            if number == N:
                line = line.replace("NEXT_CAUSE=none", "NEXT_CAUSE=Voss prepares the next expedition to the northern station")
        else:
            line = line.replace("Ines Vale", "Elara Voss").replace("Ines", "Voss") if not line.startswith("CAST:") else line
        out.append(line)
    normalized = director._normalize_story_blueprint("\n".join(out), N)
    cast = {c["name"]: c["scenes"] for c in parse_cast(normalized)}
    _assert(cast["Elara Voss"] == list(range(1, N + 1)), f"surname presence must match the ledger: {cast!r}")
    _assert(cast['K-9 Unit "Rook"'] == list(range(3, N + 1)) and cast["Sergei Petrov"] == list(range(3, N + 1)),
            f"call-sign and surname presence must match the ledger: {cast!r}")
    _assert("SCENE 12:" in normalized and normalized.rstrip().endswith("NEXT_CAUSE=none"), "the final NEXT_CAUSE must become none")
    issues = validate_blueprint_state_contract(normalized, N)
    _assert(not issues, f"second real failure must now pass the state contract: {issues!r}")


def test_third_real_failure_semantic_handoffs_block_and_death_scene_is_kept():
    from planner.config import DIRECTOR_STORY_SCENES as N
    from planner.qwen_director import QwenDirector
    from planner.story_contracts import parse_cast, split_contract_findings, validate_blueprint_state_contract
    director = QwenDirector.__new__(QwenDirector)
    lines = _story_blueprint_fixture(N).splitlines()
    lines[0] = (
        "CAST: Ines Vale (lead engineer; wants to recover the original shipping ledger before the flood; pressure fears exposing her own "
        "signature; voice precise; aliases: none; on-screen scenes: 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12; fate: survives)"
        " | Rian Kael (harbor guard; wants to protect the archive keys; pressure old debts; voice gruff; aliases: none; "
        "on-screen scenes: 02, 03; fate: dies in scene 06)"
    )
    out = []
    for line in lines:
        if line.startswith("SCENE "):
            number = int(line[6:8])
            if number in (2, 3):
                line = line.replace("ENTRY_PRESENT=Ines;", "ENTRY_PRESENT=Ines, Rian;")
                line = line.replace("ON_SCREEN_PRESENT=Ines;", "ON_SCREEN_PRESENT=Ines, Rian;")
                line = line.replace("EXIT_PRESENT=Ines;", "EXIT_PRESENT=Ines, Rian;")
            line = re.sub(r"NEXT_CAUSE=[^;]*?(?=;|$)", "NEXT_CAUSE=Ines hides the key", line) if number < N else line
        out.append(line)
    normalized = director._normalize_story_blueprint("\n".join(out), N)
    cast = {c["name"]: c for c in parse_cast(normalized)}
    _assert(6 in cast["Rian Kael"]["scenes"] and max(cast["Rian Kael"]["scenes"]) == 6,
            f"death scene must stay on screen and nothing follows it: {cast['Rian Kael']!r}")
    fatal, soft = split_contract_findings(validate_blueprint_state_contract(normalized, N))
    _assert(not any("death scene" in item for item in fatal), f"death-scene error must not return: {fatal!r}")
    # Wording overlap between NEXT_CAUSE and the next EVENT is advisory: it is reported, it does not reject the plan.
    _assert(not any("event is not anchored to the previous scene's NEXT_CAUSE" in item for item in fatal)
            and any("event is not anchored to the previous scene's NEXT_CAUSE" in item for item in soft),
            f"a paraphrased handoff must be a warning, not a fatal rejection: fatal={fatal!r} soft={soft!r}")
    _assert(not any("lacks an explicit causal connector" in item for item in soft),
            f"literal connector wording must not create false warnings: {soft!r}")


def test_setup_payoff_requires_distinctive_evidence_in_the_declared_event():
    from planner.story_contracts import validate_blueprint_state_contract
    text = _story_blueprint_fixture()
    # A character/location/topic overlap must not satisfy a planted object that is absent from EVENT.
    text = text.replace(
        "PAYOFF Scene 08: Ines uses that same shifted pressure seal notch to prove her correction changed the rescue route.",
        "PAYOFF Scene 08: Ines uses the brass access keycard to override the station security protocol.",
    )
    issues = validate_blueprint_state_contract(text)
    _assert(any("SETUP_PAYOFF_MAP payoff in scene 08" in item for item in issues),
            f"an unrelated payoff must not pass through generic character/location overlap: {issues!r}")


def test_object_inventory_is_checked_per_item_and_cannot_drop_a_keycard():
    from planner.story_contracts import validate_blueprint_state_contract
    text = _story_blueprint_fixture()
    lines = text.splitlines()
    key_line = next(i for i, line in enumerate(lines) if line.startswith("KEY_OBJECTS:"))
    lines[key_line] = lines[key_line].replace("KEY_OBJECTS:", "KEY_OBJECTS: brass keycard,", 1)
    s2 = next(i for i, line in enumerate(lines) if line.startswith("SCENE 02:"))
    lines[s2] = lines[s2].replace(
        "EVENT=", "EVENT=Ines finds and lifts the brass keycard from the desk. ", 1,
    )
    lines[s2] = lines[s2].replace(
        "EXIT_OBJECTS=original ledger and tide chart remain under Ines's control",
        "EXIT_OBJECTS=original ledger and tide chart remain under Ines's control, brass keycard held by Ines",
    )
    s3 = next(i for i, line in enumerate(lines) if line.startswith("SCENE 03:"))
    lines[s3] = lines[s3].replace(
        "ENTRY_OBJECTS=original ledger and tide chart remain under Ines's control",
        "ENTRY_OBJECTS=original ledger and tide chart remain under Ines's control, brass keycard held by Ines",
    )
    lines[s3] = lines[s3].replace(
        "EXIT_OBJECTS=original ledger and tide chart remain under Ines's control",
        "EXIT_OBJECTS=original ledger and tide chart remain under Ines's control, brass keycard held by Ines",
    )
    s4 = next(i for i, line in enumerate(lines) if line.startswith("SCENE 04:"))
    lines[s4] = lines[s4].replace(
        "ENTRY_OBJECTS=original ledger and tide chart remain under Ines's control, brass keycard held by Ines",
        "ENTRY_OBJECTS=original ledger and tide chart remain under Ines's control",
    )
    issues = validate_blueprint_state_contract("\n".join(lines))
    _assert(any("object inventory drops key object '" in item and "brass keycard" in item for item in issues),
            f"one surviving object must not be masked by other matching inventory tokens: {issues!r}")


def test_architect_revision_merge_preserves_cast_and_unaffected_scenes():
    from planner.qwen_director import QwenDirector
    from planner.story_contracts import parse_scene_state
    original_lines = _story_blueprint_fixture().splitlines()
    # The original Scene 04 already contains an explicit return from the annex to the archive.
    scene4_index = next(i for i, line in enumerate(original_lines) if line.startswith("SCENE 04:"))
    original_lines[scene4_index] = original_lines[scene4_index].replace(
        "EVENT=Ines examines transfer stage 4,",
        "EVENT=Ines descends from the archive annex into the harbor archive, then examines transfer stage 4,",
    )
    original = "\n".join(original_lines)

    revised_lines = original.splitlines()
    revised_lines[0] = revised_lines[0].replace("Ines Vale", "Ines Nova").replace("kind: person", "kind: embodied agent")
    scene3_index = next(i for i, line in enumerate(revised_lines) if line.startswith("SCENE 03:"))
    revised_lines[scene3_index] = revised_lines[scene3_index].replace(
        "EVENT=Ines examines transfer stage 3,",
        "EVENT=Ines unlocks the service hatch, climbs the ladder, and enters the archive annex to examine transfer stage 3,",
    ).replace("EXIT_LOCATION=harbor archive", "EXIT_LOCATION=archive annex")
    revised_lines[scene4_index] = revised_lines[scene4_index].replace(
        "ENTRY_LOCATION=harbor archive", "ENTRY_LOCATION=archive annex",
    )
    revised = "\n".join(revised_lines)
    problems = ["scene 03 changes its physical location without an explicit transition ('harbor archive' to 'archive annex')"]
    merged = QwenDirector._merge_story_blueprint_revision(original, revised, problems)
    _assert(merged.splitlines()[0] == original.splitlines()[0],
            "a revision must preserve the complete original canonical CAST line")
    original_scenes = {line[:9]: line for line in original.splitlines() if line.startswith("SCENE ")}
    merged_scenes = {line[:9]: line for line in merged.splitlines() if line.startswith("SCENE ")}
    _assert(merged_scenes["SCENE 12:"] == original_scenes["SCENE 12:"],
            "an unrelated scene must remain byte-for-byte unchanged")
    scene3 = parse_scene_state(merged_scenes["SCENE 03:"])
    scene4 = parse_scene_state(merged_scenes["SCENE 04:"])
    _assert(scene3["EXIT_LOCATION"] == scene4["ENTRY_LOCATION"] == "archive annex",
            "a repaired exit location must be merged with the next scene's inherited entry location")


def test_architect_revision_merge_preserves_object_boundary_and_rejects_duplicate_patch():
    from planner.qwen_director import QwenDirector
    from planner.story_contracts import parse_scene_state
    original_lines = _story_blueprint_fixture().splitlines()
    key_line = next(i for i, line in enumerate(original_lines) if line.startswith("KEY_OBJECTS:"))
    original_lines[key_line] = original_lines[key_line] + ", brass keycard"
    ledger_line = next(i for i, line in enumerate(original_lines) if line.startswith("CONTINUITY_LEDGER:"))
    original_lines[ledger_line] += " The brass keycard remains tracked as a critical prop."
    for i, line in enumerate(original_lines):
        if not line.startswith("SCENE "):
            continue
        scene_number = int(line[6:8])
        entry_state = "intact" if scene_number <= 2 else "broken"
        exit_state = "broken" if scene_number >= 2 else "intact"
        line = line.replace(
            "ENTRY_OBJECTS=original ledger and tide chart remain under Ines's control;",
            f"ENTRY_OBJECTS=original ledger and tide chart remain under Ines's control, brass keycard {entry_state};",
        )
        line = line.replace(
            "EXIT_OBJECTS=original ledger and tide chart remain under Ines's control;",
            f"EXIT_OBJECTS=original ledger and tide chart remain under Ines's control, brass keycard {exit_state};",
        )
        original_lines[i] = line
    original = "\n".join(original_lines)
    from planner.story_contracts import validate_blueprint_state_contract
    object_findings = validate_blueprint_state_contract(original)
    target = next((item for item in object_findings if "scene 02 key-object state change" in item), None)
    _assert(target is not None, f"fixture must expose a Scene 02 keycard action gap: {object_findings!r}")

    revised_lines = original.splitlines()
    scene2_index = next(i for i, line in enumerate(revised_lines) if line.startswith("SCENE 02:"))
    revised_lines[scene2_index] = revised_lines[scene2_index].replace(
        "EVENT=Ines examines transfer stage 2,",
        "EVENT=Ines breaks the brass keycard against the iron console while examining transfer stage 2,",
    )
    revised = "\n".join(revised_lines)
    merged = QwenDirector._merge_story_blueprint_revision(original, revised, [target])
    scene2 = parse_scene_state(next(line for line in merged.splitlines() if line.startswith("SCENE 02:")))
    scene3 = parse_scene_state(next(line for line in merged.splitlines() if line.startswith("SCENE 03:")))
    _assert("breaks the brass keycard" in scene2["EVENT"],
            "a key-object revision finding must be recognized and its corrected physical action merged")
    _assert(scene2["EXIT_OBJECTS"] == scene3["ENTRY_OBJECTS"],
            "an accepted object-state repair must keep the prior exit and next entry inventories synchronized")

    duplicate_lines = _story_blueprint_fixture().splitlines()
    scene11 = next(line for line in duplicate_lines if line.startswith("SCENE 11:"))
    scene12_index = next(i for i, line in enumerate(duplicate_lines) if line.startswith("SCENE 12:"))
    scene11_event = scene11.split("EVENT=", 1)[1].split("; EXIT_LOCATION=", 1)[0]
    duplicate_lines[scene12_index] = re.sub(
        r"EVENT=.*?; EXIT_LOCATION=", "EVENT=" + scene11_event + "; EXIT_LOCATION=",
        duplicate_lines[scene12_index], count=1,
    )
    duplicate_candidate = "\n".join(duplicate_lines)
    preserved = QwenDirector._merge_story_blueprint_revision(
        _story_blueprint_fixture(), duplicate_candidate,
        ["scene 12 changes its physical location without an explicit transition ('harbor archive' to 'coast')"],
    )
    source12 = parse_scene_state(next(line for line in _story_blueprint_fixture().splitlines() if line.startswith("SCENE 12:")))
    merged12 = parse_scene_state(next(line for line in preserved.splitlines() if line.startswith("SCENE 12:")))
    _assert(merged12["EVENT"] == source12["EVENT"],
            "the merge must not import a candidate event that duplicates the preceding scene")


def test_scene_local_location_validator_accepts_explicit_shelter_entry():
    from planner.story_contracts import _scene_local_transition_is_explained
    cast = [{"name": "Elena Voss", "aliases": ["Elena"]}]
    _assert(_scene_local_transition_is_explained(
        "Arctic station exterior during a storm", "Station interior",
        "Elena Voss takes shelter inside the station after crossing the broken threshold.", cast,
    ), "an explicit physical retreat inside must satisfy the location transition")


# ── Runner ────────────────────────────────────────────────────────────────

def main():
    tests = [
        test_shot_schema_tracks_instance_topology,
        test_compact_shot_batch_schema_omits_derived_and_empty_fields,
        test_compact_shot_prompt_example_is_valid_json,
        test_character_schema_does_not_silently_cap_qwen_cast_at_32,
        test_character_semantic_schema_includes_spoken_dialogue,
        test_qwen_can_choose_any_cast_size,
        test_trace_sentient_ai_candidate_survives_normalization_and_reaches_roster,
        test_sentient_ai_label_is_not_promoted_when_qwen_does_not_mark_it_sentient,
        test_qwen_negative_is_authoritative,
        test_deterministic_detection_never_adds_missing_qwen_character,
        test_qwen_extraction_does_not_force_candidates_from_hints_plan_or_dialogue,
        test_mention_only_qwen_candidates_are_filtered_without_restoring_them,
        test_non_person_qwen_candidates_are_filtered,
        test_relational_owner_must_be_qwen_approved,
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
        test_in_person_dialogue_speaker_must_exist_in_qwen_cast,
        test_source_attribution_overrides_mislabelled_ui_quote_speaker,
        test_unrostered_remote_speaker_without_source_grounding_is_dropped,
        test_unrostered_hologram_speaker_is_audio_only_not_a_crash,
        test_remote_only_shot_can_have_no_visible_source_body,
        test_remote_delivery_survives_timeline_and_context_ir_without_visible_source,
        test_remote_channel_state_ends_when_a_local_speaker_takes_over,
        test_remote_only_voice_is_removed_from_visible_scene_and_shot_bindings,
        test_dialogue_reconciliation_preserves_remote_audio_without_visible_binding,
        test_story_prompt_is_open_cast_and_fully_filled,
        test_story_prompt_values_and_dialogue_exchange_contract_are_consistent,
        test_story_prompts_encode_escalation_opposition_setup_payoff_and_scene_purpose,
        test_generated_story_cast_contract_is_plot_driven_without_a_numeric_quota,
        test_story_blueprint_normalization_requires_ordered_scene_beats,
        test_story_blueprint_accepts_wrapped_cast_formats,
        test_director_checkpoint_is_accepted_by_orchestrator_resume,
        test_critic_cannot_break_cinematic_coverage,
        test_story_contract_regressions_found_by_audit,
        test_architect_beat_length_gate_is_tolerant_but_not_empty,
        test_blueprint_size_floors_are_shared_and_tolerant,
        test_optional_plot_fields_accept_every_way_of_saying_none,
        test_cast_entry_variants_are_repaired_not_rejected,
        test_cast_wants_label_spellings_match_the_prompt_example_and_reach_downstream_parsers,
        test_scene_purpose_survives_scene_sanitizer_and_scene_floor_is_enforced,
        test_cast_and_state_ledger_are_reconciled_like_the_real_failure,
        test_second_real_failure_surnames_call_signs_final_cause_and_connectors,
        test_third_real_failure_semantic_handoffs_block_and_death_scene_is_kept,
        test_environmental_opposition_is_not_cast_and_boundary_presence_is_reconciled,
        test_prose_presence_audit_accepts_common_departure_and_behavior_cues,
        test_duplicate_adjacent_scene_block_is_cleaned_without_retry_and_isolated_repetition_is_advisory,
        test_story_run_diagnostics_and_rejected_draft_are_in_trace_bundle,
        test_successful_architect_plan_uses_one_native_bounded_thinking_call,
        test_structured_architect_json_round_trips_through_every_contract,
        test_architect_and_author_get_only_informed_hard_gate_revisions,
        test_scene_boundaries_are_derived_and_wording_only_findings_are_advisory,
        test_story_minimum_is_a_hard_floor_with_one_conditional_revision,
        test_story_hard_quality_gate_rejects_repeated_stock_phrases_and_overlong_dialogue,
        test_cast_absent_from_one_scene_is_advisory_but_never_appearing_is_fatal,
        test_blueprint_state_ledger_rejects_unrostered_people_location_jumps_and_post_death_presence,
        test_story_entity_relationship_and_person_grounding_is_a_hard_gate,
        test_a_prop_the_plan_declares_is_seeded_but_an_invented_organization_is_not,
        test_a_person_the_architect_named_but_never_put_on_screen_is_seeded,
        test_story_cinematic_coverage_enforces_unique_framing_without_camera_motion_quota,
        test_story_completion_contract_reports_an_unresolved_final_hook_as_a_warning,
        test_story_pipeline_uses_two_bounded_stages_and_only_conditional_revisions,
        test_story_writer_is_one_qwen3_nonthinking_call_with_configured_ceiling,
        test_author_min_tokens_zero_is_omitted_from_vllm_payload,
        test_rejected_author_draft_is_saved_for_diagnosis,
        test_author_wrapper_preserves_draft_when_pipeline_rejects_it,
        test_scene_markers_preserve_one_to_one_scene_boundaries,
        test_context_compaction_never_loses_closing_quote_or_splits_attribution,
        test_story_context_compactor_keeps_causal_middle_action,
        test_story_quote_cleanup_preserves_all_dialogue_words,
        test_duplicate_scene_blocks_are_hard_and_author_revision_rebuilds_from_architecture,
        test_blueprint_story_structure_failures_are_hard_and_feedback_is_actionable,
        test_architect_repair_recognizes_logged_location_and_object_verbs,
        test_architect_revision_context_preserves_the_failed_plan_and_compacts_findings,
        test_architect_revision_retry_receives_the_actual_rejected_blueprint,
        test_setup_payoff_requires_distinctive_evidence_in_the_declared_event,
        test_object_inventory_is_checked_per_item_and_cannot_drop_a_keycard,
        test_architect_revision_merge_preserves_cast_and_unaffected_scenes,
        test_architect_revision_merge_preserves_object_boundary_and_rejects_duplicate_patch,
        test_scene_local_location_validator_accepts_explicit_shelter_entry,
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

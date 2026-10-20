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


# ── Character roster behavior ──────────────────────────────────────────────

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


def test_mention_only_matching_is_case_insensitive():
    planner = _planner()
    _assert(planner._is_mention_only_identity("KOVAC had vanished.", "Kovac"), "mention-only matching must ignore case")
    _assert(not planner._is_mention_only_identity("kovac stepped into the vault.", "Kovac"), "present-tense character mention must not be filtered")


def test_sanitizer_deduplicates_normalized_identity_keys():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    result = director._sanitize_characters([
        {"name": "Elias Kade", "identity_type": "named_character"},
        {"name": "  elias   kade  ", "identity_type": "named_character"},
    ])
    _assert(len(result) == 1, f"sanitizer should deduplicate normalized identity keys: {result}")


def test_build_alias_map_skips_stopword_first_token():
    from planner.entity_resolver import EntityResolver
    aliases = EntityResolver.build_alias_map({"the man in the grey coverall"})
    _assert("the" not in aliases, "article must never become a character alias")
    _assert(aliases.get("grey coverall") == "the man in the grey coverall",
            f"stable descriptive pair alias missing: {aliases}")


def test_characters_in_scene_does_not_bind_descriptive_by_article():
    from types import SimpleNamespace
    planner = _planner()
    descriptive = {"name": "the man in the grey coverall", "role": "story character",
                   "identity_type": "descriptive_character", "semantic_aliases": []}
    other = {"name": "Eli", "role": "story character", "identity_type": "named_character", "semantic_aliases": []}
    characters = [
        SimpleNamespace(**descriptive, to_dict=lambda: dict(descriptive)),
        SimpleNamespace(**other, to_dict=lambda: dict(other)),
    ]
    result = planner._characters_in_scene("The door opened and the lights failed.", characters)
    _assert("the man in the grey coverall" not in result, f"descriptive identity was bound by article: {result}")


def test_sanitize_scene_alias_fallback_does_not_bind_descriptive_by_article():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    aliases = director._character_alias_map({"the man in the grey coverall"})
    _assert("the" not in aliases, "sanitizer alias map must not create article alias")
    _assert(aliases.get("the man in the grey coverall") == "the man in the grey coverall",
            f"descriptive canonical alias missing: {aliases}")


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


def test_qwen_extractor_is_called_even_for_obvious_named_roster():
    planner = _planner()
    story = "Elias Kade entered the station. Lin Mei waited beside the vault."
    calls = []
    def fail_deterministic_scan(*_args, **_kwargs):
        raise AssertionError("creative Qwen path must not run deterministic character scans")
    planner.detect_character_descriptors = fail_deterministic_scan
    planner._explicit_source_character_names = fail_deterministic_scan
    def extractor(_story, hints):
        calls.append(list(hints))
        return {"candidates": [
            {"name": "Elias Kade", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Lin Mei", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
        ]}
    planner.create_characters(story, qwen_character_extractor=extractor)
    _assert(calls == [[]], f"Qwen must receive the story without any deterministic name hints: {calls}")


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


def test_descriptive_alias_can_be_canonical_surface():
    planner = _planner()
    story = "Lena watched the man in the grey coverall lift the gate. The man in the grey coverall nodded."
    def extractor(_story, _hints):
        return {"candidates": [{
            "name": "man", "entity_type": "PERSON", "is_character": True,
            "aliases": ["the man in the grey coverall"],
            "identity_type": "descriptive_character",
        }]}
    characters = planner.create_characters(story, qwen_character_extractor=extractor)
    _assert([c.name for c in characters] == ["the man in the grey coverall"], "grounded descriptive alias was dropped")
    _assert(characters[0].identity_type == "descriptive_character", "descriptive identity type was lost")


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


def test_relational_character_survives_with_qwen_owner():
    planner = _planner()
    story = "Eli entered the station. Eli's father waited beside the door and warned him. The woman in the lab coat blocked the exit."
    def extractor(_story, _hints):
        return {"candidates": [
            {"name": "Eli", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Eli's father", "entity_type": "CHARACTER", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "The woman in the lab coat", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
        ]}
    characters = planner.create_characters(story, qwen_character_extractor=extractor)
    names = [c.name for c in characters]
    _assert(names == ["Eli", "Eli's father", "The woman in the lab coat"], f"Qwen-approved relational/descriptive identities were lost: {names}")
    _assert(characters[1].identity_type == "relational_character", "relational identity type was lost")
    _assert(characters[2].identity_type == "descriptive_character", "descriptive identity type was lost")


def test_no_adjudication_call_even_when_extractor_is_partial():
    planner = _planner()
    story = "Elias Kade entered the vault and Lin Mei followed him."
    calls = []
    def extractor(_story, _hints):
        calls.append("extract")
        return {"candidates": [{
            "name": "Elias Kade", "entity_type": "PERSON", "is_character": True,
            "aliases": [], "identity_type": "named_character",
        }]}
    def adjudicator(*_args):
        calls.append("adjudicate")
        raise AssertionError("adjudication must never be called")
    names = [c.name for c in planner.create_characters(
        story, qwen_character_extractor=extractor, qwen_character_adjudicator=adjudicator
    )]
    _assert(names == ["Elias Kade"], f"unexpected deterministic/adjudication recovery: {names}")
    _assert(calls == ["extract"], f"semantic path made more than one Qwen character call: {calls}")


def test_expand_is_open_cast_but_preserves_source_anchors():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    planner = _planner()
    director._planner = lambda: planner
    source = "Eli enters the abandoned station and finds a sealed vault."
    generated = "Eli enters the station. Mara meets him at the vault. Tomas unlocks the gate."
    director._validate_expand_story_cast(source, generated, source_character_names=["Eli"])
    missing = "Mara enters the station. Tomas unlocks the gate."
    try:
        director._validate_expand_story_cast(source, missing, source_character_names=["Eli"])
        raise AssertionError("dropping an established source anchor must fail")
    except RuntimeError as exc:
        _assert("omitted established source character" in str(exc), f"wrong source-anchor error: {exc}")


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


def test_named_ai_voice_from_speakers_is_source_grounded_remote():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    story = 'AURORA\'s voice echoed through the speakers: “The vault is failing.”'
    quote = QwenDirectorPromptMixin._source_quote_inventory(story)[0]
    attribution = QwenDirectorPromptMixin._source_named_quote_attribution(story, quote)
    _assert(attribution is not None and attribution["speaker"] == "AURORA"
            and attribution["delivery"] == "remote",
            f"a speaker named at the start of a paragraph must not lose its first letter: {attribution!r}")
    _assert(QwenDirectorPromptMixin._source_supports_remote_speaker(quote, attribution["speaker"]),
            "an explicit speaker identity and remote channel in source context must pass remote grounding")


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


def test_quote_inventory_preserves_repeated_quotes_and_punctuation():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    story = "Lena said, “It’s the only way…” Later, Tomas repeated, \"It's the only way...\""
    quotes = QwenDirectorPromptMixin._source_quote_inventory(story)
    _assert(len(quotes) == 2, f"quote inventory lost a source occurrence: {quotes!r}")
    _assert([item["quote_id"] for item in quotes] == [1, 2], "quote IDs must be stable and ordered")
    _assert(quotes[0]["text"] == "It’s the only way…" and quotes[1]["text"] == "It's the only way...",
            "quote inventory must preserve exact source punctuation")
    from planner.qwen_director import QwenDirector
    _assert(QwenDirector._dialogue_anchor_key(quotes[0]["text"]) == QwenDirector._dialogue_anchor_key(quotes[1]["text"]),
            "curly apostrophes and ellipses must compare equal without rewriting source text")


def test_blueprint_cast_filter_excludes_weather_and_objects_but_keeps_dynamic_people():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    blueprint = "CAST: Mara Venn (engineer; wants the truth; direct) | The Storm (non-human, relentless force; howls) | The Vault (sealed chamber; holds the core) | Ivo Chen (medic; wants to save Mara; calm)\nSCENE 01: Mara enters."
    normalized = QwenDirectorPromptMixin._normalize_story_blueprint(blueprint + "\n" + "\n".join(f"SCENE {i:02d}: Beat {i} follows the previous action." for i in range(2, 13)), 12)
    cast_line = normalized.splitlines()[0]
    _assert("Mara Venn" in cast_line and "Ivo Chen" in cast_line, f"real character anchors were lost: {cast_line}")
    _assert("The Storm" not in cast_line and "The Vault" not in cast_line, f"non-person CAST entries survived: {cast_line}")


def test_cast_static_check_is_alias_aware_and_does_not_require_fixed_recurrence():
    from planner.qwen_director import QwenDirector
    director = object.__new__(QwenDirector)
    story = "Dr. Elena Voss entered.\n\nElena found the key.\n\nAnika Patel arrived and spoke."
    blueprint = "CAST: Dr. Elena Voss (engineer; precise) | Anika Patel (researcher; sharp)\nSCENE 01: Elena arrives."
    issues = director._story_cast_contract_issues(story, blueprint)
    _assert(not any("Dr. Elena Voss" in issue for issue in issues), f"titleless first-name use caused a false cast warning: {issues}")


def test_preserve_quality_contract_accepts_source_driven_single_scene_only_for_preserve_mode():
    from planner.config import AI_STORY_MODE, PRESERVE_USER_STORY_MODE
    from planner.qwen_director import QwenDirector
    director = object.__new__(QwenDirector)
    scenes = [{"scene_id": "scene_001"}]
    shots = [{
        "shot_id": f"scene_001_shot_{i:03d}", "scene_id": "scene_001", "characters": [],
        "speaking_characters": [], "visual_prompt": "A grounded cinematic view of the supplied source event and its physical setting.",
        "camera_shot": "medium", "camera_movement": "static", "lens_and_depth_of_field": "normal, deep focus",
        "composition_notes": "Subject centered with environmental context.",
    } for i in range(1, 4)]
    director._validate_production_quality(mode=PRESERVE_USER_STORY_MODE, story="Eli finds a sealed vault.", scenes=scenes, shots=shots, characters=[])
    try:
        director._validate_production_quality(mode=AI_STORY_MODE, story="Eli finds a sealed vault.", scenes=scenes, shots=shots, characters=[{"name": "Eli"}])
    except RuntimeError as exc:
        _assert("must contain 4" in str(exc), f"unexpected generated-story topology failure: {exc}")
    else:
        raise AssertionError("AI Story incorrectly accepted a one-scene plan")


def test_preserve_short_source_is_not_duplicated_into_fake_scenes():
    from planner.production_planner import ProductionPlanner
    planner = ProductionPlanner(Path("."))
    units = planner._split_story("Eli enters the station and finds a sealed vault.")
    result = planner._rebalance_story_units(units, minimum_units=1)
    scenes = planner.create_scenes("Eli enters the station and finds a sealed vault.", [], minimum_scenes=1)
    _assert(len(result) == 1 and len(scenes) == 1, f"short Preserve Story was padded with repeated scenes: {[x.text for x in result]} / {len(scenes)} scenes")
    _assert(result[0].text == "Eli enters the station and finds a sealed vault.", "Preserve Story source text changed")


def test_watchdog_is_armed_per_qwen_http_request_not_whole_generation():
    import inspect
    from planner.qwen_director import QwenDirector
    from planner.qwen_director_runtime import QwenDirectorRuntimeMixin
    generate_source = inspect.getsource(QwenDirector.generate)
    post_source = inspect.getsource(QwenDirectorRuntimeMixin._post_chat)
    _assert("@_with_faulthandler_watchdog" not in generate_source, "the 15-minute watchdog must not wrap the entire multi-call production")
    _assert("@_with_faulthandler_watchdog" in post_source, "watchdog must reset for each Qwen HTTP request")


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
    # The shot model wrongly puts Ivo's second-scene line in scene 1. It must be dropped there.
    shots[0]["dialogue_events"] = [{"speaker": "Ivo", "text": "The bridge is gone."}]
    director._normalize_dialogue_speakers(story, scenes, shots, [{"name": "Mara"}, {"name": "Ivo"}])
    by_scene = {scene_id: [event["text"] for shot in shots if shot["scene_id"] == scene_id for event in shot["dialogue_events"]] for scene_id in ("scene_001", "scene_002")}
    _assert(by_scene["scene_001"] == ["The gate is open."], f"scene 1 dialogue drifted or source line was lost: {by_scene}")
    _assert(by_scene["scene_002"] == ["The bridge is gone."], f"scene 2 dialogue was not restored: {by_scene}")


def test_unclassified_quote_with_explicit_named_attribution_is_recovered_without_retry():
    from planner.qwen_director import QwenDirector
    story = 'Mara said, “The gate is open.” Later, Ivo replied, “The bridge is gone…”'
    semantic = [{"quote_id": 1, "text": "The gate is open.", "speaker": "Mara"}]
    segments = QwenDirector._extract_story_spoken_segments(story, semantic, allowed_speakers={"Mara", "Ivo"})
    _assert([item["display"] for item in segments] == ["The gate is open.", "The bridge is gone…"],
            f"explicitly attributed source dialogue was lost when Qwen omitted a quote ID: {segments!r}")
    _assert(segments[1]["source_speakers"] == {"ivo"}, f"source attribution did not recover Ivo's line: {segments!r}")


def test_long_source_quote_inventory_keeps_all_quotes_and_bounds_context():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    story = "\n\n".join(f'Mara entered room {i}. Mara said, “Source line number {i} remains exact.” Ivo watched.' for i in range(180))
    quotes = QwenDirectorPromptMixin._source_quote_inventory(story)
    evidence = QwenDirectorPromptMixin._semantic_extraction_evidence(story, quotes)
    _assert(len(quotes) == 180 and len(evidence["quotes"]) == 180, "long source compaction silently dropped quote candidates")
    _assert([item["quote_id"] for item in evidence["quotes"]] == list(range(1, 181)), "long-source quote IDs are not complete and ordered")
    _assert(len(evidence["story_evidence"]) < len(story), "long-source story evidence should be bounded while quote inventory remains complete")
    _assert("Ivo" in evidence["name_surface_index"] and "Mara" in evidence["name_surface_index"], "long-source name-like surfaces were lost from the recall index")
    _assert(any("Ivo" in item["source_context"] for item in evidence["name_surface_evidence"]), "long-source source contexts lost an active character mention")


def test_story_markdown_emphasis_is_removed_without_cross_paragraph_consumption():
    from planner.qwen_director import QwenDirector
    story = "Mara *opened the vault* carefully.\n\nIvo **held the door** while Elena _watched_."
    cleaned = QwenDirector._strip_story_markdown_emphasis(story)
    _assert(cleaned == "Mara opened the vault carefully.\n\nIvo held the door while Elena watched.", f"Markdown sanitizer damaged story prose: {cleaned!r}")


# ── Dialogue contract behavior ─────────────────────────────────────────────

def test_story_dialogue_contract_uses_qwen_semantic_spans():
    from planner.qwen_director import QwenDirector
    semantic = [
        {"text": "We go now", "speaker": "Lena"},
        {"text": "Nobody goes up", "speaker": "Ines"},
    ]
    human = QwenDirector._extract_story_spoken_segments(
        '"We go now," Lena whispered. She opened the door.',
        semantic, allowed_speakers={"Lena"},
    )
    _assert(human and human[0]["source_speakers"] == {"lena"}, "semantic human dialogue was not reconciled")

    machine = QwenDirector._extract_story_spoken_segments(
        'The terminal displayed "Access granted."',
        semantic, allowed_speakers={"Lena", "Ines"},
    )
    _assert(not machine, "unblessed machine/system text must not become dialogue")

    written_quote = QwenDirector._extract_story_spoken_segments(
        'Nadia opened the journal. “Seal at five,” Nadia read. She closed the book.',
        [{"text": "Seal at five", "speaker": "Nadia"}], allowed_speakers={"Nadia"},
    )
    _assert(not written_quote, "a named character reading journal text must not become present-time dialogue")

    live_quote = QwenDirector._extract_story_spoken_segments(
        'Nadia entered the room and said, “Leave now.”',
        [{"text": "Leave now", "speaker": "Nadia"}], allowed_speakers={"Nadia"},
    )
    _assert(live_quote, "generalized written-source filtering must preserve direct in-person speech")

    later_live_quote = QwenDirector._extract_story_spoken_segments(
        'He remembered his mother saying “Come home before dark.” '
        'The door flew open. Mara stepped into the room and shouted, “Get down!”',
        [
            {"text": "Come home before dark.", "speaker": "Mara"},
            {"text": "Get down!", "speaker": "Mara"},
        ],
        allowed_speakers={"Mara"},
    )
    _assert(
        len(later_live_quote) == 1 and later_live_quote[0]["display"] == "Get down!",
        "a prior memory in the same paragraph must not suppress a later present-time line",
    )

    speaker_contamination_story = (
        'A voice crackled through a long-dead speaker: “You shouldn’t be here.” '
        'Lena turned and found Dr. Elias standing beside the vault. '
        'He warned her that opening it would release something no one was prepared for. '
        '“The storm is the first warning,” he said, his voice frayed with urgency.'
    )
    speaker_contamination = QwenDirector._extract_story_spoken_segments(
        speaker_contamination_story,
        [
            {"text": "You shouldn’t be here", "speaker": "Dr. Elias"},
            {"text": "The storm is the first warning", "speaker": "Dr. Elias"},
        ],
        allowed_speakers={"Lena", "Dr. Elias"},
    )
    _assert(
        [item["text"] for item in speaker_contamination] == ["you shouldn't be here", "the storm is the first warning"],
        f"remote source dialogue should be preserved without contaminating the later in-person line: {speaker_contamination!r}",
    )
    _assert(speaker_contamination[0].get("delivery") == "remote" and speaker_contamination[1].get("delivery") == "in_person",
            f"remote-channel carry-over did not stop at the physical entrance: {speaker_contamination!r}")


def test_consecutive_quote_speaker_carries_forward_after_pronoun_attribution():
    from planner.qwen_director import QwenDirector
    story = (
        'Vex confronted Elara in the vault. “You’ll bury the cure!” she screamed. '
        '“The world needs it!” Elara met her gaze, unwavering.'
    )
    segments = QwenDirector._extract_story_spoken_segments(
        story,
        [
            {"quote_id": 1, "text": "You’ll bury the cure!", "speaker": "Mara Vex"},
            # Deliberately wrong extractor label: source grammar has no new speaker
            # attribution before the second line, so the continuing turn remains Vex's.
            {"quote_id": 2, "text": "The world needs it!", "speaker": "Elara"},
        ],
        allowed_speakers={"Mara Vex", "Elara"},
    )
    _assert(len(segments) == 2, f"both consecutive dialogue turns must survive: {segments!r}")
    _assert(segments[0]["source_speakers"] == {"mara vex"}, f"first shout lost Vex attribution: {segments!r}")
    _assert(segments[1]["source_speakers"] == {"mara vex"},
            f"continued quote was incorrectly reassigned without a new attribution: {segments!r}")


def test_explicit_empty_shot_characters_are_not_filled_from_scene_cast():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    scene = {
        "scene_id": "scene_001",
        "description": "The empty vault corridor hums under emergency lighting.",
        "location": "vault corridor",
        "characters": ["Elara Voss", "Mara Vex"],
    }
    shot_data = {
        "shot_id": "scene_001_shot_001", "scene_id": "scene_001",
        "action": "The emergency light pulses over the sealed vault door.",
        "visual_prompt": "The empty vault corridor under emergency lights, the sealed vault door and no people in view.",
        "characters": [], "dialogue_events": [], "speaking_characters": [], "speech_text": "",
        "camera_shot": "wide", "camera_movement": "static",
        "lens_and_depth_of_field": "normal lens", "composition_notes": "Center the sealed door.",
    }
    sanitized = director._sanitize_shots([shot_data], scene, {"elara voss", "mara vex"})
    _assert(len(sanitized) == 1, f"valid environment-only shot was dropped: {sanitized!r}")
    _assert(sanitized[0].get("characters") == [],
            f"explicitly empty visible cast was silently repopulated from the scene roster: {sanitized[0]!r}")
    director._validate_shot_character_contract(sanitized, [
        {"name": "Elara Voss"}, {"name": "Mara Vex"},
    ])


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


def test_unrostered_remote_speaker_survives_without_cast_addback():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.qwen_director import QwenDirector
    from pipeline.dialogue_timeline import DialogueTimeline

    story = 'Eli watched as a hologram flickered to life. “I left you a key.” His sister’s face shimmered on the terminal.'
    mixin = QwenDirectorPromptMixin()
    mixin._character_semantic_calls = 0
    mixin._semantic_spoken_dialogue = []
    mixin._anchor_semantic_dialogue_to_story = lambda _story, items: list(items)
    mixin._chat_json = lambda *args, **kwargs: {
        "candidates": [{"name": "Eli", "entity_type": "PERSON", "is_character": True, "identity_type": "named_character"}],
        "quote_classifications": [{
            "quote_id": 1,
            "classification": "remote_or_offscreen_spoken",
            "speaker": "Eli's sister (hologram)",
        }],
    }
    result = mixin.extract_character_entities(story, [])
    _assert([item["name"] for item in result["candidates"]] == ["Eli"],
            "an audio-only remote speaker must not be added to Qwen's canonical roster")
    _assert(len(mixin._semantic_spoken_dialogue) == 1
            and mixin._semantic_spoken_dialogue[0].get("delivery") == "remote",
            "explicit remote dialogue must survive even when the source is not a canonical character")

    director = QwenDirector.__new__(QwenDirector)
    director._semantic_spoken_dialogue = list(mixin._semantic_spoken_dialogue)
    director._qwen_telemetry = {}
    director._recovery_events = []
    segments = director._extract_story_spoken_segments(
        story,
        director._semantic_spoken_dialogue,
        allowed_speakers=["Eli"],
    )
    _assert(len(segments) == 1 and segments[0].get("delivery") == "remote"
            and "Eli's sister" in next(iter(segments[0].get("source_speakers", set()))),
            f"the source remote-only voice was lost or reclassified: {segments!r}")

    import json
    director._reference_visual_context = {}
    payload = json.loads(director._shot_director_batch_user(
        story,
        [{"name": "Eli"}],
        [{"scene_id": "scene_001", "characters": ["Eli"], "description": story}],
    ))
    source_dialogue = payload["scenes"][0].get("source_dialogue", [])
    _assert(len(source_dialogue) == 1
            and source_dialogue[0].get("speaker") == "Eli's sister (hologram)"
            and source_dialogue[0].get("delivery") == "remote",
            f"shot prompt dialogue whitelist did not carry the audio-only speaker: {source_dialogue!r}")

    shots = [{
        "shot_id": "scene_001_shot_001", "scene_id": "scene_001", "characters": [],
        "duration_seconds": 5.2, "action": "Eli watches the hologram on the terminal.",
        "visual_prompt": "Eli watches the dim hologram shimmer above the terminal in the abandoned station.",
        "dialogue_events": [{"speaker": "Eli's sister (hologram)", "text": "I left you a key.", "delivery": "remote"}],
    }]
    scenes = [{"scene_id": "scene_001", "characters": ["Eli"], "description": story}]
    characters = [{"name": "Eli", "character_id": "character-eli"}]
    director._normalize_dialogue_speakers(story, scenes, shots, characters)
    director._validate_dialogue_speaker_contract(shots, characters)
    director._validate_shot_character_contract(shots, characters)
    _assert(shots[0]["characters"] == [] and "Eli's sister (hologram)" not in scenes[0]["characters"],
            "a remote-only speaker was incorrectly promoted into visible shot/scene character bindings")
    _assert(len(shots[0]["dialogue_events"]) == 1
            and shots[0]["dialogue_events"][0].get("delivery") == "remote",
            f"remote-only speech did not survive deterministic dialogue reconciliation: {shots[0]!r}")

    plan = {"characters": characters, "shots": shots, "scenes": scenes}
    timeline = DialogueTimeline(characters)
    timeline.apply_to_plan(plan)
    event = plan["shots"][0]["dialogue_events"][0]
    _assert(event.get("delivery") == "remote" and event.get("speaker_id", "").startswith("remote_voice_"),
            f"timeline did not create a transient audio-only speaker binding: {event!r}")
    _assert(len(plan["characters"]) == 1 and plan["characters"][0]["name"] == "Eli",
            "timeline's transient remote speaker binding leaked into plan.characters")
    from pipeline.context_ir import H3ContextIRCompiler
    lines = H3ContextIRCompiler._dialogue_lines({"language": "English"}, plan["shots"][0])
    _assert(len(lines) == 1 and "remote channel" in lines[0].lower()
            and "eli's sister" in lines[0].lower(),
            f"Context IR lost the audio-only speaker identity or remote delivery: {lines!r}")


def test_bare_radio_prop_does_not_make_following_live_quote_remote():
    from planner.qwen_director import QwenDirector
    segments = QwenDirector._extract_story_spoken_segments(
        'A radio sat on the shelf. “I locked the vault,” Dr. Elias said.',
        [{"text": "I locked the vault", "speaker": "Dr. Elias"}],
        allowed_speakers={"Dr. Elias"},
    )
    _assert(len(segments) == 1 and segments[0]["source_speakers"] == {"dr. elias"},
            f"a bare radio mention must not classify live dialogue as remote: {segments!r}")


def test_dialogue_speaker_aliases_and_duplicate_lines_are_reconciled_structurally():
    from planner.qwen_director import QwenDirector
    semantic = [
        {"text": "Stay here", "speaker": "Mara Voss"},
        {"text": "Stay here", "speaker": "Tomas"},
    ]
    segments = QwenDirector._extract_story_spoken_segments(
        'Mara: Stay here\nTomas: Stay here',
        semantic,
        allowed_speakers={"Mara Voss", "Tomas"},
        speaker_aliases={"mara": "Mara Voss"},
    )
    _assert(len(segments) == 2, f"duplicate semantic dialogue lines were not preserved: {segments}")
    _assert(segments[0]["source_speakers"] == {"mara voss"}, f"short speaker label did not resolve through roster alias: {segments}")
    _assert(segments[1]["source_speakers"] == {"tomas"}, f"second duplicate line lost its semantic speaker: {segments}")


def test_honorific_speaker_normalization_keeps_in_person_dialogue_and_filters_remote_voice():
    from planner.qwen_director import QwenDirector
    from planner.entity_resolver import EntityResolver
    story = (
        'Lena\'s radio crackled. “Voss, you still there?” It was Dr. Renn, his voice taut. '
        '“Don’t open that vault,” he said.\n\n'
        'At the main entrance, Renn appeared, his face lit by the emergency lights. '
        '“You’ve got ten minutes before the storm breaches the outer shell,” he said, gripping her arm.\n\n'
        'Renn yanked the ignition key from the dashboard. “No one leaves with that data,” he said, tossing the key into the wind.'
    )
    semantic = [
        {"text": "Voss, you still there?", "speaker": "Dr. Renn"},
        {"text": "Don’t open that vault", "speaker": "Dr. Renn"},
        {"text": "You’ve got ten minutes before the storm breaches the outer shell,", "speaker": "Dr. Renn"},
        {"text": "No one leaves with that data,", "speaker": "Dr. Renn"},
    ]
    aliases = EntityResolver.build_character_alias_map([{"name": "Renn", "identity_type": "named_character"}])
    segments = QwenDirector._extract_story_spoken_segments(
        story, semantic, allowed_speakers=["Renn"], speaker_aliases=aliases,
    )
    texts = [segment["display"] for segment in segments]
    speakers = [segment["source_speakers"] for segment in segments]
    _assert(len(segments) == 4, f"expected two remote-channel and two in-person Renn utterances: {segments}")
    _assert(segments[0].get("delivery") == "remote" and segments[1].get("delivery") == "remote",
            f"live radio turns must be kept as remote audio: {segments!r}")
    _assert(all(segment.get("delivery") == "in_person" for segment in segments[2:]),
            f"remote-channel context leaked past the character's physical entrance: {segments!r}")
    _assert(any("You’ve got ten minutes before the storm breaches the outer shell," == text for text in texts),
            f"in-person entrance dialogue was dropped: {texts}")
    _assert("No one leaves with that data," in texts, f"in-person final-scene dialogue was dropped: {texts}")
    _assert(all(value == {"renn"} for value in speakers), f"title-bearing speaker was not canonicalized: {speakers}")


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


# ── Prompt assembly behavior ──────────────────────────────────────────────

def test_shot_planning_allows_empty_scene_binding_with_closed_roster():
    from planner.qwen_director import QwenDirector
    mixin = QwenDirector.__new__(QwenDirector)
    mixin._reference_visual_context = {}
    characters = [
        {"name": "Sara Khan", "role": "engineer"},
        {"name": "Eli Stone", "role": "scientist"},
    ]
    scenes = [{
        "scene_id": "scene_004",
        "description": "She reaches the control panel and studies the warning lights.",
        "characters": [],
    }]
    payload = mixin._shot_director_batch_user(
        "She reaches the control panel and studies the warning lights.",
        characters, scenes,
    )
    _assert("Sara Khan" in payload and "Eli Stone" in payload,
            "shot planning must still receive the closed canonical roster")
    _assert('"scene_id":"scene_004"' in payload,
            "empty-bound scene must still be represented in the shot payload")


def test_shot_context_preserves_paragraph_boundaries():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    story = "\n\n".join(
        f"Paragraph {index} opens with the setup. Paragraph {index} ends with its decisive turn."
        for index in range(1, 7)
    )
    compact = QwenDirectorPromptMixin._compact_story_context(story, 240)
    _assert("P1:" in compact and "P6:" in compact, "compact story context dropped narrative endpoints")


def test_shot_context_does_not_split_honorific_from_name():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    paragraphs = []
    for i in range(1, 7):
        paragraphs.append(
            f"Dr. Lena Voss entered chamber {i} after the pressure alarms began. "
            f"She checked the seal and read the damaged gauge while Eli watched the corridor. "
            f"A pressure seal failed, so Lena grabbed the valve and stopped the leak in chamber {i}. "
            f"The diver left chamber {i} with the valve locked and the repair recorded for the next crew."
        )
    compact = QwenDirectorPromptMixin._compact_story_context("\n\n".join(paragraphs), 1800)
    _assert("Dr. Lena Voss" in compact, "shot context split a title from the character name")
    _assert("P1:" in compact and "P6:" in compact, "title-safe compaction must preserve paragraph anchors")


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


def test_honorific_and_active_short_name_do_not_drop_character():
    planner = _planner()
    story = "Dr. Lena Voss’s voice cut through the murk. She hovered beside him. Lena’s jaw tightened."
    _assert(planner._story_has_character_name(story, "Dr. Lena Voss"), "title-free story usage must match canonical full name")
    _assert(not planner._is_mention_only_identity(story, "Dr. Lena Voss"), "active first-name usage must override a mention-only full-name occurrence")
    _assert(not planner._is_mention_only_identity(story, "Lena Voss", aliases=["Dr. Lena Voss", "Lena", "Dr. Voss"]),
            "Qwen alias evidence must preserve an actively present identity")
    recorded_only = "Dr. Lena Voss's recorded voice crackled from the console."
    _assert(planner._is_mention_only_identity(recorded_only, "Dr. Lena Voss"),
            "a recorded voice alone must not count as physical character presence")
    result = planner._reconcile_semantic_characters(story, {"candidates": [
        {"name": "Dr. Lena Voss", "entity_type": "PERSON", "is_character": True,
         "aliases": ["Lena", "Dr. Voss"], "identity_type": "named_character"}], "spoken_dialogue": []})
    _assert(result == ["Lena Voss"], f"active Lena Voss was filtered from canonical roster: {result}")


def test_sound_and_ui_words_are_not_deterministic_characters():
    planner = _planner()
    story = 'Elara called out, "Anyone?" Static answered. Silence fell. Access granted. Elara Voss stepped back.'
    names = planner.detect_character_descriptors(story)
    _assert("Static" not in names and "Silence" not in names and "Access" not in names, f"non-person surface leaked: {names}")
    _assert("Elara Voss" in names, f"real character lost: {names}")


def test_descriptive_sanitizer_preserves_qualified_identity():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    sanitized = director._sanitize_characters([{
        "name": "the man in the grey coverall",
        "identity_type": "descriptive_character",
        "semantic_aliases": ["man"],
    }])
    _assert(sanitized and sanitized[0]["identity_type"] == "descriptive_character", "qualified descriptive identity was rejected")


def test_descriptive_grounding_tolerates_minor_surface_variation():
    planner = _planner()
    story = "Lena watched the man in the grey coverall lift the gate. The man in the grey coverall nodded."
    _assert(
        planner._descriptive_identity_is_grounded(story, "the man in grey coverall"),
        "descriptive grounding should tolerate article/determiner variation",
    )


def test_descriptive_hints_populate_metadata_without_adding_cast():
    planner = _planner()
    story = "Lena watched the man in the grey coverall lift the gate."
    metadata = planner._semantic_character_metadata(
        story, ["the man in the grey coverall"], None,
        relational_hints=[], descriptive_hints=["the man in the grey coverall"],
    )
    from planner.entity_resolver import EntityResolver
    info = metadata.get(EntityResolver.normalize("the man in the grey coverall"))
    _assert(info and info.get("identity_type") == "descriptive_character",
            f"descriptive fallback metadata was not populated: {metadata}")


def test_identity_presence_accepts_title_free_and_given_name_surfaces():
    from planner.production_planner import ProductionPlanner as P
    story = (
        "Dr. Lena Voss's recorded voice crackled from the console. "
        "Lena grabbed Eli's wrist. \"Do not open it,\" Lena said."
    )
    _assert(P.identity_presence(story, "Dr. Lena Voss") == "present",
            "active given-name occurrence must override a title-bearing recording reference")
    _assert(P.identity_presence("Elara Voss entered the vault and shut the door.", "Dr. Elara Voss") == "present",
            "title-free full-name construction must match")
    _assert(P.identity_presence("Dr. Elias Kwan's final recording played; Kwan's log remained sealed.", "Dr. Elias Kwan") == "mention_only",
            "recording/document references alone must not imply physical presence")
    _assert(P.identity_presence("Mira opened the door.", "Dr. Kaelen Mira") == "absent",
            "surname-only overlap must not imply identity presence")


# ── Story pipeline behavior ───────────────────────────────────────────────

def test_short_visual_prompts_are_normalized_to_15_40_words_without_model_call():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    scene = {
        "scene_id": "scene_001", "description": "Lena studies a sealed vault beneath cold white emergency lights.",
        "lighting": "cold white emergency lights", "color_temperature": "cool", "mood": "tense",
        "location": "vault chamber", "characters": ["Lena"],
    }
    shot = {
        "scene_id": "scene_001", "shot_id": "scene_001_shot_001", "location": "vault chamber",
        "action": "Lena studies the sealed control panel.", "visual_prompt": "Lena watches the sealed vault.",
        "characters": ["Lena"], "dialogue_events": [],
    }
    long_shot = {
        **shot,
        "shot_id": "scene_001_shot_002",
        "visual_prompt": " ".join(["Lena"] + [f"detail{index}" for index in range(1, 52)]),
    }
    sanitized = director._sanitize_shots([shot, long_shot], scene, {"lena"})
    _assert(len(sanitized) == 2, "visual prompt normalization must not discard valid shots")
    for candidate in sanitized:
        words = re.findall(r"[\w'’-]+", candidate["visual_prompt"])
        _assert(15 <= len(words) <= 40,
                f"visual prompt violates 15-40 word contract for {candidate['shot_id']}: {len(words)} words")


def test_story_prompt_is_open_cast_and_fully_filled():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE, DIRECTOR_STORY_SCENES
    q = QwenDirector.__new__(QwenDirector)
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = q._story_text_system(mode)
        _assert("__" not in text, f"{mode}: unfilled length placeholder in prompt")
        _assert(f"exactly {DIRECTOR_STORY_SCENES} explicitly marked scene blocks" in text
                and "[[SCENE_01]]" in text and "[[SCENE_12]]" in text,
                f"{mode}: explicit scene-boundary contract missing from prompt")
        low = text.lower()
        _assert("names" in low and "relationships" in low, f"{mode}: Qwen must own cast, names and relationships")
        from planner.config import DIRECTOR_STORY_CAST_SIZE_POLICY, RUNTIME
        _assert(DIRECTOR_STORY_CAST_SIZE_POLICY == "plot_driven"
                and RUNTIME["director"]["story_contract"].get("cast_size_policy") == "plot_driven",
                f"{mode}: cast-size policy must be plot-driven and runtime-authoritative")
        _assert("no fixed minimum" in low and "no fixed" in low and "maximum" in low,
                f"{mode}: prompt must explicitly reject a fixed numeric cast quota")
        _assert("physically" in low and "dialogue" in low,
                f"{mode}: prompt must require active character participation and interactions")
        _assert("word counts" in low and "postscript" in low,
                f"{mode}: prompt must forbid count reports and output after the story")
        from planner.config import STORY_MIN_WORDS, STORY_WRITER_WORDS_PER_SCENE
        _assert(f"never under {STORY_MIN_WORDS}" in low,
                f"{mode}: system prompt must match the configured minimum, not a stale word floor")
        _assert(f"words total and never under {STORY_MIN_WORDS} usable story words" in q._story_text_user(
                    mode, "premise", story_blueprint="CAST: Ada\nSCENE 01: beat").lower()
                and str(STORY_WRITER_WORDS_PER_SCENE) in q._story_text_user(
                    mode, "premise", story_blueprint="CAST: Ada\nSCENE 01: beat")
                and "continuity lock" in q._story_text_user(
                    mode, "premise", story_blueprint="CAST: Ada\nSCENE 01: beat").lower(),
                f"{mode}: writer must guard minimum scene development and reconcile continuity")
        blueprint = q._story_blueprint_system(mode).lower()
        _assert("no fixed minimum" in blueprint and "maximum" in blueprint,
                f"{mode}: Architect must use a plot-driven, unbounded cast size")
        _assert(("physically present" in blueprint or "physically active" in blueprint)
                and "at least two beats" in blueprint,
                f"{mode}: architecture prompt must seed active characters and interactions")
        _assert("continuity ledger" in blueprint and "irreversible action" in blueprint,
                f"{mode}: architect must reconcile irreversible actions and character/object states")


def test_shot_sampler_uses_runtime_manifest_values():
    from planner.config import (
        DIRECTOR_SHOT_TEMPERATURE,
        DIRECTOR_SHOT_TOP_P,
        RUNTIME,
    )
    from planner.qwen_director_prompts import QwenDirectorPromptMixin

    configured = RUNTIME["director"]["story_sampling"]["shots"]
    _assert(DIRECTOR_SHOT_TEMPERATURE == float(configured["temperature"]),
            "shot temperature must come from runtime_versions.yaml")
    _assert(DIRECTOR_SHOT_TOP_P == float(configured["top_p"]),
            "shot top_p must come from runtime_versions.yaml")
    sampler = QwenDirectorPromptMixin()
    _assert(sampler._shot_sampling() == (DIRECTOR_SHOT_TEMPERATURE, DIRECTOR_SHOT_TOP_P),
            "shot sampler must return the YAML-resolved values")


def test_story_writer_minimum_output_tokens_uses_runtime_manifest():
    from planner.config import (
        DIRECTOR_STORY_MAX_TOKENS,
        DIRECTOR_STORY_WRITER_MAX_TOKENS,
        DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
        RUNTIME,
    )

    writing = RUNTIME["director"]["story_sampling"]["writing"]
    _assert(
        DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS
        == int(writing["minimum_output_tokens"]) == 0,
        "writer minimum-output floor must remain disabled; min_tokens forces EOS suppression, not useful prose",
    )
    _assert(
        0 <= DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS
        <= min(DIRECTOR_STORY_MAX_TOKENS, DIRECTOR_STORY_WRITER_MAX_TOKENS),
        "writer minimum-output floor must fit within the effective completion ceiling",
    )


def test_story_thinking_and_writing_samplers_reach_vllm_payload():
    from planner.qwen_director import QwenDirector
    from planner.config import (
        DIRECTOR_STORY_PRESENCE_PENALTY,
        DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
    )

    class FakeResponse:
        status_code = 200
        text = ""
        def json(self):
            return {"id": "test"}

    class FakeSession:
        def __init__(self):
            self.calls = []
        def post(self, url, json, timeout):
            self.calls.append({"url": url, "payload": json, "timeout": timeout})
            return FakeResponse()

    director = QwenDirector.__new__(QwenDirector)
    director._vllm_session = FakeSession()
    director._vllm_model_name = lambda: "qwen3-14b-awq"
    director._vllm_base_url = lambda: "http://127.0.0.1:8000/v1"
    director._post_chat(
        messages=[{"role": "user", "content": "Test bounded story architect."}],
        temperature=0.6, top_p=0.95, max_tokens=2200,
        enable_thinking=True, creative=False, thinking_token_budget=1280,
    )
    director._post_chat(
        messages=[{"role": "user", "content": "Test literary story writer."}],
        temperature=0.7, top_p=0.8, max_tokens=3000,
        min_tokens=DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
        enable_thinking=False, creative=True,
    )
    _assert(len(director._vllm_session.calls) == 2, "architect and writer payloads should be issued independently")
    plan_payload = director._vllm_session.calls[0]["payload"]
    _assert(plan_payload["chat_template_kwargs"]["enable_thinking"] is True,
            f"architect must explicitly enable Qwen3 thinking: {plan_payload}")
    _assert(plan_payload["temperature"] == 0.6 and plan_payload["top_p"] == 0.95
            and plan_payload["top_k"] == 20 and plan_payload["min_p"] == 0.0,
            f"Qwen3 thinking sampler not passed through: {plan_payload}")
    _assert(plan_payload["thinking_token_budget"] == 1280,
            f"architect reasoning must have a hard budget: {plan_payload}")
    _assert("presence_penalty" not in plan_payload,
            f"planning pass should not apply the prose repetition penalty: {plan_payload}")

    write_payload = director._vllm_session.calls[1]["payload"]
    _assert(write_payload["chat_template_kwargs"]["enable_thinking"] is False,
            f"prose writer must explicitly disable thinking: {write_payload}")
    _assert(write_payload["temperature"] == 0.7 and write_payload["top_p"] == 0.8
            and write_payload["top_k"] == 20 and write_payload["min_p"] == 0.0,
            f"Qwen3 non-thinking sampler not passed through: {write_payload}")
    _assert(write_payload["presence_penalty"] == DIRECTOR_STORY_PRESENCE_PENALTY == 1.5,
            f"AWQ prose writer should retain Qwen's official quantized-model penalty default: {write_payload}")
    _assert("min_tokens" not in write_payload,
            f"zero writer minimum-output floor must be omitted from the vLLM payload: {write_payload}")


def test_story_tail_cleanup_removes_metadata_and_duplicate_blocks():
    from planner.qwen_director import QwenDirector
    body = [
        f"Scene {i} paragraph describes an active character taking a concrete action that changes the situation and causes the next event. "
        + "The other named character objects and reveals a conflicting objective through a direct exchange. " * 2
        for i in range(1, 12)
    ]
    duplicate = "\n\n".join(body + body)
    cleaned, reason = QwenDirector._strip_story_nonprose_tail(duplicate)
    _assert(reason == "removed_exact_repeated_story_block", f"exact duplicated narrative block not detected: {reason}")
    _assert(cleaned == "\n\n".join(part.strip() for part in body), "duplicate block sanitizer must preserve first story and discard only repetition")

    with_postscript = "\n\n".join(body[:10] + ["---", "Note: This story meets every requirement.", "Total: **1020 words**"])
    cleaned, reason = QwenDirector._strip_story_nonprose_tail(with_postscript)
    _assert(reason == "removed_nonprose_postscript", f"prose-only postscript not detected: {reason}")
    _assert(cleaned == "\n\n".join(part.strip() for part in body[:10]), "postscript sanitizer changed story or kept checklist text")


def test_story_dialogue_surface_does_not_confuse_labels_with_speech():
    from planner.qwen_director import QwenDirector
    _assert(not QwenDirector._story_has_explicit_dialogue_surface('The door was marked “Project Permafrost.” A tag read “Subject 12.”'),
            "written tags and labels must not trigger a missing-spoken-dialogue recovery")
    _assert(QwenDirector._story_has_explicit_dialogue_surface('Elena said, “Do not open the vault.”'),
            "attributed in-person dialogue should be recognized")
    _assert(not QwenDirector._extract_story_spoken_segments(
        'A voice boomed through the chamber: “You have three minutes to evacuate.”',
        [{"text": "You have three minutes to evacuate", "speaker": "Eli"}],
        allowed_speakers={"Eli"},
    ), "an anonymous voice must not be attributed to the protagonist")


def test_generated_story_cast_contract_is_plot_driven_without_a_numeric_quota():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE, DIRECTOR_STORY_CAST_SIZE_POLICY, RUNTIME

    contract = RUNTIME["director"]["story_contract"]
    _assert(DIRECTOR_STORY_CAST_SIZE_POLICY == "plot_driven", "cast policy must be plot-driven")
    _assert("minimum_active_characters" not in contract and "target_active_characters" not in contract,
            "runtime contract must not define a fixed cast minimum or target")

    # Small and large casts are both structurally valid; no artificial 3–4-person cap.
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


def test_story_blueprint_normalization_accepts_dynamic_cast_sizes_and_hyphenated_names():
    from planner.qwen_director import QwenDirector
    from planner.config import DIRECTOR_STORY_SCENES
    director = QwenDirector.__new__(QwenDirector)
    beats = "\n".join(
        f"SCENE {i:02d}: Beat {i} changes the story and carries the state forward."
        for i in range(1, DIRECTOR_STORY_SCENES + 1)
    )
    for size in (1, 2, 4, 9, 18):
        entries = [f"Anne-Marie {i} (role {i}; distinct motive)" for i in range(1, size + 1)]
        raw = "CAST: " + " | ".join(entries) + "\n" + beats
        normalized = director._normalize_story_blueprint(raw, DIRECTOR_STORY_SCENES)
        _assert(normalized.splitlines()[0].count("|") == size - 1,
                f"CAST normalization lost characters for dynamic cast size {size}")
    duplicate = "CAST: Anne-Marie (protagonist) | Anne-Marie (duplicate)\n" + beats
    try:
        director._normalize_story_blueprint(duplicate, DIRECTOR_STORY_SCENES)
    except RuntimeError as exc:
        _assert("duplicate names" in str(exc).lower(), f"wrong duplicate-cast error: {exc}")
    else:
        raise AssertionError("duplicate character names must remain invalid")


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
    _assert(calls[0].get("max_completion") == DIRECTOR_STORY_PLAN_MAX_TOKENS == 2600,
            f"architect must use its configured hard output ceiling: {calls[0]}")
    _assert(calls[0].get("temperature") == DIRECTOR_STORY_THINKING_TEMPERATURE == 0.6
            and calls[0].get("top_p") == DIRECTOR_STORY_THINKING_TOP_P == 0.95,
            f"architect must use the official Qwen3 thinking sampler: {calls[0]}")
    _assert(blueprint.count("SCENE ") == DIRECTOR_STORY_SCENES,
            "normalized blueprint should include all configured scenes")


def test_story_minimum_word_contract_fails_closed_without_retry():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES, STORY_ACCEPT_MIN_WORDS, STORY_MIN_WORDS
    from contextlib import redirect_stdout
    from io import StringIO
    temperature, top_p = QwenDirector.__new__(QwenDirector)._sampling_for_mode(AI_STORY_MODE)

    def run(words_per_scene):
        paras = [" ".join([f"Ines{k}"] + [f"w{k}x{i}" for i in range(words_per_scene - 1)]) + "." for k in range(DIRECTOR_STORY_SCENES)]
        director = QwenDirector.__new__(QwenDirector)
        director._story_min_words = STORY_ACCEPT_MIN_WORDS
        calls = []
        def fake_chat(*a, **k):
            calls.append(k)
            return "\n\n".join(paras)
        director._chat_text = fake_chat
        # Isolate the hard word-count contract from semantic premise-coverage
        # checks; those have their own generation-contract tests.
        director._validate_story_output_contracts = lambda _mode, _input, candidate, **_kwargs: candidate
        with redirect_stdout(StringIO()):
            out = director._generate_story_once(
                AI_STORY_MODE, "premise", "system", "user",
                temperature=temperature, top_p=top_p, source_character_names=[],
            )
        return out, calls

    # 68 words/scene = 816 words: exactly the minimum production contract.
    (story, generated), calls = run(68)
    _assert(generated and len(calls) == 1, f"minimum-length story must pass with one call: {len(calls)}")
    _assert(len(re.findall(r"\b[\w'’-]+\b", story)) >= STORY_MIN_WORDS, "accepted story fell below minimum")
    # 65 words/scene = 780 words: reject before downstream work, never retry.
    try:
        run(65)
    except RuntimeError as exc:
        _assert(str(STORY_ACCEPT_MIN_WORDS) in str(exc) or "minimum production contract" in str(exc),
                f"unexpected rejection: {exc}")
    else:
        raise AssertionError("under-length story must fail closed before shot generation")
    _assert(STORY_ACCEPT_MIN_WORDS == STORY_MIN_WORDS, "acceptance floor must equal the production minimum")



def test_story_completion_contract_rejects_unresolved_final_hook():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES, STORY_MIN_WORDS
    director = QwenDirector.__new__(QwenDirector)
    director._story_min_words = STORY_MIN_WORDS
    paragraphs = []
    for scene_index in range(DIRECTOR_STORY_SCENES):
        words = [f"scene{scene_index+1}"] + [f"grounded{scene_index}_{i}" for i in range(69)]
        paragraphs.append(" ".join(words) + ".")
    paragraphs[-1] = paragraphs[-1][:-1] + " The team would discover a new threat beyond the sealed door."
    story = "\n\n".join(paragraphs)
    try:
        director._validate_story_completion_contract(AI_STORY_MODE, story)
    except RuntimeError as exc:
        _assert("unresolved future hook" in str(exc).lower() and "sentence count" in str(exc).lower(),
                f"completion gate must catch both unresolved ending and scene-depth violations: {exc}")
    else:
        raise AssertionError("a future-hook ending must fail the single-pass completion contract")


def test_story_topology_fails_closed_when_sentence_boundaries_cannot_make_scenes():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, STORY_MIN_WORDS
    story = " ".join(f"word{index}" for index in range(STORY_MIN_WORDS)) + "."
    director = QwenDirector.__new__(QwenDirector)
    director._story_min_words = STORY_MIN_WORDS
    calls = []
    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return story
    director._chat_text = fake_chat
    temperature, top_p = director._sampling_for_mode(AI_STORY_MODE)
    try:
        director._generate_story_once(
            AI_STORY_MODE, "premise", "system", "user",
            temperature=temperature, top_p=top_p, source_character_names=[],
        )
    except RuntimeError as exc:
        _assert("scene boundaries were preserved" in str(exc).lower()
                and "no split, merge" in str(exc).lower(),
                f"unexpected topology failure: {exc}")
    else:
        raise AssertionError("story without sufficient sentence boundaries must not be accepted")
    _assert(len(calls) == 1, f"topology failure must not trigger retry, got {len(calls)} calls")


def test_story_uses_one_architect_and_one_author_without_retry_or_repair():
    """Normal generation must have one thinking Architect and one non-thinking Author."""
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


def test_author_is_one_nonthinking_call_and_receives_planned_cast():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_WRITER_MAX_TOKENS, STORY_ACCEPT_MIN_WORDS
    from contextlib import redirect_stdout
    from io import StringIO
    from planner.qwen_director_prompts import QwenDirectorPromptMixin

    blueprint = (
        "CAST: Ines Vale (protagonist; save the archive; precise) | "
        "Tomas Reed (ally; expose the theft; blunt) | "
        "Mara Quinn (opponent; protect her family; guarded)\n" +
        "\n".join(f"SCENE {i:02d}: Ines chooses an action with Tomas and Mara, changing the stakes." for i in range(1, 13))
    )
    mixin = QwenDirectorPromptMixin()
    writer_user = mixin._story_text_user(
        AI_STORY_MODE, "The archive is threatened.", story_blueprint=blueprint
    )
    _assert("CAST: Ines Vale" in writer_user and "SCENE 12:" in writer_user,
            "the normalized CAST and all 12 beats must reach the prose writer")
    _assert("6–8 complete sentences" in writer_user and "protagonist's immediate outcome" in writer_user
            and "Do not end with a future hook" in writer_user,
            "writer contract must require scene depth and an on-page completed ending")

    paragraphs = [" ".join([f"Scene{i}", "Ines", "and", "Tomas", "acted."] + [f"word{i}_{j}" for j in range(75)]) + "."
                  for i in range(12)]
    director = QwenDirector.__new__(QwenDirector)
    director._story_min_words = STORY_ACCEPT_MIN_WORDS
    calls = []
    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return "\n\n".join(paragraphs)
    director._chat_text = fake_chat
    director._validate_story_output_contracts = lambda _mode, _input, candidate, **_kwargs: candidate
    with redirect_stdout(StringIO()):
        story, generated = director._generate_story_once(
            AI_STORY_MODE, "premise", mixin._story_text_system(AI_STORY_MODE), writer_user,
            temperature=0.7, top_p=0.8, source_character_names=[], story_blueprint=blueprint,
        )
    _assert(generated and len(calls) == 1, f"Author must make exactly one request, got {len(calls)}")
    call = calls[0]
    _assert(call.get("disable_thinking") is True, f"Author must run with thinking off: {call}")
    _assert(call.get("max_completion") == min(6500, DIRECTOR_STORY_WRITER_MAX_TOKENS) == 3000,
            f"Author must use the dedicated writer completion ceiling: {call}")
    _assert("thinking_token_budget" not in call,
            "Author request must not reuse the Architect thinking budget")


def test_story_plan_uses_richer_beats_and_required_cast_line():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, STORY_PLAN_BEAT_MIN_WORDS, STORY_PLAN_BEAT_MAX_WORDS
    prompt = QwenDirectorPromptMixin()._story_blueprint_system(AI_STORY_MODE)
    _assert(STORY_PLAN_BEAT_MIN_WORDS == 45 and STORY_PLAN_BEAT_MAX_WORDS == 60,
            "Architect beat targets must be 45–60 words from runtime YAML")
    _assert("45–60 words" in prompt, "Architect prompt must request richer 45–60-word beats")
    lowered = prompt.lower()
    _assert("cast:" in lowered and "at least six" in lowered,
            "Architect prompt must specify a CAST line and dialogue anchors across at least six beats")


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
    _assert(STORY_MIN_WORDS == int(DIRECTOR_STORY_SCENES * 85 * 0.8) == 816,
            f"story minimum must match prompt's 80% target: {STORY_MIN_WORDS}")
    _assert(count == DIRECTOR_STORY_SCENES, f"paragraphs must be normalized to {DIRECTOR_STORY_SCENES}, got {count}")


def test_topology_is_driven_by_scene_and_shot_settings():
    from planner.cinematic_compiler import CinematicCompiler
    from planner.config import DIRECTOR_STORY_SCENES, DIRECTOR_SHOTS_PER_SCENE
    from planner.qwen_director import QwenDirector
    _assert(QwenDirector.MAX_SCENES == DIRECTOR_STORY_SCENES, "MAX_SCENES must follow DIRECTOR_STORY_SCENES")
    _assert(QwenDirector.SHOTS_PER_SCENE == DIRECTOR_SHOTS_PER_SCENE, "SHOTS_PER_SCENE must follow config")
    scenes = [{"scene_id": f"scene_{i:03d}", "description": "Ines crosses the dock.", "location": "harbor dock",
               "characters": ["Ines"]} for i in range(1, DIRECTOR_STORY_SCENES + 1)]
    shots = CinematicCompiler(character_names={"Ines"}).compile_all(scenes, [])
    _assert(len(shots) == DIRECTOR_STORY_SCENES * DIRECTOR_SHOTS_PER_SCENE,
            f"fallback compile must yield scenes x shots, got {len(shots)}")


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


def test_batch_normalizer_reconstructs_deterministic_shot_ids():
    from planner.qwen_director_sanitize import QwenDirectorSanitizeMixin
    normalized = QwenDirectorSanitizeMixin._normalize_batch_shot_response({
        "scene_shots": [{"scene_id": "scene_004", "shots": [{"action": "A"}, {"action": "B"}, {"action": "C"}]}]
    })
    shots = normalized["scene_004"]
    _assert([s["scene_id"] for s in shots] == ["scene_004"] * 3,
            "parent scene IDs must be reconstructed for compact shot JSON")
    _assert([s["shot_id"] for s in shots] == ["scene_004_shot_001", "scene_004_shot_002", "scene_004_shot_003"],
            f"shot IDs must be reconstructed in output order: {shots}")


def test_shot_batch_context_caps_are_largest_first_and_bounded():
    from planner.qwen_director import QwenDirector
    _assert(QwenDirector._shot_batch_story_context_capacities(3000) == [3000, 2400, 1800, 1200],
            "global recap context should compact in deterministic largest-first stages")
    _assert(QwenDirector._shot_batch_story_context_capacities(1000) == [1000],
            "context caps must not inflate a smaller runtime-configured maximum")


def test_sanitizer_restores_compact_batch_fields_from_scene_metadata():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    scene = {"scene_id": "scene_001", "location": "harbor office", "lighting": "cold window light",
             "color_temperature": "cool", "mood": "tense", "description": "Mara speaks in the harbor office."}
    raw = [{
        "scene_id": "scene_001", "shot_id": "scene_001_shot_001", "action": "Mara speaks at the desk.",
        "visual_prompt": "Mara speaks at a desk inside the harbor office under cold window light, framed tightly.",
        "characters": ["Mara"], "dialogue_events": [
            {"speaker": "Mara", "text": "Open the door.", "continues_from_previous_shot": False, "continues_to_next_shot": False},
            {"speaker": "Mara", "text": "We have no time.", "continues_from_previous_shot": False, "continues_to_next_shot": False},
        ],
        "camera_shot": "medium", "camera_movement": "static", "lens_and_depth_of_field": "normal lens",
        "composition_notes": "Mara centered against the desk.", "continuity_start_state": {"state_description": "At the desk."},
        "continuity_end_state": {"state_description": "Mara turns toward the door."},
    }]
    shots = director._sanitize_shots(raw, scene, {"mara"})
    _assert(len(shots) == 1, f"sanitizer unexpectedly rejected compact shot: {shots}")
    _assert(shots[0]["speech_text"] == "Open the door. We have no time.",
            f"speech_text must contain each authoritative utterance once: {shots[0].get('speech_text')!r}")
    _assert(shots[0]["location"] == "harbor office" and shots[0]["lighting"] == "cold window light"
            and shots[0]["color_temperature"] == "cool",
            "per-shot place/light fields must be restored from canonical scene metadata")
    _assert(all(key in shots[0] for key in (
        "character_spatial_bboxes", "character_spatial_regions", "character_spatial_bboxes_start",
        "character_spatial_bboxes_end", "character_spatial_regions_start", "character_spatial_regions_end")),
        "canonical spatial fields must remain present after compact model output")


def test_shot_batch_completion_budget_uses_runtime_topology():
    from planner.config import DIRECTOR_MAX_TOKENS, DIRECTOR_SHOTS_PER_SCENE
    from planner.qwen_director import QwenDirector
    def expected(scene_count):
        count = max(1, int(scene_count))
        return min(int(DIRECTOR_MAX_TOKENS), max(320, 600 * int(DIRECTOR_SHOTS_PER_SCENE) * count))
    _assert(QwenDirector._shot_batch_completion_budget(1) == expected(1), "one-scene shot batch budget ignored YAML topology")
    _assert(QwenDirector._shot_batch_completion_budget(2) == expected(2), "two-scene shot batch budget ignored YAML topology")
    _assert(QwenDirector._shot_batch_completion_budget(0) == expected(1), "non-positive scene count must clamp to one")


def test_shot_batch_budget_and_prompt_compaction_create_more_two_scene_headroom():
    from planner.config import DIRECTOR_MAX_TOKENS, DIRECTOR_N_CTX, DIRECTOR_SHOTS_PER_SCENE
    from planner.qwen_director import QwenDirector
    from planner.qwen_director_prompts import SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT
    system_chars = len(SHOT_DIRECTOR_BATCH_SYSTEM_PROMPT)
    _assert(system_chars < 11000,
            f"shot batch system prompt regressed toward the historical 14,353-character prompt: {system_chars}")
    completion = QwenDirector._shot_batch_completion_budget(2)
    expected_completion = min(
        int(DIRECTOR_MAX_TOKENS),
        max(320, 600 * int(DIRECTOR_SHOTS_PER_SCENE) * 2),
    )
    prompt_limit = int(DIRECTOR_N_CTX) - 128 - completion
    historical_prompt_limit = int(DIRECTOR_N_CTX) - 128 - 4200
    _assert(completion == expected_completion,
            f"compact two-scene completion cap mismatch: {completion} != {expected_completion}")
    _assert(prompt_limit > historical_prompt_limit,
            f"two-scene completion reserve must leave more prompt context than the old 4,200-token reserve: {prompt_limit}")


def test_story_scene_coercion_keeps_quoted_dialogue_intact():
    from planner.qwen_director import QwenDirector
    examples = [
        ('Lena said, "Stop. Wait for me." Then she ran into the tunnel.', '"Stop. Wait for me."'),
        ('Eli opened the box. “Do not touch it. I mean it,” Mara said. The light changed.', '“Do not touch it. I mean it,” Mara said.'),
        ("Mara whispered, ‘Leave now. Do not look back.’ Then the door closed.", '‘Leave now. Do not look back.’'),
        ('He said, "Stop." Mara asked, "Why?" The alarm sounded.', '"Stop." Mara asked, "Why?"'),
    ]
    for source, expected_span in examples:
        normalized = QwenDirector._coerce_story_to_scene_paragraphs(source)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", normalized) if p.strip()]
        _assert(expected_span in normalized, f"quoted dialogue was altered: {normalized!r}")
        _assert(any(expected_span in paragraph for paragraph in paragraphs),
                f"quoted dialogue/speaker attribution was split across scene paragraphs: {paragraphs!r}")


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


# ── Runtime fingerprint / camera ──────────────────────────────────────────

def test_checkpoint_fingerprint_tracks_runtime_topology_and_frame_policy():
    from pipeline.production_checkpoint import ProductionCheckpoint
    from pipeline.production_orchestrator import ProductionOrchestrator
    keys = ("H3_DIRECTOR_STORY_SCENES", "H3_DIRECTOR_SHOTS_PER_SCENE", "H3_FPS", "H3_FRAMES_PER_SHOT")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        store = ProductionCheckpoint(ROOT)
        baseline = ProductionOrchestrator._director_fingerprint(ROOT, store)
        os.environ.update({"H3_DIRECTOR_STORY_SCENES": "19", "H3_DIRECTOR_SHOTS_PER_SCENE": "9",
                           "H3_FPS": "60", "H3_FRAMES_PER_SHOT": "141"})
        overridden = ProductionOrchestrator._director_fingerprint(ROOT, store)
        _assert(baseline == overridden,
                "environment variables must not override runtime-manifest story topology/frame settings")
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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


def test_cinematic_compiler_fallback_camera_vocabulary_is_canonical():
    from planner.cinematic_compiler import CinematicCompiler
    from planner.qwen_director_prompts import CAMERA_SHOT_OPTIONS, CAMERA_MOVEMENT_OPTIONS
    scene = {
        "scene_id": "scene_001",
        "description": "Lena watches the sealed vault under cold emergency lights.",
        "location": "vault chamber", "lighting": "cold emergency lights",
        "color_temperature": "cool", "mood": "tense", "characters": ["Lena"],
    }
    shots = CinematicCompiler(character_names={"Lena"}).compile_all([scene], [])
    _assert(bool(shots), "compiler fallback unexpectedly produced no shots")
    invalid_shots = [
        str(item.get("shot_id", ""))
        for item in shots
        if str(item.get("camera_shot", "")) not in CAMERA_SHOT_OPTIONS
        or str(item.get("camera_movement", "")) not in CAMERA_MOVEMENT_OPTIONS
    ]
    _assert(not invalid_shots, f"compiler fallback emitted noncanonical camera vocabulary: {invalid_shots!r}")


def test_camera_directive_in_action_is_replaced_from_scene_evidence():
    from planner.qwen_director import QwenDirector
    scene_description = (
        "The aftermath came in the form of a frozen silence. Lena’s body was found days later, encased in ice, the vault resealed. "
        "The storm passed, leaving the Arctic station untouched. The data was lost, but so was the vortex. "
        "The world moved on, unaware of the choice that had saved them. "
        "The compass remained in the vault, its needle finally still. The storm was gone. So was Lena."
    )
    director = QwenDirector.__new__(QwenDirector)
    scene = {
        "scene_id": "scene_006", "description": scene_description,
        "lighting": "dim ambient light from distant storm", "color_temperature": "cold blue",
        "mood": "resigned", "location": "abandoned Arctic station", "characters": ["Lena"],
    }
    shot = {
        "scene_id": "scene_006", "shot_id": "scene_006_shot_002",
        "location": "abandoned Arctic station", "characters": ["Lena"],
        "action": "The camera pans slowly across the frozen station, ending on the vault with the compass inside, its needle finally still.",
        "visual_prompt": "Frozen Arctic station with the sealed vault and compass beneath cold ambient storm light, empty and quiet.",
        "camera_shot": "wide", "camera_movement": "pan", "dialogue_events": [],
    }
    result = director._sanitize_shots([shot], scene, {"lena"})
    _assert(len(result) == 1, "sanitizer unexpectedly dropped a recoverable shot")
    action = result[0]["action"]
    _assert(not re.search(r"\b(?:the\s+)?camera\s+(?:pans?|tilts?|tracks?|dollies|pushes?|pulls?|cranes?|orbits?|zooms?)\b", action, re.I),
            f"camera direction remained in the narrative action: {action!r}")
    _assert("compass" in action.casefold() and "needle" in action.casefold(),
            f"grounded action fallback lost the shot's evidenced subject/state: {action!r}")


# ── Story buffer / critic / prompt-edge behavior ──────────────────────────

def test_unique_story_paragraph_recovers_unmatched_source_dialogue():
    import json
    from planner.qwen_director import QwenDirector
    story = (
        "Lena drove through the storm toward the station.\n\n"
        "Inside, Lena found the sealed vault and a cracked compass.\n\n"
        "A voice crackled through a long-dead speaker: “You shouldn’t be here.” "
        "Lena turned, startled, and found a man in a faded lab coat. He called himself Dr. Elias. "
        "He warned her that opening the vault would release something no one was prepared for. "
        "“The storm is the first warning,” he said, his voice frayed with urgency."
    )
    director = QwenDirector.__new__(QwenDirector)
    director._reference_visual_context = {}
    director._semantic_spoken_dialogue = [
        {"text": "You shouldn’t be here.", "speaker": "Dr. Elias"},
        {"text": "The storm is the first warning.", "speaker": "Dr. Elias"},
    ]
    payload = json.loads(director._shot_director_batch_user(
        story,
        [{"name": "Lena Korsakov"}, {"name": "Dr. Elias"}],
        [{"scene_id": "scene_003", "title": "At the vault", "location": "Arctic station",
          "description": "Lena finds an older scientist by the sealed vault.",
          "characters": ["Lena Korsakov", "Dr. Elias"]}],
    ))
    source_dialogue = payload["scenes"][0].get("source_dialogue", [])
    normalized = [
        (str(item.get("speaker", "")), re.sub(r"[^a-z0-9]+", " ", str(item.get("text", "")).casefold()).strip())
        for item in source_dialogue
    ]
    _assert(("Dr. Elias", "the storm is the first warning") in normalized,
            f"unmatched in-person dialogue was not recovered from its unique story paragraph: {source_dialogue!r}")
    remote_lines = [item for item in source_dialogue if "you shouldn t be here" in re.sub(r"[^a-z0-9]+", " ", str(item.get("text", "")).casefold()).strip()]
    _assert(len(remote_lines) == 1 and remote_lines[0].get("delivery") == "remote",
            f"live remote loudspeaker dialogue must be preserved as remote audio, not discarded or made in-person: {source_dialogue!r}")
    in_person_lines = [item for item in source_dialogue if "the storm is the first warning" in re.sub(r"[^a-z0-9]+", " ", str(item.get("text", "")).casefold()).strip()]
    _assert(len(in_person_lines) == 1 and in_person_lines[0].get("delivery", "in_person") == "in_person",
            f"the later in-person dialogue was not distinguished from remote speech: {source_dialogue!r}")


def test_exact_duplicate_terminal_sentence_is_removed_without_retry():
    from planner.qwen_director import QwenDirector
    repeated = "The station was quiet, and for the first time in years it was peaceful."
    story = (
        "The engineer reached the station door. She learned the failure had been deliberate. "
        "She chose to seal the vault. The storm finally weakened. "
        + repeated + " The records were lost, but the people survived. " + repeated
    )
    repaired, changed = QwenDirector._trim_repeated_terminal_sentence(story)
    _assert(changed, "exact duplicate final sentence was not detected")
    _assert(repaired.count(repeated) == 1, "exact duplicate final sentence was not removed exactly once")
    _assert(repaired.endswith("The records were lost, but the people survived."),
            f"prior resolution was not preserved: {repaired!r}")
    unchanged, changed = QwenDirector._trim_repeated_terminal_sentence("The station was quiet. The records survived.")
    _assert(not changed and unchanged.endswith("The records survived."), "non-repeated ending was edited")


def test_critic_filters_only_case_cross_scene_false_positives_and_noop_patches():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    plan = {
        "characters": [{"name": "Eli"}], "scenes": [],
        "shots": [
            {"shot_id": "scene_001_shot_001", "scene_id": "scene_001", "characters": ["Eli"],
             "camera_shot": "wide", "camera_movement": "static", "lighting": "blue", "location": "station", "action": "Eli enters."},
            {"shot_id": "scene_002_shot_001", "scene_id": "scene_002", "characters": ["Eli"],
             "camera_shot": "medium", "camera_movement": "pan", "lighting": "blue", "location": "station", "action": "Eli studies the lock."},
        ],
    }
    critique = {
        "overall_score": 6, "status": "review",
        "findings": [
            "Roster mismatch: shot character 'Eli' but roster lists 'eli' (case-insensitive mismatch).",
            "Action distinctness: scene_001_shot_001 and scene_002_shot_001 use similar action coverage across scenes.",
            "Fidelity mismatch: scene_001_shot_001 invents a visible explosion.",
        ],
        "shot_findings": [], "recommended_focus": [],
        "shot_patches": [
            {"shot_id": "scene_001_shot_001", "patch": {"camera_shot": "wide"}},
            {"shot_id": "scene_001_shot_001", "patch": {"camera_movement": "slow pan"}},
        ],
    }
    result = director._reconcile_critique(critique, plan)
    _assert(len(result["findings"]) == 1 and "explosion" in result["findings"][0],
            f"unsupported false-positive findings were not filtered safely: {result['findings']!r}")
    _assert(result["shot_patches"] == [{"shot_id": "scene_001_shot_001", "patch": {"camera_movement": "pan"}}],
            f"no-op patch or invalid camera value survived filtering: {result['shot_patches']!r}")


def test_director_critique_respects_manifest_default_and_explicit_opt_out():
    import os
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    calls = {"model": 0, "summary": 0}
    director._chat_json = lambda *a, **k: calls.__setitem__("model", calls["model"] + 1) or {}
    director._print_qwen_summary = lambda *a, **k: calls.__setitem__("summary", calls["summary"] + 1)
    saved = {key: os.environ.get(key) for key in ("H3_DIRECTOR_CRITIQUE", "H3_DIRECTOR_CRITIC")}
    try:
        from planner.config import director_critic_enabled
        os.environ.pop("H3_DIRECTOR_CRITIQUE", None)
        os.environ.pop("H3_DIRECTOR_CRITIC", None)
        _assert(director_critic_enabled(), "runtime manifest must enable the cinematic critic by default")
        os.environ["H3_DIRECTOR_CRITIQUE"] = "0"
        os.environ["H3_DIRECTOR_CRITIC"] = "1"
        result = director.critique_plan(mode="ai_story", user_input="p", plan={})
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    _assert(result is None, "a skipped critique must return None so orchestration does not recompile identical shots")
    _assert(calls == {"model": 0, "summary": 1}, f"critique opt-out should avoid Qwen and print final telemetry once: {calls}")


def main():
    tests = [
        # Character roster
        test_shot_schema_tracks_instance_topology,
        test_mention_only_matching_is_case_insensitive,
        test_sanitizer_deduplicates_normalized_identity_keys,
        test_build_alias_map_skips_stopword_first_token,
        test_characters_in_scene_does_not_bind_descriptive_by_article,
        test_sanitize_scene_alias_fallback_does_not_bind_descriptive_by_article,
        test_qwen_can_choose_any_cast_size,
        test_qwen_negative_is_authoritative,
        test_deterministic_detection_never_adds_missing_qwen_character,
        test_qwen_extractor_is_called_even_for_obvious_named_roster,
        test_mention_only_qwen_candidates_are_filtered_without_restoring_them,
        test_non_person_qwen_candidates_are_filtered,
        test_descriptive_alias_can_be_canonical_surface,
        test_relational_owner_must_be_qwen_approved,
        test_relational_character_survives_with_qwen_owner,
        test_no_adjudication_call_even_when_extractor_is_partial,
        test_expand_is_open_cast_but_preserves_source_anchors,
        test_character_schema_does_not_silently_cap_qwen_cast_at_32,
        test_qwen_extraction_does_not_force_candidates_from_hints_plan_or_dialogue,
        test_character_semantic_schema_includes_spoken_dialogue,
        test_in_person_dialogue_speaker_must_exist_in_qwen_cast,
        test_source_attribution_overrides_mislabelled_ui_quote_speaker,
        test_named_ai_voice_from_speakers_is_source_grounded_remote,
        test_unrostered_remote_speaker_without_source_grounding_is_dropped,
        test_unrostered_hologram_speaker_is_audio_only_not_a_crash,
        test_quote_inventory_preserves_repeated_quotes_and_punctuation,
        test_blueprint_cast_filter_excludes_weather_and_objects_but_keeps_dynamic_people,
        test_cast_static_check_is_alias_aware_and_does_not_require_fixed_recurrence,
        test_preserve_short_source_is_not_duplicated_into_fake_scenes,
        test_preserve_quality_contract_accepts_source_driven_single_scene_only_for_preserve_mode,
        test_watchdog_is_armed_per_qwen_http_request_not_whole_generation,
        test_source_dialogue_omitted_by_shot_model_is_restored_exactly_once,
        test_long_preserve_dialogue_maps_through_rebalanced_source_scenes,
        test_dialogue_cannot_drift_to_another_scene_and_is_restored_in_source_scene,
        test_unclassified_quote_with_explicit_named_attribution_is_recovered_without_retry,
        test_long_source_quote_inventory_keeps_all_quotes_and_bounds_context,
        test_story_markdown_emphasis_is_removed_without_cross_paragraph_consumption,
        # Dialogue contract
        test_story_dialogue_contract_uses_qwen_semantic_spans,
        test_consecutive_quote_speaker_carries_forward_after_pronoun_attribution,
        test_explicit_empty_shot_characters_are_not_filled_from_scene_cast,
        test_remote_only_shot_can_have_no_visible_source_body,
        test_remote_delivery_survives_timeline_and_context_ir_without_visible_source,
        test_unrostered_remote_speaker_survives_without_cast_addback,
        test_bare_radio_prop_does_not_make_following_live_quote_remote,
        test_dialogue_speaker_aliases_and_duplicate_lines_are_reconciled_structurally,
        test_honorific_speaker_normalization_keeps_in_person_dialogue_and_filters_remote_voice,
        test_dialogue_reconciliation_preserves_remote_audio_without_visible_binding,
        # Prompt assembly
        test_shot_planning_allows_empty_scene_binding_with_closed_roster,
        test_shot_context_preserves_paragraph_boundaries,
        test_shot_context_does_not_split_honorific_from_name,
        test_story_context_compactor_keeps_causal_middle_action,
        test_context_compaction_never_loses_closing_quote_or_splits_attribution,
        test_honorific_and_active_short_name_do_not_drop_character,
        test_sound_and_ui_words_are_not_deterministic_characters,
        test_descriptive_sanitizer_preserves_qualified_identity,
        test_descriptive_grounding_tolerates_minor_surface_variation,
        test_descriptive_hints_populate_metadata_without_adding_cast,
        test_identity_presence_accepts_title_free_and_given_name_surfaces,
        # Story pipeline
        test_story_prompt_is_open_cast_and_fully_filled,
        test_story_thinking_and_writing_samplers_reach_vllm_payload,
        test_shot_sampler_uses_runtime_manifest_values,
        test_story_writer_minimum_output_tokens_uses_runtime_manifest,
        test_story_tail_cleanup_removes_metadata_and_duplicate_blocks,
        test_story_dialogue_surface_does_not_confuse_labels_with_speech,
        test_generated_story_cast_contract_is_plot_driven_without_a_numeric_quota,
        test_story_blueprint_normalization_accepts_dynamic_cast_sizes_and_hyphenated_names,
        test_story_blueprint_normalization_requires_ordered_scene_beats,
        test_story_architect_uses_native_bounded_thinking_and_no_retry,
        test_story_minimum_word_contract_fails_closed_without_retry,
        test_story_completion_contract_rejects_unresolved_final_hook,
        test_story_topology_fails_closed_when_sentence_boundaries_cannot_make_scenes,
        test_story_writer_is_one_qwen3_nonthinking_call_with_configured_ceiling,
        test_story_uses_one_architect_and_one_author_without_retry_or_repair,
        test_author_is_one_nonthinking_call_and_receives_planned_cast,
        test_story_plan_uses_richer_beats_and_required_cast_line,
        test_topology_is_driven_by_scene_and_shot_settings,
        test_compact_shot_prompt_example_is_valid_json,
        test_compact_shot_batch_schema_omits_derived_and_empty_fields,
        test_source_dialogue_ids_are_stable_and_scene_scoped,
        test_repeated_dialogue_text_maps_to_the_exact_source_paragraph,
        test_scene_local_dialogue_assignments_are_unique_and_constrained,
        test_batch_normalizer_expands_scene_level_dialogue_assignments,
        test_shot_batch_dialogue_references_restore_exact_source_text_and_deduplicate,
        test_batch_normalizer_reconstructs_deterministic_shot_ids,
        test_shot_batch_context_caps_are_largest_first_and_bounded,
        test_sanitizer_restores_compact_batch_fields_from_scene_metadata,
        test_shot_batch_completion_budget_uses_runtime_topology,
        test_shot_batch_budget_and_prompt_compaction_create_more_two_scene_headroom,
        test_story_scene_coercion_keeps_quoted_dialogue_intact,
        test_scene_markers_preserve_one_to_one_scene_boundaries,
        # Runtime
        test_checkpoint_fingerprint_tracks_runtime_topology_and_frame_policy,
        test_camera_vocabulary_is_enforced_and_aliases_are_normalized,
        test_cinematic_compiler_fallback_camera_vocabulary_is_canonical,
        test_camera_directive_in_action_is_replaced_from_scene_evidence,
        test_short_visual_prompts_are_normalized_to_15_40_words_without_model_call,
        # Prompt-edge and recovery
        test_unique_story_paragraph_recovers_unmatched_source_dialogue,
        test_exact_duplicate_terminal_sentence_is_removed_without_retry,
        test_critic_filters_only_case_cross_scene_false_positives_and_noop_patches,
        test_director_critique_respects_manifest_default_and_explicit_opt_out,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"Director validation PASSED ({len(tests)} checks).")


if __name__ == "__main__":
    main()

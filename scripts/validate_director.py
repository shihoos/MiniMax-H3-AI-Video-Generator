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
    _assert(schema["properties"]["spoken_dialogue"].get("maxItems") == 128,
            "spoken-dialogue extraction should retain its independent bounded payload")


def test_character_semantic_schema_includes_spoken_dialogue():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    schema = QwenDirectorPromptMixin._character_extraction_json_schema()
    _assert("spoken_dialogue" in schema["properties"], "semantic character extraction schema lacks spoken_dialogue")
    _assert("spoken_dialogue" in schema["required"], "spoken_dialogue must be part of the single semantic extraction contract")
    _assert(schema["properties"]["spoken_dialogue"]["maxItems"] >= 64, "spoken dialogue schema is artificially capped")


def test_qwen_extraction_does_not_force_candidates_from_hints_plan_or_dialogue():
    import json
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    mixin = QwenDirectorPromptMixin()
    mixin._character_semantic_calls = 0
    mixin._semantic_spoken_dialogue = []
    mixin._anchor_semantic_dialogue_to_story = lambda _story, items: list(items)
    observed = {}
    story = 'Elara opened the vault. Kael Mercer woke and said, "I can walk." Elara helped him reach the exit.'
    def fake_chat_json(system_prompt, user_prompt, *args, **kwargs):
        observed["schema"] = kwargs["response_schema"]
        observed["system"] = system_prompt
        observed["payload"] = json.loads(user_prompt)
        return {
            "candidates": [{"name": "Elara", "entity_type": "PERSON", "is_character": True,
                            "identity_type": "named_character"}],
            "spoken_dialogue": [{"text": "I can walk.", "speaker": "Kael Mercer"}],
        }
    mixin._chat_json = fake_chat_json
    result = mixin.extract_character_entities(
        story, deterministic_candidates=["Elara", "Kael Mercer", "Station Echo-7"]
    )
    _assert(set(observed["payload"]) == {"story"},
            "deterministic hint names and planned cast must not be sent as roster directives")
    _assert(set(observed["schema"]["properties"]) == {"candidates", "spoken_dialogue"},
            "Qwen should decide the cast directly, not fill a per-name classification array")
    _assert("physically participate" in observed["system"] and "no cast-count quota" in observed["system"].lower(),
            "extractor must request all current-action participants without a cast quota")
    _assert([item["name"] for item in result["candidates"]] == ["Elara"],
            "the extractor must not programmatically add a missing character to Qwen's roster")
    _assert(mixin._semantic_spoken_dialogue == [{"text": "I can walk.", "speaker": "Kael Mercer"}],
            "dialogue extraction should remain available without silently rewriting the roster")


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
        [item["text"] for item in speaker_contamination] == ["the storm is the first warning"],
        f"remote 'speaker' context incorrectly removed later in-person dialogue: {speaker_contamination!r}",
    )


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
        {"text": "You’ve got ten minutes before the storm breaches the outer shell", "speaker": "Dr. Renn"},
        {"text": "No one leaves with that data", "speaker": "Dr. Renn"},
    ]
    aliases = EntityResolver.build_character_alias_map([{"name": "Renn", "identity_type": "named_character"}])
    segments = QwenDirector._extract_story_spoken_segments(
        story, semantic, allowed_speakers=["Renn"], speaker_aliases=aliases,
    )
    texts = [segment["display"] for segment in segments]
    speakers = [segment["source_speakers"] for segment in segments]
    _assert(len(segments) == 2, f"expected two visible Renn utterances and no radio dialogue: {segments}")
    _assert(any("you've got ten minutes before the storm breaches the outer shell" == text.casefold() for text in texts),
            f"in-person entrance dialogue was dropped: {texts}")
    _assert("No one leaves with that data" in texts, f"in-person final-scene dialogue was dropped: {texts}")
    _assert(all(value == {"renn"} for value in speakers), f"title-bearing speaker was not canonicalized: {speakers}")


def test_dialogue_reconciliation_keeps_only_visible_canonical_speaker_events():
    from planner.qwen_director import QwenDirector
    story = (
        'Lena’s radio crackled. “Voss, you still there?” It was Dr. Renn, his voice taut. '
        '“Don’t open that vault,” he said.\n\n'
        'At the main entrance, Renn appeared, his face lit by the emergency lights. '
        '“You’ve got ten minutes before the storm breaches the outer shell,” he said, gripping her arm.\n\n'
        'Renn yanked the ignition key from the dashboard. “No one leaves with that data,” he said, tossing the key into the wind.'
    )
    director = QwenDirector.__new__(QwenDirector)
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
    _assert(shots[0]["dialogue_events"] == [], f"remote radio speech must not become visible shot dialogue: {shots[0]}")
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
        _assert(f"exactly {DIRECTOR_STORY_SCENES} paragraphs" in text, f"{mode}: scene count missing from prompt")
        low = text.lower()
        _assert("names" in low and "relationships" in low, f"{mode}: Qwen must own cast, names and relationships")
        _assert(not re.search(r"(at least|exactly|minimum of)\s+(two|three|four|\d)\s+(named\s+)?characters", low),
                f"{mode}: prompt must not impose a cast quota")


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
        == int(writing["minimum_output_tokens"]) == 1350,
        "writer minimum-output floor must be owned by runtime_versions.yaml",
    )
    _assert(
        1 <= DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS
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
    _assert(write_payload["min_tokens"] == DIRECTOR_STORY_WRITER_MIN_OUTPUT_TOKENS,
            f"writer minimum-output floor must reach the vLLM payload: {write_payload}")


def test_story_blueprint_normalization_requires_ordered_scene_beats():
    from planner.qwen_director import QwenDirector
    from planner.config import DIRECTOR_STORY_SCENES
    director = QwenDirector.__new__(QwenDirector)
    source = "\n".join(f"SCENE {i:02d}: Beat {i} changes the story and carries the state forward." for i in range(1, DIRECTOR_STORY_SCENES + 1))
    normalized = director._normalize_story_blueprint(source, DIRECTOR_STORY_SCENES)
    lines = normalized.splitlines()
    _assert(len(lines) == DIRECTOR_STORY_SCENES, "blueprint must have one line per scene")
    _assert(lines[0].startswith("SCENE 01:") and lines[-1].startswith(f"SCENE {DIRECTOR_STORY_SCENES:02d}:"),
            "blueprint numbering must be normalized and ordered")
    try:
        director._normalize_story_blueprint("SCENE 01: only one beat", DIRECTOR_STORY_SCENES)
    except RuntimeError as exc:
        _assert("missing" in str(exc).lower(), f"unexpected invalid-blueprint error: {exc}")
    else:
        raise AssertionError("incomplete story blueprint must fail closed without a retry")
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
    raw = "\n".join(f"SCENE {i:02d}: Beat {i} changes the situation and causes the next event." for i in range(1, DIRECTOR_STORY_SCENES + 1))
    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return raw
    director._chat_text = fake_chat
    blueprint = director._generate_story_blueprint(AI_STORY_MODE, "A premise.")
    _assert(len(calls) == 1, "architect pass must be single-call with no retry")
    _assert(calls[0].get("disable_thinking") is False, "architect must use thinking mode")
    _assert(calls[0].get("thinking_token_budget") == DIRECTOR_STORY_PLAN_THINKING_BUDGET == 1280,
            f"architect must use its configured reasoning budget: {calls[0]}")
    _assert(calls[0].get("max_completion") == DIRECTOR_STORY_PLAN_MAX_TOKENS == 2200,
            f"architect must use its configured hard output ceiling: {calls[0]}")
    _assert(calls[0].get("temperature") == DIRECTOR_STORY_THINKING_TEMPERATURE == 0.6
            and calls[0].get("top_p") == DIRECTOR_STORY_THINKING_TOP_P == 0.95,
            f"architect must use the official Qwen3 thinking sampler: {calls[0]}")
    _assert(blueprint.count("SCENE ") == DIRECTOR_STORY_SCENES,
            "normalized blueprint should include all configured scenes")


def test_short_story_below_prompt_minimum_fails_without_retry():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES, STORY_MIN_WORDS
    paragraph = " ".join(["Ines"] + [f"word{i}" for i in range(59)]) + "."
    story = "\n\n".join(paragraph for _ in range(DIRECTOR_STORY_SCENES))
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
        _assert("need 816+" in str(exc), f"unexpected short-story rejection: {exc}")
    else:
        raise AssertionError("story below prompt's 816-word minimum should fail closed")
    _assert(len(calls) == 1, f"short story must not trigger retry, got {len(calls)} calls")


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
        _assert("Could not normalize the story to exactly" in str(exc),
                f"unexpected topology failure: {exc}")
    else:
        raise AssertionError("story without sufficient sentence boundaries must not be accepted")
    _assert(len(calls) == 1, f"topology failure must not trigger retry, got {len(calls)} calls")


def test_story_uses_one_bounded_plan_and_one_writer_without_retry_or_repair():
    """Guard the normal fresh-story path against extra expensive model calls."""
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    generate_start = source.index("story_blueprint = self._generate_story_blueprint(", source.index("def generate("))
    normal_path_end = source.index("# PASS 1B: deterministic production foundation", generate_start)
    normal_path = source[generate_start:normal_path_end]
    _assert("_maybe_plan_story_bible" not in source and "_format_story_bible" not in source,
            "the removed Story Bible must not return")
    _assert(normal_path.count("self._generate_story_blueprint(") == 1,
            "normal fresh generation must create one causal blueprint")
    _assert(normal_path.count("self._generate_story_once(") == 1,
            "normal fresh generation must make one prose-writing call")
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
        " ".join([f"Scene{i}", "Ines", "carried", "the", "lantern", "down", "the", "cellar", "stairs", "and", "counted", "each", "step."]
                 + ["She", "listened", "for", "the", "door", "above", "her", "head."] * 8)
        for i in range(DIRECTOR_STORY_SCENES + 3)
    ]
    story = "\n\n".join(paragraphs)
    director = QwenDirector.__new__(QwenDirector)
    director._story_min_words = STORY_MIN_WORDS
    calls = []
    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return story
    director._chat_text = fake_chat
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


def test_shot_batch_completion_budget_uses_runtime_topology():
    from planner.config import DIRECTOR_MAX_TOKENS, DIRECTOR_SHOTS_PER_SCENE
    from planner.qwen_director import QwenDirector
    def expected(scene_count):
        count = max(1, int(scene_count))
        return min(int(DIRECTOR_MAX_TOKENS), max(320, 700 * int(DIRECTOR_SHOTS_PER_SCENE) * count))
    _assert(QwenDirector._shot_batch_completion_budget(1) == expected(1), "one-scene shot batch budget ignored YAML topology")
    _assert(QwenDirector._shot_batch_completion_budget(2) == expected(2), "two-scene shot batch budget ignored YAML topology")
    _assert(QwenDirector._shot_batch_completion_budget(0) == expected(1), "non-positive scene count must clamp to one")


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
    _assert(all("you shouldn t be here" not in text for _, text in normalized),
            f"remote loudspeaker dialogue leaked into scene-local visible speech: {source_dialogue!r}")


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
        # Dialogue contract
        test_story_dialogue_contract_uses_qwen_semantic_spans,
        test_bare_radio_prop_does_not_make_following_live_quote_remote,
        test_dialogue_speaker_aliases_and_duplicate_lines_are_reconciled_structurally,
        test_honorific_speaker_normalization_keeps_in_person_dialogue_and_filters_remote_voice,
        test_dialogue_reconciliation_keeps_only_visible_canonical_speaker_events,
        # Prompt assembly
        test_shot_planning_allows_empty_scene_binding_with_closed_roster,
        test_shot_context_preserves_paragraph_boundaries,
        test_shot_context_does_not_split_honorific_from_name,
        test_story_context_compactor_keeps_causal_middle_action,
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
        test_story_blueprint_normalization_requires_ordered_scene_beats,
        test_story_architect_uses_native_bounded_thinking_and_no_retry,
        test_short_story_below_prompt_minimum_fails_without_retry,
        test_story_topology_fails_closed_when_sentence_boundaries_cannot_make_scenes,
        test_story_writer_is_one_qwen3_nonthinking_call_with_configured_ceiling,
        test_story_uses_one_bounded_plan_and_one_writer_without_retry_or_repair,
        test_topology_is_driven_by_scene_and_shot_settings,
        test_shot_batch_completion_budget_uses_runtime_topology,
        test_story_scene_coercion_keeps_quoted_dialogue_intact,
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

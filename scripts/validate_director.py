from pathlib import Path
import os
import re
import sys
import ast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _planner():
    from planner.production_planner import ProductionPlanner
    return ProductionPlanner(ROOT)


def test_qwen_is_the_only_creative_character_authority():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    planner_source = (ROOT / "planner/production_planner.py").read_text(encoding="utf-8")
    _assert("qwen_character_extractor=self.extract_character_entities" in source,
            "creative path must call the Qwen character extractor")
    _assert("Qwen is the sole semantic authority" in planner_source,
            "planner must document Qwen roster authority")


def test_single_character_semantic_call_budget():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert("if self._character_semantic_calls > 1:" in source,
            "character semantic budget must reject a second call")
    _assert("max 1" in source,
            "character semantic call limit must remain one")


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
    planner_source = (ROOT / "planner/production_planner.py").read_text(encoding="utf-8")
    _assert("TIME_WORDS" not in planner_source and "WEATHER_WORDS" not in planner_source and "MOOD_WORDS" not in planner_source,
            "production planner must not infer scene semantics from fixed keyword tables")
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

    descriptive = {
        "name": "the man in the grey coverall",
        "role": "story character",
        "identity_type": "descriptive_character",
        "semantic_aliases": [],
    }
    other = {
        "name": "Eli",
        "role": "story character",
        "identity_type": "named_character",
        "semantic_aliases": [],
    }
    characters = [
        SimpleNamespace(
            **descriptive,
            to_dict=lambda: dict(descriptive),
        ),
        SimpleNamespace(
            **other,
            to_dict=lambda: dict(other),
        ),
    ]

    result = planner._characters_in_scene(
        "The door opened and the lights failed.",
        characters,
    )
    _assert(
        "the man in the grey coverall" not in result,
        f"descriptive identity was bound by article: {result}",
    )


def test_sanitize_scene_alias_fallback_does_not_bind_descriptive_by_article():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    aliases = director._character_alias_map({"the man in the grey coverall"})
    _assert("the" not in aliases, "sanitizer alias map must not create article alias")
    _assert(aliases.get("the man in the grey coverall") == "the man in the grey coverall",
            f"descriptive canonical alias missing: {aliases}")


def test_qwen_can_choose_any_cast_size():
    from planner.production_planner import ProductionPlanner
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
    _assert(planner._drop_mention_only_identities("Kovac had vanished.", ["Kovac"], preserve_empty=False) == [], "mention-only filter must be able to return empty")


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
            # Reproduce the October 16 trace: Qwen approved the identity but mislabeled its type.
            {"name": "Eli's father", "entity_type": "CHARACTER", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "The woman in the lab coat", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
        ]}

    characters = planner.create_characters(story, qwen_character_extractor=extractor)
    names = [c.name for c in characters]
    _assert(names == ["Eli", "Eli's father", "The woman in the lab coat"], f"Qwen-approved relational/descriptive identities were lost: {names}")
    _assert(characters[1].identity_type == "relational_character", "relational identity type was lost")
    _assert(characters[2].identity_type == "descriptive_character", "descriptive identity type was lost")


def test_director_create_characters_keywords_match_planner_signature():
    import inspect
    from planner.production_planner import ProductionPlanner

    planner_params = set(inspect.signature(ProductionPlanner.create_characters).parameters)
    planner_params.discard("self")
    planner_params.discard("story")

    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "create_characters"
            and isinstance(func.value, ast.Name)
            and func.value.id == "planner"
        ):
            calls.append(node)

    _assert(calls, "qwen_director must call planner.create_characters")
    for call in calls:
        for keyword in call.keywords:
            _assert(
                keyword.arg in planner_params,
                f"qwen_director calls create_characters with unknown keyword {keyword.arg!r}; "
                f"planner accepts {sorted(planner_params)}",
            )


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
    _assert("hint_classifications" not in schema["properties"] and "hint_classifications" not in schema["required"],
            "the roster contract must not require per-name hint classifications")


def test_qwen_extraction_does_not_force_candidates_from_hints_plan_or_dialogue():
    import json
    from planner.qwen_director_prompts import QwenDirectorPromptMixin

    mixin = QwenDirectorPromptMixin()
    mixin._character_semantic_calls = 0
    mixin._semantic_spoken_dialogue = []
    # This unit isolates roster authorship; dialogue anchoring is tested separately.
    mixin._anchor_semantic_dialogue_to_story = lambda _story, items: list(items)
    observed = {}
    story = 'Elara opened the vault. Kael Mercer woke and said, "I can walk." Elara helped him reach the exit.'

    def fake_chat_json(system_prompt, user_prompt, *args, **kwargs):
        observed["schema"] = kwargs["response_schema"]
        observed["system"] = system_prompt
        observed["payload"] = json.loads(user_prompt)
        # Simulate Qwen returning an incomplete roster. The code must not inject
        # Kael from deterministic hints or the separate dialogue list.
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


def test_ai_story_does_not_inherit_template_cast_anchors():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    start = source.index("source_character_names: list[str] =", source.index("def generate("))
    end = source.index("if resuming and prior_director_plan.get(", start)
    assignment = source[start:end]
    _assert("if mode == EXPAND_USER_STORY_MODE" in assignment,
            "only source-preserving Expand Story may receive base-plan character anchors")








def test_story_dialogue_contract_uses_qwen_semantic_spans():
    from planner.qwen_director import QwenDirector
    semantic = [
        {"text": "We go now", "speaker": "Lena"},
        {"text": "Nobody goes up", "speaker": "Ines"},
    ]
    human = QwenDirector._extract_story_spoken_segments(
        '"We go now," Lena whispered. She opened the door.',
        semantic,
        allowed_speakers={"Lena"},
    )
    _assert(human and human[0]["source_speakers"] == {"lena"}, "semantic human dialogue was not reconciled")

    machine = QwenDirector._extract_story_spoken_segments(
        'The terminal displayed "Access granted."',
        semantic,
        allowed_speakers={"Lena", "Ines"},
    )
    _assert(not machine, "unblessed machine/system text must not become dialogue")

    unusual_verb = QwenDirector._extract_story_spoken_segments(
        '"Nobody goes up," Ines snapped.',
        semantic,
        allowed_speakers={"Ines"},
    )
    _assert(unusual_verb and unusual_verb[0]["source_speakers"] == {"ines"}, "semantic dialogue must not depend on a speech-verb list")

    written_quote = QwenDirector._extract_story_spoken_segments(
        'Nadia opened the journal. “Seal at five,” Nadia read. She closed the book.',
        [{"text": "Seal at five", "speaker": "Nadia"}],
        allowed_speakers={"Nadia"},
    )
    _assert(not written_quote, "a named character reading journal text must not become present-time dialogue")

    live_quote = QwenDirector._extract_story_spoken_segments(
        'Nadia entered the room and said, “Leave now.”',
        [{"text": "Leave now", "speaker": "Nadia"}],
        allowed_speakers={"Nadia"},
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

    written_then_live_quote = QwenDirector._extract_story_spoken_segments(
        'Mara read the log, then turned toward Eli and said, “Leave now.”',
        [{"text": "Leave now", "speaker": "Mara"}],
        allowed_speakers={"Mara"},
    )
    _assert(
        written_then_live_quote and written_then_live_quote[0]["display"] == "Leave now.",
        "a preceding document mention must not suppress subsequent present-time speech",
    )


def test_semantic_dialogue_is_persisted_and_restored_across_resume():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert('"_semantic_spoken_dialogue": deepcopy(self._semantic_spoken_dialogue)' in source,
            "director checkpoints must persist semantic spoken dialogue")
    _assert('restored_semantic_dialogue = prior_director_plan.get("_semantic_spoken_dialogue", [])' in source,
            "resume path must restore semantic spoken dialogue")
    _assert('"_semantic_spoken_dialogue" not in prior_director_plan' in source,
            "resume path must fail closed when the semantic dialogue field is absent")
    _assert('"_semantic_spoken_dialogue" in creative' in source,
            "enrich_plan must preserve the semantic dialogue payload")


def test_preserve_story_uses_the_single_semantic_pass_without_overriding_roster():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert('qwen_character_extractor=None' in source,
            "Preserve Story must keep its deterministic character roster")
    _assert('_preserve_story_requires_semantic_dialogue' in source,
            "Preserve Story must gate semantic Qwen use on quote-style prose dialogue")


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
        characters,
        scenes,
    )

    _assert("Sara Khan" in payload and "Eli Stone" in payload,
            "shot planning must still receive the closed canonical roster")
    _assert('"scene_id":"scene_004"' in payload,
            "empty-bound scene must still be represented in the shot payload")


def test_resume_never_repeats_semantic_qwen_call():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert("_semantic_spoken_dialogue was not persisted" in source, "legacy semantic-dialogue checkpoints must fail closed")
    resume_section = source[source.index("if resume_roster:"):source.index("else:\n            if mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):")]
    _assert("self.extract_character_entities(" not in resume_section, "resume path must not re-run semantic extraction")


def test_qwen_prompt_imports_entity_resolver_for_shot_dialogue():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert("from planner.entity_resolver import EntityResolver" in source,
            "shot dialogue prompt builder must import EntityResolver explicitly")


def test_dialogue_anchor_normalization_matches_terminal_punctuation_rules():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert(r"!?\.\-\s" in source,
            "dialogue anchor key must normalize terminal periods consistently")

def test_empty_character_extraction_result_keeps_dialogue_schema_contract():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert('return {"candidates": [], "spoken_dialogue": []}' in source,
            "empty semantic extraction must return only Qwen roster and dialogue fields")


def test_preserve_dialogue_detection_uses_normalized_story():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert("_preserve_story_requires_semantic_dialogue(story)" in source,
            "Preserve semantic-dialogue detection must use the normalized story")


def test_production_shot_error_message_tracks_instance_topology():
    source = (ROOT / "planner/qwen_director_sanitize.py").read_text(encoding="utf-8")
    _assert('f"Every scene must contain exactly {self.SHOTS_PER_SCENE} production shots."' in source,
            "production shot validation message must track instance topology")

def test_expand_semantic_dialogue_loss_is_telemetried():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert('"expand_semantic_dialogue_empty"' in source,
            "Expand semantic dialogue loss must be visible in telemetry")


def test_preserve_semantic_call_is_conditional():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert("_preserve_story_requires_semantic_dialogue" in source, "Preserve mode semantic extraction must be conditional")


def test_shot_dialogue_uses_roster_aliases():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert("speaker_aliases=EntityResolver.build_character_alias_map(characters)" in source, "shot dialogue extraction must use roster aliases")


def test_preserve_semantic_empty_dialogue_is_telemetried():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert("preserve_semantic_dialogue_empty" in source, "Preserve semantic dialogue misses must be visible in telemetry")


def test_shot_dialogue_extraction_degrades_deterministically():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert("source_dialogue = []" in source and "source dialogue extraction failed" in source, "shot dialogue extraction failure must not abort scene batching")




def test_semantic_dialogue_empty_is_telemetried():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert("expand_semantic_dialogue_empty" in source, "Expand semantic-dialogue misses must be visible in telemetry")
    _assert("preserve_semantic_dialogue_empty" in source, "Preserve semantic-dialogue misses must be visible in telemetry")
    _assert("ai_story_semantic_dialogue_empty" in source, "AI Story semantic-dialogue misses must be visible in telemetry")

def test_character_semantic_schema_includes_spoken_dialogue():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    schema = QwenDirectorPromptMixin._character_extraction_json_schema()
    _assert("spoken_dialogue" in schema["properties"], "semantic character extraction schema lacks spoken_dialogue")
    _assert("spoken_dialogue" in schema["required"], "spoken_dialogue must be part of the single semantic extraction contract")
    _assert(schema["properties"]["spoken_dialogue"]["maxItems"] >= 64, "spoken dialogue schema is artificially capped")



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
        story,
        ["the man in the grey coverall"],
        None,
        relational_hints=[],
        descriptive_hints=["the man in the grey coverall"],
    )
    from planner.entity_resolver import EntityResolver
    info = metadata.get(EntityResolver.normalize("the man in the grey coverall"))
    _assert(
        info and info.get("identity_type") == "descriptive_character",
        f"descriptive fallback metadata was not populated: {metadata}",
    )


def test_mention_only_filter_can_fail_closed():
    planner = _planner()
    _assert(planner._drop_mention_only_identities("Kovac had vanished.", ["Kovac"], preserve_empty=False) == [], "mention-only filter must not restore rejected roster")





def test_final_telemetry_is_not_printed_twice_when_critique_is_skipped():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert(source.count('self._print_qwen_summary("FINAL")') == 2,
            "only the enabled-critique finally and disabled-critique branch should print final telemetry")
    enrich = source[source.index("    def enrich_plan("):source.index("    def critique_plan(")]
    _assert('self._print_qwen_summary("FINAL")' not in enrich,
            "enrich_plan must not duplicate the summary owned by critique_plan")


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

        # The new explicit opt-out wins over the legacy feature variable.
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
    director_source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    orchestrator_source = (ROOT / "pipeline/production_orchestrator.py").read_text(encoding="utf-8")
    _assert("director_critic_enabled()" in director_source,
            "the critique entrypoint must use the shared feature resolver")
    _assert("director_critic_enabled()" in orchestrator_source,
            "the orchestrator must use the same feature resolver as the critic")








def test_story_prompt_has_no_artificial_subtargets():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8").lower()
    director_source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8").lower()
    _assert("about 500 words" not in source, "story prompt must not reintroduce an artificial 500-word target")
    _assert("six to eight sentences" not in source, "story prompt must not impose a per-paragraph sentence quota")
    _assert("at least 60 words" not in source, "story prompt must not impose an artificial final-paragraph word floor")
    _assert("470 to 520 words" not in director_source and "470-520 words" not in director_source,
            "story path must not impose a conflicting narrower word-count subtarget")
    _assert("about 70 to 95 words" not in director_source,
            "story path must not impose a conflicting per-paragraph word quota")
    _assert("at least 50 words" not in director_source,
            "story path must not impose a conflicting final-paragraph floor")
    _assert("_maybe_plan_story_bible" not in director_source.lower(),
            "Story Bible planner must be removed, not left as dormant code")
    _assert("story contract miss; one guided revision follows" not in director_source,
            "single-pass story profile must not automatically retry a failed soft contract")

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


def test_best_of_n_is_not_wired_into_normal_story_generation():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    start = source.index("story_user = self._story_text_user(", source.index("def generate("))
    end = source.index("# PASS 1B: deterministic production foundation", start)
    normal_path = source[start:end]
    _assert("_resample_for_audit(" not in normal_path, "normal path must not generate a challenger story")
    _assert("_pairwise_story_judgment(" not in normal_path, "normal path must not call pairwise story judge")
    _assert(normal_path.count("self._generate_story_once(") == 1, "story writer must be called once")
def test_story_writer_skips_story_plan_and_uses_one_pass_without_repair():
    """Normal generation skips the costly Story Bible and makes exactly one writer call."""
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    start = source.index("story_user = self._story_text_user(", source.index("def generate("))
    end = source.index("# PASS 1B: deterministic production foundation", start)
    normal_path = source[start:end]
    _assert("_maybe_plan_story_bible" not in source and "_format_story_bible" not in source,
            "Story Bible code must be removed, not merely bypassed")
    _assert(normal_path.count("self._generate_story_once(") == 1, "exactly one prose-writing call")
    _assert("_repair_story_if_needed(" not in normal_path and "story_repair_pass" not in normal_path,
            "default path must not retry or automatically repair the story")

def test_shot_critique_checks_object_state_across_story():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    # Normalize whitespace so this contract checks the wording, not incidental wrapping/indentation.
    normalized_source = " ".join(source.split()).lower()
    _assert("track story-established key objects across the full story" in normalized_source,
            "shot critique must explicitly track story-established object state across shots")
    _assert("after it was moved" in normalized_source,
            "shot critique must catch objects reappearing at old locations")
    _assert("capitalization alone" in normalized_source and "case-insensitively for identity" in normalized_source,
            "shot critique must not report a roster mismatch for capitalization alone")
    _assert("verbatim quote from the story" in normalized_source,
            "shot critique must ground findings in source evidence")
    _assert("remote speaker as missing" in normalized_source,
            "shot critique must not demand visible binding for remote-only dialogue")
    _assert("action distinctness" in normalized_source and "two paraphrases of the same observable action" in normalized_source,
            "shot critique must identify repeated actions, not only repeated camera framing")
    _assert('"color_temperature": str(shot.get("color_temperature"' in normalized_source,
            "critic context must include each shot's color temperature for continuity review")
    _assert('"location": str(shot.get("location"' in normalized_source,
            "critic context must include each shot's location for continuity review")




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
    aliases = EntityResolver.build_character_alias_map([
        {"name": "Renn", "identity_type": "named_character"},
    ])
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


def test_story_writer_is_one_thinking_call_and_never_rejects_on_style():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, DIRECTOR_STORY_SCENES
    paragraphs = [
        " ".join([f"Scene{i}", "Ines", "carried", "the", "lantern", "down", "the", "cellar", "stairs", "and", "counted", "each", "step."]
                 + ["She", "listened", "for", "the", "door", "above", "her", "head."] * 8)
        for i in range(DIRECTOR_STORY_SCENES + 3)   # wrong paragraph count on purpose
    ]
    story = "\n\n".join(paragraphs)
    director = QwenDirector.__new__(QwenDirector)
    calls = []
    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return story
    director._chat_text = fake_chat
    from contextlib import redirect_stdout
    from io import StringIO
    with redirect_stdout(StringIO()):
        result, generated = director._generate_story_once(
            AI_STORY_MODE, "premise", "system", "user", temperature=0.6, top_p=0.95, source_character_names=[],
        )
    count = len([p for p in re.split(r"\n\s*\n+", result) if p.strip()])
    _assert(generated and len(calls) == 1, f"story must be exactly one model call: {len(calls)}")
    _assert(calls[0].get("disable_thinking") is False, "story call must run with thinking on")
    _assert(calls[0].get("creative") is True, "story call must use the Qwen3 presence penalty for the AWQ model")
    _assert(calls[0].get("max_completion", 0) >= 6000, f"thinking needs a large completion ceiling: {calls[0]}")
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


def main():
    tests = [
        test_qwen_is_the_only_creative_character_authority,
        test_single_character_semantic_call_budget,
        test_shot_schema_tracks_instance_topology,
        test_mention_only_matching_is_case_insensitive,
        test_sanitizer_deduplicates_normalized_identity_keys,
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
        test_director_create_characters_keywords_match_planner_signature,
        test_expand_is_open_cast_but_preserves_source_anchors,
        test_character_schema_does_not_silently_cap_qwen_cast_at_32,
        test_qwen_extraction_does_not_force_candidates_from_hints_plan_or_dialogue,
        test_ai_story_does_not_inherit_template_cast_anchors,
        test_semantic_dialogue_is_persisted_and_restored_across_resume,
        test_preserve_story_uses_the_single_semantic_pass_without_overriding_roster,
        test_dialogue_speaker_aliases_and_duplicate_lines_are_reconciled_structurally,
        test_honorific_speaker_normalization_keeps_in_person_dialogue_and_filters_remote_voice,
        test_dialogue_reconciliation_keeps_only_visible_canonical_speaker_events,
        test_shot_planning_allows_empty_scene_binding_with_closed_roster,
        test_story_dialogue_contract_uses_qwen_semantic_spans,
        test_resume_never_repeats_semantic_qwen_call,
        test_preserve_semantic_call_is_conditional,
        test_qwen_prompt_imports_entity_resolver_for_shot_dialogue,
        test_dialogue_anchor_normalization_matches_terminal_punctuation_rules,
        test_empty_character_extraction_result_keeps_dialogue_schema_contract,
        test_preserve_dialogue_detection_uses_normalized_story,
        test_production_shot_error_message_tracks_instance_topology,
        test_expand_semantic_dialogue_loss_is_telemetried,
        test_shot_dialogue_uses_roster_aliases,
        test_preserve_semantic_empty_dialogue_is_telemetried,
        test_shot_dialogue_extraction_degrades_deterministically,
        test_semantic_dialogue_empty_is_telemetried,
        test_character_semantic_schema_includes_spoken_dialogue,
        test_shot_context_preserves_paragraph_boundaries,
        test_shot_context_does_not_split_honorific_from_name,
        test_honorific_and_active_short_name_do_not_drop_character,
        test_story_context_compactor_keeps_causal_middle_action,
        test_sound_and_ui_words_are_not_deterministic_characters,
        test_descriptive_sanitizer_preserves_qualified_identity,
        test_descriptive_grounding_tolerates_minor_surface_variation,
        test_descriptive_hints_populate_metadata_without_adding_cast,
        test_mention_only_filter_can_fail_closed,
        test_final_telemetry_is_not_printed_twice_when_critique_is_skipped,
        test_director_critique_respects_manifest_default_and_explicit_opt_out,
        test_story_prompt_has_no_artificial_subtargets,
        test_build_alias_map_skips_stopword_first_token,
        test_characters_in_scene_does_not_bind_descriptive_by_article,
        test_sanitize_scene_alias_fallback_does_not_bind_descriptive_by_article,
        test_identity_presence_accepts_title_free_and_given_name_surfaces,
        test_story_prompt_is_open_cast_and_fully_filled,
        test_story_writer_is_one_thinking_call_and_never_rejects_on_style,
        test_topology_is_driven_by_scene_and_shot_settings,
        test_best_of_n_is_not_wired_into_normal_story_generation,
        test_story_writer_skips_story_plan_and_uses_one_pass_without_repair,
        test_shot_critique_checks_object_state_across_story,
        test_short_visual_prompts_are_normalized_to_15_40_words_without_model_call,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"Director validation PASSED ({len(tests)} checks).")


if __name__ == "__main__":
    main()

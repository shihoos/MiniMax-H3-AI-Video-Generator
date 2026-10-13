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

    def extractor(_story, hints):
        calls.append(list(hints))
        return {"candidates": [
            {"name": "Elias Kade", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Lin Mei", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
        ]}

    planner.create_characters(story, qwen_character_extractor=extractor)
    _assert(len(calls) == 1, f"Qwen character extraction was bypassed or repeated: {calls}")


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
    story = "Eli entered the station. Eli's father waited beside the door and warned him."

    def extractor(_story, _hints):
        return {"candidates": [
            {"name": "Eli", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character"},
            {"name": "Eli's father", "entity_type": "CHARACTER", "is_character": True, "aliases": ["his father", "father"], "identity_type": "relational_character", "relationship_to": "Eli", "relationship": "father"},
        ]}

    characters = planner.create_characters(story, qwen_character_extractor=extractor)
    names = [c.name for c in characters]
    _assert(names == ["Eli", "Eli's father"], f"valid Qwen relational character was lost: {names}")
    _assert(characters[1].identity_type == "relational_character", "relational identity type was lost")


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


def test_expand_prompt_does_not_impose_cast_limit():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import EXPAND_USER_STORY_MODE, AI_STORY_MODE
    mixin = QwenDirectorPromptMixin()
    expand = mixin._story_text_system(EXPAND_USER_STORY_MODE).lower()
    ai = mixin._story_text_system(AI_STORY_MODE).lower()
    for text, label in ((expand, "expand"), (ai, "ai story")):
        _assert(
            ("no fixed cast size" in text
             or "no numeric cast limit" in text
             or "not a cast limit" in text
             or "do not force one character, two characters, or any fixed cast size" in text
             or ("you decide the cast" in text and "there is no target size" in text)),
            f"{label} prompt still constrains cast size",
        )
        _assert(
            "decorative character" in text
            or "decorative cast" in text
            or "do not add a character merely" in text,
            f"{label} prompt lacks anti-decorative-cast rule",
        )
    _assert("only named people allowed" not in expand, "Expand prompt still treats source anchors as a cast whitelist")
    _assert("exactly one" not in expand or "exactly one" in expand and "six paragraphs" in expand, "Expand prompt contains an unintended cast-count instruction")


def test_character_schema_does_not_silently_cap_qwen_cast_at_32():
    from planner.qwen_director import QwenDirector
    schema = QwenDirector._character_extraction_json_schema()
    candidates = schema["properties"]["candidates"]
    _assert("maxItems" not in candidates, "character extraction schema must not impose a cast-size cap")
    _assert(schema["properties"]["spoken_dialogue"].get("maxItems") == 128,
            "spoken-dialogue extraction should retain its independent bounded payload")


def test_story_generation_is_single_call_with_seed_offsets_and_no_floor():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    body = source.split("def _generate_story_once")[1].split("def _coerce_story_to_six_paragraphs")[0]
    _assert(body.count("self._chat_text(") == 1, "story generation must use one primary Qwen text call")
    _assert('seed=DIRECTOR_VLLM_SEED + int(getattr(self, "_story_seed_offset", 0) or 0)' in body,
            "candidate seeds must be distinct while the baseline offset remains zero")
    _assert("story_min_output_tokens = 0" in body, "story generation must not impose a minimum-token padding floor")
    _assert("minimum_output_tokens=story_min_output_tokens" in body, "story minimum must reach the chat call")


def test_story_prompt_has_causal_reversal_not_fixed_mystery_template():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE
    mixin = QwenDirectorPromptMixin()
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = mixin._story_text_system(mode).lower()
        _assert("paragraph 4 delivers a reversal caused by something established earlier" in text,
                f"{mode} must require a planted, causal reversal")
        _assert("every important event follow from a visible decision" in text,
                f"{mode} must require event-level cause and effect")
        _assert("plant a concrete detail" in text and "pay it off through action or consequence" in text,
                f"{mode} must establish and pay off a concrete detail")
        _assert("the turn must change what the protagonist does" in text,
                f"{mode} reversal must change protagonist action")
        _assert("paragraph 5 forces a consequential choice with a visible price" in text,
                f"{mode} must require a consequential choice with a price")
        _assert("end the central conflict" in text and "settled aftermath" in text,
                f"{mode} must require resolution rather than a future hook")
        _assert("think through cause and effect before drafting" in text,
                f"{mode} must prompt Qwen to plan causal relationships before drafting")
        _assert("no fixed cast size, target, or maximum" in text,
                f"{mode} must leave cast size to Qwen")
        _assert("always use a mystery" not in text and "always use containment" not in text,
                f"{mode} must not force a fixed genre template")

def test_story_contract_remains_six_paragraphs_420_560():
    prompt_source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    sanitize_source = (ROOT / "planner/qwen_director_sanitize.py").read_text(encoding="utf-8")
    combined = prompt_source + "\n" + sanitize_source
    _assert("420 to 560 words" in combined or "420-560" in combined, "story word contract missing")
    _assert("six paragraphs" in combined.lower(), "six-paragraph contract missing")


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
            "empty semantic extraction must return both schema fields")


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
    critique = " ".join((ROOT / "planner/qwen_director.py").read_text(encoding="utf-8").split()).lower()
    _assert("track story-established key objects across the full story" in critique and "after it was moved" in critique, "shot critique must check object continuity")

def test_honorific_and_active_short_name_do_not_drop_character():
    from planner.production_planner import ProductionPlanner as P
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
    _assert(P.identity_presence("Elara Voss entered the vault and shut the door.", "Dr. Elara Voss") == "present", "title-free full name must match")
    _assert(P.identity_presence(recorded_only, "Dr. Lena Voss") == "mention_only", "a recording alone must not imply physical presence")
    _assert(P.identity_presence("Mira opened the door.", "Dr. Kaelen Mira") == "absent", "surname overlap must not merge different people")

def test_story_context_compactor_keeps_causal_middle_action():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    story = "\n\n".join(
        f"Dr. Lena Voss reached chamber {i} after a long descent through a narrow access tunnel lined with old warning labels and corroded support ribs. "
        f"The station had been silent for decades, and the pressure readings moved unpredictably as the diver crossed the damaged threshold. "
        f"A pressure seal failed, so Lena grabbed the valve and stopped the leak in chamber {i}. "
        f"The diver left chamber {i} with the valve locked, marking the repair with a strip of orange emergency tape for the next crew to find."
        for i in range(1, 7)
    )
    compact = QwenDirectorPromptMixin._compact_story_context(story, 1800)
    _assert("Dr. Lena Voss" in compact, "shot context split the honorific from the name")
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


def test_story_prompt_preserves_causal_cast_and_live_dialogue_contract():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE

    mixin = QwenDirectorPromptMixin()
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = mixin._story_text_system(mode).lower()
        _assert("caused by something established earlier" in text, f"{mode} must require a grounded reversal")
        _assert("at least two concrete sentences" in text, f"{mode} must preserve protagonist interiority")
        _assert("short, purposeful line of live dialogue" in text, f"{mode} must require functional live dialogue")
        _assert("no fixed cast size, target, or maximum" in text, f"{mode} must leave cast size to Qwen")
        _assert("physically present counterpart" in text, f"{mode} should prefer a consequential counterpart when the premise supports one")
        _assert("exactly six paragraphs" in text and "420-560 words" in text, f"{mode} must retain the production format contract")


def test_story_bible_cast_schema_is_unbounded_and_plan_is_not_duplicated():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    schema = QwenDirectorPromptMixin._story_bible_json_schema()
    cast_schema = schema["properties"]["cast"]
    _assert("maxItems" not in cast_schema, "Story Bible cast schema must not impose a fixed maximum")
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    prompt_source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert('max_completion=2800' in prompt_source, "thinking-mode open-cast planning needs a bounded JSON/reasoning budget")
    _assert("protagonist_name" in schema["properties"] and "causal_mechanism" in schema["properties"],
            "Story Bible must state one protagonist and the causal rule")
    _assert("not a fixed cast quota" in prompt_source.lower(), "the outline must not force an unnecessary cast")
    _assert("not automatically sent to editor" in source, "outline discrepancies must stay advisory")
    _assert('os.getenv("H3_STORY_BIBLE", "0")' in source, "Story Bible stays opt-in until controlled runs justify it")

def test_editor_uses_same_story_contract_and_can_repair_supported_counterpart():
    from planner.qwen_director import QwenDirector
    editor = QwenDirector._STORY_EDITOR_SYSTEM.lower()
    _assert("420-560 words" in editor, "editor must match the primary output contract")
    _assert("story bible is provisional" in editor, "editor must treat the outline as advisory")
    _assert("do not force every outline beat or planned character" in editor, "editor must be free to drop nonessential planned cast")
    _assert("consequential present-day counterpart" in editor, "editor should repair a missing consequential counterpart when premise supports one")
    _assert('os.getenv("h3_story_editor", "auto")' in (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8").lower(), "editor should default to audit-gated (auto) mode")


def test_thinking_mode_is_enabled_for_story_planner_writer_and_editor():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    prompts = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    runtime = (ROOT / "planner/qwen_director_runtime.py").read_text(encoding="utf-8")
    story_start = source.index("def _generate_story_once")
    story_end = source.index("def _story_craft_issues", story_start)
    story_code = source[story_start:story_end]
    bible_start = prompts.index("def _plan_story_bible(")
    bible_end = prompts.index("def _format_story_bible(", bible_start)
    bible_code = prompts[bible_start:bible_end]
    _assert("disable_thinking=False" in story_code, "main story call must enable Qwen3 thinking")
    _assert('"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}' in runtime, "runtime must propagate thinking mode to vLLM")
    _assert("disable_thinking=False" in source[source.index("def _edit_story_draft"):source.index("def _enforce_scene_coherence")], "editor call must also use thinking mode")
    _assert("disable_thinking=False" in bible_code, "Story Bible planner must also use Qwen3 thinking mode")
    _assert("temperature=0.60" in bible_code and "top_p=0.95" in bible_code, "thinking planner sampling should use Qwen3 recommended values")


def test_story_quality_review_is_evidence_bound_and_precedes_editor():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE
    from planner.qwen_director_prompts import QwenDirectorPromptMixin

    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    generation_start = source.index("story_review_findings = self._collect_story_findings(")
    editor_call = source.index("story = self._edit_story_draft(", generation_start)
    _assert(generation_start < editor_call, "story review must feed the editor before final story validation")
    review_start = source.index("def _review_story_draft(")
    review_end = source.index("_STORY_EDITOR_SYSTEM", review_start)
    review_code = source[review_start:review_end]
    _assert('call_name="story_quality_review"' in review_code, "review must be separately observable in Qwen telemetry")
    _assert("disable_thinking=False" in review_code, "story review must use Qwen3 thinking mode")
    _assert("evidence_norm not in normalized_story" in review_code, "unsupported critic quotes must not trigger edits")
    _assert("review_findings=story_review_findings" in source, "the editor must receive evidence-backed findings")

    schema = QwenDirector._story_review_json_schema()
    findings = schema["properties"]["findings"]
    _assert(findings.get("maxItems") == 5, "review payload must remain bounded")
    allowed = {item for item in schema["properties"]["findings"]["items"]["properties"]["code"]["enum"]}
    _assert("planned_cast_presence" in allowed and "unresolved_conflict" in allowed,
            "review must inspect on-screen cast and ending resolution")

    director = QwenDirector.__new__(QwenDirector)
    director._planned_cast = ["Elias Kwan"]
    story = "Lena watched the sealed door. Dr. Elias Kwan only speaks through the intercom. The door never opens."
    def fake_chat_json(*args, **kwargs):
        return {"findings": [
            {"code": "planned_cast_presence", "severity": "major",
             "evidence": "Dr. Elias Kwan only speaks through the intercom.",
             "problem": "Elias Kwan is remote-only although his planned action is essential.", "repair": "Bring Elias Kwan into the current scene if his causal role is essential."},
            {"code": "unresolved_conflict", "severity": "major",
             "evidence": "The final chamber is permanently calm and the conflict is closed.",
             "problem": "Fabricated evidence must be rejected.", "repair": "Do not use this unsupported note."},
        ]}
    director._chat_json = fake_chat_json
    previous = os.environ.get("H3_STORY_CRITIC")
    os.environ["H3_STORY_CRITIC"] = "1"
    try:
        findings_out = director._review_story_draft(AI_STORY_MODE, "A scientist reaches a sealed station.", story, None, [])
    finally:
        if previous is None:
            os.environ.pop("H3_STORY_CRITIC", None)
        else:
            os.environ["H3_STORY_CRITIC"] = previous
    _assert(len(findings_out) == 1, f"only exact evidence-backed findings should survive, got {findings_out}")
    _assert(findings_out[0]["code"] == "planned_cast_presence", "wrong evidence-backed finding survived")
    import inspect
    _assert(inspect.signature(QwenDirector._validate_story_output_contracts).parameters["enforce_planned_cast"].default is False, "planned cast must remain advisory")
    _assert('call_name="story_quality_repair_pass"' in source and "exactly one targeted repair pass" in source, "verification repair must be bounded")
    mixin = QwenDirectorPromptMixin()
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        _assert(len(mixin._story_text_system(mode)) < 3500, f"{mode} story prompt exceeded compact budget")

def test_audit_gated_editor_and_quality_critic_default():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE

    director = QwenDirector.__new__(QwenDirector)
    director._fallback_planner = None
    director.project_root = ROOT
    director._planned_cast = ["Elias Kwan"]
    calls = {"chat": 0, "json": 0}
    director._chat_text = lambda *a, **k: calls.__setitem__("chat", calls["chat"] + 1) or ""
    director._chat_json = lambda *a, **k: calls.__setitem__("json", calls["json"] + 1) or {"findings": []}
    saved = {k: os.environ.pop(k, None) for k in ("H3_STORY_EDITOR", "H3_STORY_CRITIC")}
    try:
        story = "Lena opened the hatch."
        # The editor remains auto-gated: it must not rewrite a draft with no finding.
        _assert(director._edit_story_draft(AI_STORY_MODE, "p", story, None, [], temperature=0.6, top_p=0.95) == story,
                "auto editor must not touch a draft with no findings")
        _assert(director._review_story_draft(AI_STORY_MODE, "p", story, None, []) == [],
                "an empty quality review should return no findings")
        _assert(calls == {"chat": 0, "json": 1}, f"quality-first default must run one bounded review but no edit: {calls}")
        # Explicit latency mode disables the semantic critic.
        os.environ["H3_STORY_CRITIC"] = "0"
        _assert(director._review_story_draft(AI_STORY_MODE, "p", story, None, []) == [],
                "H3_STORY_CRITIC=0 must disable the review")
        _assert(calls == {"chat": 0, "json": 1}, f"critic-off mode made an unexpected call: {calls}")
        # Missing planned cast is diagnostic/advisory, not a blocking contract failure.
        found = director._planned_cast_findings(AI_STORY_MODE, story)
        _assert(found and found[0]["code"] == "planned_cast_presence" and found[0]["severity"] == "advisory",
                f"outline drift must remain advisory: {found}")
        director._planned_cast = []
        _assert(director._planned_cast_findings(AI_STORY_MODE, story) == [], "no plan means no cast findings")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_writer_path_does_not_abort_on_planned_cast_heuristic():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    start = source.index("def _generate_story_once")
    end = source.index("def _story_craft_issues", start)
    _assert(source[start:end].count("enforce_planned_cast=False") >= 3,
            "writer-path validation must leave planned-cast repair to the targeted editor")


def test_story_prompt_has_no_artificial_subtargets():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8").lower()
    director_source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8").lower()
    _assert("about 500 words" not in source, "story prompt must not reintroduce an artificial 500-word target")
    _assert("six to eight sentences" not in source, "story prompt must not impose a per-paragraph sentence quota")
    _assert("at least 60 words" not in source, "story prompt must not impose an artificial final-paragraph word floor")
    _assert("470 to 520 words" not in director_source and "470-520 words" not in director_source,
            "guided story revision must not impose a conflicting narrower word-count subtarget")
    _assert("about 70 to 95 words" not in director_source,
            "guided revision must not impose a conflicting per-paragraph word quota")
    _assert("at least 50 words" not in director_source,
            "guided revision must not impose a conflicting final-paragraph floor")
    _assert("420-560 words total" in director_source,
            "guided revision must restate the authoritative story-length contract")

def test_best_of_n_requires_grounded_pairwise_and_position_swap():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE
    director = QwenDirector.__new__(QwenDirector)
    director._fallback_planner = None
    director.project_root = ROOT
    director._planned_cast = ["Unneeded Outline Character"]
    incumbent = "Mara pocketed the brass key and shut the vault. Later, she found the key in the snow without returning outside."
    challenger = "Mara pocketed the brass key and shut the vault. The sealed door never opened again; the key stayed inside her coat."
    calls = {"n": 0, "seeds": [], "judge": 0}
    def fake_generate(*args, **kwargs):
        calls["n"] += 1
        calls["seeds"].append(director._story_seed_offset)
        return challenger, True
    director._generate_story_once = fake_generate
    director._story_style_findings = lambda story: []
    director._collect_story_findings = lambda *args, **kwargs: []
    def fake_judge(mode, user_input, story_a, story_b, findings_a, findings_b, story_bible, *, call_name):
        calls["judge"] += 1
        return {"winner": "B" if calls["judge"] == 1 else "A", "confidence": "high",
                "reason": "Challenger resolves the object-state contradiction.",
                "evidence_a": "Later, she found the key in the snow without returning outside." if calls["judge"] == 1 else "The sealed door never opened again; the key stayed inside her coat.",
                "evidence_b": "The sealed door never opened again; the key stayed inside her coat." if calls["judge"] == 1 else "Later, she found the key in the snow without returning outside."}
    director._pairwise_story_judgment = fake_judge
    saved = os.environ.get("H3_STORY_CANDIDATES")
    os.environ["H3_STORY_CANDIDATES"] = "2"
    try:
        director._story_seed_offset = 0
        chosen, findings = director._resample_for_audit(
            AI_STORY_MODE, "premise", incumbent, "system", "user",
            temperature=0.6, top_p=0.95, source_character_names=[],
            initial_findings=[{"code": "object_or_location_continuity", "severity": "major", "evidence": "Later, she found the key in the snow without returning outside."}],
        )
        _assert(chosen == challenger, "challenger should win only after both candidate orderings agree")
        _assert(calls["n"] == 1 and calls["seeds"] == [1], f"candidate generation must use new seed: {calls}")
        _assert(calls["judge"] == 2, f"a proposed challenger win must be confirmed after swapping positions: {calls}")
        _assert(director._story_seed_offset == 0, "seed offset must be reset after selection")
    finally:
        if saved is None:
            os.environ.pop("H3_STORY_CANDIDATES", None)
        else:
            os.environ["H3_STORY_CANDIDATES"] = saved
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert("position-swap confirmation failed" in source and "_quote_is_grounded" in source, "candidate choice must be grounded and position-swap confirmed")
    _assert("source_fidelity" in source and "do not force" in source.lower(), "candidate choice must preserve source facts without cast quotas")

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
        test_expand_prompt_does_not_impose_cast_limit,
        test_character_schema_does_not_silently_cap_qwen_cast_at_32,
        test_story_generation_is_single_call_with_seed_offsets_and_no_floor,
        test_story_prompt_has_causal_reversal_not_fixed_mystery_template,
        test_story_contract_remains_six_paragraphs_420_560,
        test_semantic_dialogue_is_persisted_and_restored_across_resume,
        test_preserve_story_uses_the_single_semantic_pass_without_overriding_roster,
        test_dialogue_speaker_aliases_and_duplicate_lines_are_reconciled_structurally,
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
        test_honorific_and_active_short_name_do_not_drop_character,
        test_story_context_compactor_keeps_causal_middle_action,
        test_sound_and_ui_words_are_not_deterministic_characters,
        test_descriptive_sanitizer_preserves_qualified_identity,
        test_descriptive_grounding_tolerates_minor_surface_variation,
        test_descriptive_hints_populate_metadata_without_adding_cast,
        test_mention_only_filter_can_fail_closed,
        test_story_prompt_preserves_causal_cast_and_live_dialogue_contract,
        test_story_bible_cast_schema_is_unbounded_and_plan_is_not_duplicated,
        test_editor_uses_same_story_contract_and_can_repair_supported_counterpart,
        test_thinking_mode_is_enabled_for_story_planner_writer_and_editor,
        test_story_quality_review_is_evidence_bound_and_precedes_editor,
        test_audit_gated_editor_and_quality_critic_default,
        test_writer_path_does_not_abort_on_planned_cast_heuristic,
        test_story_prompt_has_no_artificial_subtargets,
        test_build_alias_map_skips_stopword_first_token,
        test_characters_in_scene_does_not_bind_descriptive_by_article,
        test_sanitize_scene_alias_fallback_does_not_bind_descriptive_by_article,
        test_best_of_n_requires_grounded_pairwise_and_position_swap,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"Director validation PASSED ({len(tests)} checks).")


if __name__ == "__main__":
    main()

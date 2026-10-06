from pathlib import Path
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
    _assert("named_only_deterministic" not in source, "creative path must not bypass Qwen extraction")
    _assert("_high_confidence_deterministic_character" not in source, "Director must not protect a deterministic character over Qwen")
    _assert("qwen_character_extractor=self.extract_character_entities" in source, "creative path must call Qwen character extraction")
    _assert("qwen_character_adjudicator" not in source, "creative Director path must not expose a dead adjudicator call")
    _assert("Qwen is the sole semantic authority" in planner_source, "planner must document Qwen roster authority")


def test_single_character_semantic_call_budget():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert("if self._character_semantic_calls > 1:" in source, "character semantic budget must be one call")
    _assert("max 1" in source, "character semantic call limit must be one")


def test_reconcile_api_has_no_dead_deterministic_parameter():
    import inspect
    from planner.production_planner import ProductionPlanner
    parameters = list(inspect.signature(ProductionPlanner._reconcile_semantic_characters).parameters)
    _assert(parameters == ["story", "semantic_result"], f"unexpected reconciliation API parameters: {parameters}")


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
        _assert("no fixed cast size" in text or "no numeric cast limit" in text, f"{label} prompt still constrains cast size")
        _assert("decorative character" in text or "decorative cast" in text, f"{label} prompt lacks anti-decorative-cast rule")
    _assert("only named people allowed" not in expand, "Expand prompt still treats source anchors as a cast whitelist")
    _assert("exactly one" not in expand or "exactly one" in expand and "six paragraphs" in expand, "Expand prompt contains an unintended cast-count instruction")


def test_character_schema_does_not_silently_cap_qwen_cast_at_32():
    from planner.qwen_director import QwenDirector
    schema = QwenDirector._character_extraction_json_schema()
    _assert(schema["properties"]["candidates"]["maxItems"] >= 64, "character schema still silently caps Qwen cast at 32")


def test_story_generation_is_single_call_fixed_seed_no_floor():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    body = source.split("def _generate_story_once")[1].split("def _coerce_story_to_six_paragraphs")[0]
    _assert("H3_STORY_MAX_ATTEMPTS" not in source, "story retry setting must not exist")
    _assert("for attempt" not in body and "_retry" not in body, "story generation must not retry")
    _assert("seed=DIRECTOR_VLLM_SEED" in body, "story seed must remain fixed")
    _assert("story_min_output_tokens = 0" in body, "story generation must not impose the old 1450 floor")
    _assert("minimum_output_tokens=story_min_output_tokens" in body, "story minimum must reach the chat call")


def test_story_prompt_has_causal_reversal_not_fixed_mystery_template():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE
    mixin = QwenDirectorPromptMixin()
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = mixin._story_text_system(mode).lower()
        _assert("goal" in text and "resistance" in text and "reversal" in text and "choice" in text and "consequence" in text, "causal spine incomplete")
        _assert("cause, choose, or misjudge" in text, "reversal is not explicitly action-driven")
        _assert("sentient" in text, "generic sentient-system reveal guard missing")
        _assert("completed past-tense action" in text, "settled aftermath requirement missing")


def test_story_contract_remains_six_paragraphs_420_560():
    prompt_source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    sanitize_source = (ROOT / "planner/qwen_director_sanitize.py").read_text(encoding="utf-8")
    combined = prompt_source + "\n" + sanitize_source
    _assert("420 to 560 words" in combined or "420-560" in combined, "story word contract missing")
    _assert("six paragraphs" in combined.lower(), "six-paragraph contract missing")


def test_non_human_speech_tags_are_not_speakers():
    from planner.qwen_director import QwenDirector
    for text, expected in (
        ('"Nobody goes up," Ines said. She held the chain.', {"ines"}),
        ('"Access granted," the terminal said.', set()),
        ('Static answered.', set()),
    ):
        segments = QwenDirector._extract_story_spoken_segments(text)
        got = set().union(*[s.get("tag_speakers", set()) for s in segments]) if segments else set()
        _assert(got == expected, f"speech-tag attribution wrong for {text!r}: {got}")


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


def test_no_deterministic_cast_floor_symbols_in_qwen_path():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert("deterministic_hard_named" not in source, "Director must not maintain a deterministic cast floor")
    _assert("deterministic_descriptive" not in source, "Director must not maintain a deterministic descriptive cast floor")


def main():
    tests = [
        test_qwen_is_the_only_creative_character_authority,
        test_single_character_semantic_call_budget,
        test_reconcile_api_has_no_dead_deterministic_parameter,
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
        test_story_generation_is_single_call_fixed_seed_no_floor,
        test_story_prompt_has_causal_reversal_not_fixed_mystery_template,
        test_story_contract_remains_six_paragraphs_420_560,
        test_non_human_speech_tags_are_not_speakers,
        test_sound_and_ui_words_are_not_deterministic_characters,
        test_descriptive_sanitizer_preserves_qualified_identity,
        test_descriptive_grounding_tolerates_minor_surface_variation,
        test_descriptive_hints_populate_metadata_without_adding_cast,
        test_mention_only_filter_can_fail_closed,
        test_no_deterministic_cast_floor_symbols_in_qwen_path,
        test_build_alias_map_skips_stopword_first_token,
        test_characters_in_scene_does_not_bind_descriptive_by_article,
        test_sanitize_scene_alias_fallback_does_not_bind_descriptive_by_article,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"Director validation PASSED ({len(tests)} checks).")


if __name__ == "__main__":
    main()

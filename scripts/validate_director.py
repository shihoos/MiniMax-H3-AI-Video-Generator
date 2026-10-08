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
    _assert(
        "qwen_character_extractor=self.extract_character_entities" in source,
        "creative character planning must use the Qwen semantic extractor",
    )

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
            "no fixed cast size" in text
            or "there is no fixed cast size" in text
            or "there is no target size" in text
            or "the cast size is yours" in text,
            f"{label} prompt does not preserve Qwen-selected cast size",
        )
        _assert(
            "decorative character" in text
            or "decorative cast" in text
            or "do not add a character merely" in text
            or "every recurring character materially affects" in text,
            f"{label} prompt lacks the current causal-cast rule",
        )

def test_character_schema_does_not_silently_cap_qwen_cast_at_32():
    from planner.qwen_director import QwenDirector
    schema = QwenDirector._character_extraction_json_schema()
    _assert(schema["properties"]["candidates"]["maxItems"] >= 64, "character schema still silently caps Qwen cast at 32")


def test_story_generation_uses_current_output_contract():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    body = source.split("def _generate_story_once")[1].split("def _coerce_story_to_six_paragraphs")[0]
    _assert("seed=DIRECTOR_VLLM_SEED" in body, "story generation must use the configured deterministic seed")
    _assert("story_min_output_tokens = 0" in body, "story generation must not impose a minimum-output floor")
    _assert(
        "minimum_output_tokens=story_min_output_tokens" in body,
        "story minimum-output setting must reach the model call",
    )

def test_story_prompt_has_causal_reversal_structure():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE

    mixin = QwenDirectorPromptMixin()
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = mixin._story_text_system(mode).lower()
        for line in (
            "want:",
            "resistance:",
            "cast:",
            "setup:",
            "turn (reversal):",
            "choice:",
            "result:",
        ):
            _assert(line in text, f"{mode} story plan is missing the {line!r} line")
        _assert("in your reasoning" in text, f"{mode} prompt does not place the plan in reasoning")
        _assert(
            "causes, chooses, or misjudges" in text
            or "act, choose, refuse, misjudge" in text
            or "cause, choose, or misjudge" in text,
            f"{mode} reversal must remain action-driven",
        )
        _assert("completed past-tense action" in text, f"{mode} settled-aftermath requirement missing")
        _assert("hold the turn until" in text, f"{mode} prompt lacks the late-turn rule")
        _assert(
            "first consequential attempt" in text and "meets resistance" in text,
            f"{mode} prompt lacks the setback rhythm",
        )
        _assert(
            "forces the protagonist to decide or act" in text,
            f"{mode} prompt must make the turn force a decision",
        )

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


def test_scene_cast_guard_checks_scene_payload_not_global_roster():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert('if not scene_payload.get("characters"):' in source,
            "shot planning must reject a scene with no bound characters")


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


def test_story_prompt_prefers_causal_human_conflict_without_forcing_cast_size():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE
    mixin = QwenDirectorPromptMixin()
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = mixin._story_text_system(mode).lower()
        _assert(
            "include another person whenever their goal, action, or knowledge would change what the protagonist does" in text
            or "another character may create pressure" in text
            or "present counterpart" in text
            or "another present person" in text
            or ("relationship" in text and "causal story" in text),
            f"{mode} prompt lacks consequential-interpersonal guidance",
        )
        _assert(
            "no fixed cast size" in text
            or "there is no fixed cast size" in text
            or "there is no target size" in text
            or "the cast size is yours" in text,
            f"{mode} prompt accidentally constrains cast size",
        )
        _assert(
            "solitary story" in text
            or "one person" in text
            or "isolation" in text,
            f"{mode} prompt lacks valid solitary-story guidance",
        )
        _assert(
            "one person, two people, or any larger number" in text
            or "one person, two, or many" in text
            or "one, two, or several" in text
            or "cast may contain one person, two people" in text
            or "the cast size is yours: one, two, or several" in text
            or ("you decide the cast" in text and "there is no target size" in text),
            f"{mode} prompt does not preserve Qwen-selected cast size",
        )
        _assert(
            "dialogue is optional" in text
            or "do not add someone merely to create dialogue" in text
            or "never add a speaker merely to satisfy a dialogue requirement" in text
            or "do not add a character merely" in text,
            f"{mode} prompt still risks forcing dialogue/cast",
        )


def _unbound_loads(relative_path, watched):
    """Names in `watched` that a function reads without a valid lexical binding."""
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    problems = []

    def bindings(node):
        names = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs:
                names.add(arg.arg)
            for extra in (args.vararg, args.kwarg):
                if extra is not None:
                    names.add(extra.arg)
        stack = list(ast.iter_child_nodes(node))
        while stack:
            child = stack.pop()
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(child.name)
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                names.add(child.id)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            stack.extend(ast.iter_child_nodes(child))
        return names

    def visit(node, inherited):
        scope = inherited | bindings(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, scope)
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in watched and child.id not in scope:
                    problems.append((relative_path, child.lineno, child.id))
            visit(child, scope)

    visit(tree, set())
    return sorted(set(problems))



def test_known_wiring_names_are_bound():
    """Guard cross-module NameError regressions outside the Director validator itself."""
    for path, names in (
        ("planner/qwen_director_runtime.py", {"deepcopy"}),
        ("execution/production_runner.py", {"production_plan"}),
        ("ui/storyboard_gradio.py", {"gr"}),
    ):
        problems = _unbound_loads(path, names)
        _assert(not problems, f"unbound names would raise NameError at runtime: {problems}")
    runner = (ROOT / "execution/production_runner.py").read_text(encoding="utf-8")
    _assert(
        "production_plan=self._active_plan," in runner,
        "auto-retake must pass the active production plan",
    )

def main():
    tests = [
        test_qwen_is_the_only_creative_character_authority,
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
        test_story_generation_uses_current_output_contract,
        test_story_prompt_has_causal_reversal_structure,
        test_story_contract_remains_six_paragraphs_420_560,
        test_semantic_dialogue_is_persisted_and_restored_across_resume,
        test_preserve_story_uses_the_single_semantic_pass_without_overriding_roster,
        test_dialogue_speaker_aliases_and_duplicate_lines_are_reconciled_structurally,
        test_scene_cast_guard_checks_scene_payload_not_global_roster,
        test_story_dialogue_contract_uses_qwen_semantic_spans,
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
        test_character_semantic_schema_includes_spoken_dialogue,
        test_shot_context_preserves_paragraph_boundaries,
        test_sound_and_ui_words_are_not_deterministic_characters,
        test_descriptive_sanitizer_preserves_qualified_identity,
        test_descriptive_grounding_tolerates_minor_surface_variation,
        test_descriptive_hints_populate_metadata_without_adding_cast,
        test_mention_only_filter_can_fail_closed,
        test_story_prompt_prefers_causal_human_conflict_without_forcing_cast_size,
        test_build_alias_map_skips_stopword_first_token,
        test_characters_in_scene_does_not_bind_descriptive_by_article,
        test_sanitize_scene_alias_fallback_does_not_bind_descriptive_by_article,
        test_known_wiring_names_are_bound,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"Director validation PASSED ({len(tests)} checks).")


if __name__ == "__main__":
    main()

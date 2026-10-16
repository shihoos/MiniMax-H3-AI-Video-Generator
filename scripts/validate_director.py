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


def test_story_prompt_does_not_force_cast_size():
    """Qwen must choose the cast; story quality must not depend on a numeric quota."""
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import EXPAND_USER_STORY_MODE, AI_STORY_MODE

    mixin = QwenDirectorPromptMixin()
    for mode in (EXPAND_USER_STORY_MODE, AI_STORY_MODE):
        text = " ".join(mixin._story_text_system(mode).lower().split())
        _assert(
            "qwen decides names, identities, relationships, and final cast size" in text
            and "no hardcoded roster or numeric cast quota" in text,
            f"{mode} must leave cast authorship to Qwen",
        )
        _assert("include a second physically present" not in text,
                f"{mode} must not force a second character")
        _assert("at least two characters" not in text and "two-character minimum" not in text,
                f"{mode} must not enforce a numeric cast quota")
    _assert("only named people allowed" not in " ".join(mixin._story_text_system(EXPAND_USER_STORY_MODE).lower().split()),
            "Expand prompt must not treat source anchors as a cast whitelist")


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
    mixin._planned_cast = ["Planned Person"]
    # This unit isolates roster authorship; dialogue anchoring is tested separately.
    mixin._anchor_semantic_dialogue_to_story = lambda _story, items: list(items)
    observed = {}
    story = 'Elara opened the vault. Kael Mercer woke and said, "I can walk." Elara helped him reach the exit.'

    def fake_chat_json(system_prompt, user_prompt, *args, **kwargs):
        observed["schema"] = kwargs["response_schema"]
        observed["system"] = system_prompt
        observed["payload"] = json.loads(user_prompt)
        # Simulate Qwen returning an incomplete roster. The code must not inject
        # Kael from hint scans, its planned cast, or the separate dialogue list.
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


def test_story_generation_is_single_pass_thinking_3200_no_floor():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    body = source.split("def _generate_story_once")[1].split("def _story_craft_issues")[0]
    generation_start = source.index("story_user = self._story_text_user(", source.index("def generate("))
    generation_end = source.index("# PASS 1B: deterministic production foundation", generation_start)
    generation = source[generation_start:generation_end]
    _assert(body.count("self._chat_text(") == 1, "story generation must use one primary Qwen text call")
    _assert("story_budget = DIRECTOR_STORY_MAX_TOKENS" in body and "max_completion=story_budget" in body,
            "story writer must use the runtime-configured 3,200-token completion cap")
    _assert("disable_thinking=False" in body, "story writer must use thinking mode")
    _assert("seed=DIRECTOR_VLLM_SEED," in body, "single-pass story generation must use the stable Director seed")
    _assert("minimum_output_tokens=story_min_output_tokens" in body and "story_min_output_tokens = 0" in body,
            "story writer must not force minimum-token padding")
    _assert(generation.count("self._generate_story_once(") == 1,
            "normal story path must invoke the writer exactly once")
    for forbidden in ("_collect_story_findings(", "_resample_for_audit(", "_edit_story_draft(", "story_quality_repair_pass"):
        _assert(forbidden not in generation, f"normal story path must not call expensive optional stage {forbidden}")
    _assert("temperature, top_p = self._sampling_for_mode(mode)" in source,
            "story writer must use its configured story sampler")
def test_qwen3_story_sampler_is_runtime_config_driven():
    import yaml
    from planner.config import (
        DIRECTOR_STORY_MAX_TOKENS,
        DIRECTOR_STORY_THINKING_TEMPERATURE,
        DIRECTOR_STORY_THINKING_TOP_P,
        DIRECTOR_STORY_THINKING_TOP_K,
        DIRECTOR_STORY_THINKING_MIN_P,
        DIRECTOR_STORY_PRESENCE_PENALTY,
    )
    config = yaml.safe_load((ROOT / "configs/runtime_versions.yaml").read_text(encoding="utf-8"))
    profile = config["director"]["story_sampling"]["thinking"]
    _assert(config["director"]["story_max_tokens"] == 3200 and DIRECTOR_STORY_MAX_TOKENS == 3200,
            "story completion cap must be runtime-configured at 3,200 tokens")
    expected = {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
    }
    _assert(profile == expected, f"wrong runtime story profile: {profile}")
    _assert(config["director"]["temperature"] == 0.2 and config["director"]["top_p"] == 0.85,
            "generic Director defaults must remain unchanged; story sampler is a dedicated profile")
    _assert((DIRECTOR_STORY_THINKING_TEMPERATURE, DIRECTOR_STORY_THINKING_TOP_P,
             DIRECTOR_STORY_THINKING_TOP_K, DIRECTOR_STORY_THINKING_MIN_P,
             DIRECTOR_STORY_PRESENCE_PENALTY) == (0.6, 0.95, 20, 0.0, 1.5),
            "loaded story profile differs from runtime_versions.yaml")
    runtime = (ROOT / "planner/qwen_director_runtime.py").read_text(encoding="utf-8")
    _assert("payload[\"top_k\"] = int(DIRECTOR_STORY_THINKING_TOP_K)" in runtime,
            "top_k must come from runtime configuration")
    _assert("payload[\"min_p\"] = float(DIRECTOR_STORY_THINKING_MIN_P)" in runtime,
            "min_p must come from runtime configuration")
    # Exercise the actual request payload without starting a live vLLM server.
    from planner.qwen_director_runtime import QwenDirectorRuntimeMixin
    class Response:
        status_code = 200
        text = ""
        def json(self):
            return {"ok": True}
    class Session:
        def __init__(self):
            self.payload = None
        def post(self, url, json, timeout):
            self.payload = dict(json)
            return Response()
    probe = QwenDirectorRuntimeMixin.__new__(QwenDirectorRuntimeMixin)
    probe._vllm_session = Session()
    probe._vllm_model_name = lambda: "qwen3-14b-awq"
    probe._vllm_base_url = lambda: "http://127.0.0.1:8000/v1"
    probe._post_chat(
        messages=[{"role": "user", "content": "test"}],
        temperature=DIRECTOR_STORY_THINKING_TEMPERATURE,
        top_p=DIRECTOR_STORY_THINKING_TOP_P,
        max_tokens=20,
        creative=True,
        enable_thinking=True,
    )
    sent = probe._vllm_session.payload
    _assert((sent["temperature"], sent["top_p"], sent["top_k"], sent["min_p"], sent["presence_penalty"])
            == (0.6, 0.95, 20, 0.0, 1.5), f"wrong effective sampler sent to vLLM: {sent}")
    _assert(sent["chat_template_kwargs"]["enable_thinking"] is True,
            "request must explicitly retain thinking mode")


def test_story_prompt_restores_september_25_craft_contract():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE

    mixin = QwenDirectorPromptMixin()
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = " ".join(mixin._story_text_system(mode).lower().split())
        _assert("unexpected reveal or reversal" in text,
                f"{mode} must preserve the September 25 reversal requirement")
        _assert("at least two sentences" in text and "interiority" in text,
                f"{mode} must preserve the baseline interiority requirement")
        _assert("short line of spoken dialogue by a named character" in text,
                f"{mode} must preserve purposeful spoken dialogue")
        _assert("400-650 words" in text,
                f"{mode} must restore the September 25 word-range guidance")
        _assert(("complete aftermath paragraph" in text if mode == AI_STORY_MODE else "complete resolution paragraph" in text),
                f"{mode} must require a completed aftermath")
        _assert("include a second physically present" not in text,
                f"{mode} must not force a second character")


def test_story_contract_remains_six_paragraphs_420_560():
    director_source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    prompt_source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    sanitize_source = (ROOT / "planner/qwen_director_sanitize.py").read_text(encoding="utf-8")
    combined = prompt_source + "\n" + sanitize_source
    _assert("six paragraphs" in combined.lower(), "six-paragraph contract missing")
    _assert("self._story_min_words: int = 420" in director_source,
            "story validator must retain the 420-word minimum")
    _assert('minimum_words = int(getattr(self, "_story_min_words", 420) or 420)' in sanitize_source,
            "story validator must use the configured minimum word count")
    _assert("if not minimum_words <= word_count <= 560:" in sanitize_source,
            "story validator must enforce the 560-word maximum")


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


def test_director_critique_is_opt_in_without_empty_recompile_payload():
    import os
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    calls = {"model": 0, "summary": 0}
    director._chat_json = lambda *a, **k: calls.__setitem__("model", calls["model"] + 1) or {}
    director._print_qwen_summary = lambda *a, **k: calls.__setitem__("summary", calls["summary"] + 1)
    saved = {key: os.environ.get(key) for key in ("H3_DIRECTOR_CRITIQUE", "H3_DIRECTOR_CRITIC")}
    try:
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
    _assert('os.getenv("H3_DIRECTOR_CRITIQUE")' in director_source,
            "the canonical critique switch must control both the model call and final telemetry")
    _assert('os.getenv("H3_DIRECTOR_CRITIQUE")' in orchestrator_source,
            "the canonical critique switch must override the runtime manifest feature flag")


def test_story_prompt_uses_baseline_per_mode_sampling_and_source_fidelity():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE

    mixin = QwenDirectorPromptMixin()
    from planner.config import DIRECTOR_STORY_THINKING_TEMPERATURE, DIRECTOR_STORY_THINKING_TOP_P
    expected_sampling = (DIRECTOR_STORY_THINKING_TEMPERATURE, DIRECTOR_STORY_THINKING_TOP_P)
    _assert(mixin._sampling_for_mode(AI_STORY_MODE) == expected_sampling,
            "AI Story should use the configured Qwen3 thinking-mode profile")
    _assert(mixin._sampling_for_mode(EXPAND_USER_STORY_MODE) == expected_sampling,
            "Expand Story should use the configured Qwen3 thinking-mode profile")
    _assert(expected_sampling == (0.6, 0.95),
            f"story sampler should match Qwen3's documented thinking-mode baseline: {expected_sampling}")
    ai = " ".join(mixin._story_text_system(AI_STORY_MODE).lower().split())
    expand = " ".join(mixin._story_text_system(EXPAND_USER_STORY_MODE).lower().split())
    _assert("subvert the obvious" in ai and "specific fear, memory, desire" in ai,
            "AI Story lost the baseline reversal/interiority craft guidance")
    _assert("preserving its important characters, events, chronology, setting, outcome" in expand,
            "Expand Story must preserve its source's established plot")
    _assert("qwen decides names, identities, relationships, and final cast size" in ai and
            "qwen decides names, identities, relationships, and final cast size" in expand,
            "both modes must keep open-cast authorship with Qwen")


def test_story_prompt_preserves_baseline_dialogue_without_cast_quota():
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE

    mixin = QwenDirectorPromptMixin()
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        text = " ".join(mixin._story_text_system(mode).lower().split())
        _assert("spoken dialogue by a named character" in text,
                f"{mode} must preserve the baseline dialogue requirement")
        _assert("qwen decides names, identities, relationships, and final cast size" in text,
                f"{mode} must leave cast authorship to Qwen")
        _assert("include a second physically present, consequential character" not in text,
                f"{mode} must not impose a second-person preference")


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


def test_editor_uses_same_story_contract_and_can_repair_supported_counterpart():
    from planner.qwen_director import QwenDirector
    editor = QwenDirector._STORY_EDITOR_SYSTEM.lower()
    _assert("420-560 words" in editor, "editor must match the primary output contract")
    _assert("story bible is provisional" in editor, "editor must treat the outline as advisory")
    _assert("do not force every outline beat or planned character" in editor, "editor must be free to drop nonessential planned cast")
    _assert("consequential present-day counterpart" in editor, "editor should repair a missing consequential counterpart when premise supports one")
    _assert('os.getenv("h3_story_editor", "0")' in (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8").lower(), "editor should default to audit-gated (auto) mode")


def test_story_profile_is_single_pass_and_skips_optional_story_calls():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    prompt = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    start = source.index("story_user = self._story_text_user(", source.index("def generate("))
    end = source.index("# PASS 1B: deterministic production foundation", start)
    normal_path = source[start:end]
    writer_start = source.index("def _generate_story_once")
    writer_end = source.index("def _story_craft_issues", writer_start)
    writer = source[writer_start:writer_end]
    _assert("max_completion=story_budget" in writer and "story_budget = DIRECTOR_STORY_MAX_TOKENS" in writer,
            "story writer must use the runtime-configured 3,200-token cap")
    _assert("disable_thinking=False" in writer, "story generation must keep thinking enabled")
    _assert("story_thinking=on story_max_tokens={story_budget} temperature={temperature:.2f} top_p={top_p:.2f} top_k={DIRECTOR_STORY_THINKING_TOP_K} min_p={DIRECTOR_STORY_THINKING_MIN_P:.1f} presence_penalty={DIRECTOR_STORY_PRESENCE_PENALTY:.1f} single_pass=true" in writer,
            "story telemetry must identify the runtime-configured sampler and no-retry profile")
    _assert(normal_path.count("self._generate_story_once(") == 1,
            "normal story path must make exactly one writer call")
    for stage in ("_collect_story_findings(", "_resample_for_audit(", "_edit_story_draft(", "story_quality_repair_pass"):
        _assert(stage not in normal_path, f"unnecessary story-stage call remains wired into normal path: {stage}")
    _assert("400-650 words" in prompt, "writing prompt must restore the September 25 word-range guidance")
def test_story_quality_review_helpers_remain_optional_but_are_not_called_by_story_path():
    from planner.qwen_director import QwenDirector
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    generation_start = source.index("story_user = self._story_text_user(", source.index("def generate("))
    generation_end = source.index("# PASS 1B: deterministic production foundation", generation_start)
    generation = source[generation_start:generation_end]
    for forbidden in ("_collect_story_findings(", "_review_story_draft(", "_resample_for_audit(", "_edit_story_draft("):
        _assert(forbidden not in generation, f"normal single-pass path must not invoke {forbidden}")
    review_start = source.index("def _review_story_draft(")
    review_end = source.index("_STORY_EDITOR_SYSTEM", review_start)
    review_code = source[review_start:review_end]
    _assert('call_name="story_quality_review"' in review_code, "optional review helper should remain separately observable")
    _assert("evidence_norm not in normalized_story" in review_code, "optional critic must reject evidence not quoted from draft")
    schema = QwenDirector._story_review_json_schema()
    _assert(schema["properties"]["findings"].get("maxItems") == 5, "optional review schema must remain bounded")
def test_optional_story_review_helpers_are_opt_in_only():
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
        # Optional helpers remain available for explicit experiments, but are not
        # invoked by the normal single-pass story-generation path.
        _assert(director._edit_story_draft(AI_STORY_MODE, "p", story, None, [], temperature=0.6, top_p=0.95) == story,
                "baseline editor must not touch a draft")
        _assert(director._review_story_draft(AI_STORY_MODE, "p", story, None, []) == [],
                "the critic must be off by default")
        _assert(calls == {"chat": 0, "json": 0}, f"baseline should make no review/editor calls: {calls}")
        # The critic remains available as an explicit experiment.
        os.environ["H3_STORY_CRITIC"] = "1"
        _assert(director._review_story_draft(AI_STORY_MODE, "p", story, None, []) == [],
                "an empty explicitly-enabled review should return no findings")
        _assert(calls == {"chat": 0, "json": 1}, f"explicit critic opt-in made an unexpected call: {calls}")
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



def test_story_contract_warnings_do_not_trigger_verification_or_repair_calls():
    import inspect
    from planner.qwen_director import QwenDirector
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    default = inspect.signature(QwenDirector._validate_story_output_contracts).parameters["enforce_planned_cast"].default
    _assert(default is False, "planned cast must not become a hard contract")
    start = source.index("story_user = self._story_text_user(", source.index("def generate("))
    end = source.index("# PASS 1B: deterministic production foundation", start)
    generation = source[start:end]
    _assert("story_quality_verification" not in generation, "normal path must not run a follow-up verification call")
    _assert("story_quality_repair_pass" not in generation, "normal path must not perform automatic story repair")
    _assert("strict=False" in source[source.index("def _generate_story_once"):source.index("def _story_craft_issues")],
            "soft word/source-coverage misses must warn rather than cause another story call")
    mixin = QwenDirectorPromptMixin()
    for mode in (AI_STORY_MODE, EXPAND_USER_STORY_MODE):
        _assert(len(mixin._story_text_system(mode)) < 3500, f"{mode} writing prompt exceeds compact budget")
def test_writer_path_warns_on_soft_contract_misses_without_retry():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    start = source.index("def _generate_story_once")
    end = source.index("def _story_craft_issues", start)
    writer = source[start:end]
    _assert("strict=False" in writer, "single-pass story writer must not abort on soft word/source-coverage misses")
    _assert(writer.count("self._chat_text(") == 1, "soft validation must not trigger another story call")
    _assert("story_static_checks=WARN" in writer, "non-passing static checks must be logged as warnings")
    _assert("semantic_quality=NOT_EVALUATED" in writer, "static contract validation must not claim semantic quality")
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
    _assert("one deliberate plan-before-write call is enabled by default" in director_source, "plan-before-write path marker is missing")
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
def test_candidate_judge_requires_exact_evidence_and_prioritizes_source_fidelity():
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    _assert("position-swap confirmation failed" in source,
            "pairwise choice must fail closed when the reversed comparison disagrees")
    _assert("_quote_is_grounded" in source,
            "candidate preference must cite exact evidence from both drafts")
    _assert("source_fidelity" in source and "do not force" in source.lower(),
            "candidate comparison must prioritize source facts and avoid cast quotas")


def test_story_plan_precedes_one_writer_and_repair_stays_off_by_default():
    import os
    from planner.qwen_director import QwenDirector
    source = (ROOT / "planner/qwen_director.py").read_text(encoding="utf-8")
    start = source.index("story_user = self._story_text_user(", source.index("def generate("))
    end = source.index("# PASS 1B: deterministic production foundation", start)
    normal_path = source[start:end]
    _assert("self._maybe_plan_story_bible(" in normal_path, "plan call must precede prose generation")
    _assert(normal_path.index("self._maybe_plan_story_bible(") < normal_path.index("self._generate_story_once("),
            "the plan must be supplied before writing")
    _assert(normal_path.count("self._generate_story_once(") == 1, "exactly one prose-writing call")
    _assert("_repair_story_if_needed(" not in normal_path and "story_repair_pass" not in normal_path,
            "default path must not retry or automatically repair the story")
    director = QwenDirector.__new__(QwenDirector)
    director._planned_cast = ["stale"]
    calls = []
    director._plan_story_bible = lambda *args: calls.append(args) or {"cast": [{"name": "Mira"}], "beats": ["b"] * 6}
    old = os.environ.pop("H3_STORY_PLAN", None)
    try:
        result = director._maybe_plan_story_bible("ai_story", "premise", [])
        _assert(result is not None and len(calls) == 1, "story planning should default on")
        os.environ["H3_STORY_PLAN"] = "0"
        calls.clear()
        result = director._maybe_plan_story_bible("ai_story", "premise", [])
        _assert(result is None and not calls, "H3_STORY_PLAN=0 must disable planning")
    finally:
        os.environ.pop("H3_STORY_PLAN", None)
        if old is not None:
            os.environ["H3_STORY_PLAN"] = old
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


def test_story_writer_makes_exactly_one_3200_token_thinking_call():
    from planner.qwen_director import QwenDirector
    from planner.config import AI_STORY_MODE

    story_paragraphs = []
    for scene_index in range(6):
        tokens = [
            f"Scene{scene_index + 1}", "Lena", "crossed", "the", "silent", "room", "and", "studied", "the", "marked", "panel."
        ]
        # Build a harmless 450-word test draft; only call count and request kwargs matter.
        while len(tokens) < 75:
            tokens.extend(["She", "checked", "the", "door", "before", "moving", "toward", "the", "next", "marked", "panel."])
        story_paragraphs.append(" ".join(tokens[:75]))
    story = "\n\n".join(story_paragraphs)
    director = QwenDirector.__new__(QwenDirector)
    calls = []

    def fake_chat(*args, **kwargs):
        calls.append(kwargs.copy())
        return story

    director._chat_text = fake_chat
    director._reflow_story_paragraphs = lambda value: value
    def fake_validate(mode, user, value, **kwargs):
        director._last_story_contract_warnings = ["source coverage miss"]
        return value
    director._validate_story_output_contracts = fake_validate
    director._ends_cleanly = lambda value: True
    director._story_quality_issues = lambda value: []
    director._story_craft_issues = lambda value: []
    from contextlib import redirect_stdout
    from io import StringIO
    captured = StringIO()
    with redirect_stdout(captured):
        result, generated = director._generate_story_once(
            AI_STORY_MODE, "premise", "system", "user", temperature=0.6, top_p=0.95,
            source_character_names=[],
        )
    _assert(generated and result == story, "single-pass writer should return the one generated draft")
    _assert(len(calls) == 1, f"writer made more than one model call: {len(calls)}")
    _assert(calls[0].get("max_completion") == 3200, f"writer budget differs from configured 3,200-token contract: {calls[0]}")
    _assert(calls[0].get("disable_thinking") is False, f"thinking mode is not enabled: {calls[0]}")
    _assert(calls[0].get("minimum_output_tokens") == 0, f"writer unexpectedly forced output padding: {calls[0]}")
    log = captured.getvalue()
    _assert("story_static_checks=WARN" in log and "story_static_checks=PASS" not in log,
            f"a source-contract warning must not be followed by a misleading PASS: {log}")


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
        test_story_prompt_does_not_force_cast_size,
        test_character_schema_does_not_silently_cap_qwen_cast_at_32,
        test_qwen_extraction_does_not_force_candidates_from_hints_plan_or_dialogue,
        test_ai_story_does_not_inherit_template_cast_anchors,
        test_story_generation_is_single_pass_thinking_3200_no_floor,
        test_story_writer_makes_exactly_one_3200_token_thinking_call,
        test_qwen3_story_sampler_is_runtime_config_driven,
        test_story_prompt_restores_september_25_craft_contract,
        test_story_contract_remains_six_paragraphs_420_560,
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
        test_director_critique_is_opt_in_without_empty_recompile_payload,
        test_story_prompt_uses_baseline_per_mode_sampling_and_source_fidelity,
        test_story_prompt_preserves_baseline_dialogue_without_cast_quota,
        test_story_bible_cast_schema_is_unbounded_and_plan_is_not_duplicated,
        test_editor_uses_same_story_contract_and_can_repair_supported_counterpart,
        test_story_profile_is_single_pass_and_skips_optional_story_calls,
        test_story_quality_review_helpers_remain_optional_but_are_not_called_by_story_path,
        test_optional_story_review_helpers_are_opt_in_only,
        test_story_contract_warnings_do_not_trigger_verification_or_repair_calls,
        test_writer_path_warns_on_soft_contract_misses_without_retry,
        test_story_prompt_has_no_artificial_subtargets,
        test_build_alias_map_skips_stopword_first_token,
        test_characters_in_scene_does_not_bind_descriptive_by_article,
        test_sanitize_scene_alias_fallback_does_not_bind_descriptive_by_article,
        test_identity_presence_accepts_title_free_and_given_name_surfaces,
        test_best_of_n_is_not_wired_into_normal_story_generation,
        test_story_plan_precedes_one_writer_and_repair_stays_off_by_default,
        test_shot_critique_checks_object_state_across_story,
        test_short_visual_prompts_are_normalized_to_15_40_words_without_model_call,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"Director validation PASSED ({len(tests)} checks).")


if __name__ == "__main__":
    main()

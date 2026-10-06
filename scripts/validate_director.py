from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _has_normalized(source, phrase):
    """Check prompt contracts without depending on source line wrapping."""
    import re
    normalized_source = re.sub(r"\s+", " ", source)
    normalized_phrase = re.sub(r"\s+", " ", phrase)
    return normalized_phrase in normalized_source


def _planner():
    from planner.production_planner import ProductionPlanner
    return ProductionPlanner(ROOT)


def test_deterministic_character_regressions():
    planner = _planner()
    for story, expected in planner.character_detection_regression_cases():
        actual = set(planner.detect_character_descriptors(story))
        _assert(actual == expected, f"detector regression mismatch: {story!r}: {actual} != {expected}")

    single = planner.detect_character_descriptors("Eli entered the vault and Eli looked back.")
    _assert("Eli" in single, f"one-word deterministic character was lost: {single}")


def test_story_prompt_restores_successful_compact_narrative_contract():
    source = Path(ROOT, "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    from planner.qwen_director_prompts import QwenDirectorPromptMixin
    from planner.config import AI_STORY_MODE, EXPAND_USER_STORY_MODE
    _mixin = QwenDirectorPromptMixin()
    ai = _mixin._story_text_system(AI_STORY_MODE)
    expand = _mixin._story_text_system(EXPAND_USER_STORY_MODE)
    for label, text in (("AI story", ai), ("Expand story", expand)):
        _assert("exactly six paragraphs" in text.lower(), f"{label} prompt must produce six scene-sized paragraphs")
        _assert(("420 to 560 words" in text) or ("420-560" in text), f"{label} prompt must state its word target")
        _assert("hard format" in text.lower() and "absolute allowed range 420-560" in text.lower(), f"{label} prompt must make the word budget operationally hard")
        _assert("canonical" in text.lower() and "name" in text.lower(), f"{label} prompt must require canonical character identity")
        for concept in ("goal", "resistance", "reversal", "choice", "consequence"):
            _assert(concept in text.lower(), f"{label} prompt is missing the core {concept} concept")
        _assert(
            "cause, choose, or misjudge" in text.lower() and "must not be" in text.lower() and "sentient" in text.lower(),
            f"{label} prompt must make the reversal action-driven and ban the generic hidden-object/sentient-system reveal",
        )
        _assert(
            "price" in text.lower() and "choice" in text.lower(),
            f"{label} prompt must make the choice cost something concrete",
        )
        _assert("no markdown" in text.lower(), f"{label} prompt must forbid markdown emphasis")
        _assert("dialogue" in text.lower() and "recordings" in text.lower(), f"{label} prompt must keep non-spoken media out of dialogue")
        _assert("completed past-tense action" in text.lower(), f"{label} prompt must require a completed final action")
        _assert("organism" not in text.lower(), f"{label} prompt must not prime hook imagery")
    _assert("short lines" in ai.lower() and "spoken dialogue" in ai.lower(), "AI story prompt must keep dialogue compact and spoken")
    _assert("stable canonical name" in expand.lower() and "preserve established characters" in expand.lower(), "Expand prompt must preserve source character identity")
    _assert("add additional named, relational, or descriptive recurring characters" in expand.lower(), "Expand prompt must allow Qwen to create additional consequential characters")
    _assert("not a cast limit" in expand.lower(), "Expand source anchors must not become a cast whitelist")
    _assert("exactly as given" not in expand.lower(), "Expand prompt must not over-constrain source names with an artificial exact-only cast rule")
    _assert("concrete" in ai.lower() and "physical action" in ai.lower(), "AI story prompt must prioritize immediate filmable action")
    _assert("PLAN:" not in ai and "LEDGER:" not in ai, "AI story prompt must not expose a planning/checklist format")
    _assert("<option A>" not in ai and "<option B>" not in ai, "AI story prompt must not force a binary choice template")
    _assert("source_dialogue" in source, "shot prompt must receive an explicit source-dialogue whitelist")
    _assert("If `is_character=true`, `entity_type` MUST be PERSON, CHARACTER, or SENTIENT" in source, "character extraction must keep identity type and entity type consistent")
    _assert("Reject pronouns, contractions, sentence fragments" in source, "character extraction must reject prose fragments as identities")
    _assert("final character must end" not in source.lower(), "obsolete final-character wording remains")
    for leaked_name in ("Eli's father", "Sara's sister", "Mira's commander", "Eli", "Lin Mei"):
        _assert(leaked_name not in source, f"prompt must not seed a hard-coded character name: {leaked_name}")


def test_expand_source_fallback_removed():
    source = Path(ROOT, "planner", "qwen_director.py").read_text(encoding="utf-8")
    _assert("expand_story_source_fallback" not in source, "Expand Story must not silently fall back to preserve-story mode")
    prompts = Path(ROOT, "planner", "qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert("Expand Story generation failed validation: " in source, "Expand Story must still fail closed when no attempt is valid")


def test_logged_story_rosters_are_not_poisoned_by_prose_surfaces():
    from planner.production_planner import ProductionPlanner
    planner = ProductionPlanner(ROOT)
    ai_story = (
        "Elara Voss clawed through the snowdrifts. Dr. Kess's final transmission had warned her. Dr. Kess entered the vault. "
        "Below, the air was thick with ozone. A voice announced, \"Unauthorized access detected. Initiation of Protocol Epsilon.\" "
        "Elara sprinted to the terminal."
    )
    ai_names = planner.detect_character_descriptors(ai_story)
    _assert(set(ai_names) == {"Elara Voss", "Kess"}, f"AI Story prose poisoned the deterministic roster: {ai_names}")

    expand_story = (
        "Eli reached the vault. His sister's voice warned him. His father had vanished years ago. "
        "A photograph of a child with his sister's eyes lay inside the crate. Eli sprinted to the exit."
    )
    expand_names = planner.detect_character_descriptors(expand_story)
    _assert(expand_names == ["Eli"], f"Expand Story visual/relational prose poisoned the deterministic roster: {expand_names}")
    relations = planner._extract_relational_character_hints(expand_story, expand_names)
    _assert(
        [item["name"] for item in relations] == ["Eli's sister", "Eli's father"],
        f"Expand Story relational identities were not grounded correctly: {relations}",
    )



def test_malformed_qwen_character_roster_fails_closed_without_adjudication():
    planner = _planner()
    story = "Liora Venn reached the Arctic station. Everything had changed. Elias Kren waited beside the vault."

    def extractor(*_args):
        raise ValueError("malformed character JSON")

    try:
        planner.create_characters(
            story,
            qwen_character_extractor=extractor,
            qwen_character_adjudicator=None,
        )
    except ValueError as exc:
        _assert("malformed character JSON" in str(exc), f"wrong failure from Qwen character pass: {exc}")
    else:
        raise AssertionError("malformed Qwen character output must fail closed; deterministic roster discovery must not take over")


def test_dialogue_speaker_is_bound_even_when_shot_starts_empty():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    director._qwen_telemetry = {"deterministic_recoveries": 0}
    story = 'Eli faced the door. "Open it," Eli said.'
    characters = [{"name": "Eli", "identity_type": "named_character", "semantic_aliases": []}]
    scenes = [{"scene_id": "scene_001", "characters": []}]
    shots = [{
        "shot_id": "scene_001_shot_001",
        "scene_id": "scene_001",
        "characters": [],
        "dialogue_events": [{"speaker": "Eli", "text": "Open it,"}],
    }]
    director._normalize_dialogue_speakers(story, scenes, shots, characters)
    _assert(shots[0]["characters"] == ["Eli"], f"empty shot lost its dialogue speaker binding: {shots[0]}")
    _assert(scenes[0]["characters"] == ["Eli"], f"empty scene lost its dialogue speaker binding: {scenes[0]}")
    director._validate_dialogue_speaker_contract(shots, characters)


def test_semantic_named_surface_safety_boundary():
    from planner.production_planner import ProductionPlanner
    _assert(ProductionPlanner._semantic_named_surface_is_safe("Elara Voss"), "valid proper name was rejected")
    for value in ("child with his sister's eyes", "father was", "her instead", "Everything"):
        _assert(
            not ProductionPlanner._semantic_named_surface_is_safe(value),
            f"non-name semantic surface was accepted: {value!r}",
        )
    protocol_context = "Protocol Epsilon was a self-destruct sequence. Unauthorized access detected."
    _assert(
        "Protocol Epsilon" not in ProductionPlanner(".").detect_character_descriptors(protocol_context),
        "contextually explicit protocol entity leaked into the deterministic roster",
    )


def test_qwen_character_roster_is_authoritative_and_filters_noncharacters():
    planner = _planner()
    story = "Elias Kade entered the vault and Lin Mei followed him. Access appeared on the monitor."
    calls = []

    def extractor(_story, required):
        calls.append(("extract", list(required)))
        return {"candidates": [
            {"name": "Elias Kade", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
            {"name": "Lin Mei", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
            {"name": "Access", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
        ]}

    names = [c.name for c in planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=None,
    )]
    _assert(names == ["Elias Kade", "Lin Mei"], f"planner failed to bounded-filter the Qwen roster: {names}")
    _assert(len(calls) == 1, f"creative roster must use exactly one Qwen character pass: {calls}")


def test_qwen_character_roster_can_introduce_new_characters_without_source_whitelist():
    planner = _planner()
    story = "Elias Kade entered the vault. Lin Mei followed him. Mara Venn locked the door behind them."

    def extractor(_story, required):
        _assert(required == [], f"AI Story must not receive an artificial source-cast whitelist: {required}")
        return {"candidates": [
            {"name": "Elias Kade", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
            {"name": "Lin Mei", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
            {"name": "Mara Venn", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
        ]}

    names = [c.name for c in planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=None,
    )]
    _assert(names == ["Elias Kade", "Lin Mei", "Mara Venn"], f"Qwen-created character was incorrectly filtered by source whitelist: {names}")


def test_qwen_negative_character_decision_is_respected():
    planner = _planner()
    story = "Elias Kade entered the vault and Lin Mei followed him."

    def extractor(*_args):
        return {"candidates": [
            {"name": "Elias Kade", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
            {"name": "Lin Mei", "entity_type": "PERSON", "is_character": False, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
        ]}

    names = [c.name for c in planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=None,
    )]
    _assert(names == ["Elias Kade"], f"planner overrode Qwen's explicit negative character decision: {names}")


def test_expand_qwen_roster_preserves_source_and_allows_new_characters():
    planner = _planner()
    story = "Eli entered the station. His father waited by the door. Mara Venn checked the generator."

    def extractor(_story, required):
        _assert(required == ["Eli"], f"Expand should pass only established source anchors: {required}")
        return {"candidates": [
            {"name": "Eli", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
            {"name": "Eli's father", "entity_type": "PERSON", "is_character": True, "aliases": ["his father"], "identity_type": "relational_character", "relationship_to": "Eli", "relationship": "father"},
            {"name": "Mara Venn", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
        ]}

    names = [c.name for c in planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=None,
        required_character_names=["Eli"],
    )]
    _assert(names == ["Eli", "Eli's father", "Mara Venn"], f"Expand cast was incorrectly source-locked: {names}")


def test_expand_qwen_relational_character_is_accepted_without_adjudication():
    planner = _planner()
    story = (
        "Eli stepped through the station. His father stood beside the vault and watched him. "
        "Mara Venn checked the generator before the lights failed."
    )

    def extractor(_story, required):
        _assert(required == ["Eli"], f"unexpected Expand source anchors: {required}")
        return {"candidates": [
            {"name": "Eli", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
            {"name": "Eli's father", "entity_type": "CHARACTER", "is_character": True, "aliases": ["his father"], "identity_type": "relational_character", "relationship_to": "Eli", "relationship": "father"},
            {"name": "Mara Venn", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
        ]}

    characters = planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=None,
        required_character_names=["Eli"],
    )
    names = [c.name for c in characters]
    _assert(names == ["Eli", "Eli's father", "Mara Venn"], f"relational/new Qwen roster was not preserved: {names}")


def test_relational_character_survives_qwen_roster_and_resolves_dialogue():
    from planner.qwen_director import QwenDirector

    story = (
        "Eli entered the vault. His father had disappeared years ago. "
        "An older man stepped forward and said, \"You should not be here.\""
    )
    planner = _planner()

    def extractor(*_args):
        return {"candidates": [
            {"name": "Eli", "entity_type": "PERSON", "is_character": True,
             "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
            {"name": "Eli's father", "entity_type": "CHARACTER", "is_character": True,
             "aliases": ["his father", "man"], "identity_type": "relational_character",
             "relationship_to": "Eli", "relationship": "father"},
        ]}

    characters = planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=None,
    )
    names = [c.name for c in characters]
    _assert("Eli's father" in names and "man" not in names and "All" not in names,
            f"generic role leaked into Qwen canonical roster: {names}")

    payload = [c.to_dict() for c in characters]
    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    scenes = [{"scene_id": "scene_001", "characters": ["Eli"]}]
    shots = [{
        "shot_id": "scene_001_shot_001",
        "scene_id": "scene_001",
        "characters": ["Eli"],
        "dialogue_events": [{
            "speaker": "man",
            "text": "You should not be here.",
            "continues_from_previous_shot": False,
            "continues_to_next_shot": False,
        }],
        "speaking_characters": ["man"],
        "speech_text": "You should not be here.",
    }]
    director._normalize_dialogue_speakers(story, scenes, shots, payload)
    event = shots[0]["dialogue_events"][0]
    _assert(event["speaker"] == "Eli's father", f"generic dialogue speaker did not resolve: {event}")
    _assert("Eli's father" in shots[0]["characters"], f"shot binding missing relational speaker: {shots[0]['characters']}")
    _assert("Eli's father" in scenes[0]["characters"], f"scene binding missing relational speaker: {scenes[0]['characters']}")


def test_sanitizer_identity_contract():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    payload = [{
        "name": "Eli's father",
        "identity_type": "relational_character",
        "relationship_to": "Eli",
        "relationship": "father",
        "semantic_aliases": ["man"],
        "identity_profile": {
            "identity_type": "relational_character",
            "relationship_to": "Eli",
            "relationship": "father",
            "semantic_aliases": ["man"],
        },
    }]
    result = director._sanitize_characters(payload)
    _assert(len(result) == 1, f"sanitizer unexpectedly removed valid relational character: {result}")
    char = result[0]
    _assert(char["identity_type"] == char["identity_profile"]["identity_type"], "identity_type mismatch")
    _assert(char["relationship_to"] == char["identity_profile"]["relationship_to"], "relationship_to mismatch")
    _assert(char["relationship"] == char["identity_profile"]["relationship"], "relationship mismatch")
    _assert(char["semantic_aliases"] == char["identity_profile"]["semantic_aliases"], "semantic_aliases mismatch")


def test_qwen_cache_generation_contract():
    # Static behavioral contract: the cache key must change when material
    # generation configuration changes.
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    director._model_path = "/model/a"
    director._cache_namespace = "test"
    base = dict(
        call_name="characters",
        system_prompt="system",
        user_prompt="story",
        response_schema={"type": "object"},
        json_mode=True,
        disable_thinking=True,
        temperature=0.1,
        top_p=0.9,
        max_tokens=256,
    )
    first = director._cache_key(**base)
    second = director._cache_key(**{**base, "temperature": 0.2})
    third = director._cache_key(**{**base, "max_tokens": 512})
    _assert(first != second, "temperature must participate in Qwen cache key")
    _assert(first != third, "max_tokens must participate in Qwen cache key")


def test_disabled_director_path():
    from planner.qwen_director import QwenDirector

    previous = os.environ.get("H3_DIRECTOR_ENABLED")
    os.environ["H3_DIRECTOR_ENABLED"] = "0"
    try:
        director = QwenDirector(ROOT)
        plan = {
            "story": "Eli enters a station.",
            "characters": [{"name": "Eli"}],
            "scenes": [{
                "scene_id": "scene_001",
                "characters": ["Eli"],
                "description": "An abandoned station.",
                "lighting": "cool natural light",
                "color_temperature": "cool",
                "mood": "tense",
                "location": "abandoned station",
            }],
            "shots": [{
                "shot_id": "scene_001_shot_001",
                "scene_id": "scene_001",
                "characters": ["Eli"],
                "action": "Eli enters the station.",
                "duration_seconds": 5.0,
            }],
        }
        result = director.generate(
            mode="preserve_user_story",
            user_input=plan["story"],
            base_plan=plan,
        )
        _assert(result["enabled"] is False, "disabled Director must remain disabled")
        _assert(result["plan"]["shots"], "disabled Director must produce deterministic shots")
        shot = result["plan"]["shots"][0]
        for field in ("camera_shot", "camera_movement", "lens_and_depth_of_field", "composition_notes", "lighting", "color_temperature", "mood", "visual_prompt"):
            _assert(str(shot.get(field, "")).strip(), f"disabled Director path left required shot field empty: {field}")
    finally:
        if previous is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = previous



def test_story_token_budget_contract():
    source = Path(ROOT, "planner/qwen_director.py").read_text(encoding="utf-8")
    # Production story generation is intentionally locked to one Qwen3 thinking configuration.
    _assert(
        "minimum_completion=600," in source,
        "story pass must reserve enough completion budget for the context-length check",
    )
    _assert(
        "minimum_output_tokens=story_min_output_tokens," in source and "story_min_output_tokens = 0" in source,
        "story pass must NOT use a min_tokens floor: it suppresses EOS and forces padding after a finished story",
    )
    _assert(
        '"[QWEN] story_thinking=on story_max_tokens=3200"' in source,
        "story pass must be locked to thinking mode with a 3200-token ceiling",
    )
    _assert(
        "max_completion=3200," in source and "disable_thinking=False," in source,
        "story pass must route fixed thinking mode into Qwen3",
    )
    _assert(
        "H3_DIRECTOR_STORY_THINKING" not in source,
        "production story generation must not expose an A/B environment switch",
    )
    _assert(
        "thinking_token_budget=1600," not in source,
        "story pass must not impose an explicit thinking-token budget",
    )
    _assert(
        "seed=DIRECTOR_VLLM_SEED," in source,
        "story pass must remain deterministic without retry-dependent seed changes",
    )
    _assert("_extract_story_body" not in source, "story flow must not require a visible PLAN/STORY wrapper")
    _assert("ai_story_text_retry" not in source, "story generation must not contain a second Qwen creative pass")
    _assert("expand_story_text_retry" not in source, "expand story generation must not contain a second Qwen creative pass")
    _assert("H3_DIRECTOR_STORY_ATTEMPTS" not in source and "H3_STORY_MAX_ATTEMPTS" not in source, "story generation must not expose a retry loop")
    _assert("qwen_character_extractor=self.extract_character_entities" in source, "creative story path must always resolve characters through Qwen semantic extraction")
    _assert("qwen_character_adjudicator=None" in source, "creative story path must not perform a second character adjudication call")
    _san = Path(ROOT, "planner", "qwen_director_sanitize.py").read_text(encoding="utf-8")
    _assert("minimum_token_overlap=0.40" not in _san, "AI Story premise gate must use the permissive default, not the 0.40 override")
    _assert("max_completion=1500" not in source, "obsolete 1500-token story ceiling remains")

def test_story_salvage_and_quality_gate():
    from planner.qwen_director import QwenDirector

    d = QwenDirector.__new__(QwenDirector)
    paras = [f"Paragraph {i} walks the quiet harbour and ends cleanly." for i in range(6)]
    looped = "\n\n".join(paras + paras[:3] + paras[:3])
    out = d._salvage_runaway_story(looped)
    _assert(out.count("\n\n") == 5, "salvage must keep exactly the first six paragraphs")
    _assert(d._salvage_runaway_story("one\n\ntwo") == "", "too-short drafts must not be salvaged")
    cliche = ("Her heart pounded. Little did she know. " * 3 + "She ran. " * 6)
    _assert(d._story_quality_issues(cliche), "cliché/repetition must be flagged")
    good = (
        "Marta wedged the crowbar under the shutter and leaned until the rust gave. "
        "Salt wind pushed through the gap, smelling of diesel and low tide. "
        "\"Leave it,\" Joao said from the pier. She did not. "
        "Inside, rows of crates sagged under tarpaulin, each stencilled with her father's initials. "
        "A gull screamed. She dragged the nearest crate into the light, cut the cord, and counted the reels."
    )
    _assert(len(d._story_quality_issues(good)) == 0, f"clean prose flagged: {d._story_quality_issues(good)}")


def test_story_sampling_guards_are_story_only_and_warmup_matches():
    source = Path(ROOT, "planner/qwen_director_runtime.py").read_text(encoding="utf-8")
    _assert('payload["top_k"] = 20' in source, "story top_k guard must be present")
    _assert('if creative:' in source and 'payload["presence_penalty"]' in source, "presence penalty must be story-only")
    _assert('warmup_payload' in source and '"top_k": 20' in source, "sampler warmup must cover story top-k path")
    _assert('"chat_template_kwargs": {"enable_thinking": True}' in source, "warmup must use the thinking chat template")
    _assert('"--reasoning-parser",\n                "qwen3"' in source, "vLLM server must enable the Qwen3 reasoning parser")
    _assert('"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}' in source, "runtime must explicitly control Qwen3 thinking mode")
    _assert(
        "any(\n            key in response.text for key in (\"presence_penalty\", \"top_k\")\n        )" in source,
        "sampling fallback must remain scoped to optional extras",
    )
    _assert('payload.pop("chat_template_kwargs"' not in source and 'payload.pop("thinking_token_budget"' not in source, "fallback must never silently disable thinking")


def test_story_thinking_arguments_reach_vllm_without_a_retry():
    from planner.qwen_director_runtime import QwenDirectorRuntimeMixin

    runtime = QwenDirectorRuntimeMixin.__new__(QwenDirectorRuntimeMixin)
    runtime._vllm_session = object()
    runtime._qwen_telemetry = {
        "calls": [],
        "total_elapsed_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "retries": 0,
        "cache_hits": 0,
        "deterministic_recoveries": 0,
    }
    captured = {}

    runtime._available_output_tokens = lambda *_args, **_kwargs: (None, 3200)

    def fake_post_chat(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [{
                "message": {"content": "A complete test story."},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 18,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        }

    runtime._post_chat = fake_post_chat
    runtime._trace_call = lambda *args, **kwargs: None

    out = runtime._chat_text(
        "system",
        "user",
        minimum_completion=600,
        temperature=0.6,
        top_p=0.95,
        call_name="ai_story_text_pass",
        max_completion=3200,
        disable_thinking=False,
        minimum_output_tokens=0,
        seed=123,
        creative=True,
    )
    _assert(out == "A complete test story.", "story test generation returned unexpected text")
    _assert(captured.get("enable_thinking") is True, "story thinking mode did not reach _post_chat")
    _assert("thinking_token_budget" not in captured or captured.get("thinking_token_budget") is None, "story path must not send an explicit thinking token budget")
    _assert(captured.get("max_tokens") == 3200, "story max token budget did not reach _post_chat")
    _assert(captured.get("creative") is True, "story creative sampling flag did not reach _post_chat")
    _assert(captured.get("messages", [{}])[-1].get("content") == "user", "thinking story must not append /no_think")
    _assert(runtime._qwen_telemetry["calls"][0]["reasoning_tokens"] == 7, "reasoning token telemetry was not captured")


def test_parallel_stage_copy_is_byte_exact():
    import hashlib, tempfile
    from planner import qwen_director_runtime as runtime

    root = tempfile.mkdtemp()
    for size in (5, 64 * 1024 * 1024 + 1):
        src, dst = f"{root}/s{size}", f"{root}/d{size}"
        with open(src, "wb") as handle:
            handle.write(os.urandom(size))
        runtime._parallel_copy_file(src, dst, workers=4)
        digest = lambda path: hashlib.sha256(open(path, "rb").read()).hexdigest()
        _assert(digest(src) == digest(dst), f"parallel copy corrupted a {size}-byte file")
    previous = os.environ.pop("H3_DIRECTOR_STAGE_LOCAL", None)
    try:
        _assert(runtime._stage_mode() == "auto", "staging must default to filesystem-adaptive auto mode")
    finally:
        if previous is not None:
            os.environ["H3_DIRECTOR_STAGE_LOCAL"] = previous


def test_quoted_dialogue_speaker_comes_from_speech_tag():
    from planner.qwen_director import QwenDirector
    cases = (
        ('"Nobody goes up," Ines said. She held the chain.', {"ines"}),
        ('Tomas turned. "Move aside," said Tomas.', {"tomas"}),
        ('Ines whispered, "It is already open."', {"ines"}),
        ('"Access granted," the terminal said.', set()),
        ('"Run," she said.', set()),
    )
    for text, expected in cases:
        segments = QwenDirector._extract_story_spoken_segments(text)
        got = set().union(*[s.get("tag_speakers", set()) for s in segments]) if segments else set()
        _assert(got == expected, f"speech-tag attribution wrong for {text!r}: {got} != {expected}")

def test_creative_named_only_story_still_calls_qwen_character_extractor():
    planner = _planner()
    story = "Elias Kade entered the Arctic station. Lin Mei followed him into the vault."
    calls = []

    def extractor(_story, required):
        calls.append("extract")
        _assert(required == [], f"AI Story received an artificial cast limit: {required}")
        return {"candidates": [
            {"name": "Elias Kade", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
            {"name": "Lin Mei", "entity_type": "PERSON", "is_character": True, "aliases": [], "identity_type": "named_character", "relationship_to": "", "relationship": ""},
        ]}

    names = [c.name for c in planner.create_characters(
        story,
        qwen_character_extractor=extractor,
        qwen_character_adjudicator=None,
    )]
    _assert(names == ["Elias Kade", "Lin Mei"], f"Qwen roster changed unexpectedly: {names}")
    _assert(calls == ["extract"], f"creative roster must always use the Qwen character pass: {calls}")

def test_context_ir_capture_root():
    from pipeline.context_ir import H3ContextIRCompiler
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ctx"
        plan = {
            "production_id": "p1",
            "context_ir_capture_root": str(root),
            "story": "Eli enters the station.",
        }
        shot = {
            "shot_id": "shot_001",
            "scene_id": "scene_001",
            "location": "station",
            "duration_seconds": 5.0,
            "camera_shot": "medium",
            "camera_movement": "static",
            "lens_and_depth_of_field": "deep focus",
            "composition_notes": "centered",
            "lighting": "soft light",
            "mood": "tense",
            "action": "Eli enters.",
            "visual_prompt": "Eli enters the station.",
            "characters": ["Eli"],
            "dialogue_events": [],
            "reference_images": [],
            "reference_videos": [],
            "reference_audio_paths": [],
            "reference_roles": [],
        }
        result = H3ContextIRCompiler().compile(plan, shot)
        path = Path(result["context_ir_provenance"]["capture_path"])
        _assert(path.parent == root.resolve(), f"Context-IR ignored requested capture root: {path}")


def test_checkpoint_digest_excludes_runtime_outputs():
    from pipeline.production_checkpoint import ProductionCheckpoint
    base = {"story": "Eli enters.", "shots": [{"shot_id": "s1", "action": "enter"}]}
    digest = ProductionCheckpoint.plan_digest(base)
    mutated = dict(base)
    mutated.update({
        "final_video": "/tmp/final.mp4",
        "shot_outputs": ["/tmp/s1.mp4"],
        "job_result": {"final_video": "/tmp/final.mp4"},
        "completed_shot_ids": ["s1"],
    })
    _assert(digest == ProductionCheckpoint.plan_digest(mutated), "output/runtime metadata changed plan digest")


def test_job_state_clears_stale_completion():
    from pipeline.production_plan_store import ProductionPlanStore
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plan.json"
        ProductionPlanStore.atomic_save(path, {
            "job_id": "old",
            "job_status": "completed",
            "final_video": "/old/final.mp4",
            "job_result": {"final_video": "/old/final.mp4"},
        })
        updated = ProductionPlanStore.set_job_state_unlocked(
            path,
            job_id="new",
            status="queued",
            result={},
        )
        _assert("final_video" not in updated and "job_result" not in updated, "stale completion state survived queue transition")



def test_unresolved_explicit_dialogue_fails_closed():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    director._qwen_telemetry = {"deterministic_recoveries": 0}
    story = 'Eli entered the vault. A stranger said, "Run."'
    characters = [{
        "name": "Eli",
        "identity_type": "named_character",
        "semantic_aliases": [],
    }]
    scenes = [{"scene_id": "scene_001", "characters": ["Eli"]}]
    shots = [{
        "shot_id": "scene_001_shot_001",
        "scene_id": "scene_001",
        "characters": ["Eli"],
        "dialogue_events": [{
            "speaker": "stranger",
            "text": "Run.",
        }],
    }]
    try:
        director._normalize_dialogue_speakers(story, scenes, shots, characters)
    except RuntimeError as exc:
        _assert("speaker" in str(exc).lower() and "stranger" in str(exc),
                f"unexpected unresolved-dialogue error: {exc}")
    else:
        raise AssertionError("unresolved explicit dialogue was silently accepted/dropped")


def test_wrong_shot_speaker_is_corrected_by_speech_tag():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    director._qwen_telemetry = {"deterministic_recoveries": 0}
    story = 'Tomas pushed the door. "Nobody goes up," Ines said. "Then stop me," Tomas said.'
    characters = [
        {"name": "Tomas", "identity_type": "named_character", "semantic_aliases": []},
        {"name": "Ines", "identity_type": "named_character", "semantic_aliases": []},
    ]
    scenes = [{"scene_id": "scene_001", "characters": ["Tomas", "Ines"]}]
    shots = [{
        "shot_id": "scene_001_shot_001",
        "scene_id": "scene_001",
        "characters": ["Tomas", "Ines"],
        "dialogue_events": [
            {"speaker": "Tomas", "text": "Nobody goes up,"},
            {"speaker": "Tomas", "text": "Then stop me,"},
        ],
    }]
    director._normalize_dialogue_speakers(story, scenes, shots, characters)
    speakers = [e["speaker"] for e in shots[0]["dialogue_events"]]
    _assert(speakers == ["Ines", "Tomas"], f"speech tag must correct a wrong shot speaker: {speakers}")
    _assert(shots[0]["speaking_characters"] == ["Ines", "Tomas"], "speaking_characters must follow corrected speakers")


def test_canonical_dialogue_speaker_is_rebound_into_shot_and_scene():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    director._qwen_telemetry = {"deterministic_recoveries": 0}
    story = 'Eli met his father. "I was trying to protect you," his father said.'
    characters = [
        {
            "name": "Eli",
            "identity_type": "named_character",
            "semantic_aliases": [],
        },
        {
            "name": "Eli's father",
            "identity_type": "relational_character",
            "relationship_to": "Eli",
            "relationship": "father",
            "semantic_aliases": ["his father", "father", "man"],
        },
    ]
    scenes = [{"scene_id": "scene_001", "characters": ["Eli"]}]
    shots = [{
        "shot_id": "scene_001_shot_001",
        "scene_id": "scene_001",
        "characters": ["Eli"],
        "dialogue_events": [{
            "speaker": "Eli's father",
            "text": "I was trying to protect you,",
        }],
    }]
    director._normalize_dialogue_speakers(story, scenes, shots, characters)
    event = shots[0]["dialogue_events"][0]
    _assert(event["speaker"] == "Eli's father", f"speaker was not canonicalized: {event}")
    _assert("Eli's father" in shots[0]["characters"], "canonical speaker missing from shot binding")
    _assert("Eli's father" in scenes[0]["characters"], "canonical speaker missing from scene binding")


def test_downstream_production_preserves_dialogue_multiset():
    from pipeline.production_orchestrator import ProductionOrchestrator

    before = ProductionOrchestrator._snapshot_dialogue_contract({
        "shots": [{
            "shot_id": "scene_001_shot_001",
            "dialogue_events": [{
                "speaker": "Eli",
                "text": "I was trying to protect you.",
            }],
        }]
    })
    ProductionOrchestrator._assert_dialogue_contract_preserved(before, {
        "shots": [{
            "shot_id": "scene_001_shot_001",
            "dialogue_events": [{
                "speaker_name": "Eli",
                "speaker_id": "char_eli",
                "text": "I was trying to protect you.",
            }],
        }]
    })

    try:
        ProductionOrchestrator._assert_dialogue_contract_preserved(before, {
            "shots": [{
                "shot_id": "scene_001_shot_001",
                "dialogue_events": [],
            }]
        })
    except RuntimeError:
        pass
    else:
        raise AssertionError("downstream dialogue deletion was not detected")

def test_dialogue_scene_boundary_clears_stale_continuation():
    from pipeline.dialogue_timeline import DialogueTimeline

    characters = [{"name": "Eli", "character_id": "eli"}]
    plan = {
        "shots": [
            {
                "shot_id": "scene_001_shot_001",
                "scene_id": "scene_001",
                "is_scene_boundary": True,
                "duration_seconds": 4.0,
                "characters": ["Eli"],
                "dialogue_events": [
                    {
                        "speaker": "Eli",
                        "text": "A fresh scene begins here.",
                        "continues_from_previous_shot": True,
                        "continues_to_next_shot": False,
                    }
                ],
            }
        ]
    }
    DialogueTimeline(characters).apply_to_plan(plan)
    event = plan["shots"][0]["dialogue_events"][0]
    _assert(
        event["continues_from_previous_shot"] is False,
        "scene-boundary normalization did not persist the cleared continuation flag",
    )


def test_shot_prompt_requires_same_speaker_continuation_rule():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert(
        "A continuation edge is SAME-SPEAKER ONLY" in source,
        "shot prompt must require same-speaker continuation",
    )
    _assert(
        "Never set `continues_to_next_shot=true` when the next shot begins with a different speaker." in source,
        "shot prompt must forbid cross-speaker continuation",
    )
    _assert(
        "source_dialogue` whitelist" in source,
        "shot prompt must identify the exact source-dialogue whitelist",
    )
    _assert(
        "must never invent a new line from narrative prose" in source,
        "shot prompt must forbid narrative-to-dialogue invention",
    )


def test_dialogue_continuation_requires_same_canonical_speaker():
    from copy import deepcopy
    from planner.qwen_director import QwenDirector
    from pipeline.dialogue_timeline import DialogueTimeline

    previous = os.environ.get("H3_DIRECTOR_ENABLED")
    os.environ["H3_DIRECTOR_ENABLED"] = "0"
    try:
        director = QwenDirector(ROOT)
    finally:
        if previous is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = previous

    characters = [
        {"name": "Elias Kade", "character_id": "elias_kade"},
        {"name": "Lin Mei", "character_id": "lin_mei"},
    ]
    scenes = [{"scene_id": "scene_001"}]

    # A safe alias for the SAME character must preserve continuation.
    alias_same = [
        {
            "shot_id": "scene_001_shot_001",
            "scene_id": "scene_001",
            "characters": ["Elias Kade"],
            "duration_seconds": 6.0,
            "dialogue_events": [{
                "speaker": "Elias",
                "text": "We have to go now.",
                "continues_from_previous_shot": False,
                "continues_to_next_shot": True,
            }],
        },
        {
            "shot_id": "scene_001_shot_002",
            "scene_id": "scene_001",
            "characters": ["Elias Kade"],
            "duration_seconds": 6.0,
            "dialogue_events": [{
                "speaker": "Elias Kade",
                "text": "Before they find us.",
                "continues_from_previous_shot": False,
                "continues_to_next_shot": False,
            }],
        },
    ]
    director._normalize_dialogue_continuations(scenes, alias_same, characters)
    _assert(
        alias_same[0]["dialogue_events"][0]["continues_to_next_shot"] is True
        and alias_same[1]["dialogue_events"][0]["continues_from_previous_shot"] is True,
        "same-character aliases were incorrectly broken at the continuation boundary",
    )

    # A real speaker change must break continuation deterministically.
    speaker_change = [
        {
            "shot_id": "scene_001_shot_001",
            "scene_id": "scene_001",
            "characters": ["Elias Kade"],
            "duration_seconds": 6.0,
            "dialogue_events": [{
                "speaker": "Elias Kade",
                "text": "We have to go now.",
                "continues_from_previous_shot": False,
                "continues_to_next_shot": True,
            }],
        },
        {
            "shot_id": "scene_001_shot_002",
            "scene_id": "scene_001",
            "characters": ["Lin Mei"],
            "duration_seconds": 6.0,
            "dialogue_events": [{
                "speaker": "Lin Mei",
                "text": "No. We stay.",
                "continues_from_previous_shot": False,
                "continues_to_next_shot": False,
            }],
        },
    ]
    director._normalize_dialogue_continuations(scenes, speaker_change, characters)
    _assert(
        speaker_change[0]["dialogue_events"][0]["continues_to_next_shot"] is False
        and speaker_change[1]["dialogue_events"][0]["continues_from_previous_shot"] is False,
        "speaker-change boundary was incorrectly preserved as a continuation",
    )
    try:
        DialogueTimeline(characters).apply_to_plan({"shots": deepcopy(speaker_change)})
    except Exception as exc:
        raise AssertionError(
            f"speaker-change continuation repair still violates the dialogue scheduler: {exc}"
        ) from exc


def test_dialogue_h3_feasibility_rebalances_generated_shots():
    from copy import deepcopy
    from planner.qwen_director import QwenDirector
    from pipeline.dialogue_timeline import DialogueTimeline

    previous = os.environ.get("H3_DIRECTOR_ENABLED")
    os.environ["H3_DIRECTOR_ENABLED"] = "0"
    try:
        director = QwenDirector(ROOT)
    finally:
        if previous is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = previous

    characters = [{"name": "Eli", "character_id": "eli"}]

    def event(text, *, continues_from=False, continues_to=False):
        return {
            "speaker": "Eli",
            "text": text,
            "continues_from_previous_shot": continues_from,
            "continues_to_next_shot": continues_to,
        }

    shots = [
        {
            "shot_id": "scene_005_shot_001",
            "scene_id": "scene_005",
            "duration_seconds": 13.667,
            "characters": ["Eli"],
            "dialogue_events": [
                event("one two three four five six seven eight nine ten"),
                event("one two three four five six seven eight nine ten"),
                event("one two three four five six seven eight nine ten"),
                event("one two three four five six seven", continues_to=True),
            ],
        },
        {
            "shot_id": "scene_005_shot_002",
            "scene_id": "scene_005",
            "duration_seconds": 5.2,
            "characters": ["Eli"],
            "dialogue_events": [
                event("the sentence continues here", continues_from=True),
            ],
        },
    ]
    scenes = [{"scene_id": "scene_005", "characters": ["Eli"]}]

    source_text = [
        event_data["text"]
        for shot in shots
        for event_data in shot["dialogue_events"]
    ]

    try:
        DialogueTimeline(characters).apply_to_plan({"shots": deepcopy(shots)})
    except ValueError:
        pass
    else:
        raise AssertionError("regression fixture unexpectedly fit before Director H3 rebalancing")

    director._normalize_dialogue_h3_feasibility(scenes, shots, characters)

    final_text = [
        event_data["text"]
        for shot in shots
        for event_data in shot["dialogue_events"]
    ]
    _assert(final_text == source_text, "H3 dialogue rebalancing changed spoken text or order")

    try:
        DialogueTimeline(characters).apply_to_plan({"shots": deepcopy(shots)})
    except Exception as exc:
        raise AssertionError(f"rebalanced dialogue still violates H3 scheduling: {exc}") from exc

    _assert(
        len(shots[0]["dialogue_events"]) < 4 or len(shots[1]["dialogue_events"]) > 1,
        "H3 rebalancing did not redistribute the overfull shot",
    )


def test_dialogue_h3_feasibility_searches_the_whole_scene():
    import planner.qwen_director as qwen_director_module
    from planner.qwen_director import QwenDirector

    previous = os.environ.get("H3_DIRECTOR_ENABLED")
    os.environ["H3_DIRECTOR_ENABLED"] = "0"
    try:
        director = QwenDirector(ROOT)
    finally:
        if previous is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = previous

    characters = [{"name": "Eli", "character_id": "eli"}]

    def event(number):
        return {
            "speaker": "Eli",
            "text": f"line {number}",
            "continues_from_previous_shot": False,
            "continues_to_next_shot": False,
        }

    shots = [
        {
            "shot_id": "scene_global_shot_001",
            "scene_id": "scene_global",
            "characters": ["Eli"],
            "dialogue_events": [event(1), event(2), event(3), event(4)],
        },
        {
            "shot_id": "scene_global_shot_002",
            "scene_id": "scene_global",
            "characters": ["Eli"],
            "dialogue_events": [],
        },
        {
            "shot_id": "scene_global_shot_003",
            "scene_id": "scene_global",
            "characters": ["Eli"],
            "dialogue_events": [],
        },
    ]
    scenes = [{"scene_id": "scene_global", "characters": ["Eli"]}]

    original_fit = QwenDirector._dialogue_scene_fits_h3

    def fake_fit(scene_shots, _characters):
        counts = tuple(len(shot.get("dialogue_events", []) or []) for shot in scene_shots)
        # Force the only legal scene-global arrangement to be [2, 1, 1].
        if counts == (2, 1, 1):
            return True, ""
        return False, "synthetic H3 timing overflow"

    qwen_director_module.QwenDirector._dialogue_scene_fits_h3 = staticmethod(fake_fit)
    try:
        director._normalize_dialogue_h3_feasibility(scenes, shots, characters)
    finally:
        qwen_director_module.QwenDirector._dialogue_scene_fits_h3 = staticmethod(original_fit)

    counts = [len(shot.get("dialogue_events", []) or []) for shot in shots]
    _assert(
        counts == [2, 1, 1],
        f"scene-global feasibility search selected the wrong partition: {counts}",
    )


def test_dialogue_h3_feasibility_propagates_non_timing_errors():
    import planner.qwen_director as qwen_director_module
    from planner.qwen_director import QwenDirector

    class DefectTimeline:
        def __init__(self, *_args, **_kwargs):
            pass

        def apply_to_plan(self, _plan):
            raise ValueError("continued dialogue must use the previous shot's speaker.")

    original = qwen_director_module.DialogueTimeline
    qwen_director_module.DialogueTimeline = DefectTimeline
    try:
        try:
            QwenDirector._dialogue_scene_fits_h3([], [])
        except ValueError as exc:
            _assert(
                "continued dialogue must use the previous shot's speaker" in str(exc),
                "non-H3 scheduler errors must propagate from feasibility probes",
            )
        else:
            raise AssertionError(
                "non-H3 scheduler ValueError was swallowed by the feasibility probe"
            )
    finally:
        qwen_director_module.DialogueTimeline = original



def test_story_completion_contracts():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    sentence = "Eli crossed the abandoned station with measured steps, checked the sealed vault, and listened to the cooling machinery."
    body = " ".join([sentence] * 4)
    paragraphs = [body for _ in range(6)]
    paragraphs[0] = 'Eli entered the abandoned station and found a sealed vault. "I should have left it closed," he whispered. ' + body
    open_story = "\n\n".join(paragraphs[:-1] + [
        body + " Eli escaped after the station shook. He knew he had set something in motion that could never be undone."
    ])
    try:
        director._validate_story_completion_contract("expand_user_story", open_story)
    except RuntimeError as exc:
        _assert("unresolved future hook" in str(exc), f"wrong completion failure: {exc}")
    else:
        raise AssertionError("open-ended story ending was not rejected")

    closed_paragraphs = [body for _ in range(6)]
    closed_paragraphs[0] = 'Eli entered the abandoned station and found a sealed vault. "I should have left it closed," he whispered. ' + body
    closed_paragraphs[-1] = (
        "Eli shut the system down before dawn, sealed the journal away, and secured the station again. "
        "He finally stopped searching for the answers his father had taken to his grave. " + body
    )
    closed_story = "\n\n".join(closed_paragraphs)
    director._validate_story_completion_contract("expand_user_story", closed_story)


def test_story_normalization_preserves_scene_paragraphs():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    normalized = director._normalize_story(
        " First  paragraph.\n\n Second paragraph.\n\n\nThird   paragraph. "
    )
    _assert(
        normalized == "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        f"story normalization did not preserve paragraph topology: {normalized!r}",
    )


def test_story_topology_fallback_is_last_resort_and_prose_preserving():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    seven = "\n\n".join(
        f"Beat {index} happened with concrete action and a settled consequence."
        for index in range(1, 8)
    )
    repaired = director._coerce_story_to_six_paragraphs(seven)
    paragraphs = [part for part in repaired.split("\n\n") if part.strip()]
    _assert(len(paragraphs) == 6, f"seven-paragraph fallback did not reach six: {len(paragraphs)}")
    _assert(
        "".join(repaired.split()) == "".join(seven.split()),
        "topology fallback changed story prose tokens",
    )

    five = "\n\n".join(
        f"Beat {index} opened the door and found the same concrete evidence. The character acted."
        for index in range(1, 6)
    )
    repaired_five = director._coerce_story_to_six_paragraphs(five)
    five_paragraphs = [part for part in repaired_five.split("\n\n") if part.strip()]
    _assert(len(five_paragraphs) == 6, f"five-paragraph fallback did not reach six: {len(five_paragraphs)}")
    _assert(
        "".join(repaired_five.split()) == "".join(five.split()),
        "five-to-six topology fallback changed story prose tokens",
    )


def test_story_validation_precedes_topology_fallback():
    source = Path(ROOT, "planner", "qwen_director.py").read_text(encoding="utf-8")
    start = source.index("def _generate_story_once")
    primary = source[start:source.index("def _coerce_story_to_six_paragraphs", start)]
    _assert(
        primary.index("self._validate_story_output_contracts")
        < primary.index("self._coerce_story_to_six_paragraphs(raw)"),
        "story flow must validate raw output before any topology repair",
    )
    _assert(
        "story = self._coerce_story_to_six_paragraphs(story)" not in primary,
        "primary story flow must not silently coerce topology before validation",
    )


def test_six_story_paragraphs_become_six_production_units():
    from planner.production_planner import ProductionPlanner

    planner = ProductionPlanner(ROOT)
    story = "\n\n".join(
        f"Paragraph {index} establishes a complete narrative beat in one physical setting."
        for index in range(1, 7)
    )
    units = planner._split_story(story)
    _assert(len(units) == 6, f"six story paragraphs became {len(units)} units")
    _assert(
        [unit.text for unit in units] == story.split("\n\n"),
        "production units changed six-paragraph story order/content",
    )


def test_four_to_six_scene_functions_are_structural():
    from planner.qwen_director_scene import QwenDirectorSceneMixin

    expected = {
        4: ["setup", "catalyst", "climax", "finale"],
        5: ["setup", "catalyst", "development", "climax", "finale"],
        6: ["setup", "catalyst", "development", "midpoint", "climax", "finale"],
    }
    for count, target in expected.items():
        scenes = [{"description": "generic prose with no structural keywords"} for _ in range(count)]
        actual = [
            scene["scene_function"]
            for scene in QwenDirectorSceneMixin._annotate_scene_functions(scenes)
        ]
        _assert(actual == target, f"{count}-scene structural function mapping drifted: {actual}")


def test_story_completion_contract_rejects_wrong_structure_and_word_budget():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    sentence = "Eli moved through the station and checked the sealed vault before dawn."
    short_body = " ".join([sentence] * 8)
    five_paragraph_story = "\n\n".join([short_body] * 5)
    try:
        director._validate_story_completion_contract("expand_user_story", five_paragraph_story)
    except RuntimeError as exc:
        _assert("exactly six paragraphs" in str(exc), f"wrong structural error: {exc}")
    else:
        raise AssertionError("five-paragraph story passed the six-paragraph contract")

    six_paragraph_short_story = "\n\n".join([short_body] * 6)
    try:
        director._validate_story_completion_contract("expand_user_story", six_paragraph_short_story)
    except RuntimeError as exc:
        _assert("420 to 560 words" in str(exc), f"wrong word-budget error: {exc}")
    else:
        raise AssertionError("under-length story passed the word-budget contract")


def test_text_generation_length_reason_is_fail_closed():
    source = Path(ROOT, "planner/qwen_director_runtime.py").read_text(encoding="utf-8")
    _assert('if finish_reason == "length":' in source, "text length finish reason is not handled")
    _assert("completion limit" in source, "length failure does not identify the completion limit")


def test_critic_roster_is_bidirectional_and_patch_consumer_remains_wired():
    director_source = Path(ROOT, "planner/qwen_director.py").read_text(encoding="utf-8")
    orchestrator_source = Path(ROOT, "pipeline/production_orchestrator.py").read_text(encoding="utf-8")
    _assert("enforce the roster in both directions" in director_source, "critic roster check is not bidirectional")
    _assert('critique.get("shot_patches", [])' in orchestrator_source, "critic patch consumer was removed despite being wired")


def test_dialogue_source_occurrence_budget():
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    director._recovery_events = []
    director._qwen_telemetry = {"deterministic_recoveries": 0}
    characters = [{"name": "Eli", "identity_type": "named_character", "semantic_aliases": []}]
    scenes = [{"scene_id": "scene_001", "characters": ["Eli"]}]

    def run(story, texts):
        shots = []
        for index, text in enumerate(texts, start=1):
            shots.append({
                "shot_id": f"scene_001_shot_{index:03d}",
                "scene_id": "scene_001",
                "characters": ["Eli"],
                "dialogue_events": [{
                    "speaker": "Eli",
                    "text": text,
                    "continues_from_previous_shot": False,
                    "continues_to_next_shot": False,
                }],
            })
        director._normalize_dialogue_speakers(story, scenes, shots, characters)
        return shots

    duplicate = run('Eli said, "Stay here."', ["Stay here.", "Stay here."])
    _assert(len(duplicate[0]["dialogue_events"]) == 1, "first source line was lost")
    _assert(not duplicate[1]["dialogue_events"], "same source line was duplicated")

    split = run('Eli said, "We need to leave before dawn."', ["We need to leave", "before dawn."])
    _assert(all(len(shot["dialogue_events"]) == 1 for shot in split), "split source dialogue was lost")

    repeated = run(
        'Eli said, "Stay here." Later he said, "Stay here."',
        ["Stay here.", "Stay here."],
    )
    _assert(all(len(shot["dialogue_events"]) == 1 for shot in repeated), "distinct repeated source lines were collapsed")


def test_source_contracts():
    from pathlib import Path
    orchestrator = (ROOT / "pipeline/production_orchestrator.py").read_text(encoding="utf-8")
    runner = (ROOT / "execution/production_runner.py").read_text(encoding="utf-8")
    _assert("require_context_ir_results=False" in orchestrator, "planning manifest must be non-strict before rendering")
    _assert('"h3_context_ir"' in runner and '"h3_effective_prompt"' in runner, "completed shot records must persist local Context-IR provenance")
    _assert("production_plan[\"final_video\"] = str(final_video)" in runner, "final video must be bound before final manifest")
    _assert(runner.find('require_context_ir_results=True') > runner.find('production_plan["final_video"]'), "strict final manifest must be written after assembly")


def test_sentence_initial_common_words_never_skip_semantic_roster():
    from planner.production_planner import ProductionPlanner

    story = (
        'Elias trudged on. Dust swirled in the light. '
        '\u201cAccess granted,\u201d the terminal said. Anika Marlowe stared at Elias and waited.'
    )
    for word in ("Dust", "Access"):
        assert not ProductionPlanner._high_confidence_deterministic_character(story, word), word
    assert ProductionPlanner._high_confidence_deterministic_character(story, "Elias")
    assert ProductionPlanner._high_confidence_deterministic_character(
        "Eli, a scientist, enters the station.", "Eli"
    )


def test_story_is_single_call_fixed_seed_no_floor_and_all_defects():
    """Regression: 5 paragraphs + short + invented 'Eli's brother' must fail closed after ONE call."""
    from planner.qwen_director import QwenDirector
    from planner.production_planner import ProductionPlanner

    source_text = Path(ROOT, "planner", "qwen_director.py").read_text(encoding="utf-8")
    prompts = Path(ROOT, "planner", "qwen_director_prompts.py").read_text(encoding="utf-8")
    body = source_text.split("def _generate_story_once")[1].split("def _coerce_story_to_six_paragraphs")[0]
    _assert("H3_STORY_MAX_ATTEMPTS" not in source_text, "no configurable story retry may exist")
    _assert("_retry" not in body and "for attempt" not in body, "no retry loop or retry call name")
    _assert("PREVIOUS DRAFT FAILED" not in prompts, "no retry feedback prompt")
    _assert("story_min_output_tokens = 0" in body and "1450" not in body, "no min_tokens floor: it forces filler after the story ends")
    _assert("minimum_output_tokens=story_min_output_tokens" in body, "floor must reach _chat_text")
    _assert("seed=DIRECTOR_VLLM_SEED," in body, "story seed must be fixed")

    director = QwenDirector.__new__(QwenDirector)
    planner = ProductionPlanner(ROOT)
    director._planner = lambda: planner
    director._qwen_telemetry = {"retries": 0}
    calls = []
    bad = "\n\n".join(
        ["Eli enters the abandoned station and finds a sealed vault. He follows his brother's coordinates and tests the vault door."] * 5
    )

    def fake_chat(system, user, **kwargs):
        calls.append((kwargs["call_name"], kwargs["minimum_output_tokens"], kwargs["seed"]))
        return bad

    director._chat_text = fake_chat
    source = "Eli enters the abandoned station and finds a sealed vault."
    try:
        director._generate_story_once(
            "expand_user_story", source, "s",
            director._story_text_user("expand_user_story", source, ["Eli"]),
            temperature=0.6, top_p=0.95, source_character_names=["Eli"],
        )
        raise AssertionError("bad draft must fail closed")
    except RuntimeError as exc:
        text = str(exc).lower()
        _assert(
            "exactly six paragraphs" in text and "420 to 560 words" in text and "brother" in text,
            f"paragraph-count, word-count and invented-relative defects must be reported together: {exc}",
        )
    _assert(len(calls) == 1, f"exactly ONE Qwen story call allowed: {calls}")
    _assert(calls[0][1] == 0, f"story call must not set a min_tokens floor: {calls}")
    _assert(director._qwen_telemetry["retries"] == 0, "no retries")

    ai_premise = "A polar systems engineer reaches an abandoned Arctic station during a violent storm and discovers a sealed underground vault."
    ai_story = "Elara Voss kicked snow from her boots inside the abandoned station while the storm howled. The sealed vault waited below."
    coverage, _ = director._global_premise_coverage(ai_premise, ai_story)
    _assert(coverage >= 0.5, f"permissive premise gate rejected a faithful AI story: {coverage}")


def test_non_human_speech_tags_never_become_speakers():
    from planner.qwen_director import QwenDirector

    for tag in ("Static answered.", "Silence fell.", "The radio answered.", "Alarm called.", "Echo replied."):
        speakers = QwenDirector._speech_tag_speakers('"Hello? Anyone?"', tag)
        _assert(speakers == set(), f"{tag!r} produced speaker {speakers}")
    _assert(
        QwenDirector._speech_tag_speakers('"Run."', "Eli said.") == {"eli"},
        "a real speech tag must still resolve its speaker",
    )
    _assert(
        QwenDirector._speech_tag_speakers('"Run."', "Elara Voss whispered.") == {"elara voss"},
        "a full-name speech tag must still resolve its speaker",
    )


def test_runaway_tail_after_a_complete_story_is_trimmed_and_emoji_stripped():
    """Regression from a live run: valid 6 paragraphs / 469 words, then 5 padding paragraphs of
    repetition ending in an emoji (caused by a min_tokens floor)."""
    from planner.qwen_director import QwenDirector

    director = QwenDirector.__new__(QwenDirector)
    body = (
        "Eli pried the sealed vault hatch while the damp concrete of the abandoned station dripped onto his collar and "
        "the crowbar bit into the rusted seam until the metal finally shrieked and moved under his boots again today. "
        "He wiped his sleeve across his brow, shouldered the hatch, and held the flashlight steady on the dark gap "
        "while water ran along the tiles, pooled against the pillar, and carried a thin line of rust toward the tracks."
    )
    paragraphs = [body] * 6
    tail = ["The flask stayed with him, its dents a quiet record of the night. \U0001F311."] * 5
    raw = "\n\n".join(paragraphs + tail)
    trimmed = director._trim_runaway_tail(raw)
    parts = [p for p in trimmed.split("\n\n") if p.strip()]
    _assert(len(parts) == 6, f"runaway tail must be cut back to six paragraphs: {len(parts)}")
    _assert("\U0001F311" not in director._normalize_story("Done \U0001F311."), "emoji must be stripped")
    short = "\n\n".join(["Too short."] * 8)
    _assert(director._trim_runaway_tail(short) == short, "an out-of-range head must never be trimmed into validity")


def test_attributed_dialogue_and_signage_words():
    from planner.qwen_director import QwenDirector
    from planner.production_planner import ProductionPlanner

    _assert(QwenDirector._story_has_attributed_dialogue('"Stop," Mara said. He froze.'), "tagged line is dialogue")
    _assert(QwenDirector._story_has_attributed_dialogue('Mara said, "Stop right there."'), "leading tag is dialogue")
    _assert(
        not QwenDirector._story_has_attributed_dialogue("Kovac had vanished during the thaw, his last transmission citing \u201cunstable core readings.\u201d Her hand trembled as she opened the container."),
        "a quoted phrase inside narration is not attributed dialogue",
    )
    planner = ProductionPlanner(ROOT)
    names = planner.detect_character_descriptors("Lena Voss read the pad. Access granted to L. Voss. Lena Voss stepped back.")
    _assert("Access" not in names, f"signage word became a character: {names}")


def test_expand_cast_is_not_count_restricted():
    from planner.qwen_director import QwenDirector
    director = QwenDirector.__new__(QwenDirector)
    director._planner = lambda: _planner()
    source = "Eli enters the station and finds a sealed vault."
    generated = (
        "Eli entered the station and met Mara Venn by the vault. "
        "His father appeared in the control room and warned them to leave. "
        "Mara ignored him and Eli shut the vault before dawn."
    )
    director._validate_expand_story_cast(source, generated, source_character_names=["Eli"])


def test_mention_only_people_never_enter_the_roster():
    """Regression from live runs: 'Kovac' (a name on a container + 'had vanished'), 'Access' (on-screen
    'Access granted') and 'Renn' (a radio message) became characters because a regex produced the roster
    and the Qwen extraction call was skipped."""
    from planner.production_planner import ProductionPlanner

    planner = ProductionPlanner(ROOT)
    story = (
        "Lena Voss crouched beside the vault and pried at the hatch. A data pad blinked: Access granted to L. Voss, 2018. "
        "Her fingers brushed a name etched into the metal: Kovac. She knew that name\u2014Kovac had vanished during the 2017 thaw, "
        "his last transmission citing unstable readings. Lena Voss slammed the switch and walked out, "
        "leaving Kovac\u2019s secrets beneath the ice. A message was from Dr. Renn, the station biologist."
    )
    _assert(planner._is_mention_only_identity(story, "Kovac"), "Kovac is evidence, not a character")
    _assert(planner._is_mention_only_identity(story, "Access"), "UI text is not a character")
    _assert(planner._is_mention_only_identity(story, "Renn"), "a message sender who never appears is not a character")
    _assert(not planner._is_mention_only_identity(story, "Lena Voss"), "the protagonist must stay")
    _assert(
        planner._drop_mention_only_identities(story, ["Lena Voss", "Kovac", "Access", "Renn"]) == ["Lena Voss"],
        "roster filter must keep only people who are actually present",
    )
    present = 'Mira followed Arun into the hall. "Wait," Arun said. Kovac had vanished years ago, but Kovac turned and spoke: Kovac said, "Run."'
    _assert(not planner._is_mention_only_identity(present, "Kovac"), "a name that later speaks or acts is a real character")
    _assert(
        planner._drop_mention_only_identities("Kovac had vanished.", ["Kovac"]) == ["Kovac"],
        "the filter must never empty a roster",
    )
    characters = planner.create_characters(story)
    names = [getattr(c, "name", "") for c in characters]
    _assert(names == ["Lena Voss"], f"create_characters must not re-add mention-only names: {names}")


def test_craft_diagnostics_are_logged_not_fatal():
    from planner.qwen_director import QwenDirector
    from planner.production_planner import ProductionPlanner

    director = QwenDirector.__new__(QwenDirector)
    planner = ProductionPlanner(ROOT)
    director._planner = lambda: planner
    story = "\n\n".join(
        ["Lena Voss opened the hatch."] * 5
        + ["The system was alive. Kovac had vanished long ago, and a name etched into the wall said Kovac."]
    )
    issues = director._story_craft_issues(story)
    joined = " | ".join(issues)
    _assert("paragraph" in joined and "reveal" in joined, f"short paragraph and generic reveal must be reported: {issues}")


def test_sound_nouns_never_become_characters():
    planner = _planner()
    names = planner.detect_character_descriptors(
        'Elara called out, "Anyone?" Static answered. Silence fell over the corridor. Elara Voss stepped back.'
    )
    _assert("Static" not in names and "Silence" not in names, f"sound noun became a character: {names}")
    _assert("Elara Voss" in names, f"real character lost: {names}")



def test_story_prompt_has_no_numeric_cast_contract():
    source = (ROOT / "planner/qwen_director_prompts.py").read_text(encoding="utf-8")
    _assert("Do not force one character, two characters, or any fixed cast size." in source,
            "AI Story prompt must leave character count to Qwen")
    _assert("not a cast limit" in source,
            "Expand Story source anchors must not become a cast whitelist")
    _assert("exactly ONE unnamed counterpart" not in source,
            "Expand Story must not force exactly one new character")

def main():
    tests = [
        test_deterministic_character_regressions,
        test_story_prompt_restores_successful_compact_narrative_contract,
        test_expand_source_fallback_removed,
        test_semantic_named_surface_safety_boundary,
        test_logged_story_rosters_are_not_poisoned_by_prose_surfaces,
        test_qwen_character_roster_is_authoritative_and_filters_noncharacters,
        test_malformed_qwen_character_roster_fails_closed_without_adjudication,
        test_qwen_character_roster_can_introduce_new_characters_without_source_whitelist,
        test_qwen_negative_character_decision_is_respected,
        test_expand_qwen_roster_preserves_source_and_allows_new_characters,
        test_expand_qwen_relational_character_is_accepted_without_adjudication,
        test_relational_character_survives_qwen_roster_and_resolves_dialogue,
        test_sanitizer_identity_contract,
        test_qwen_cache_generation_contract,
        test_disabled_director_path,
        test_story_token_budget_contract,
        test_story_salvage_and_quality_gate,
        test_story_sampling_guards_are_story_only_and_warmup_matches,
        test_story_thinking_arguments_reach_vllm_without_a_retry,
        test_parallel_stage_copy_is_byte_exact,
        test_quoted_dialogue_speaker_comes_from_speech_tag,
        test_creative_named_only_story_still_calls_qwen_character_extractor,
        test_story_prompt_has_no_numeric_cast_contract,
        test_context_ir_capture_root,
        test_checkpoint_digest_excludes_runtime_outputs,
        test_job_state_clears_stale_completion,
        test_unresolved_explicit_dialogue_fails_closed,
        test_wrong_shot_speaker_is_corrected_by_speech_tag,
        test_canonical_dialogue_speaker_is_rebound_into_shot_and_scene,
        test_dialogue_speaker_is_bound_even_when_shot_starts_empty,
        test_downstream_production_preserves_dialogue_multiset,
        test_dialogue_scene_boundary_clears_stale_continuation,
        test_shot_prompt_requires_same_speaker_continuation_rule,
        test_dialogue_continuation_requires_same_canonical_speaker,
        test_dialogue_h3_feasibility_rebalances_generated_shots,
        test_dialogue_h3_feasibility_searches_the_whole_scene,
        test_dialogue_h3_feasibility_propagates_non_timing_errors,
        test_story_completion_contracts,
        test_story_normalization_preserves_scene_paragraphs,
        test_story_topology_fallback_is_last_resort_and_prose_preserving,
        test_story_validation_precedes_topology_fallback,
        test_six_story_paragraphs_become_six_production_units,
        test_four_to_six_scene_functions_are_structural,
        test_story_completion_contract_rejects_wrong_structure_and_word_budget,
        test_text_generation_length_reason_is_fail_closed,
        test_critic_roster_is_bidirectional_and_patch_consumer_remains_wired,
        test_dialogue_source_occurrence_budget,
        test_source_contracts,
        test_sentence_initial_common_words_never_skip_semantic_roster,
        test_sound_nouns_never_become_characters,
        test_mention_only_people_never_enter_the_roster,
        test_craft_diagnostics_are_logged_not_fatal,
        test_expand_cast_is_not_count_restricted,
        test_runaway_tail_after_a_complete_story_is_trimmed_and_emoji_stripped,
        test_attributed_dialogue_and_signage_words,
        test_non_human_speech_tags_never_become_speakers,
        test_story_is_single_call_fixed_seed_no_floor_and_all_defects,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("Director and integration validation PASSED.")


if __name__ == "__main__":
    main()

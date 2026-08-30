from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep the logic validator model-free when run directly.
# QwenDirector otherwise performs model discovery during construction.
os.environ.setdefault(
    "H3_DIRECTOR_ENABLED",
    "0",
)

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from planner.production_planner import (
    ProductionPlanner,
)
from planner.qwen_director import (
    QwenDirector,
)


def check(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise RuntimeError(
            message
        )


def test_story_modes() -> None:

    director = QwenDirector(
        ROOT
    )

    source = (
        "A lone man walks through an endless abyss "
        "at the edge of a collapsing world."
    )

    # AI STORY
    ai_result = (
        "A lone man walks through an endless abyss "
        "at the edge of a collapsing world. "
        "He discovers a hidden signal beneath the ruins "
        "and realizes that the collapse is leading him "
        "toward a final choice."
    )

    director._validate_mode_output(
        "ai_story",
        source,
        ai_result,
    )

    try:

        director._validate_mode_output(
            "ai_story",
            source,
            source,
        )

    except RuntimeError:
        pass

    else:

        raise RuntimeError(
            "AI Story accepted an unchanged premise."
        )

    # EXPAND
    expanded = (
        "A lone man walks through an endless abyss "
        "at the edge of a collapsing world. "
        "He carries the memories of the life he lost "
        "before the collapse and slowly realizes that "
        "the destruction is not random. Each step brings "
        "him closer to the source of the catastrophe, "
        "forcing him to decide whether survival is still "
        "possible."
    )

    director._validate_mode_output(
        "expand_user_story",
        source,
        expanded,
    )

    try:

        director._validate_mode_output(
            "expand_user_story",
            source,
            source,
        )

    except RuntimeError:
        pass

    else:

        raise RuntimeError(
            "Expand Story accepted unchanged input."
        )

    # PRESERVE
    director._validate_mode_output(
        "preserve_user_story",
        source,
        source,
    )

    try:

        director._validate_mode_output(
            "preserve_user_story",
            source,
            source + " Extra event.",
        )

    except RuntimeError:
        pass

    else:

        raise RuntimeError(
            "Preserve Story accepted modified text."
        )


def test_expand_preservation_gates() -> None:
    director = QwenDirector(
        ROOT
    )

    source = (
        "A man named Eli enters the abandoned station. "
        "A woman named Sara gives Eli a map to the underground vault. "
        "The vault contains 7 sealed chambers."
    )

    valid = (
        "Eli enters the abandoned station and searches the ruined platform. "
        "Sara gives Eli a map to the underground vault, explaining why she "
        "believes it matters. Eli follows the map and discovers 7 sealed "
        "chambers, realizing the station hides a much larger secret."
    )

    director._validate_mode_output(
        "expand_user_story",
        source,
        valid,
    )

    missing_name = valid.replace("Sara", "Mara")
    try:
        director._validate_mode_output(
            "expand_user_story",
            source,
            missing_name,
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "Expand Story accepted output that dropped a named source anchor."
        )

    unrelated = (
        "A pilot crosses a desert, discovers a hidden temple, and escapes "
        "before sunset. The journey ends with a mysterious transmission."
    )
    try:
        director._validate_mode_output(
            "expand_user_story",
            source,
            unrelated,
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "Expand Story accepted output with insufficient source overlap."
        )


def test_shot_batch_contract() -> None:
    director = QwenDirector(
        ROOT
    )

    prompt = director._shot_director_batch_system()

    check(
        "Create exactly TWO production-ready shots for EACH supplied scene." in prompt,
        "Batch shot prompt does not enforce two shots per scene.",
    )

    check(
        "Do not add characters" in prompt or "Do not create new characters" in prompt,
        "Batch prompt lost the explicit character restriction.",
    )

    normalized = director._normalize_batch_shot_response(
        {
            "scene_shots": [
                {
                    "scene_id": "scene_001",
                    "shots": [
                        {"shot_id": "a"},
                        {"shot_id": "b"},
                    ],
                },
                {
                    "scene_id": "scene_002",
                    "shots": [
                        {"shot_id": "c"},
                        {"shot_id": "d"},
                    ],
                },
            ]
        }
    )

    check(
        set(normalized) == {"scene_001", "scene_002"},
        "Batch response normalization lost a scene.",
    )

    check(
        all(len(values) == 2 for values in normalized.values()),
        "Batch response normalization did not preserve both shots.",
    )


def test_text_generation_disables_thinking_by_default() -> None:
    director = QwenDirector(
        ROOT
    )

    import inspect

    parameter = inspect.signature(
        director._chat_text
    ).parameters["disable_thinking"]

    check(
        parameter.default is True,
        "Narrative text generation still enables Qwen reasoning by default.",
    )


def test_character_sanitization() -> None:

    director = QwenDirector(
        ROOT
    )

    values = (
        director._sanitize_characters(
            [
                {
                    "name": "Elias",
                    "role": "protagonist",
                },
                {
                    "name": "The Vortex",
                    "role": "entity",
                },
                {
                    "name": "Visual",
                    "role": "metadata",
                },
                {
                    "name": "Camera",
                    "role": "metadata",
                },
                {
                    "name": "Elias",
                    "role": "duplicate",
                },
            ]
        )
    )

    names = [
        value["name"].lower()
        for value
        in values
    ]

    check(
        names.count("elias") == 1,
        "Character sanitizer failed to remove duplicate Elias.",
    )

    check(
        "visual" not in names,
        "Character sanitizer accepted metadata word Visual.",
    )

    check(
        "camera" not in names,
        "Character sanitizer accepted metadata word Camera.",
    )


def test_scene_id_sanitization_before_batching() -> None:
    director = QwenDirector(ROOT)
    scenes = director._sanitize_scenes(
        [
            {"scene_id": "scene_001", "description": "First event."},
            {"scene_id": "scene_001", "description": "Second event."},
        ],
        set(),
    )
    ids = [str(scene.get("scene_id", "")) for scene in scenes]
    check(ids == ["scene_001", "scene_001_2"], "Duplicate scene IDs must be repaired before batching/resume.")


def test_shot_id_normalization() -> None:

    director = QwenDirector(
        ROOT
    )

    scenes = [
        {
            "scene_id": "scene_001",
            "title": "Beginning",
        },
        {
            "scene_id": "scene_002",
            "title": "Escalation",
        },
    ]

    shots = [
        {
            "shot_id": "shot_001",
            "scene_id": "scene_001",
        },
        {
            "shot_id": "shot_001",
            "scene_id": "scene_001",
        },
        {
            "shot_id": "shot_001",
            "scene_id": "scene_002",
        },
    ]

    director._normalize_ids(
        scenes,
        shots,
    )

    ids = [
        shot["shot_id"]
        for shot
        in shots
    ]

    check(
        len(ids) == len(set(ids)),
        "Shot IDs are not globally unique.",
    )

    for value in ids:

        check(
            value.strip(),
            "Shot ID is empty.",
        )


def test_character_descriptor_deduplication() -> None:

    planner = ProductionPlanner(
        ROOT
    )

    story = (
        "A young man named Eli explores an "
        "abandoned city after a war. "
        "He meets a woman named Sara near "
        "the ruined railway station."
    )

    values = (
        planner.detect_character_descriptors(
            story
        )
    )

    names = {
        str(value).lower()
        for value
        in values
    }

    check(
        "eli" in names,
        "Named character Eli was not detected.",
    )

    check(
        "sara" in names,
        "Named character Sara was not detected.",
    )

    check(
        not (
            "man" in names
            and "eli" in names
        ),
        "Generic 'man' was duplicated alongside named Eli.",
    )

    check(
        not (
            "woman" in names
            and "sara" in names
        ),
        "Generic 'woman' was duplicated alongside named Sara.",
    )


def test_single_paragraph_segmentation() -> None:

    planner = ProductionPlanner(
        ROOT
    )

    story = (
        "Eli enters the abandoned city. "
        "He finds the ruined station and discovers "
        "a strange signal. "
        "The signal leads him underground, where "
        "the city begins to collapse around him."
    )

    units = planner._split_story(
        story
    )

    check(
        len(units) > 1,
        "A multi-event single paragraph remained a single scene.",
    )


def test_director_prompt_contract() -> None:

    director = QwenDirector(
        ROOT
    )

    ai = director._story_director_system(
        "ai_story"
    )

    expand = director._story_director_system(
        "expand_user_story"
    )

    preserve = director._story_director_system(
        "preserve_user_story"
    )

    shots = director._shot_director_system()

    check(
        "AI STORY MODE" in ai,
        "AI Story prompt is missing its mode contract.",
    )

    check(
        "EXPAND STORY MODE" in expand,
        "Expand Story prompt is missing its mode contract.",
    )

    check(
        "PRESERVE STORY MODE" in preserve,
        "Preserve Story prompt is missing its mode contract.",
    )

    check(
        '"characters"' in ai,
        "AI Story prompt does not request characters.",
    )

    check(
        '"scenes"' in ai,
        "AI Story prompt does not request scenes.",
    )

    check(
        '"shots"' not in ai,
        "Story pass should not request shots.",
    )

    check(
        "visual_language" in ai,
        "Story prompt does not request the visual-language bible.",
    )

    check(
        "time_of_day" in ai,
        "Story prompt does not request time_of_day.",
    )

    check(
        "environment_details" in ai,
        "Story prompt does not request environment details.",
    )

    check(
        "color_temperature" in ai,
        "Story prompt does not request color temperature.",
    )

    check(
        "lens_and_depth_of_field" in shots,
        "Shot prompt does not request lens/depth-of-field direction.",
    )

    check(
        "composition_notes" in shots,
        "Shot prompt does not request composition direction.",
    )

    check(
        "SHOT / FRAMING VOCABULARY" in shots,
        "Shot prompt is missing framing vocabulary.",
    )

    check(
        "CAMERA MOVEMENT VOCABULARY" in shots,
        "Shot prompt is missing camera-movement vocabulary.",
    )

    check(
        "LIGHTING VOCABULARY" in shots,
        "Shot prompt is missing lighting vocabulary.",
    )

    check(
        "Do not create new characters" in shots,
        "Shot director does not protect character identity.",
    )


def test_shot_sampling_contract() -> None:

    director = QwenDirector(
        ROOT
    )

    temperature, top_p = (
        director._shot_sampling()
    )

    check(
        temperature == 0.68,
        "Shot temperature is not 0.68.",
    )

    check(
        top_p == 0.92,
        "Shot top_p is not 0.92.",
    )


def test_visual_schema_sanitization() -> None:

    director = QwenDirector(
        ROOT
    )

    visual_language = (
        director._sanitize_visual_language(
            {
                "genre_tone": "dark cinematic sci-fi",
                "color_palette": "charcoal, amber, cold blue",
                "lighting_philosophy": "low-key motivated practical light",
                "camera_philosophy": "deliberate movement with deep spatial compositions",
                "pacing": "slow build with sharp escalation",
                "unexpected": "ignored",
            }
        )
    )

    check(
        set(visual_language) == {
            "genre_tone",
            "color_palette",
            "lighting_philosophy",
            "camera_philosophy",
            "pacing",
        },
        "Visual-language sanitizer returned unexpected fields.",
    )


def test_shot_schema_cardinality_is_grammar_constrained() -> None:
    # Grammar-constrained (JSON Schema) decoding is already the
    # project's chosen speed/correctness strategy for JSON calls.
    # This locks in that the shot-count contracts are enforced at the
    # schema level (so the model literally cannot sample an invalid
    # shot count), not left to be caught only after generation by the
    # retry/recovery machinery.
    normal = QwenDirector._shot_json_schema()
    check(
        normal["properties"]["shots"]["minItems"] == 2
        and normal["properties"]["shots"]["maxItems"] == 2,
        "Normal/retry shot schema must constrain to exactly "
        "SHOTS_PER_SCENE shots.",
    )

    recovery = QwenDirector._shot_json_schema(
        min_items=1,
        max_items=1,
    )
    check(
        recovery["properties"]["shots"]["minItems"] == 1
        and recovery["properties"]["shots"]["maxItems"] == 1,
        "Recovery shot schema must constrain to exactly one shot.",
    )

    batch = QwenDirector._shot_batch_json_schema(
        scene_count=2,
    )
    check(
        batch["properties"]["scene_shots"]["minItems"] == 2
        and batch["properties"]["scene_shots"]["maxItems"] == 2,
        "Batch schema must constrain scene_shots to the actual "
        "batch size.",
    )
    check(
        batch["properties"]["scene_shots"]["items"]["properties"][
            "shots"
        ]["minItems"]
        == QwenDirector.SHOTS_PER_SCENE
        and batch["properties"]["scene_shots"]["items"]["properties"][
            "shots"
        ]["maxItems"]
        == QwenDirector.SHOTS_PER_SCENE,
        "Batch schema must constrain each scene's shots to "
        "SHOTS_PER_SCENE.",
    )


def test_shot_sanitization_cinematography_fields() -> None:

    director = QwenDirector(
        ROOT
    )

    values = director._sanitize_shots(
        [
            {
                "shot_id": "shot_001",
                "scene_id": "scene_001",
                "characters": [],
                "camera_shot": "close-up",
                "camera_movement": "slow push-in",
                "lens_and_depth_of_field": "telephoto compression with shallow depth of field",
                "composition_notes": "rule of thirds with foreground framing",
                "lighting": "cool moonlight",
                "color_temperature": "cool 4300K",
                "mood": "tense and isolated",
                "visual_prompt": "A lone figure stands among ruined stone structures under cool moonlight.",
            }
        ],
        {
            "scene_id": "scene_001",
            "location": "ruins",
        },
        set(),
    )

    check(
        len(values) == 1,
        "Shot sanitizer rejected a valid shot.",
    )

    shot = values[0]

    check(
        shot["lens_and_depth_of_field"].startswith("telephoto"),
        "Shot lens/DOF field was not preserved.",
    )

    check(
        "rule of thirds" in shot["composition_notes"],
        "Shot composition field was not preserved.",
    )

    check(
        shot["color_temperature"] == "cool 4300K",
        "Shot color temperature was not preserved.",
    )




def _sample_scene(scene_id: str, characters=None, order: int = 1) -> dict:
    return {
        "scene_id": scene_id,
        "title": f"Beat {scene_id}",
        "order": order,
        "location": "ruined city",
        "time_of_day": "night",
        "weather": "rain",
        "atmosphere": "wet neon streets",
        "description": f"A real narrative event unfolds in {scene_id}.",
        "mood": "tense",
        "lighting": "cool neon",
        "color_temperature": "cool 4300K",
        "environment_details": ["ruined buildings"],
        "key_props": ["signal device"],
        "characters": list(characters or []),
        "scene_objective": "Advance the story.",
        "continuity_notes": "",
        "story_summary": f"Summary {scene_id}",
        "shot_ids": [],
    }


def _sample_shot(scene_id: str, ordinal: int, characters=None) -> dict:
    return {
        "shot_id": f"{scene_id}_shot_{ordinal}",
        "scene_id": scene_id,
        "duration_seconds": 5.2,
        "characters": list(characters or []),
        "location": "ruined city",
        "action": f"Action beat {ordinal}.",
        "camera_shot": "wide" if ordinal == 1 else "close-up",
        "camera_movement": "slow pan" if ordinal == 1 else "push-in",
        "lens_and_depth_of_field": (
            "normal perspective with deep focus"
            if ordinal == 1
            else "telephoto compression with shallow depth of field"
        ),
        "composition_notes": (
            "leading lines and layered depth"
            if ordinal == 1
            else "subject isolation with foreground framing"
        ),
        "lighting": "cool neon",
        "color_temperature": "cool 4300K",
        "mood": "tense",
        "visual_prompt": "A filmable cinematic shot.",
        "speaking_characters": [],
        "speech_text": "",
    }


def test_canonical_roster_not_overwritten_by_qwen() -> None:
    # P0 regression guard: enrich_plan() must never let Qwen's creative
    # output replace the deterministic canonical character roster or
    # scene topology (scene_id / order / characters / shot_ids). This
    # locks in the fix described in the V2.1 architecture review.
    director = QwenDirector(ROOT)

    base_plan = {
        "story": "Elias walked into the ruined city looking for Mara.",
        "story_mode": "preserve_user_story",
        "characters": [
            {
                "character_id": "char_elias",
                "name": "Elias",
                "role": "protagonist",
                "description": "",
                "personality": "",
            },
            {
                "character_id": "char_mara",
                "name": "Mara",
                "role": "supporting",
                "description": "",
                "personality": "",
            },
        ],
        "scenes": [
            _sample_scene("scene_001", ["Elias", "Mara"], 1),
        ],
        "shots": [],
        "visual_language": {},
    }

    # Simulate a Qwen response that tries to invent an entirely
    # different roster and scene topology -- this must be rejected,
    # not merged in, regardless of what the model returns.
    def fake_generate(
        self,
        *,
        mode,
        user_input,
        base_plan,
        checkpoint_session_id=None,
        resume_state=None,
    ):
        return {
            "enabled": True,
            "plan": {
                "story": base_plan["story"],
                "director_notes": "creative notes",
                "visual_language": {
                    "genre_tone": "noir",
                    "color_palette": "desaturated blues",
                    "lighting_philosophy": "hard chiaroscuro",
                    "camera_philosophy": "handheld",
                    "pacing": "slow",
                },
                "characters": [
                    {
                        "character_id": "char_invented",
                        "name": "Invented Stranger",
                        "role": "protagonist",
                        "description": "",
                        "personality": "",
                    }
                ],
                "scenes": [
                    {
                        "scene_id": "scene_001",
                        "order": 99,
                        "characters": ["Invented Stranger"],
                        "shot_ids": ["fake_shot"],
                        "mood": "eerie",
                        "lighting": "moonlight",
                    }
                ],
                "shots": [
                    _sample_shot("scene_001", 1, ["Invented Stranger"]),
                    _sample_shot("scene_001", 2, ["Invented Stranger"]),
                ],
            },
        }

    original_generate = QwenDirector.generate
    QwenDirector.generate = fake_generate
    try:
        merged = director.enrich_plan(
            mode="preserve_user_story",
            user_input=base_plan["story"],
            base_plan=base_plan,
        )
    finally:
        QwenDirector.generate = original_generate

    check(
        merged["characters"] == base_plan["characters"],
        "enrich_plan() let Qwen overwrite the canonical character roster.",
    )

    merged_scene = merged["scenes"][0]

    check(
        merged_scene["scene_id"] == "scene_001"
        and merged_scene["order"] == 1
        and merged_scene["characters"] == ["Elias", "Mara"],
        "enrich_plan() let Qwen overwrite protected scene topology "
        "(scene_id/order/characters).",
    )

    check(
        merged_scene.get("mood") == "eerie"
        and merged_scene.get("lighting") == "moonlight",
        "enrich_plan() failed to apply Qwen's non-structural creative "
        "enrichment (mood/lighting) onto the canonical scene.",
    )

    check(
        merged["visual_language"].get("genre_tone") == "noir",
        "enrich_plan() failed to merge the visual_language bible.",
    )


def test_entity_resolver_shot_rebinding() -> None:
    # P0 regression guard: shot/scene character references must resolve
    # through EntityResolver (aliases, honorifics) rather than exact-name
    # matching only, and an explicit-but-unresolved character reference
    # must never silently fall back to "all scene characters" (that
    # would invent presence the model didn't actually establish).
    from pipeline.production_orchestrator import ProductionOrchestrator
    from schemas.character import Character

    characters = [
        Character(
            character_id="char_elias",
            name="Elias",
            role="protagonist",
            description="",
            personality="",
        ),
        Character(
            character_id="char_mara",
            name="Mara",
            role="supporting",
            description="",
            personality="",
        ),
    ]

    plan = {
        "scenes": [
            _sample_scene("scene_001", ["Elias", "Mara"], 1),
        ],
        "shots": [
            # Honorific + case variation should resolve to Elias.
            {
                **_sample_shot("scene_001", 1, ["Dr. elias"]),
            },
            # No character field at all -- must inherit scene characters.
            {
                k: v
                for k, v in _sample_shot("scene_001", 2, []).items()
                if k != "characters"
            },
            # Explicit reference to someone not in the roster -- must
            # resolve to nobody, NOT fall back to the full scene cast.
            {
                **_sample_shot("scene_001", 3, ["Totally Unknown Person"]),
            },
        ],
    }

    orchestrator = ProductionOrchestrator.__new__(
        ProductionOrchestrator
    )
    ProductionOrchestrator._rebind_shots(
        orchestrator,
        plan,
        characters,
    )

    shots = plan["shots"]

    check(
        shots[0]["characters"] == ["Elias"],
        "EntityResolver honorific/case normalization did not resolve "
        "'Dr. elias' to the canonical character 'Elias'.",
    )

    check(
        set(shots[1]["characters"]) == {"Elias", "Mara"},
        "A shot with no character field at all should inherit the "
        "scene's full character list.",
    )

    check(
        shots[2]["characters"] == [],
        "An explicit but unresolved character reference must resolve "
        "to no characters, not silently fall back to the full scene "
        "cast (that would invent presence the model never established).",
    )


def test_scene_budget_contract_and_fallback() -> None:
    director = QwenDirector(ROOT)
    scenes = [
        _sample_scene(f"scene_{index:03d}", ["Elias"], index)
        for index in range(1, 9)
    ]

    original_chat = director._chat_json
    try:
        def fail_compression(*args, **kwargs):
            raise RuntimeError("forced validation fallback")

        director._chat_json = fail_compression
        reduced = director._compress_scenes_to_budget(
            "ai_story",
            "A story about Elias discovering a signal before the city collapses.",
            [{"name": "Elias", "role": "protagonist"}],
            scenes,
            {"elias"},
        )
    finally:
        director._chat_json = original_chat

    check(len(reduced) == director.MAX_SCENES, "Scene budget fallback did not enforce MAX_SCENES.")
    check(reduced[0]["scene_id"] == "scene_001", "Scene budget fallback lost the opening scene.")
    check(reduced[-1]["scene_id"] == "scene_008", "Scene budget fallback lost the closing scene.")
    check(
        [scene["order"] for scene in reduced] == list(range(1, director.MAX_SCENES + 1)),
        "Scene budget fallback did not normalize scene order.",
    )


def test_scene_budget_semantic_repair_contract() -> None:
    director = QwenDirector(ROOT)
    scenes = [
        _sample_scene(f"scene_{index:03d}", ["Elias"], index)
        for index in range(1, 8)
    ]

    def fake_chat(*args, **kwargs):
        return {
            "scenes": [
                _sample_scene("scene_001", ["Elias"], 1),
                _sample_scene("scene_002", ["Elias"], 2),
                _sample_scene("scene_003", ["Elias"], 3),
                _sample_scene("scene_004", ["Elias"], 4),
                _sample_scene("scene_005", ["Elias"], 5),
            ]
        }

    original_chat = director._chat_json
    try:
        director._chat_json = fake_chat
        repaired = director._compress_scenes_to_budget(
            "ai_story",
            "Elias discovers the signal and reaches the final beacon.",
            [{"name": "Elias", "role": "protagonist"}],
            scenes,
            {"elias"},
        )
    finally:
        director._chat_json = original_chat

    check(len(repaired) == 5, "Semantic scene compression did not accept a valid 5-scene repair.")
    check(all(scene.get("description") for scene in repaired), "Compressed scenes contain an empty description.")
    check([scene["order"] for scene in repaired] == [1, 2, 3, 4, 5], "Semantic repair returned non-contiguous orders.")


def test_batch_planning_runtime_contract() -> None:
    director = QwenDirector(ROOT)
    import inspect
    source = inspect.getsource(director.generate)

    check(
        director.MAX_SHOT_BATCH_SCENES == 2,
        "Shot batch size must remain capped at 2 scenes.",
    )
    check(
        "_shot_director_batch_system" in source
        and "_normalize_batch_shot_response" in source,
        "Generate path is missing the batched shot-planning path.",
    )
    check(
        "_generate_scene_shots" in source,
        "Generate path is missing the per-scene fallback path.",
    )

    normalized = director._normalize_batch_shot_response(
        {
            "scene_shots": [
                {"scene_id": "scene_001", "shots": [_sample_shot("scene_001", 1), _sample_shot("scene_001", 2)]},
                {"scene_id": "scene_002", "shots": [_sample_shot("scene_002", 1), _sample_shot("scene_002", 2)]},
            ]
        }
    )
    check(set(normalized) == {"scene_001", "scene_002"}, "Batch normalization lost a scene.")
    check(all(len(value) == 2 for value in normalized.values()), "Batch normalization lost required shots.")


def test_batch_prompt_is_compact() -> None:
    director = QwenDirector(ROOT)
    scenes = [
        _sample_scene("scene_001", ["Elias"]),
        _sample_scene("scene_002", ["Sara"]),
    ]
    huge_story = " ".join(["A detailed narrative event about Elias and Sara and the ruined city."] * 500)
    payload = director._shot_director_batch_user(
        huge_story,
        [{"name": "Elias", "role": "protagonist"}, {"name": "Sara", "role": "supporting"}],
        scenes,
        {"genre_tone": "cinematic", "color_palette": "cold blue"},
    )
    check(len(payload) < 10000, "Batch shot prompt grew beyond the intended compact payload budget.")
    check("visual_language" in payload and "scenes" in payload, "Compact batch prompt lost required context.")
    check("personality" not in payload and "distinctive_features" not in payload, "Batch prompt included heavyweight character descriptors.")


def test_shot_prompt_is_compact() -> None:
    director = QwenDirector(ROOT)
    huge_story = " ".join(["Elias crosses the ruined city and follows the signal."] * 500)
    payload = director._shot_director_user(
        huge_story,
        [{"name": "Elias", "role": "protagonist"}],
        _sample_scene("scene_001", ["Elias"]),
        {"genre_tone": "cinematic", "pacing": "controlled"},
    )
    check(len(payload) < 12000, "Per-scene shot prompt is not compact enough.")
    check('"story_context"' in payload, "Shot prompt lost compact narrative context.")



def test_mult_word_character_extraction_regression() -> None:
    # P0 regression guard for the ProductionPlanner character extractor.
    # Multi-word names must remain intact and extraction cannot depend on a
    # closed hand-maintained verb list.
    planner = ProductionPlanner(ROOT)

    story = (
        "Marcus Chen arrived at the station just as "
        "Dr. Elara Voss finished her readings."
    )
    names = planner.detect_character_descriptors(story)
    normalized = {name.lower() for name in names}

    check(
        "marcus chen" in normalized,
        "ProductionPlanner truncated or missed multi-word name Marcus Chen.",
    )
    check(
        "elara voss" in normalized,
        "ProductionPlanner truncated or missed honorific multi-word name Dr. Elara Voss.",
    )
    check(
        "chen" not in normalized and "voss" not in normalized,
        "ProductionPlanner reduced a multi-word character to a surname.",
    )

    suffix_story = "Ava Morgan sprinted across the bridge while Daniel Stone watched."
    suffix_names = {
        name.lower()
        for name in planner.detect_character_descriptors(suffix_story)
    }
    check(
        {"ava morgan", "daniel stone"}.issubset(suffix_names),
        "ProductionPlanner morphological verb fallback missed multi-word narrative subjects.",
    )


def test_visual_language_partial_merge_preserves_base_fields() -> None:
    director = QwenDirector(ROOT)
    base_plan = {
        "story": "Elias walks home.",
        "characters": [],
        "scenes": [],
        "shots": [],
        "visual_language": {
            "genre_tone": "base tone",
            "color_palette": "base palette",
            "lighting_philosophy": "base lighting",
            "camera_philosophy": "base camera",
            "pacing": "base pacing",
        },
    }

    def fake_generate(self, **kwargs):
        return {
            "enabled": True,
            "plan": {
                "story": base_plan["story"],
                "visual_language": {
                    "genre_tone": "qwen tone",
                    "pacing": "",
                },
                "characters": [],
                "scenes": [],
                "shots": [],
            },
        }

    original = QwenDirector.generate
    QwenDirector.generate = fake_generate
    try:
        merged = director.enrich_plan(
            mode="preserve_user_story",
            user_input=base_plan["story"],
            base_plan=base_plan,
        )
    finally:
        QwenDirector.generate = original

    check(
        merged["visual_language"]["genre_tone"] == "qwen tone",
        "Creative visual-language field was not applied.",
    )
    check(
        merged["visual_language"]["camera_philosophy"] == "base camera",
        "Partial Qwen visual-language output erased a base field.",
    )
    check(
        merged["visual_language"]["pacing"] == "base pacing",
        "Empty Qwen visual-language value erased a base field.",
    )


def test_final_generation_uses_compiler_before_quality_validation() -> None:
    director = QwenDirector(ROOT)
    import inspect
    source = inspect.getsource(director.generate)
    compile_pos = source.find("CinematicCompiler(")
    quality_pos = source.find("self._validate_production_quality(")
    check(
        compile_pos >= 0 and quality_pos > compile_pos,
        "Final generation validates production quality before deterministic compilation.",
    )


def test_resume_does_not_rewrite_scene_ids() -> None:
    director = QwenDirector(ROOT)
    import inspect
    source = inspect.getsource(director.generate)
    check(
        'prior_director_plan.get(' in source and 'scene_id' in source,
        "Resume path does not retain prior director scene addressing.",
    )
    check(
        'if len(existing_scene_shots) >= self.SHOTS_PER_SCENE' in source,
        "Resume path does not preserve completed scene shots.",
    )



def test_deterministic_foundation_when_director_enabled() -> None:
    from planner.config import director_enabled

    original = os.environ.get("H3_DIRECTOR_ENABLED")
    try:
        os.environ["H3_DIRECTOR_ENABLED"] = "1"
        planner = ProductionPlanner(ROOT)
        result = planner.build(
            mode="preserve_user_story",
            user_input=(
                "Dr. Elara Voss enters the station. "
                "Marcus Chen follows her. "
                "They discover a hidden signal."
            ),
        )
        check(
            result["characters"],
            "Director-enabled build returned no deterministic characters.",
        )
        check(
            result["scenes"],
            "Director-enabled build returned no deterministic scenes.",
        )
    finally:
        if original is None:
            os.environ.pop("H3_DIRECTOR_ENABLED", None)
        else:
            os.environ["H3_DIRECTOR_ENABLED"] = original


def test_abbreviation_safe_story_split() -> None:
    planner = ProductionPlanner(ROOT)
    units = planner._split_story(
        "Dr. Elara Voss entered the station. Marcus Chen followed her. "
        "They found the signal."
    )
    text = " ".join(unit.text for unit in units)
    check(
        "dr. elara voss" in text.lower(),
        "Abbreviation-safe splitter broke 'Dr. Elara Voss'.",
    )


def test_expand_failure_is_source_fallback_without_retry() -> None:
    director = QwenDirector(ROOT)
    source = "Eli enters the abandoned station and finds a sealed vault."
    calls = []
    original = director._chat_text
    original_load = director.load

    def fake_load():
        director._llama = object()

    director.load = fake_load

    def fake_chat(*args, **kwargs):
        calls.append(kwargs.get("call_name", ""))
        if kwargs.get("call_name") == "expand_story_text_pass":
            raise RuntimeError("forced validation failure")
        raise RuntimeError("unexpected retry")

    director._chat_text = fake_chat
    try:
        # Exercise only the contract: a failed expansion must not invoke a retry.
        try:
            director.generate(
                mode="expand_user_story",
                user_input=source,
                base_plan={
                    "story": source,
                    "characters": [{"name": "Eli"}],
                    "scenes": [{"scene_id": "scene_001", "order": 1, "characters": ["Eli"], "shot_ids": []}]*4,
                    "shots": [],
                },
            )
        except RuntimeError:
            pass
    finally:
        director._chat_text = original
        director.load = original_load
        director._llama = None

    check(
        "expand_story_text_retry" not in calls,
        "Expand mode still attempted an expensive Qwen retry.",
    )

def main() -> None:
    test_deterministic_foundation_when_director_enabled()
    test_abbreviation_safe_story_split()
    test_expand_failure_is_source_fallback_without_retry()

    tests = [
        test_story_modes,
        test_expand_preservation_gates,
        test_shot_batch_contract,
        test_text_generation_disables_thinking_by_default,
        test_character_sanitization,
        test_scene_id_sanitization_before_batching,
        test_shot_id_normalization,
        test_character_descriptor_deduplication,
        test_single_paragraph_segmentation,
        test_director_prompt_contract,
        test_shot_sampling_contract,
        test_visual_schema_sanitization,
        test_shot_schema_cardinality_is_grammar_constrained,
        test_shot_sanitization_cinematography_fields,
        test_canonical_roster_not_overwritten_by_qwen,
        test_entity_resolver_shot_rebinding,
        test_mult_word_character_extraction_regression,
        test_visual_language_partial_merge_preserves_base_fields,
        test_final_generation_uses_compiler_before_quality_validation,
        test_scene_budget_contract_and_fallback,
        test_scene_budget_semantic_repair_contract,
        test_batch_planning_runtime_contract,
        test_batch_prompt_is_compact,
        test_shot_prompt_is_compact,
        test_resume_does_not_rewrite_scene_ids,
    ]

    for test in tests:

        test()

        print(
            f"PASS: {test.__name__}"
        )

    print(
        "Director validation PASSED."
    )


if __name__ == "__main__":
    main()

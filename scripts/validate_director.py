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


def main() -> None:

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
        test_shot_sanitization_cinematography_fields,
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

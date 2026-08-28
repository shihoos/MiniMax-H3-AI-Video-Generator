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


def main() -> None:

    tests = [
        test_story_modes,
        test_expand_preservation_gates,
        test_shot_batch_contract,
        test_text_generation_disables_thinking_by_default,
        test_character_sanitization,
        test_shot_id_normalization,
        test_character_descriptor_deduplication,
        test_single_paragraph_segmentation,
        test_director_prompt_contract,
        test_shot_sampling_contract,
        test_visual_schema_sanitization,
        test_shot_sanitization_cinematography_fields,
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

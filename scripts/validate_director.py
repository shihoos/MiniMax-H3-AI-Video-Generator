from __future__ import annotations

import re
from pathlib import Path

from planner.production_planner import (
    ProductionPlanner,
)
from planner.qwen_director import (
    QwenDirector,
)


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
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
        for shot in shots
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
        "Do not create new characters"
        in shots,
        "Shot director does not protect character identity.",
    )


def test_config_contract() -> None:

    from planner.config import (
        DIRECTOR_MAX_TOKENS,
        DIRECTOR_N_BATCH,
        DIRECTOR_N_CTX,
        DIRECTOR_N_GPU_LAYERS,
        DIRECTOR_TOP_P,
        DIRECTOR_TEMPERATURE,
    )

    check(
        DIRECTOR_N_CTX == 8192,
        f"Unexpected director context: {DIRECTOR_N_CTX}",
    )

    check(
        DIRECTOR_MAX_TOKENS > 0,
        "Director max tokens must be positive.",
    )

    check(
        DIRECTOR_N_BATCH > 0,
        "Director batch must be positive.",
    )

    check(
        DIRECTOR_N_GPU_LAYERS != 0,
        "Director GPU layer setting is invalid.",
    )

    check(
        0.0 < DIRECTOR_TEMPERATURE <= 2.0,
        "Director temperature is outside valid range.",
    )

    check(
        0.0 < DIRECTOR_TOP_P <= 1.0,
        "Director top_p is outside valid range.",
    )


def main() -> None:

    tests = [
        test_story_modes,
        test_character_sanitization,
        test_shot_id_normalization,
        test_character_descriptor_deduplication,
        test_single_paragraph_segmentation,
        test_director_prompt_contract,
        test_config_contract,
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

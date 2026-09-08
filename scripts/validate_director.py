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
from planner.entity_resolver import (
    EntityResolver,
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
        f"Create exactly {QwenDirector.SHOTS_PER_SCENE} production-ready shots for EACH supplied scene." in prompt,
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
    director = QwenDirector(ROOT)

    ai = director._story_text_system("ai_story")
    expand = director._story_text_system("expand_user_story")
    shots = director._shot_director_batch_system()

    # AI Story: validate behavior, not a fragile literal heading.
    check(
        "complete cinematic short-film story" in ai.lower(),
        "AI Story text prompt does not require a complete cinematic story.",
    )

    check(
        "Output ONLY the story prose" in ai,
        "AI Story text prompt does not enforce prose-only output.",
    )

    check(
        "Do not output JSON" in ai,
        "AI Story text prompt still permits JSON output.",
    )

    # Expand Story: validate the actual expansion contract.
    check(
        "Expand the supplied story substantially" in expand,
        "Expand Story text prompt does not require substantial expansion.",
    )

    check(
        "Output ONLY the expanded story prose" in expand,
        "Expand Story text prompt does not enforce prose-only output.",
    )

    check(
        "Do not replace the original plot" in expand,
        "Expand Story text prompt does not protect the original plot.",
    )

    # Preserve Story intentionally has no story-text generation pass.
    try:
        director._story_text_system("preserve_user_story")
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "Preserve Story should not use the story-text generation pass."
        )

    # Active cinematography/shot-generation contract.
    check(
        "visual-language consistency" in shots,
        "Shot prompt lost visual-language continuity requirements.",
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

    check(
        "SCENE-FUNCTION DIRECTING" in shots
        and "obligatory_moment" in shots,
        "Shot prompt is missing scene-function / obligatory-moment directing constraints.",
    )

    check(
        "Do NOT output compiler-owned fields." in shots,
        "Shot prompt is missing compiler-ownership boundaries.",
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
    # Grammar-constrained (JSON Schema) decoding is the project's
    # chosen speed/correctness strategy for JSON calls. This locks in
    # that the valid shot-count contracts are enforced at the schema
    # level; missing shots are repaired deterministically later rather
    # than by another Qwen recovery call.
    normal = QwenDirector._shot_json_schema()
    check(
        normal["properties"]["shots"]["minItems"] == 2
        and normal["properties"]["shots"]["maxItems"] == 2,
        "Normal/retry shot schema must constrain to exactly "
        "SHOTS_PER_SCENE shots.",
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





def test_entity_resolution_adversarial_regressions() -> None:
    planner = ProductionPlanner(ROOT)

    cases = {
        "Eli enters the station. Sara gives Eli a map.": {"Eli", "Sara"},
        "Eli enters the station. Sara helps Eli escape.": {"Eli", "Sara"},
        "Eli enters the station. Sara says the signal is dangerous.": {"Eli", "Sara"},
        "Eli, a scientist, enters the station.": {"Eli"},
        "Mira, the detective, follows Arun.": {"Mira", "Arun"},
        "Sara, an engineer, and Eli, a pilot, arrive.": {"Sara", "Eli"},
        "Mira and Arun arrive at the station.": {"Mira", "Arun"},
        "Dr. Elara Voss entered the station. Marcus Chen followed her.": {"Elara Voss", "Marcus Chen"},
        "Sara said, \"Eli, run!\"": {"Sara", "Eli"},
    }

    for story, expected in cases.items():
        actual = set(planner.detect_character_descriptors(story))
        check(
            expected.issubset(actual),
            f"Entity detector missed canonical names for: {story}; actual={sorted(actual)}",
        )

    roles = {
        value.lower()
        for value in planner.detect_character_descriptors(
            "Sara, an engineer, and Eli, a pilot, arrive."
        )
    }
    check(
        not ({"engineer", "pilot"} & roles),
        "Appositive role descriptors leaked into the canonical roster.",
    )

    false_positive = {
        value.lower()
        for value in planner.detect_character_descriptors(
            "The Station opened at dawn."
        )
    }
    check(
        "station" not in false_positive,
        "A location noun was classified as a character.",
    )


def test_character_appearance_is_locally_scoped() -> None:
    planner = ProductionPlanner(ROOT)
    appearance = planner._appearance_from_story(
        "Eli",
        "Eli has long hair. Sara wears a red coat and has short hair.",
    )
    check(
        appearance["hair"] == "long hair",
        "Eli inherited Sara's unrelated hair description.",
    )
    check(
        "red" not in appearance["clothing"],
        "Eli inherited Sara's unrelated clothing description.",
    )


def test_cinematic_compiler_cannot_promote_scene_identity() -> None:
    from planner.cinematic_compiler import CinematicCompiler

    compiler = CinematicCompiler({"Eli", "Sara"})
    scene = _sample_scene("scene_001", ["Eli"], 1)
    shot = _sample_shot("scene_001", 1)
    shot["characters"] = ["Eli", "Invented Character"]
    compiled = compiler.compile_shot(scene, shot, 1)
    check(
        compiled["characters"] == ["Eli"],
        "Compiler promoted an unrecognized Qwen character into the output roster.",
    )
    check(
        "invented character" not in compiler.character_names,
        "Compiler mutated the canonical identity set.",
    )


def test_h3_optimizer_ownership_guard() -> None:
    from execution.h3_workflow_builder import H3WorkflowBuilder

    workflow = {
        "nodes": [
            {"id": 1, "type": "UNETLoader", "outputs": [{"name": "MODEL", "links": [1, 2]}]},
            {"id": 2, "type": "H3MemoryOptimization", "inputs": [{"name": "model", "type": "MODEL", "link": 1}], "outputs": [{"name": "MODEL", "links": [3, 4]}]},
            {"id": 3, "type": "BasicScheduler", "inputs": [{"name": "model", "type": "MODEL", "link": 3}]},
            {"id": 4, "type": "BasicGuider", "inputs": [{"name": "model", "type": "MODEL", "link": 4}]},
        ],
        "links": [
            [1, 1, 0, 2, 0, "MODEL"],
            [2, 1, 0, 3, 0, "MODEL"],
            [3, 2, 0, 3, 0, "MODEL"],
            [4, 2, 0, 4, 0, "MODEL"],
        ],
    }
    try:
        H3WorkflowBuilder._assert_optimizer_ownership(
            workflow,
            "UNETLoader",
        )
    except RuntimeError:
        return
    raise RuntimeError(
        "H3 optimizer ownership guard accepted a bypassing MODEL edge."
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



def test_h3_workflow_duration_updates_float_source() -> None:
    from execution.h3_workflow_builder import H3WorkflowBuilder

    builder = H3WorkflowBuilder(ROOT, None)
    workflow = builder.load("ref2va")

    expression = next(
        node
        for node in workflow["nodes"]
        if node.get("type") == "ComfyMathExpression"
    )
    primitive = next(
        node
        for node in workflow["nodes"]
        if node.get("type") == "PrimitiveFloat"
        and node.get("title") == "Float (Duration)"
    )

    original_expression = expression["widgets_values"][0]
    builder._set_duration(workflow, 5.0)

    check(
        expression["widgets_values"][0] == original_expression,
        "H3 duration update modified the ComfyMathExpression formula.",
    )
    check(
        primitive["widgets_values"][0] == 5.0,
        "H3 duration update did not modify the PrimitiveFloat source.",
    )
    check(
        primitive.get("widgets_values_named", {}).get("value") == 5.0,
        "H3 duration named widget value was not updated.",
    )

    ref_node = builder._one(
        workflow,
        "MiniMaxH3ReferenceToVideo",
    )
    check(
        ref_node["widgets_values"][3] == 124,
        "5.0 seconds did not resolve to the H3-legal 124-frame length.",
    )


def test_h3_workflow_resolution_selector_mapping() -> None:
    from execution.h3_workflow_builder import H3WorkflowBuilder

    builder = H3WorkflowBuilder(ROOT, None)
    expected = {
        (1344, 768): 0.98,
        (1216, 672): 0.80,
        (1056, 608): 0.60,
        (1920, 1088): 2.00,
    }

    for (width, height), megapixels in expected.items():
        workflow = builder.load("ref2va")
        builder._set_resolution(
            workflow,
            width,
            height,
        )
        selector = builder._one(
            workflow,
            "ResolutionSelector",
        )
        widgets = selector["widgets_values"]

        check(
            widgets[0] == "16:9 (Widescreen)"
            and float(widgets[1]) == megapixels
            and int(widgets[2]) == 32,
            f"Resolution selector mapping failed for {width}x{height}.",
        )

    try:
        workflow = builder.load("ref2va")
        builder._set_resolution(
            workflow,
            1400,
            800,
        )
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "Unsupported H3 resolution was accepted instead of failing loudly."
        )


def test_short_story_rebalances_to_four_units_without_losing_source_text() -> None:
    planner = ProductionPlanner(ROOT)

    source = "Eli enters the station."
    units = planner._rebalance_story_units(
        planner._split_story(source)
    )

    check(
        len(units) == 4,
        "Short story did not rebalance to four structural planning units.",
    )

    joined = " ".join(
        unit.text
        for unit in units
    )
    check(
        "Eli" in joined and "enters" in joined and "station" in joined,
        "Short-story rebalance lost source narrative content.",
    )

    tiny = planner._rebalance_story_units(
        planner._split_story("Run.")
    )
    check(
        len(tiny) == 4,
        "Extremely short story did not produce four structural units.",
    )
    check(
        all(unit.text == "Run." for unit in tiny),
        "Tiny-source fallback changed the source text.",
    )


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


def test_verified_roster_flag_propagates_to_orchestrator() -> None:
    # Regression guard for a real production failure reproduced from a
    # live Kaggle benchmark: AI Story mode produced "characters=0" and
    # a hard AssertionError on a story that clearly named two
    # characters ("Elena Kovalenko", "Anton"). Root cause: enrich_plan()
    # correctly computed merged["characters"] from generate()'s
    # verified, story-derived roster, but never copied the
    # "_canonical_character_roster_verified" flag itself into the
    # returned dict. The orchestrator's boundary check
    # (production_orchestrator.py) reads exactly this key to decide
    # whether it may trust the roster enrich_plan() just computed; with
    # the flag missing, it always fell back to its own premise-derived
    # roster -- which is empty for AI Story mode, since the user's
    # premise rarely names the characters Qwen goes on to invent in the
    # final story. This test simulates that exact orchestrator check.
    director = QwenDirector(ROOT)

    premise = (
        "Write a sci-fi thriller about a researcher who discovers "
        "something dangerous in an abandoned Arctic station."
    )

    base_plan = {
        "story": premise,
        # What a premise-only deterministic pass would find: nobody,
        # since the premise itself never names a character.
        "characters": [],
        "scenes": [
            {
                "scene_id": f"scene_{i:03d}",
                "order": i,
                "characters": [],
                "shot_ids": [],
            }
            for i in range(1, 5)
        ],
        "shots": [],
        "visual_language": {},
    }

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
                "story": (
                    "Elena Kovalenko stumbled through the blinding "
                    "snow. Anton had left notes behind."
                ),
                "director_notes": "",
                "visual_language": {},
                "characters": [
                    {
                        "character_id": "char_elena",
                        "name": "Elena Kovalenko",
                        "role": "protagonist",
                        "description": "",
                        "personality": "",
                    },
                    {
                        "character_id": "char_anton",
                        "name": "Anton",
                        "role": "supporting",
                        "description": "",
                        "personality": "",
                    },
                ],
                "_canonical_character_roster_verified": True,
                "scenes": base_plan["scenes"],
                "shots": [],
            },
        }

    original_generate = QwenDirector.generate
    QwenDirector.generate = fake_generate
    try:
        merged = director.enrich_plan(
            mode="ai_story",
            user_input=premise,
            base_plan=base_plan,
        )
    finally:
        QwenDirector.generate = original_generate

    check(
        merged.get("_canonical_character_roster_verified") is True,
        "enrich_plan() did not propagate the "
        "_canonical_character_roster_verified flag into its returned "
        "dict, even though generate() marked the roster verified.",
    )

    # Simulate the orchestrator's own boundary check verbatim.
    plan = merged
    premise_derived_characters: list = []
    if (
        not isinstance(plan, dict)
        or plan.get("_canonical_character_roster_verified") is not True
    ):
        plan["characters"] = premise_derived_characters

    check(
        {c["name"] for c in plan["characters"]}
        == {"Elena Kovalenko", "Anton"},
        "A verified, story-derived character roster was lost when "
        "passed through the orchestrator's boundary check -- this is "
        "the exact 'characters=0' production failure.",
    )


def test_verified_scene_topology_not_orphaned() -> None:
    # Regression guard for a real production failure reproduced from a
    # live Kaggle benchmark: AI Story mode produced "scenes=4, shots=8"
    # when generate() had actually produced a verified 6-scene,
    # 12-shot topology derived from the final story. Root cause:
    # enrich_plan() always anchored canonical_scenes to the premise-
    # derived base_plan (computed by planner.build() before the
    # director ever ran), discarding any story-derived scene beyond
    # what the short premise alone produced -- along with the shots
    # generated for it, silently throwing away real Qwen shot-batch
    # compute. This mirrors the character-roster fix: a verified
    # director pass's own scene topology must take priority over the
    # premise-derived skeleton.
    director = QwenDirector(ROOT)

    premise = (
        "A polar systems engineer reaches an abandoned Arctic "
        "station during a violent storm and discovers a sealed "
        "underground vault."
    )

    # What planner.build(premise) actually produces: a short,
    # premise-derived 4-scene skeleton (the real observed behavior --
    # a short premise naturally splits into fewer scenes than a full
    # generated story).
    base_plan = {
        "story": premise,
        "characters": [],
        "scenes": [
            {
                "scene_id": f"scene_{i:03d}",
                "order": i,
                "characters": [],
                "shot_ids": [],
                "title": f"premise scene {i}",
            }
            for i in range(1, 5)
        ],
        "shots": [],
        "visual_language": {},
    }

    def fake_generate(
        self,
        *,
        mode,
        user_input,
        base_plan,
        checkpoint_session_id=None,
        resume_state=None,
    ):
        scenes = [
            {
                "scene_id": f"scene_{i:03d}",
                "order": i,
                "characters": ["Elena Kovalenko"],
                "shot_ids": [],
                "title": f"story scene {i}",
                "mood": "tense",
            }
            for i in range(1, 7)
        ]
        shots = [
            {
                "shot_id": f"scene_{i:03d}_shot_{j:03d}",
                "scene_id": f"scene_{i:03d}",
                "characters": ["Elena Kovalenko"],
            }
            for i in range(1, 7)
            for j in range(1, 3)
        ]
        return {
            "enabled": True,
            "plan": {
                "story": (
                    "Elena Kovalenko stumbled through the blinding "
                    "snow toward the Arctic station..."
                ),
                "director_notes": "",
                "visual_language": {},
                "characters": [
                    {
                        "character_id": "char_elena",
                        "name": "Elena Kovalenko",
                        "role": "protagonist",
                        "description": "",
                        "personality": "",
                    },
                ],
                "_canonical_character_roster_verified": True,
                "scenes": scenes,
                "shots": shots,
            },
        }

    original_generate = QwenDirector.generate
    QwenDirector.generate = fake_generate
    try:
        merged = director.enrich_plan(
            mode="ai_story",
            user_input=premise,
            base_plan=base_plan,
        )
    finally:
        QwenDirector.generate = original_generate

    check(
        len(merged["scenes"]) == 6,
        "Verified story-derived scene topology was truncated to the "
        f"premise-derived scene count: got {len(merged['scenes'])} "
        "scenes, expected 6. This silently discards real Qwen "
        "shot-batch work for the dropped scenes.",
    )

    check(
        len(merged["shots"]) == 12,
        "Shots for story-derived scenes beyond the premise-derived "
        f"count were dropped: got {len(merged['shots'])} shots, "
        "expected 12.",
    )


def test_qwen_excluded_candidate_not_silently_readded() -> None:
    # Regression guard for the live "Arctic" leak: Qwen's character
    # extraction correctly returned only ["Elena Kovalenko", "Anton"],
    # but the final roster contained a third, wrong entry ("Arctic")
    # that neither the deterministic detector nor Qwen's own raw
    # result actually named. Root cause was the reconciliation layer
    # silently re-adding deterministically-flagged candidates Qwen had
    # excluded. Once Qwen produces a usable roster, it is the semantic
    # authority; deterministic candidates are a fallback only, never a
    # silent addition on top of a valid Qwen answer.
    planner = ProductionPlanner(ROOT)

    story = (
        "The wind screamed like a wounded beast as Elena Kovalenko "
        "stumbled through the blinding snow. The Arctic station had "
        "been abandoned for years. Inside, the walls were covered in "
        "scrawled equations and desperate notes, the handwriting of "
        "the previous engineer, a man named Anton."
    )

    def fake_extractor(story_text, descriptors):
        # Exact recorded Qwen response from the live benchmark run.
        return {"characters": ["Elena Kovalenko", "Anton"]}

    characters = planner.create_characters(
        story,
        qwen_character_extractor=fake_extractor,
    )

    names = {c.name for c in characters}

    check(
        names == {"Elena Kovalenko", "Anton"},
        "A valid Qwen character roster was contaminated by a "
        f"silently re-added deterministic candidate: got {sorted(names)}, "
        "expected exactly {'Elena Kovalenko', 'Anton'}.",
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
        "Shot batch max scenes must be 2.",
    )

    check(
        "_shot_director_batch_system" in source
        and "_shot_director_batch_user" in source
        and "_normalize_batch_shot_response" in source,
        "Generate path is missing the batched shot-planning path.",
    )

   
    normalized = director._normalize_batch_shot_response(
        {
            "scene_shots": [
                {
                    "scene_id": "scene_001",
                    "shots": [
                        _sample_shot("scene_001", 1),
                        _sample_shot("scene_001", 2),
                    ],
                },
                {
                    "scene_id": "scene_002",
                    "shots": [
                        _sample_shot("scene_002", 1),
                        _sample_shot("scene_002", 2),
                    ],
                },
            ]
        }
    )

    check(
        set(normalized) == {"scene_001", "scene_002"},
        "Batch normalization lost a scene.",
    )

    check(
        all(len(value) == 2 for value in normalized.values()),
        "Batch normalization lost required shots.",
    )

    single = director._shot_batch_json_schema(scene_count=1)
    check(
        single["properties"]["scene_shots"]["minItems"] == 1
        and single["properties"]["scene_shots"]["maxItems"] == 1,
        "Single-scene batch schema is missing.",
    )

    five = director._shot_batch_json_schema(scene_count=5)
    check(
        five["properties"]["scene_shots"]["minItems"] == 5
        and five["properties"]["scene_shots"]["maxItems"] == 5,
        "Five-scene batch schema is missing.",
    )

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


# Removed test_shot_prompt_is_compact since _shot_director_user is gone.


def test_qwen_semantic_negative_is_authoritative() -> None:
    planner = ProductionPlanner(ROOT)
    story = (
        "Elena Kovalenko stumbled through the blinding snow. "
        "The Arctic station had been abandoned for years."
    )
    deterministic = planner.detect_character_descriptors(story)
    check(
        "Elena Kovalenko" in deterministic,
        f"Deterministic extraction lost Elena Kovalenko: {deterministic}",
    )

    semantic = {
        "candidates": [
            {
                "name": "Elena Kovalenko",
                "entity_type": "PERSON",
                "is_character": False,
                "aliases": [],
            },
        ]
    }

    names = {
        value.lower()
        for value in planner._reconcile_semantic_characters(
            story,
            deterministic,
            semantic,
        )
    }
    check(
        "elena kovalenko" not in names,
        "Planner incorrectly overrode a Qwen semantic negative.",
    )


def test_qwen_semantic_character_reconciliation() -> None:
    planner = ProductionPlanner(ROOT)

    story = (
        "The Research Station was silent. Dr. Elara Voss checked Sara's notebook. "
        "Marcus Chen waited outside. The United Nations issued a warning. "
        "Captain Rho, exhausted after the journey, entered the chamber."
    )

    deterministic = planner.detect_character_descriptors(story)

    semantic = {
        "candidates": [
            {"name": "Elara Voss", "entity_type": "PERSON", "is_character": True, "aliases": ["Dr. Voss"]},
            {"name": "Sara", "entity_type": "CHARACTER", "is_character": True, "aliases": []},
            {"name": "Marcus Chen", "entity_type": "PERSON", "is_character": True, "aliases": ["Marcus"]},
            {"name": "Captain Rho", "entity_type": "CHARACTER", "is_character": True, "aliases": ["Rho"]},
            {"name": "Research Station", "entity_type": "FACILITY", "is_character": False, "aliases": []},
            {"name": "United Nations", "entity_type": "ORGANIZATION", "is_character": False, "aliases": ["UN"]},
            {"name": "Invented Person", "entity_type": "PERSON", "is_character": True, "aliases": []},
        ]
    }

    names = {
        value.lower()
        for value in planner._reconcile_semantic_characters(
            story,
            deterministic,
            semantic,
        )
    }

    check("elara voss" in names, "Qwen recovery lost Elara Voss.")
    check("sara" in names, "Qwen recovery lost possessive character Sara.")
    check("marcus chen" in names, "Qwen recovery lost Marcus Chen.")
    check("rho" in names, "Qwen recovery missed the named title-form character.")
    check("research station" not in names, "Qwen reconciliation kept a facility as a character.")
    check("united nations" not in names, "Qwen reconciliation kept an organization as a character.")
    check("invented person" not in names, "Qwen reconciliation accepted a hallucinated name.")
    check("voss" not in names, "Qwen reconciliation retained a shorter surname beside Elara Voss.")
    check("scientist" not in names, "Qwen reconciliation promoted an anonymous role to a canonical character.")



def test_character_pipeline_has_no_external_ner_dependency() -> None:
    import inspect
    source = inspect.getsource(ProductionPlanner).lower()
    forbidden_package = "spa" + "cy"
    forbidden_model = "en_core" + "_web_sm"
    check(forbidden_package not in source, "ProductionPlanner contains a forbidden external NER package.")
    check(forbidden_model not in source, "ProductionPlanner references a forbidden external NER model.")


def test_semantic_character_reconciliation_adversarial_matrix() -> None:
    planner = ProductionPlanner(ROOT)

    cases = (
        ("Dr. Elara Voss entered the station.", {"elara voss"}),
        ("Sara's notebook was open beside Marcus Chen.", {"sara", "marcus chen"}),
        ("Eli whispered, \"Sara, run!\"", {"eli", "sara"}),
        ("Captain Rho, exhausted after the journey, entered.", {"rho"}),
        ("The United Nations issued a warning while Marcus watched.", {"marcus"}),
        ("The Research Station was silent. Naomi Reyes checked the console.", {"naomi reyes"}),
        ("Paris was quiet before Elena arrived.", {"elena"}),
        ("Washington was evacuated after Marcus left.", {"marcus"}),
        ("Amazon delivered the package while Sara waited.", {"sara"}),
        ("The Apollo Mission launched as Eli watched.", {"eli"}),
        ("A scientist named Mira entered. Arun followed.", {"mira", "arun"}),
        ("The pilot and Sara arrived together.", {"sara"}),
        ("Sara was a scientist at the station.", {"sara"}),
        ("Mira, a systems engineer, arrived.", {"mira"}),
        ("The old commander, Marcus, raised his weapon.", {"marcus"}),
        ("Zara-Lin activated the console and Nex'to watched.", {"zara-lin", "nex'to"}),
        ("John Doe arrived while Ava Morgan waited.", {"john doe", "ava morgan"}),
        ("Prof. Amina al-Rashid arrived before Daniel Stone.", {"amina al-rashid", "daniel stone"}),
        ("The woman everyone called Elara stepped forward.", {"elara"}),
        ("Later, Elias understood what Sara meant.", {"elias", "sara"}),
        ("Behind her, Marcus opened the door.", {"marcus"}),
        ("Across the room stood Dr. Lina Park.", {"lina park"}),
        ("Eli followed Marcus into Central Command.", {"eli", "marcus"}),
        ("The Frozen Lake was empty; Talia waited nearby.", {"talia"}),
        ("Monday arrived cold, and Elena smiled.", {"elena"}),
        ("Renn led Kass through the tunnels while Odile followed.", {"renn", "kass", "odile"}),
        ("Sara said that the Warden was coming.", {"sara", "the warden"}),
        ("The Warden watched from the tower while Eli waited.", {"the warden", "eli"}),
        ("Nex'to's signal reached Zara-Lin first.", {"nex'to", "zara-lin"}),
        ("The commander known as Marcus spoke to Eli.", {"marcus", "eli"}),
    )

    for story, expected in cases:
        deterministic = planner.detect_character_descriptors(story)
        deterministic_names = {str(value).strip() for value in deterministic if str(value).strip()}
        semantic_candidates = []
        for name in sorted(expected | deterministic_names):
            semantic_candidates.append(
                {
                    "name": name,
                    "entity_type": "PERSON" if name.lower() not in {"the warden"} else "CHARACTER",
                    "is_character": name.lower() in {value.lower() for value in expected},
                    "aliases": [],
                }
            )
        semantic_candidates.extend([
            {"name": "United Nations", "entity_type": "ORGANIZATION", "is_character": False, "aliases": []},
            {"name": "Research Station", "entity_type": "FACILITY", "is_character": False, "aliases": []},
            {"name": "Invented Person", "entity_type": "PERSON", "is_character": True, "aliases": []},
            {"name": "scientist", "entity_type": "ROLE", "is_character": True, "aliases": []},
        ])
        semantic = {"candidates": semantic_candidates}
        got = {
            value.lower()
            for value in planner._reconcile_semantic_characters(story, deterministic, semantic)
        }
        check(
            got == {value.lower() for value in expected},
            f"Semantic reconciliation mismatch for {story!r}: expected {sorted(expected)}, got {sorted(got)}",
        )


def test_verified_semantic_character_roster_reaches_final_plan() -> None:
    director = QwenDirector(ROOT)
    base_plan = {
        "story": "Eli entered the station.",
        "characters": [{"name": "Eli"}],
        "scenes": [],
        "shots": [],
        "visual_language": {},
    }

    original_generate = director.generate
    try:
        director.generate = lambda **kwargs: {
            "enabled": True,
            "plan": {
                "story": "Eli entered the station.",
                "characters": [{"name": "Eli"}, {"name": "Sara"}],
                "scenes": [],
                "shots": [],
                "visual_language": {},
                "_canonical_character_roster_verified": True,
            },
        }
        merged = director.enrich_plan(
            mode="PRESERVE_USER_STORY_MODE",
            user_input=base_plan["story"],
            base_plan=base_plan,
        )
    finally:
        director.generate = original_generate

    names = {str(item.get("name", "")).lower() for item in merged["characters"] if isinstance(item, dict)}
    check(names == {"eli", "sara"}, "Verified semantic roster did not reach the final production plan.")


def test_qwen_semantic_character_extractor_contract() -> None:
    planner = ProductionPlanner(ROOT)

    calls = []

    def fake_extractor(story, candidates):
        calls.append((story, list(candidates)))
        return {
            "candidates": [
                {
                    "name": "Eli",
                    "entity_type": "PERSON",
                    "is_character": True,
                    "aliases": [],
                },
                {
                    "name": "Station",
                    "entity_type": "FACILITY",
                    "is_character": False,
                    "aliases": [],
                },
            ]
        }

    characters = planner.create_characters(
        "Eli entered the station.",
        qwen_character_extractor=fake_extractor,
    )
    names = {character.name.lower() for character in characters}

    check(calls, "Qwen character extractor callback was not invoked.")
    check("eli" in names, "Semantic extraction failed to preserve Eli.")
    check("station" not in names, "Semantic extraction allowed a facility into the canonical roster.")


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


def test_cinematic_compiler_deterministic_fallback() -> None:
    from planner.cinematic_compiler import CinematicCompiler

    scene = _sample_scene("scene_001", ["Elias", "Mara"], 1)
    compiler = CinematicCompiler(
        character_names={"Elias", "Mara"},
    )

    compiled = compiler.compile_all(
        [scene],
        [],
    )

    check(
        len(compiled) == 2,
        "CinematicCompiler fallback did not produce exactly two shots.",
    )
    check(
        all(shot.get("scene_id") == "scene_001" for shot in compiled),
        "Compiler fallback changed the canonical scene ID.",
    )
    check(
        len({shot.get("shot_id") for shot in compiled}) == 2
        and all(str(shot.get("shot_id", "")).strip() for shot in compiled),
        "Compiler fallback did not produce unique non-empty shot IDs.",
    )
    check(
        all(
            str(shot.get("camera_shot", "")).strip()
            and str(shot.get("camera_movement", "")).strip()
            and str(shot.get("lens_and_depth_of_field", "")).strip()
            and str(shot.get("composition_notes", "")).strip()
            and str(shot.get("lighting", "")).strip()
            and str(shot.get("color_temperature", "")).strip()
            and str(shot.get("mood", "")).strip()
            and str(shot.get("visual_prompt", "")).strip()
            for shot in compiled
        ),
        "Compiler fallback did not satisfy required creative shot fields.",
    )
    check(
        all(set(shot.get("characters", [])) <= {"Elias", "Mara"} for shot in compiled),
        "Compiler fallback introduced a character outside the canonical roster.",
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
    director._count_tokens = lambda text: 100

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
        test_h3_workflow_duration_updates_float_source,
        test_h3_workflow_resolution_selector_mapping,
        test_short_story_rebalances_to_four_units_without_losing_source_text,
        test_canonical_roster_not_overwritten_by_qwen,
        test_verified_roster_flag_propagates_to_orchestrator,
        test_verified_scene_topology_not_orphaned,
        test_qwen_excluded_candidate_not_silently_readded,
        test_entity_resolver_shot_rebinding,
        test_entity_resolution_adversarial_regressions,
        test_character_appearance_is_locally_scoped,
        test_cinematic_compiler_cannot_promote_scene_identity,
        test_h3_optimizer_ownership_guard,
        test_character_pipeline_has_no_external_ner_dependency,
        test_semantic_character_reconciliation_adversarial_matrix,
        test_qwen_semantic_negative_is_authoritative,
        test_qwen_semantic_authority_prefers_qwen_roster,
        test_qwen_semantic_alias_anchor,
        test_entity_resolver_has_no_qwen_handle,
        test_recorded_semantic_payloads_are_terminal,
        test_character_semantic_call_budget_is_bounded,
        test_qwen_semantic_character_reconciliation,
        test_verified_semantic_character_roster_reaches_final_plan,
        test_qwen_semantic_character_extractor_contract,
        test_mult_word_character_extraction_regression,
        test_visual_language_partial_merge_preserves_base_fields,
        test_final_generation_uses_compiler_before_quality_validation,
        test_cinematic_compiler_deterministic_fallback,
        test_scene_budget_contract_and_fallback,
        test_scene_budget_semantic_repair_contract,
        test_batch_planning_runtime_contract,
        test_batch_prompt_is_compact,
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


def test_qwen_semantic_authority_prefers_qwen_roster() -> None:
    planner = ProductionPlanner(ROOT)
    story = "Elena Kovalenko entered the Arctic station. Anton repaired the generator."
    result = planner.create_characters(
        story,
        qwen_character_extractor=lambda _story, _hints: {
            "candidates": [
                {"name": "Elena Kovalenko", "entity_type": "PERSON", "is_character": True, "aliases": ["Elena"]},
                {"name": "Anton", "entity_type": "PERSON", "is_character": True, "aliases": []},
            ]
        },
    )
    check(
        {c.name for c in result} == {"Elena Kovalenko", "Anton"},
        "Qwen semantic roster was not treated as authoritative.",
    )


def test_qwen_semantic_alias_anchor() -> None:
    planner = ProductionPlanner(ROOT)
    story = "Elena Kovalenko entered the room. Later Elena waited outside."
    result = planner.create_characters(
        story,
        qwen_character_extractor=lambda _story, _hints: {
            "candidates": [
                {"name": "Elena", "entity_type": "PERSON", "is_character": True, "aliases": ["Elena Kovalenko"]},
            ]
        },
    )
    check(len(result) == 1 and result[0].name == "Elena", "Alias-tolerant anchoring failed.")


def test_entity_resolver_has_no_qwen_handle() -> None:
    import inspect
    signature = inspect.signature(EntityResolver.resolve_scene_aliases)
    check("qwen_chat" not in signature.parameters, "EntityResolver still exposes a live Qwen callback.")


def test_recorded_semantic_payloads_are_terminal() -> None:
    planner = ProductionPlanner(ROOT)
    cases = [
        (
            "Eli entered the station and waited outside.",
            {"characters": ["Eli"]},
        ),
        (
            "The station was empty. Dust covered the floor.",
            {"characters": ["Dust", "They're", "man"]},
        ),
    ]
    for story, semantic in cases:
        values = planner._reconcile_semantic_characters(
            story,
            planner.detect_character_descriptors(story),
            semantic,
        )
        check(isinstance(values, list), "Recorded semantic payload did not terminate with a finite roster.")


def test_character_semantic_call_budget_is_bounded() -> None:
    director = QwenDirector(ROOT)
    director._character_semantic_calls = 2
    failed = False
    try:
        director.adjudicate_character_entities(
            "Eli entered the station.",
            ["Eli"],
            {"candidates": []},
        )
    except RuntimeError as exc:
        failed = "max 2" in str(exc)
    check(
        failed,
        "Character semantic Qwen call budget is not hard-bounded at two.",
    )


if __name__ == "__main__":
    main()

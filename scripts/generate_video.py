from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

# The validator is executed as:
#
#     python scripts/validate_project.py
#
# Python otherwise puts scripts/ on sys.path rather than
# the repository root. Add the repository root once so
# planner/, execution/, pipeline/, etc. are importable.
if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


PRODUCTION_WORKFLOWS = {
    "ref2v":
        ROOT
        / "workflows"
        / "generation"
        / "H3_Ref2V_Production.json",

    "turbo_ref2v":
        ROOT
        / "workflows"
        / "generation"
        / "H3_Turbo_Ref2V_Production.json",

    "upscale":
        ROOT
        / "workflows"
        / "postprocess"
        / "H3_Ref2V_UltimateUpscale_Production.json",
}

SOURCE_WORKFLOWS = {
    "turbo_source":
        ROOT
        / "workflows"
        / "sources"
        / "H3_Turbo_Reference_Source.json",

    "upscale_source":
        ROOT
        / "workflows"
        / "sources"
        / "H3_LatentUpscaler_Source.json",
}

LOCKED_MODELS = {
    "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
    "minimax_h3_turbo_v4_step600_ema.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_latent_upscaler_3d_fp16.safetensors",
}

FORBIDDEN_EXECUTABLE_TOKENS = {
    "minimax_h3_fl2va",
    "minimax_h3_fl2v",
    "qwen3-4b",
    "qwen3vl_32b_minimax_h3-Q4",
    "minimax_h3_ref2va_pruned_int8_convrot",
    "minimax_h3_video_vae_int8_convrot",
    "minimax_h3_ref2v_turbo_4step",
}

REQUIRED_CUSTOM_NODES = {
    "Comfyui_Minimax_h3_latent_Upscaler",
    "Comfyui-MMH3-UltimateUpscale",
    "ComfyUI-MiniMax-H3-Turbo",
    "ComfyUI-Workflow-To-API-Converter",
    "ComfyUI-VideoHelperSuite",
}


def fail(
    message: str,
):
    raise RuntimeError(
        message
    )


def load_json(
    path: Path,
):

    if not path.is_file():
        fail(
            f"Missing workflow: {path}"
        )

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        fail(
            f"Invalid JSON in {path}: {exc}"
        )

    if not isinstance(
        data,
        dict,
    ):
        fail(
            f"Workflow root must be an object: {path}"
        )

    if not isinstance(
        data.get(
            "nodes"
        ),
        list,
    ):
        fail(
            f"Workflow has no node list: {path}"
        )

    return data


def node_types(
    graph: dict,
):
    return {
        str(
            node.get(
                "type"
            )
        )
        for node in graph.get(
            "nodes",
            [],
        )
        if isinstance(
            node,
            dict,
        )
        and node.get(
            "type"
        )
    }


def executable_values(
    graph: dict,
):

    executable = {
        "UNETLoader",
        "CLIPLoader",
        "CLIPLoaderGGUF",
        "VAELoader",
        "MiniMaxH3TurboLoRA",
        "MMH3LatentUpscaleWithModelParams",
    }

    values = []

    for node in graph.get(
        "nodes",
        [],
    ):

        if node.get(
            "type"
        ) not in executable:
            continue

        for value in node.get(
            "widgets_values",
            [],
        ):

            if isinstance(
                value,
                str,
            ):
                values.append(
                    value
                )

    return values


def validate_python():

    for path in ROOT.rglob(
        "*.py"
    ):

        if (
            ".git"
            in path.parts
            or "ComfyUI"
            in path.parts
        ):
            continue

        try:
            ast.parse(
                path.read_text(
                    encoding="utf-8"
                ),
                filename=str(
                    path
                ),
            )
        except SyntaxError as exc:
            fail(
                f"Python syntax error in {path}: {exc}"
            )

    print(
        "PASS Python syntax"
    )


def validate_workflows():

    workflows = {
        **PRODUCTION_WORKFLOWS,
        **SOURCE_WORKFLOWS,
    }

    for name, path in workflows.items():

        load_json(
            path
        )

        print(
            "PASS workflow:",
            name,
        )


def validate_ref2v():

    graph = load_json(
        PRODUCTION_WORKFLOWS[
            "ref2v"
        ]
    )

    required = {
        "UNETLoader",
        "CLIPLoader",
        "MiniMaxH3ReferenceToVideo",
        "BasicGuider",
        "BasicScheduler",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    }

    missing = (
        required
        - node_types(
            graph
        )
    )

    if missing:
        fail(
            "Ref2V missing nodes: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    values = executable_values(
        graph
    )

    for filename in (
        "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors",
        "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    ):

        if filename not in values:
            fail(
                f"Ref2V missing locked model: {filename}"
            )

    print(
        "PASS Ref2V contract"
    )


def validate_turbo():

    graph = load_json(
        PRODUCTION_WORKFLOWS[
            "turbo_ref2v"
        ]
    )

    types = node_types(
        graph
    )

    required = {
        "UNETLoader",
        "CLIPLoader",
        "MiniMaxH3ReferenceToVideo",
        "BasicGuider",
        "BasicScheduler",
        "SamplerCustomAdvanced",
        "MiniMaxH3TurboLoRA",
        "MiniMaxH3TurboSampler",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    }

    missing = (
        required
        - types
    )

    if missing:
        fail(
            "Turbo missing nodes: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    loras = [
        node
        for node in graph[
            "nodes"
        ]
        if node.get(
            "type"
        )
        == "MiniMaxH3TurboLoRA"
    ]

    if len(loras) != 1:
        fail(
            "Turbo must contain exactly one Turbo LoRA node."
        )

    values = loras[0].get(
        "widgets_values",
        [],
    )

    if (
        not values
        or values[0]
        != "minimax_h3_turbo_v4_step600_ema.safetensors"
    ):
        fail(
            "Turbo is not using Step600."
        )

    schedulers = [
        node
        for node in graph[
            "nodes"
        ]
        if node.get(
            "type"
        )
        == "BasicScheduler"
    ]

    if len(schedulers) != 1:
        fail(
            "Turbo must contain exactly one BasicScheduler."
        )

    widgets = schedulers[0].get(
        "widgets_values",
        [],
    )

    if (
        len(widgets) < 3
        or widgets[1] != 8
        or widgets[2] != 1
    ):
        fail(
            "Turbo scheduler is not locked to 8 steps."
        )

    print(
        "PASS Turbo 8-step contract"
    )


def validate_upscale():

    graph = load_json(
        PRODUCTION_WORKFLOWS[
            "upscale"
        ]
    )

    required = {
        "MMH3LatentUpscaleWithModelParams",
        "MMH3TemporalSplitParams",
        "MMH3SpatialSplitParams",
        "MMH3UltimateUpscale",
    }

    missing = (
        required
        - node_types(
            graph
        )
    )

    if missing:
        fail(
            "Upscale workflow missing nodes: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    text = json.dumps(
        graph
    )

    if (
        "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        not in text
    ):
        fail(
            "Upscale workflow is missing the locked "
            "3D H3 upscaler."
        )

    print(
        "PASS Ultimate Upscale contract"
    )


def validate_models():

    for name, path in PRODUCTION_WORKFLOWS.items():

        graph = load_json(
            path
        )

        values = executable_values(
            graph
        )

        for value in values:

            lowered = value.lower()

            for token in (
                FORBIDDEN_EXECUTABLE_TOKENS
            ):

                if (
                    token.lower()
                    in lowered
                ):
                    fail(
                        f"Forbidden executable "
                        f"asset '{token}' in {name}"
                    )

            if (
                value.endswith(
                    ".safetensors"
                )
                and value
                not in LOCKED_MODELS
            ):
                fail(
                    f"Unapproved executable "
                    f"model '{value}' in {name}"
                )

    print(
        "PASS locked executable model inventory"
    )


def validate_config():

    from planner.config import (
        DELIVERY_HEIGHT,
        DELIVERY_WIDTH,
        H3_HEIGHT,
        H3_STEPS,
        H3_WIDTH,
        H3_REF_IMAGE_SIZE,
        TURBO_STEPS,
        UPSCALE_HEIGHT,
        UPSCALE_WIDTH,
    )

    assert (
        H3_WIDTH,
        H3_HEIGHT,
    ) == (
        1344,
        768,
    )

    assert (
        H3_STEPS,
        TURBO_STEPS,
    ) == (
        20,
        8,
    )

    assert (
        H3_REF_IMAGE_SIZE
        == "match"
    )

    assert (
        UPSCALE_WIDTH,
        UPSCALE_HEIGHT,
    ) == (
        1920,
        1088,
    )

    assert (
        DELIVERY_WIDTH,
        DELIVERY_HEIGHT,
    ) == (
        1280,
        720,
    )

    print(
        "PASS centralized runtime configuration"
    )


def validate_runtime_imports():

    modules = [
        "planner.production_planner",
        "planner.qwen_director",
        "pipeline.production_orchestrator",
        "pipeline.identity_anchor_store",
        "pipeline.h3_scene_continuity",
        "execution.h3_workflow_builder",
        "execution.h3_upscaled_workflow_builder",
        "execution.shot_executor",
        "execution.production_runner",
        "execution.h3_runtime",
        "execution.assembly_manager",
        "ui.storyboard_server",
    ]

    for module in modules:

        importlib.import_module(
            module
        )

    print(
        "PASS production runtime imports"
    )


def validate_upscale_builder():

    from execution.h3_upscaled_workflow_builder import (
        H3UpscaledWorkflowBuilder,
    )

    assert hasattr(
        H3UpscaledWorkflowBuilder,
        "build_upscaled",
    )

    print(
        "PASS combined H3 upscale builder"
    )


def validate_storyboard_ui():

    html = (
        ROOT
        / "ui"
        / "storyboard.html"
    )

    server = (
        ROOT
        / "ui"
        / "storyboard_server.py"
    )

    if not html.is_file():
        fail(
            "Storyboard HTML missing."
        )

    if not server.is_file():
        fail(
            "Storyboard server missing."
        )

    text = html.read_text(
        encoding="utf-8"
    )

    required_ui_tokens = [
        "Approve Storyboard",
        "Save Draft",
        "characters",
        "scenes",
        "shots",
        "camera_shot",
        "camera_movement",
        "lighting",
        "overall_soundscape",
    ]

    for token in required_ui_tokens:

        if token not in text:
            fail(
                "Storyboard UI missing token: "
                + token
            )

    print(
        "PASS interactive storyboard UI"
    )


def validate_cleanup():

    for path in ROOT.rglob(
        "*_TEST.json"
    ):

        if path.is_file():
            fail(
                f"Temporary TEST workflow remains: {path}"
            )

    print(
        "PASS repository cleanup"
    )


def main():

    validate_python()
    validate_workflows()
    validate_ref2v()
    validate_turbo()
    validate_upscale()
    validate_models()
    validate_config()
    validate_runtime_imports()
    validate_upscale_builder()
    validate_storyboard_ui()
    validate_cleanup()

    print(
        "=" * 80
    )

    print(
        "MiniMax H3 PROJECT VALIDATION PASSED."
    )


if __name__ == "__main__":
    main()

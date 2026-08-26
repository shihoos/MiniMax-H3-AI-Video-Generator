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

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )


REQUIRED_FILES = [
    # Configuration
    "configs/runtime_versions.yaml",
    "configs/model_inventory.yaml",
    "configs/custom_nodes.yaml",

    # Planner
    "planner/config.py",
    "planner/production_planner.py",
    "planner/qwen_director.py",

    # Pipeline
    "pipeline/production_orchestrator.py",
    "pipeline/reference_manager.py",
    "pipeline/identity_continuity.py",
    "pipeline/h3_reference_binding.py",
    "pipeline/h3_scene_continuity.py",
    "pipeline/identity_anchor_store.py",

    # Execution
    "execution/assembly_manager.py",
    "execution/checkpoint_manager.py",
    "execution/comfy_client.py",
    "execution/h3_runtime.py",
    "execution/h3_workflow_builder.py",
    "execution/h3_upscaled_workflow_builder.py",
    "execution/production_runner.py",
    "execution/shot_executor.py",

    # Scheduler
    "scheduler/gpu_scheduler.py",

    # Schemas
    "schemas/character.py",
    "schemas/scene.py",
    "schemas/shot.py",
    "schemas/parser.py",

    # Kaggle
    "kaggle/bootstrap.py",
    "kaggle/preflight_h3.py",
    "kaggle/start_comfyui.py",
    "kaggle/verify_live_runtime.py",

    # UI / scripts
    "ui/storyboard_gradio.py",
    "scripts/generate_video.py",
    "scripts/validate_reference_wiring.py",

    # Workflows
    "workflows/generation/H3_Ref2V_Production.json",
    "workflows/generation/H3_Turbo_Ref2V_Production.json",
    "workflows/postprocess/H3_Ref2V_UltimateUpscale_Production.json",
    "workflows/sources/H3_Turbo_Reference_Source.json",
    "workflows/sources/H3_LatentUpscaler_Source.json",
]


RUNTIME_IMPORTS = [
    "planner.production_planner",
    "planner.qwen_director",

    "pipeline.production_orchestrator",
    "pipeline.reference_manager",
    "pipeline.identity_continuity",
    "pipeline.h3_reference_binding",
    "pipeline.h3_scene_continuity",
    "pipeline.identity_anchor_store",

    "execution.assembly_manager",
    "execution.comfy_client",
    "execution.h3_runtime",
    "execution.h3_workflow_builder",
    "execution.h3_upscaled_workflow_builder",
    "execution.production_runner",
    "execution.shot_executor",

    "scheduler.gpu_scheduler",

    "ui.storyboard_gradio",
]


PRODUCTION_WORKFLOWS = {
    "ref2v": (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Ref2V_Production.json"
    ),
    "turbo_ref2v": (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Turbo_Ref2V_Production.json"
    ),
    "upscale": (
        ROOT
        / "workflows"
        / "postprocess"
        / "H3_Ref2V_UltimateUpscale_Production.json"
    ),
}


SOURCE_WORKFLOWS = {
    "turbo_source": (
        ROOT
        / "workflows"
        / "sources"
        / "H3_Turbo_Reference_Source.json"
    ),
    "upscale_source": (
        ROOT
        / "workflows"
        / "sources"
        / "H3_LatentUpscaler_Source.json"
    ),
}


LOCKED_MODELS = {
    "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
    "minimax_h3_turbo_v4_step600_ema.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "minimax_h3_latent_upscaler_3d_fp16.safetensors",
}


FORBIDDEN_MODEL_TOKENS = {
    "minimax_h3_fl2va",
    "minimax_h3_fl2v",
    "qwen3-4b",
    "qwen3vl_32b_minimax_h3-Q4",
    "minimax_h3_ref2va_pruned_int8_convrot",
    "minimax_h3_video_vae_int8_convrot",
    "minimax_h3_ref2v_turbo_4step",
}


def fail(
    message: str,
) -> None:

    raise RuntimeError(
        message
    )


def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        fail(
            message
        )


def load_json(
    path: Path,
) -> dict:

    require(
        path.is_file(),
        f"Missing JSON workflow: {path}",
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

    require(
        isinstance(
            data,
            dict,
        ),
        f"Workflow root must be an object: {path}",
    )

    require(
        isinstance(
            data.get("nodes"),
            list,
        ),
        f"Workflow has no node list: {path}",
    )

    return data


def node_types(
    graph: dict,
) -> set[str]:

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
        if (
            isinstance(
                node,
                dict,
            )
            and node.get(
                "type"
            )
        )
    }


def executable_model_values(
    graph: dict,
) -> list[str]:

    executable_nodes = {
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
        ) not in executable_nodes:

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


def validate_files() -> None:

    for relative in REQUIRED_FILES:

        path = (
            ROOT
            / relative
        )

        require(
            path.is_file(),
            (
                "Required repository file "
                f"is missing:\n{path}"
            ),
        )

    print(
        "PASS repository file layout"
    )


def validate_python() -> None:

    excluded = {
        ".git",
        "ComfyUI",
        "__pycache__",
        ".runtime_ltx098",
    }

    count = 0

    for path in ROOT.rglob(
        "*.py"
    ):

        relative = (
            path.relative_to(
                ROOT
            )
        )

        if any(
            part in excluded
            for part
            in relative.parts
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
                f"Python syntax error in "
                f"{path}: {exc}"
            )

        count += 1

    require(
        count > 0,
        "No Python files found.",
    )

    print(
        f"PASS Python syntax ({count} files)"
    )


def validate_workflows() -> None:

    for name, path in {
        **PRODUCTION_WORKFLOWS,
        **SOURCE_WORKFLOWS,
    }.items():

        load_json(
            path
        )

        print(
            "PASS workflow:",
            name,
        )

    ref2v = load_json(
        PRODUCTION_WORKFLOWS[
            "ref2v"
        ]
    )

    ref2v_required = {
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
        ref2v_required
        - node_types(
            ref2v
        )
    )

    require(
        not missing,
        "Ref2V workflow missing nodes: "
        + ", ".join(
            sorted(
                missing
            )
        ),
    )

    turbo = load_json(
        PRODUCTION_WORKFLOWS[
            "turbo_ref2v"
        ]
    )

    turbo_required = {
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
        turbo_required
        - node_types(
            turbo
        )
    )

    require(
        not missing,
        "Turbo workflow missing nodes: "
        + ", ".join(
            sorted(
                missing
            )
        ),
    )

    turbo_text = json.dumps(
        turbo
    )

    require(
        "minimax_h3_turbo_v4_step600_ema.safetensors"
        in turbo_text,
        "Turbo workflow is not using Step600.",
    )

    require(
        "minimax_h3_ref2v_turbo_4step"
        not in turbo_text,
        "Obsolete 4-step H3 Turbo workflow detected.",
    )

    upscale = load_json(
        PRODUCTION_WORKFLOWS[
            "upscale"
        ]
    )

    upscale_required = {
        "MMH3LatentUpscaleWithModelParams",
        "MMH3TemporalSplitParams",
        "MMH3SpatialSplitParams",
        "MMH3UltimateUpscale",
    }

    missing = (
        upscale_required
        - node_types(
            upscale
        )
    )

    require(
        not missing,
        "Upscale workflow missing nodes: "
        + ", ".join(
            sorted(
                missing
            )
        ),
    )

    require(
        "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        in json.dumps(
            upscale
        ),
        (
            "Upscale workflow is missing "
            "the 3D H3 upscaler."
        ),
    )

    print(
        "PASS workflow contracts"
    )


def validate_model_inventory() -> None:

    for name, path in {
        **PRODUCTION_WORKFLOWS,
        **SOURCE_WORKFLOWS,
    }.items():

        graph = load_json(
            path
        )

        values = (
            executable_model_values(
                graph
            )
        )

        for value in values:

            lowered = (
                value.lower()
            )

            for forbidden in (
                FORBIDDEN_MODEL_TOKENS
            ):

                require(
                    forbidden.lower()
                    not in lowered,
                    (
                        f"Forbidden model token "
                        f"'{forbidden}' detected "
                        f"in {name}."
                    ),
                )

            if value.endswith(
                ".safetensors"
            ):

                require(
                    value in LOCKED_MODELS,
                    (
                        f"Unapproved executable "
                        f"model '{value}' detected "
                        f"in {name}."
                    ),
                )

    print(
        "PASS executable model inventory"
    )


def validate_config() -> None:

    from planner.config import (
        DELIVERY_FPS,
        DELIVERY_HEIGHT,
        DELIVERY_WIDTH,
        H3_HEIGHT,
        H3_REF_IMAGE_SIZE,
        H3_STEPS,
        H3_WIDTH,
        TURBO_STEPS,
        UPSCALE_HEIGHT,
        UPSCALE_WIDTH,
    )

    require(
        (
            H3_WIDTH,
            H3_HEIGHT,
        )
        == (
            1344,
            768,
        ),
        (
            "H3 generation resolution "
            "is not 1344x768."
        ),
    )

    require(
        (
            H3_STEPS,
            TURBO_STEPS,
        )
        == (
            20,
            8,
        ),
        (
            "H3/Turbo step configuration "
            "is incorrect."
        ),
    )

    require(
        H3_REF_IMAGE_SIZE == "match",
        (
            "H3 reference image policy "
            "must be 'match'."
        ),
    )

    require(
        (
            UPSCALE_WIDTH,
            UPSCALE_HEIGHT,
        )
        == (
            1920,
            1088,
        ),
        "Upscale dimensions are incorrect.",
    )

    require(
        (
            DELIVERY_WIDTH,
            DELIVERY_HEIGHT,
        )
        == (
            1280,
            720,
        ),
        "Delivery dimensions are incorrect.",
    )

    require(
        DELIVERY_FPS == 24,
        "Delivery FPS must be 24.",
    )

    print(
        "PASS centralized runtime configuration"
    )


def validate_runtime_imports() -> None:

    for module_name in (
        RUNTIME_IMPORTS
    ):

        importlib.import_module(
            module_name
        )

    print(
        "PASS runtime imports"
    )


def validate_gradio_ui() -> None:

    path = (
        ROOT
        / "ui"
        / "storyboard_gradio.py"
    )

    require(
        path.is_file(),
        f"Missing Gradio UI: {path}",
    )

    text = path.read_text(
        encoding="utf-8"
    )

    required_tokens = (
        "ProductionController",
        "generate_storyboard",
        "approve_and_generate",
        "Your Story",
        "AI Story",
        "Expand Story",
        "Preserve Story",
        "Generate Storyboard",
        "Approve & Generate Video",
        "share=True",
        "H3_DIRECTOR_ENABLED",
        "ProductionRunner",
        "H3Runtime",
        "check_worker",
    )

    for token in required_tokens:

        require(
            token in text,
            (
                "Gradio UI missing "
                f"contract token: {token}"
            ),
        )

    require(
        "def build_app(" in text,
        "Gradio UI build_app() missing.",
    )

    require(
        "def serve_storyboard_gradio(" in text,
        (
            "Gradio UI "
            "serve_storyboard_gradio() missing."
        ),
    )

    print(
        "PASS Gradio UI contract"
    )


def validate_reference_wiring() -> None:

    path = (
        ROOT
        / "scripts"
        / "validate_reference_wiring.py"
    )

    require(
        path.is_file(),
        "Reference wiring validator missing.",
    )

    text = path.read_text(
        encoding="utf-8"
    )

    for token in (
        "copy_input",
        "_add_load_image",
        "Production isolation wiring PASSED",
    ):

        require(
            token in text,
            (
                "Reference wiring validator "
                f"missing: {token}"
            ),
        )

    print(
        "PASS reference path validator"
    )


def validate_plan_persistence_boundary() -> None:

    orchestrator_path = (
        ROOT
        / "pipeline"
        / "production_orchestrator.py"
    )

    cli_path = (
        ROOT
        / "scripts"
        / "generate_video.py"
    )

    orchestrator_text = (
        orchestrator_path.read_text(
            encoding="utf-8"
        )
    )

    cli_text = (
        cli_path.read_text(
            encoding="utf-8"
        )
    )

    require(
        "PRODUCTION_DIR" not in orchestrator_text,
        (
            "ProductionOrchestrator must not "
            "own plan persistence."
        ),
    )

    require(
        "production_plan_path"
        not in orchestrator_text,
        (
            "ProductionOrchestrator must not "
            "expose a persisted plan path."
        ),
    )

    require(
        "story_preview.json"
        not in orchestrator_text,
        (
            "ProductionOrchestrator must not "
            "write the legacy global "
            "story_preview.json."
        ),
    )

    require(
        "def create_cli_plan_path("
        in cli_text,
        (
            "CLI plan persistence helper "
            "is missing."
        ),
    )

    require(
        "create_cli_plan_path("
        in cli_text,
        (
            "CLI must own creation "
            "of its plan path."
        ),
    )

    require(
        "production_id"
        in cli_text,
        (
            "CLI must persist plans "
            "under a production ID."
        ),
    )

    print(
        "PASS plan persistence boundary"
    )


def validate_cleanup() -> None:

    for path in ROOT.rglob(
        "*_TEST.json"
    ):

        if path.is_file():

            fail(
                (
                    "Temporary workflow artifact "
                    f"remains: {path}"
                )
            )

    print(
        "PASS repository cleanup"
    )


def main() -> None:

    validate_files()
    validate_python()
    validate_workflows()
    validate_model_inventory()
    validate_config()
    validate_runtime_imports()
    validate_gradio_ui()
    validate_reference_wiring()
    validate_plan_persistence_boundary()
    validate_cleanup()

    print(
        "=" * 80
    )

    print(
        "MiniMax H3 PROJECT VALIDATION PASSED."
    )


if __name__ == "__main__":
    main()

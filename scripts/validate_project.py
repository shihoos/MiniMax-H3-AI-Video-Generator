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
    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    "configs/runtime_versions.yaml",
    "configs/model_inventory.yaml",
    "configs/custom_nodes.yaml",

    # --------------------------------------------------------
    # DEPENDENCY / PROJECT DOCUMENTATION
    # --------------------------------------------------------

    "requirements.txt",
    "README.md",

    # --------------------------------------------------------
    # PLANNER
    # --------------------------------------------------------

    "planner/config.py",
    "planner/production_planner.py",
    "planner/qwen_director.py",

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------

    "pipeline/production_orchestrator.py",
    "pipeline/reference_manager.py",
    "pipeline/identity_continuity.py",
    "pipeline/h3_scene_continuity.py",
    "pipeline/identity_anchor_store.py",

    # --------------------------------------------------------
    # EXECUTION
    # --------------------------------------------------------

    "execution/assembly_manager.py",
    "execution/comfy_client.py",
    "execution/h3_runtime.py",
    "execution/h3_workflow_builder.py",
    "execution/h3_upscaled_workflow_builder.py",
    "execution/production_runner.py",
    "execution/shot_executor.py",

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    "scheduler/gpu_scheduler.py",

    # --------------------------------------------------------
    # SCHEMAS
    # --------------------------------------------------------

    "schemas/character.py",
    "schemas/scene.py",
    "schemas/shot.py",
    "schemas/parser.py",

    # --------------------------------------------------------
    # KAGGLE
    # --------------------------------------------------------

    "kaggle/bootstrap.py",
    "kaggle/preflight_h3.py",
    "kaggle/verify_live_runtime.py",

    # --------------------------------------------------------
    # UI / SCRIPTS
    # --------------------------------------------------------

    "ui/storyboard_gradio.py",
    "scripts/generate_video.py",
    "scripts/validate_reference_wiring.py",

    # --------------------------------------------------------
    # WORKFLOWS
    # --------------------------------------------------------

    "workflows/generation/H3_Ref2V_Production.json",
    "workflows/generation/H3_Turbo_Ref2V_Production.json",
    "workflows/postprocess/H3_Ref2V_UltimateUpscale_Production.json",
    "workflows/sources/H3_Turbo_Reference_Source.json",
    "workflows/sources/H3_LatentUpscaler_Source.json",
]


RUNTIME_IMPORTS = [
    # Planner
    "planner.production_planner",
    "planner.qwen_director",

    # Pipeline
    "pipeline.production_orchestrator",
    "pipeline.reference_manager",
    "pipeline.identity_continuity",
    "pipeline.h3_scene_continuity",
    "pipeline.identity_anchor_store",

    # Execution
    "execution.h3_workflow_builder",
    "execution.h3_upscaled_workflow_builder",
    "execution.shot_executor",
    "execution.production_runner",
    "execution.h3_runtime",
    "execution.assembly_manager",

    # Scheduler
    "scheduler.gpu_scheduler",

    # UI
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
        f"Missing JSON workflow:\n{path}",
    )

    try:

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:

        fail(
            f"Invalid JSON in {path}:\n{exc}"
        )

    require(
        isinstance(
            data,
            dict,
        ),
        f"Workflow root must be an object:\n{path}",
    )

    require(
        isinstance(
            data.get(
                "nodes"
            ),
            list,
        ),
        f"Workflow has no node list:\n{path}",
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

        if (
            not isinstance(
                node,
                dict,
            )
        ):
            continue

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


def validate_dependency_manifests() -> None:
    base = ROOT / "requirements.txt"

    require(
        base.is_file(),
        f"Dependency manifest is missing: {base}",
    )

    base_text = base.read_text(encoding="utf-8")

    require(
        "PyYAML==" in base_text,
        "requirements.txt must pin PyYAML.",
    )

    require(
        "gradio==" in base_text,
        "requirements.txt must pin Gradio.",
    )

    legacy = ROOT / "requirements-kaggle.txt"
    require(
        not legacy.exists(),
        "requirements-kaggle.txt is obsolete; Kaggle dependencies are controlled by configs/runtime_versions.yaml and kaggle/bootstrap.py.",
    )

    print("PASS dependency manifests")

def validate_files() -> None:

    for relative in REQUIRED_FILES:

        path = (
            ROOT
            / relative
        )

        require(
            path.is_file(),
            (
                "Required repository file is missing:\n"
                f"{path}"
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
    }

    count = 0

    for path in ROOT.rglob(
        "*.py"
    ):

        relative_parts = (
            path.relative_to(
                ROOT
            ).parts
        )

        if any(
            part in excluded
            for part
            in relative_parts
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
                "Python syntax error:\n"
                f"{path}\n"
                f"{exc}"
            )

        count += 1

    require(
        count > 0,
        "No Python files found.",
    )

    print(
        f"PASS Python syntax ({count} files)"
    )


def validate_workflow_graph_integrity(graph: dict, name: str) -> None:
    nodes = {
        int(node["id"]): node
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and "id" in node
    }

    links = {}
    for row in graph.get("links", []):
        require(
            isinstance(row, list) and len(row) >= 6,
            f"{name}: malformed workflow link: {row!r}",
        )
        link_id = int(row[0])
        require(
            link_id not in links,
            f"{name}: duplicate workflow link id {link_id}.",
        )
        links[link_id] = row

    for node_id, node in nodes.items():
        for slot, item in enumerate(node.get("inputs", []) or []):
            require(isinstance(item, dict), f"{name}: node {node_id} input {slot} is not an object.")
            link_id = item.get("link")
            if link_id is None:
                continue
            link_id = int(link_id)
            require(link_id in links, f"{name}: node {node_id} input {slot} references missing link {link_id}.")
            row = links[link_id]
            require(int(row[3]) == node_id and int(row[4]) == slot, f"{name}: input link {link_id} does not point to node {node_id}:{slot}.")

        for slot, item in enumerate(node.get("outputs", []) or []):
            require(isinstance(item, dict), f"{name}: node {node_id} output {slot} is not an object.")
            for link_id in item.get("links") or []:
                link_id = int(link_id)
                require(link_id in links, f"{name}: node {node_id} output {slot} references missing link {link_id}.")
                row = links[link_id]
                require(int(row[1]) == node_id and int(row[2]) == slot, f"{name}: output link {link_id} does not point from node {node_id}:{slot}.")

    for link_id, row in links.items():
        source_id, source_slot = int(row[1]), int(row[2])
        target_id, target_slot = int(row[3]), int(row[4])
        require(source_id in nodes, f"{name}: link {link_id} references missing source node {source_id}.")
        require(target_id in nodes, f"{name}: link {link_id} references missing destination node {target_id}.")
        require(source_slot < len(nodes[source_id].get("outputs", []) or []), f"{name}: link {link_id} has invalid source slot {source_slot}.")
        require(target_slot < len(nodes[target_id].get("inputs", []) or []), f"{name}: link {link_id} has invalid target slot {target_slot}.")


def validate_workflows() -> None:

    all_workflows = {
        **PRODUCTION_WORKFLOWS,
        **SOURCE_WORKFLOWS,
    }

    for name, path in all_workflows.items():

        graph = load_json(
            path
        )

        validate_workflow_graph_integrity(
            graph,
            name,
        )

        print(
            "PASS workflow:",
            name,
        )

    # --------------------------------------------------------
    # REF2V
    # --------------------------------------------------------

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
        "Ref2V workflow is missing nodes:\n"
        + "\n".join(
            sorted(
                missing
            )
        ),
    )

    # --------------------------------------------------------
    # TURBO REF2V
    # --------------------------------------------------------

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
        "Turbo workflow is missing nodes:\n"
        + "\n".join(
            sorted(
                missing
            )
        ),
    )

    turbo_text = json.dumps(
        turbo
    )

    require(
        (
            "minimax_h3_turbo_v4_step600_ema.safetensors"
            in turbo_text
        ),
        "Turbo workflow is not using the locked Step600 LoRA.",
    )

    

    # --------------------------------------------------------
    # UPSCALE
    # --------------------------------------------------------

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
        "Upscale workflow is missing nodes:\n"
        + "\n".join(
            sorted(
                missing
            )
        ),
    )

    require(
        (
            "minimax_h3_latent_upscaler_3d_fp16.safetensors"
            in json.dumps(
                upscale
            )
        ),
        "Upscale workflow is missing the locked H3 3D upscaler.",
    )

    print(
        "PASS workflow contracts"
    )


def validate_model_inventory() -> None:

    all_workflows = {
        **PRODUCTION_WORKFLOWS,
        **SOURCE_WORKFLOWS,
    }

    for name, path in all_workflows.items():

        graph = load_json(
            path
        )

        values = (
            executable_model_values(
                graph
            )
        )

        for value in values:

            lowered = value.lower()

            
            if value.endswith(
                ".safetensors"
            ):

                require(
                    value in LOCKED_MODELS,
                    (
                        f"Unapproved executable model "
                        f"'{value}' found in {name}."
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
        "H3 generation resolution must be 1344x768.",
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
        "H3/Turbo step configuration is incorrect.",
    )

    require(
        H3_REF_IMAGE_SIZE == "match",
        "H3 reference image policy must be 'match'.",
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
        "Upscale resolution must be 1920x1088.",
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
        "Delivery resolution must be 1280x720.",
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

        try:

            importlib.import_module(
                module_name
            )

        except Exception as exc:

            fail(
                "Runtime import failed:\n"
                f"{module_name}\n"
                f"{type(exc).__name__}: {exc}"
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
        "Gradio UI file is missing.",
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
        "H3_DIRECTOR_ENABLED",
        "ProductionRunner",
        "H3Runtime",
        "check_worker",
        "storyboard_share_enabled",
    )

    for token in required_tokens:

        require(
            token in text,
            (
                "Gradio UI is missing required "
                f"contract token: {token}"
            ),
        )

    require(
        "def build_app(" in text,
        "Gradio build_app() is missing.",
    )

    require(
        "def serve_storyboard_gradio(" in text,
        "Gradio serve_storyboard_gradio() is missing.",
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
        "Reference wiring validator is missing.",
    )

    text = path.read_text(
        encoding="utf-8"
    )

    required_tokens = (
        "copy_input",
        "_add_load_image",
        "Production isolation wiring PASSED",
    )

    for token in required_tokens:

        require(
            token in text,
            (
                "Reference wiring validator is missing "
                f"required token: {token}"
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
        "def create_production_plan("
        in orchestrator_text,
        "ProductionOrchestrator plan method is missing.",
    )

    require(
        "return plan"
        in orchestrator_text,
        "ProductionOrchestrator must return the production plan.",
    )

    require(
        "def create_cli_plan_path("
        in cli_text,
        "CLI plan persistence helper is missing.",
    )

    require(
        "story_preview.json"
        in cli_text,
        "CLI plan filename is missing.",
    )

    require(
        "data"
        in cli_text
        and "production"
        in cli_text,
        "CLI production storage path is missing.",
    )

    require(
        "save_plan("
        in cli_text,
        "CLI save_plan() call is missing.",
    )

    print(
        "PASS plan persistence boundary"
    )



def validate_production_templates() -> None:
    for name, path in PRODUCTION_WORKFLOWS.items():
        graph = load_json(path)
        for node in graph.get("nodes", []):
            if node.get("type") != "LoadImage":
                continue
            widgets = node.get("widgets_values", [])
            filename = widgets[0] if isinstance(widgets, list) and widgets else ""
            require(
                not (isinstance(filename, str) and filename.startswith("__H3_REF_IMAGE_")),
                f"{name}: contains a runtime placeholder reference {filename!r}.",
            )
    print("PASS production reference templates")


def validate_examples() -> None:
    path = ROOT / "examples" / "production_config.json"
    require(path.is_file(), "examples/production_config.json is missing.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Invalid production example: {exc}")
    require(isinstance(data, dict), "Production example must be a JSON object.")
    for key in ("story_mode", "workflow_mode", "profile", "upscale_enabled", "story"):
        require(key in data, f"Production example is missing {key}.")
    require(data["story_mode"] in {"ai_story", "expand_user_story", "preserve_user_story"}, "Example story_mode is invalid.")
    require(data["workflow_mode"] in {"auto", "ref2v", "turbo_ref2v", "upscale"}, "Example workflow_mode is invalid.")
    require(data["profile"] in {"base", "turbo", "upscale"}, "Example profile is invalid.")
    require(isinstance(data["upscale_enabled"], bool), "Example upscale_enabled must be boolean.")
    require(bool(str(data["story"]).strip()), "Example story cannot be empty.")

    stale_fields = {
        "use_character_references",
        "use_general_references",
        "use_audio",
        "use_music",
        "use_ic_lora_detailer",
        "use_ltx_spatial_upscaler",
        "multi_gpu_mode",
    }
    stale = stale_fields & set(data)
    require(
        not stale,
        "Production example contains obsolete fields: "
        + ", ".join(sorted(stale)),
    )

    print("PASS production example configuration")


def validate_ui_security_defaults() -> None:
    text = (ROOT / "planner" / "config.py").read_text(encoding="utf-8")
    section = text[text.index("def storyboard_share_enabled"):text.index("# ============================================================\n# BASIC RUNTIME VALIDATION")]
    require('        "0",' in section, "Gradio public sharing must be disabled by default.")
    print("PASS UI security defaults")


def main() -> None:

    validate_files()
    validate_dependency_manifests()
    validate_examples()
    validate_python()
    validate_workflows()
    validate_production_templates()
    validate_model_inventory()
    validate_config()
    validate_runtime_imports()
    validate_gradio_ui()
    validate_reference_wiring()
    validate_plan_persistence_boundary()
    validate_ui_security_defaults()

    
    print(
        "=" * 80
    )

    print(
        "MiniMax H3 PROJECT VALIDATION PASSED."
    )


if __name__ == "__main__":
    main()

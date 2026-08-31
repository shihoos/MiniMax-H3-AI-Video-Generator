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
    "execution/metrics.py",

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
    "schemas/dialogue.py",
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
    "pipeline.storyboard_reference_builder",
    "pipeline.continuity_ledger",
    "pipeline.dialogue_timeline",
    "pipeline.dialogue_duration",
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
    "execution.metrics",

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


def validate_upscale_audio_semantics() -> None:
    """Verify the H3 upscale graph preserves the original audio-bearing latent."""
    graph = load_json(PRODUCTION_WORKFLOWS["upscale"])
    nodes = {int(node["id"]): node for node in graph.get("nodes", [])}
    links = {int(row[0]): row for row in graph.get("links", []) if isinstance(row, list) and len(row) >= 6}

    def find_node(node_type: str):
        for node in nodes.values():
            if node.get("type") == node_type:
                return node
        raise RuntimeError(f"Upscale workflow is missing {node_type}.")

    sampler = find_node("SamplerCustomAdvanced")
    video = find_node("VAEDecode")
    audio = find_node("VAEDecodeAudio")
    upscale = find_node("MMH3UltimateUpscale")

    def input_link(node, name: str):
        for item in node.get("inputs", []):
            if item.get("name") == name:
                link_id = item.get("link")
                return links.get(int(link_id)) if link_id is not None else None
        raise RuntimeError(f"{node.get('type')} is missing input {name}.")

    video_edge = input_link(video, "samples")
    audio_edge = input_link(audio, "samples")
    require(video_edge is not None, "Upscale VAEDecode.samples is not connected.")
    require(audio_edge is not None, "Upscale VAEDecodeAudio.samples is not connected.")
    require(int(video_edge[1]) == int(upscale["id"]), "Upscale VAEDecode must consume MMH3UltimateUpscale output.")
    require(int(audio_edge[1]) == int(sampler["id"]), "Upscale VAEDecodeAudio must consume the original SamplerCustomAdvanced latent.")
    require(int(audio_edge[1]) != int(upscale["id"]), "Upscale audio must never consume MMH3UltimateUpscale output.")
    print("PASS upscale audio semantic wiring")


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


def validate_ui_share_configuration() -> None:
    text = (ROOT / "planner" / "config.py").read_text(encoding="utf-8")

    require(
        "GRADIO_SHARE_ENV = \"H3_GRADIO_SHARE\"" in text,
        "Gradio share environment variable is missing.",
    )

    start = text.index("def storyboard_share_enabled")
    end = text.index(
        "# ============================================================\n# BASIC RUNTIME VALIDATION"
    )
    section = text[start:end]

    require(
        '        "1",' in section,
        "Gradio public sharing must remain enabled by default for the configured remote-access workflow.",
    )

    import os

    from planner.config import storyboard_share_enabled

    previous = os.environ.get("H3_GRADIO_SHARE")
    try:
        os.environ["H3_GRADIO_SHARE"] = "1"
        require(
            storyboard_share_enabled() is True,
            "H3_GRADIO_SHARE=1 must enable Gradio sharing.",
        )

        os.environ["H3_GRADIO_SHARE"] = "0"
        require(
            storyboard_share_enabled() is False,
            "H3_GRADIO_SHARE=0 must disable Gradio sharing.",
        )
    finally:
        if previous is None:
            os.environ.pop("H3_GRADIO_SHARE", None)
        else:
            os.environ["H3_GRADIO_SHARE"] = previous

    print("PASS Gradio share configuration")



def validate_execution_integration() -> None:
    """Validate runtime cross-file contracts beyond syntax and static graph checks."""
    metrics_path = ROOT / "execution" / "metrics.py"
    shot_path = ROOT / "execution" / "shot_executor.py"
    runner_path = ROOT / "execution" / "production_runner.py"
    scheduler_path = ROOT / "scheduler" / "gpu_scheduler.py"

    metrics_text = metrics_path.read_text(encoding="utf-8")
    require("class MetricsRecorder" in metrics_text, "MetricsRecorder class is missing.")
    require("def record(" in metrics_text, "MetricsRecorder.record() is missing.")
    require("json.dumps(payload" in metrics_text, "Metrics recorder must serialize JSONL records.")
    require("os.fsync(handle.fileno())" in metrics_text, "Metrics recorder must durably flush records.")

    client_path = ROOT / "execution" / "comfy_client.py"
    client_text = client_path.read_text(encoding="utf-8")
    shot_text = shot_path.read_text(encoding="utf-8")
    require("self.gpu_id = int(gpu_id)" in shot_text, "ShotExecutor must store gpu_id for telemetry.")
    require("self.metrics = MetricsRecorder(Path(metrics_path))" in shot_text, "ShotExecutor must initialize MetricsRecorder when configured.")
    require("self.metrics.record(" in shot_text, "ShotExecutor must write telemetry events.")
    require("def cancel_prompt(" in client_text, "ComfyClient must provide prompt cancellation.")
    require("liveness_interval" in client_text and "max_liveness_failures" in client_text, "ComfyClient wait_for_prompt must expose liveness controls.")
    require("outputs[-1]" not in shot_text, "ShotExecutor must not select an arbitrary last video output.")
    require("expected exactly one SaveVideo node" in shot_text, "ShotExecutor must enforce a unique SaveVideo output.")

    runner_text = runner_path.read_text(encoding="utf-8")
    require("self._completed_shots_lock = threading.RLock()" in runner_text, "ProductionRunner must protect shared completed-shot state.")
    require("self._add_identity_anchors(\n                shot,\n                character_map,\n            )" in runner_text, "ProductionRunner identity-anchor call signature is incorrect.")

    scheduler_text = scheduler_path.read_text(encoding="utf-8")
    require("class GPUScheduler" in scheduler_text, "GPUScheduler class is missing.")
    require("run_independent" in scheduler_text, "GPUScheduler.run_independent() is missing.")
    require("threading.Thread" in scheduler_text, "GPUScheduler must execute independent GPU jobs concurrently.")

    print("PASS execution integration contracts")


def validate_execution_runtime_contracts() -> None:
    """Check cross-module contracts that syntax/JSON validation cannot catch."""
    import tempfile
    import inspect

    metrics_path = ROOT / "execution" / "metrics.py"
    shot_path = ROOT / "execution" / "shot_executor.py"
    runner_path = ROOT / "execution" / "production_runner.py"

    require(metrics_path.is_file(), "execution/metrics.py is missing.")
    metrics_text = metrics_path.read_text(encoding="utf-8")
    require("class MetricsRecorder" in metrics_text, "MetricsRecorder class is missing.")
    require("def record(" in metrics_text, "MetricsRecorder.record() is missing.")

    # Verify the actual ShotExecutor constructor/runtime state, not only text tokens.
    from execution.shot_executor import ShotExecutor
    from execution.metrics import MetricsRecorder

    sig = inspect.signature(ShotExecutor.__init__)
    require("gpu_id" in sig.parameters, "ShotExecutor.__init__ must accept gpu_id.")
    require("metrics_path" in sig.parameters, "ShotExecutor.__init__ must accept metrics_path.")

    class _DummyClient:
        pass

    with tempfile.TemporaryDirectory(prefix="h3-validator-") as td:
        tmp_root = Path(td)
        (tmp_root / "ComfyUI" / "input").mkdir(parents=True, exist_ok=True)
        executor = ShotExecutor(
            comfy_client=_DummyClient(),
            project_root=tmp_root,
            comfy_input_dir=tmp_root / "ComfyUI" / "input",
            gpu_id=1,
            metrics_path=tmp_root / "metrics.jsonl",
        )
        require(executor.gpu_id == 1, "ShotExecutor did not retain gpu_id.")
        require(isinstance(executor.metrics, MetricsRecorder), "ShotExecutor did not initialize MetricsRecorder.")
        executor._record("validator_smoke", check=True)
        metrics_file = tmp_root / "metrics.jsonl"
        require(metrics_file.is_file(), "ShotExecutor failed to write metrics JSONL.")
        require(metrics_file.read_text(encoding="utf-8").strip(), "Metrics JSONL record is empty.")

    # Verify the ProductionRunner call to _add_identity_anchors has the exact signature.
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    anchor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_add_identity_anchors"
    ]
    require(len(anchor_calls) == 1, "ProductionRunner must have exactly one _add_identity_anchors call in the scene path.")
    require(len(anchor_calls[0].args) == 2, "ProductionRunner _add_identity_anchors call must pass only shot and character_map.")

    runner_text = runner_path.read_text(encoding="utf-8")
    require("self._completed_shots_lock = threading.RLock()" in runner_text, "ProductionRunner shared completed-shot lock is missing.")
    require("with self._completed_shots_lock:" in runner_text, "ProductionRunner must lock shared completed-shot state.")

    print("PASS execution runtime contracts")

def main() -> None:

    validate_files()
    validate_dependency_manifests()
    validate_examples()
    validate_python()
    validate_workflows()
    validate_upscale_audio_semantics()
    validate_production_templates()
    validate_model_inventory()
    validate_config()
    validate_runtime_imports()
    validate_execution_runtime_contracts()
    validate_execution_integration()
    validate_gradio_ui()
    validate_reference_wiring()
    validate_plan_persistence_boundary()
    validate_ui_share_configuration()

    
    print(
        "=" * 80
    )

    print(
        "MiniMax H3 PROJECT VALIDATION PASSED."
    )


if __name__ == "__main__":
    main()

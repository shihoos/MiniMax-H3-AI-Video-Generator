from __future__ import annotations

import ast
import importlib
import json
import os
import re
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
    "scripts/validate_director.py",
    "scripts/validate_production_continuity.py",

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
    "planner.cinematic_compiler",

    # Pipeline
    "pipeline.production_orchestrator",
    "pipeline.reference_manager",
    "pipeline.identity_continuity",
    "pipeline.h3_scene_continuity",
    "pipeline.dialogue_duration",
    "pipeline.dialogue_timeline",
    "pipeline.continuity_ledger",
    "pipeline.storyboard_reference_builder",
    "pipeline.identity_anchor_store",

    # Schemas
    "schemas.dialogue",

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


MODEL_WIDGET_INDEX = {
    "UNETLoader": 0,
    "CLIPLoader": 0,
    "CLIPLoaderGGUF": 0,
    "VAELoader": 0,
    "MiniMaxH3TurboLoRA": 0,
    "MMH3LatentUpscaleWithModelParams": 0,
}


def executable_model_values(
    graph: dict,
    workflow_name: str = "workflow",
) -> list[tuple[str, str]]:
    """Return model-selector values and fail on malformed known loaders."""
    values: list[tuple[str, str]] = []
    seen_known_loader = False
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type", ""))
        index = MODEL_WIDGET_INDEX.get(node_type)
        if index is None:
            continue
        seen_known_loader = True
        widgets = node.get("widgets_values", [])
        require(
            isinstance(widgets, list) and index < len(widgets),
            f"{workflow_name}: {node_type} has no model-selector widget at index {index}.",
        )
        value = widgets[index]
        require(
            isinstance(value, str) and value.strip(),
            f"{workflow_name}: {node_type} has an empty model selector.",
        )
        values.append((node_type, value.strip()))
    require(
        seen_known_loader,
        f"{workflow_name}: no supported executable model loader nodes were found.",
    )
    return values


def model_basename(value: str) -> str:
    """Normalize Windows/Unix ComfyUI model selectors to their final filename."""
    return Path(value.replace("\\", "/")).name


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

    require(
        re.search(r"(?m)^Pillow==[^\s#]+\s*$", base_text) is not None
        or re.search(r"(?m)^Pillow>=[^\s#]+(?:,[^\s#]+)*\s*$", base_text) is not None,
        "requirements.txt must declare Pillow for the storyboard builder.",
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


def validate_workflow_graph_integrity(
    graph: dict,
    name: str,
) -> None:
    nodes: dict[int, dict] = {}
    for raw_node in graph.get("nodes", []):
        require(isinstance(raw_node, dict), f"{name}: workflow node is not an object: {raw_node!r}")
        raw_id = raw_node.get("id")
        try:
            node_id = int(raw_id)
        except (TypeError, ValueError):
            fail(f"{name}: workflow node id must be numeric, got {raw_id!r}.")
        require(node_id not in nodes, f"{name}: duplicate workflow node id {node_id}.")
        nodes[node_id] = raw_node

    links: dict[int, list] = {}
    for row in graph.get("links", []):
        require(isinstance(row, list) and len(row) >= 6, f"{name}: malformed workflow link: {row!r}")
        try:
            link_id = int(row[0])
        except (TypeError, ValueError):
            fail(f"{name}: workflow link id must be numeric: {row!r}")
        require(link_id not in links, f"{name}: duplicate workflow link id {link_id}.")
        links[link_id] = row

    for node_id, node in nodes.items():
        for slot, item in enumerate(node.get("inputs", []) or []):
            require(isinstance(item, dict), f"{name}: node {node_id} input {slot} is not an object.")
            link_id = item.get("link")
            if link_id is None:
                continue
            try:
                link_id = int(link_id)
            except (TypeError, ValueError):
                fail(f"{name}: node {node_id} input {slot} has non-numeric link {link_id!r}.")
            require(link_id in links, f"{name}: node {node_id} input {slot} references missing link {link_id}.")
            row = links[link_id]
            try:
                target_id, target_slot = int(row[3]), int(row[4])
            except (TypeError, ValueError):
                fail(f"{name}: malformed target in link {link_id}: {row!r}")
            require(target_id == node_id and target_slot == slot, f"{name}: input link {link_id} does not point to node {node_id}:{slot}.")

        for slot, item in enumerate(node.get("outputs", []) or []):
            require(isinstance(item, dict), f"{name}: node {node_id} output {slot} is not an object.")
            for link_id in item.get("links") or []:
                try:
                    link_id = int(link_id)
                except (TypeError, ValueError):
                    fail(f"{name}: node {node_id} output {slot} has non-numeric link {link_id!r}.")
                require(link_id in links, f"{name}: node {node_id} output {slot} references missing link {link_id}.")
                row = links[link_id]
                try:
                    source_id, source_slot = int(row[1]), int(row[2])
                except (TypeError, ValueError):
                    fail(f"{name}: malformed source in link {link_id}: {row!r}")
                require(source_id == node_id and source_slot == slot, f"{name}: output link {link_id} does not point from node {node_id}:{slot}.")

    for link_id, row in links.items():
        try:
            source_id, source_slot = int(row[1]), int(row[2])
            target_id, target_slot = int(row[3]), int(row[4])
        except (TypeError, ValueError):
            fail(f"{name}: malformed workflow link {link_id}: {row!r}")
        require(source_id in nodes, f"{name}: link {link_id} references missing source node {source_id}.")
        require(target_id in nodes, f"{name}: link {link_id} references missing destination node {target_id}.")
        require(0 <= source_slot < len(nodes[source_id].get("outputs", []) or []), f"{name}: link {link_id} has invalid source slot {source_slot}.")
        require(0 <= target_slot < len(nodes[target_id].get("inputs", []) or []), f"{name}: link {link_id} has invalid target slot {target_slot}.")



def _workflow_node_map(graph: dict, name: str) -> tuple[dict[int, dict], dict[int, list]]:
    """Build validated node/link lookup maps for semantic workflow checks."""
    nodes: dict[int, dict] = {}
    for raw_node in graph.get("nodes", []):
        require(isinstance(raw_node, dict), f"{name}: workflow node is not an object.")
        try:
            node_id = int(raw_node.get("id"))
        except (TypeError, ValueError):
            fail(f"{name}: workflow node id must be numeric.")
        require(node_id not in nodes, f"{name}: duplicate workflow node id {node_id}.")
        nodes[node_id] = raw_node
    links: dict[int, list] = {}
    for row in graph.get("links", []):
        require(isinstance(row, list) and len(row) >= 6, f"{name}: malformed workflow link: {row!r}")
        try:
            link_id = int(row[0])
        except (TypeError, ValueError):
            fail(f"{name}: workflow link id must be numeric: {row!r}")
        require(link_id not in links, f"{name}: duplicate workflow link id {link_id}.")
        links[link_id] = row
    return nodes, links


def _linked_input_source(
    graph: dict,
    target_node: dict,
    input_name: str,
    name: str,
) -> tuple[dict, list]:
    """Return source node and link row for a named linked input."""
    nodes, links = _workflow_node_map(graph, name)
    item = next(
        (value for value in target_node.get("inputs", []) or [] if value.get("name") == input_name),
        None,
    )
    require(item is not None, f"{name}: node {target_node.get('id')} has no input {input_name!r}.")
    link_id = item.get("link")
    require(link_id is not None, f"{name}: node {target_node.get('id')} input {input_name!r} is not linked.")
    try:
        link_id = int(link_id)
    except (TypeError, ValueError):
        fail(f"{name}: input {input_name!r} has non-numeric link {link_id!r}.")
    require(link_id in links, f"{name}: input {input_name!r} references missing link {link_id}.")
    row = links[link_id]
    try:
        source_id = int(row[1])
    except (TypeError, ValueError):
        fail(f"{name}: link {link_id} has invalid source node id.")
    require(source_id in nodes, f"{name}: link {link_id} references missing source node {source_id}.")
    return nodes[source_id], row


def _h3_legal_frames(seconds: float, fps: float = 24.0) -> int:
    value = max(4.0, min(15.0, float(seconds)))
    requested = round(value * fps)
    n = max(0, (requested - 5 + 16) // 17)
    return max(124, min(362, 17 * n + 5))


def validate_h3_duration_chain() -> None:
    """Validate the live H3 workflow's duration source→math→length chain."""
    expected_expr_fragments = ("a * 24", "% 17", "(5 -")
    for name in ("ref2v", "turbo_ref2v"):
        graph = load_json(PRODUCTION_WORKFLOWS[name])
        refs = [n for n in graph.get("nodes", []) if n.get("type") == "MiniMaxH3ReferenceToVideo"]
        require(len(refs) == 1, f"{name}: expected exactly one MiniMaxH3ReferenceToVideo node.")
        ref = refs[0]
        expr, _ = _linked_input_source(graph, ref, "length", name)
        require(expr.get("type") == "ComfyMathExpression", f"{name}: length must be driven by ComfyMathExpression.")
        duration_source, _ = _linked_input_source(graph, expr, "values.a", name)
        require(duration_source.get("type") == "PrimitiveFloat", f"{name}: ComfyMathExpression.a must be driven by PrimitiveFloat.")
        expr_text = str((expr.get("widgets_values_named") or {}).get("expression") or "")
        for fragment in expected_expr_fragments:
            require(fragment in expr_text, f"{name}: duration expression missing {fragment!r}.")
        print(f"PASS H3 duration chain: {name}")


def validate_h3_resolution_selector_contract() -> None:
    """Validate generation ResolutionSelector links and production target dimensions."""
    for name in ("ref2v", "turbo_ref2v"):
        graph = load_json(PRODUCTION_WORKFLOWS[name])
        selectors = [n for n in graph.get("nodes", []) if n.get("type") == "ResolutionSelector"]
        require(len(selectors) == 1, f"{name}: expected exactly one ResolutionSelector.")
        selector = selectors[0]
        widgets = selector.get("widgets_values") or []
        require(len(widgets) >= 3, f"{name}: ResolutionSelector widgets are incomplete.")
        require(widgets[0] == "16:9 (Widescreen)", f"{name}: ResolutionSelector aspect ratio must be 16:9 (Widescreen).")
        require(int(widgets[2]) == 32, f"{name}: ResolutionSelector multiple must be 32.")
        ref = next(n for n in graph.get("nodes", []) if n.get("type") == "MiniMaxH3ReferenceToVideo")
        width_source, _ = _linked_input_source(graph, ref, "width", name)
        height_source, _ = _linked_input_source(graph, ref, "height", name)
        require(width_source.get("type") == "ResolutionSelector", f"{name}: width must come from ResolutionSelector.")
        require(height_source.get("type") == "ResolutionSelector", f"{name}: height must come from ResolutionSelector.")
        ref_widgets = ref.get("widgets_values") or []
        require(len(ref_widgets) >= 3, f"{name}: ReferenceToVideo dimensions are incomplete.")
        require((int(ref_widgets[1]), int(ref_widgets[2])) == (1344, 768), f"{name}: production H3 canvas must be 1344x768.")
        print(f"PASS H3 resolution selector contract: {name}")


def validate_h3_builder_behavior() -> None:
    """Execute H3WorkflowBuilder's duration/resolution mutators on template copies."""
    from copy import deepcopy
    from execution.h3_workflow_builder import H3WorkflowBuilder

    builder = H3WorkflowBuilder(ROOT, None)
    # Duration: the PrimitiveFloat value must change; the math expression must remain unchanged.
    for name in ("ref2v", "turbo_ref2v"):
        workflow = builder.load(name)
        before_expr = str(
            next(n for n in workflow["nodes"] if n.get("type") == "ComfyMathExpression")
            .get("widgets_values_named", {})
            .get("expression", "")
        )
        builder._set_duration(workflow, 5.0)
        expr = next(n for n in workflow["nodes"] if n.get("type") == "ComfyMathExpression")
        after_expr = str((expr.get("widgets_values_named") or {}).get("expression", ""))
        duration_nodes = [n for n in workflow["nodes"] if n.get("type") == "PrimitiveFloat"]
        require(duration_nodes, f"{name}: no PrimitiveFloat duration node found.")
        duration_node = next((n for n in duration_nodes if "Duration" in str(n.get("title", ""))), duration_nodes[0])
        named_value = (duration_node.get("widgets_values_named") or {}).get("value")
        widgets = duration_node.get("widgets_values") or []
        value = named_value if named_value is not None else (widgets[0] if widgets else None)
        require(abs(float(value) - 5.0) < 1e-9, f"{name}: _set_duration() did not update PrimitiveFloat value to requested seconds.")
        require(after_expr == before_expr, f"{name}: _set_duration() overwrote the ComfyMathExpression formula.")
        require(
            next(n for n in workflow["nodes"] if n.get("type") == "MiniMaxH3ReferenceToVideo").get("widgets_values", [])[3]
            == _h3_legal_frames(5.0),
            f"{name}: _set_duration() did not preserve the H3 legal frame fallback value.",
        )
    # Resolution: supported H3 production target must produce the selector mapping used by the builder.
    workflow = builder.load("ref2v")
    builder._set_resolution(workflow, 1344, 768)
    selector = next(n for n in workflow["nodes"] if n.get("type") == "ResolutionSelector")
    widgets = selector.get("widgets_values") or []
    require(abs(float(widgets[1]) - 0.98) < 1e-9, "H3WorkflowBuilder._set_resolution() must map 1344x768 to the selector's 0.98 MP entry.")
    builder._set_resolution(workflow, 1920, 1088)
    selector = next(n for n in workflow["nodes"] if n.get("type") == "ResolutionSelector")
    widgets = selector.get("widgets_values") or []
    require(abs(float(widgets[1]) - 2.0) < 1e-9, "H3WorkflowBuilder._set_resolution() must map 1920x1088 to the selector's 2.0 MP entry.")
    try:
        builder._set_resolution(deepcopy(workflow), 1280, 720)
    except ValueError:
        pass
    else:
        fail("H3WorkflowBuilder._set_resolution() must reject unsupported production resolutions.")
    print("PASS H3WorkflowBuilder behavioral contracts")


def validate_short_story_planner_contract() -> None:
    """Exercise the deterministic planner's minimum four-unit behavior without loading Qwen."""
    from planner.production_planner import ProductionPlanner, StoryUnit
    planner = ProductionPlanner(ROOT)
    for text in ("John enters the room.", "John enters the room and sees a light."):
        units = planner._split_story(text)
        rebalanced = planner._rebalance_story_units(units)
        require(len(rebalanced) >= 4, "ProductionPlanner must create at least four planning units for non-empty short stories.")
        joined = " ".join(unit.text for unit in rebalanced).strip()
        original = planner._clean_text(text)
        require(original in joined or joined in original or len(original.split()) <= 2, "Short-story rebalancing must preserve the source narrative text.")
    print("PASS short-story planner contract")


def validate_runtime_config_alignment() -> None:
    """Verify planner/config.py agrees with the centralized runtime YAML."""
    path = ROOT / "configs" / "runtime_versions.yaml"
    require(path.is_file(), "configs/runtime_versions.yaml is missing.")
    try:
        import yaml
    except ImportError as exc:
        fail(f"PyYAML is required for runtime configuration validation: {exc}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    generation = data.get("generation") or {}
    upscale = data.get("upscale") or {}
    delivery = data.get("delivery") or {}
    from planner import config as runtime
    checks = (
        ("H3_WIDTH", runtime.H3_WIDTH, generation.get("width")),
        ("H3_HEIGHT", runtime.H3_HEIGHT, generation.get("height")),
        ("H3_FPS", runtime.H3_FPS, generation.get("fps")),
        ("H3_FRAMES_PER_SHOT", runtime.H3_FRAMES_PER_SHOT, generation.get("frames_per_shot")),
        ("H3_STEPS", runtime.H3_STEPS, generation.get("normal_steps")),
        ("TURBO_STEPS", runtime.TURBO_STEPS, generation.get("turbo_steps")),
        ("H3_REF_IMAGE_SIZE", runtime.H3_REF_IMAGE_SIZE, generation.get("ref_image_size")),
        ("UPSCALE_WIDTH", runtime.UPSCALE_WIDTH, upscale.get("width")),
        ("UPSCALE_HEIGHT", runtime.UPSCALE_HEIGHT, upscale.get("height")),
        ("DELIVERY_WIDTH", runtime.DELIVERY_WIDTH, delivery.get("width")),
        ("DELIVERY_HEIGHT", runtime.DELIVERY_HEIGHT, delivery.get("height")),
        ("DELIVERY_FPS", runtime.DELIVERY_FPS, delivery.get("fps")),
    )
    for key, actual, configured in checks:
        require(configured == actual, f"Runtime config mismatch for {key}: planner/config.py={actual!r}, runtime_versions.yaml={configured!r}.")
    print("PASS runtime configuration alignment")



def validate_gradio_director_override() -> None:
    """Ensure the Gradio entry point preserves an explicit director-enable environment value."""
    path = ROOT / "ui" / "storyboard_gradio.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    setdefault_ok = False
    unconditional_override = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr != "setdefault":
                continue
            receiver = node.func.value
            if not (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "environ"
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "os"
            ):
                continue
            if len(node.args) != 2:
                continue
            if all(isinstance(arg, ast.Constant) for arg in node.args):
                setdefault_ok = (
                    node.args[0].value == "H3_DIRECTOR_ENABLED"
                    and node.args[1].value == "1"
                )
                if setdefault_ok:
                    break

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            receiver = target.value
            if not (
                isinstance(receiver, ast.Attribute)
                and receiver.attr == "environ"
                and isinstance(receiver.value, ast.Name)
                and receiver.value.id == "os"
            ):
                continue
            key_node = target.slice
            if isinstance(key_node, ast.Constant) and key_node.value == "H3_DIRECTOR_ENABLED":
                unconditional_override = True

    require(
        setdefault_ok,
        "Gradio entry point must use os.environ.setdefault('H3_DIRECTOR_ENABLED', '1').",
    )
    require(
        not unconditional_override,
        "Gradio entry point must not unconditionally assign H3_DIRECTOR_ENABLED.",
    )
    print("PASS Gradio director environment contract")


def validate_production_runtime_isolation() -> None:
    """Verify continuity and identity artifacts are isolated by production_id."""
    runner_path = ROOT / "execution" / "production_runner.py"
    continuity_path = ROOT / "pipeline" / "h3_scene_continuity.py"
    identity_path = ROOT / "pipeline" / "identity_anchor_store.py"

    runner = runner_path.read_text(encoding="utf-8")
    continuity = continuity_path.read_text(encoding="utf-8")
    identity = identity_path.read_text(encoding="utf-8")

    require(
        "production_id=production_id" in runner,
        "ProductionRunner must pass production_id to production-scoped stores.",
    )
    require(
        "H3SceneContinuity(" in runner,
        "ProductionRunner must initialize H3SceneContinuity.",
    )
    require(
        "IdentityAnchorStore(" in runner,
        "ProductionRunner must initialize IdentityAnchorStore.",
    )
    require(
        "base / self.production_id" in continuity,
        "H3SceneContinuity must isolate its root when production_id is supplied.",
    )
    require(
        "base / self.production_id" in identity,
        "IdentityAnchorStore must isolate its root when production_id is supplied.",
    )
    print("PASS production runtime isolation")


def validate_manifest_locking() -> None:
    """Verify every ProductionRunner storyboard manifest write is protected."""
    path = ROOT / "execution" / "production_runner.py"
    text = path.read_text(encoding="utf-8")
    require(
        "self._manifest_lock = threading.RLock()" in text,
        "ProductionRunner must create a manifest lock.",
    )

    tree = ast.parse(text, filename=str(path))
    update_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "update_manifest"
    ]
    require(update_calls, "ProductionRunner must update the storyboard reference manifest.")

    protected_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            context = item.context_expr
            if (
                isinstance(context, ast.Attribute)
                and context.attr == "_manifest_lock"
                and isinstance(context.value, ast.Name)
                and context.value.id == "self"
            ):
                protected_ranges.append(
                    (node.lineno, getattr(node, "end_lineno", node.lineno))
                )

    require(protected_ranges, "ProductionRunner has no _manifest_lock context.")
    for call in update_calls:
        require(
            any(start <= call.lineno <= end for start, end in protected_ranges),
            f"ProductionRunner manifest update at line {call.lineno} is not under _manifest_lock.",
        )
    print("PASS manifest locking")


def validate_ffprobe_stream_duration_semantics() -> None:
    """Ensure A/V duration validation measures stream duration rather than container duration."""
    path = ROOT / "pipeline" / "dialogue_duration.py"
    text = path.read_text(encoding="utf-8")
    require("-select_streams" in text, "FFprobe provider must select an individual stream.")
    require("stream=duration" in text, "FFprobe provider must request stream duration.")
    require(
        "format=duration" not in text,
        "FFprobe A/V sync validation must not use container format duration for stream comparisons.",
    )
    print("PASS FFprobe stream-duration semantics")


def validate_production_orchestrator_contract() -> None:
    """Ensure post-Qwen production enforcement is deterministic and occurs after unload."""
    path = ROOT / "pipeline" / "production_orchestrator.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    enforce = _find_function(tree, "_enforce_production_contracts")
    create = _find_function(tree, "create_production_plan")
    require(enforce is not None, "ProductionOrchestrator._enforce_production_contracts() is missing.")
    require(create is not None, "ProductionOrchestrator.create_production_plan() is missing.")

    repair_calls = [
        node
        for node in ast.walk(enforce)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "repair_continuity_violation"
    ]
    require(
        not repair_calls,
        "_enforce_production_contracts must not call Qwen continuity repair after unload.",
    )
    require(
        "apply_field_level_fallback" in ast.unparse(enforce),
        "Deterministic continuity fallback is missing from _enforce_production_contracts.",
    )

    unload_lines = [
        node.lineno
        for node in ast.walk(create)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "unload"
    ]
    enforce_lines = [
        node.lineno
        for node in ast.walk(create)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_enforce_production_contracts"
    ]
    require(unload_lines, "create_production_plan() must unload the director.")
    require(enforce_lines, "create_production_plan() must call deterministic production enforcement.")
    require(
        min(unload_lines) < min(enforce_lines),
        "Qwen director must be unloaded before deterministic production enforcement.",
    )
    print("PASS production orchestrator deterministic contract")


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

    # Safe conversion of node/link IDs, consistent with validate_workflow_graph_integrity
    nodes: dict[int, dict] = {}
    for raw_node in graph.get("nodes", []):
        raw_id = raw_node.get("id")
        try:
            node_id = int(raw_id)
        except (TypeError, ValueError):
            fail(f"Upscale workflow node id must be numeric, got {raw_id!r}.")
        nodes[node_id] = raw_node

    links: dict[int, list] = {}
    for row in graph.get("links", []):
        if not isinstance(row, list) or len(row) < 6:
            continue
        try:
            link_id = int(row[0])
        except (TypeError, ValueError):
            fail(f"Upscale workflow link id must be numeric: {row!r}")
        links[link_id] = row

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
                if link_id is None:
                    return None
                try:
                    link_id = int(link_id)
                except (TypeError, ValueError):
                    fail(f"Upscale workflow node has non-numeric link for {name}: {link_id!r}.")
                return links.get(link_id)
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

        values = executable_model_values(graph, name)

        for node_type, value in values:
            basename = model_basename(value)
            if basename.lower().endswith(".safetensors"):
                require(
                    basename in LOCKED_MODELS,
                    f"Unapproved executable model '{value}' in {node_type} found in {name}.",
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


def _find_function(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def validate_gradio_ui() -> None:
    path = ROOT / "ui" / "storyboard_gradio.py"
    require(path.is_file(), "Gradio UI file is missing.")
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text)
    tree = ast.parse(text, filename=str(path))

    for token in (
        "ProductionController",
        "generate_storyboard",
        "approve_and_generate",
        "ProductionRunner",
        "H3Runtime",
        "check_worker",
        "storyboard_share_enabled",
    ):
        require(token in normalized, f"Gradio UI is missing required contract: {token}")

    require(_find_function(tree, "build_app") is not None, "Gradio build_app() is missing.")
    require(_find_function(tree, "serve_storyboard_gradio") is not None, "Gradio serve_storyboard_gradio() is missing.")
    require("Your Story" in normalized, "Gradio story input is missing.")
    require(all(value in normalized for value in ("AI Story", "Expand Story", "Preserve Story")), "Gradio story-mode controls are incomplete.")
    require("Generate Storyboard" in normalized and "Approve & Generate Video" in normalized, "Gradio production controls are incomplete.")

    print("PASS Gradio UI contract")


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
    orchestrator_path = ROOT / "pipeline" / "production_orchestrator.py"
    cli_path = ROOT / "scripts" / "generate_video.py"
    orchestrator_text = orchestrator_path.read_text(encoding="utf-8")
    cli_text = cli_path.read_text(encoding="utf-8")

    orchestrator_tree = ast.parse(orchestrator_text, filename=str(orchestrator_path))
    cli_tree = ast.parse(cli_text, filename=str(cli_path))
    require(_find_function(orchestrator_tree, "create_production_plan") is not None, "ProductionOrchestrator plan method is missing.")
    require(_find_function(cli_tree, "create_cli_plan_path") is not None, "CLI plan persistence helper is missing.")
    require("story_preview.json" in cli_text, "CLI plan filename is missing.")
    require("save_plan(" in cli_text, "CLI save_plan() call is missing.")
    require(
        re.search(r"(?m)ROOT\s*/\s*[\'\"]data[\'\"]\s*/\s*[\'\"]production[\'\"]", cli_text)
        or re.search(r"(?m)Path\(\s*[\'\"]data/production[\'\"]", cli_text)
        or re.search(r"(?m)[\'\"]data/production/", cli_text),
        "CLI production plan path must explicitly contain data/production.",
    )
    print("PASS plan persistence boundary")


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
    path = ROOT / "planner" / "config.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    function = _find_function(tree, "storyboard_share_enabled")
    require(function is not None, "storyboard_share_enabled() is missing.")

    env_assignment = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "GRADIO_SHARE_ENV":
                    env_assignment = isinstance(node.value, ast.Constant) and node.value.value == "H3_GRADIO_SHARE"
                    if env_assignment:
                        break
            if env_assignment:
                break
    require(env_assignment, "GRADIO_SHARE_ENV must equal H3_GRADIO_SHARE.")

    from planner.config import storyboard_share_enabled
    previous = os.environ.get("H3_GRADIO_SHARE")
    try:
        for value, expected in (("1", True), ("0", False)):
            os.environ["H3_GRADIO_SHARE"] = value
            require(storyboard_share_enabled() is expected, f"H3_GRADIO_SHARE={value} contract failed.")
        os.environ.pop("H3_GRADIO_SHARE", None)
        require(storyboard_share_enabled() is True, "Gradio sharing must default to enabled.")
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
    require("self._manifest_lock = threading.RLock()" in runner_text, "ProductionRunner must protect shared reference-role manifest updates.")
    require("production_id=production_id" in runner_text, "ProductionRunner must pass production_id into production-scoped continuity/identity stores.")
    require("H3SceneContinuity(" in runner_text, "ProductionRunner must initialize H3SceneContinuity.")
    require("IdentityAnchorStore(" in runner_text, "ProductionRunner must initialize IdentityAnchorStore.")

    # Replace brittle whitespace-sensitive string check with AST-based call validation.
    tree = ast.parse(runner_text, filename=str(runner_path))
    anchor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_add_identity_anchors"
    ]
    require(anchor_calls, "ProductionRunner must invoke _add_identity_anchors().")
    for call in anchor_calls:
        keyword_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        positional_ok = len(call.args) == 2 and not keyword_names
        keyword_ok = len(call.args) == 0 and {"shot", "character_map"} <= keyword_names
        require(positional_ok or keyword_ok, "Each _add_identity_anchors call must pass shot and character_map only.")

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

    # Verify the helper signature, while allowing positional or keyword calls.
    tree = ast.parse(runner_path.read_text(encoding="utf-8"))
    helper = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_add_identity_anchors":
            helper = node
            break
    require(helper is not None, "ProductionRunner._add_identity_anchors() is missing.")
    params = [arg.arg for arg in helper.args.args]
    require(params == ["self", "shot", "character_map"], "_add_identity_anchors signature must be (self, shot, character_map).")

    anchor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_add_identity_anchors"
    ]
    require(anchor_calls, "ProductionRunner must invoke _add_identity_anchors().")
    for call in anchor_calls:
        keyword_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        positional_ok = len(call.args) == 2 and not keyword_names
        keyword_ok = len(call.args) == 0 and {"shot", "character_map"} <= keyword_names
        require(positional_ok or keyword_ok, "Each _add_identity_anchors call must pass shot and character_map only.")

    runner_text = runner_path.read_text(encoding="utf-8")
    require("self._completed_shots_lock = threading.RLock()" in runner_text, "ProductionRunner shared completed-shot lock is missing.")
    require("with self._completed_shots_lock:" in runner_text, "ProductionRunner must lock shared completed-shot state.")

    # Exercise the actual Shot dataclass and prompt serializer.
    from schemas.shot import Shot
    shot = Shot(
        shot_id="validator_shot",
        scene_id="validator_scene",
        order=1,
        duration_seconds=5.2,
        characters=[],
        location="test location",
        action="test action",
        camera_shot="medium shot",
        camera_movement="static",
        lens_and_depth_of_field="35mm",
        composition_notes="centered",
        lighting="soft",
        color_temperature="neutral",
        mood="calm",
        visual_prompt="A test shot.",
        retention_analysis="Preserve the subject.",
        detailed_description="A test cinematic description.",
        overall_soundscape="Quiet ambience.",
        non_diegetic_music="N/A",
        negative_prompt="artifacts",
        continuity_notes="Stable.",
        seed=1,
        workflow_mode="ref2v",
    )
    prompt = shot.h3_prompt()
    for section in (
        "subject_definitions:",
        "reference_bindings:",
        "summary:",
        "retention_analysis:",
        "detailed_description:",
        "overall_soundscape:",
        "non_diegetic_music:",
    ):
        require(section in prompt, f"Shot.h3_prompt() is missing required section: {section}")

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
    validate_gradio_director_override()
    validate_reference_wiring()
    validate_plan_persistence_boundary()
    validate_ui_share_configuration()
    validate_h3_duration_chain()
    validate_h3_resolution_selector_contract()
    validate_h3_builder_behavior()
    validate_short_story_planner_contract()
    validate_production_runtime_isolation()
    validate_manifest_locking()
    validate_ffprobe_stream_duration_semantics()
    validate_production_orchestrator_contract()
    validate_runtime_config_alignment()

    
    print(
        "=" * 80
    )

    print(
        "MiniMax H3 PROJECT VALIDATION PASSED."
    )


if __name__ == "__main__":
    main()

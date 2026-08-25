from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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

FORBIDDEN_EXECUTABLE_TOKENS = {
    "minimax_h3_fl2va",
    "minimax_h3_fl2v",
    "qwen3-4b",
    "Q4_K_M",
    "qwen3vl_32b_minimax_h3-Q4",
    "minimax_h3_ref2va_pruned_int8_convrot",
    "minimax_h3_video_vae_int8_convrot",
}

REQUIRED_CUSTOM_NODES = {
    "Comfyui_Minimax_h3_latent_Upscaler",
    "Comfyui-MMH3-UltimateUpscale",
    "ComfyUI-MiniMax-H3-Turbo",
    "ComfyUI-Workflow-To-API-Converter",
    "ComfyUI-VideoHelperSuite",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"Missing workflow: {path}")

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

    if not isinstance(data, dict):
        fail(
            f"Workflow root must be an object: {path}"
        )

    if not isinstance(
        data.get("nodes"),
        list,
    ):
        fail(
            f"Workflow has no node list: {path}"
        )

    return data


def node_types(workflow: dict) -> set[str]:
    return {
        str(node.get("type"))
        for node in workflow.get(
            "nodes",
            []
        )
        if isinstance(node, dict)
        and node.get("type")
    }


def executable_model_values(
    workflow: dict,
) -> list[str]:

    executable = {
        "UNETLoader",
        "CLIPLoader",
        "CLIPLoaderGGUF",
        "VAELoader",
        "LoraLoaderModelOnly",
        "MiniMaxH3TurboLoRA",
        "MMH3LatentUpscaleWithModelParams",
    }

    values = []

    for node in workflow.get(
        "nodes",
        []
    ):
        if node.get("type") not in executable:
            continue

        for value in node.get(
            "widgets_values",
            []
        ):
            if isinstance(value, str):
                values.append(value)

    return values


def validate_python() -> None:

    excluded = {
        ".git",
        "ComfyUI",
        "__pycache__",
    }

    for path in ROOT.rglob("*.py"):

        if any(
            part in excluded
            for part in path.parts
        ):
            continue

        ast.parse(
            path.read_text(
                encoding="utf-8"
            ),
            filename=str(path),
        )

    print("PASS Python syntax")


def validate_workflow_files() -> None:

    for name, path in PRODUCTION_WORKFLOWS.items():
        load_json(path)
        print(
            "PASS production workflow:",
            name,
        )

    for name, path in SOURCE_WORKFLOWS.items():
        load_json(path)
        print(
            "PASS source workflow:",
            name,
        )


def validate_ref2v() -> None:

    graph = load_json(
        PRODUCTION_WORKFLOWS["ref2v"]
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
        - node_types(graph)
    )

    if missing:
        fail(
            "Ref2V missing nodes: "
            + ", ".join(sorted(missing))
        )

    values = executable_model_values(graph)

    for filename in [
        "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors",
        "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    ]:
        if filename not in values:
            fail(
                f"Ref2V missing locked executable model: "
                f"{filename}"
            )

    print("PASS Ref2V contract")


def validate_turbo() -> None:

    graph = load_json(
        PRODUCTION_WORKFLOWS["turbo_ref2v"]
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
        - node_types(graph)
    )

    if missing:
        fail(
            "Turbo missing nodes: "
            + ", ".join(sorted(missing))
        )

    loras = [
        node
        for node in graph["nodes"]
        if node.get("type")
        == "MiniMaxH3TurboLoRA"
    ]

    if len(loras) != 1:
        fail(
            "Turbo must contain exactly one "
            "MiniMaxH3TurboLoRA."
        )

    widgets = loras[0].get(
        "widgets_values",
        []
    )

    if (
        not widgets
        or widgets[0]
        != "minimax_h3_turbo_v4_step600_ema.safetensors"
    ):
        fail(
            "Turbo is not using the locked Step600 LoRA."
        )

    schedulers = [
        node
        for node in graph["nodes"]
        if node.get("type")
        == "BasicScheduler"
    ]

    if len(schedulers) != 1:
        fail(
            "Turbo must contain exactly one BasicScheduler."
        )

    scheduler_widgets = schedulers[0].get(
        "widgets_values",
        []
    )

    if (
        len(scheduler_widgets) < 3
        or scheduler_widgets[1] != 8
        or scheduler_widgets[2] != 1
    ):
        fail(
            "Turbo scheduler must be [scheduler, 8, 1]."
        )

    print("PASS Turbo 8-step contract")


def validate_upscale() -> None:

    graph = load_json(
        PRODUCTION_WORKFLOWS["upscale"]
    )

    required = {
        "MMH3LatentUpscaleWithModelParams",
        "MMH3TemporalSplitParams",
        "MMH3SpatialSplitParams",
        "MMH3UltimateUpscale",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    }

    missing = (
        required
        - node_types(graph)
    )

    if missing:
        fail(
            "Ultimate Upscale missing nodes: "
            + ", ".join(sorted(missing))
        )

    params = [
        node
        for node in graph["nodes"]
        if node.get("type")
        == "MMH3LatentUpscaleWithModelParams"
    ]

    if len(params) != 1:
        fail(
            "Ultimate Upscale must contain exactly one "
            "MMH3LatentUpscaleWithModelParams."
        )

    widgets = params[0].get(
        "widgets_values",
        []
    )

    if (
        not widgets
        or widgets[0]
        != "minimax_h3_latent_upscaler_3d_fp16.safetensors"
    ):
        fail(
            "Ultimate Upscale does not use the locked "
            "3D H3 upscaler."
        )

    print("PASS Ultimate Upscale contract")


def validate_models_and_legacy_text() -> None:

    for name, path in PRODUCTION_WORKFLOWS.items():

        graph = load_json(path)

        values = executable_model_values(graph)

        for value in values:
            lowered = value.lower()

            for token in FORBIDDEN_EXECUTABLE_TOKENS:
                if token.lower() in lowered:
                    fail(
                        f"Forbidden executable asset '{token}' "
                        f"found in {name}"
                    )

        # Production executable values must come from the
        # locked inventory for model filenames.
        for value in values:
            if (
                value.endswith(".safetensors")
                and value not in LOCKED_MODELS
            ):
                fail(
                    f"Unapproved executable model '{value}' "
                    f"found in {name}"
                )

    print("PASS locked executable model inventory")


def validate_custom_nodes() -> None:

    path = (
        ROOT
        / "configs"
        / "custom_nodes.yaml"
    )

    text = path.read_text(
        encoding="utf-8"
    )

    for node in REQUIRED_CUSTOM_NODES:
        if node not in text:
            fail(
                f"Required custom node missing "
                f"from custom_nodes.yaml: {node}"
            )

    print("PASS custom-node manifest")


def validate_no_test_artifacts() -> None:

    forbidden_files = [
        ROOT
        / "workflows"
        / "generation"
        / "aa",

        ROOT
        / "workflows"
        / "sources"
        / "a",
    ]

    for path in forbidden_files:
        if path.exists():
            fail(
                f"Stray repository artifact exists: {path}"
            )

    for path in ROOT.rglob("*_TEST.json"):
        if path.is_file():
            fail(
                f"Temporary TEST workflow remains: {path}"
            )

    print("PASS repository artifact cleanup")


def main() -> None:

    validate_python()
    validate_workflow_files()
    validate_ref2v()
    validate_turbo()
    validate_upscale()
    validate_models_and_legacy_text()
    validate_custom_nodes()
    validate_no_test_artifacts()

    print(
        "\nMiniMax H3 project validation PASSED."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

COMFY = ROOT / "ComfyUI"

KAGGLE_INPUT = Path(
    "/kaggle/input"
)

MODEL_MANIFEST = (
    ROOT
    / "configs"
    / "model_inventory.yaml"
)

NODE_MANIFEST = (
    ROOT
    / "configs"
    / "custom_nodes.yaml"
)

DIRECTOR_MODEL_FILENAME = (
    "Qwen3-14B-Q4_K_M.gguf"
)

PRODUCTION_WORKFLOWS = [
    (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Ref2V_Production.json"
    ),
    (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Turbo_Ref2V_Production.json"
    ),
    (
        ROOT
        / "workflows"
        / "postprocess"
        / "H3_Ref2V_UltimateUpscale_Production.json"
    ),
]

SOURCE_WORKFLOWS = [
    (
        ROOT
        / "workflows"
        / "sources"
        / "H3_Turbo_Reference_Source.json"
    ),
    (
        ROOT
        / "workflows"
        / "sources"
        / "H3_LatentUpscaler_Source.json"
    ),
]


# ============================================================
# HELPERS
# ============================================================

def load_yaml(
    path: Path,
):
    if not path.is_file():
        raise RuntimeError(
            f"Missing manifest: {path}"
        )

    return yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_json(
    path: Path,
):
    if not path.is_file():
        raise RuntimeError(
            f"Missing workflow: {path}"
        )

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Invalid JSON: {path}: {exc}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):

        raise RuntimeError(
            f"Workflow root is not an object: {path}"
        )

    if not isinstance(
        data.get("nodes"),
        list,
    ):

        raise RuntimeError(
            f"Invalid workflow nodes: {path}"
        )

    return data


# ============================================================
# DIRECTOR MODEL
# ============================================================

def find_director_model() -> Path:

    matches = []

    for path in KAGGLE_INPUT.rglob("*"):

        if (
            path.is_file()
            and path.name.lower()
            == DIRECTOR_MODEL_FILENAME.lower()
        ):
            matches.append(
                path
            )

    if len(matches) != 1:

        raise RuntimeError(
            "Qwen director model contract failed.\n"
            f"Expected exactly one: "
            f"{DIRECTOR_MODEL_FILENAME}\n"
            f"Found: {len(matches)}\n"
            + (
                "\n".join(
                    str(path)
                    for path in matches
                )
                if matches
                else "No matching file found."
            )
        )

    return matches[0]


def check_director_model() -> None:

    path = find_director_model()

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"Qwen director model is empty: {path}"
        )

    print(
        "DIRECTOR MODEL OK:",
        path,
    )


def check_director_runtime() -> None:

    try:
        import llama_cpp  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "llama-cpp-python is unavailable."
        ) from exc

    print(
        "DIRECTOR RUNTIME OK: llama-cpp-python"
    )


# ============================================================
# H3 MODELS
# ============================================================

def check_models() -> None:

    manifest = load_yaml(
        MODEL_MANIFEST
    )

    for model in manifest[
        "models"
    ].values():

        path = (
            COMFY
            / "models"
            / model[
                "directory"
            ]
            / model[
                "filename"
            ]
        )

        if not path.is_file():

            raise RuntimeError(
                f"Missing locked model: {path}"
            )

        print(
            "H3 MODEL OK:",
            model[
                "filename"
            ],
        )


# ============================================================
# WORKFLOWS
# ============================================================

def check_workflows() -> None:

    for path in (
        PRODUCTION_WORKFLOWS
        + SOURCE_WORKFLOWS
    ):

        load_json(
            path
        )

        print(
            "WORKFLOW OK:",
            path.relative_to(ROOT),
        )

    ref2v = load_json(
        PRODUCTION_WORKFLOWS[0]
    )

    turbo = load_json(
        PRODUCTION_WORKFLOWS[1]
    )

    upscale = load_json(
        PRODUCTION_WORKFLOWS[2]
    )

    ref_types = {
        node.get("type")
        for node in ref2v[
            "nodes"
        ]
    }

    turbo_types = {
        node.get("type")
        for node in turbo[
            "nodes"
        ]
    }

    upscale_types = {
        node.get("type")
        for node in upscale[
            "nodes"
        ]
    }

    ref_required = {
        "UNETLoader",
        "CLIPLoader",
        "MiniMaxH3ReferenceToVideo",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    }

    turbo_required = {
        "MiniMaxH3TurboLoRA",
        "MiniMaxH3TurboSampler",
    }

    upscale_required = {
        "MinimaxH3LatentUpscaler3D",
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
        ref_required
        - ref_types
    )

    if missing:
        raise RuntimeError(
            "Ref2V missing nodes: "
            + ", ".join(
                sorted(missing)
            )
        )

    missing = (
        turbo_required
        - turbo_types
    )

    if missing:
        raise RuntimeError(
            "Turbo missing nodes: "
            + ", ".join(
                sorted(missing)
            )
        )

    missing = (
        upscale_required
        - upscale_types
    )

    if missing:
        raise RuntimeError(
            "Upscale missing nodes: "
            + ", ".join(
                sorted(missing)
            )
        )

    # Confirm the actual Turbo checkpoint remains the
    # Step600 model and the obsolete 4-step asset is absent.
    turbo_text = json.dumps(
        turbo
    )

    if (
        "minimax_h3_turbo_v4_step600_ema.safetensors"
        not in turbo_text
    ):
        raise RuntimeError(
            "Turbo workflow does not contain "
            "the locked Step600 LoRA."
        )

    if (
        "minimax_h3_ref2v_turbo_4step"
        in turbo_text
    ):
        raise RuntimeError(
            "Obsolete Turbo 4-step model remains."
        )

    # Confirm the locked 3D latent upscaler.
    upscale_text = json.dumps(
        upscale
    )

    if (
        "minimax_h3_latent_upscaler_3d_fp16.safetensors"
        not in upscale_text
    ):
        raise RuntimeError(
            "Upscale workflow does not use the "
            "locked H3 3D latent upscaler."
        )


# ============================================================
# CUSTOM NODES
# ============================================================

def check_custom_nodes() -> None:

    manifest = load_yaml(
        NODE_MANIFEST
    )

    nodes = (
        manifest[
            "custom_nodes"
        ][
            "required"
        ]
        + manifest[
            "custom_nodes"
        ][
            "supporting"
        ]
    )

    for node in nodes:

        path = (
            COMFY
            / "custom_nodes"
            / node[
                "name"
            ]
        )

        if not path.is_dir():

            raise RuntimeError(
                f"Missing custom node: {path}"
            )

        print(
            "NODE OK:",
            node["name"],
        )


# ============================================================
# LEGACY WORKFLOW CHECK
# ============================================================

def check_no_legacy() -> None:

    forbidden = {
        "qwen3-4b",
        "minimax_h3_fl2va",
        "minimax_h3_fl2v",
        "minimax_h3_ref2va_pruned_int8_convrot",
        "minimax_h3_video_vae_int8_convrot",
        "minimax_h3_ref2v_turbo_4step",
    }

    executable_types = {
        "UNETLoader",
        "CLIPLoader",
        "CLIPLoaderGGUF",
        "VAELoader",
        "LoraLoaderModelOnly",
        "MMH3LatentUpscaleWithModelParams",
    }

    for path in PRODUCTION_WORKFLOWS:

        graph = load_json(
            path
        )

        for node in graph[
            "nodes"
        ]:

            if (
                node.get(
                    "type"
                )
                not in executable_types
            ):
                continue

            for value in node.get(
                "widgets_values",
                [],
            ):

                if not isinstance(
                    value,
                    str,
                ):
                    continue

                lowered = value.lower()

                for token in forbidden:

                    if (
                        token.lower()
                        in lowered
                    ):

                        raise RuntimeError(
                            "Forbidden legacy "
                            "executable asset "
                            f"{token} in {path}"
                        )


# ============================================================
# DELIVERY CONTRACT
# ============================================================

def check_delivery_contract() -> None:

    from planner.config import (
        DELIVERY_FPS,
        DELIVERY_HEIGHT,
        DELIVERY_WIDTH,
        UPSCALE_HEIGHT,
        UPSCALE_WIDTH,
    )

    if (
        DELIVERY_WIDTH,
        DELIVERY_HEIGHT,
        DELIVERY_FPS,
    ) != (
        1280,
        720,
        24,
    ):
        raise RuntimeError(
            "Invalid 720p delivery contract."
        )

    if (
        UPSCALE_WIDTH,
        UPSCALE_HEIGHT,
    ) != (
        1920,
        1080,
    ):
        raise RuntimeError(
            "Invalid 1080p internal upscale contract."
        )

    print(
        "DELIVERY OK: 1920x1080 upscale -> 1280x720"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    check_director_model()
    check_director_runtime()

    check_models()
    check_workflows()
    check_custom_nodes()
    check_no_legacy()
    check_delivery_contract()

    print(
        "=" * 80
    )

    print(
        "MiniMax H3 PRE-FLIGHT PASSED."
    )

    print(
        "Director: Qwen3-14B Q4_K_M"
    )

    print(
        "Generator: MiniMax H3"
    )

    print(
        "Final delivery: 1280x720"
    )


if __name__ == "__main__":
    main()

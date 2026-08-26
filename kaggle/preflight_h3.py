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

RUNTIME_MANIFEST = (
    ROOT
    / "configs"
    / "runtime_versions.yaml"
)

PRODUCTION_WORKFLOWS = [
    ROOT
    / "workflows"
    / "generation"
    / "H3_Ref2V_Production.json",

    ROOT
    / "workflows"
    / "generation"
    / "H3_Turbo_Ref2V_Production.json",

    ROOT
    / "workflows"
    / "postprocess"
    / "H3_Ref2V_UltimateUpscale_Production.json",
]

SOURCE_WORKFLOWS = [
    ROOT
    / "workflows"
    / "sources"
    / "H3_Turbo_Reference_Source.json",

    ROOT
    / "workflows"
    / "sources"
    / "H3_LatentUpscaler_Source.json",
]


def load_yaml(
    path,
):
    return yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_json(
    path,
):
    if not path.is_file():
        raise RuntimeError(
            f"Missing workflow: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def find_input_file(
    filename,
):

    matches = [
        path
        for path
        in KAGGLE_INPUT.rglob(
            "*"
        )
        if (
            path.is_file()
            and path.name.lower()
            == filename.lower()
        )
    ]

    if len(matches) != 1:

        raise RuntimeError(
            f"Expected exactly one Kaggle file "
            f"{filename}; found {len(matches)}"
        )

    return matches[0]


def check_director():

    runtime = load_yaml(
        RUNTIME_MANIFEST
    )

    filename = (
        runtime[
            "director"
        ][
            "model_filename"
        ]
    )

    path = find_input_file(
        filename
    )

    if path.stat().st_size <= 0:
        raise RuntimeError(
            f"Director model is empty: {path}"
        )

    try:
        import llama_cpp  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "llama-cpp-python is unavailable."
        ) from exc

    print(
        "DIRECTOR OK:",
        path,
    )


def check_models():

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
                f"Missing H3 model: {path}"
            )

        print(
            "MODEL OK:",
            model[
                "filename"
            ],
        )


def check_workflows():

    for path in (
        PRODUCTION_WORKFLOWS
        + SOURCE_WORKFLOWS
    ):

        graph = load_json(
            path
        )

        if not isinstance(
            graph.get("nodes"),
            list,
        ):
            raise RuntimeError(
                f"Workflow has no node list: {path}"
            )

        print(
            "WORKFLOW OK:",
            path.relative_to(ROOT),
        )


def check_upscale_contract():

    graph = load_json(
        PRODUCTION_WORKFLOWS[2]
    )

    types = {
        node.get(
            "type"
        )
        for node in graph[
            "nodes"
        ]
    }

    required = {
        "MMH3LatentUpscaleWithModelParams",
        "MMH3TemporalSplitParams",
        "MMH3SpatialSplitParams",
        "MMH3UltimateUpscale",
    }

    missing = (
        required
        - types
    )

    if missing:
        raise RuntimeError(
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
        raise RuntimeError(
            "Upscale workflow does not reference "
            "the locked H3 3D upscaler."
        )

    print(
        "UPSCALE CONTRACT OK"
    )


def check_custom_nodes():

    manifest = load_yaml(
        NODE_MANIFEST
    )

    for node in (
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
    ):

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
            node[
                "name"
            ],
        )


def check_delivery():

    runtime = load_yaml(
        RUNTIME_MANIFEST
    )

    generation = runtime[
        "generation"
    ]

    upscale = runtime[
        "upscale"
    ]

    delivery = runtime[
        "delivery"
    ]

    assert (
        generation[
            "width"
        ],
        generation[
            "height"
        ]
    ) == (
        1344,
        768,
    )

    assert (
        generation[
            "normal_steps"
        ],
        generation[
            "turbo_steps"
        ]
    ) == (
        20,
        8,
    )

    assert (
        delivery[
            "width"
        ],
        delivery[
            "height"
        ]
    ) == (
        1280,
        720,
    )

    assert (
        upscale[
            "width"
        ],
        upscale[
            "height"
        ]
    ) == (
        1920,
        1088,
    )

    print(
        "DELIVERY CONTRACT OK"
    )


def main():

    check_director()
    check_models()
    check_workflows()
    check_upscale_contract()
    check_custom_nodes()
    check_delivery()

    print(
        "=" * 80
    )

    print(
        "MiniMax H3 PRE-FLIGHT PASSED."
    )


if __name__ == "__main__":
    main()

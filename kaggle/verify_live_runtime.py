from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PRODUCTION = {
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

REQUIRED_LIVE_NODES = {
    "BasicScheduler",
    "CLIPLoader",
    "CreateVideo",
    "MMH3LatentUpscaleWithModelParams",
    "MMH3SpatialSplitParams",
    "MMH3TemporalSplitParams",
    "MMH3UltimateUpscale",
    "MiniMaxH3ReferenceToVideo",
    "MiniMaxH3TurboLoRA",
    "MiniMaxH3TurboSampler",
    "MinimaxH3LatentUpscaler3D",
    "SamplerCustomAdvanced",
    "SaveVideo",
    "UNETLoader",
    "VAEDecode",
    "VAEDecodeAudio",
}


def get(
    base: str,
    endpoint: str,
):
    with urllib.request.urlopen(
        base + endpoint,
        timeout=60,
    ) as response:
        return json.loads(
            response.read().decode(
                "utf-8"
            )
        )


def workflow_types(
    path: Path,
) -> set[str]:

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    return {
        str(node.get("type"))
        for node in data.get(
            "nodes",
            []
        )
        if isinstance(node, dict)
        and node.get("type")
    }


def check_worker(
    port: int,
) -> None:

    base = (
        f"http://127.0.0.1:{port}"
    )

    objects = get(
        base,
        "/object_info",
    )

    available = set(
        objects
    )

    print(
        f"\n=== WORKER {port} ==="
    )

    missing = (
        REQUIRED_LIVE_NODES
        - available
    )

    for node in sorted(
        REQUIRED_LIVE_NODES
    ):
        print(
            "OK   "
            if node in available
            else "FAIL ",
            node,
        )

    if missing:
        raise RuntimeError(
            "Worker is missing required H3 nodes: "
            + ", ".join(
                sorted(missing)
            )
        )

    # Verify every executable production workflow
    # contains only registered runtime node types.
    ignored = {
        "MarkdownNote",
        "Note",
        "PrimitiveFloat",
        "PrimitiveInt",
        "PrimitiveBoolean",
        "PrimitiveString",
        "PrimitiveStringMultiline",
        "ComfyMathExpression",
        "INTConstant",
        "LoadImage",
        "ResolutionSelector",
        "KSamplerSelect",
        "RandomNoise",
    }

    for name, path in PRODUCTION.items():

        types = (
            workflow_types(path)
            - ignored
        )

        unknown = (
            types
            - available
        )

        if unknown:
            raise RuntimeError(
                f"{name}: unregistered executable "
                "node types: "
                + ", ".join(sorted(unknown))
            )

        print(
            f"PASS {name} live node compatibility"
        )


def main() -> None:

    configured = os.getenv(
        "H3_GPU_IDS",
        "0",
    )

    ids = [
        int(value.strip())
        for value in configured.split(",")
        if value.strip()
    ]

    if not ids:
        ids = [0]

    for index, _ in enumerate(ids):
        check_worker(
            8188 + index
        )

    print(
        "\nAll configured H3 workers are ready."
    )


if __name__ == "__main__":
    main()

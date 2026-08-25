from __future__ import annotations

import json
import os
import urllib.request


REQUIRED_H3_RUNTIME_NODES = {
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
    base,
    endpoint,
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


def check_worker(
    port,
):

    base = (
        f"http://127.0.0.1:{port}"
    )

    objects = get(
        base,
        "/object_info",
    )

    missing = sorted(
        REQUIRED_H3_NODES
        - set(objects)
    )

    print(
        f"\n=== WORKER {port} ==="
    )

    for node in sorted(
        REQUIRED_H3_NODES
    ):

        print(
            "OK  "
            if node in objects
            else "FAIL",
            node,
        )

    if missing:
        raise RuntimeError(
            f"Worker {port} is missing: "
            + ", ".join(
                missing
            )
        )

    return True


def main():

    configured = os.getenv(
        "H3_GPU_IDS",
        "0",
    )

    ids = [
        int(value.strip())
        for value in configured.split(",")
        if value.strip()
    ]

    for index, _ in enumerate(
        ids
    ):

        check_worker(
            8188 + index
        )

    print(
        "\nAll configured H3 workers are ready."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import urllib.error
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
    url = (
        base.rstrip("/")
        + endpoint
    )

    request = urllib.request.Request(
        url=url,
        method="GET",
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:

            body = response.read()

    except urllib.error.HTTPError as exc:

        details = (
            exc.read()
            .decode(
                "utf-8",
                errors="replace",
            )
        )

        raise RuntimeError(
            f"ComfyUI returned HTTP {exc.code} "
            f"for {endpoint}.\n"
            f"{details}"
        ) from exc

    except (
        urllib.error.URLError,
        TimeoutError,
    ) as exc:

        raise RuntimeError(
            "Unable to contact ComfyUI.\n"
            f"URL: {url}\n"
            f"Error: {exc}"
        ) from exc

    if not body:

        raise RuntimeError(
            f"ComfyUI returned an empty response "
            f"for {endpoint}."
        )

    try:

        return json.loads(
            body.decode(
                "utf-8"
            )
        )

    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as exc:

        raise RuntimeError(
            f"ComfyUI returned invalid JSON "
            f"for {endpoint}."
        ) from exc


def check_basic_endpoints(
    base: str,
) -> None:

    # --------------------------------------------------------
    # SYSTEM HEALTH
    # --------------------------------------------------------

    system_stats = get(
        base,
        "/system_stats",
    )

    if not isinstance(
        system_stats,
        dict,
    ):

        raise RuntimeError(
            "ComfyUI /system_stats returned "
            "an invalid response."
        )

    print(
        "PASS /system_stats"
    )

    # --------------------------------------------------------
    # NODE REGISTRY
    # --------------------------------------------------------

    object_info = get(
        base,
        "/object_info",
    )

    if not isinstance(
        object_info,
        dict,
    ):

        raise RuntimeError(
            "ComfyUI /object_info returned "
            "an invalid response."
        )

    print(
        "PASS /object_info"
    )


def workflow_types(
    path: Path,
) -> set[str]:

    if not path.is_file():

        raise RuntimeError(
            f"Production workflow is missing:\n{path}"
        )

    try:

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            f"Invalid workflow JSON:\n{path}\n{exc}"
        ) from exc

    if not isinstance(
        data,
        dict,
    ):

        raise RuntimeError(
            f"Workflow root must be an object:\n{path}"
        )

    return {
        str(
            node.get(
                "type"
            )
        )
        for node in data.get(
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


def check_worker(
    port: int,
) -> None:

    base = (
        f"http://127.0.0.1:{port}"
    )

    print(
        f"\n=== WORKER {port} ==="
    )

    # --------------------------------------------------------
    # HTTP HEALTH
    # --------------------------------------------------------

    check_basic_endpoints(
        base
    )

    # --------------------------------------------------------
    # LIVE NODE REGISTRY
    # --------------------------------------------------------

    objects = get(
        base,
        "/object_info",
    )

    available = set(
        objects
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
                sorted(
                    missing
                )
            )
        )

    print(
        "PASS required H3 node inventory"
    )

    # --------------------------------------------------------
    # WORKFLOW → LIVE NODE COMPATIBILITY
    # --------------------------------------------------------

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

    for name, path in (
        PRODUCTION.items()
    ):

        types = (
            workflow_types(
                path
            )
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
                + ", ".join(
                    sorted(
                        unknown
                    )
                )
            )

        print(
            f"PASS {name} live node compatibility"
        )


def main() -> None:

    configured = os.getenv(
        "H3_GPU_IDS",
        "0",
    )

    ids = []

    for value in (
        configured.split(",")
    ):

        value = value.strip()

        if not value:
            continue

        try:

            ids.append(
                int(value)
            )

        except ValueError as exc:

            raise RuntimeError(
                "H3_GPU_IDS contains an invalid GPU ID: "
                f"{value!r}"
            ) from exc

    if not ids:

        ids = [0]

    for index, _gpu_id in enumerate(
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

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import yaml


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


PRODUCTION = {
    "ref2va": (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Ref2VA_Production.json"
    ),
    "turbo_ref2va": (
        ROOT
        / "workflows"
        / "generation"
        / "H3_Turbo_Ref2VA_Production.json"
    ),
    "upscale": (
        ROOT
        / "workflows"
        / "postprocess"
        / "H3_Ref2VA_UltimateUpscale_Production.json"
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


def _director_request_json(url: str, payload: dict | None = None, timeout: int = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url=url,
        data=data,
        headers=headers,
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Director vLLM returned HTTP {exc.code} for {url}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Unable to contact Director vLLM at {url}: {exc}") from exc
    if not raw:
        raise RuntimeError(f"Director vLLM returned an empty JSON response for {url}")
    try:
        result = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Director vLLM returned invalid JSON for {url}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"Director vLLM returned a non-object response for {url}")
    return result


def check_director_vllm() -> None:
    """Optional live vLLM smoke test; opt in after the Director server is ready."""
    enabled = os.getenv("H3_VERIFY_DIRECTOR_VLLM", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        print("SKIP Director vLLM live smoke test (set H3_VERIFY_DIRECTOR_VLLM=1 after the Director is ready)")
        return
    runtime_path = ROOT / "configs" / "runtime_versions.yaml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}
    director = runtime.get("director", {}) or {}
    expected_version = str(director.get("vllm_version", "") or "").strip()
    if expected_version != "0.32.0":
        raise RuntimeError(f"Director runtime manifest must pin vLLM 0.32.0, got {expected_version!r}")
    configured_base = os.getenv(
        "H3_DIRECTOR_VLLM_BASE_URL",
        f"http://{director.get('host', '127.0.0.1')}:{int(director.get('port', 8000))}/v1",
    ).rstrip("/")
    root = configured_base[:-3] if configured_base.endswith("/v1") else configured_base
    model_name = os.getenv(
        "H3_DIRECTOR_VLLM_MODEL_NAME",
        Path(str(director.get("model_path", "Qwen3-14B-AWQ"))).name,
    )
    timeout = max(5, int(os.getenv("H3_DIRECTOR_VLLM_SMOKE_TIMEOUT", "180")))

    health_url = root + "/health"
    health_request = urllib.request.Request(health_url, method="GET")
    try:
        with urllib.request.urlopen(health_request, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Director vLLM health returned HTTP {response.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Director vLLM health returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Director vLLM health check failed: {exc}") from exc

    version = _director_request_json(root + "/version", timeout=30)
    observed_version = str(version.get("version", "") or "").strip()
    if observed_version != expected_version:
        raise RuntimeError(f"Director vLLM version mismatch: expected={expected_version}, actual={observed_version!r}")
    models = _director_request_json(root + "/v1/models", timeout=30)
    model_ids = {str(item.get("id", "")) for item in models.get("data", []) if isinstance(item, dict)}
    if model_name not in model_ids:
        raise RuntimeError(f"Director vLLM model {model_name!r} is missing; observed={sorted(model_ids)}")

    # One generated token exercises the real chat endpoint and EAGLE-3 path,
    # rather than accepting a health-only success as proof of readiness.
    completion = _director_request_json(
        root + "/v1/chat/completions",
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1,
            "seed": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=timeout,
    )
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Director vLLM chat completion returned no choices")
    print(f"PASS Director vLLM version={observed_version} model={model_name} EAGLE-3 chat endpoint")


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

        if "MiniMaxH3ContextIR" in workflow_types(path):
            raise RuntimeError(
                f"{name}: retired external Context-IR node is still present."
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
    check_director_vllm()


if __name__ == "__main__":
    main()

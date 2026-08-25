from __future__ import annotations

import argparse
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


def discover_gpu_ids():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "NVIDIA CUDA GPU is required "
            "for production H3 generation."
        )

    count = torch.cuda.device_count()

    if count <= 0:
        raise RuntimeError(
            "No CUDA GPUs detected."
        )

    return list(
        range(count)
    )


def load_clients(
    urls,
):
    from execution.comfy_client import (
        ComfyClient,
    )

    clients = {}

    for index, url in enumerate(
        urls
    ):
        client = ComfyClient(
            base_url=url,
            timeout=60,
            request_retries=3,
        )

        if not client.health_check():
            raise RuntimeError(
                f"ComfyUI worker unavailable: {url}"
            )

        clients[index] = client

    return clients


def main():

    parser = argparse.ArgumentParser(
        description=(
            "MiniMax H3 production video generator"
        )
    )

    parser.add_argument(
        "--story",
        required=True,
    )

    parser.add_argument(
        "--mode",
        default="preserve_user_story",
        choices=[
            "ai_story",
            "preserve_user_story",
            "expand_user_story",
        ],
    )

    parser.add_argument(
        "--profile",
        default="base",
        choices=[
            "base",
            "turbo",
            "upscale",
        ],
    )

    parser.add_argument(
        "--worker",
        action="append",
        dest="workers",
        help=(
            "Existing ComfyUI worker URL. "
            "Repeat for multiple GPUs."
        ),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
    )

    args = parser.parse_args()

    from pipeline.production_orchestrator import (
        ProductionOrchestrator,
    )

    orchestrator = (
        ProductionOrchestrator()
    )

    plan = (
        orchestrator
        .create_production_plan(
            mode=args.mode,
            user_input=args.story,
            profile=args.profile,
        )
    )

    print(
        json.dumps(
            {
                "plan": plan.get(
                    "production_plan_path"
                ),
                "profile": args.profile,
                "shots": len(
                    plan.get(
                        "shots",
                        [],
                    )
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.plan_only:
        return 0

    runtime_workers = None

    if args.workers:
        workers = args.workers
    else:
        from execution.h3_runtime import (
            H3Runtime,
        )

        gpu_ids = discover_gpu_ids()

        runtime_workers = (
            H3Runtime.launch_workers(
                ROOT,
                gpu_ids,
            )
        )

        workers = [
            item["url"]
            for item in (
                runtime_workers
                .values()
            )
        ]

    try:

        clients = load_clients(
            workers
        )

        from execution.production_runner import (
            ProductionRunner,
        )

        result = (
            ProductionRunner(
                project_root=ROOT,
                comfy_clients=clients,
            )
            .run(
                plan
            )
        )

        print(
            json.dumps(
                {
                    "status": "completed",
                    "profile": args.profile,
                    "shot_outputs": [
                        str(path)
                        for path in result[
                            "shot_outputs"
                        ]
                    ],
                    "final_video": str(
                        result[
                            "final_video"
                        ]
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    finally:

        if runtime_workers is not None:
            from execution.h3_runtime import (
                H3Runtime,
            )

            H3Runtime.stop_workers(
                runtime_workers
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

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
            "for production generation."
        )

    return list(
        range(
            torch.cuda.device_count()
        )
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
        "--preview",
        "--plan-only",
        action="store_true",
        dest="preview",
        help=(
            "Create the story/scenes/shots preview "
            "without starting GPU generation."
        ),
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
            workflow_mode="auto",
            profile=args.profile,
        )
    )

    preview = {
        "preview": True,
        "story_mode": args.mode,
        "profile": args.profile,
        "characters": plan[
            "character_count"
        ],
        "scenes": plan[
            "scene_count"
        ],
        "shots": plan[
            "shot_count"
        ],
        "audio_policy": plan[
            "audio_policy"
        ],
        "preview_file": plan[
            "production_plan_path"
        ],
    }

    print(
        json.dumps(
            preview,
            indent=2,
            ensure_ascii=False,
        )
    )

    if args.preview:
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
            for item in runtime_workers.values()
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

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


WORKFLOW_CHOICES = [
    "auto",
    "hard_r2v",
    "hard_chained",
    "seamless_v2",
    "seamless_core",
    "keyframes",
    "extend_take",
    "turbo_i2v",
    "turbo_ref2v",
    "turbo_t2v",
]


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
            "MiniMax H3 production pipeline"
        )
    )

    parser.add_argument(
        "--story",
        required=True,
    )

    parser.add_argument(
        "--mode",
        default="ai_story",
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
        ],
    )

    parser.add_argument(
        "--workflow",
        default="auto",
        choices=WORKFLOW_CHOICES,
    )

    parser.add_argument(
        "--worker",
        action="append",
        dest="workers",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
    )

    args = parser.parse_args()

    workers = (
        args.workers
        if args.workers
        else [
            "http://127.0.0.1:8188"
        ]
    )

    from pipeline.production_orchestrator import (
        ProductionOrchestrator,
    )

    orchestrator = (
        ProductionOrchestrator()
    )

    try:

        plan = (
            orchestrator
            .create_production_plan(
                mode=args.mode,
                user_input=args.story,
                workflow_mode=args.workflow,
                profile=args.profile,
            )
        )

    finally:

        orchestrator.unload_models()

    print(
        json.dumps(
            {
                "plan": plan.get(
                    "production_plan_path"
                ),
                "workflow": args.workflow,
                "profile": args.profile,
                "shots": len(
                    plan.get(
                        "shots",
                        [],
                    )
                ),
            },
            indent=2,
        )
    )

    if args.plan_only:
        return

    from execution.production_runner import (
        ProductionRunner,
    )

    clients = load_clients(
        workers
    )

    result = (
        ProductionRunner(
            project_root=ROOT,
            comfy_clients=clients,
        ).run(
            plan
        )
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "video": str(result),
                "profile": args.profile,
                "workflow": args.workflow,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

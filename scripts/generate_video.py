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
    "turbo_ref2v",
    "ref2v",
]


def load_clients(
    urls: list[str],
):
    from execution.comfy_client import (
        ComfyClient,
    )

    clients = {}

    for index, url in enumerate(urls):

        client = ComfyClient(
            base_url=url,
            timeout=60,
            request_retries=3,
        )

        if not client.health_check():
            raise RuntimeError(
                "ComfyUI worker unavailable:\n"
                f"{url}"
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
        help="Story text or path to story input.",
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
        help=(
            "ComfyUI worker URL. Repeat for multiple GPUs."
        ),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Create the production plan but do not execute.",
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

    orchestrator = ProductionOrchestrator()

    try:
        plan = (
            orchestrator.create_production_plan(
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
            ensure_ascii=False,
        )
    )

    if args.plan_only:
        return 0

    clients = load_clients(
        workers
    )

    from execution.production_runner import (
        ProductionRunner,
    )

    result = ProductionRunner(
        project_root=ROOT,
        comfy_clients=clients,
    ).run(
        plan
    )

    print(
        json.dumps(
            {
                "status": "completed",
                "shot_outputs": [
                    str(path)
                    for path in result
                ],
                "profile": args.profile,
                "workflow": args.workflow,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

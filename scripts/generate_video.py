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


def load_plan(
    path: Path,
) -> dict:

    if not path.is_file():
        raise FileNotFoundError(
            path
        )

    plan = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        plan,
        dict,
    ):
        raise RuntimeError(
            "Production plan must be a JSON object."
        )

    return plan


def save_plan(
    path: Path,
    plan: dict,
) -> None:

    path.write_text(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def require_approval(
    plan: dict,
) -> None:

    approval = plan.get(
        "approval",
        {},
    )

    if (
        approval.get(
            "status"
        )
        != "approved"
    ):
        raise RuntimeError(
            "Storyboard has not been approved.\n"
            "Run with --storyboard, approve it in the "
            "browser, then continue to generation."
        )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "MiniMax H3 production video generator"
        )
    )

    parser.add_argument(
        "--story",
        default=None,
        help=(
            "Story or subject used for planning."
        ),
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
        "--plan",
        default=None,
        help=(
            "Use an existing production plan."
        ),
    )

    parser.add_argument(
        "--storyboard",
        action="store_true",
        help=(
            "Create the plan, launch the interactive "
            "visual storyboard, wait for approval, "
            "then continue to generation."
        ),
    )

    parser.add_argument(
        "--storyboard-port",
        type=int,
        default=8765,
    )

    parser.add_argument(
        "--preview",
        "--plan-only",
        action="store_true",
        dest="preview",
        help=(
            "Create the production plan only."
        ),
    )

    parser.add_argument(
        "--upscale",
        action="store_true",
        help=(
            "Run the H3 latent 3D upscaler and "
            "MMH3 Ultimate Upscale before final "
            "720p delivery."
        ),
    )

    args = parser.parse_args()

    from pipeline.production_orchestrator import (
        ProductionOrchestrator,
    )

    # --------------------------------------------------------
    # LOAD OR CREATE PLAN
    # --------------------------------------------------------

    created_new_plan = False

    if args.plan:

        plan_path = Path(
            args.plan
        )

        plan = load_plan(
            plan_path
        )

    else:

        if not args.story:

            raise RuntimeError(
                "--story is required unless "
                "--plan is supplied."
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

        plan_path = Path(
            plan[
                "production_plan_path"
            ]
        )

        created_new_plan = True

    # --------------------------------------------------------
    # UPSCALE OVERRIDE
    # --------------------------------------------------------

    if args.upscale:

        plan[
            "upscale_enabled"
        ] = True

    elif args.profile == "upscale":

        plan[
            "upscale_enabled"
        ] = True

    # Save new plans and any explicit CLI override.
    if (
        created_new_plan
        or args.upscale
        or args.profile == "upscale"
    ):
        save_plan(
            plan_path,
            plan,
        )

    # --------------------------------------------------------
    # PREVIEW SUMMARY
    # --------------------------------------------------------

    preview = {
        "preview": True,
        "story_mode": plan.get(
            "story_mode"
        ),
        "profile": plan.get(
            "profile"
        ),
        "characters": len(
            plan.get(
                "characters",
                [],
            )
        ),
        "scenes": len(
            plan.get(
                "scenes",
                [],
            )
        ),
        "shots": len(
            plan.get(
                "shots",
                [],
            )
        ),
        "upscale_enabled": bool(
            plan.get(
                "upscale_enabled",
                False,
            )
        ),
        "preview_file": str(
            plan_path
        ),
    }

    print(
        json.dumps(
            preview,
            indent=2,
            ensure_ascii=False,
        )
    )

    # --------------------------------------------------------
    # STORYBOARD
    # --------------------------------------------------------

    if args.storyboard:

    from ui.storyboard_gradio import (
        serve_storyboard_gradio,
    )

    approved_path = (
        serve_storyboard_gradio(
            plan_path,
            wait_for_approval=True,
        )
    )

    plan = load_plan(
        approved_path
    )

    # --------------------------------------------------------
    # GENERATION REQUIRES APPROVAL
    # --------------------------------------------------------

    require_approval(
        plan
    )

    # --------------------------------------------------------
    # WORKERS
    # --------------------------------------------------------

    runtime_workers = None

    if args.workers:

        workers = list(
            args.workers
        )

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
            item[
                "url"
            ]
            for item
            in runtime_workers.values()
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
                    "status":
                        "completed",
                    "profile":
                        plan.get(
                            "profile",
                            args.profile,
                        ),
                    "upscale_enabled":
                        result[
                            "upscale_enabled"
                        ],
                    "shot_outputs": [
                        str(path)
                        for path
                        in result[
                            "shot_outputs"
                        ]
                    ],
                    "final_video":
                        str(
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

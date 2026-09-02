from __future__ import annotations

import argparse
import json
import sys
import uuid
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


from planner.config import RUNTIME

def discover_gpu_ids():

    import torch

    if not torch.cuda.is_available():

        raise RuntimeError(
            "NVIDIA CUDA GPU is required "
            "for production generation."
        )

    count = (
        torch.cuda.device_count()
    )

    if count <= 0:

        raise RuntimeError(
            "CUDA is available but no GPU "
            "devices were detected."
        )

    return list(
        range(
            count
        )
    )


def load_clients(
    workers,
):
    """Create ComfyUI clients while preserving physical GPU IDs.

    ``workers`` may contain either plain URLs (assigned IDs 0..N-1 for
    backwards compatibility) or ``(gpu_id, url)`` pairs produced by
    H3Runtime.launch_workers(). Keeping the actual GPU ID is important
    when H3_GPU_IDS is non-contiguous, such as ``2,3``.
    """

    from execution.comfy_client import (
        ComfyClient,
    )

    clients = {}

    for index, item in enumerate(
        workers
    ):

        if isinstance(item, (tuple, list)) and len(item) == 2:
            gpu_id = int(item[0])
            url = str(item[1]).strip()
        elif isinstance(item, dict):
            if "gpu_id" not in item or "url" not in item:
                raise ValueError(
                    "Worker mapping dictionaries must contain gpu_id and url."
                )
            gpu_id = int(item["gpu_id"])
            url = str(item["url"]).strip()
        else:
            gpu_id = index
            url = str(item).strip()

        if not url:
            raise ValueError(
                f"ComfyUI worker URL is empty for GPU {gpu_id}."
            )

        if gpu_id in clients:
            raise ValueError(
                f"Duplicate ComfyUI worker GPU ID: {gpu_id}."
            )

        runtime_cfg = dict(RUNTIME.get("runtime", {}) or {})
        client = ComfyClient(
            base_url=url,
            timeout=float(runtime_cfg.get("comfyui_request_timeout_seconds", 60)),
            request_retries=int(runtime_cfg.get("comfyui_request_retries", 3)),
        )

        if not client.health_check():

            raise RuntimeError(
                "ComfyUI worker unavailable: "
                f"GPU {gpu_id} at {url}"
            )

        clients[gpu_id] = client

    if not clients:
        raise RuntimeError(
            "At least one ComfyUI worker is required."
        )

    return clients


def load_plan(
    path: Path,
) -> dict:

    path = (
        Path(path)
        .resolve()
    )

    if not path.is_file():

        raise FileNotFoundError(
            path
        )

    try:

        plan = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Production plan contains invalid JSON:\n"
            f"{path}\n"
            f"{exc}"
        ) from exc

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

    path = (
        Path(path)
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    temporary.write_text(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def _safe_production_id(
    value: str,
) -> str:

    cleaned = "".join(
        character
        if (
            character.isalnum()
            or character in "._-"
        )
        else "_"
        for character
        in str(value)
    ).strip(
        "._-"
    )

    if not cleaned:

        return (
            "production_"
            + uuid.uuid4().hex
        )

    return cleaned[:128]


def create_cli_plan_path(
    plan: dict,
) -> Path:

    production_id = str(
        plan.get(
            "production_id",
            "",
        )
        or ""
    ).strip()

    if not production_id:

        production_id = (
            "production_"
            + uuid.uuid4().hex
        )

    production_id = (
        _safe_production_id(
            production_id
        )
    )

    plan[
        "production_id"
    ] = production_id

    production_dir = (
        ROOT
        / "data"
        / "production"
        / production_id
    )

    production_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        production_dir
        / "story_preview.json"
    )


def require_approval(
    plan: dict,
) -> None:

    approval = (
        plan.get(
            "approval",
            {},
        )
        or {}
    )

    status = (
        approval.get(
            "status"
        )
    )

    if status not in {
        "approved",
        "completed",
    }:

        raise RuntimeError(
            "Storyboard has not been approved.\n"
            "Use --storyboard to open the Gradio UI, "
            "or provide an approved plan with --plan."
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
        help=(
            "Story direction mode."
        ),
    )

    parser.add_argument(
        "--profile",
        default="base",
        choices=[
            "base",
            "turbo",
            "upscale",
        ],
        help=(
            "H3 production profile."
        ),
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
            "Launch the canonical Gradio storyboard UI. "
            "Approval and H3 generation are handled by the UI."
        ),
    )

    # Retained for command-line compatibility with older
    # launchers. The canonical Gradio implementation owns
    # its current port configuration.
    parser.add_argument(
        "--storyboard-port",
        type=int,
        default=8765,
        help=argparse.SUPPRESS,
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
            "Enable H3 3D latent upscaling and "
            "MMH3 Ultimate Upscale before final "
            "720p delivery."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # CANONICAL INTERACTIVE FLOW
    # ========================================================

    if args.storyboard:

        if args.plan:

            raise RuntimeError(
                "--storyboard is an interactive UI flow "
                "and cannot be combined with --plan."
            )

        from ui.storyboard_gradio import (
            serve_storyboard_gradio,
        )

        serve_storyboard_gradio(
            initial_story=args.story,
            initial_mode=args.mode,
        )

        return 0

    # ========================================================
    # LOAD OR CREATE PLAN
    # ========================================================

    created_new_plan = False

    if args.plan:

        plan_path = (
            Path(
                args.plan
            )
            .resolve()
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

        plan_path = (
            create_cli_plan_path(
                plan
            )
        )

        created_new_plan = True

    # ========================================================
    # UPSCALE OVERRIDE
    # ========================================================

    if args.upscale:

        plan[
            "upscale_enabled"
        ] = True

    elif (
        args.profile
        == "upscale"
    ):

        plan[
            "upscale_enabled"
        ] = True

    # ========================================================
    # PLAN PERSISTENCE
    # ========================================================

    if (
        created_new_plan
        or args.upscale
        or args.profile == "upscale"
    ):

        save_plan(
            plan_path,
            plan,
        )

    # ========================================================
    # PREVIEW
    # ========================================================

    preview = {
        "preview": True,

        "story_mode": plan.get(
            "story_mode"
        ),

        "profile": plan.get(
            "profile"
        ),

        "production_id": plan.get(
            "production_id"
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

    if args.preview:

        return 0

    # ========================================================
    # APPROVAL
    # ========================================================

    require_approval(
        plan
    )

    # ========================================================
    # WORKERS
    # ========================================================

    runtime_workers = None

    if args.workers:

        workers = list(
            args.workers
        )

    else:

        from execution.h3_runtime import (
            H3Runtime,
        )

        gpu_ids = (
            discover_gpu_ids()
        )

        runtime_workers = (
            H3Runtime.launch_workers(
                ROOT,
                gpu_ids,
            )
        )

        workers = [
            (
                int(gpu_id),
                item[
                    "url"
                ],
            )
            for gpu_id, item
            in runtime_workers.items()
        ]

    try:

        clients = (
            load_clients(
                workers
            )
        )

        # ----------------------------------------------------
        # Live worker validation
        # ----------------------------------------------------

        if runtime_workers is not None:

            from kaggle.verify_live_runtime import (
                check_worker,
            )

            for worker in (
                runtime_workers.values()
            ):

                check_worker(
                    worker[
                        "port"
                    ]
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

        final_video = Path(
            result[
                "final_video"
            ]
        ).resolve()

        if not final_video.is_file():

            raise RuntimeError(
                "Production runner reported completion "
                "but final video does not exist:\n"
                f"{final_video}"
            )

        if final_video.stat().st_size <= 0:

            raise RuntimeError(
                "Production runner produced an empty final video:\n"
                f"{final_video}"
            )

        print(
            json.dumps(
                {
                    "status":
                        "completed",

                    "production_id":
                        result[
                            "production_id"
                        ],

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
                        str(
                            path
                        )
                        for path
                        in result[
                            "shot_outputs"
                        ]
                    ],

                    "final_video":
                        str(
                            final_video
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

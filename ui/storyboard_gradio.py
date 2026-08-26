from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import traceback
import uuid
from datetime import datetime
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


class ProductionController:

    def __init__(
        self,
    ):
        # One H3 generation is intentionally allowed at a time
        # because the current runtime owns the complete local
        # GPU/ComfyUI worker lifecycle.
        #
        # Storyboard state itself is session-scoped through
        # gr.State and is not stored on this controller.
        self._lock = threading.Lock()

    @staticmethod
    def _production_id() -> str:
        return (
            "production_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + "_"
            + uuid.uuid4().hex[:12]
        )

    @staticmethod
    def _session_dir(
        production_id: str,
    ) -> Path:

        path = (
            ROOT
            / "data"
            / "production"
            / "sessions"
            / production_id
        )

        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        return path

    @staticmethod
    def _save_session_plan(
        plan: dict,
    ) -> tuple[str, Path]:

        production_id = str(
            plan.get(
                "production_id",
                "",
            )
            or ""
        ).strip()

        if not production_id:
            production_id = (
                ProductionController
                ._production_id()
            )

        plan[
            "production_id"
        ] = production_id

        session_dir = (
            ProductionController
            ._session_dir(
                production_id
            )
        )

        plan_path = (
            session_dir
            / "story_preview.json"
        )

        plan[
            "production_plan_path"
        ] = str(
            plan_path
        )

        plan_path.write_text(
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return (
            production_id,
            plan_path,
        )

    @staticmethod
    def _load_state_plan(
        state: dict | None,
    ) -> tuple[dict, Path]:

        if not isinstance(
            state,
            dict,
        ):
            raise RuntimeError(
                "Storyboard session state is missing."
            )

        plan_path_value = str(
            state.get(
                "plan_path",
                "",
            )
            or ""
        ).strip()

        if not plan_path_value:
            raise RuntimeError(
                "Generate a valid storyboard first."
            )

        plan_path = Path(
            plan_path_value
        ).resolve()

        if not plan_path.is_file():
            raise FileNotFoundError(
                f"Storyboard plan does not exist:\n"
                f"{plan_path}"
            )

        plan = json.loads(
            plan_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            plan,
            dict,
        ):
            raise RuntimeError(
                "Storyboard plan must be a JSON object."
            )

        return (
            plan,
            plan_path,
        )

    # ========================================================
    # STORYBOARD GENERATION
    # ========================================================

    def generate_storyboard(
        self,
        story: str,
        mode: str,
        state: dict | None = None,
    ):

        story = str(
            story or ""
        ).strip()

        if not story:
            return (
                "### ERROR\nPlease write a story or premise.",
                "",
                "",
                "",
                "",
                None,
                state
                or {
                    "status": "empty"
                },
            )

        if not self._lock.acquire(
            blocking=False
        ):
            return (
                "### BUSY\nAnother production operation is running.",
                "",
                "",
                "",
                "",
                None,
                state
                or {
                    "status": "busy"
                },
            )

        try:

            # Production Gradio must always use the local director.
            os.environ[
                "H3_DIRECTOR_ENABLED"
            ] = "1"

            from planner.config import (
                director_enabled,
            )

            if not director_enabled():

                raise RuntimeError(
                    "Qwen director is disabled. "
                    "Production Gradio requires the local "
                    "Qwen3-14B director."
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
                    mode=mode,
                    user_input=story,
                    workflow_mode="auto",
                    profile="turbo",
                )
            )

            if plan.get(
                "director_pending",
                False,
            ):
                raise RuntimeError(
                    "Production plan is still waiting for "
                    "the Qwen director."
                )

            characters = (
                plan.get(
                    "characters",
                    [],
                )
                or []
            )

            scenes = (
                plan.get(
                    "scenes",
                    [],
                )
                or []
            )

            shots = (
                plan.get(
                    "shots",
                    [],
                )
                or []
            )

            if not scenes:
                raise RuntimeError(
                    "Qwen director returned no scenes."
                )

            if not shots:
                raise RuntimeError(
                    "Qwen director returned no shots."
                )

            if not isinstance(
                characters,
                list,
            ):
                raise RuntimeError(
                    "Invalid Qwen character plan."
                )

            plan[
                "profile"
            ] = "turbo"

            plan[
                "upscale_enabled"
            ] = True

            plan[
                "approval"
            ] = {
                "status": "draft",
                "approved_at": None,
            }

            production_id = (
                self._production_id()
            )

            plan[
                "production_id"
            ] = production_id

            (
                production_id,
                plan_path,
            ) = self._save_session_plan(
                plan
            )

            # CHARACTER PREVIEW

            character_text = "\n\n".join(
                (
                    f"### {character.get('name', '')}\n"
                    f"**Role:** {character.get('role', '')}\n\n"
                    f"{character.get('description', '')}\n\n"
                    f"**Personality:** "
                    f"{character.get('personality', '')}\n\n"
                    f"**Appearance:** "
                    f"{json.dumps(character.get('appearance', {}), ensure_ascii=False)}\n\n"
                    f"**Continuity:** "
                    f"{', '.join(character.get('continuity_rules', []) or [])}"
                )
                for character
                in characters
            )

            # SCENE PREVIEW

            scene_text = "\n\n".join(
                (
                    f"### {scene.get('scene_id', '')} — "
                    f"{scene.get('location', '')}\n\n"
                    f"{scene.get('description', '')}\n\n"
                    f"**Time:** {scene.get('time_of_day', '')}\n"
                    f"**Mood:** {scene.get('mood', '')}\n"
                    f"**Lighting:** {scene.get('lighting', '')}\n"
                    f"**Characters:** "
                    f"{', '.join(scene.get('characters', []) or [])}\n\n"
                    f"**Continuity:** "
                    f"{scene.get('continuity_notes', '')}"
                )
                for scene
                in scenes
            )

            # SHOT PREVIEW

            shot_text = "\n\n".join(
                (
                    f"### {shot.get('shot_id', '')}\n"
                    f"**Scene:** {shot.get('scene_id', '')}\n"
                    f"**Duration:** {shot.get('duration_seconds', '')} sec\n"
                    f"**Characters:** "
                    f"{', '.join(shot.get('characters', []) or [])}\n"
                    f"**Camera:** {shot.get('camera_shot', '')}\n"
                    f"**Movement:** {shot.get('camera_movement', '')}\n"
                    f"**Lighting:** {shot.get('lighting', '')}\n"
                    f"**Action:** {shot.get('action', '')}\n\n"
                    f"**Visual Direction:** "
                    f"{shot.get('detailed_description', '') or shot.get('visual_prompt', '')}\n\n"
                    f"**Soundscape:** "
                    f"{shot.get('overall_soundscape', '')}\n"
                    f"**Music:** "
                    f"{shot.get('non_diegetic_music', '')}\n"
                    f"**Dialogue:** "
                    f"{shot.get('speech_text', '')}\n"
                    f"**Continuity:** "
                    f"{shot.get('continuity_notes', '')}"
                )
                for shot
                in shots
            )

            summary = (
                "### STORYBOARD READY\n\n"
                f"**Production ID:** `{production_id}`\n\n"
                f"**Mode:** `{mode}`\n\n"
                f"**Characters:** {len(characters)}\n\n"
                f"**Scenes:** {len(scenes)}\n\n"
                f"**Shots:** {len(shots)}\n\n"
                "**Production:** H3 Turbo 8-step + "
                "3D latent upscale + Ultimate Upscale\n\n"
                "**Delivery:** 1280×720"
            )

            session_state = {
                "status": "draft",
                "production_id": production_id,
                "plan_path": str(
                    plan_path
                ),
            }

            return (
                summary,
                character_text
                or "No named characters generated.",
                scene_text,
                shot_text,
                "READY — review the storyboard, then approve it.",
                None,
                session_state,
            )

        except Exception as exc:

            traceback.print_exc()

            details = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc()}"
            )

            return (
                "### STORYBOARD GENERATION FAILED",
                "",
                "",
                "",
                "```text\n"
                + details
                + "\n```",
                None,
                state
                or {
                    "status": "error"
                },
            )

        finally:

            self._lock.release()

    # ========================================================
    # APPROVAL + H3 PRODUCTION
    # ========================================================

    def approve_and_generate(
        self,
        state: dict | None,
    ):

        try:
            (
                plan,
                plan_path,
            ) = self._load_state_plan(
                state
            )

        except Exception as exc:

            return (
                "### ERROR\n"
                + str(exc),
                None,
                state
                or {
                    "status": "error"
                },
            )

        if not self._lock.acquire(
            blocking=False
        ):
            return (
                "### BUSY\nA production job is already running.",
                None,
                state,
            )

        runtime_workers = None

        try:

            scenes = (
                plan.get(
                    "scenes",
                    [],
                )
                or []
            )

            shots = (
                plan.get(
                    "shots",
                    [],
                )
                or []
            )

            if not scenes or not shots:
                raise RuntimeError(
                    "The storyboard is incomplete."
                )

            approval = (
                plan.get(
                    "approval",
                    {},
                )
                or {}
            )

            if (
                approval.get(
                    "status"
                )
                == "completed"
            ):
                return (
                    "### COMPLETE\n"
                    "This production has already completed.",
                    plan.get(
                        "final_video"
                    ),
                    state,
                )

            plan[
                "approval"
            ] = {
                "status": "approved",
                "approved_at": (
                    datetime.now().isoformat()
                ),
            }

            plan_path.write_text(
                json.dumps(
                    plan,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "No NVIDIA CUDA GPU is available."
                )

            from execution.h3_runtime import (
                H3Runtime,
            )

            from execution.comfy_client import (
                ComfyClient,
            )

            from execution.production_runner import (
                ProductionRunner,
            )

            gpu_ids = list(
                range(
                    torch.cuda.device_count()
                )
            )

            if not gpu_ids:
                raise RuntimeError(
                    "No CUDA GPU was detected."
                )

            runtime_workers = (
                H3Runtime.launch_workers(
                    ROOT,
                    gpu_ids,
                )
            )

            clients = {}

            for gpu_id, worker in (
                runtime_workers.items()
            ):

                client = ComfyClient(
                    base_url=worker[
                        "url"
                    ],
                    timeout=60,
                    request_retries=3,
                )

                if not client.health_check():
                    raise RuntimeError(
                        "ComfyUI worker unavailable: "
                        f"{worker['url']}"
                    )

                # Verify actual live custom-node registration,
                # not just HTTP health.
                from kaggle.verify_live_runtime import (
                    check_worker,
                )

                check_worker(
                    worker["port"]
                )

                clients[
                    gpu_id
                ] = client

            plan[
                "production_id"
            ] = str(
                plan.get(
                    "production_id",
                    "",
                )
                or self._production_id()
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
            )

            plan[
                "production_id"
            ] = result[
                "production_id"
            ]

            plan[
                "final_video"
            ] = str(
                final_video
            )

            plan[
                "approval"
            ] = {
                "status": "completed",
                "approved_at": (
                    approval.get(
                        "approved_at"
                    )
                ),
                "completed_at": (
                    datetime.now().isoformat()
                ),
            }

            plan_path.write_text(
                json.dumps(
                    plan,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            updated_state = dict(
                state
                or {}
            )

            updated_state[
                "status"
            ] = "completed"

            updated_state[
                "production_id"
            ] = result[
                "production_id"
            ]

            updated_state[
                "plan_path"
            ] = str(
                plan_path
            )

            return (
                "### VIDEO GENERATION COMPLETE ✅\n\n"
                f"Production: `{result['production_id']}`\n\n"
                f"Final video: `{final_video}`\n\n"
                "Profile: H3 Turbo 8-step\n\n"
                "Upscale: H3 3D latent + "
                "MMH3 Ultimate Upscale\n\n"
                "Delivery: 1280×720",
                str(
                    final_video
                ),
                updated_state,
            )

        except Exception as exc:

            traceback.print_exc()

            details = (
                f"{type(exc).__name__}: {exc}\n\n"
                f"{traceback.format_exc()}"
            )

            return (
                "### PRODUCTION FAILED\n\n"
                "```text\n"
                + details
                + "\n```",
                None,
                state,
            )

        finally:

            if runtime_workers is not None:

                try:

                    from execution.h3_runtime import (
                        H3Runtime,
                    )

                    H3Runtime.stop_workers(
                        runtime_workers
                    )

                except Exception:
                    traceback.print_exc()

            self._lock.release()


def build_app(
    controller: ProductionController | None = None,
    initial_story: str | None = None,
    initial_mode: str = "ai_story",
):

    try:
        import gradio as gr

    except ImportError as exc:

        raise RuntimeError(
            "Gradio is not installed. "
            "Run kaggle/bootstrap.py first."
        ) from exc

    controller = (
        controller
        or ProductionController()
    )

    session_state = gr.State(
        {
            "status": "empty"
        }
    )

    with gr.Blocks(
        title=(
            "MiniMax H3 AI Video Generator"
        ),
    ) as demo:

        gr.Markdown(
            "# MiniMax H3 AI Video Generator\n\n"
            "Write your story naturally. "
            "Qwen3-14B acts as the director and develops "
            "the story, characters, scenes and cinematic shots."
        )

        story = gr.Textbox(
            label="Your Story",
            value=(
                initial_story
                or ""
            ),
            placeholder=(
                "Tell me what you want the video to be about..."
            ),
            lines=16,
        )

        mode = gr.Radio(
            choices=[
                (
                    "AI Story",
                    "ai_story",
                ),
                (
                    "Expand Story",
                    "expand_user_story",
                ),
                (
                    "Preserve Story",
                    "preserve_user_story",
                ),
            ],
            value=(
                initial_mode
                if initial_mode
                in {
                    "ai_story",
                    "expand_user_story",
                    "preserve_user_story",
                }
                else "ai_story"
            ),
            label="Story Mode",
        )

        generate = gr.Button(
            "Generate Storyboard",
            variant="primary",
        )

        status = gr.Markdown(
            "Write your story and press Enter."
        )

        with gr.Accordion(
            "Characters",
            open=True,
        ):
            characters = gr.Markdown()

        with gr.Accordion(
            "Scenes",
            open=True,
        ):
            scenes = gr.Markdown()

        with gr.Accordion(
            "Shots & Director Plan",
            open=True,
        ):
            shots = gr.Markdown()

        approve = gr.Button(
            "Approve & Generate Video",
            variant="primary",
        )

        result_status = gr.Markdown()

        final_video = gr.Video(
            label="Final Video",
        )

        generate_outputs = [
            status,
            characters,
            scenes,
            shots,
            result_status,
            final_video,
            session_state,
        ]

        generate.click(
            fn=controller.generate_storyboard,
            inputs=[
                story,
                mode,
                session_state,
            ],
            outputs=generate_outputs,
        )

        story.submit(
            fn=controller.generate_storyboard,
            inputs=[
                story,
                mode,
                session_state,
            ],
            outputs=generate_outputs,
        )

        approve.click(
            fn=controller.approve_and_generate,
            inputs=[
                session_state,
            ],
            outputs=[
                result_status,
                final_video,
                session_state,
            ],
        )

    return demo


def serve_storyboard_gradio(
    plan_path: Path | None = None,
    wait_for_approval: bool = False,
    initial_story: str | None = None,
    initial_mode: str = "ai_story",
):

    # plan_path and wait_for_approval are retained in the
    # function signature for compatibility with older callers.
    #
    # The canonical production flow is now fully session-scoped
    # inside the Gradio application.

    os.environ[
        "H3_DIRECTOR_ENABLED"
    ] = "1"

    controller = (
        ProductionController()
    )

    demo = build_app(
        controller=controller,
        initial_story=initial_story,
        initial_mode=initial_mode,
    )

    # Keep share=True because Kaggle is expected to expose
    # the application through a Gradio share URL.
    demo.launch(
        server_name="0.0.0.0",
        server_port=8765,
        share=True,
        show_error=True,
    )

    return (
        Path(plan_path)
        if plan_path is not None
        else None
    )


def main():

    serve_storyboard_gradio()


if __name__ == "__main__":
    main()

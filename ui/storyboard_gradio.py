from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


class ProductionController:

    def __init__(
        self,
    ):
        self._lock = threading.Lock()
        self._plan = None
        self._plan_path = None

    # ========================================================
    # STORYBOARD GENERATION
    # ========================================================

    def generate_storyboard(
        self,
        story: str,
        mode: str,
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
            )

        try:

            from pipeline.production_orchestrator import (
                ProductionOrchestrator,
            )

            orchestrator = (
                ProductionOrchestrator()
            )

            # Production policy:
            # Turbo 8-step + 3D latent upscale
            # + Ultimate Upscale + 1280x720 delivery.
            plan = (
                orchestrator
                .create_production_plan(
                    mode=mode,
                    user_input=story,
                    workflow_mode="auto",
                    profile="turbo",
                )
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

            plan_path = Path(
                plan[
                    "production_plan_path"
                ]
            )

            plan_path.write_text(
                json.dumps(
                    plan,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            self._plan = plan
            self._plan_path = plan_path

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
                f"**Mode:** `{mode}`\n\n"
                f"**Characters:** {len(characters)}\n\n"
                f"**Scenes:** {len(scenes)}\n\n"
                f"**Shots:** {len(shots)}\n\n"
                "**Production:** H3 Turbo 8-step + "
                "3D latent upscale + Ultimate Upscale\n\n"
                "**Delivery:** 1280×720"
            )

            return (
                summary,
                character_text
                or "No characters generated.",
                scene_text
                or "No scenes generated.",
                shot_text
                or "No shots generated.",
                "READY — review the storyboard, then approve it.",
                None,
            )

        except Exception as exc:

            return (
                "### STORYBOARD GENERATION FAILED",
                "",
                "",
                "",
                (
                    f"**{type(exc).__name__}:** "
                    f"{exc}"
                ),
                None,
            )

    # ========================================================
    # APPROVAL + H3 PRODUCTION
    # ========================================================

    def approve_and_generate(
        self,
    ):

        if (
            self._plan is None
            or self._plan_path is None
        ):

            return (
                "### ERROR\nGenerate a storyboard first.",
                None,
            )

        if not self._lock.acquire(
            blocking=False
        ):

            return (
                "### BUSY\nA production job is already running.",
                None,
            )

        runtime_workers = None

        try:

            self._plan[
                "approval"
            ] = {
                "status": "approved",
                "approved_at": (
                    datetime.now().isoformat()
                ),
            }

            self._plan_path.write_text(
                json.dumps(
                    self._plan,
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

                clients[
                    gpu_id
                ] = client

            result = (
                ProductionRunner(
                    project_root=ROOT,
                    comfy_clients=clients,
                )
                .run(
                    self._plan
                )
            )

            final_video = Path(
                result[
                    "final_video"
                ]
            )

            return (
                "### VIDEO GENERATION COMPLETE ✅\n\n"
                f"Final video: `{final_video}`\n\n"
                "Profile: H3 Turbo 8-step\n\n"
                "Upscale: H3 3D latent + "
                "MMH3 Ultimate Upscale\n\n"
                "Delivery: 1280×720",
                str(final_video),
            )

        except Exception as exc:

            return (
                "### PRODUCTION FAILED\n\n"
                f"**{type(exc).__name__}:** {exc}",
                None,
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
                    pass

            self._lock.release()


def build_app():

    try:

        import gradio as gr

    except ImportError as exc:

        raise RuntimeError(
            "Gradio is not installed. "
            "Run kaggle/bootstrap.py first."
        ) from exc

    controller = (
        ProductionController()
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
            value="ai_story",
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
        ]

        generate.click(
            fn=controller.generate_storyboard,
            inputs=[
                story,
                mode,
            ],
            outputs=generate_outputs,
        )

        story.submit(
            fn=controller.generate_storyboard,
            inputs=[
                story,
                mode,
            ],
            outputs=generate_outputs,
        )

        approve.click(
            fn=controller.approve_and_generate,
            inputs=[],
            outputs=[
                result_status,
                final_video,
            ],
        )

    return demo


def serve_storyboard_gradio(
    plan_path: Path | None = None,
    wait_for_approval: bool = False,
):

    demo = build_app()

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

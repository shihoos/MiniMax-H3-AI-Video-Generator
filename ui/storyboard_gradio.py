from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def load_plan(
    plan_path: Path,
) -> dict:

    if not plan_path.is_file():
        raise FileNotFoundError(
            plan_path
        )

    data = json.loads(
        plan_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        dict,
    ):
        raise RuntimeError(
            "Production plan must be a JSON object."
        )

    data.setdefault(
        "approval",
        {
            "status": "draft",
            "approved_at": None,
        },
    )

    return data


def save_plan(
    plan_path: Path,
    plan: dict,
) -> None:

    plan["updated_at"] = (
        datetime.now().isoformat()
    )

    plan_path.write_text(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _shot_text(
    shot: dict,
) -> str:

    return "\n".join(
        [
            f"Shot: {shot.get('shot_id', '')}",
            f"Scene: {shot.get('scene_id', '')}",
            f"Duration: {shot.get('duration_seconds', '')}",
            f"Characters: {', '.join(shot.get('characters', []) or [])}",
            f"Camera: {shot.get('camera_shot', '')}",
            f"Movement: {shot.get('camera_movement', '')}",
            f"Lighting: {shot.get('lighting', '')}",
            f"Action: {shot.get('action', '')}",
            f"Visual: {shot.get('visual_prompt', '')}",
            f"Dialogue: {shot.get('speech_text', '')}",
            f"Sound: {shot.get('overall_soundscape', '')}",
            f"Continuity: {shot.get('continuity_notes', '')}",
        ]
    )


def build_interface(
    plan_path: Path,
    approval_event: threading.Event,
):

    plan = load_plan(
        plan_path
    )

    with gr.Blocks(
        title="MiniMax H3 Storyboard",
    ) as demo:

        gr.Markdown(
            "# MiniMax H3 Storyboard\n"
            "Qwen3-14B Director → Characters → Scenes → Shots → H3"
        )

        status = gr.Markdown(
            "### DRAFT — review the production plan before generation."
        )

        with gr.Row():

            with gr.Column(
                scale=1
            ):

                story_mode = gr.Textbox(
                    value=str(
                        plan.get(
                            "story_mode",
                            "",
                        )
                    ),
                    label="Story Mode",
                    interactive=False,
                )

                profile = gr.Textbox(
                    value=str(
                        plan.get(
                            "profile",
                            "",
                        )
                    ),
                    label="Generation Profile",
                    interactive=False,
                )

                story = gr.Textbox(
                    value=str(
                        plan.get(
                            "story",
                            "",
                        )
                    ),
                    label="Story",
                    lines=12,
                )

            with gr.Column(
                scale=1
            ):

                gr.Markdown(
                    "## Characters"
                )

                character_summary = "\n\n".join(
                    [
                        (
                            f"**{character.get('name', '')}** "
                            f"— {character.get('role', '')}\n\n"
                            f"{character.get('description', '')}"
                        )
                        for character
                        in (
                            plan.get(
                                "characters",
                                [],
                            )
                            or []
                        )
                    ]
                )

                gr.Markdown(
                    character_summary
                    or "No characters."
                )

        gr.Markdown(
            "## Scenes"
        )

        scene_summary = "\n\n".join(
            [
                (
                    f"### {scene.get('scene_id', '')} "
                    f"— {scene.get('location', '')}\n\n"
                    f"{scene.get('description', '')}\n\n"
                    f"**Mood:** {scene.get('mood', '')}  "
                    f"**Lighting:** {scene.get('lighting', '')}"
                )
                for scene
                in (
                    plan.get(
                        "scenes",
                        [],
                    )
                    or []
                )
            ]
        )

        gr.Markdown(
            scene_summary
            or "No scenes."
        )

        gr.Markdown(
            "## Shots"
        )

        shot_components = []

        for shot in (
            plan.get(
                "shots",
                [],
            )
            or []
        ):

            shot_box = gr.Textbox(
                value=_shot_text(
                    shot
                ),
                label=str(
                    shot.get(
                        "shot_id",
                        "Shot",
                    )
                ),
                lines=10,
            )

            shot_components.append(
                (
                    shot,
                    shot_box,
                )
            )

        save_button = gr.Button(
            "Save Draft",
            variant="secondary",
        )

        approve_button = gr.Button(
            "Approve Storyboard",
            variant="primary",
        )

        def save_current(
            story_value,
            *shot_values,
        ):

            current = load_plan(
                plan_path
            )

            current[
                "story"
            ] = story_value

            for (
                shot,
                value,
            ), edited in zip(
                shot_components,
                shot_values,
            ):

                shot[
                    "director_review"
                ] = edited

            current[
                "approval"
            ] = {
                "status": "draft",
                "approved_at": None,
            }

            save_plan(
                plan_path,
                current,
            )

            return (
                "### DRAFT SAVED"
            )

        save_button.click(
            fn=save_current,
            inputs=[
                story,
                *[
                    component
                    for (
                        _shot,
                        component
                    )
                    in shot_components
                ],
            ],
            outputs=status,
        )

        def approve_current(
            story_value,
            *shot_values,
        ):

            current = load_plan(
                plan_path
            )

            current[
                "story"
            ] = story_value

            for (
                shot,
                value,
            ), edited in zip(
                shot_components,
                shot_values,
            ):

                shot[
                    "director_review"
                ] = edited

            current[
                "approval"
            ] = {
                "status": "approved",
                "approved_at": (
                    datetime.now().isoformat()
                ),
            }

            save_plan(
                plan_path,
                current,
            )

            approval_event.set()

            return (
                "### APPROVED — generation may continue."
            )

        approve_button.click(
            fn=approve_current,
            inputs=[
                story,
                *[
                    component
                    for (
                        _shot,
                        component
                    )
                    in shot_components
                ],
            ],
            outputs=status,
        )

    return demo


def serve_storyboard_gradio(
    plan_path: Path,
    wait_for_approval: bool = True,
):

    approval_event = (
        threading.Event()
    )

    demo = build_interface(
        Path(
            plan_path
        ).resolve(),
        approval_event,
    )

    print(
        "=" * 80
    )

    print(
        "MINIMAX H3 GRADIO STORYBOARD UI"
    )

    print(
        "=" * 80
    )

    result = demo.launch(
        server_name="0.0.0.0",
        server_port=8765,
        share=True,
        show_error=True,
        prevent_thread_lock=True,
    )

    if isinstance(
        result,
        tuple,
    ):

        for value in result:
            if isinstance(
                value,
                str,
            ) and (
                "gradio.live"
                in value
            ):
                print(
                    f"STORYBOARD PUBLIC URL: {value}"
                )

    if not wait_for_approval:
        return Path(
            plan_path
        )

    print(
        "Waiting for storyboard approval..."
    )

    approval_event.wait()

    return Path(
        plan_path
    )

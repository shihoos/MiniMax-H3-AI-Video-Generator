from __future__ import annotations

import json
import os
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


from planner.config import (
    GRADIO_SHARE_ENV,
    STORYBOARD_HOST,
    STORYBOARD_PORT,
    storyboard_share_enabled,
)


SESSIONS_ROOT = (
    ROOT
    / "data"
    / "production"
    / "sessions"
)


class ProductionController:

    def __init__(self):

        self._lock = (
            threading.Lock()
        )

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
            SESSIONS_ROOT
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

        temporary = plan_path.with_name(
            f".{plan_path.name}.{uuid.uuid4().hex}.tmp"
        )

        try:
            temporary.write_text(
                json.dumps(
                    plan,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            temporary.replace(
                plan_path
            )
        finally:
            temporary.unlink(
                missing_ok=True
            )

        return (
            production_id,
            plan_path,
        )

    # ========================================================
    # SAVED DRAFTS
    # ========================================================

    @staticmethod
    def _draft_label(
        plan: dict,
        path: Path,
    ) -> str:

        mode = str(
            plan.get(
                "story_mode",
                "",
            )
            or ""
        )

        mode_labels = {
            "ai_story":
                "AI Story",

            "expand_user_story":
                "Expand Story",

            "preserve_user_story":
                "Preserve Story",
        }

        label = mode_labels.get(
            mode,
            mode,
        )

        production_id = str(
            plan.get(
                "production_id",
                path.parent.name,
            )
        )

        return (
            f"{label} — "
            f"{production_id}"
        )

    @classmethod
    def _saved_drafts(
        cls,
    ) -> list[tuple[str, str]]:

        if not SESSIONS_ROOT.exists():
            return []

        drafts: list[tuple[str, str]] = []

        for session_dir in (
            SESSIONS_ROOT.iterdir()
        ):

            if not session_dir.is_dir():
                continue

            plan_path = (
                session_dir
                / "story_preview.json"
            )

            if not plan_path.is_file():
                continue

            try:

                plan = json.loads(
                    plan_path.read_text(
                        encoding="utf-8"
                    )
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):

                continue

            if not isinstance(
                plan,
                dict,
            ):

                continue

            drafts.append(
                (
                    cls._draft_label(
                        plan,
                        plan_path,
                    ),
                    str(
                        plan_path
                    ),
                )
            )

        drafts.sort(
            key=lambda item: (
                Path(
                    item[1]
                ).stat().st_mtime
            ),
            reverse=True,
        )

        return drafts

    @classmethod
    def _draft_choices(
        cls,
    ) -> list[str]:

        return [
            label
            for label, _ in cls._saved_drafts()
        ]

    @classmethod
    def _draft_path_from_label(
        cls,
        label: str | None,
    ) -> str:

        wanted = str(
            label or ""
        ).strip()

        if not wanted:
            return ""

        for current_label, path in (
            cls._saved_drafts()
        ):

            if current_label == wanted:
                return path

        return ""

    @staticmethod
    def _load_plan(
        plan_path_value: str,
    ) -> tuple[dict, Path]:

        value = str(
            plan_path_value or ""
        ).strip()

        if not value:

            raise RuntimeError(
                "No saved storyboard was selected."
            )

        plan_path = (
            Path(value)
            .resolve()
        )

        try:

            plan_path.relative_to(
                SESSIONS_ROOT.resolve()
            )

        except ValueError as exc:

            raise RuntimeError(
                "Selected storyboard is outside "
                "the managed session directory."
            ) from exc

        if not plan_path.is_file():

            raise FileNotFoundError(
                plan_path
            )

        try:

            plan = json.loads(
                plan_path.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Saved storyboard is invalid JSON."
            ) from exc

        if not isinstance(
            plan,
            dict,
        ):

            raise RuntimeError(
                "Saved storyboard must be a JSON object."
            )

        return (
            plan,
            plan_path,
        )

    @staticmethod
    def _render_plan(
        plan: dict,
        plan_path: Path,
    ):

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

        mode = str(
            plan.get(
                "story_mode",
                "",
            )
            or ""
        )

        production_id = str(
            plan.get(
                "production_id",
                plan_path.parent.name,
            )
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

        if not character_text:

            character_text = (
                "No named characters are required "
                "for this story."
            )

        scene_text = "\n\n".join(
            (
                f"### {scene.get('scene_id', '')} — "
                f"{scene.get('title', '') or scene.get('location', '')}\n\n"
                f"{scene.get('description', '')}\n\n"
                f"**Location:** {scene.get('location', '')}\n"
                f"**Time:** {scene.get('time_of_day', '')}\n"
                f"**Weather:** {scene.get('weather', '')}\n"
                f"**Atmosphere:** {scene.get('atmosphere', '')}\n"
                f"**Mood:** {scene.get('mood', '')}\n"
                f"**Lighting:** {scene.get('lighting', '')}\n"
                f"**Color Temperature:** "
                f"{scene.get('color_temperature', '')}\n"
                f"**Environment Details:** "
                f"{', '.join(scene.get('environment_details', []) or [])}\n"
                f"**Key Props:** "
                f"{', '.join(scene.get('key_props', []) or [])}\n"
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
                f"**Duration:** "
                f"{shot.get('duration_seconds', '')} sec\n"
                f"**Characters:** "
                f"{', '.join(shot.get('characters', []) or [])}\n"
                f"**Camera:** "
                f"{shot.get('camera_shot', '')}\n"
                f"**Movement:** "
                f"{shot.get('camera_movement', '')}\n"
                f"**Lens / DOF:** "
                f"{shot.get('lens_and_depth_of_field', '')}\n"
                f"**Composition:** "
                f"{shot.get('composition_notes', '')}\n"
                f"**Lighting:** "
                f"{shot.get('lighting', '')}\n"
                f"**Color Temperature:** "
                f"{shot.get('color_temperature', '')}\n"
                f"**Mood:** "
                f"{shot.get('mood', '')}\n"
                f"**Action:** "
                f"{shot.get('action', '')}\n\n"
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

        visual_language = (
            plan.get(
                "visual_language",
                {},
            )
            or {}
        )

        if visual_language:

            visual_language_text = (
                "\n\n### VISUAL LANGUAGE\n\n"
                f"**Genre / Tone:** "
                f"{visual_language.get('genre_tone', '')}\n\n"
                f"**Color Palette:** "
                f"{visual_language.get('color_palette', '')}\n\n"
                f"**Lighting Philosophy:** "
                f"{visual_language.get('lighting_philosophy', '')}\n\n"
                f"**Camera Philosophy:** "
                f"{visual_language.get('camera_philosophy', '')}\n\n"
                f"**Pacing:** "
                f"{visual_language.get('pacing', '')}"
            )

        else:

            visual_language_text = (
                "\n\n### VISUAL LANGUAGE\n\n"
                "Not available for this storyboard "
                "(director did not run, or this draft "
                "predates the visual language feature)."
            )

        summary = (
            "### STORYBOARD READY\n\n"
            f"**Production ID:** `{production_id}`\n\n"
            f"**Mode:** `{mode}`\n\n"
            f"**Characters:** {len(characters)}\n\n"
            f"**Scenes:** {len(scenes)}\n\n"
            f"**Shots:** {len(shots)}\n\n"
            f"**Production profile:** "
            f"{plan.get('profile', 'base')}\n\n"
            f"**Workflow:** "
            f"{plan.get('workflow_mode', 'auto')}\n\n"
            f"**Upscale:** "
            f"{'enabled' if plan.get('upscale_enabled', False) else 'disabled'}\n\n"
            f"**Delivery:** "
            f"{plan.get('delivery_width', 1280)}×"
            f"{plan.get('delivery_height', 720)} @ "
            f"{plan.get('delivery_fps', 24)} FPS"
            f"{visual_language_text}"
        )

        approval = (
            plan.get(
                "approval",
                {},
            )
            or {}
        )

        status = (
            "READY — review the storyboard, "
            "then approve it."
        )

        if (
            approval.get(
                "status"
            )
            == "completed"
        ):

            status = (
                "### COMPLETE\n"
                "This production has already completed."
            )

        return (
            summary,
            character_text,
            scene_text,
            shot_text,
            status,
            plan.get(
                "final_video"
            ),
            str(
                plan_path
            ),
        )

    # ========================================================
    # GENERATION
    # ========================================================

    def generate_storyboard(
        self,
        story: str,
        mode: str,
        resume_id: str | None = None,
    ):

        story = str(
            story or ""
        ).strip()

        resume_id = str(
            resume_id or ""
        ).strip()

        if not story:

            return (
                "### ERROR\nPlease write a story or premise.",
                "",
                "",
                "",
                "",
                None,
                "",
                self._draft_choices(),
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
                "",
                self._draft_choices(),
            )

        try:

            os.environ[
                "H3_DIRECTOR_ENABLED"
            ] = "1"

            from planner.config import (
                director_enabled,
            )

            if not director_enabled():

                raise RuntimeError(
                    "Qwen director is disabled."
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
                    resume_session_id=(
                        resume_id
                        or None
                    ),
                )
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
                    "Production planner produced no scenes."
                )

            if not shots:

                raise RuntimeError(
                    "Production planner produced no shots."
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
                "status":
                    "draft",

                "approved_at":
                    None,
            }

            # ProductionOrchestrator already created and checkpointed the
            # canonical production_id. Preserve it so the UI storyboard and
            # director checkpoint refer to the same production session.
            (
                _,
                plan_path,
            ) = self._save_session_plan(
                plan
            )

            (
                summary,
                character_text,
                scene_text,
                shot_text,
                status,
                final_video,
                stored_path,
            ) = self._render_plan(
                plan,
                plan_path,
            )

            choices = (
                self._draft_choices()
            )

            return (
                summary,
                character_text,
                scene_text,
                shot_text,
                status,
                final_video,
                stored_path,
                choices,
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
                "",
                self._draft_choices(),
            )

        finally:

            self._lock.release()

    # ========================================================
    # PREVIEW
    # ========================================================

    def preview_saved(
        self,
        selected_label: str | None,
    ):

        try:

            path = (
                self._draft_path_from_label(
                    selected_label
                )
            )

            if not path:

                raise RuntimeError(
                    "Select a saved storyboard first."
                )

            (
                plan,
                plan_path,
            ) = self._load_plan(
                path
            )

            return self._render_plan(
                plan,
                plan_path,
            )

        except Exception as exc:

            return (
                "### PREVIEW FAILED\n"
                + str(exc),
                "",
                "",
                "",
                "",
                None,
                "",
            )

    def preview_latest_for_mode(
        self,
        mode: str,
    ):

        drafts = (
            self._saved_drafts()
        )

        for label, path in drafts:

            try:

                plan = json.loads(
                    Path(
                        path
                    ).read_text(
                        encoding="utf-8"
                    )
                )

            except (
                OSError,
                json.JSONDecodeError,
            ):

                continue

            if (
                plan.get(
                    "story_mode"
                )
                == mode
            ):

                return (
                    label,
                    *self._render_plan(
                        plan,
                        Path(path),
                    )
                )

        return (
            None,
            "No saved storyboard exists for this mode.",
            "",
            "",
            "",
            "",
            None,
            "",
        )

    # ========================================================
    # APPROVAL / VIDEO
    # ========================================================

    def approve_and_generate(
        self,
        plan_path_value: str,
    ):

        try:

            plan, plan_path = (
                self._load_plan(
                    plan_path_value
                )
            )

        except Exception as exc:

            return (
                "### ERROR\n"
                + str(exc),
                None,
                plan_path_value,
            )

        if not self._lock.acquire(
            blocking=False
        ):

            return (
                "### BUSY\nA production job is already running.",
                None,
                plan_path_value,
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
                    plan_path_value,
                )

            plan[
                "approval"
            ] = {
                "status":
                    "approved",

                "approved_at":
                    datetime.now().isoformat(),
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

                from kaggle.verify_live_runtime import (
                    check_worker,
                )

                check_worker(
                    worker[
                        "port"
                    ]
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
                    plan
                )
            )

            final_video = (
                Path(
                    result[
                        "final_video"
                    ]
                )
                .resolve()
            )

            if not final_video.is_file():

                raise RuntimeError(
                    "Production runner completed but "
                    "final video was not found:\n"
                    f"{final_video}"
                )

            if final_video.stat().st_size <= 0:

                raise RuntimeError(
                    "Production runner produced an empty "
                    "final video:\n"
                    f"{final_video}"
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
                "status":
                    "completed",

                "approved_at":
                    approval.get(
                        "approved_at"
                    ),

                "completed_at":
                    datetime.now().isoformat(),
            }

            plan_path.write_text(
                json.dumps(
                    plan,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            return (
                "### VIDEO GENERATION COMPLETE ✅\n\n"
                f"Production: `{result['production_id']}`\n\n"
                f"Final video: `{final_video}`\n\n"
                f"Profile: `{plan.get('profile', 'base')}`\n\n"
                f"Upscale: `{'enabled' if plan.get('upscale_enabled', False) else 'disabled'}`\n\n"
                f"Delivery: "
                f"{plan.get('delivery_width', 1280)}×"
                f"{plan.get('delivery_height', 720)} @ "
                f"{plan.get('delivery_fps', 24)} FPS",
                str(
                    final_video
                ),
                plan_path_value,
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
                plan_path_value,
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
            "Gradio is not installed."
        ) from exc

    controller = (
        controller
        or ProductionController()
    )

    def dropdown_update(
        selected_path: str | None = None,
        selected_label: str | None = None,
        preserve_value: str | None = None,
    ):
        choices = (
            controller._draft_choices()
        )

        value = None

        if selected_label:
            if selected_label in choices:
                value = selected_label

        elif selected_path:

            for label, path in (
                controller._saved_drafts()
            ):

                if str(path) == str(
                    selected_path
                ):
                    value = label
                    break

        elif preserve_value:

            if preserve_value in choices:
                value = preserve_value

        return gr.Dropdown(
        choices=choices,
        value=value,
        interactive=True,
        allow_custom_value=False,
        )

    def _stream_generation_ui(
        story_value: str,
        mode_value: str,
        resume_id_value: str,
    ):
        # Run the existing synchronous controller in a worker thread while
        # the Gradio callback yields heartbeat/status frames. This keeps the
        # browser/proxy connection alive during long Qwen planning without
        # changing the controller's locking/checkpoint behavior.
        result_holder = {}
        error_holder = {}
        done = threading.Event()

        def worker():
            try:
                result_holder["result"] = controller.generate_storyboard(
                    story_value,
                    mode_value,
                    resume_id_value,
                )
            except Exception as exc:  # controller already formats normal failures
                error_holder["error"] = exc
            finally:
                done.set()

        threading.Thread(
            target=worker,
            name="storyboard-generation",
            daemon=True,
        ).start()

        yield (
            "### PLANNING\nGenerating the storyboard with the Qwen Director...",
            "", "", "", "", None, "",
            dropdown_update(preserve_value=None),
        )

        while not done.wait(2.0):
            yield (
                "### PLANNING\nStill working — Qwen scene/shot planning is in progress...",
                "", "", "", "", None, "",
                dropdown_update(preserve_value=None),
            )

        if "error" in error_holder:
            raise error_holder["error"]

        result = list(result_holder["result"])
        stored_path = result[6] if len(result) > 6 else ""
        result[7] = dropdown_update(
            selected_path=stored_path if stored_path else None,
        )
        yield tuple(result)

    generate_storyboard_ui = _stream_generation_ui

    def regenerate_storyboard_ui(
        story_value: str,
        mode_value: str,
        resume_id_value: str,
    ):
        yield from _stream_generation_ui(
            story_value,
            mode_value,
            resume_id_value,
        )

    def preview_latest_ui(
        mode_value: str,
    ):
        result = (
            controller.preview_latest_for_mode(
                mode_value
            )
        )

        result = list(result)

        selected_label = (
            result[0]
            if result
            else None
        )

        result[0] = (
            dropdown_update(
                selected_label=selected_label
            )
        )

        return tuple(result)

    def refresh_saved_drafts_ui(
        current_value: str | None,
    ):
        return (
            dropdown_update(
                preserve_value=current_value
            )
        )

    with gr.Blocks(
        title="MiniMax H3 AI Video Generator",
    ) as demo:

        gr.Markdown(
            "# MiniMax H3 AI Video Generator\n\n"
            "Create a cinematic storyboard with Qwen3-14B, "
            "review saved drafts, and approve the version "
            "you want to render."
        )

        story = gr.Textbox(
            label="Your Story",
            value=initial_story or "",
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
            info=(
                "AI Story: create a new cinematic story from your premise. "
                "Expand Story: enrich the existing story while preserving its core. "
                "Preserve Story: keep the supplied story unchanged and only "
                "structure it for production."
            ),
        )

        resume_id = gr.Textbox(
            label="Resume Production ID (optional)",
            value="",
            placeholder=(
                "Paste a production_id here to resume a "
                "generation that stopped partway through "
                "(e.g. after a Kaggle session died). Leave "
                "empty to start a new production. The story "
                "text and mode above must exactly match the "
                "original request."
            ),
            lines=1,
        )

        with gr.Row():

            generate = gr.Button(
                "Generate Storyboard",
                variant="primary",
            )

            regenerate = gr.Button(
                "Regenerate",
            )

        gr.Markdown(
            "### Saved Storyboards"
        )

        saved_draft = gr.Dropdown(
            choices=(
                controller._draft_choices()
            ),
            label="Saved Draft",
            value=None,
            interactive=True,
            allow_custom_value=False,
        )

        with gr.Row():

            preview = gr.Button(
                "Preview Selected"
            )

            latest = gr.Button(
                "Preview Latest For Mode"
            )

            refresh = gr.Button(
                "Refresh Saved Drafts"
            )

        status = gr.Markdown(
            "Write your story and choose a mode."
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

        session_plan_path = gr.Textbox(
            value="",
            visible=False,
            interactive=False,
        )

        generation_outputs = [
            status,
            characters,
            scenes,
            shots,
            result_status,
            final_video,
            session_plan_path,
            saved_draft,
        ]

        generate.click(
            fn=generate_storyboard_ui,
            inputs=[
                story,
                mode,
                resume_id,
            ],
            outputs=generation_outputs,
        )

        regenerate.click(
            fn=regenerate_storyboard_ui,
            inputs=[
                story,
                mode,
                resume_id,
            ],
            outputs=generation_outputs,
        )

        preview.click(
            fn=controller.preview_saved,
            inputs=[
                saved_draft,
            ],
            outputs=[
                status,
                characters,
                scenes,
                shots,
                result_status,
                final_video,
                session_plan_path,
            ],
        )

        latest.click(
            fn=preview_latest_ui,
            inputs=[
                mode,
            ],
            outputs=[
                saved_draft,
                status,
                characters,
                scenes,
                shots,
                result_status,
                final_video,
                session_plan_path,
            ],
        )

        refresh.click(
            fn=refresh_saved_drafts_ui,
            inputs=[
                saved_draft,
            ],
            outputs=[
                saved_draft,
            ],
        )

        approve.click(
            fn=controller.approve_and_generate,
            inputs=[
                session_plan_path,
            ],
            outputs=[
                result_status,
                final_video,
                session_plan_path,
            ],
        )

    return demo


def serve_storyboard_gradio(
    plan_path: Path | None = None,
    wait_for_approval: bool = False,
    initial_story: str | None = None,
    initial_mode: str = "ai_story",
):

    del plan_path
    del wait_for_approval

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

    demo.launch(
        server_name=STORYBOARD_HOST,
        server_port=STORYBOARD_PORT,
        share=storyboard_share_enabled(),
        show_error=True,
    )

def main():
    serve_storyboard_gradio()


if __name__ == "__main__":
    main()

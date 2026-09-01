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


from pipeline.job_queue import ProductionJobQueue
from pipeline.timeline import ProductionTimeline
from pipeline.runtime_diagnostics import RuntimeDiagnostics
from pipeline.production_checkpoint import ProductionCheckpoint
from pipeline.retake_manager import RetakeManager
from ui.shot_view_model import shot_choices, render_shot_card
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
        self._job_queue = ProductionJobQueue(
            ROOT / "data" / "production" / "jobs.sqlite3"
        )
        self._job_queue.recover_stale(max_age_seconds=21600.0)
        self._queue_stop = threading.Event()
        self._queue_thread = threading.Thread(
            target=self._queue_worker_loop,
            name="h3-production-queue",
            daemon=True,
        )
        self._queue_thread.start()

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
    def _safe_name(value: str) -> str:
        """Sanitize a string for safe use as a filesystem directory name."""
        text = str(value or "").strip()
        cleaned = "".join(
            char if (char.isalnum() or char in "._-") else "_"
            for char in text
        )
        return cleaned[:96] or "production"

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
                f"{shot.get('continuity_notes', '')}\n"
                f"**Continuity Mode:** {shot.get('continuity_mode', 'chained')}\n"
                f"**QA:** {json.dumps(shot.get('quality_gate', {}) or {}, ensure_ascii=False)}\n"
                f"**Retake Recommended:** {bool(shot.get('retake_recommended', False))}"
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

            os.environ.setdefault(
                "H3_DIRECTOR_ENABLED",
                "1",
            )

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

    def shot_options(self, plan_path_value: str):
        try:
            plan, _ = self._load_plan(plan_path_value)
            choices = shot_choices(plan)
            return gr.update(choices=choices, value=(choices[0] if choices else None)), (render_shot_card(plan, choices[0]) if choices else "### No shots available")
        except Exception as exc:
            return gr.update(choices=[], value=None), "### ERROR\n" + str(exc)

    def shot_detail(self, plan_path_value: str, shot_id: str):
        try:
            plan, _ = self._load_plan(plan_path_value)
            return render_shot_card(plan, shot_id)
        except Exception as exc:
            return "### ERROR\n" + str(exc)

    def shot_preview(self, plan_path_value: str, shot_id: str):
        try:
            plan, _ = self._load_plan(plan_path_value)
            wanted = str(shot_id or "").strip()
            shot = next((s for s in (plan.get("shots", []) or []) if isinstance(s, dict) and str(s.get("shot_id", "")).strip() == wanted), None)
            if not shot:
                return None, "No shot selected."
            candidates = []
            for key in ("preview_path", "preview_file", "output_video", "video_path", "rendered_video"):
                value = str(shot.get(key, "") or "").strip()
                if value:
                    path = Path(value)
                    if path.is_file():
                        candidates.append(path)
            production_id = str(plan.get("production_id", "")).strip()
            preview_root = ROOT / "data" / "production" / production_id / "previews"
            if preview_root.is_dir():
                shot_preview_root = preview_root / self._safe_name(wanted)
                if shot_preview_root.is_dir():
                    candidates.extend(
                        path
                        for path in shot_preview_root.rglob("latest.*")
                        if path.is_file()
                    )
            if not candidates:
                return None, f"No visual preview is available for `{wanted}` yet."
            path = max(candidates, key=lambda item: item.stat().st_mtime)
            return str(path), f"Preview: `{path.name}` · Shot `{wanted}`"
        except Exception as exc:
            return None, "### ERROR\n" + str(exc)

    def shot_retake_defaults(self, plan_path_value: str, shot_id: str):
        try:
            plan, _ = self._load_plan(plan_path_value)
            wanted = str(shot_id or "").strip()
            shot = next((s for s in (plan.get("shots", []) or []) if isinstance(s, dict) and str(s.get("shot_id", "")).strip() == wanted), None)
            if not shot:
                return "", 0.0, 4.0, ""
            duration = max(4.0, min(15.0, float(shot.get("duration_seconds", 4.0) or 4.0)))
            q = shot.get("quality_gate", {}) or {}
            reason = str(q.get("decision_reason") or "Selective quality retake")
            return wanted, 0.0, duration, reason
        except Exception:
            return str(shot_id or ""), 0.0, 4.0, "Selective quality retake"

    # ========================================================
    # PREVIEW
    # ========================================================

    def latest_live_preview(self, plan_path_value: str):
        try:
            plan, _ = self._load_plan(plan_path_value)
            production_id = str(plan.get("production_id", "")).strip()
            root = ROOT / "data" / "production" / production_id / "previews"
            candidates = []
            if root.is_dir():
                for path in root.rglob("latest.*"):
                    if path.is_file():
                        candidates.append(path)
            if not candidates:
                return None, "No sampling preview is available yet."
            path = max(candidates, key=lambda item: item.stat().st_mtime)
            return str(path), f"Live preview: `{path.parent.name}`"
        except Exception as exc:
            return None, "### ERROR\n" + str(exc)

    def timeline_table(self, plan_path_value: str):
        try:
            plan, _ = self._load_plan(plan_path_value)
            timeline = ProductionTimeline(plan)
            return timeline.table(), f"### TIMELINE\n{len(plan.get('shots', []) or [])} shots · {plan.get('timeline', {}).get('total_duration_seconds', 0):.2f}s total"
        except Exception as exc:
            return [], "### ERROR\n" + str(exc)

    def apply_timeline_edits(self, plan_path_value: str, rows):
        try:
            plan, plan_path = self._load_plan(plan_path_value)
            ProductionTimeline(plan).apply_table(rows)
            ProductionTimeline.validate(plan)
            plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

            # Keep the durable planning checkpoint synchronized with any user
            # timeline edits. Otherwise the render runner would correctly
            # reject the edited plan as a fingerprint mismatch.
            production_id = str(plan.get("production_id", "")).strip()
            if production_id:
                store = ProductionCheckpoint(ROOT)
                try:
                    state = store.load(production_id)
                    if isinstance(state.get("director_plan"), dict):
                        state["director_plan"] = plan
                        state["plan_sha256"] = store.plan_digest(plan)
                        state["status"] = "ready"
                        state["stage"] = "timeline_edited"
                        state["error"] = ""
                        store.save(production_id, state)
                except FileNotFoundError:
                    pass

            return ProductionTimeline(plan).table(), "### TIMELINE UPDATED\nChanges are persisted to the production plan and checkpoint.", str(plan_path)
        except Exception as exc:
            return [], "### ERROR\n" + str(exc), str(plan_path_value or "")

    def runtime_health(self, plan_path_value: str):
        try:
            plan, _ = self._load_plan(plan_path_value)
            report = RuntimeDiagnostics(ROOT).collect()
            report["production_id"] = plan.get("production_id", "")
            return report, "### RUNTIME HEALTH\nDiagnostics collected."
        except Exception as exc:
            return {}, "### ERROR\n" + str(exc)

    def create_retake_request(self, plan_path_value: str, shot_id: str, start_seconds: float, end_seconds: float, reason: str):
        try:
            plan, _ = self._load_plan(plan_path_value)
            production_id = str(plan.get("production_id", "")).strip()
            request_path = RetakeManager(ROOT).request(
                production_id, shot_id,
                start_seconds=float(start_seconds or 0.0),
                end_seconds=float(end_seconds),
                reason=reason,
            )
            for shot in plan.get("shots", []) or []:
                if str(shot.get("shot_id", "")) == str(shot_id):
                    shot["retake_requested"] = True
                    shot["retake_start_seconds"] = float(start_seconds or 0.0)
                    shot["retake_end_seconds"] = float(end_seconds)
            production_id, plan_path = self._save_session_plan(plan)

            # Retake metadata is part of the semantic plan, so keep the
            # durable checkpoint fingerprint synchronized with the edited plan.
            if production_id:
                store = ProductionCheckpoint(ROOT)
                try:
                    state = store.load(production_id)
                    if isinstance(state.get("director_plan"), dict):
                        state["director_plan"] = plan
                        state["plan_sha256"] = store.plan_digest(plan)
                        state["status"] = "ready"
                        state["stage"] = "retake_requested"
                        state["error"] = ""
                        store.save(production_id, state)
                except FileNotFoundError:
                    pass

            return f"### RETAKE REQUESTED\n`{request_path}`", str(plan_path)
        except Exception as exc:
            return "### ERROR\n" + str(exc), str(plan_path_value or "")

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
        """Approve a saved plan and persist a render job without tying it to the browser lifetime."""
        try:
            plan, plan_path = self._load_plan(plan_path_value)
        except Exception as exc:
            return "### ERROR\n" + str(exc), None, plan_path_value

        approval = plan.get("approval", {}) or {}
        if approval.get("status") == "completed":
            return "### COMPLETE\nThis production has already completed.", plan.get("final_video"), plan_path_value

        production_id = str(plan.get("production_id", plan_path.parent.name)).strip()
        plan["production_id"] = production_id
        plan["approval"] = {
            "status": "approved",
            "approved_at": datetime.now().isoformat(),
        }
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        job_id = self._job_queue.submit(
            production_id,
            plan_path,
            {"profile": plan.get("profile", "turbo"), "upscale_enabled": bool(plan.get("upscale_enabled", False))},
        )
        plan["job_id"] = job_id
        plan["job_status"] = "queued"
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"### QUEUED\nProduction `{production_id}` has been queued.\n\nJob ID: `{job_id}`\n\nYou can close the browser; the queue state is persisted on disk.", None, str(plan_path)

    def refresh_job_status(self, plan_path_value: str):
        try:
            plan, plan_path = self._load_plan(plan_path_value)
        except Exception as exc:
            return "### ERROR\n" + str(exc), None, plan_path_value
        job_id = str(plan.get("job_id", "") or "").strip()
        if not job_id:
            return "No persistent render job is attached to this storyboard.", plan.get("final_video"), str(plan_path)
        row = self._job_queue.get(job_id)
        if not row:
            return "### ERROR\nPersistent render job was not found.", None, str(plan_path)
        status = str(row.get("status", "unknown"))
        if status == "completed" and row.get("result_json"):
            try:
                result = json.loads(row["result_json"])
                final_video = result.get("final_video")
            except Exception:
                final_video = plan.get("final_video")
            return f"### COMPLETE\nJob `{job_id}` completed.", final_video, str(plan_path)
        if status == "failed":
            return f"### FAILED\n{row.get('error', 'Unknown render failure.')}", None, str(plan_path)
        return f"### {status.upper()}\nJob `{job_id}` is {status}.", plan.get("final_video"), str(plan_path)

    def _queue_worker_loop(self):
        while not self._queue_stop.wait(0.5):
            job = self._job_queue.claim_next()
            if not job:
                continue
            job_id = str(job["job_id"])
            worker_token = str(job.get("worker_token", "") or "")
            heartbeat_stop = threading.Event()
            heartbeat_thread = None
            try:
                def _heartbeat():
                    while not heartbeat_stop.wait(30.0):
                        if not self._job_queue.heartbeat(job_id, worker_token):
                            break
                heartbeat_thread = threading.Thread(target=_heartbeat, name=f"h3-job-heartbeat-{job_id[:8]}", daemon=True)
                heartbeat_thread.start()

                status_message, final_video, plan_path_value = self._execute_approved_plan(str(job["plan_path"]))
                if not str(status_message).startswith("### VIDEO GENERATION COMPLETE"):
                    raise RuntimeError(str(status_message))
                plan, _ = self._load_plan(str(job["plan_path"]))
                result = {
                    "production_id": plan.get("production_id", job.get("production_id")),
                    "final_video": final_video or plan.get("final_video"),
                }
                self._job_queue.complete(job_id, result, worker_token=worker_token)
                heartbeat_stop.set()
                plan["job_status"] = "completed"
                Path(job["plan_path"]).write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                try:
                    self._job_queue.fail(job_id, str(exc), worker_token=worker_token)
                except Exception:
                    pass
                try:
                    plan, plan_path = self._load_plan(str(job["plan_path"]))
                    plan["job_status"] = "failed"
                    plan["job_error"] = str(exc)
                    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
                heartbeat_stop.set()

    def _execute_approved_plan(
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

        # Queue workers must wait for an active storyboard-planning operation
        # instead of converting normal lock contention into a permanent job
        # failure. Keep the timeout bounded so a genuinely wedged controller
        # cannot block the persistent queue forever.
        lock_acquired = self._lock.acquire(
            timeout=3600.0
        )

        if not lock_acquired:

            return (
                "### ERROR\nLock acquisition timed out after 1 hour.",
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

            if lock_acquired:
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

    css = """
    .h3-shell { max-width: 1500px; margin: 0 auto; }
    .h3-hero { padding: 20px 24px; border-radius: 18px; border: 1px solid var(--border-color-primary); background: linear-gradient(135deg, var(--background-fill-secondary), var(--background-fill-primary)); }
    .h3-title { font-size: 28px; font-weight: 700; margin: 0 0 6px 0; }
    .h3-subtitle { opacity: .78; margin: 0; }
    .h3-panel { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 10px; }
    .h3-actions button { min-height: 46px; }
    .h3-status { min-height: 42px; }
    .h3-preview img { object-fit: contain !important; }
    .h3-shot-card { border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 16px; background: var(--background-fill-secondary); }
    .h3-shot-title { font-size: 1.15rem; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    .h3-shot-meta { margin: 10px 0; display: flex; gap: 6px; flex-wrap: wrap; }
    .h3-shot-grid { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 14px; }
    .h3-shot-grid section { min-width: 0; }
    .h3-shot-grid pre { white-space: pre-wrap; max-height: 260px; overflow: auto; padding: 10px; border-radius: 10px; }
    .h3-badge { display: inline-block; border: 1px solid var(--border-color-primary); border-radius: 999px; padding: 3px 8px; margin: 2px; font-size: 0.78rem; }
    .h3-shot-inspector-hint { padding: 10px 12px; border: 1px solid var(--border-color-primary); border-radius: 10px; margin: 8px 0; opacity: 0.82; }
    .h3-good { border-color: #3b8f5f; }
    .h3-warn { border-color: #c48a2b; }
    .h3-bad { border-color: #bd4a4a; }
    @media (max-width: 1000px) { .h3-shot-grid { grid-template-columns: 1fr; } }
    """

    try:
        theme = gr.themes.Soft(spacing_size="sm", radius_size="md")
    except Exception:
        theme = None

    with gr.Blocks(
        title="MiniMax H3 Film Studio",
        theme=theme,
        css=css,
    ) as demo:

        with gr.Column(elem_classes=["h3-shell"]):
            gr.HTML(
                "<div class='h3-hero'><div class='h3-title'>MiniMax H3 Film Studio</div>"
                "<div class='h3-subtitle'>Story → entities → Qwen Director → cinematic compiler → H3 → visual QA → retake → upscale → final film</div></div>"
            )

            with gr.Tabs():
                with gr.Tab("🎬 Studio", id="studio"):
                    story = gr.Textbox(
                        label="Your Story",
                        value=initial_story or "",
                        placeholder="Describe the story, characters, world and desired outcome…",
                        lines=10,
                    )
                    with gr.Row():
                        mode = gr.Radio(
                            choices=[("AI Story", "ai_story"), ("Expand Story", "expand_user_story"), ("Preserve Story", "preserve_user_story")],
                            value=initial_mode if initial_mode in {"ai_story", "expand_user_story", "preserve_user_story"} else "ai_story",
                            label="Story Mode",
                            info="AI Story creates from the premise. Expand enriches it. Preserve keeps the supplied story and structures it for production.",
                        )
                        resume_id = gr.Textbox(
                            label="Resume Production ID (optional)",
                            value="",
                            placeholder="production_YYYYMMDD_…",
                            lines=1,
                        )
                    with gr.Row(elem_classes=["h3-actions"]):
                        generate = gr.Button("Generate Storyboard", variant="primary")
                        regenerate = gr.Button("Regenerate")
                        refresh = gr.Button("Refresh Saved Drafts")
                    saved_draft = gr.Dropdown(
                        choices=controller._draft_choices(),
                        label="Saved Storyboard",
                        value=None,
                        interactive=True,
                        allow_custom_value=False,
                    )
                    with gr.Row(elem_classes=["h3-actions"]):
                        preview = gr.Button("Preview Selected")
                        latest = gr.Button("Preview Latest For Mode")
                    status = gr.Markdown("Write your story and generate a storyboard.", elem_classes=["h3-status"])
                    with gr.Row():
                        with gr.Column(elem_classes=["h3-panel"]):
                            with gr.Accordion("Characters", open=True):
                                characters = gr.Markdown()
                        with gr.Column(elem_classes=["h3-panel"]):
                            with gr.Accordion("Scenes", open=True):
                                scenes = gr.Markdown()
                    with gr.Accordion("Shots & Director Plan", open=True):
                        shots = gr.Markdown()
                    with gr.Row():
                        shot_selector = gr.Dropdown(label="Selected Shot", choices=[], value=None, interactive=True)
                        shot_refresh = gr.Button("Load Shot Details")
                    shot_detail = gr.Markdown("### Select a shot\nThe selected shot's prompt, references, continuity, critic, VLM and QA state will appear here.", elem_classes=["h3-panel"])
                    shot_inspector_hint = gr.Markdown("Select a shot to inspect its visual result, quality decision, and retake controls.", elem_classes=["h3-shot-inspector-hint"])
                    with gr.Row():
                        with gr.Column(scale=2):
                            shot_visual = gr.Image(label="Selected Shot Preview", type="filepath", height=360, elem_classes=["h3-preview"])
                            shot_visual_status = gr.Markdown()
                        with gr.Column(scale=1, elem_classes=["h3-panel"]):
                            gr.Markdown("### Shot Actions")
                            shot_load_retake = gr.Button("Use Selected Shot for Retake", variant="primary")
                            gr.Markdown("Select a shot above; the retake controls are populated automatically from its duration and current QA reason.")

                with gr.Tab("⏱ Timeline & Preview", id="timeline"):
                    gr.Markdown("### Timeline Editor\nEdit duration and continuity mode, then apply the changes. The cinematic compiler remains the single production source of truth.")
                    timeline_table = gr.Dataframe(
                        headers=["Shot", "Scene", "Start (s)", "End (s)", "Duration (s)", "Continuity"],
                        datatype=["str", "str", "number", "number", "number", "str"],
                        value=[], interactive=True, row_count=(1, "dynamic"), col_count=(6, "fixed"),
                    )
                    with gr.Row(elem_classes=["h3-actions"]):
                        load_timeline = gr.Button("Load Timeline")
                        apply_timeline = gr.Button("Apply Timeline Edits", variant="primary")
                    timeline_status = gr.Markdown()
                    gr.Markdown("### Live H3 Sampling Preview\nThe preview is optional and does not replace the ComfyUI rendering backend. Refresh is safe during long renders.")
                    live_preview = gr.Image(label="Current Sampling Preview", type="filepath", height=480, elem_classes=["h3-preview"])
                    live_preview_status = gr.Markdown()
                    with gr.Row(elem_classes=["h3-actions"]):
                        refresh_live_preview = gr.Button("Refresh Live Preview", variant="primary")
                    live_preview_timer = gr.Timer(value=3.0, active=False)
                    gr.Markdown("### Final Output")
                    final_video = gr.Video(label="Final Film / Latest Output")

                with gr.Tab("🛠 Production", id="production"):
                    with gr.Row():
                        runtime_status_card = gr.Markdown("**Runtime:** press Check Runtime Health", elem_classes=["h3-panel"])
                        vlm_status_card = gr.Markdown(
                            "**VLM:** configured automatically from `H3_VLM_ENABLED`, `H3_VLM_ENDPOINT`, and `H3_VLM_MODEL`.",
                            elem_classes=["h3-panel"],
                        )
                    with gr.Accordion("Runtime Diagnostics", open=True):
                        runtime_json = gr.JSON()
                        runtime_status = gr.Markdown()
                        runtime_check = gr.Button("Check Runtime Health", variant="primary")
                    with gr.Accordion("Selective Retake", open=True):
                        gr.Markdown("Mark only the bad range. The retake manager persists the request so the renderer can replace that range and reassemble the shot.")
                        retake_shot_id = gr.Textbox(label="Shot ID")
                        with gr.Row():
                            retake_start = gr.Number(label="Start (s)", value=0.0, minimum=0.0)
                            retake_end = gr.Number(label="End (s)", value=4.0, minimum=0.01)
                        retake_reason = gr.Textbox(label="Why retake this range?", lines=3)
                        request_retake = gr.Button("Create Retake Request", variant="primary")
                        retake_status = gr.Markdown()
                    with gr.Row(elem_classes=["h3-actions"]):
                        approve = gr.Button("Approve & Generate Video", variant="primary", scale=2)
                        refresh_job = gr.Button("Refresh Generation Status")
                    result_status = gr.Markdown(elem_classes=["h3-status"])

            session_plan_path = gr.Textbox(value="", visible=False, interactive=False)

            generation_outputs = [status, characters, scenes, shots, result_status, final_video, session_plan_path, saved_draft]

            generate.click(fn=generate_storyboard_ui, inputs=[story, mode, resume_id], outputs=generation_outputs)
            regenerate.click(fn=regenerate_storyboard_ui, inputs=[story, mode, resume_id], outputs=generation_outputs)
            preview.click(fn=controller.preview_saved, inputs=[saved_draft], outputs=[status, characters, scenes, shots, result_status, final_video, session_plan_path])
            latest.click(fn=preview_latest_ui, inputs=[mode], outputs=[saved_draft, status, characters, scenes, shots, result_status, final_video, session_plan_path])
            refresh.click(fn=refresh_saved_drafts_ui, inputs=[saved_draft], outputs=[saved_draft])

            approve.click(fn=controller.approve_and_generate, inputs=[session_plan_path], outputs=[result_status, final_video, session_plan_path])
            refresh_job.click(fn=controller.refresh_job_status, inputs=[session_plan_path], outputs=[result_status, final_video, session_plan_path])
            refresh_live_preview.click(fn=controller.latest_live_preview, inputs=[session_plan_path], outputs=[live_preview, live_preview_status])
            live_preview_timer.tick(fn=controller.latest_live_preview, inputs=[session_plan_path], outputs=[live_preview, live_preview_status])
            load_timeline.click(fn=controller.timeline_table, inputs=[session_plan_path], outputs=[timeline_table, timeline_status])
            load_timeline.click(fn=controller.shot_options, inputs=[session_plan_path], outputs=[shot_selector, shot_detail])
            shot_selector.change(fn=controller.shot_detail, inputs=[session_plan_path, shot_selector], outputs=[shot_detail])
            shot_refresh.click(fn=controller.shot_detail, inputs=[session_plan_path, shot_selector], outputs=[shot_detail])
            shot_selector.change(fn=controller.shot_preview, inputs=[session_plan_path, shot_selector], outputs=[shot_visual, shot_visual_status])
            shot_selector.change(fn=controller.shot_retake_defaults, inputs=[session_plan_path, shot_selector], outputs=[retake_shot_id, retake_start, retake_end, retake_reason])
            shot_load_retake.click(fn=controller.shot_retake_defaults, inputs=[session_plan_path, shot_selector], outputs=[retake_shot_id, retake_start, retake_end, retake_reason])
            apply_timeline.click(fn=controller.apply_timeline_edits, inputs=[session_plan_path, timeline_table], outputs=[timeline_table, timeline_status, session_plan_path])
            runtime_check.click(fn=controller.runtime_health, inputs=[session_plan_path], outputs=[runtime_json, runtime_status])
            request_retake.click(fn=controller.create_retake_request, inputs=[session_plan_path, retake_shot_id, retake_start, retake_end, retake_reason], outputs=[retake_status, session_plan_path])

            def runtime_card_ui():
                try:
                    report = RuntimeDiagnostics(ROOT).collect()
                    gpus = report.get("gpus", []) or []
                    gpu_text = ", ".join(map(str, gpus)) if gpus else "unavailable"
                    return f"**Runtime:** 🟢 {report.get('python', '')} · GPUs: {gpu_text}", (
                        "**VLM:** 🟢 enabled/configured" if os.getenv("H3_VLM_ENDPOINT", "").strip() and os.getenv("H3_VLM_MODEL", "").strip() else
                        "**VLM:** 🟡 enabled but endpoint/model not configured"
                    )
                except Exception as exc:
                    return "**Runtime:** 🔴 diagnostics failed", f"**VLM:** status unavailable — {exc}"

            runtime_check.click(fn=runtime_card_ui, inputs=[], outputs=[runtime_status_card, vlm_status_card])

    return demo


def serve_storyboard_gradio(
    plan_path: Path | None = None,
    wait_for_approval: bool = False,
    initial_story: str | None = None,
    initial_mode: str = "ai_story",
):

    del plan_path
    del wait_for_approval

    os.environ.setdefault("H3_DIRECTOR_ENABLED", "1")

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

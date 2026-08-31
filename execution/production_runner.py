from __future__ import annotations

import errno
import hashlib
import os
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from execution.assembly_manager import (
    AssemblyManager,
)
from execution.h3_runtime import H3Runtime
from execution.shot_executor import (
    ShotExecutor,
)
from pipeline.storyboard_reference_builder import StoryboardReferenceBuilder
from pipeline.dialogue_duration import FFProbeMediaDurationProvider
from pipeline.seed_lineage import ensure_shot_uid, semantic_content_digest, stable_seed
from pipeline.visual_state_observer import VisualStateObserver
from pipeline.visual_feedback import VisualFeedbackEngine
from pipeline.h3_scene_continuity import (
    H3SceneContinuity,
)
from pipeline.continuity_ledger import ContinuityLedger
from pipeline.production_checkpoint import (
    ProductionCheckpoint,
)
from pipeline.identity_anchor_store import (
    IdentityAnchorStore,
)
from planner.config import (
    DELIVERY_FPS,
    DELIVERY_HEIGHT,
    DELIVERY_WIDTH,
    H3_MAX_REFERENCE_IMAGES,
    PROFILE_BASE,
    PROFILE_TURBO,
    PROFILE_UPSCALE,
    WORKFLOW_REF2V,
    WORKFLOW_TURBO_REF2V,
)
from scheduler.gpu_scheduler import (
    GPUScheduler,
)


class ProductionRunner:

    def __init__(
        self,
        project_root: Path,
        comfy_clients: dict[int, object],
    ):

        self.project_root = Path(
            project_root
        ).resolve()

        self.clients = dict(
            comfy_clients
        )

        if not self.clients:
            raise ValueError(
                "At least one ComfyUI client is required."
            )

        self.input_root = (
            self.project_root
            / "ComfyUI"
            / "input"
        )

        self.production_id = None
        self._active_plan_sha256 = ""
        self._completed_shots_lock = threading.RLock()
        self._manifest_lock = threading.RLock()
        self.visual_observer = VisualStateObserver(self.project_root)
        self.visual_feedback = VisualFeedbackEngine(self.visual_observer)

        self.production_input_root = (
            self.input_root
        )

        self.output_root = (
            self.project_root
            / "data"
            / "production"
            / "h3"
        )


    @staticmethod
    def _safe_name(
        value,
    ) -> str:

        text = str(
            value or ""
        ).strip()

        cleaned = "".join(
            char
            if (
                char.isalnum()
                or char in "._-"
            )
            else "_"
            for char in text
        )

        return (
            cleaned[:96]
            or "production"
        )

    def _resolve_production_id(
        self,
        production_plan: dict,
    ) -> str:

        requested = str(
            production_plan.get(
                "production_id",
                "",
            )
            or ""
        ).strip()

        if requested:
            return self._safe_name(
                requested
            )

        return (
            "production_"
            f"{uuid.uuid4().hex}"
        )

    @staticmethod
    def _plan_hash(production_plan: dict[str, Any]) -> str:
        return ProductionCheckpoint.plan_digest(production_plan)

    @staticmethod
    def _validate_checkpoint_plan(
        production_plan: dict[str, Any],
        checkpoint: dict | None,
    ) -> str:
        plan_hash = ProductionCheckpoint.plan_digest(production_plan)
        if not checkpoint:
            return plan_hash

        existing_hash = str(checkpoint.get("plan_sha256", "") or "").strip()
        status = str(checkpoint.get("status", "") or "").strip().lower()
        resume_statuses = {"rendering", "interrupted", "failed", "ready", "running"}

        if existing_hash:
            if existing_hash != plan_hash:
                raise RuntimeError(
                    "Production checkpoint plan fingerprint does not match the supplied plan. "
                    "Refusing to reuse completed shots from a different plan. "
                    f"checkpoint={existing_hash} supplied={plan_hash}"
                )
            return plan_hash

        if status in resume_statuses:
            raise RuntimeError(
                "This production checkpoint predates plan fingerprinting. "
                "Refusing unsafe resume; start a new production or migrate the checkpoint explicitly."
            )
        return plan_hash

    def _prepare_production_paths(
        self,
        production_id: str,
    ) -> None:

        self.production_id = (
            production_id
        )

        self.production_input_root = (
            self.input_root
            / production_id
        )

        self.output_root = (
            self.project_root
            / "data"
            / "production"
            / production_id
            / "h3"
        )

        self.production_input_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Production-scoped continuity state.
        #
        # The production ID is only known when run() starts,
        # so these stores must be created here rather than left
        # as the global instances created during __init__.
        self.continuity = (
            H3SceneContinuity(
                self.project_root,
                production_id=production_id,
            )
        )

        self.identity_anchors = (
            IdentityAnchorStore(
                self.project_root,
                production_id=production_id,
            )
        )

    def _executor(
        self,
        gpu_id: int,
        scene_id: str,
    ):

        from execution.metrics import MetricsRecorder
        metrics_path = self.project_root / "data" / "production" / str(self.production_id) / "metrics.jsonl"
        return ShotExecutor(
            comfy_client=self.clients[gpu_id],
            project_root=self.project_root,
            comfy_input_dir=(
                self.production_input_root
                / f"gpu_{gpu_id}"
                / self._safe_name(scene_id)
            ),
            gpu_id=gpu_id,
            metrics_path=metrics_path,
        )

    @staticmethod
    def _scene_batches(scene_jobs: list[tuple[str, list[dict]]]) -> list[list[tuple[str, list[dict]]]]:
        """Create ordered batches where no two concurrent scenes share characters.

        Scenes sharing a character are forced into later batches so identity-anchor
        writes remain deterministic. Unrelated scenes can occupy the same batch and
        therefore use both T4 GPUs even when the overall production is not globally
        marked parallel-safe.
        """
        batches: list[list[tuple[str, list[dict]]]] = []
        scene_characters: dict[str, set[str]] = {}
        scene_batch: dict[str, int] = {}

        for scene_id, scene_shots in scene_jobs:
            chars = {
                str(name).strip().lower()
                for shot in scene_shots
                for name in (shot.get("characters", []) or [])
                if str(name).strip()
            }
            scene_characters[scene_id] = chars
            earliest_batch = 0
            for previous_id, previous_chars in scene_characters.items():
                if previous_id == scene_id:
                    continue
                if chars and previous_chars and chars.intersection(previous_chars):
                    earliest_batch = max(earliest_batch, scene_batch[previous_id] + 1)
            while len(batches) <= earliest_batch:
                batches.append([])
            batches[earliest_batch].append((scene_id, scene_shots))
            scene_batch[scene_id] = earliest_batch

        return batches

    @staticmethod
    def _workflow_for_shot(
        shot: dict,
        profile: str,
    ) -> str:

        explicit = str(
            shot.get(
                "workflow_mode",
                "",
            )
            or ""
        ).strip()

        if explicit in {
            WORKFLOW_REF2V,
            WORKFLOW_TURBO_REF2V,
        }:
            return explicit

        if profile == PROFILE_TURBO:
            return WORKFLOW_TURBO_REF2V

        return WORKFLOW_REF2V

    @staticmethod
    def _sort_shots(
        shots,
    ):

        return sorted(
            shots,
            key=lambda shot: int(
                shot.get(
                    "order",
                    0,
                )
            ),
        )

    def _character_map(
        self,
        production_plan: dict,
    ) -> dict:

        return {
            str(
                character.get(
                    "name",
                    "",
                )
            ).strip().lower():
                character
            for character
            in production_plan.get(
                "characters",
                [],
            )
            if str(
                character.get(
                    "name",
                    "",
                )
            ).strip()
        }

    def _add_identity_anchors(
        self,
        shot: dict,
        character_map: dict,
    ) -> None:

        references = list(
            shot.get(
                "reference_images",
                [],
            )
            or []
        )
        roles = [
            dict(item)
            for item in (shot.get("reference_roles", []) or [])
            if isinstance(item, dict)
        ]
        role_by_path = {
            str(item.get("path", "")): item
            for item in roles
            if str(item.get("path", "")).strip()
        }

        for name in (
            shot.get(
                "characters",
                [],
            )
            or []
        ):

            character = character_map.get(
                str(name).lower()
            )

            if not character:
                continue

            if (
                character.get(
                    "reference_mode"
                )
                == "provided"
            ):
                continue

            character_id = (
                character.get(
                    "character_id"
                )
            )

            if not character_id:
                continue

            anchor = (
                self.identity_anchors
                .latest_anchor(
                    character_id
                )
            )

            if (
                anchor
                and str(anchor)
                not in references
            ):
                anchor_path = str(anchor)
                references.insert(0, anchor_path)
                role_by_path[anchor_path] = {
                    "path": anchor_path,
                    "role": "character_identity",
                    "character_name": character.get("name", name),
                    "character_id": character_id,
                    "label": (
                        f"Production identity anchor for {character.get('name', name)}; "
                        "use for stable identity only."
                    ),
                    "priority": 95,
                }

        normalized_roles = []
        for path in references[:H3_MAX_REFERENCE_IMAGES]:
            role = dict(
                role_by_path.get(
                    str(path),
                    {
                        "path": str(path),
                        "role": "visual_reference",
                        "priority": 50,
                    },
                )
            )
            role["path"] = str(path)
            normalized_roles.append(role)

        shot["reference_images"] = [item["path"] for item in normalized_roles]
        shot["reference_roles"] = normalized_roles

    def _persist_first_appearance_anchors(
        self,
        shot: dict,
        character_map: dict,
        frame_path: Path,
    ) -> None:

        for name in (
            shot.get(
                "characters",
                [],
            )
            or []
        ):

            character = character_map.get(
                str(name).lower()
            )

            if not character:
                continue

            if (
                character.get(
                    "reference_mode"
                )
                == "provided"
            ):
                continue

            character_id = (
                character.get(
                    "character_id"
                )
            )

            if not character_id:
                continue

            self.identity_anchors.save_first_anchor(
                character_id=character_id,
                shot_id=str(
                    shot[
                        "shot_id"
                    ]
                ),
                source_frame=frame_path,
            )

    # ========================================================
    # RENDER CHECKPOINT / RESUME
    # ========================================================

    def _checkpoint_store(self) -> ProductionCheckpoint:
        return ProductionCheckpoint(
            self.project_root
        )

    def _load_render_checkpoint(
        self,
        production_id: str,
    ) -> dict | None:
        store = self._checkpoint_store()
        try:
            return store.load(production_id)
        except FileNotFoundError:
            return None
        except (RuntimeError, ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Production checkpoint is unreadable or invalid: {production_id}"
            ) from exc

    def _update_render_checkpoint(
        self,
        production_id: str,
        *,
        status: str,
        stage: str,
        current_scene_id: str = "",
        completed_shot_ids: list[str] | None = None,
        completed_shot: dict[str, dict] | None = None,
        error: str = "",
        final_video: str = "",
        shot_outputs: list[str] | None = None,
    ) -> None:
        store = self._checkpoint_store()

        updates = {
            "status": status,
            "stage": stage,
            "current_scene_id": current_scene_id,
            "completed_shot_ids": list(completed_shot_ids or []),
            "error": str(error or ""),
        }

        if self._active_plan_sha256:
            updates["plan_sha256"] = self._active_plan_sha256

        if completed_shot is not None:
            updates[
                "completed_shots"
            ] = dict(
                completed_shot
            )

        if final_video:
            updates[
                "final_video"
            ] = str(
                final_video
            )

        if shot_outputs is not None:
            updates[
                "shot_outputs"
            ] = [
                str(path)
                for path in shot_outputs
            ]

        store.update(
            production_id,
            updates,
        )

    def _validate_completed_record(
        self,
        shot_id: str,
        record: dict,
        production_id: str,
    ) -> Path | None:
        record_hash = str(record.get("plan_sha256", "") or "").strip()
        if not record_hash:
            raise RuntimeError(
                f"Checkpoint record for {shot_id} has no plan fingerprint; refusing unsafe reuse."
            )
        if record_hash != self._active_plan_sha256:
            raise RuntimeError(
                f"Checkpoint record for {shot_id} belongs to a different plan."
            )
        recorded_output = str(record.get("output", "") or "").strip()
        if not recorded_output:
            return None
        path = self._existing_shot_output(Path(recorded_output))
        if path is None:
            return None
        production_root = (
            self.project_root / "data" / "production" / self._safe_name(production_id)
        ).resolve()
        try:
            path.relative_to(production_root)
        except ValueError as exc:
            raise RuntimeError(
                f"Checkpoint output for {shot_id} is outside the production directory: {path}"
            ) from exc
        return path

    @staticmethod
    def _existing_shot_output(
        output: Path,
        shot_id: str | None = None,
    ) -> Path | None:
        path = (
            Path(output)
            if shot_id is None
            else Path(output) / f"{shot_id}.mp4"
        )

        if (
            path.is_file()
            and path.stat().st_size > 0
        ):
            return path.resolve()

        return None

    @staticmethod
    def _reference_binding_text(
        reference_roles: list[dict],
    ) -> list[str]:
        bindings = []
        for index, role in enumerate(reference_roles, start=1):
            kind = str(role.get("role", "")).strip().lower()
            label = str(role.get("label", "")).strip()
            if not label:
                if kind == "storyboard":
                    label = "Unified storyboard for sequencing, composition and blocking."
                elif kind == "previous_shot_last_frame":
                    label = "Previous shot final-frame continuity reference."
                else:
                    character = str(role.get("character_name", "")).strip()
                    label = f"Canonical visual identity reference for {character}." if character else "Production visual reference."
            bindings.append(
                f"<Picture {index}> = {label}"
            )
        return bindings

    def _rebuild_reference_contract(
        self,
        shot: dict,
        references: list[str],
        roles: list[dict],
    ) -> None:
        if len(references) != len(roles):
            raise RuntimeError(
                f"Reference contract mismatch for {shot.get('shot_id', '')}: "
                f"{len(references)} images vs {len(roles)} roles before normalization."
            )
        normalized = []
        seen = set()
        for path, role in zip(references, roles):
            p = str(path).strip()
            if not p or p in seen:
                continue
            seen.add(p)
            item = dict(role)
            item["path"] = p
            normalized.append(item)
        shot["reference_images"] = [item["path"] for item in normalized][:9]
        shot["reference_roles"] = normalized[:9]
        shot["reference_bindings"] = self._reference_binding_text(shot["reference_roles"])

        # Rebuild the stored prompt through the schema object so Picture N
        # numbering always matches the actual runtime reference order.
        from schemas.shot import Shot
        field_names = {field.name for field in __import__("dataclasses").fields(Shot)}
        payload = {key: value for key, value in shot.items() if key in field_names}
        payload["reference_images"] = shot["reference_images"]
        payload["reference_roles"] = shot["reference_roles"]
        payload["reference_bindings"] = shot["reference_bindings"]
        shot_obj = Shot(**payload)
        shot["h3_prompt"] = shot_obj.h3_prompt()

        manifest_path = shot.get("reference_role_manifest")
        if manifest_path:
            with self._manifest_lock:
                entry = StoryboardReferenceBuilder.update_manifest(
                    manifest_path,
                    str(shot.get("shot_id", "")),
                    shot["reference_images"],
                    shot["reference_roles"],
                    shot["reference_bindings"],
                )
                StoryboardReferenceBuilder.assert_manifest_invariant(
                    entry,
                    shot["reference_images"],
                    shot["reference_bindings"],
                )

    def _run_scene(
        self,
        gpu_id,
        scene_id,
        shots,
        profile,
        character_map,
        upscale_enabled,
        production_id: str,
        completed_shots: dict[str, dict],
    ):

        executor = self._executor(
            gpu_id,
            scene_id,
        )

        output_dir = (
            self.output_root
            / f"gpu_{gpu_id}"
            / self._safe_name(scene_id)
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        results = []
        previous_video = None
        previous_shot = None

        for original in self._sort_shots(
            shots
        ):

            shot = dict(
                original
            )

            shot_id = str(
                shot.get(
                    "shot_id",
                    "",
                )
                or ""
            ).strip()

            if not shot_id:
                raise RuntimeError(
                    "Shot is missing shot_id."
                )

            # Preserve any explicit user/planner seed. Only synthesize a
            # deterministic seed when the plan did not provide one.
            ensure_shot_uid(shot, production_id)
            shot["semantic_content_digest"] = semantic_content_digest(shot)
            if shot.get("seed") in (None, ""):
                shot["seed"] = stable_seed(production_id, shot)

            # ------------------------------------------------
            # RESUME BY RECORDED OUTPUT PATH + GPU.
            # Never assume the original output was rendered
            # on the GPU currently executing the resume.
            # ------------------------------------------------
            with self._completed_shots_lock:
                record = completed_shots.get(
                    shot_id
                )

            existing = None

            if isinstance(
                record,
                dict,
            ):

                existing = self._validate_completed_record(
                    shot_id,
                    record,
                    production_id,
                )

                # Backward compatibility for older checkpoints that
                # have GPU ownership but no absolute output path.
                if existing is None:
                    recorded_gpu = record.get(
                        "gpu_id",
                        gpu_id,
                    )

                    legacy_output_dir = (
                        self.output_root
                        / f"gpu_{recorded_gpu}"
                        / self._safe_name(
                            scene_id
                        )
                    )

                    existing = (
                        self._existing_shot_output(
                            legacy_output_dir,
                            shot_id,
                        )
                    )

                if existing is not None:

                    result = Path(
                        existing
                    )

                    results.append(
                        result
                    )

                    previous_video = result
                    previous_shot = shot

                    print(
                        "[H3 RESUME] skipping completed shot",
                        shot_id,
                        "->",
                        result,
                        flush=True,
                    )

                    continue

            self._update_render_checkpoint(
                production_id,
                status="rendering",
                stage="rendering",
                current_scene_id=scene_id,
            )

            self._add_identity_anchors(
                shot,
                character_map,
            )

            if previous_shot is not None and not bool(shot.get("is_scene_boundary", False)):
                observed = previous_shot.get("observed_visual_state")
                if isinstance(observed, dict) and observed:
                    shot["observed_previous_shot_state"] = dict(observed)

            if previous_video is not None and not bool(shot.get("is_scene_boundary", False)):

                last_frame = (
                    self.continuity
                    .prepare_next_shot(
                        previous_video,
                        scene_id,
                        previous_shot[
                            "shot_id"
                        ],
                    )
                )

                references = list(
                    shot.get(
                        "reference_images",
                        [],
                    )
                    or []
                )
                roles = [
                    dict(item)
                    for item in (
                        shot.get("reference_roles", []) or []
                    )
                    if isinstance(item, dict)
                ]
                role_by_path = {
                    str(item.get("path", "")): item
                    for item in roles
                    if str(item.get("path", "")).strip()
                }

                ordered: list[tuple[str, dict]] = []
                for path in references:
                    role = dict(
                        role_by_path.get(
                            str(path),
                            {
                                "path": str(path),
                                "role": "visual_reference",
                                "priority": 50,
                            },
                        )
                    )
                    ordered.append((str(path), role))

                ordered = [
                    item
                    for item in ordered
                    if item[0] != str(last_frame)
                ]
                storyboard = [
                    item
                    for item in ordered
                    if item[1].get("role") == "storyboard"
                ]
                identity = [
                    item
                    for item in ordered
                    if item[1].get("role") != "storyboard"
                ]
                identity.sort(
                    key=lambda item: (
                        -int(item[1].get("priority", 50)),
                        item[0],
                    )
                )
                role = {
                    "path": str(last_frame),
                    "role": "previous_shot_last_frame",
                    "label": (
                        f"Exact final frame of previous shot "
                        f"{previous_shot['shot_id']}; use for temporal and visual continuity."
                    ),
                    "priority": 100,
                }
                final_items = identity + storyboard + [(str(last_frame), role)]
                final_items = final_items[:H3_MAX_REFERENCE_IMAGES]
            else:
                # First shot of a scene or an explicit scene boundary: no
                # previous-frame relay. Identity/storyboard references remain.
                final_items = []
                for path, role in zip(
                    list(shot.get("reference_images", []) or []),
                    list(shot.get("reference_roles", []) or []),
                ):
                    if isinstance(role, dict):
                        item = dict(role)
                        item["path"] = str(path)
                        final_items.append((str(path), item))
                final_items = final_items[:H3_MAX_REFERENCE_IMAGES]

            # Exactly one contract rebuild per shot. This incorporates identity
            # anchors, storyboard references, and optional last-frame continuity
            # before the manifest is persisted and <Picture N> is rebuilt.
            self._rebuild_reference_contract(
                shot,
                [path for path, _ in final_items],
                [role for _, role in final_items],
            )

            workflow_mode = (
                self._workflow_for_shot(
                    shot,
                    profile,
                )
            )

            result = (
                executor.execute_shot(
                    shot=shot,
                    workflow_mode=workflow_mode,
                    output_dir=output_dir,
                    upscale=upscale_enabled,
                )
            )

            result = Path(
                result
            ).resolve()

            if (
                not result.is_file()
                or result.stat().st_size <= 0
            ):
                raise RuntimeError(
                    "Shot execution returned an invalid output: "
                    f"{result}"
                )

            try:
                av_result = FFProbeMediaDurationProvider().validate_video_audio_sync(
                    result,
                    tolerance_seconds=0.30,
                )
                if av_result.get("audio_stream_present"):
                    shot["audio_duration_seconds"] = av_result.get("audio_duration_seconds")
                    shot["audio_duration_source"] = "ffprobe_rendered_output"
                elif shot.get("dialogue_events"):
                    raise RuntimeError(
                        f"Rendered shot {shot_id} contains dialogue events but no audio stream was detected."
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                print("[H3 A/V VALIDATION] warning:", exc, flush=True)

            results.append(
                result
            )

            anchor_frame = (
                self.continuity
                .prepare_next_shot(
                    result,
                    scene_id,
                    shot["shot_id"],
                )
            )

            try:
                shot["visual_feedback"] = self.visual_feedback.analyze(
                    result,
                    anchor_frame,
                    shot.get("continuity_end_state", {}) or {},
                )
                shot["observed_visual_state"] = dict(
                    shot["visual_feedback"].get("observed_state", {})
                )
            except Exception as feedback_error:
                shot["visual_feedback"] = {
                    "deterministic_observation": False,
                    "warning": str(feedback_error),
                }

            self._persist_first_appearance_anchors(
                shot,
                character_map,
                anchor_frame,
            )

            previous_video = result
            previous_shot = shot

            with self._completed_shots_lock:
                completed_shots[shot_id] = {
                    "gpu_id": int(gpu_id),
                    "scene_id": str(scene_id),
                    "output": str(result),
                    "plan_sha256": self._active_plan_sha256,
                    "shot_uid": str(shot.get("shot_uid", "")),
                    "semantic_content_digest": str(shot.get("semantic_content_digest", "")),
                    "observed_visual_state": dict(shot.get("observed_visual_state", {}) or {}),
                    "visual_feedback": dict(shot.get("visual_feedback", {}) or {}),
                    "audio_duration_seconds": shot.get("audio_duration_seconds"),
                }
                completed_record = dict(completed_shots[shot_id])

            self._update_render_checkpoint(
                production_id,
                status="rendering",
                stage="rendering",
                current_scene_id=scene_id,
                completed_shot_ids=[
                    shot_id
                ],
                completed_shot={
                    shot_id: completed_record,
                },
            )

        return results

    @contextmanager
    def _production_lock(self, production_id: str):
        lock_dir = self.project_root / "data" / "production" / self._safe_name(production_id)
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / ".run.lock"
        handle = lock_path.open("a+")
        try:
            try:
                import fcntl
            except ImportError as exc:
                raise RuntimeError("Production locking requires fcntl on Linux production targets.") from exc
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise RuntimeError(f"Production {production_id} is already running in another process.") from exc
                raise
            handle.seek(0)
            handle.truncate(0)
            handle.write(f"pid={os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            handle.close()

    def run(
        self,
        production_plan: dict[str, Any],
    ):
        if not isinstance(production_plan, dict):
            raise TypeError("production_plan must be a mapping.")
        production_id = self._resolve_production_id(production_plan)
        production_plan["production_id"] = production_id
        with self._production_lock(production_id):
            return self._run_locked(production_plan)

    def _run_locked(
        self,
        production_plan: dict[str, Any],
    ):

        shots = list(
            production_plan.get(
                "shots",
                [],
            )
        )

        if not shots:
            raise RuntimeError(
                "Production plan contains no shots."
            )

        production_id = (
            self._resolve_production_id(
                production_plan
            )
        )

        production_plan[
            "production_id"
        ] = production_id

        
        self._prepare_production_paths(production_id)

        checkpoint = self._load_render_checkpoint(production_id)
        self._active_plan_sha256 = self._validate_checkpoint_plan(
            production_plan, checkpoint
        )
        production_plan["plan_sha256"] = self._active_plan_sha256

        profile = str(
            production_plan.get(
                "profile",
                PROFILE_BASE,
            )
            or PROFILE_BASE
        ).strip().lower()

        if profile not in {
            PROFILE_BASE,
            PROFILE_TURBO,
            PROFILE_UPSCALE,
        }:
            raise RuntimeError(
                f"Unsupported profile: {profile}"
            )

        upscale_enabled = bool(
            production_plan.get(
                "upscale_enabled",
                profile
                == PROFILE_UPSCALE,
            )
        )

        checkpoint_status = str(
            (checkpoint or {}).get(
                "status",
                "",
            )
            or ""
        ).strip().lower()

        if checkpoint_status == "completed":
            final_video = str(
                (checkpoint or {}).get(
                    "final_video",
                    "",
                )
                or ""
            ).strip()

            if final_video:
                final_path = Path(
                    final_video
                ).resolve()

                if (
                    final_path.is_file()
                    and final_path.stat().st_size > 0
                ):
                    raise RuntimeError(
                        "Production is already completed. "
                        "Start a new production instead of "
                        "rendering the completed session: "
                        f"{production_id}"
                    )

        resume_active = checkpoint_status in {
            "rendering",
            "interrupted",
            "failed",
        }

        completed_shots = {
            str(shot_id): dict(
                record
            )
            for shot_id, record
            in (
                (checkpoint or {}).get(
                    "completed_shots",
                    {},
                )
                or {}
            ).items()
            if isinstance(
                record,
                dict,
            )
        }

        # Backward-compatible migration from checkpoints that only have IDs.
        legacy_ids = {
            str(value).strip()
            for value in (
                (checkpoint or {}).get(
                    "completed_shot_ids",
                    [],
                )
                or []
            )
            if str(value).strip()
        }

        for shot_id in legacy_ids:
            completed_shots.setdefault(
                shot_id,
                {},
            )

        all_planned_shot_ids = {
            str(
                shot.get(
                    "shot_id",
                    "",
                )
                or ""
            ).strip()
            for shot in shots
            if str(
                shot.get(
                    "shot_id",
                    "",
                )
                or ""
            ).strip()
        }

        completed_shots = {
            shot_id: record
            for shot_id, record
            in completed_shots.items()
            if shot_id in all_planned_shot_ids
        }

        # Establish the durable render state before workers start.
        self._update_render_checkpoint(
            production_id,
            status="rendering",
            stage="rendering",
            current_scene_id=str(
                (checkpoint or {}).get(
                    "current_scene_id",
                    "",
                )
                or ""
            ),
            completed_shot_ids=list(
                completed_shots.keys()
            ),
        )

        scenes: dict[
            str,
            list[dict],
        ] = {}

        for shot in shots:

            scene_id = str(
                shot.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()

            if not scene_id:
                raise RuntimeError(
                    "Shot is missing scene_id."
                )

            scenes.setdefault(
                scene_id,
                []
            ).append(
                shot
            )

        character_map = (
            self._character_map(
                production_plan
            )
        )

        scene_jobs = list(
            scenes.items()
        )

        scene_results = []

        try:
            # Hard handoff: release any remotely resident ComfyUI models and
            # clear local CUDA allocator state before concurrent H3 workers start.
            H3Runtime.vram_handoff(self.clients, unload_models=True)

            scheduler = GPUScheduler(
                gpu_ids=sorted(self.clients)
            )

            batches = self._scene_batches(scene_jobs)
            for batch_index, batch in enumerate(batches):
                if len(batch) > 1 and len(self.clients) > 1:
                    batch_results = scheduler.run_independent(
                        batch,
                        lambda gpu_id, job: (
                            job[0],
                            self._run_scene(
                                gpu_id,
                                job[0],
                                job[1],
                                profile,
                                character_map,
                                upscale_enabled,
                                production_id,
                                completed_shots,
                            ),
                        ),
                    )
                    scene_results.extend(batch_results)
                else:
                    gpu_id = sorted(self.clients)[batch_index % len(self.clients)]
                    scene_id, scene_shots = batch[0]
                    scene_results.append(
                        (
                            scene_id,
                            self._run_scene(
                                gpu_id,
                                scene_id,
                                scene_shots,
                                profile,
                                character_map,
                                upscale_enabled,
                                production_id,
                                completed_shots,
                            ),
                        )
                    )
                self._update_render_checkpoint(
                    production_id,
                    status="rendering",
                    stage="rendering",
                    current_scene_id="batch_%s" % batch_index,
                    completed_shot_ids=list(completed_shots.keys()),
                )

            # Re-read the checkpoint after all workers finish so the durable
            # merged state, rather than one worker's local dictionary, is
            # authoritative.
            final_checkpoint = (
                self._load_render_checkpoint(
                    production_id
                )
            )

            completed_shots = (
                self._completed_shot_records(
                    final_checkpoint
                )
            )

            # Every planned shot must have a durable record and a real file.
            videos = []

            for shot in self._sort_shots(
                shots
            ):

                shot_id = str(
                    shot.get(
                        "shot_id",
                        "",
                    )
                    or ""
                ).strip()

                record = (
                    completed_shots.get(
                        shot_id
                    )
                )

                if not isinstance(
                    record,
                    dict,
                ):
                    raise RuntimeError(
                        "Missing completed render record for shot: "
                        + shot_id
                    )

                output = str(
                    record.get(
                        "output",
                        "",
                    )
                    or ""
                ).strip()

                if not output:
                    raise RuntimeError(
                        "Completed render record has no output path for shot: "
                        + shot_id
                    )

                output_path = (
                    self._existing_shot_output(
                        Path(
                            output
                        )
                    )
                )

                if output_path is None:
                    raise RuntimeError(
                        "Completed render record points to a missing/invalid "
                        f"output for shot {shot_id}: {output}"
                    )

                videos.append(
                    output_path
                )

            if len(videos) != len(shots):
                raise RuntimeError(
                    "Rendered output count does not match "
                    f"planned shot count: {len(videos)} != {len(shots)}"
                )

            assembly_dir = (
                self.project_root
                / "data"
                / "production"
                / production_id
                / "final"
            )

            assembly_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            assembler = AssemblyManager(
                assembly_dir
            )

            final_video = (
                assembler.assemble(
                    videos,
                    final_name="final.mp4",
                    width=int(
                        production_plan.get(
                            "delivery_width",
                            DELIVERY_WIDTH,
                        )
                    ),
                    height=int(
                        production_plan.get(
                            "delivery_height",
                            DELIVERY_HEIGHT,
                        )
                    ),
                    fps=int(
                        production_plan.get(
                            "delivery_fps",
                            DELIVERY_FPS,
                        )
                    ),
                )
            )

            final_video = Path(
                final_video
            ).resolve()

            if (
                not final_video.is_file()
                or final_video.stat().st_size <= 0
            ):
                raise RuntimeError(
                    "Final video assembly produced no valid file."
                )

            self._update_render_checkpoint(
                production_id,
                status="completed",
                stage="render_complete",
                current_scene_id="",
                completed_shot_ids=sorted(
                    all_planned_shot_ids
                ),
                final_video=str(
                    final_video
                ),
                shot_outputs=[
                    str(path)
                    for path in videos
                ],
            )

            return {
                "production_id": production_id,
                "shot_outputs": videos,
                "final_video": final_video,
                "profile": profile,
                "upscale_enabled": upscale_enabled,
            }

        except Exception as exc:

            try:
                checkpoint_now = (
                    self._load_render_checkpoint(
                        production_id
                    )
                )

                persisted = (
                    self._completed_shot_records(
                        checkpoint_now
                    )
                )

                self._update_render_checkpoint(
                    production_id,
                    status="failed",
                    stage="rendering",
                    current_scene_id=str(
                        (checkpoint_now or {}).get(
                            "current_scene_id",
                            "",
                        )
                        or ""
                    ),
                    completed_shot_ids=list(
                        persisted.keys()
                    ),
                    error=str(
                        exc
                    ),
                )

            except Exception as checkpoint_error:

                print(
                    "[CHECKPOINT] render failure update failed:",
                    str(
                        checkpoint_error
                    ),
                    flush=True,
                )

            raise

    def _completed_shot_records(
        self,
        checkpoint: dict | None,
    ) -> dict[str, dict]:
        records = (
            (checkpoint or {}).get(
                "completed_shots",
                {},
            )
            or {}
        )

        if isinstance(
            records,
            dict,
        ):
            return {
                str(shot_id): dict(
                    record
                )
                for shot_id, record
                in records.items()
                if isinstance(
                    record,
                    dict,
                )
            }

        # Legacy checkpoints can still provide completed_shot_ids, but without
        # output paths those records are not enough for safe cross-GPU resume.
        return {}


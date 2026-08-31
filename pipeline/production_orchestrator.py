from __future__ import annotations

import json
import threading
from datetime import datetime
from copy import deepcopy
from pathlib import Path
import uuid

from pipeline.identity_continuity import (
    IdentityContinuity,
)
from pipeline.production_checkpoint import (
    ProductionCheckpoint,
)
from pipeline.reference_manager import (
    ReferenceManager,
)
from planner.config import (
    DELIVERY_FPS,
    DELIVERY_HEIGHT,
    DELIVERY_WIDTH,
    H3_FPS,
    H3_FRAMES_PER_SHOT,
    H3_HEIGHT,
    H3_MAX_REFERENCE_AUDIO,
    H3_MAX_REFERENCE_FILES,
    H3_MAX_REFERENCE_IMAGES,
    H3_MAX_REFERENCE_VIDEOS,
    H3_STEPS,
    H3_WIDTH,
    PROFILE_TURBO,
    TURBO_STEPS,
    WORKFLOW_AUTO,
    WORKFLOW_REF2V,
    WORKFLOW_TURBO_REF2V,
    ensure_directories,
    PRESERVE_USER_STORY_MODE,
    UPSCALE_HEIGHT,
    UPSCALE_WIDTH,
)
from planner.entity_resolver import EntityResolver

from planner.production_planner import (
    ProductionPlanner,
)
from planner.qwen_director import (
    QwenDirector,
)
from pipeline.dialogue_timeline import DialogueTimeline
from pipeline.continuity_ledger import ContinuityLedger, ContinuityViolation
from pipeline.storyboard_reference_builder import StoryboardReferenceBuilder
from pipeline.seed_lineage import ensure_plan_lineage
from schemas.character import (
    Character,
)
from schemas.shot import (
    Shot,
)


class ProductionOrchestrator:

    def __init__(
        self,
    ):

        ensure_directories()

        self.project_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

        self.planner = ProductionPlanner(
            self.project_root
        )

        self.references = (
            ReferenceManager(
                self.project_root
            )
        )

        self.director = QwenDirector(
            self.project_root
        )

        # Protect shared reference-role manifest updates performed by this
        # orchestrator instance. ProductionRunner has its own lock for render
        # time updates; this lock covers planning-time updates.
        self._manifest_lock = threading.RLock()

    # ========================================================
    # CHARACTER REBINDING
    # ========================================================

    def _character_objects(
        self,
        values: list[dict],
    ) -> list[Character]:

        characters = []

        for value in values:

            character = Character(
                character_id=str(
                    value.get(
                        "character_id",
                        "",
                    )
                ),
                name=str(
                    value.get(
                        "name",
                        "",
                    )
                ),
                role=str(
                    value.get(
                        "role",
                        "story character",
                    )
                ),
                description=str(
                    value.get(
                        "description",
                        "",
                    )
                ),
                personality=str(
                    value.get(
                        "personality",
                        "",
                    )
                ),
                appearance=dict(
                    value.get(
                        "appearance",
                        {},
                    )
                    or {}
                ),
                clothing=dict(
                    value.get(
                        "clothing",
                        {},
                    )
                    or {}
                ),
                distinctive_features=list(
                    value.get(
                        "distinctive_features",
                        [],
                    )
                    or []
                ),
                character_state=dict(
                    value.get(
                        "character_state",
                        {},
                    )
                    or {}
                ),
                continuity_rules=list(
                    value.get(
                        "continuity_rules",
                        [],
                    )
                    or []
                ),
            )

            source = (
                self.references.resolve_character(
                    character.name
                )
            )

            character.reference_paths = (
                source[
                    "reference_paths"
                ]
            )

            character.reference_video_paths = (
                source[
                    "reference_video_paths"
                ]
            )

            character.reference_audio_paths = (
                source[
                    "reference_audio_paths"
                ]
            )

            character.reference_path = (
                character.reference_paths[0]
                if character.reference_paths
                else None
            )

            character.reference_video_path = (
                character.reference_video_paths[0]
                if character.reference_video_paths
                else None
            )

            character.reference_audio_path = (
                character.reference_audio_paths[0]
                if character.reference_audio_paths
                else None
            )

            character.reference_mode = (
                "provided"
                if (
                    character.reference_paths
                    or character.reference_video_paths
                    or character.reference_audio_paths
                )
                else "story_generated"
            )

            character.build_identity_profile()
            character.build_story_state_profile()

            characters.append(
                character
            )

        self.references.validate(
            characters,
            require_images=False,
        )

        return characters

    # ========================================================
    # SHOT REBINDING
    # ========================================================

    @staticmethod
    def _native_audio_policy(
        soundscape: str,
    ) -> str:

        return (
            f"{soundscape.strip()} "
            "Native H3 audio policy: when no supplied "
            "reference audio exists, generate suitable "
            "scene ambience, dialogue and music natively "
            "from the production context."
        )

    def _rebind_shots(
        self,
        plan: dict,
        characters: list[Character],
    ) -> None:

        by_name = {
            EntityResolver.normalize(character.name): character
            for character in characters
        }
        canonical_names = set(by_name)
        aliases = EntityResolver.build_alias_map(canonical_names)

        scenes_by_id = {
            str(
                scene.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip(): scene
            for scene in plan.get(
                "scenes",
                [],
            )
            if isinstance(
                scene,
                dict,
            )
            and str(
                scene.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()
        }

        for scene in plan.get(
            "scenes",
            [],
        ):
            scene["shot_ids"] = []

        for index, raw in enumerate(
            plan.get(
                "shots",
                [],
            ),
            start=1,
        ):

            scene_id = str(
                raw.get(
                    "scene_id",
                    "",
                )
                or ""
            ).strip()

            raw_characters = raw.get(
                "characters",
                [],
            )

            if not raw_characters:

                scene = scenes_by_id.get(
                    scene_id,
                    {},
                )

                raw_characters = scene.get(
                    "characters",
                    [],
                )

            names = []

            for value in (
                raw_characters
                if isinstance(
                    raw_characters,
                    (list, tuple, set),
                )
                else [raw_characters]
            ):

                name = str(
                    value
                ).strip()

                normalized = EntityResolver.normalize(name)
                resolved = (
                    aliases.get(normalized)
                    or aliases.get(EntityResolver.strip_honorific(normalized))
                )

                if (
                    resolved
                    and resolved in by_name
                    and resolved not in {
                        EntityResolver.normalize(item)
                        for item in names
                    }
                ):
                    names.append(
                        by_name[resolved].name
                    )

            selected = [
                by_name[EntityResolver.normalize(name)]
                for name in names
                if EntityResolver.normalize(name) in by_name
            ]

            images = []
            videos = []
            audio = []
            reference_roles = []
            bindings = {}

            for character in selected:

                character_images = (
                    character.normalized_reference_paths()
                )

                character_videos = (
                    character.normalized_video_paths()
                )

                character_audio = (
                    character.normalized_audio_paths()
                )

                bindings[
                    character.name
                ] = character_images

                # Reserve capacity for runtime storyboard and (when this is not
                # the first shot in a scene) the previous-shot final-frame relay.
                # First shots need one reserved slot (storyboard); continuation
                # shots need two (storyboard + previous frame).
                has_same_scene_previous = any(
                    isinstance(previous, dict)
                    and str(previous.get("scene_id", "") or "").strip() == scene_id
                    for previous in plan.get("shots", [])[: max(0, index - 1)]
                )
                reserved_runtime_slots = 1 + int(has_same_scene_previous)
                identity_capacity = max(1, H3_MAX_REFERENCE_IMAGES - reserved_runtime_slots)
                for path in character_images:
                    if (
                        path not in images
                        and len(images) < identity_capacity
                    ):
                        images.append(
                            path
                        )
                        reference_roles.append({
                            "path": path,
                            "role": "character_identity",
                            "character_name": character.name,
                            "character_id": character.character_id,
                            "label": (
                                f"Canonical visual identity reference for {character.name}; "
                                "use for face, hair, body structure and stable identity only."
                            ),
                            "priority": 80,
                        })

                for path in character_videos:
                    if (
                        path not in videos
                        and len(videos)
                        < H3_MAX_REFERENCE_VIDEOS
                    ):
                        videos.append(
                            path
                        )

                for path in character_audio:
                    if (
                        path not in audio
                        and len(audio)
                        < H3_MAX_REFERENCE_AUDIO
                    ):
                        audio.append(
                            path
                        )

            locks = (
                IdentityContinuity.build_locks(
                    selected,
                    names,
                )
            )

            reference_bindings = (
                IdentityContinuity
                .build_reference_bindings(
                    images,
                    bindings,
                )
            )

            workflow_mode = str(
                plan.get(
                    "workflow_mode",
                    WORKFLOW_AUTO,
                )
            )

            if (
                workflow_mode
                == WORKFLOW_AUTO
            ):

                if plan.get(
                    "profile"
                ) == PROFILE_TURBO:

                    workflow_mode = (
                        WORKFLOW_TURBO_REF2V
                    )

                else:

                    workflow_mode = (
                        WORKFLOW_REF2V
                    )

            if (
                workflow_mode
                == WORKFLOW_TURBO_REF2V
                or plan.get(
                    "profile"
                ) == PROFILE_TURBO
            ):

                steps = TURBO_STEPS

                workflow_mode = (
                    WORKFLOW_TURBO_REF2V
                )

            else:

                steps = H3_STEPS

                if workflow_mode not in {
                    WORKFLOW_REF2V,
                }:

                    workflow_mode = (
                        WORKFLOW_REF2V
                    )

            soundscape = str(
                raw.get(
                    "overall_soundscape",
                    "",
                )
                or ""
            )

            negative = str(
                raw.get(
                    "negative_prompt",
                    "",
                )
                or ""
            )

            description = str(
                raw.get(
                    "detailed_description",
                    "",
                )
                or raw.get(
                    "visual_prompt",
                    "",
                )
            )

            h3_object = Shot(
                shot_id=str(
                    raw.get(
                        "shot_id",
                        f"shot_{index:03d}",
                    )
                ),

                scene_id=str(
                    raw.get(
                        "scene_id",
                        "",
                    )
                ),

                order=int(
                    raw.get(
                        "order",
                        index,
                    )
                ),

                duration_seconds=float(
                    raw.get(
                        "duration_seconds",
                        5.2,
                    )
                ),

                characters=names,

                location=str(
                    raw.get(
                        "location",
                        "",
                    )
                ),

                action=str(
                    raw.get(
                        "action",
                        "",
                    )
                ),

                camera_shot=str(
                    raw.get(
                        "camera_shot",
                        "",
                    )
                ),

                camera_movement=str(
                    raw.get(
                        "camera_movement",
                        "",
                    )
                ),

                lens_and_depth_of_field=str(
                    raw.get(
                        "lens_and_depth_of_field",
                        "",
                    )
                    or ""
                ),

                composition_notes=str(
                    raw.get(
                        "composition_notes",
                        "",
                    )
                    or ""
                ),

                lighting=str(
                    raw.get(
                        "lighting",
                        "",
                    )
                ),

                color_temperature=str(
                    raw.get(
                        "color_temperature",
                        "",
                    )
                    or ""
                ),

                mood=str(
                    raw.get(
                        "mood",
                        "",
                    )
                ),

                visual_prompt=str(
                    raw.get(
                        "visual_prompt",
                        "",
                    )
                ),

                retention_analysis=str(
                    raw.get(
                        "retention_analysis",
                        "",
                    )
                ),

                detailed_description=description,

                overall_soundscape=(
                    self._native_audio_policy(
                        soundscape
                    )
                ),

                non_diegetic_music=str(
                    raw.get(
                        "non_diegetic_music",
                        "",
                    )
                ),

                negative_prompt=negative,

                continuity_notes=str(
                    raw.get(
                        "continuity_notes",
                        "",
                    )
                ),

                seed=(
                    int(
                        raw["seed"]
                    )
                    if raw.get(
                        "seed"
                    ) is not None
                    else (
                        100000
                        + index
                    )
                ),

                reference_images=images,
                reference_videos=videos,
                reference_roles=reference_roles,

                reference_audio=(
                    audio[0]
                    if audio
                    else None
                ),

                reference_audio_paths=audio,

                reference_audio_by_character={
                    name: (
                        by_name[
                            name.lower()
                        ]
                        .normalized_audio_paths()
                    )
                    for name in names
                    if name.lower()
                    in by_name
                },

                reference_video_by_character={
                    name: (
                        by_name[
                            name.lower()
                        ]
                        .normalized_video_paths()
                    )
                    for name in names
                    if name.lower()
                    in by_name
                },

                speaking_characters=[
                    str(value).strip()
                    for value in (
                        raw.get(
                            "speaking_characters",
                            names,
                        )
                        or []
                    )
                    if (
                        str(value).strip().lower()
                        in {
                            item.lower()
                            for item in names
                        }
                    )
                ],

                speech_text=str(
                    raw.get(
                        "speech_text",
                        "",
                    )
                ),

                dialogue_events=list(
                    raw.get(
                        "dialogue_events",
                        [],
                    )
                    or []
                ),

                is_scene_boundary=bool(
                    raw.get(
                        "is_scene_boundary",
                        False,
                    )
                ),

                character_spatial_bboxes=dict(
                    raw.get(
                        "character_spatial_bboxes",
                        {},
                    )
                    or {}
                ),

                character_spatial_regions=dict(
                    raw.get(
                        "character_spatial_regions",
                        {},
                    )
                    or {}
                ),

                character_spatial_bboxes_start=dict(
                    raw.get(
                        "character_spatial_bboxes_start",
                        {},
                    )
                    or {}
                ),

                character_spatial_bboxes_end=dict(
                    raw.get(
                        "character_spatial_bboxes_end",
                        {},
                    )
                    or {}
                ),

                character_spatial_regions_start=dict(
                    raw.get(
                        "character_spatial_regions_start",
                        {},
                    )
                    or {}
                ),

                character_spatial_regions_end=dict(
                    raw.get(
                        "character_spatial_regions_end",
                        {},
                    )
                    or {}
                ),

                continuity_start_state=dict(
                    raw.get(
                        "continuity_start_state",
                        {},
                    )
                    or {}
                ),

                continuity_end_state=dict(
                    raw.get(
                        "continuity_end_state",
                        {},
                    )
                    or {}
                ),

                continuity_repair_applied=bool(
                    raw.get(
                        "continuity_repair_applied",
                        False,
                    )
                ),

                identity_fingerprints=dict(
                    raw.get(
                        "identity_fingerprints",
                        {},
                    )
                    or {}
                ),

                storyboard_reference=raw.get(
                    "storyboard_reference"
                ),

                reference_role_manifest=raw.get(
                    "reference_role_manifest"
                ),

                reference_bindings=(
                    reference_bindings
                ),

                identity_locks=locks,

                workflow_mode=workflow_mode,

                keyframe_images=[],
                keyframe_positions=[],
                extend_take_source_video=None,

                width=H3_WIDTH,
                height=H3_HEIGHT,
                fps=H3_FPS,
                frames_per_shot=(
                    H3_FRAMES_PER_SHOT
                ),
                steps=steps,
            )

            updated = (
                h3_object.to_dict()
            )

            raw.clear()
            raw.update(
                updated
            )

            for scene in plan.get(
                "scenes",
                [],
            ):

                if scene.get(
                    "scene_id"
                ) == raw.get(
                    "scene_id"
                ):

                    scene.setdefault(
                        "shot_ids",
                        [],
                    ).append(
                        raw["shot_id"]
                    )

        plan[
            "characters"
        ] = [
            character.to_dict()
            for character
            in characters
        ]


    # ========================================================
    # CHECKPOINT / RESUME
    # ========================================================

    def _new_session_id(self) -> str:
        return (
            "production_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:12]
        )

    def _checkpoint_store(self) -> ProductionCheckpoint:
        return ProductionCheckpoint(
            self.project_root
        )

    def _save_checkpoint(
        self,
        session_id: str,
        *,
        mode: str,
        user_input: str,
        workflow_mode: str,
        profile: str,
        status: str,
        stage: str,
        base_plan: dict,
        director_plan: dict,
        completed_scene_ids: list[str] | None = None,
        current_scene_id: str = "",
        error: str = "",
    ) -> None:
        store = self._checkpoint_store()
        state = {
            "mode": mode,
            "user_input": str(user_input or ""),
            "user_input_sha256": store.digest_text(user_input),
            "plan_sha256": (
                store.plan_digest(director_plan)
                if isinstance(director_plan, dict) and director_plan
                else ""
            ),
            "director_sha256": store.digest_file(
                self.project_root / "planner" / "qwen_director.py"
            ),
            "workflow_mode": workflow_mode,
            "profile": profile,
            "status": status,
            "stage": stage,
            "base_plan": base_plan,
            "director_plan": director_plan,
            "completed_scene_ids": list(
                completed_scene_ids or []
            ),
            "current_scene_id": current_scene_id,
            "error": error,
            "updated_at": datetime.now().isoformat(),
        }
        store.save(
            session_id,
            state,
        )

    def _load_resume_state(
        self,
        session_id: str,
        mode: str,
        user_input: str,
        workflow_mode: str,
        profile: str,
    ) -> dict:
        store = self._checkpoint_store()
        state = store.load(
            session_id
        )

        if state.get("mode") != mode:
            raise RuntimeError(
                "Checkpoint mode does not match the requested mode."
            )

        if state.get("user_input_sha256") != store.digest_text(user_input):
            raise RuntimeError(
                "Checkpoint story/input does not match the requested story."
            )

        if state.get("workflow_mode", workflow_mode) != workflow_mode:
            raise RuntimeError(
                "Checkpoint workflow mode does not match the requested workflow."
            )

        if state.get("profile", profile) != profile:
            raise RuntimeError(
                "Checkpoint profile does not match the requested profile."
            )

        expected_director = store.digest_file(
            self.project_root / "planner" / "qwen_director.py"
        )
        if state.get("director_sha256") != expected_director:
            raise RuntimeError(
                "Checkpoint was created by a different qwen_director.py. "
                "Start a new production instead of mixing director versions."
            )

        status = str(
            state.get("status", "") or ""
        )
        if status == "completed":
            raise RuntimeError(
                "Production checkpoint is already completed."
            )

        if not isinstance(
            state.get("base_plan"),
            dict,
        ):
            raise RuntimeError(
                "Checkpoint is missing base_plan."
            )

        if not isinstance(state.get("director_plan"), dict):
            raise RuntimeError("Checkpoint is missing director_plan.")

        checkpoint_plan_hash = str(state.get("plan_sha256", "") or "").strip()
        if not checkpoint_plan_hash:
            raise RuntimeError(
                "Checkpoint has no plan fingerprint; refusing unsafe resume."
            )
        actual_plan_hash = store.plan_digest(state["director_plan"])
        if checkpoint_plan_hash != actual_plan_hash:
            raise RuntimeError(
                "Checkpoint plan fingerprint is invalid; the persisted production plan may have been modified."
            )

        return state

    def resume_production_plan(
        self,
        session_id: str,
    ) -> dict:
        store = self._checkpoint_store()
        state = store.load(
            session_id
        )

        mode = str(
            state.get("mode", "") or ""
        ).strip()

        original_user_input = str(
            state.get(
                "user_input",
                "",
            )
            or ""
        ).strip()

        if not original_user_input:
            raise RuntimeError(
                "Checkpoint is missing the original user input."
            )

        return self.create_production_plan(
            mode=mode,
            user_input=original_user_input,
            workflow_mode=str(
                state.get(
                    "workflow_mode",
                    WORKFLOW_AUTO,
                )
            ),
            profile=str(
                state.get(
                    "profile",
                    "base",
                )
            ),
            resume_session_id=session_id,
        )

    # ========================================================
    # PLAN
    # ========================================================

    @staticmethod
    def _write_storyboard(
        path: Path,
        plan: dict,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_name(
            f".{path.name}.{uuid.uuid4().hex}.tmp"
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
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink(
                    missing_ok=True
                )

    @staticmethod
    def _refresh_shot_prompt(shot: dict) -> None:
        from dataclasses import fields
        field_names = {field.name for field in fields(Shot)}
        payload = {key: value for key, value in shot.items() if key in field_names}
        shot_obj = Shot(**payload)
        shot["h3_prompt"] = shot_obj.h3_prompt()

    def _enforce_production_contracts(
        self,
        plan: dict,
        characters: list[Character],
    ) -> dict:
        """Run deterministic production contracts after the Qwen pass.

        The director is intentionally unloaded before this method is called.
        Continuity repair therefore MUST remain deterministic here: never call
        the Qwen director from this stage.
        """
        character_dicts = [
            character.to_dict()
            for character in characters
        ]

        self._rebind_shots(
            plan,
            characters,
        )

        DialogueTimeline(
            character_dicts
        ).apply_to_plan(plan)

        ledger = ContinuityLedger(
            self.project_root,
            str(
                plan.get(
                    "production_id",
                    "",
                )
            ),
        )

        try:
            ledger.apply(
                plan,
                character_dicts,
            )
            return plan
        except ContinuityViolation:
            # Deterministic field-level fallback only. The creative shot is
            # never regenerated here, and the unloaded Qwen director is never
            # called from this post-director phase.
            return ledger.apply_field_level_fallback(
                plan,
                character_dicts,
            )

    def create_production_plan(
        self,
        mode: str,
        user_input: str,
        workflow_mode: str = WORKFLOW_AUTO,
        profile: str = "base",
        resume_session_id: str | None = None,
    ) -> dict:

        checkpoint_store = self._checkpoint_store()

        if resume_session_id:

            state = self._load_resume_state(
                resume_session_id,
                mode,
                user_input,
                workflow_mode,
                profile,
            )

            production_id = str(
                resume_session_id
            )

            base_plan = deepcopy(
                state[
                    "base_plan"
                ]
            )

            director_resume_state = deepcopy(
                state
            )

        else:

            production_id = self._new_session_id()

            base_plan = (
                self.planner.build(
                    mode=mode,
                    user_input=user_input,
                    workflow_mode=workflow_mode,
                    profile=profile,
                )
            )

            director_resume_state = None

            try:
                self._save_checkpoint(
                    production_id,
                    mode=mode,
                    user_input=user_input,
                    workflow_mode=workflow_mode,
                    profile=profile,
                    status="running",
                    stage="initialized",
                    base_plan=deepcopy(base_plan),
                    director_plan={},
                )
            except Exception as checkpoint_error:
                raise RuntimeError(
                    "Initial production checkpoint could not be persisted: "
                    + str(checkpoint_error)
                ) from checkpoint_error

        try:

            try:

                plan = self.director.enrich_plan(
                    mode=mode,
                    user_input=user_input,
                    base_plan=base_plan,
                    checkpoint_session_id=production_id,
                    resume_state=director_resume_state,
                )

            except Exception as exc:

                try:
                    state = checkpoint_store.load(
                        production_id
                    )
                    state["status"] = "failed"
                    state["stage"] = "director"
                    state["error"] = str(exc)
                    state["updated_at"] = datetime.now().isoformat()
                    checkpoint_store.save(
                        production_id,
                        state,
                    )
                except Exception:
                    pass

                raise

        finally:

            self.director.unload()

        if mode == PRESERVE_USER_STORY_MODE:

            plan["story"] = base_plan["story"]

        characters = self._character_objects(
            plan.get(
                "characters",
                [],
            )
        )

        self._rebind_shots(
            plan,
            characters,
        )

        # Deterministic production-enforcement passes. Qwen remains the
        # creative source, while timing and continuity are finalized here.
        plan["production_id"] = production_id
        plan = self._enforce_production_contracts(plan, characters)
        ensure_plan_lineage(plan, production_id)
        character_dicts = [character.to_dict() for character in characters]

        # Finalize scene-boundary flags before building the storyboard manifest.
        # The manifest and all downstream reference contracts must see the same
        # final scene-boundary semantics.
        previous_by_scene = {}
        for shot in plan.get("shots", []):
            scene_id = str(shot.get("scene_id", "")).strip()
            previous = previous_by_scene.get(scene_id)
            explicit_boundary = bool(shot.get("is_scene_boundary", False))
            shot["is_scene_boundary"] = previous is None or explicit_boundary
            if previous is not None:
                shot["previous_shot"] = previous.get("shot_id")
                previous["next_shot"] = shot.get("shot_id")
            previous_by_scene[scene_id] = shot

        storyboard = StoryboardReferenceBuilder(
            self.project_root,
            production_id,
        ).build(
            plan,
            character_dicts,
        )
        plan["storyboard_reference"] = storyboard["path"]
        plan["storyboard_reference_manifest"] = storyboard["manifest_path"]

        def _bindings_from_roles(reference_roles: list[dict]) -> list[str]:
            bindings = []
            for index, role in enumerate(reference_roles, start=1):
                kind = str(role.get("role", "")).strip().lower()
                if kind == "storyboard":
                    label = "Unified storyboard for sequencing, composition and blocking; not the canonical character identity source."
                elif kind == "previous_shot_last_frame":
                    label = "Previous shot final-frame continuity reference."
                else:
                    character = str(role.get("character_name", "")).strip()
                    label = (
                        f"Canonical visual identity reference for {character}; use for face, hair, body structure and stable identity only."
                        if character else "Production visual reference."
                    )
                bindings.append(f"<Picture {index}> = {label}")
            return bindings

        for shot in plan.get("shots", []):
            shot["storyboard_reference"] = storyboard["path"]
            shot["reference_role_manifest"] = storyboard["manifest_path"]
            shot["reference_bindings"] = _bindings_from_roles(
                list(shot.get("reference_roles", []) or [])
            )
            with self._manifest_lock:
                entry = StoryboardReferenceBuilder.update_manifest(
                    storyboard["manifest_path"],
                    str(shot.get("shot_id", "")),
                    list(shot.get("reference_images", []) or []),
                    list(shot.get("reference_roles", []) or []),
                    list(shot.get("reference_bindings", []) or []),
                    actual_runtime_order=False,
                )
                StoryboardReferenceBuilder.assert_manifest_invariant(
                    entry,
                    list(shot.get("reference_images", []) or []),
                    list(shot.get("reference_bindings", []) or []),
                )
            self._refresh_shot_prompt(shot)

        scene_characters = {}

        for shot in plan.get(
            "shots",
            [],
        ):

            scene_characters.setdefault(
                shot["scene_id"],
                set(),
            ).update(
                str(name).lower()
                for name in (
                    shot.get(
                        "characters",
                        [],
                    )
                    or []
                )
            )

        seen_characters = {}
        shared = False

        for scene_id, names in scene_characters.items():
            for name in names:
                if name in seen_characters:
                    shared = True
                seen_characters[name] = scene_id

        plan["parallel_safe"] = not shared

        plan["delivery_width"] = DELIVERY_WIDTH
        plan["delivery_height"] = DELIVERY_HEIGHT
        plan["delivery_fps"] = DELIVERY_FPS
        plan["upscale_width"] = int(
            plan.get(
                "upscale_width",
                UPSCALE_WIDTH,
            )
        )
        plan["upscale_height"] = int(
            plan.get(
                "upscale_height",
                UPSCALE_HEIGHT,
            )
        )

        plan["preview_ready"] = True
        plan["created_at"] = datetime.now().isoformat()
        plan["production_id"] = production_id

        plan.setdefault(
            "profile",
            profile,
        )
        plan.setdefault(
            "workflow_mode",
            workflow_mode,
        )

        # This is the durable final storyboard used by the UI/render boundary.
        session_dir = checkpoint_store.session_dir(
            production_id
        )
        plan_path = (
            session_dir
            / "story_preview.json"
        )
        self._write_storyboard(
            plan_path,
            plan,
        )

        # Mark the planning stage READY. Rendering is a separate persisted
        # stage owned by ProductionRunner; do not claim production completed
        # until the final H3 video has actually been assembled.
        try:
            self._save_checkpoint(
                production_id,
                mode=mode,
                user_input=user_input,
                workflow_mode=workflow_mode,
                profile=profile,
                status="ready",
                stage="production_plan",
                base_plan=deepcopy(base_plan),
                director_plan=deepcopy(plan),
                completed_scene_ids=[],
            )
        except Exception as exc:
            raise RuntimeError(
                "Production plan checkpoint could not be persisted: "
                + str(exc)
            ) from exc

        return plan

    def resume_latest_production_plan(
        self,
        mode: str,
        user_input: str,
    ) -> dict:

        state = self._checkpoint_store().latest_resumable(
            mode,
            user_input,
        )

        if state is None:
            raise RuntimeError(
                "No resumable production checkpoint was found "
                "for the supplied mode and story."
            )

        return self.resume_production_plan(
            str(
                state["session_id"]
            )
        )

    def unload_models(
        self,
    ):

        self.director.unload()

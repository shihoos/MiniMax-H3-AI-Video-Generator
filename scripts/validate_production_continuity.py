from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planner.cinematic_compiler import CinematicCompiler
from pipeline.continuity_ledger import ContinuityLedger
from pipeline.dialogue_timeline import DialogueTimeline
from pipeline.timeline import ProductionTimeline
from pipeline.storyboard_reference_builder import StoryboardReferenceBuilder
from schemas.character import Character
from schemas.shot import Shot


def make_character() -> Character:
    return Character(
        character_id="char_alex",
        name="Alex",
        role="protagonist",
        description="A determined young man.",
        personality="Focused.",
        appearance={
            "facial_features": "angular face",
            "hair": "short dark hair",
            "body_build": "lean",
            "body_proportions": "balanced",
            "skin_tone": "warm brown",
            "age_range": "20s",
        },
        clothing={"jacket": "navy"},
    )


def qwen_shot(
    scene_id: str,
    shot_id: str,
    *,
    boundary: bool,
    start_state: dict | None = None,
    start_bbox: list[float] | None = None,
    end_bbox: list[float] | None = None,
    start_region: str = "foreground_left",
    end_region: str = "foreground_center",
) -> dict:
    return {
        "shot_id": shot_id,
        "scene_id": scene_id,
        "duration_seconds": 5.0,
        "characters": ["Alex"],
        "location": "station" if scene_id == "scene_001" else "alley",
        "action": "Alex moves through the scene.",
        "camera_shot": "wide",
        "camera_movement": "tracking",
        "lens_and_depth_of_field": "deep focus",
        "composition_notes": "leading lines",
        "lighting": "cool overhead light" if scene_id == "scene_001" else "cool night light",
        "color_temperature": "cool",
        "mood": "urgent",
        "visual_prompt": "Alex moves through the scene.",
        "speaking_characters": ["Alex"],
        "speech_text": 'Alex: "We have to go now."',
        "dialogue_events": [
            {
                "speaker": "Alex",
                "text": '"We have to go now."',
                "continues_from_previous_shot": False,
                "continues_to_next_shot": False,
            }
        ],
        "continuity_start_state": start_state
        or {
            "location": "station" if scene_id == "scene_001" else "alley",
            "lighting": "cool overhead light" if scene_id == "scene_001" else "cool night light",
            "state_description": "scene opening",
        },
        "continuity_end_state": {
            "location": "station" if scene_id == "scene_001" else "alley",
            "lighting": "cool overhead light" if scene_id == "scene_001" else "cool night light",
            "state_description": "scene end",
        },
        "is_scene_boundary": boundary,
        "character_spatial_bboxes_start": {"Alex": list(start_bbox or [0.20, 0.20, 0.50, 0.95])},
        "character_spatial_bboxes_end": {"Alex": list(end_bbox or [0.35, 0.20, 0.65, 0.95])},
        "character_spatial_regions_start": {"Alex": start_region},
        "character_spatial_regions_end": {"Alex": end_region},
    }


def main() -> None:
    character = make_character()
    characters = [character.to_dict()]
    compiler = CinematicCompiler({"alex"})

    scene = {
        "scene_id": "scene_001",
        "characters": ["Alex"],
        "location": "station",
        "description": "Alex moves through the station.",
        "scene_objective": "Reach the platform door.",
        "lighting": "cool overhead light",
        "color_temperature": "cool",
        "mood": "urgent",
        "continuity_notes": "Preserve station continuity.",
    }

    first_qwen = qwen_shot("scene_001", "shot_001", boundary=True)
    first = compiler.compile_shot(scene, first_qwen, 1)

    # Critical integration check: real compiler output must instantiate the real Shot dataclass.
    assert isinstance(first["continuity_start_state"], dict)
    assert isinstance(first["continuity_end_state"], dict)
    shot_obj = Shot(**first)
    assert isinstance(shot_obj.continuity_start_state, dict)
    assert isinstance(shot_obj.continuity_end_state, dict)
    assert '"We have to go now."' in shot_obj.h3_prompt()
    prompt_text = shot_obj.h3_prompt()
    assert "<d>[English]" in prompt_text
    assert "subject_definitions:" in prompt_text
    assert "summary:" in prompt_text
    assert "retention_analysis:" in prompt_text
    assert "detailed_description:" in prompt_text
    assert "overall_soundscape:" in prompt_text
    assert "non_diegetic_music:" in prompt_text

    second_qwen = qwen_shot(
        "scene_001",
        "shot_002",
        boundary=False,
        start_state=first["continuity_end_state"],
        start_bbox=[0.35, 0.20, 0.65, 0.95],
        start_region="foreground_center",
    )
    second = compiler.compile_shot(scene, second_qwen, 2)
    Shot(**second)

    plan = {
        "production_id": "validation",
        "characters": characters,
        "shots": [first, second],
    }
    DialogueTimeline(characters).apply_to_plan(plan)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ledger = ContinuityLedger(root, "validation")
        ledger.apply(plan, characters)
        ContinuityLedger.validate(plan)

        storyboard = StoryboardReferenceBuilder(root, "validation").build(
            plan,
            characters,
        )
        manifest_path = Path(storyboard["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Builder must include the storyboard reference before the manifest is finalized.
        for entry in manifest["shots"].values():
            assert storyboard["path"] in entry["reference_images"]
            assert len(entry["reference_images"]) == len(entry["references"])
            assert len(entry["reference_images"]) == len(entry["picture_bindings"])
            for index, (ref, record) in enumerate(
                zip(entry["reference_images"], entry["references"]), start=1
            ):
                assert record["path"] == ref
                assert int(record["picture_index"]) == index

        # Simulate the runner's final runtime order and verify the manifest invariant.
        shot = plan["shots"][1]
        refs = list(shot["reference_images"])
        roles = [dict(item) for item in shot["reference_roles"]]
        last_frame = str(root / "shot_001_last_frame.png")
        from PIL import Image
        Image.new("RGB", (8, 8), "white").save(last_frame)
        refs = [ref for ref in refs if ref != last_frame]
        roles = [role for role in roles if str(role.get("path", "")) != last_frame]
        refs.append(last_frame)
        roles.append({
            "path": last_frame,
            "role": "previous_shot_last_frame",
            "label": "Previous shot final-frame continuity reference.",
            "priority": 100,
        })
        bindings = [
            (
                f"<Picture {index}> = "
                + (
                    "Previous shot final-frame continuity reference."
                    if role["role"] == "previous_shot_last_frame"
                    else "Unified storyboard for sequencing, composition and blocking; not the canonical character identity source."
                    if role["role"] == "storyboard"
                    else f"Canonical visual identity reference for {role.get('character_name', '')}; use for stable identity only."
                )
            )
            for index, role in enumerate(roles, start=1)
        ]
        entry = StoryboardReferenceBuilder.update_manifest(
            manifest_path,
            shot["shot_id"],
            refs,
            roles,
            bindings,
        )
        StoryboardReferenceBuilder.assert_manifest_invariant(entry, refs, bindings)

        # Scene-boundary reset: environment/spatial state flushes, identity fingerprints remain.
        boundary = compiler.compile_shot(
            {**scene, "scene_id": "scene_002", "location": "alley"},
            qwen_shot("scene_002", "shot_003", boundary=True),
            1,
        )
        boundary_plan = {
            "production_id": "validation_boundary",
            "characters": characters,
            "shots": [first, boundary],
        }
        ledger2 = ContinuityLedger(root, "validation_boundary")
        ledger2.apply(boundary_plan, characters)
        boundary_shot = boundary_plan["shots"][1]
        assert boundary_shot["is_scene_boundary"] is True
        assert boundary_shot["continuity_start_state"]["scene_boundary_reset"] is True
        assert boundary_shot["identity_fingerprints"]["Alex"] == character.identity_fingerprint

        # Continuity mismatch must be rejected rather than silently overwritten.
        bad = dict(second)
        bad["continuity_start_state"] = {"state_description": "wrong"}
        bad_plan = {"shots": [first, bad]}
        try:
            ContinuityLedger.validate_proposed(bad_plan)
        except Exception:
            pass
        else:
            raise AssertionError("Continuity mismatch was not rejected.")

    # Legacy string compatibility must be normalized by the compiler, not passed into Shot.
    legacy = dict(first)
    legacy.pop("continuity_start_state", None)
    legacy.pop("continuity_end_state", None)
    legacy["continuity_state_start"] = '{"state_description":"legacy start"}'
    legacy["continuity_state_end"] = "legacy end"
    normalized = compiler.compile_shot(scene, legacy, 1)
    assert normalized["continuity_start_state"] == {"state_description": "legacy start"}
    assert normalized["continuity_end_state"] == {"state_description": "legacy end"}
    Shot(**normalized)

    # H3 duration-grid contract.
    assert DialogueTimeline.h3_legal_frames(5.0) == 124
    assert abs(DialogueTimeline.h3_effective_duration_seconds(5.0) - 124 / 24.0) < 1e-9
    assert DialogueTimeline.h3_legal_frames(15.0) == 362

    # Requested shot timing must be canonicalized to the exact H3 frame-grid
    # duration so the plan/timeline/ComfyUI render cannot silently diverge.
    timing_plan = {
        "shots": [
            {
                "shot_id": "timing_001",
                "scene_id": "timing_scene",
                "order": 1,
                "duration_seconds": 5.2,
            }
        ]
    }
    ProductionTimeline(timing_plan).build()
    timing_shot = timing_plan["shots"][0]
    assert abs(timing_shot["duration_seconds"] - 141 / 24.0) < 1e-9
    assert timing_shot["frames_per_shot"] == 141
    assert abs(timing_shot["h3_effective_duration_seconds"] - timing_shot["duration_seconds"]) < 1e-9
    assert abs(timing_shot["requested_duration_seconds"] - 5.2) < 1e-9

    max_timing_plan = {
        "shots": [
            {
                "shot_id": "timing_max",
                "scene_id": "timing_scene",
                "order": 1,
                "duration_seconds": 15.0,
            }
        ]
    }
    ProductionTimeline(max_timing_plan).build()
    max_timing_shot = max_timing_plan["shots"][0]
    assert max_timing_shot["frames_per_shot"] == 362
    assert abs(max_timing_shot["duration_seconds"] - 362 / 24.0) < 1e-4

    print("PRODUCTION CONTINUITY VALIDATION PASSED")



def validate_production_isolation(root: Path) -> None:
    from pipeline.h3_scene_continuity import H3SceneContinuity
    from pipeline.identity_anchor_store import IdentityAnchorStore

    a = H3SceneContinuity(root, "production_A")
    b = H3SceneContinuity(root, "production_B")
    assert a.root != b.root
    assert str(a.root).endswith("production_A")
    assert str(b.root).endswith("production_B")

    ia = IdentityAnchorStore(root, production_id="production_A")
    ib = IdentityAnchorStore(root, production_id="production_B")
    assert ia.root != ib.root


def validate_reference_manifest_semantics(root: Path) -> None:
    from pipeline.storyboard_reference_builder import StoryboardReferenceBuilder

    builder = StoryboardReferenceBuilder(root, "manifest_test")
    manifest_path = builder.root / "reference_role_manifest.json"
    payload = {
        "manifest_version": 2,
        "production_id": "manifest_test",
        "storyboard": {},
        "shots": {},
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    refs = ["/tmp/identity.png", "/tmp/storyboard.png"]
    roles = [
        {"path": refs[0], "role": "character_identity", "priority": 95},
        {"path": refs[1], "role": "storyboard", "priority": 90},
    ]
    bindings = ["<Picture 1> = identity", "<Picture 2> = storyboard"]
    entry = StoryboardReferenceBuilder.update_manifest(
        manifest_path, "shot_001", refs, roles, bindings, actual_runtime_order=False
    )
    assert entry["actual_runtime_order"] is False
    StoryboardReferenceBuilder.assert_manifest_invariant(entry, refs, bindings)
    entry = StoryboardReferenceBuilder.update_manifest(
        manifest_path, "shot_001", list(reversed(refs)), list(reversed(roles)),
        ["<Picture 1> = storyboard", "<Picture 2> = identity"],
        actual_runtime_order=True,
    )
    assert entry["actual_runtime_order"] is True
    StoryboardReferenceBuilder.assert_manifest_invariant(
        entry, list(reversed(refs)), ["<Picture 1> = storyboard", "<Picture 2> = identity"]
    )


def validate_ffprobe_stream_duration() -> None:
    import shutil
    import subprocess
    import tempfile
    from pipeline.dialogue_duration import FFProbeMediaDurationProvider

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        print("ffprobe stream-duration regression skipped: ffmpeg/ffprobe unavailable.")
        return

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        video = root / "test.mp4"
        command = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=24:d=2.0",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=2.0",
            "-t", "2.0",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(video),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])

        # Create a second file whose video is 2 seconds and audio only 1 second.
        mismatch = root / "mismatch.mp4"
        command = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:r=24:d=2.0",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono:d=1.0",
            "-t", "2.0",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(mismatch),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])

        provider = FFProbeMediaDurationProvider(ffprobe)
        video_seconds = provider.duration_seconds(mismatch, stream_selector="v:0")
        audio_seconds = provider.duration_seconds(mismatch, stream_selector="a:0")
        assert video_seconds > audio_seconds + 0.4, (video_seconds, audio_seconds)


if __name__ == "__main__":
    main()
    # P0/P1/P2 regression gates.
    validate_production_isolation(ROOT)
    validate_reference_manifest_semantics(ROOT)
    validate_ffprobe_stream_duration()
    print("P0/P1/P2 regression gates PASSED.")

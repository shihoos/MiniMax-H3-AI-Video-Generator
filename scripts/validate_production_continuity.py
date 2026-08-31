from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.continuity_ledger import ContinuityLedger
from pipeline.dialogue_timeline import DialogueTimeline
from pipeline.storyboard_reference_builder import StoryboardReferenceBuilder
from pipeline.continuity_ledger import ContinuityViolation
from schemas.character import Character


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        character = Character(
            character_id="char_alex",
            name="Alex",
            role="protagonist",
            description="A determined young man.",
            personality="Focused.",
            appearance={"hair": "short dark hair", "skin_tone": "warm brown"},
            clothing={"jacket": "navy"},
        )
        chars = [character.to_dict()]
        plan = {
            "production_id": "validation",
            "characters": chars,
            "shots": [
                {
                    "shot_id": "scene_001_shot_001",
                    "scene_id": "scene_001",
                    "order": 1,
                    "duration_seconds": 5.2,
                    "characters": ["Alex"],
                    "location": "station",
                    "action": "Alex enters the station.",
                    "camera_shot": "wide",
                    "camera_movement": "tracking",
                    "lens_and_depth_of_field": "normal deep focus",
                    "composition_notes": "leading lines",
                    "dialogue_events": [{
                        "speaker": "Alex",
                        "text": '"We have to go now."',
                        "continues_to_next_shot": False,
                    }],
                    "speaking_characters": ["Alex"],
                    "speech_text": '"We have to go now."',
                    "visual_prompt": "Alex enters the station.",
                    "continuity_state_start": "standing at the station entrance",
                    "continuity_state_end": "standing beside the platform door",
                },
                {
                    "shot_id": "scene_001_shot_002",
                    "scene_id": "scene_001",
                    "order": 2,
                    "duration_seconds": 5.2,
                    "characters": ["Alex"],
                    "location": "station",
                    "action": "Alex reaches the platform door.",
                    "camera_shot": "medium",
                    "camera_movement": "push-in",
                    "lens_and_depth_of_field": "normal shallow focus",
                    "composition_notes": "subject isolation",
                    "dialogue_events": [],
                    "visual_prompt": "Alex reaches the platform door.",
                    "continuity_state_start": "incorrect state that will be replaced",
                    "continuity_state_end": "touching the platform door",
                    "character_spatial_bboxes_start": {"Alex": [0.45, 0.20, 0.75, 0.95]},
                    "is_scene_boundary": False,
                    "character_spatial_bboxes_end": {"Alex": [0.45, 0.20, 0.75, 0.95]},
                },
                {
                    "shot_id": "scene_002_shot_001",
                    "scene_id": "scene_002",
                    "order": 3,
                    "duration_seconds": 5.2,
                    "characters": ["Alex"],
                    "location": "alley",
                    "action": "Alex exits into the alley.",
                    "camera_shot": "wide",
                    "camera_movement": "static",
                    "lens_and_depth_of_field": "deep focus",
                    "composition_notes": "open negative space",
                    "lighting": "cool night light",
                    "continuity_state_start": "must reset scene state",
                    "continuity_state_end": "standing in the alley",
                    "is_scene_boundary": True,
                    "character_spatial_bboxes": {"Alex": [0.55, 0.20, 0.95, 0.95]},
                },
            ],
        }
        DialogueTimeline(chars).apply_to_plan(plan)
        ContinuityLedger(tmp_root, "validation").apply(plan, chars)
        ContinuityLedger.validate(plan)
        storyboard = StoryboardReferenceBuilder(tmp_root, "validation").build(plan, chars)
        assert Path(storyboard["path"]).is_file()
        assert plan["shots"][1]["continuity_start_state"] == plan["shots"][0]["continuity_end_state"]
        assert plan["shots"][0]["dialogue_events"][0]["speaker_id"] == "char_alex"
        assert plan["shots"][1]["continuity_start_state"] == plan["shots"][0]["continuity_end_state"]
        assert plan["shots"][2]["is_scene_boundary"] is True
        assert plan["shots"][2]["continuity_start_state"]["scene_boundary_reset"] is True
        entry = storyboard["manifest"]["shots"]["scene_001_shot_001"] if "manifest" in storyboard else json.loads(Path(storyboard["manifest_path"]).read_text(encoding="utf-8"))["shots"]["scene_001_shot_001"]
        assert entry["reference_images"] == []
        print("PRODUCTION CONTINUITY VALIDATION PASSED")


if __name__ == "__main__":
    main()

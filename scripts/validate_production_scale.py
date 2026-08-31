from __future__ import annotations

import ast
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    from pipeline.job_queue import ProductionJobQueue
    from pipeline.seed_lineage import ensure_plan_lineage, stable_seed
    from pipeline.storyboard_cache import StoryboardCache
    from pipeline.visual_state_observer import VisualStateObserver
    from pipeline.visual_feedback import VisualFeedbackEngine

    with tempfile.TemporaryDirectory(prefix="h3-scale-") as td:
        root = Path(td)
        queue = ProductionJobQueue(root / "jobs.sqlite3")
        job_id = queue.submit("prod", root / "plan.json", {"kind": "smoke"})
        check(queue.get(job_id)["status"] == "queued", "SQLite queue submit failed.")
        claimed = queue.claim_next()
        check(claimed and claimed["status"] == "running", "SQLite queue claim failed.")
        queue.complete(job_id, {"ok": True})
        check(queue.get(job_id)["status"] == "completed", "SQLite queue completion failed.")

        plan = {"shots": [{"shot_id": "shot_001", "scene_id": "scene_001", "order": 1, "location": "room", "action": "enter", "characters": [], "camera_shot": "wide", "camera_movement": "static", "lens_and_depth_of_field": "normal", "composition_notes": "centered", "lighting": "soft", "color_temperature": "neutral", "mood": "calm", "visual_prompt": "A cinematic entrance."}]}
        ensure_plan_lineage(plan, "prod")
        first = dict(plan["shots"][0])
        first_seed = stable_seed("prod", first)
        check(plan["shots"][0]["shot_uid"], "shot_uid missing.")
        plan["shots"][0]["order"] = 99
        check(stable_seed("prod", plan["shots"][0]) == first_seed, "Shot seed changed after order mutation.")

        cache = StoryboardCache(root)
        digest = cache.digest({"shot": "same"})
        image = root / "a.png"; manifest = root / "a.json"
        image.write_bytes(b"png"); manifest.write_text("{}", encoding="utf-8")
        cache.store(digest, image, manifest)
        image.unlink(); manifest.unlink()
        check(cache.restore(digest, image, manifest), "Storyboard cache restore failed.")

        observer = VisualStateObserver(root)
        check(callable(observer.observe_frame), "Visual observer is unavailable.")
        feedback = VisualFeedbackEngine(observer)
        check(callable(feedback.analyze), "Visual feedback engine is unavailable.")

    runner = (ROOT / "execution" / "production_runner.py").read_text(encoding="utf-8")
    check("self._manifest_lock = threading.RLock()" in runner, "ProductionRunner manifest lock is missing.")
    check("production_id=production_id" in runner, "ProductionRunner production-scoped initialization is missing.")

    print("Production scale contracts PASSED.")


if __name__ == "__main__":
    main()

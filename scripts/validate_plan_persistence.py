from __future__ import annotations

import ast
import tempfile
import threading
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.production_plan_store import ProductionPlanStore


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_controller_contract() -> None:
    path = ROOT / "ui" / "storyboard_gradio.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    controller = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ProductionController"
    )
    methods = {
        node.name: node
        for node in controller.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    required = {
        "apply_timeline_edits",
        "approve_and_generate",
        "create_retake_request",
        "_queue_worker_loop",
        "_execute_approved_plan",
    }
    require(required <= methods.keys(), "ProductionController persistence methods are incomplete.")

    for name in ("apply_timeline_edits", "approve_and_generate", "create_retake_request"):
        source = ast.get_source_segment(text, methods[name]) or ""
        require("self._lock.acquire" in source, f"{name} must acquire the controller lock.")
        require("self._plan_store.lock" in source, f"{name} must acquire the per-plan lock.")
        require("ProductionPlanStore.atomic_save_unlocked" in source, f"{name} must use atomic plan persistence.")

    worker_source = ast.get_source_segment(text, methods["_queue_worker_loop"]) or ""
    require(worker_source.count("self._plan_store.lock") >= 2, "Queue worker must lock plan persistence on success and failure paths.")
    require("job_completed = False" in worker_source, "Queue worker must distinguish DB completion from post-completion persistence errors.")

    execute_source = ast.get_source_segment(text, methods["_execute_approved_plan"]) or ""
    require("self._lock.acquire(timeout=3600.0)" in execute_source, "Execution must acquire the bounded controller lock.")
    require("self._plan_store.lock(plan_path)" in execute_source, "Execution must lock plan persistence at plan boundaries.")
    require("render_plan_sha = ProductionCheckpoint.plan_digest(plan)" in execute_source, "Execution must fingerprint the render plan before rendering.")
    require("Production plan changed while rendering" in execute_source, "Execution must reject stale-plan finalization.")
    require("ProductionPlanStore.atomic_save_unlocked" in execute_source, "Execution must use atomic plan persistence.")


def validate_atomic_concurrent_updates() -> None:
    with tempfile.TemporaryDirectory(prefix="h3-plan-store-") as td:
        path = Path(td) / "story_preview.json"
        ProductionPlanStore.atomic_save(path, {"counter": 0})
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    with ProductionPlanStore.lock(path):
                        plan = ProductionPlanStore.load_unlocked(path)
                        plan["counter"] = int(plan.get("counter", 0)) + 1
                        ProductionPlanStore.atomic_save_unlocked(path, plan)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        require(not any(thread.is_alive() for thread in threads), "Plan-store worker thread did not finish.")
        require(not errors, f"Plan-store concurrent update failed: {errors!r}")
        final_plan = ProductionPlanStore.load(path)
        require(final_plan.get("counter") == 200, f"Lost concurrent plan updates: {final_plan!r}")
        require(not list(Path(td).glob("*.tmp")), "Temporary plan files were left behind.")


def main() -> None:
    validate_controller_contract()
    validate_atomic_concurrent_updates()
    print("PASS plan persistence race contracts")


if __name__ == "__main__":
    main()

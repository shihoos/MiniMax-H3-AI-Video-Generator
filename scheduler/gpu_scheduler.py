from __future__ import annotations

import os
import threading
import traceback
from collections import deque


SCHEDULER_JOIN_TIMEOUT_SECONDS = float(
    os.getenv(
        "H3_SCHEDULER_JOIN_TIMEOUT",
        str(6 * 60 * 60),
    )
)


class GPUScheduler:
    """Run independent scene jobs across GPUs without feeding new jobs after failure."""

    def __init__(self, gpu_ids=None):
        self.gpu_ids = sorted({
            int(gpu) for gpu in (gpu_ids if gpu_ids is not None else [0])
        })
        if not self.gpu_ids:
            raise ValueError("At least one GPU is required.")

    def run_independent(self, jobs, worker_function):
        indexed_jobs = deque(enumerate(jobs))
        queue_lock = threading.Lock()
        result_lock = threading.Lock()
        stop_event = threading.Event()
        results = []
        failures = []

        def worker(gpu_id):
            while not stop_event.is_set():
                with queue_lock:
                    if not indexed_jobs:
                        return
                    index, job = indexed_jobs.popleft()
                try:
                    result = worker_function(gpu_id, job)
                    with result_lock:
                        results.append((index, result))
                except Exception as error:
                    with result_lock:
                        failures.append(
                            (
                                gpu_id,
                                index,
                                job,
                                error,
                                traceback.format_exc(),
                            )
                        )
                    stop_event.set()
                    return

        threads = [
            threading.Thread(
                target=worker,
                args=(gpu_id,),
                name=f"h3-gpu-{gpu_id}",
                daemon=False,
            )
            for gpu_id in self.gpu_ids
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(
                timeout=SCHEDULER_JOIN_TIMEOUT_SECONDS
            )

        alive = [
            thread.name
            for thread in threads
            if thread.is_alive()
        ]

        if alive:
            stop_event.set()
            for thread in threads:
                if thread.is_alive():
                    thread.join(
                        timeout=5.0
                    )

            raise TimeoutError(
                "GPU scheduler workers did not finish within "
                f"{SCHEDULER_JOIN_TIMEOUT_SECONDS}s: "
                + ", ".join(alive)
            )

        if failures:
            messages = "\n".join(
                (
                    f"GPU {gpu}: job {job!r} failed:\n"
                    f"{traceback_text}"
                )
                for gpu, _index, job, _error, traceback_text
                in failures
            )
            raise RuntimeError(
                "GPU jobs failed:\n" + messages
            )

        results.sort(key=lambda item: item[0])
        return [result for _index, result in results]

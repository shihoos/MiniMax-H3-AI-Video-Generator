from __future__ import annotations

import threading
from collections import deque


class GPUScheduler:

    def __init__(
        self,
        gpu_ids=None,
    ):
        self.gpu_ids = [
            int(gpu)
            for gpu in (
                gpu_ids
                if gpu_ids is not None
                else [0]
            )
        ]

        if not self.gpu_ids:
            raise ValueError(
                "At least one GPU is required."
            )

    def run_independent(
        self,
        jobs,
        worker_function,
    ):
        """
        One worker thread per physical GPU.

        Each GPU owns at most one active scene at a time.
        Shots inside a scene remain sequential in ProductionRunner.
        """

        queue = deque(jobs)
        queue_lock = threading.Lock()
        result_lock = threading.Lock()

        results = []
        failures = []

        def worker(
            gpu_id,
        ):
            while True:

                with queue_lock:
                    if not queue:
                        return

                    job = queue.popleft()

                try:
                    result = worker_function(
                        gpu_id,
                        job,
                    )

                    with result_lock:
                        results.append(
                            result
                        )

                except Exception as error:

                    with result_lock:
                        failures.append(
                            (
                                gpu_id,
                                job,
                                error,
                            )
                        )

        threads = [
            threading.Thread(
                target=worker,
                args=(gpu_id,),
                name=f"h3-gpu-{gpu_id}",
                daemon=True,
            )
            for gpu_id in self.gpu_ids
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        if failures:
            messages = "\n".join(
                f"GPU {gpu}: {error}"
                for gpu, _job, error
                in failures
            )

            raise RuntimeError(
                "GPU jobs failed:\n"
                + messages
            )

        return results

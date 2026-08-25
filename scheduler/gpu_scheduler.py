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
                else [0, 1]
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
        Parallel execution is allowed only for jobs
        explicitly marked independent.
        """

        queue = deque(jobs)
        queue_lock = threading.Lock()
        result_lock = threading.Lock()

        results = []
        failures = []

        def worker(gpu_id):

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

    def run_sequential(
        self,
        jobs,
        worker_function,
        gpu_id=None,
    ):
        """
        Continuity-dependent shots/scenes must use this path.
        """

        if gpu_id is None:
            gpu_id = self.gpu_ids[0]

        results = []

        for job in jobs:

            results.append(
                worker_function(
                    gpu_id,
                    job,
                )
            )

        return results

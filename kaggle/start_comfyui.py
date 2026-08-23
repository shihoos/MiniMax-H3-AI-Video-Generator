from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

COMFY = ROOT / "ComfyUI"

LOG_DIR = (
    ROOT
    / "data"
    / "comfy_logs"
)


def gpu_ids():

    configured = os.getenv(
        "H3_GPU_IDS"
    )

    if configured:
        return [
            int(value.strip())
            for value in configured.split(",")
            if value.strip()
        ]

    try:

        import torch

        count = torch.cuda.device_count()

        if count >= 2:
            return [0, 1]

        if count == 1:
            return [0]

    except Exception:
        pass

    return [0]


def start_worker(
    gpu_id,
    port,
):

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logfile = (
        LOG_DIR
        / f"gpu_{gpu_id}.log"
    )

    env = os.environ.copy()

    env[
        "CUDA_VISIBLE_DEVICES"
    ] = str(gpu_id)

    env[
        "PYTORCH_CUDA_ALLOC_CONF"
    ] = (
        "expandable_segments:True"
    )

    handle = logfile.open(
        "a",
        encoding="utf-8",
    )

    process = subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--lowvram",
            "--cpu-vae",
        ],
        cwd=str(COMFY),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    return (
        process,
        handle,
    )


def main():

    ids = gpu_ids()

    processes = []

    for index, gpu_id in enumerate(
        ids
    ):

        port = (
            8188 + index
        )

        process, handle = (
            start_worker(
                gpu_id,
                port,
            )
        )

        processes.append(
            (
                process,
                handle,
                gpu_id,
                port,
            )
        )

        print(
            f"GPU {gpu_id}: "
            f"http://127.0.0.1:{port} "
            f"PID={process.pid}"
        )

    try:

        while any(
            process.poll() is None
            for process, _, _, _
            in processes
        ):
            time.sleep(2)

    except KeyboardInterrupt:

        for process, handle, _, _ in (
            processes
        ):

            if process.poll() is None:
                process.send_signal(
                    signal.SIGTERM
                )

            handle.close()


if __name__ == "__main__":
    main()

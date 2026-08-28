from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from pathlib import Path


class H3Runtime:

    @staticmethod
    def clear_cuda():
        try:
            import torch

            gc.collect()

            if not torch.cuda.is_available():
                return

            for device_id in range(
                torch.cuda.device_count()
            ):
                with torch.cuda.device(
                    device_id
                ):
                    torch.cuda.empty_cache()

                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass

        except Exception:
            pass

    @staticmethod
    def worker_environment(
        gpu_id,
    ):
        env = os.environ.copy()

        env[
            "CUDA_VISIBLE_DEVICES"
        ] = str(gpu_id)

        env[
            "PYTORCH_CUDA_ALLOC_CONF"
        ] = (
            "expandable_segments:True"
        )

        env[
            "PYTHONUNBUFFERED"
        ] = "1"

        return env

    @staticmethod
    def launch_worker(
        comfy_root: Path,
        gpu_id: int,
        port: int,
        log_path: Path,
    ):
        log_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        handle = log_path.open(
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
            cwd=str(
                comfy_root
            ),
            env=(
                H3Runtime
                .worker_environment(
                    gpu_id
                )
            ),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        return (
            process,
            handle,
        )

    @staticmethod
    def wait_http(
        url,
        timeout=300,
    ):
        import urllib.request

        started = time.time()

        last_error = None

        while (
            time.time() - started
            < timeout
        ):
            try:
                with urllib.request.urlopen(
                    url,
                    timeout=5,
                ) as response:
                    if response.status == 200:
                        return
            except Exception as error:
                last_error = error

            time.sleep(2)

        detail = (
            f" Last error: {last_error}"
            if last_error is not None
            else ""
        )

        raise TimeoutError(
            f"ComfyUI did not start: {url}."
            + detail
        )

    @classmethod
    def launch_workers(
        cls,
        project_root: Path,
        gpu_ids: list[int],
        base_port: int = 8188,
    ):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "NVIDIA CUDA is required for "
                "the production H3 runtime."
            )

        available = torch.cuda.device_count()

        for gpu_id in gpu_ids:
            if gpu_id < 0 or gpu_id >= available:
                raise RuntimeError(
                    f"GPU {gpu_id} is unavailable. "
                    f"Detected {available} GPUs."
                )

        comfy_root = (
            Path(project_root)
            / "ComfyUI"
        )

        log_root = (
            Path(project_root)
            / "data"
            / "workers"
        )

        processes = {}

        try:
            for offset, gpu_id in enumerate(
                gpu_ids
            ):
                port = base_port + offset

                process, handle = (
                    cls.launch_worker(
                        comfy_root,
                        gpu_id,
                        port,
                        log_root
                        / f"gpu_{gpu_id}.log",
                    )
                )

                try:
                    cls.wait_http(
                        f"http://127.0.0.1:{port}/system_stats"
                    )
                except Exception:
                    # A failed startup must not leak previously started workers.
                    try:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=10)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
                    finally:
                        handle.close()
                    raise

                if process.poll() is not None:
                    handle.close()
                    raise RuntimeError(
                        f"ComfyUI worker for GPU {gpu_id} exited "
                        f"before becoming ready."
                    )

                processes[gpu_id] = {
                    "process": process,
                    "handle": handle,
                    "url": (
                        f"http://127.0.0.1:{port}"
                    ),
                    "port": port,
                }

        except Exception:
            cls.stop_workers(
                processes
            )
            raise

        return processes

    @staticmethod
    def stop_workers(
        workers,
    ):
        for item in workers.values():
            process = item[
                "process"
            ]

            if process.poll() is None:
                process.terminate()

                try:
                    process.wait(
                        timeout=15
                    )
                except subprocess.TimeoutExpired:
                    process.kill()

            handle = item.get(
                "handle"
            )

            if handle:
                handle.close()

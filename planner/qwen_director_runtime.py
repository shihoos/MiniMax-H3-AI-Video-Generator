from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import faulthandler
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from functools import wraps
from pathlib import Path

import requests

from planner.config import (
    DIRECTOR_KAGGLE_INPUT_ROOT,
    DIRECTOR_MAX_TOKENS,
    DIRECTOR_MODEL_ENV,
    DIRECTOR_MODEL_PATH,
    DIRECTOR_N_CTX,
    DIRECTOR_TEMPERATURE,
    DIRECTOR_TOP_P,
    DIRECTOR_STORY_THINKING_TEMPERATURE,
    DIRECTOR_STORY_THINKING_TOP_P,
    DIRECTOR_STORY_THINKING_TOP_K,
    DIRECTOR_STORY_THINKING_MIN_P,
    DIRECTOR_STORY_WRITER_TOP_K,
    DIRECTOR_STORY_WRITER_MIN_P,
    DIRECTOR_STORY_PRESENCE_PENALTY,
    DIRECTOR_VLLM_GPU_MEMORY_UTILIZATION,
    DIRECTOR_VLLM_MAX_MODEL_LEN,
    DIRECTOR_VLLM_PORT,
    DIRECTOR_VLLM_HOST,
    DIRECTOR_VLLM_TENSOR_PARALLEL_SIZE,
    DIRECTOR_VLLM_ENV_DIR,
    DIRECTOR_VLLM_MAX_NUM_SEQS,
    DIRECTOR_VLLM_SPECULATIVE_METHOD,
    DIRECTOR_VLLM_SPECULATIVE_MODEL_PATH,
    DIRECTOR_VLLM_SPECULATIVE_TOKENS,
    DIRECTOR_VLLM_GENERATION_CONFIG,
    DIRECTOR_VLLM_SAFETENSORS_LOAD_STRATEGY,
    DIRECTOR_VLLM_ENFORCE_EAGER,
    DIRECTOR_VLLM_JIT_WARMUP,
    DIRECTOR_VLLM_VERSION,
    DIRECTOR_VLLM_ALLOW_SPECULATIVE_MODEL_OVERRIDE,
    director_enabled,
)

NO_THINK_SUFFIX = "\n/no_think"


def _verbose() -> bool:
    return os.getenv("H3_DIRECTOR_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}


# H3_AUTOFIX_FAST_PORT_PROBE_V1
def _h3_probe_host(host: str, port: int, timeout: float = 0.25) -> bool:
    """H3_AUTOFIX_FAST_PORT_PROBE_V1: quick TCP connect probe."""
    import socket as _h3_socket
    _ph = host
    if _ph in ("0.0.0.0", "::", ""):
        _ph = "127.0.0.1"
    try:
        with _h3_socket.socket(_h3_socket.AF_INET, _h3_socket.SOCK_STREAM) as _s:
            _s.settimeout(float(timeout))
            return _s.connect_ex((_ph, int(port))) == 0
    except Exception:
        return False


# H3_V6_PARALLEL_STAGING: Kaggle's /kaggle/input is a GCSFUSE mount. One sequential reader gets ~10-15 MB/s per stream, and
# two tensor-parallel workers each read every shard, so loading straight from it took 6+ minutes per shard. Copying the
# checkpoint once with many concurrent ranged reads, then loading from local disk, restores the ~5 minute start.
def _stage_mode() -> str:
    value = os.getenv("H3_DIRECTOR_STAGE_LOCAL", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "on"
    if value in {"0", "false", "no", "off"}:
        return "off"
    if value in {"auto", "default", ""}:
        return "auto"
    raise RuntimeError("H3_DIRECTOR_STAGE_LOCAL must be one of: auto, 0, 1.")


def _filesystem_type(path: Path) -> str:
    try:
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", str(path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        return result.stdout.strip().lower()
    except Exception:
        return ""


def _should_stage_local(source_path: Path) -> bool:
    """Stage only checkpoints on Kaggle's network-backed input mount (or when forced)."""
    mode = _stage_mode()
    if mode == "off":
        return False
    try:
        resolved = source_path.resolve()
    except OSError:
        resolved = source_path
    if not str(resolved).startswith("/kaggle/input/"):
        return False
    if mode == "on":
        return True
    fs_type = _filesystem_type(resolved)
    # An unreadable filesystem type under /kaggle/input is treated as network storage: copying is the safe choice.
    return fs_type == "" or fs_type.startswith(("nfs", "lustre", "s3fs", "gcsfuse", "fuse"))


def _parallel_copy_file(src, dst, workers: int = 64, chunk_bytes: int = 32 * 1024 * 1024, progress=None) -> None:
    """Copy one large file with many concurrent ranged reads (pread/pwrite are thread-safe)."""
    src, dst = str(src), str(dst)
    size = os.path.getsize(src)
    with open(dst, "wb") as handle:
        handle.truncate(size)
    ranges = [(offset, min(chunk_bytes, size - offset)) for offset in range(0, size, chunk_bytes)]
    io_block = 8 * 1024 * 1024

    def copy_range(item):
        offset, length = item
        fd_src = os.open(src, os.O_RDONLY)
        fd_dst = os.open(dst, os.O_WRONLY)
        try:
            done = 0
            while done < length:
                buf = os.pread(fd_src, min(io_block, length - done), offset + done)
                if not buf:
                    raise IOError(f"short read at byte {offset + done} of {src}")
                written = 0
                while written < len(buf):
                    written += os.pwrite(fd_dst, buf[written:], offset + done + written)
                done += len(buf)
                if progress is not None:
                    progress(len(buf))
        finally:
            os.close(fd_src)
            os.close(fd_dst)

    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="h3-stage-copy") as pool:
        for _ in pool.map(copy_range, ranges):
            pass
    if os.path.getsize(dst) != size:
        raise IOError(f"size mismatch after copy: {dst}")


_STAGE_SPACE_LOCK = threading.Lock()
_STAGE_RESERVED_BYTES = 0


class StoryTruncated(RuntimeError):
    """Raised when a text pass stops on the token cap; carries the partial text."""

    def __init__(self, message: str, partial: str = ""):
        super().__init__(message)
        self.partial = partial


class QwenJsonTruncated(RuntimeError):
    """Raised when structured Qwen output hits the token cap before completion."""

    def __init__(self, message: str, partial: str = ""):
        super().__init__(message)
        self.partial = partial

# Pin the Director sampling stream at the request layer as well as the server layer.
# This removes run-to-run RNG-state drift while preserving the existing temperature/top-p profile.
DIRECTOR_VLLM_SEED = int(os.getenv("H3_DIRECTOR_VLLM_SEED", "0"))

# Keep one Director server alive for the whole Python/Kaggle session.
# The orchestrator intentionally calls director.unload() after each operation;
# treating that as a hard vLLM shutdown causes a multi-minute cold start for
# every subsequent Director mode. The shared process/session below separates
# request-client lifetime from GPU model lifetime.
_SHARED_VLLM_LOCK = threading.RLock()
_SHARED_VLLM_PROCESS: subprocess.Popen | None = None
_SHARED_VLLM_LOG_HANDLE = None
_SHARED_VLLM_LOG_PATH: Path | None = None
_SHARED_VLLM_TOKENIZER = None

# H3_V51_KEEP_ALIVE_DEFAULT_V1: a story-only run leaves the server running so the next run attaches in seconds (set H3_DIRECTOR_VLLM_KEEP_ALIVE=0 to turn off).
os.environ.setdefault("H3_DIRECTOR_VLLM_KEEP_ALIVE", "1")


def _shutdown_shared_vllm_at_exit() -> None:
    global _SHARED_VLLM_PROCESS, _SHARED_VLLM_LOG_HANDLE, _SHARED_VLLM_TOKENIZER
    if os.getenv("H3_DIRECTOR_VLLM_KEEP_ALIVE", "0").strip().lower() in {"1", "true", "yes", "on"}:
        # Leave the server running so the next cell/process attaches in seconds.
        # Free the GPUs before rendering: pkill -f vllm.entrypoints
        _SHARED_VLLM_PROCESS = None
        return
    process = _SHARED_VLLM_PROCESS
    _SHARED_VLLM_PROCESS = None
    if process is not None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        except Exception:
            pass
    handle = _SHARED_VLLM_LOG_HANDLE
    _SHARED_VLLM_LOG_HANDLE = None
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
    _SHARED_VLLM_TOKENIZER = None


atexit.register(_shutdown_shared_vllm_at_exit)


def _with_faulthandler_watchdog(func):
    """Arm a long-lived traceback watchdog around one Director operation."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            seconds = float(os.getenv("H3_DIRECTOR_WATCHDOG_SECONDS", "900"))
        except (TypeError, ValueError):
            seconds = 900.0
        armed = seconds > 0
        if armed:
            try:
                faulthandler.dump_traceback_later(seconds, repeat=False, file=sys.stderr)
            except Exception:
                armed = False
        try:
            return func(*args, **kwargs)
        finally:
            if armed:
                try:
                    faulthandler.cancel_dump_traceback_later()
                except Exception:
                    pass
    return wrapped


# H3_V51_STRAY_VLLM_V1
def _h3_pid_alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
    except Exception:
        return False
    return stat.rsplit(")", 1)[-1].strip()[:1] not in {"Z", "X", ""}


def _h3_find_vllm_pids() -> list:
    """PIDs of a Director vLLM server (same cmdline identity check as shutdown_qwen_before_render), whichever process started it."""
    found = []
    need_port = f"--port {int(DIRECTOR_VLLM_PORT)}"
    try:
        names = os.listdir("/proc")
    except Exception:
        return found
    for name in names:
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            cmd = Path(f"/proc/{name}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except Exception:
            continue
        if ("vllm.entrypoints.cli.main" in cmd and " serve" in cmd
                and "qwen3-14b-awq" in cmd and need_port in cmd and _h3_pid_alive(int(name))):
            found.append(int(name))
    return found


def _h3_kill_stray_vllm(reason: str = "", grace: float = 30.0) -> int:
    """SIGTERM (then SIGKILL) every Director vLLM server and wait until the port is free. Returns how many were found."""
    import signal as _sig
    pids = _h3_find_vllm_pids()
    if not pids:
        return 0
    print(f"[QWEN] stopping vLLM Director pid(s) {pids} ({reason})", flush=True)
    for pid in pids:
        try:
            os.kill(pid, _sig.SIGTERM)
        except Exception:
            pass
    end = time.monotonic() + float(grace)
    while time.monotonic() < end and any(_h3_pid_alive(p) for p in pids):
        time.sleep(0.25)
    for pid in pids:
        if _h3_pid_alive(pid):
            try:
                os.killpg(os.getpgid(pid), _sig.SIGKILL)
            except Exception:
                try:
                    os.kill(pid, _sig.SIGKILL)
                except Exception:
                    pass
    end = time.monotonic() + 20.0
    while time.monotonic() < end and (_h3_probe_host(DIRECTOR_VLLM_HOST, DIRECTOR_VLLM_PORT) or any(_h3_pid_alive(p) for p in pids)):
        time.sleep(0.25)
    return len(pids)


# H3_AUTOFIX_SHUTDOWN_BEFORE_RENDER_V1
def shutdown_qwen_before_render(director=None) -> None:
    """SIGTERM the shared vLLM process only after verifying cmdline identity."""
    import signal as _h3_signal
    proc = _SHARED_VLLM_PROCESS
    if proc is None:
        # H3_V51: a server kept alive from an earlier run is not our child, but it still holds the GPUs the renderer needs.
        _h3_kill_stray_vllm("freeing the GPUs before render")
        return
    pid = proc.pid
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return
    except Exception as _exc:
        print(f"[QWEN] shutdown_qwen_before_render: cannot read cmdline: {_exc}", flush=True)
        return
    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
    need_port = f"--port {int(DIRECTOR_VLLM_PORT)}"
    if not ("vllm.entrypoints.cli.main" in cmdline
            and " serve" in cmdline
            and "qwen3-14b-awq" in cmdline
            and need_port in cmdline):
        print(
            f"[QWEN] shutdown_qwen_before_render: refusing SIGTERM pid {pid}; "
            f"cmdline mismatch (tail={cmdline[:300]!r})",
            flush=True,
        )
        return
    print(f"[QWEN] shutdown_qwen_before_render: SIGTERM pid {pid}", flush=True)
    try:
        proc.send_signal(_h3_signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass


class QwenDirectorRuntimeMixin:
    @staticmethod
    def _optional_directory_env(name: str) -> Path | None:
        value = os.getenv(name, "").strip()
        if not value:
            return None
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _strip_thinking(text: str) -> str:
        value = str(text or "").strip()
        value = re.sub(
            r"<think>.*?</think>",
            "",
            value,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if re.search(r"<think>", value, flags=re.IGNORECASE):
            value = re.split(
                r"<think>",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()
        return value

    @staticmethod
    def _stage_to_local_scratch(source_path: Path, label: str) -> Path:
        """Return a local copy of a /kaggle/input checkpoint (copied once with parallel ranged reads and reused).

        Falls back to the original path when staging is off, not needed, impossible (space), or fails.
        H3_DIRECTOR_STAGE_LOCAL=0 forces direct loading; =1 forces staging. Scratch is /kaggle/tmp or /tmp.
        """
        global _STAGE_RESERVED_BYTES
        source_path = Path(source_path).expanduser()
        if not source_path.is_dir():
            raise FileNotFoundError(f"{label} checkpoint directory is missing: {source_path}")
        if not _should_stage_local(source_path):
            if _verbose():
                print(f"[QWEN] checkpoint_source label={label!r} path={source_path} staging=off", flush=True)
            return source_path
        resolved = source_path.resolve()
        scratch = Path("/kaggle/tmp") if Path("/kaggle/tmp").is_dir() else Path("/tmp")
        target_dir = scratch / "staged_models" / resolved.name
        manifest_name = ".h3_stage_manifest.json"
        files = sorted(p for p in resolved.rglob("*") if p.is_file())
        fingerprint = {
            "schema": 2,
            "files": {str(p.relative_to(resolved)): int(p.stat().st_size) for p in files},
        }
        total = sum(fingerprint["files"].values())

        def valid(directory: Path) -> bool:
            try:
                manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
                if manifest != fingerprint:
                    return False
                return all(
                    (directory / name).is_file() and (directory / name).stat().st_size == size
                    for name, size in fingerprint["files"].items()
                ) and (directory / "config.json").is_file()
            except Exception:
                return False

        if target_dir.is_dir() and valid(target_dir):
            if _verbose():
                print(f"[QWEN] staged_checkpoint=reused label={label!r} path={target_dir}", flush=True)
            return target_dir

        reservation = total + 2 * (1024 ** 3)
        try:
            avail = os.statvfs(scratch).f_bavail * os.statvfs(scratch).f_frsize
            with _STAGE_SPACE_LOCK:
                if avail - _STAGE_RESERVED_BYTES < reservation:
                    print(
                        f"[QWEN] staging_skipped label={label!r}: {(avail - _STAGE_RESERVED_BYTES) / 2**30:.1f} GiB free, "
                        f"{reservation / 2**30:.1f} GiB needed; loading directly",
                        flush=True,
                    )
                    return source_path
                _STAGE_RESERVED_BYTES += reservation
        except Exception:
            print(f"[QWEN] staging_skipped label={label!r}: cannot verify scratch space; loading directly", flush=True)
            return source_path

        try:
            # Measured on Kaggle GCSFUSE: 1 reader 68 MiB/s, 16 -> 463, 64 -> 813 MiB/s (still rising).
            workers = max(1, int(os.getenv("H3_DIRECTOR_STAGE_WORKERS", "64")))
        except ValueError:
            workers = 64
        started = time.perf_counter()
        tmp_target = target_dir.with_name(f"{target_dir.name}.tmp_{os.getpid()}")
        copied = [0]
        last_print = [started]
        lock = threading.Lock()

        def note(count: int) -> None:
            with lock:
                copied[0] += count
                now = time.perf_counter()
                if _verbose() and now - last_print[0] >= 10.0:
                    last_print[0] = now
                    rate = copied[0] / max(now - started, 1e-6) / (1024 ** 2)
                    eta = (total - copied[0]) / max(rate * (1024 ** 2), 1.0)
                    print(
                        f"[QWEN] staging {label}: {copied[0] / 2**30:.2f}/{total / 2**30:.2f} GiB "
                        f"{rate:.0f} MiB/s eta={eta:.0f}s",
                        flush=True,
                    )

        if _verbose():
            print(
                f"[QWEN] staging_start label={label!r} size={total / 2**30:.2f} GiB workers={workers} "
                f"source={resolved} target={target_dir}",
                flush=True,
            )
        try:
            if tmp_target.exists():
                shutil.rmtree(tmp_target, ignore_errors=True)
            for path in files:
                relative = path.relative_to(resolved)
                destination = tmp_target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if path.stat().st_size >= 64 * 1024 * 1024:
                    _parallel_copy_file(path, destination, workers=workers, progress=note)
                else:
                    shutil.copyfile(path, destination)
                    note(path.stat().st_size)
            (tmp_target / manifest_name).write_text(json.dumps(fingerprint, sort_keys=True), encoding="utf-8")
            if not valid(tmp_target):
                raise RuntimeError("staged checkpoint failed post-copy validation")
            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            tmp_target.rename(target_dir)
        except Exception as exc:
            shutil.rmtree(tmp_target, ignore_errors=True)
            print(f"[QWEN] staging_failed label={label!r}: {exc}; loading directly from the dataset mount", flush=True)
            return source_path
        finally:
            with _STAGE_SPACE_LOCK:
                _STAGE_RESERVED_BYTES = max(0, _STAGE_RESERVED_BYTES - reservation)
        # A fresh 12 GiB copy leaves dirty page-cache pages that cannot be reclaimed until written out. Flush them in the
        # background while vLLM boots, so loading never competes with them for memory.
        threading.Thread(target=os.sync, name="h3-stage-sync", daemon=True).start()
        elapsed = time.perf_counter() - started
        if _verbose():
            print(
                f"[QWEN] staging_done label={label!r} {total / 2**30:.2f} GiB in {elapsed:.0f}s "
                f"({total / max(elapsed, 1e-6) / 2**20:.0f} MiB/s)",
                flush=True,
            )
        return target_dir

    def _record_qwen_call(
        self,
        *,
        call_name: str,
        elapsed: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        max_tokens: int = 0,
        min_tokens: int = 0,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format=None,
        finish_reason: str = "",
        cache_hit: bool = False,
        error: str = "",
        thinking_enabled: bool = False,
        thinking_token_budget: int | None = None,
        reasoning_tokens: int = 0,
    ) -> None:
        """Record and print bounded runtime telemetry for one Qwen call."""
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
        reasoning_tokens = max(0, int(reasoning_tokens or 0))
        elapsed = max(0.0, float(elapsed or 0.0))

        record = {
            "call_name": str(call_name or "unknown"),
            "elapsed_seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "decode_tps": (
                completion_tokens / elapsed
                if elapsed > 0 and completion_tokens > 0
                else 0.0
            ),
            "max_tokens": int(max_tokens or 0),
            "min_tokens": int(min_tokens or 0),
            "temperature": temperature,
            "top_p": top_p,
            "thinking_enabled": bool(thinking_enabled),
            "thinking_token_budget": (
                int(thinking_token_budget)
                if thinking_token_budget is not None
                else None
            ),
            "reasoning_tokens": reasoning_tokens,
            "response_format": (
                "json_schema"
                if isinstance(response_format, dict)
                and response_format.get("type") == "json_schema"
                else (
                    response_format.get("type")
                    if isinstance(response_format, dict)
                    else None
                )
            ),
            "finish_reason": str(finish_reason or ""),
            "cache_hit": bool(cache_hit),
            "error": str(error or ""),
        }

        calls = self._qwen_telemetry.setdefault("calls", [])
        calls.append(record)

        self._qwen_telemetry["total_elapsed_seconds"] += elapsed
        self._qwen_telemetry["prompt_tokens"] += prompt_tokens
        self._qwen_telemetry["completion_tokens"] += completion_tokens
        if cache_hit:
            self._qwen_telemetry["cache_hits"] += 1

        if "retry" in str(call_name).lower():
            self._qwen_telemetry["retries"] += 1

        print(
            "[QWEN]",
            call_name,
            f"elapsed={elapsed:.2f}s",
            f"prompt_tokens={prompt_tokens}",
            f"completion_tokens={completion_tokens}",
            f"reasoning_tokens={reasoning_tokens}" if reasoning_tokens else "",
            f"total_tokens={prompt_tokens + completion_tokens}",
            f"decode_tps={record['decode_tps']:.2f}",
            f"max_tokens={int(max_tokens or 0)}",
            f"min_tokens={int(min_tokens or 0)}" if int(min_tokens or 0) > 0 else "",
            (f"finish_reason={record['finish_reason']}" if record["finish_reason"] else ""),
            (f"cache_hit={cache_hit}" if cache_hit else ""),
            (f"error={error}" if error else ""),
            flush=True,
        )

    def _record_recovery(
        self,
        recovery_type: str,
        detail: str = "",
    ) -> None:
        self._qwen_telemetry["deterministic_recoveries"] += 1
        print(
            "[QWEN]",
            "recovery",
            f"type={recovery_type}",
            (f"detail={detail}" if detail else ""),
            flush=True,
        )

    def _print_qwen_summary(self, label: str = "SUMMARY") -> None:
        """Print a compact production-level Qwen accounting summary."""
        calls = list(self._qwen_telemetry.get("calls", []) or [])
        total_elapsed = float(
            self._qwen_telemetry.get("total_elapsed_seconds", 0.0) or 0.0
        )
        prompt_tokens = int(
            self._qwen_telemetry.get("prompt_tokens", 0) or 0
        )
        completion_tokens = int(
            self._qwen_telemetry.get("completion_tokens", 0) or 0
        )
        retries = int(self._qwen_telemetry.get("retries", 0) or 0)
        cache_hits = int(self._qwen_telemetry.get("cache_hits", 0) or 0)
        recoveries = int(
            self._qwen_telemetry.get("deterministic_recoveries", 0) or 0
        )

        if label == "FINAL":
            # One FINAL block per telemetry session; a second request with the
            # same call count would only repeat identical numbers.
            if self._qwen_telemetry.get("_final_printed_calls") == len(calls):
                return
            self._qwen_telemetry["_final_printed_calls"] = len(calls)

        print(f"[QWEN] ==================== {label} ====================", flush=True)
        print("[QWEN] calls=" + str(len(calls)), flush=True)
        print("[QWEN] prompt_tokens=" + str(prompt_tokens), flush=True)
        print("[QWEN] completion_tokens=" + str(completion_tokens), flush=True)
        print("[QWEN] total_tokens=" + str(prompt_tokens + completion_tokens), flush=True)
        print("[QWEN] total_elapsed=" + f"{total_elapsed:.2f}s", flush=True)
        print("[QWEN] retries=" + str(retries), flush=True)
        print("[QWEN] cache_hits=" + str(cache_hits), flush=True)
        print("[QWEN] deterministic_recoveries=" + str(recoveries), flush=True)
        if calls:
            names = ", ".join(str(item.get("call_name", "unknown")) for item in calls)
            print("[QWEN] call_sequence=" + names, flush=True)
        print("[QWEN] =====================================================", flush=True)

    def _trace_call(
        self,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        response,
        elapsed: float,
        error: str = "",
        response_format=None,
    ) -> None:
        if self._trace_dir is None:
            return

        raw_content = ""
        if isinstance(response, dict):
            try:
                message = response["choices"][0]["message"]
                raw_content = str(message.get("content") or "")
            except Exception:
                raw_content = ""

        reasoning_tokens = 0
        safe_response = response
        if isinstance(response, dict):
            try:
                usage = response.get("usage", {}) or {}
                details = usage.get("completion_tokens_details", {}) or {}
                reasoning_tokens = int(
                    details.get("reasoning_tokens", 0)
                    or details.get("reasoning", 0)
                    or 0
                )
            except Exception:
                reasoning_tokens = 0
            try:
                safe_response = deepcopy(response)
                message = safe_response["choices"][0]["message"]
                if isinstance(message, dict):
                    # Never persist chain-of-thought in trace files.
                    message.pop("reasoning_content", None)
                    message.pop("reasoning", None)
            except Exception:
                safe_response = {"omitted": "raw_response_unavailable"}

        payload = {
            "call_name": call_name,
            "elapsed_seconds": elapsed,
            "error": error,
            "response_format": response_format,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_content": raw_content,
            "reasoning_tokens": reasoning_tokens,
            "raw_response": safe_response,
        }
        digest = hashlib.sha256(
            (
                self._cache_namespace
                + "\n"
                + call_name
                + "\n"
                + system_prompt
                + "\n"
                + user_prompt
            ).encode("utf-8")
        ).hexdigest()
        path = self._trace_dir / f"{call_name.replace(':', '_')}_{digest[:16]}.json"
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(
                "[QWEN]",
                "trace_write_failed",
                call_name,
                str(exc),
                flush=True,
            )

    def _cache_key(
        self,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict | None,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        json_mode: bool,
        disable_thinking: bool,
        thinking_token_budget: int | None = None,
    ) -> str:
        try:
            model_name = self._vllm_model_name()
        except Exception:
            model_name = "unknown"
        material = json.dumps(
            {
                "namespace": self._cache_namespace,
                "call_name": call_name,
                "system": system_prompt,
                "user": user_prompt,
                "schema": response_schema,
                "model": model_name,
                "model_path": str(self._model_path) if getattr(self, "_model_path", None) else "",
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_tokens),
                "json_mode": bool(json_mode),
                "disable_thinking": bool(disable_thinking),
                "thinking_token_budget": (
                    int(thinking_token_budget)
                    if thinking_token_budget is not None
                    else None
                ),
                "generation_config": str(DIRECTOR_VLLM_GENERATION_CONFIG),
                "tensor_parallel": int(DIRECTOR_VLLM_TENSOR_PARALLEL_SIZE),
                "max_model_len": int(DIRECTOR_VLLM_MAX_MODEL_LEN),
                "seed": int(DIRECTOR_VLLM_SEED),
                "speculative_method": str(DIRECTOR_VLLM_SPECULATIVE_METHOD),
                "speculative_model_path": str(DIRECTOR_VLLM_SPECULATIVE_MODEL_PATH),
                "speculative_tokens": int(DIRECTOR_VLLM_SPECULATIVE_TOKENS),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(
            material.encode("utf-8")
        ).hexdigest()

    def _cache_read(
        self,
        key: str,
    ) -> dict | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"{key}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
            result = payload.get("parsed")
            return result if isinstance(result, dict) else None
        except Exception:
            return None

    def _cache_write(
        self,
        key: str,
        parsed: dict,
        *,
        call_name: str,
        system_prompt: str,
        user_prompt: str,
        elapsed: float,
        raw_content: str,
    ) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_dir / f"{key}.json"
        payload = {
            "namespace": self._cache_namespace,
            "call_name": call_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "elapsed_seconds": elapsed,
            "raw_content": raw_content,
            "parsed": parsed,
        }
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            print(
                "[QWEN]",
                "cache_write_failed",
                call_name,
                str(exc),
                flush=True,
            )

    def _find_model(
        self,
    ) -> Path:
        """Resolve the directory containing the sharded AWQ Director checkpoint."""
        explicit = os.getenv(DIRECTOR_MODEL_ENV, "").strip()
        candidates: list[Path] = []

        if explicit:
            candidates.append(Path(explicit).expanduser())

        candidates.append(Path(DIRECTOR_MODEL_PATH))

        if DIRECTOR_KAGGLE_INPUT_ROOT.is_dir():
            try:
                candidates.extend(
                    p
                    for p in DIRECTOR_KAGGLE_INPUT_ROOT.rglob(
                        Path(DIRECTOR_MODEL_PATH).name
                    )
                    if p.is_dir()
                )
            except OSError:
                pass

        unique: list[Path] = []
        seen: set[str] = set()

        for candidate in candidates:
            candidate = Path(candidate)
            try:
                key = str(candidate.resolve())
            except OSError:
                key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)

        required = (
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "tokenizer.json",
        )

        for path in unique:
            if not path.is_dir():
                continue
            if all((path / item).is_file() for item in required):
                return path

        raise FileNotFoundError(
            "Qwen3-14B-AWQ director model was not found as a complete "
            "sharded checkpoint.\n"
            f"Expected directory: {DIRECTOR_MODEL_PATH}\n"
            f"Required files: {', '.join(required)}\n"
            "Attach the Kaggle qwen3-14b-awq dataset or set "
            f"{DIRECTOR_MODEL_ENV} to its directory."
        )

    def _find_speculator_model(self) -> Path:
        """Resolve the pinned local Qwen3-14B EAGLE-3 speculator checkpoint."""
        explicit = os.getenv("H3_DIRECTOR_VLLM_SPECULATIVE_MODEL_PATH", "").strip()
        if explicit and not DIRECTOR_VLLM_ALLOW_SPECULATIVE_MODEL_OVERRIDE:
            raise RuntimeError(
                "H3_DIRECTOR_VLLM_SPECULATIVE_MODEL_PATH is set, but speculative-model "
                "overrides are disabled. Set H3_DIRECTOR_VLLM_ALLOW_SPECULATIVE_MODEL_OVERRIDE=1 "
                "only for a deliberate non-production checkpoint override."
            )
        configured = (
            Path(explicit).expanduser()
            if explicit and DIRECTOR_VLLM_ALLOW_SPECULATIVE_MODEL_OVERRIDE
            else DIRECTOR_VLLM_SPECULATIVE_MODEL_PATH
        )

        candidates = [configured]

        if DIRECTOR_KAGGLE_INPUT_ROOT.is_dir():
            try:
                candidates.extend(
                    p
                    for p in DIRECTOR_KAGGLE_INPUT_ROOT.rglob(
                        configured.name
                    )
                    if p.is_dir()
                )
            except OSError:
                pass

        if DIRECTOR_VLLM_SPECULATIVE_METHOD != "eagle3":
            raise RuntimeError(
                "runtime_versions.yaml director.speculative_method must be eagle3 "
                "for the locked EAGLE-3 Director runtime."
            )

        if not DIRECTOR_VLLM_ALLOW_SPECULATIVE_MODEL_OVERRIDE and str(DIRECTOR_VLLM_SPECULATIVE_MODEL_PATH) != "/kaggle/input/eagle-3":
            raise RuntimeError(
                "runtime_versions.yaml director.speculative_model_path must be /kaggle/input/eagle-3 "
                "for the locked Eagle-3 Kaggle dataset."
            )

        for candidate in candidates:
            candidate = Path(candidate)
            config_path = candidate / "config.json"
            if not (candidate.is_dir() and config_path.is_file()):
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Invalid Eagle-3 speculator config: {config_path}") from exc

            spec = config.get("speculators_config", {}) or {}
            verifier = spec.get("verifier", {}) or {}
            if str(spec.get("algorithm", "")).strip().lower() != DIRECTOR_VLLM_SPECULATIVE_METHOD:
                continue
            if str(verifier.get("name_or_path", "")).strip() != "Qwen/Qwen3-14B":
                continue

            index_files = list(candidate.glob("*.index.json"))
            if index_files:
                for index_path in index_files:
                    try:
                        index = json.loads(index_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        raise RuntimeError(
                            f"Invalid EAGLE-3 model index: {index_path}"
                        ) from exc
                    weight_map = index.get("weight_map") if isinstance(index, dict) else None
                    if not isinstance(weight_map, dict) or not weight_map:
                        raise RuntimeError(
                            f"EAGLE-3 model index has no usable weight_map: {index_path}"
                        )
                    missing = sorted(
                        {
                            str(name)
                            for name in weight_map.values()
                            if not (candidate / str(name)).is_file()
                        }
                    )
                    if missing:
                        raise RuntimeError(
                            "EAGLE-3 checkpoint is incomplete; missing indexed weight files: "
                            + ", ".join(missing)
                        )
                return candidate.resolve()

            if any(any(candidate.glob(pattern)) for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")):
                return candidate.resolve()

            raise RuntimeError(
                f"EAGLE-3 checkpoint has config.json but no model weight files: {candidate}"
            )

        raise FileNotFoundError(
            "Qwen3-14B EAGLE-3 speculator checkpoint was not found. "
            "Expected Kaggle dataset path: /kaggle/input/eagle-3. "
            f"Resolved configuration path: {configured}."
        )

    @property
    def model_path(
        self,
    ) -> Path | None:
        return self._model_path

    @property
    def available(
        self,
    ) -> bool:
        return bool(
            director_enabled()
            and self._model_path is not None
            and self._model_path.is_dir()
            and (self._model_path / "model.safetensors.index.json").is_file()
        )

    def _vllm_base_url(self) -> str:
        return os.getenv(
            "H3_DIRECTOR_VLLM_BASE_URL",
            f"http://{DIRECTOR_VLLM_HOST}:{DIRECTOR_VLLM_PORT}/v1",
        ).rstrip("/")

    def _vllm_model_name(self) -> str:
        return os.getenv(
            "H3_DIRECTOR_VLLM_MODEL_NAME",
            self._model_path.name if self._model_path is not None else "Qwen3-14B-AWQ",
        )

    @staticmethod
    def _checkpoint_shard_summary(checkpoint: Path) -> dict:
        """Return shard sizes from the index without opening multi-GB weight files."""
        checkpoint = Path(checkpoint).expanduser().resolve()
        files: set[str] = set()
        for index_path in sorted(checkpoint.glob("*.index.json")):
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map", {}) if isinstance(index, dict) else {}
                if isinstance(weight_map, dict):
                    files.update(str(name) for name in weight_map.values())
            except Exception as exc:
                raise RuntimeError(f"Invalid checkpoint shard index: {index_path}: {exc}") from exc
        if not files:
            for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
                files.update(path.name for path in checkpoint.glob(pattern) if path.is_file())
        sizes = []
        for name in sorted(files):
            path = (checkpoint / name).resolve()
            if not path.is_relative_to(checkpoint):
                raise RuntimeError(f"Checkpoint index references a path outside its dataset directory: {name!r}")
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint index references a missing shard: {path}")
            sizes.append((name, int(path.stat().st_size)))
        largest_name, largest_bytes = max(sizes, key=lambda item: item[1], default=(None, 0))
        return {
            "directory": str(checkpoint),
            "shard_count": len(sizes),
            "total_shard_bytes": sum(size for _, size in sizes),
            "largest_shard_name": largest_name,
            "largest_shard_bytes": largest_bytes,
        }

    @staticmethod
    def _container_available_memory() -> int:
        """Free RAM as the container sees it: the smaller of psutil's host figure and the cgroup limit minus usage.

        The host-wide figure alone is misleading on Kaggle (seen 5.78 GiB and 29.9 GiB in different sessions) and ignores
        the notebook kernel's own memory. Page cache counts as used here, which errs on the safe side.
        """
        import psutil
        available = int(psutil.virtual_memory().available)
        for limit_file, usage_file in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ):
            try:
                raw = Path(limit_file).read_text().strip()
                if raw == "max" or int(raw) >= 1 << 60:
                    continue
                available = min(available, int(raw) - int(Path(usage_file).read_text().strip()))
                break
            except Exception:
                continue
        return max(0, available)

    @classmethod
    def _resolve_safetensors_strategy(cls, qwen_path: Path, eagle_path: Path) -> str:
        """Pick the safetensors loader.

        - Local disk (the staged copy): always lazy (mmap). Eager makes each tensor-parallel worker pull its whole shard into
          RAM (~tensor_parallel x 4.65 GiB) with no benefit on local disk; it exhausted memory and killed the notebook.
        - Network mount (direct loading): eager only if the container has the RAM headroom, otherwise lazy.
        An explicit H3_DIRECTOR_VLLM_SAFETENSORS_LOAD_STRATEGY is always honoured.
        """
        configured = DIRECTOR_VLLM_SAFETENSORS_LOAD_STRATEGY
        if configured != "eager" or os.getenv("H3_DIRECTOR_VLLM_SAFETENSORS_LOAD_STRATEGY", "").strip():
            return configured
        if not any(str(p).startswith("/kaggle/input/") for p in (qwen_path, eagle_path)):
            print("[QWEN] safetensors_load_strategy=lazy (checkpoints are on local disk; eager would only add RAM pressure)", flush=True)
            return "lazy"
        try:
            shard = max(
                cls._checkpoint_shard_summary(qwen_path)["largest_shard_bytes"],
                cls._checkpoint_shard_summary(eagle_path)["largest_shard_bytes"],
            )
            director = __import__("planner.config", fromlist=["RUNTIME"]).RUNTIME["director"]
            factor = float(director.get("eager_ram_safety_factor", 1.25))
            reserve = float(director.get("eager_ram_warning_reserve_gib", 4.0)) * (1024 ** 3)
            required = int(int(DIRECTOR_VLLM_TENSOR_PARALLEL_SIZE) * shard * factor + reserve)
            available = cls._container_available_memory()
        except Exception as exc:
            print(f"[QWEN] safetensors_load_strategy=lazy (eager headroom could not be measured: {exc})", flush=True)
            return "lazy"
        if available < required:
            print(
                "[QWEN] safetensors_load_strategy=lazy "
                f"(eager needs ~{required / 2**30:.1f} GiB free RAM, container has {available / 2**30:.1f} GiB)",
                flush=True,
            )
            return "lazy"
        return "eager"

    def _wait_for_vllm(self, session: requests.Session, process: subprocess.Popen | None) -> None:
        health_url = self._vllm_base_url().rsplit("/v1", 1)[0] + "/health"
        models_url = self._vllm_base_url() + "/models"
        version_url = self._vllm_base_url().rsplit("/v1", 1)[0] + "/version"
        deadline = time.monotonic() + float(
            os.getenv("H3_DIRECTOR_VLLM_STARTUP_TIMEOUT", "900")
        )

        last_error = ""
        health_ok = False
        model_ok = False
        version_ok = False
        while time.monotonic() < deadline:
            health_ok = False
            model_ok = False
            version_ok = False
            if process is not None and process.poll() is not None:
                log_path = Path(getattr(self, "_vllm_log_path", ""))
                log_tail = ""
                if log_path.is_file():
                    try:
                        lines = log_path.read_text(errors="replace").splitlines()
                        log_tail = "\n".join(lines[-120:])
                    except Exception:
                        log_tail = "<unable to read vLLM startup log>"
                raise RuntimeError(
                    "vLLM Director server exited during startup "
                    f"with code {process.returncode}. "
                    f"See {log_path or 'vLLM log'}.\n"
                    "Last vLLM log lines:\n"
                    f"{log_tail}"
                )
            try:
                response = session.get(health_url, timeout=5)
                health_ok = response.status_code == 200
            except Exception as exc:
                last_error = f"health {type(exc).__name__}: {exc}"
                health_ok = False
            try:
                response = session.get(models_url, timeout=5)
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    model_ok = bool(data) and self._vllm_model_name() in {
                        str(item.get("id", "")) for item in data
                    }
            except Exception as exc:
                last_error = f"models {type(exc).__name__}: {exc}"
                model_ok = False
            try:
                response = session.get(version_url, timeout=5)
                if response.status_code == 200:
                    version_data = response.json()
                    observed_version = str(version_data.get("version", "")).strip() if isinstance(version_data, dict) else ""
                    version_ok = observed_version == DIRECTOR_VLLM_VERSION
                    if not version_ok:
                        last_error = f"vLLM version mismatch: observed={observed_version!r}, expected={DIRECTOR_VLLM_VERSION!r}"
            except Exception as exc:
                last_error = f"version {type(exc).__name__}: {exc}"
                version_ok = False
            if health_ok and model_ok and version_ok:
                return
            time.sleep(0.5)

        raise RuntimeError(
            "Timed out waiting for the Qwen3-14B-AWQ vLLM server. "
            f"health_ok={health_ok}, model_ok={model_ok}, version_ok={version_ok}, last_error={last_error}. "
            f"Log: {getattr(self, '_vllm_log_path', 'unknown')}"
        )

    def load(
        self,
    ) -> None:
        global _SHARED_VLLM_PROCESS, _SHARED_VLLM_LOG_HANDLE, _SHARED_VLLM_LOG_PATH
        global _SHARED_VLLM_TOKENIZER

        if not self.available:
            return

        if self._vllm_session is not None:
            return

        with _SHARED_VLLM_LOCK:
            session = requests.Session()
            model_name = self._vllm_model_name()

            try:
                reuse_timeout = float(
                    os.getenv("H3_DIRECTOR_VLLM_REUSE_TIMEOUT", "8")
                )
            except (TypeError, ValueError):
                reuse_timeout = 8.0
            reuse_timeout = max(1.0, reuse_timeout)

            health_url = self._vllm_base_url().rsplit("/v1", 1)[0] + "/health"
            models_url = self._vllm_base_url() + "/models"
            version_url = self._vllm_base_url().rsplit("/v1", 1)[0] + "/version"
            keep_alive = os.getenv("H3_DIRECTOR_VLLM_KEEP_ALIVE", "0").strip().lower() in {
                "1", "true", "yes", "on"
            }
            external_server = os.getenv("H3_DIRECTOR_VLLM_EXTERNAL", "0").strip().lower() in {
                "1", "true", "yes", "on"
            }
            shared_process_alive = (
                _SHARED_VLLM_PROCESS is not None
                and _SHARED_VLLM_PROCESS.poll() is None
            )
            allow_port_reuse = shared_process_alive or keep_alive or external_server
            reuse_deadline = time.monotonic() + reuse_timeout
            last_error = ""
            # H3_AUTOFIX_FAST_PORT_PROBE_V1: skip reuse loop entirely if port closed
            _h3_port_open_initial = _h3_probe_host(DIRECTOR_VLLM_HOST, DIRECTOR_VLLM_PORT)
            # H3_AUTOFIX_UNCONDITIONAL_PREFLIGHT_V1
            # H3_V51: a server left from an aborted or finished run may still be loading (its port opens only at the end); wait for it instead of failing or starting a second one.
            _h3_waiting_stray = False
            if not external_server and _h3_find_vllm_pids():
                try:
                    _h3_healthy = session.get(health_url, timeout=1.5).status_code == 200
                except Exception:
                    _h3_healthy = False
                if not _h3_healthy:
                    _h3_waiting_stray = True
                    _h3_port_open_initial = True
                    try:
                        _h3_wait = float(os.getenv("H3_DIRECTOR_VLLM_STARTUP_TIMEOUT", "900"))
                    except (TypeError, ValueError):
                        _h3_wait = 900.0
                    reuse_deadline = time.monotonic() + _h3_wait
                    print("[QWEN] a vLLM Director server from an earlier run is still starting; waiting for it instead of restarting", flush=True)
            while _h3_port_open_initial and time.monotonic() < reuse_deadline:
                try:
                    health = session.get(health_url, timeout=1.5)
                    if health.status_code == 200:
                        models = session.get(models_url, timeout=1.5)
                        if models.status_code == 200:
                            data = models.json().get("data", [])
                            if data and model_name in {
                                str(item.get("id", "")) for item in data
                            }:
                                version_response = session.get(version_url, timeout=1.5)
                                version_data = version_response.json() if version_response.status_code == 200 else {}
                                observed_server_version = str(version_data.get("version", "")).strip() if isinstance(version_data, dict) else ""
                                if observed_server_version != DIRECTOR_VLLM_VERSION:
                                    last_error = (
                                        f"running vLLM version={observed_server_version!r}; "
                                        f"expected={DIRECTOR_VLLM_VERSION!r}"
                                    )
                                    if not external_server:
                                        _h3_kill_stray_vllm("stale vLLM version")
                                        _h3_port_open_initial = False
                                        break
                                    raise RuntimeError(last_error)
                                try:
                                    _h3_ml = int(next((item.get("max_model_len") or 0 for item in data if str(item.get("id", "")) == model_name), 0))
                                except Exception:
                                    _h3_ml = 0
                                if (not external_server) and _h3_ml and _h3_ml != int(DIRECTOR_VLLM_MAX_MODEL_LEN):
                                    print(f"[QWEN] running vLLM has context={_h3_ml} but {int(DIRECTOR_VLLM_MAX_MODEL_LEN)} is configured; restarting it", flush=True)
                                    _h3_kill_stray_vllm("stale context")
                                    _h3_waiting_stray = False
                                    break
                                self._vllm_session = session
                                self._vllm_process = _SHARED_VLLM_PROCESS
                                self._vllm_log_handle = None
                                self._vllm_log_path = _SHARED_VLLM_LOG_PATH
                                if _SHARED_VLLM_TOKENIZER is None:
                                    from transformers import AutoTokenizer
                                    _SHARED_VLLM_TOKENIZER = AutoTokenizer.from_pretrained(
                                        str(self._model_path),
                                        local_files_only=True,
                                        trust_remote_code=True,
                                        use_fast=True,
                                    )
                                self._tokenizer = _SHARED_VLLM_TOKENIZER
                                print(
                                    "[QWEN] vLLM Director reused",
                                    f"model={model_name}",
                                    f"pid={_SHARED_VLLM_PROCESS.pid if _SHARED_VLLM_PROCESS is not None else 'external'}",
                                    f"reason={'shared-process' if shared_process_alive else 'keep-alive/external'}",
                                    flush=True,
                                )
                                return
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                if _h3_waiting_stray and not _h3_find_vllm_pids():
                    print("[QWEN] the earlier vLLM server exited; starting a fresh one", flush=True)
                    _h3_waiting_stray = False
                    break
                time.sleep(0.5)

            if external_server:
                try:
                    self._wait_for_vllm(session, None)
                except Exception:
                    session.close()
                    raise
                self._vllm_session = session
                if _SHARED_VLLM_TOKENIZER is None:
                    from transformers import AutoTokenizer
                    _SHARED_VLLM_TOKENIZER = AutoTokenizer.from_pretrained(
                        str(self._model_path),
                        local_files_only=True,
                        trust_remote_code=True,
                        use_fast=True,
                    )
                self._tokenizer = _SHARED_VLLM_TOKENIZER
                print(
                    "[QWEN] vLLM Director attached to external server",
                    f"model={model_name}",
                    flush=True,
                )
                return

            if _h3_waiting_stray and _h3_find_vllm_pids():
                print("[QWEN] the earlier vLLM server never became ready; stopping it and starting fresh", flush=True)
                _h3_kill_stray_vllm("startup wait expired")
            # H3_AUTOFIX_PORT_GUARD_V1: refuse to spawn on occupied-but-unhealthy port
            if _h3_probe_host(DIRECTOR_VLLM_HOST, DIRECTOR_VLLM_PORT):
                raise RuntimeError(
                    f"[H3_AUTOFIX_PORT_GUARD_V1] Port "
                    f"{DIRECTOR_VLLM_HOST}:{DIRECTOR_VLLM_PORT} is already occupied "
                    f"by a server that is not a healthy {model_name!r} instance. "
                    f"last_reuse_error={last_error!r}. "
                    f"Stop the offending process, or set H3_DIRECTOR_VLLM_EXTERNAL=1 to attach externally."
                )

            host = DIRECTOR_VLLM_HOST
            port = DIRECTOR_VLLM_PORT
            log_path = Path(
                os.getenv(
                    "H3_DIRECTOR_VLLM_LOG",
                    str(self.project_root / "qwen3_vllm_server.log"),
                )
            )
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                session.close()
                raise

            stale_process = _SHARED_VLLM_PROCESS
            if stale_process is not None and stale_process.poll() is not None:
                _SHARED_VLLM_PROCESS = None
                stale_handle = _SHARED_VLLM_LOG_HANDLE
                _SHARED_VLLM_LOG_HANDLE = None
                if stale_handle is not None:
                    try:
                        stale_handle.close()
                    except Exception:
                        pass

            vllm_python = DIRECTOR_VLLM_ENV_DIR / "bin" / "python"
            if not vllm_python.is_file():
                session.close()
                raise RuntimeError(
                    "Qwen Director vLLM environment is missing: "
                    f"{vllm_python}. Run kaggle/bootstrap.py first."
                )

            try:
                # Stage both checkpoints concurrently (parallel ranged reads); falls back to the dataset path.
                _speculator_source = self._find_speculator_model()
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="h3-stage") as _stage_pool:
                    _qwen_future = _stage_pool.submit(
                        self._stage_to_local_scratch, self._model_path, "Qwen3-14B-AWQ"
                    )
                    _spec_future = _stage_pool.submit(
                        self._stage_to_local_scratch, _speculator_source, "Qwen3-14B EAGLE-3 speculator"
                    )
                    self._model_path = _qwen_future.result()
                    speculator_model = _spec_future.result()
            except Exception:
                session.close()
                raise
            if DIRECTOR_VLLM_SPECULATIVE_TOKENS <= 0:
                session.close()
                raise RuntimeError(
                    "Director speculative_tokens must be positive in runtime configuration."
                )

            eagle_enforce_eager_value = os.getenv(
                "H3_DIRECTOR_VLLM_EAGLE_ENFORCE_EAGER",
                "1",
            ).strip().lower()
            if eagle_enforce_eager_value not in {
                "0", "1", "false", "true", "no", "yes", "off", "on"
            }:
                session.close()
                raise RuntimeError(
                    "H3_DIRECTOR_VLLM_EAGLE_ENFORCE_EAGER must be a boolean environment value."
                )
            eagle_enforce_eager = eagle_enforce_eager_value in {
                "1", "true", "yes", "on"
            }

            speculative_config = json.dumps({
                "model": str(speculator_model),
                "method": DIRECTOR_VLLM_SPECULATIVE_METHOD,
                "num_speculative_tokens": DIRECTOR_VLLM_SPECULATIVE_TOKENS,
                "enforce_eager": eagle_enforce_eager,
            }, separators=(",", ":"))

            max_batched_tokens = os.getenv("H3_DIRECTOR_VLLM_MAX_NUM_BATCHED_TOKENS", "10240").strip()

            command = [
                str(vllm_python),
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                str(self._model_path),
                "--host",
                host,
                "--port",
                str(port),
                "--served-model-name",
                model_name,
                "--tensor-parallel-size",
                str(DIRECTOR_VLLM_TENSOR_PARALLEL_SIZE),
                "--max-model-len",
                str(DIRECTOR_VLLM_MAX_MODEL_LEN),
                "--gpu-memory-utilization",
                str(DIRECTOR_VLLM_GPU_MEMORY_UTILIZATION),
                "--max-num-seqs",
                str(DIRECTOR_VLLM_MAX_NUM_SEQS),
                "--max-num-batched-tokens",
                str(max_batched_tokens),
                "--seed",
                str(DIRECTOR_VLLM_SEED),
                "--generation-config",
                DIRECTOR_VLLM_GENERATION_CONFIG,
                "--reasoning-parser",
                "qwen3",
                "--reasoning-config",
                json.dumps({
                    "reasoning_start_str": "<think>",
                    "reasoning_end_str": (
                        "Considering the limited time by the user, I have to give "
                        "the solution based on the thinking directly now.</think>"
                    ),
                }, ensure_ascii=False, separators=(",", ":")),
                "--speculative-config",
                speculative_config,
                "--trust-remote-code",
                "--disable-custom-all-reduce",
            ]

            enforce_eager_base_val = "1" if DIRECTOR_VLLM_ENFORCE_EAGER else "0"
            if enforce_eager_base_val in {"1", "true", "yes", "on"}:
                command.append("--enforce-eager")
            elif not DIRECTOR_VLLM_JIT_WARMUP:
                command.extend(["--kernel-config", json.dumps({"enable_jit_warmup": False}, separators=(",", ":"))])

            safetensors_strategy = self._resolve_safetensors_strategy(self._model_path, speculator_model)
            self._effective_safetensors_strategy = safetensors_strategy
            command.extend([
                "--safetensors-load-strategy",
                safetensors_strategy,
            ])
            print(
                f"[QWEN] safetensors_load_strategy={safetensors_strategy} "
                f"vllm={DIRECTOR_VLLM_VERSION} enforce_eager={DIRECTOR_VLLM_ENFORCE_EAGER} jit_warmup={DIRECTOR_VLLM_JIT_WARMUP}",
                flush=True,
            )

            kv_cache_memory_bytes = os.getenv(
                "H3_DIRECTOR_VLLM_KV_CACHE_MEMORY_BYTES",
                "",
            ).strip()
            if kv_cache_memory_bytes:
                if not kv_cache_memory_bytes.isdigit() or int(kv_cache_memory_bytes) <= 0:
                    session.close()
                    raise RuntimeError(
                        "H3_DIRECTOR_VLLM_KV_CACHE_MEMORY_BYTES must be a positive integer byte count."
                    )
                command.extend([
                    "--kv-cache-memory-bytes",
                    kv_cache_memory_bytes,
                ])

            cudagraph_capture_sizes = os.getenv(
                "H3_DIRECTOR_VLLM_CUDAGRAPH_CAPTURE_SIZES",
                "",
            ).strip()
            if cudagraph_capture_sizes:
                try:
                    capture_sizes = [
                        int(value.strip())
                        for value in cudagraph_capture_sizes.split(",")
                        if value.strip()
                    ]
                except ValueError as exc:
                    session.close()
                    raise RuntimeError(
                        "H3_DIRECTOR_VLLM_CUDAGRAPH_CAPTURE_SIZES must be a comma-separated list of positive integers."
                    ) from exc
                if not capture_sizes or any(value <= 0 for value in capture_sizes):
                    session.close()
                    raise RuntimeError(
                        "H3_DIRECTOR_VLLM_CUDAGRAPH_CAPTURE_SIZES must contain positive integers."
                    )
                if len(set(capture_sizes)) != len(capture_sizes):
                    session.close()
                    raise RuntimeError(
                        "H3_DIRECTOR_VLLM_CUDAGRAPH_CAPTURE_SIZES must not contain duplicates."
                    )
                capture_sizes = sorted(capture_sizes)
                command.extend([
                    "--compilation-config",
                    json.dumps(
                        {"cudagraph_capture_sizes": capture_sizes},
                        separators=(",", ":"),
                    ),
                ])

            child_env = os.environ.copy()
            # Disable FlashInfer sampler probe on SM75 (T4) to silence warnings and use native paths cleanly
            child_env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
            # H3_AUTOFIX_FORCE_AOT_LOAD_OFF_V1
            child_env["VLLM_FORCE_AOT_LOAD"] = "0"

            # H3_PERSISTENT_VLLM_CACHE_ROOT_V1
            # Keep compiled artifacts inside the existing repository cache.
            # Full model checkpoints remain in Kaggle Input.
            configured_cache_root = os.getenv(
                "H3_DIRECTOR_VLLM_CACHE_ROOT", ""
            ).strip()

            cache_path = (
                Path(configured_cache_root).expanduser()
                if configured_cache_root
                else Path(self.project_root) / ".vllm_cache"
            ).resolve()

            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                session.close()
                raise

            child_env["VLLM_CACHE_ROOT"] = str(cache_path)

            startup_plan_value = os.getenv(
                "H3_DIRECTOR_VLLM_ENABLE_STARTUP_PLAN",
                "0",
            ).strip().lower()
            if startup_plan_value:
                if startup_plan_value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
                    session.close()
                    raise RuntimeError(
                        "H3_DIRECTOR_VLLM_ENABLE_STARTUP_PLAN must be a boolean environment value."
                    )
                child_env["VLLM_ENABLE_STARTUP_PLAN"] = (
                    "1"
                    if startup_plan_value in {"1", "true", "yes", "on"}
                    else "0"
                )

            warmup_sampler_jit_value = os.getenv(
                "H3_DIRECTOR_VLLM_WARMUP_SAMPLER_JIT",
                "1",
            ).strip().lower()
            if warmup_sampler_jit_value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
                session.close()
                raise RuntimeError(
                    "H3_DIRECTOR_VLLM_WARMUP_SAMPLER_JIT must be a boolean environment value."
                )
            warmup_sampler_jit = warmup_sampler_jit_value in {
                "1", "true", "yes", "on"
            }

            _startup_t0 = time.perf_counter()
            if _verbose(): print(
                "[QWEN] vLLM startup config",
                f"cache_root={cache_path or 'default(/root/.cache/vllm)'}",
                f"startup_plan={'1' if startup_plan_value in {'1', 'true', 'yes', 'on'} else '0'}",
                f"enforce_eager={enforce_eager_base_val in {'1', 'true', 'yes', 'on'}}",
                f"eagle_enforce_eager={eagle_enforce_eager}",
                f"max_batched_tokens={max_batched_tokens}",
                f"cudagraph_capture_sizes={capture_sizes if cudagraph_capture_sizes else 'auto'}",
                f"kv_cache_memory_bytes={kv_cache_memory_bytes or 'auto'}",
                f"warmup_sampler_jit={warmup_sampler_jit}",
                flush=True,
            )

            try:
                log_handle = log_path.open("ab")
            except Exception:
                session.close()
                raise

            process = None
            # Record the log path before starting the process.
            self._vllm_log_path = log_path
            self._vllm_process = None
            try:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                    start_new_session=True,
                )
                _SHARED_VLLM_PROCESS = process
                _SHARED_VLLM_LOG_HANDLE = log_handle
                _SHARED_VLLM_LOG_PATH = log_path

                self._vllm_process = process
                self._vllm_log_handle = log_handle
                self._vllm_log_path = log_path
                self._vllm_session = session

                self._wait_for_vllm(
                    session,
                    process,
                )

                # H3_AUTOFIX_TWO_MODE_WARMUP_V1: exercise both sampler paths so
                # _thinking_budget_kernel and _topp_sb_* JIT once at startup.
                # Warn, do not raise, on failure.
                if warmup_sampler_jit:
                    warmup_url = self._vllm_base_url() + "/chat/completions"
                    _h3_thinking = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": "Say OK."}],
                        "max_tokens": 5,
                        "thinking_token_budget": 1280,
                        "temperature": DIRECTOR_STORY_THINKING_TEMPERATURE,
                        "top_p": DIRECTOR_STORY_THINKING_TOP_P,
                        "top_k": DIRECTOR_STORY_THINKING_TOP_K,
                        "min_p": DIRECTOR_STORY_THINKING_MIN_P,
                        "chat_template_kwargs": {"enable_thinking": True},
                    }
                    try:
                        _r = session.post(warmup_url, json=_h3_thinking, timeout=60)
                        _r.raise_for_status()
                        print("[QWEN] sampler-JIT warmup completed mode=thinking", flush=True)
                    except Exception as _h3_exc:
                        print(f"[QWEN] sampler-JIT warmup warning mode=thinking: {_h3_exc}", flush=True)
                    _h3_creative = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": "Say OK."}],
                        "max_tokens": 5,
                        "temperature": DIRECTOR_STORY_THINKING_TEMPERATURE,
                        "top_p": DIRECTOR_STORY_THINKING_TOP_P,
                        "top_k": DIRECTOR_STORY_WRITER_TOP_K,
                        "min_p": DIRECTOR_STORY_WRITER_MIN_P,
                        "presence_penalty": DIRECTOR_STORY_PRESENCE_PENALTY,
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    try:
                        _r = session.post(warmup_url, json=_h3_creative, timeout=60)
                        _r.raise_for_status()
                        print("[QWEN] sampler-JIT warmup completed mode=creative", flush=True)
                    except Exception as _h3_exc:
                        print(f"[QWEN] sampler-JIT warmup warning mode=creative: {_h3_exc}", flush=True)

                if _SHARED_VLLM_TOKENIZER is None:
                    from transformers import AutoTokenizer
                    _SHARED_VLLM_TOKENIZER = AutoTokenizer.from_pretrained(
                        str(self._model_path),
                        local_files_only=True,
                        trust_remote_code=True,
                        use_fast=True,
                    )
                self._tokenizer = _SHARED_VLLM_TOKENIZER

                print(
                    "[QWEN] vLLM Director ready",
                    f"model={model_name}",
                    f"tp={DIRECTOR_VLLM_TENSOR_PARALLEL_SIZE}",
                    f"context={DIRECTOR_VLLM_MAX_MODEL_LEN}",
                    f"startup={time.perf_counter() - _startup_t0:.0f}s",
                    f"log={log_path}",
                    flush=True,
                )

            except Exception as exc:
                if _SHARED_VLLM_PROCESS is process:
                    _SHARED_VLLM_PROCESS = None
                    _SHARED_VLLM_LOG_HANDLE = None
                    _SHARED_VLLM_LOG_PATH = None
                if process is not None:
                    try:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=15)
                    except Exception:
                        try:
                            process.kill()
                            process.wait(timeout=5)
                        except Exception:
                            pass
                try:
                    log_handle.close()
                except Exception:
                    pass
                try:
                    session.close()
                except Exception:
                    pass
                self._vllm_process = None
                self._vllm_log_handle = None
                self._vllm_session = None
                raise RuntimeError(
                    "Failed to initialize Qwen3-14B-AWQ vLLM director. "
                    f"Model: {self._model_path}\nError: {exc}"
                ) from exc

    def _shutdown_vllm(self) -> None:
        global _SHARED_VLLM_PROCESS, _SHARED_VLLM_LOG_HANDLE
        with _SHARED_VLLM_LOCK:
            process = _SHARED_VLLM_PROCESS
            _SHARED_VLLM_PROCESS = None

            if process is not None:
                try:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                except Exception:
                    pass

            log_handle = _SHARED_VLLM_LOG_HANDLE
            _SHARED_VLLM_LOG_HANDLE = None
            if log_handle is not None:
                try:
                    log_handle.close()
                except Exception:
                    pass

    def unload(
        self,
    ) -> None:
        session = self._vllm_session
        self._vllm_session = None

        if session is not None:
            try:
                session.close()
            except Exception:
                pass

        tokenizer = getattr(self, "_tokenizer", None)
        self._tokenizer = None
        if tokenizer is not None:
            if tokenizer is not _SHARED_VLLM_TOKENIZER:
                del tokenizer

        self._vllm_process = _SHARED_VLLM_PROCESS
        self._vllm_log_handle = None
        self._vllm_log_path = _SHARED_VLLM_LOG_PATH

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                pass
        except Exception:
            pass

    def _count_tokens(
        self,
        text: str,
    ) -> int:
        if self._vllm_session is None:
            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "Qwen director tokenizer is not loaded."
            )

        return int(
            len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True,
                )
            )
        )

    def _available_output_tokens(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 512,
    ) -> tuple[int, int]:
        context = int(
            DIRECTOR_N_CTX
        )

        safety = 128

        prompt_tokens = (
            self._count_tokens(
                system_prompt
                + "\n\n"
                + user_prompt
            )
        )

        available = (
            context
            - prompt_tokens
            - safety
        )

        if available < minimum_completion:
            raise RuntimeError(
                "Qwen director prompt is too large "
                f"for the {context}-token context window.\n"
                f"Prompt tokens: {prompt_tokens}.\n"
                f"Available completion tokens: {available}."
            )

        return (
            prompt_tokens,
            min(
                int(
                    DIRECTOR_MAX_TOKENS
                ),
                available,
            ),
        )

    @staticmethod
    def _limit_text(
        text: str,
        max_chars: int,
    ) -> str:
        value = str(
            text or ""
        ).strip()

        if len(value) <= max_chars:
            return value

        return (
            value[
                :max_chars
            ]
            .rstrip()
            + "…"
        )

    @staticmethod
    def _extract_json(
        text: str,
    ) -> dict:
        value = QwenDirectorRuntimeMixin._strip_thinking(text)
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value).strip()

        try:
            result = json.loads(value)
            if isinstance(result, dict):
                return result
            if isinstance(result, list):
                return {"items": result}
            raise RuntimeError("Qwen director output must be a JSON object or array.")
        except json.JSONDecodeError:
            pass

        for start_index, char in enumerate(value):
            if char not in "{[":
                continue
            close_char = "}" if char == "{" else "]"
            depth = 0
            in_string = False
            escaped = False
            for index in range(start_index, len(value)):
                current = value[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif current == "\\":
                        escaped = True
                    elif current == '"':
                        in_string = False
                    continue
                if current == '"':
                    in_string = True
                    continue
                if current == char:
                    depth += 1
                elif current == close_char:
                    depth -= 1
                    if depth == 0:
                        candidate = value[start_index:index + 1]
                        try:
                            result = json.loads(candidate)
                        except json.JSONDecodeError:
                            break
                        if isinstance(result, dict):
                            return result
                        if isinstance(result, list):
                            return {"items": result}
                        break
        raise RuntimeError("Qwen director returned invalid JSON.")

    @_with_faulthandler_watchdog
    def _post_chat(
        self,
        *,
        messages: list[dict],
        temperature: float,
        top_p: float,
        max_tokens: int,
        response_format: dict | None = None,
        min_tokens: int = 0,
        seed: int | None = None,
        creative: bool = False,
        enable_thinking: bool = False,
        thinking_token_budget: int | None = None,
        n_samples: int = 1,
    ) -> dict:
        if self._vllm_session is None:
            raise RuntimeError("Qwen director model is not loaded.")

        requested_output_tokens = int(max_tokens)
        if requested_output_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {requested_output_tokens}.")
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Qwen director tokenizer is not loaded for context validation.")
        try:
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=bool(enable_thinking),
            )
            prompt_token_count = len(prompt_ids)
        except Exception as exc:
            raise RuntimeError(
                "Unable to count the exact Qwen chat-template prompt before sending "
                "the vLLM request; refusing to risk context-window overflow."
            ) from exc
        context_limit = int(DIRECTOR_VLLM_MAX_MODEL_LEN)
        context_safety_tokens = 128
        if prompt_token_count + requested_output_tokens + context_safety_tokens > context_limit:
            raise RuntimeError(
                "Qwen Director request exceeds the configured context reservation: "
                f"prompt={prompt_token_count}, output={requested_output_tokens}, "
                f"safety={context_safety_tokens}, context={context_limit}."
            )

        payload = {
            "model": self._vllm_model_name(),
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "seed": int(DIRECTOR_VLLM_SEED if seed is None else seed),
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
        if enable_thinking:
            # Qwen3 documented thinking-mode sampler; keep these extras explicit.
            payload["top_k"] = int(DIRECTOR_STORY_THINKING_TOP_K)
            payload["min_p"] = float(DIRECTOR_STORY_THINKING_MIN_P)
        elif creative:
            # Separate Qwen3 non-thinking writing sampler for clean prose output.
            payload["top_k"] = int(DIRECTOR_STORY_WRITER_TOP_K)
            payload["min_p"] = float(DIRECTOR_STORY_WRITER_MIN_P)
        if creative:
            # Qwen recommends 1.5 for quantized checkpoints but notes a language-mixing
            # tradeoff at higher values. Keep the model-card default configurable and
            # inspect real output traces before tuning it; architect calls stay unpenalized.
            payload["presence_penalty"] = float(DIRECTOR_STORY_PRESENCE_PENALTY)
        if int(min_tokens or 0) > 0:
            payload["min_tokens"] = int(min_tokens)
        if thinking_token_budget is not None:
            payload["thinking_token_budget"] = int(thinking_token_budget)
        if response_format is not None:
            payload["response_format"] = response_format
        if int(n_samples or 1) > 1:
            payload["n"] = int(n_samples)

        url = self._vllm_base_url() + "/chat/completions"
        timeout = float(os.getenv("H3_DIRECTOR_VLLM_REQUEST_TIMEOUT", "1800"))
        response = self._vllm_session.post(url, json=payload, timeout=timeout)
        # Deliberately do not retry a rejected completion with weakened sampling.
        # The pinned vLLM 0.32 API supports top_k, min_p and presence_penalty;
        # a compatibility retry would silently change the author's sampling
        # contract and make a supposedly single-pass story issue two requests.
        if response.status_code >= 400:
            body = response.text[:4000]
            raise RuntimeError(
                f"vLLM request failed with HTTP {response.status_code}: {body}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                "vLLM returned a non-JSON response."
            ) from exc

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 512,
        temperature: float | None = None,
        top_p: float | None = None,
        call_name: str = "unknown",
        max_completion: int | None = None,
        json_mode: bool = True,
        disable_thinking: bool = True,
        response_schema: dict | None = None,
    ) -> dict:
        if self._vllm_session is None:
            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        if temperature is None:
            temperature = DIRECTOR_TEMPERATURE

        if top_p is None:
            top_p = DIRECTOR_TOP_P

        if disable_thinking:
            user_prompt = user_prompt.rstrip() + NO_THINK_SUFFIX

        _, max_tokens = self._available_output_tokens(
            system_prompt,
            user_prompt,
            minimum_completion=minimum_completion,
        )

        if max_completion is not None:
            max_tokens = min(
                max_tokens,
                int(max_completion),
            )

        if max_tokens <= 0:
            raise RuntimeError(
                f"No completion budget remains for {call_name}."
            )

        cache_key = None
        if self._cache_dir is not None:
            cache_key = self._cache_key(
                call_name,
                system_prompt,
                user_prompt,
                response_schema,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                json_mode=json_mode,
                disable_thinking=disable_thinking,
            )
            cached = self._cache_read(cache_key)
            if cached is not None:
                self._record_qwen_call(
                    call_name=call_name,
                    elapsed=0.0,
                    max_tokens=0,
                    temperature=temperature,
                    top_p=top_p,
                    response_format=(
                        {"type": "json_object", "schema": response_schema}
                        if json_mode and response_schema is not None
                        else None
                    ),
                    cache_hit=True,
                )
                return cached

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if json_mode:
            if response_schema is None:
                raise RuntimeError(
                    f"No JSON schema supplied for {call_name}."
                )
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": re.sub(r"[^a-zA-Z0-9_-]+", "_", call_name)[:64] or "director_json",
                    "schema": response_schema,
                },
            }

        started = time.perf_counter()
        response = None
        error_text = ""

        print(f"[QWEN] START {call_name}", flush=True)

        try:
            response = self._post_chat(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                response_format=kwargs.get("response_format"),
                enable_thinking=not disable_thinking,
            )
        except Exception as exc:
            error_text = (
                f"{type(exc).__name__}: {exc}"
            )
            raise
        finally:
            elapsed = (
                time.perf_counter()
                - started
            )

            usage = (
                response.get("usage", {})
                if isinstance(response, dict)
                else {}
            )

            prompt_tokens = int(
                usage.get("prompt_tokens", 0) or 0
            )
            completion_tokens = int(
                usage.get("completion_tokens", 0) or 0
            )
            finish_reason = ""
            if isinstance(response, dict):
                try:
                    finish_reason = str(
                        response["choices"][0].get("finish_reason", "")
                        or ""
                    )
                except (KeyError, IndexError, TypeError, AttributeError):
                    finish_reason = ""

            self._record_qwen_call(
                call_name=call_name,
                elapsed=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=kwargs.get("response_format"),
                finish_reason=finish_reason,
                error=error_text,
            )

            self._trace_call(
                call_name,
                system_prompt,
                user_prompt,
                response,
                elapsed,
                error_text,
                kwargs.get("response_format"),
            )

        try:
            content = str(
                response["choices"][0]["message"]["content"] or ""
            )
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "Qwen director returned an unexpected completion structure."
            ) from exc

        # Never pass a clipped structured response to JSON recovery or planner
        # reconciliation: doing so can turn truncation into a misleading empty roster.
        if finish_reason == "length":
            raise QwenJsonTruncated(
                f"Qwen structured output hit the completion limit for {call_name} "
                "before producing a complete JSON response.",
                partial=content,
            )

        if not content.strip():
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        if disable_thinking and re.search(
            r"<think>",
            content,
            flags=re.IGNORECASE,
        ):
            content = self._strip_thinking(content)

        parsed = self._extract_json(content)

        if not parsed:
            raise RuntimeError(
                f"Qwen returned an empty JSON object for {call_name}."
            )

        if cache_key is not None:
            self._cache_write(
                cache_key,
                parsed,
                call_name=call_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                elapsed=elapsed,
                raw_content=content,
            )

        return parsed

    def _chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        minimum_completion: int = 256,
        temperature: float | None = None,
        top_p: float | None = None,
        call_name: str = "unknown",
        max_completion: int | None = None,
        disable_thinking: bool = True,
        minimum_output_tokens: int = 0,
        seed: int | None = None,
        creative: bool = False,
        thinking_token_budget: int | None = None,
    ) -> str:
        if self._vllm_session is None:
            raise RuntimeError(
                "Qwen director model is not loaded."
            )

        if temperature is None:
            temperature = DIRECTOR_TEMPERATURE

        if top_p is None:
            top_p = DIRECTOR_TOP_P

        if disable_thinking:
            user_prompt = user_prompt.rstrip() + NO_THINK_SUFFIX

        _, max_tokens = (
            self._available_output_tokens(
                system_prompt,
                user_prompt,
                minimum_completion=minimum_completion,
            )
        )

        if max_completion is not None:
            max_tokens = min(
                max_tokens,
                int(max_completion),
            )

        minimum_output_tokens = max(0, int(minimum_output_tokens or 0))
        if minimum_output_tokens > max_tokens:
            raise RuntimeError(
                f"Minimum output token floor {minimum_output_tokens} exceeds the available "
                f"completion budget {max_tokens} for {call_name}."
            )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        started = time.perf_counter()
        response = None
        error_text = ""

        # One Author completion only. No n=3 candidate fan-out, ranking, or revision pass.
        _h3_n = 1

        print(f"[QWEN] START {call_name}", flush=True)

        try:
            response = self._post_chat(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                min_tokens=minimum_output_tokens,
                seed=seed,
                creative=creative,
                enable_thinking=not disable_thinking,
                thinking_token_budget=thinking_token_budget,
                n_samples=_h3_n,
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = (
                time.perf_counter()
                - started
            )

            usage = (
                response.get(
                    "usage",
                    {},
                )
                if isinstance(
                    response,
                    dict,
                )
                else {}
            )

            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
            completion_details = usage.get("completion_tokens_details", {}) or {}
            reasoning_tokens = int(
                completion_details.get("reasoning_tokens", 0)
                or completion_details.get("reasoning", 0)
                or usage.get("reasoning_tokens", 0)
                or 0
            )
            finish_reason = ""
            if isinstance(response, dict):
                try:
                    finish_reason = str(
                        response["choices"][0].get("finish_reason", "")
                        or ""
                    )
                except (KeyError, IndexError, TypeError, AttributeError):
                    finish_reason = ""
            self._record_qwen_call(
                call_name=call_name,
                elapsed=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                max_tokens=max_tokens,
                min_tokens=minimum_output_tokens,
                temperature=temperature,
                top_p=top_p,
                response_format=None,
                finish_reason=finish_reason,
                thinking_enabled=not disable_thinking,
                thinking_token_budget=thinking_token_budget,
                reasoning_tokens=reasoning_tokens,
                error=error_text,
            )

            self._trace_call(
                call_name,
                system_prompt,
                user_prompt,
                response,
                elapsed,
                error_text,
                None,
            )

        try:
            content = (
                response[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ]
            )
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            raise RuntimeError(
                "Qwen director returned an "
                "unexpected completion structure."
            ) from exc

        if finish_reason == "length":
            raise StoryTruncated(
                f"Qwen text generation hit the completion limit for {call_name} before producing a complete response.",
                partial=str(content or ""),
            )

        content = str(
            content or ""
        ).strip()

        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if re.search(r"<think>", content, flags=re.IGNORECASE):
            content = re.split(r"<think>", content, maxsplit=1, flags=re.IGNORECASE)[0].strip()

        if not content:
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        return content


def prestage_director_checkpoints() -> None:
    """Copy both Director checkpoints off the network-backed /kaggle/input mount into local scratch.

    kaggle/bootstrap.py runs this in the background while the install phases are busy, so the first
    Director start finds the checkpoints already staged and skips the copy. Staging is the same
    function the runtime uses (identical target and validation), so a missing or failed pre-stage
    only means the runtime stages at startup as before.
    """
    try:
        mixin = QwenDirectorRuntimeMixin
        qwen_path = mixin._find_model(None)
        speculator_path = mixin._find_speculator_model(None)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="h3-prestage") as pool:
            futures = [
                pool.submit(mixin._stage_to_local_scratch, qwen_path, "Qwen3-14B-AWQ"),
                pool.submit(mixin._stage_to_local_scratch, speculator_path, "Qwen3-14B EAGLE-3 speculator"),
            ]
            for future in futures:
                future.result()
        # Flush the freshly written pages now so the Director's weight loading never competes with writeback.
        os.sync()
    except Exception as exc:
        print(f"[QWEN] checkpoint pre-staging skipped: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)


def prewarm_director_server(project_root: str) -> None:
    """Start the Director vLLM server and leave it running (called from a detached bootstrap subprocess).

    This is the planner's own load() (identical command, version and context checks); with
    H3_DIRECTOR_VLLM_KEEP_ALIVE=1 the server outlives this process, and the planner later finds it
    healthy and reuses it, or waits for it if it is still starting. If this fails nothing is lost: the
    planner starts the server itself as before.
    """
    try:
        from planner.qwen_director import QwenDirector

        director = QwenDirector(Path(project_root))
        if not director.available:
            return
        director.load()
    except BaseException as exc:
        print(f"[QWEN] early Director start skipped: {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
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
    DIRECTOR_VLLM_ALLOW_SPECULATIVE_MODEL_OVERRIDE,
    director_enabled,
)

NO_THINK_SUFFIX = "\n/no_think"


def _verbose() -> bool:
    return os.getenv("H3_DIRECTOR_VERBOSE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _stage_mode() -> str:
    value = os.getenv("H3_DIRECTOR_STAGE_LOCAL", "auto").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return "on"
    if value in {"0", "false", "no", "off"}:
        return "off"
    if value in {"auto", "default", ""}:
        return "auto"
    raise RuntimeError(
        "H3_DIRECTOR_STAGE_LOCAL must be one of: auto, 0, 1."
    )


def _stage_enabled() -> bool:
    return _stage_mode() == "on"


def _filesystem_type(path: Path) -> str:
    try:
        result = subprocess.run(
            ["stat", "-f", "-c", "%T", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return result.stdout.strip().lower()
    except Exception:
        return ""


def _should_stage_local(source_path: Path) -> bool:
    mode = _stage_mode()
    if mode == "off":
        return False

    try:
        resolved = source_path.resolve()
    except OSError:
        resolved = source_path

    try:
        under_kaggle_input = (
            Path("/kaggle/input").resolve() in resolved.parents
            or str(resolved).startswith("/kaggle/input/")
            or str(resolved) == "/kaggle/input"
        )
    except OSError:
        under_kaggle_input = "/kaggle/input/" in str(resolved)

    if not under_kaggle_input:
        return False
    if mode == "on":
        return True

    fs_type = _filesystem_type(resolved)
    return fs_type.startswith(("nfs", "lustre", "s3fs", "gcsfuse", "fuse"))


class StoryTruncated(RuntimeError):
    """Raised when a text pass stops on the token cap; carries the partial text."""

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

def _parallel_copy_file(src, dst, workers: int = 8, chunk_bytes: int = 64 * 1024 * 1024, progress=None) -> None:
    """Copy one large file with several concurrent ranged reads.

    A single sequential reader is latency-bound on network-mounted datasets; disjoint
    byte ranges read in parallel (pread/pwrite, thread-safe) keep the link busy.
    The destination is preallocated and its final size is verified.
    """
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
        """Stage network-mounted Kaggle models to transient local scratch when policy selects staging."""
        # AUTO stages only when the dataset path is on a recognized network filesystem.
        # Set H3_DIRECTOR_STAGE_LOCAL=0 to force direct dataset loading, or =1 to force
        # local scratch staging. Scratch is /kaggle/tmp or /tmp, never /kaggle/working.
        if not _should_stage_local(source_path):
            return source_path

        resolved = source_path.resolve()
        is_kaggle_input = False
        try:
            is_kaggle_input = (
                Path("/kaggle/input").resolve() in resolved.parents
                or str(resolved).startswith("/kaggle/input")
            )
        except Exception:
            is_kaggle_input = "/kaggle/input" in str(resolved)

        if not is_kaggle_input:
            return source_path

        scratch = Path("/kaggle/tmp") if Path("/kaggle/tmp").is_dir() else Path("/tmp")
        target_dir = scratch / "staged_models" / resolved.name
        manifest_name = ".h3_stage_manifest.json"

        def checkpoint_fingerprint(directory: Path) -> dict:
            """Build a cheap content/layout identity without hashing multi-GB weight files."""
            config_path = directory / "config.json"
            index_path = directory / "model.safetensors.index.json"

            def sha256_small_file(path: Path) -> str:
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while True:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                return digest.hexdigest()

            fingerprint: dict = {
                "schema": 1,
                "config_sha256": sha256_small_file(config_path) if config_path.is_file() else "",
                "index_sha256": sha256_small_file(index_path) if index_path.is_file() else "",
                "files": {},
            }

            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map", {})
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError(f"Invalid checkpoint weight_map: {index_path}")
                names = sorted({str(value) for value in weight_map.values()})
            else:
                names = sorted(
                    str(path.relative_to(directory))
                    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")
                    for path in directory.glob(pattern)
                    if path.is_file()
                )

            for name in names:
                path = directory / name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing checkpoint weight file: {path}")
                fingerprint["files"][name] = path.stat().st_size

            tokenizer_path = directory / "tokenizer.json"
            if tokenizer_path.is_file():
                fingerprint["tokenizer_size"] = tokenizer_path.stat().st_size
            return fingerprint

        def is_valid_checkpoint(directory: Path, expected_fingerprint: dict | None = None) -> bool:
            """Fail closed on structure and, when present, the staged identity manifest."""
            config_path = directory / "config.json"
            if not config_path.is_file():
                return False

            index_file = directory / "model.safetensors.index.json"
            try:
                if index_file.is_file():
                    index = json.loads(index_file.read_text(encoding="utf-8"))
                    weight_map = index.get("weight_map", {})
                    if not isinstance(weight_map, dict) or not weight_map:
                        return False
                    if not all(
                        (directory / str(filename)).is_file()
                        for filename in set(weight_map.values())
                    ):
                        return False
                elif not any(
                    any(directory.glob(pattern))
                    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth")
                ):
                    return False
            except Exception:
                return False

            # Preserve tokenizer completeness for models that provide tokenizer.json.
            source_has_tokenizer = (resolved / "tokenizer.json").is_file()
            if source_has_tokenizer and not (directory / "tokenizer.json").is_file():
                return False

            if expected_fingerprint is not None:
                manifest_path = directory / manifest_name
                if not manifest_path.is_file():
                    return False
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    return False
                if manifest != expected_fingerprint:
                    return False

            return True

        try:
            source_fingerprint = checkpoint_fingerprint(resolved)
        except Exception as exc:
            print(
                f"[QWEN] Failed to fingerprint {label}; using direct mount. Error: {exc}",
                flush=True,
            )
            return source_path

        if target_dir.is_dir() and is_valid_checkpoint(target_dir, source_fingerprint):
            if _verbose():
                print(f"[QWEN] Reusing locally staged {label} at {target_dir}", flush=True)
            return target_dir

        reservation_bytes = 0
        try:
            stat = os.statvfs(scratch)
            avail_bytes = stat.f_bavail * stat.f_frsize
            source_size = sum(
                path.stat().st_size
                for path in resolved.rglob("*")
                if path.is_file()
            )
            reservation_bytes = source_size + 2 * (1024**3)
            global _STAGE_RESERVED_BYTES
            with _STAGE_SPACE_LOCK:
                effective_free = avail_bytes - _STAGE_RESERVED_BYTES
                if effective_free < reservation_bytes:
                    print(
                        f"[QWEN] Insufficient scratch space to stage {label} "
                        f"({effective_free / 1e9:.1f}GB available, {source_size / 1e9:.1f}GB needed). "
                        "Using direct mount.",
                        flush=True,
                    )
                    return source_path
                _STAGE_RESERVED_BYTES += reservation_bytes
        except Exception:
            print(
                f"[QWEN] Could not verify scratch capacity for {label}; using direct mount.",
                flush=True,
            )
            return source_path

        if _verbose():
            print(
                f"[QWEN] Staging {label} to local scratch ({target_dir})...",
                flush=True,
            )
        start_time = time.perf_counter()
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target_dir.with_name(f"{target_dir.name}.tmp_{os.getpid()}")

        try:
            if tmp_target.exists():
                shutil.rmtree(tmp_target, ignore_errors=True)

            copied_bytes = [0]
            last_print_time = [time.perf_counter()]

            def tracked_copy(src, dst, *, follow_symlinks=True):
                """Chunked copy function to track and report staging progress."""
                try:
                    stage_workers = int(os.getenv("H3_DIRECTOR_STAGE_WORKERS", "8"))
                except ValueError:
                    stage_workers = 8
                if stage_workers > 1 and os.path.getsize(src) >= 256 * 1024 * 1024:
                    def _note(n):
                        copied_bytes[0] += n
                    try:
                        _parallel_copy_file(src, dst, workers=stage_workers, progress=_note)
                        shutil.copystat(src, dst, follow_symlinks=follow_symlinks)
                        return dst
                    except Exception as exc:  # fall back to the proven sequential path
                        if _verbose():
                            print(f"[QWEN] parallel copy failed ({exc}); retrying sequentially", flush=True)
                length = 16 * 1024 * 1024
                with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                    while True:
                        buf = fsrc.read(length)
                        if not buf:
                            break
                        fdst.write(buf)
                        copied_bytes[0] += len(buf)
                        now = time.perf_counter()
                        if _verbose() and now - last_print_time[0] > 5.0:
                            print(
                                f"[QWEN] Staging {label}: "
                                f"{copied_bytes[0] / 1e9:.1f}GB / {source_size / 1e9:.1f}GB...",
                                flush=True,
                            )
                            last_print_time[0] = now
                shutil.copystat(src, dst, follow_symlinks=follow_symlinks)
                return dst

            shutil.copytree(resolved, tmp_target, copy_function=tracked_copy)
            (tmp_target / manifest_name).write_text(
                json.dumps(source_fingerprint, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            for _name, _size in source_fingerprint.get("files", {}).items():
                _staged = tmp_target / _name
                if not _staged.is_file() or _staged.stat().st_size != _size:
                    raise RuntimeError(f"staged file size mismatch: {_name}")

            if not is_valid_checkpoint(tmp_target, source_fingerprint):
                raise RuntimeError(
                    f"staged {label} checkpoint failed post-copy validation"
                )

            if target_dir.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
            tmp_target.rename(target_dir)
        except Exception as exc:
            if tmp_target.exists():
                shutil.rmtree(tmp_target, ignore_errors=True)
            print(
                f"[QWEN] Failed to stage {label}, falling back to direct mount. Error: {exc}",
                flush=True,
            )
            return source_path
        finally:
            if reservation_bytes:
                with _STAGE_SPACE_LOCK:
                    _STAGE_RESERVED_BYTES = max(0, _STAGE_RESERVED_BYTES - reservation_bytes)

        elapsed = time.perf_counter() - start_time
        if _verbose():
            print(f"[QWEN] Staged {label} in {elapsed:.2f}s", flush=True)
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

    def _wait_for_vllm(self, session: requests.Session, process: subprocess.Popen | None) -> None:
        health_url = self._vllm_base_url().rsplit("/v1", 1)[0] + "/health"
        models_url = self._vllm_base_url() + "/models"
        deadline = time.monotonic() + float(
            os.getenv("H3_DIRECTOR_VLLM_STARTUP_TIMEOUT", "900")
        )

        last_error = ""
        health_ok = False
        model_ok = False
        while time.monotonic() < deadline:
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
            if health_ok and model_ok:
                return
            time.sleep(2.0)

        raise RuntimeError(
            "Timed out waiting for the Qwen3-14B-AWQ vLLM server. "
            f"health_ok={health_ok}, model_ok={model_ok}, last_error={last_error}. "
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
            while allow_port_reuse and time.monotonic() < reuse_deadline:
                try:
                    health = session.get(health_url, timeout=1.5)
                    if health.status_code == 200:
                        models = session.get(models_url, timeout=1.5)
                        if models.status_code == 200:
                            data = models.json().get("data", [])
                            if data and model_name in {
                                str(item.get("id", "")) for item in data
                            }:
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
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="h3-model-prep") as executor:
                    model_future = executor.submit(
                        self._stage_to_local_scratch,
                        self._model_path,
                        "Qwen3-14B-AWQ",
                    )
                    speculator_source_future = executor.submit(self._find_speculator_model)
                    speculator_source = speculator_source_future.result()
                    speculator_future = executor.submit(
                        self._stage_to_local_scratch,
                        speculator_source,
                        "Qwen3-14B EAGLE-3 speculator",
                    )
                    self._model_path = model_future.result()
                    speculator_model = speculator_future.result()
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

            max_batched_tokens = os.getenv("H3_DIRECTOR_VLLM_MAX_NUM_BATCHED_TOKENS", "4096").strip()

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
                "--speculative-config",
                speculative_config,
                "--trust-remote-code",
                "--disable-custom-all-reduce",
            ]

            enforce_eager_base_val = os.getenv("H3_DIRECTOR_VLLM_ENFORCE_EAGER", "0").strip().lower()
            if enforce_eager_base_val in {"1", "true", "yes", "on"}:
                command.append("--enforce-eager")

            safetensors_strategy = os.getenv(
                "H3_DIRECTOR_VLLM_SAFETENSORS_LOAD_STRATEGY",
                "",
            ).strip().lower()
            if safetensors_strategy in {"eager", "lazy", "prefetch", "torchao"}:
                command.extend([
                    "--safetensors-load-strategy",
                    safetensors_strategy,
                ])
            elif safetensors_strategy not in {"", "none", "default"}:
                session.close()
                raise RuntimeError(
                    "Unsupported H3_DIRECTOR_VLLM_SAFETENSORS_LOAD_STRATEGY: "
                    f"{safetensors_strategy!r}"
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

            configured_cache_root = os.getenv("H3_DIRECTOR_VLLM_CACHE_ROOT", "").strip()
            cache_root = configured_cache_root or ""

            cache_path = None
            if cache_root:
                cache_path = Path(cache_root).expanduser()
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

                if warmup_sampler_jit:
                    warmup_url = self._vllm_base_url() + "/chat/completions"
                    warmup_payload = {
                        "model": model_name,
                        "messages": [{"role": "user", "content": "Say OK."}],
                        "max_tokens": 2,
                        "temperature": 0.6,
                        "top_p": 0.95,
                        # Match the story request's sampler/reasoning path so the first real
                        # thinking-enabled story call does not discover a different JIT path.
                        "top_k": 20,
                        "presence_penalty": float(
                            os.getenv("H3_DIRECTOR_STORY_PRESENCE_PENALTY", "1.5")
                        ),
                        "chat_template_kwargs": {"enable_thinking": True},
                    }
                    try:
                        warmup_response = session.post(
                            warmup_url,
                            json=warmup_payload,
                            timeout=60,
                        )
                        warmup_response.raise_for_status()
                        if _verbose():
                            print("[QWEN] vLLM sampler-JIT warmup completed", flush=True)
                    except Exception as exc:
                        raise RuntimeError(
                            "vLLM sampler-JIT warmup failed. Disable "
                            "H3_DIRECTOR_VLLM_WARMUP_SAMPLER_JIT to skip it."
                        ) from exc

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
    ) -> dict:
        if self._vllm_session is None:
            raise RuntimeError("Qwen director model is not loaded.")

        payload = {
            "model": self._vllm_model_name(),
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "seed": int(DIRECTOR_VLLM_SEED if seed is None else seed),
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
        if creative:
            # Story-only anti-runaway sampling. Do not alter the structured JSON passes
            # with a story-specific top-k or presence penalty.
            payload["top_k"] = 20
            payload["presence_penalty"] = float(
                os.getenv("H3_DIRECTOR_STORY_PRESENCE_PENALTY", "1.5")
            )
        if int(min_tokens or 0) > 0:
            payload["min_tokens"] = int(min_tokens)
        if thinking_token_budget is not None:
            payload["thinking_token_budget"] = int(thinking_token_budget)
        if response_format is not None:
            payload["response_format"] = response_format

        url = self._vllm_base_url() + "/chat/completions"
        timeout = float(os.getenv("H3_DIRECTOR_VLLM_REQUEST_TIMEOUT", "1800"))
        response = self._vllm_session.post(url, json=payload, timeout=timeout)
        if response.status_code == 400 and any(
            key in response.text for key in ("presence_penalty", "top_k")
        ):
            # Sampling extras are optional. Never drop chat_template_kwargs or the reasoning
            # budget on fallback: changing the thinking mode would silently change story quality.
            for key in ("presence_penalty", "top_k"):
                payload.pop(key, None)
            response = self._vllm_session.post(url, json=payload, timeout=timeout)

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

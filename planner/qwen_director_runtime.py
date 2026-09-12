from __future__ import annotations

import faulthandler
import sys
import gc
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
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
    director_enabled,
)


NO_THINK_SUFFIX = "\n/no_think"


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

    def _record_qwen_call(
        self,
        *,
        call_name: str,
        elapsed: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        max_tokens: int = 0,
        temperature: float | None = None,
        top_p: float | None = None,
        response_format=None,
        finish_reason: str = "",
        cache_hit: bool = False,
        error: str = "",
    ) -> None:
        """Record and print bounded runtime telemetry for one Qwen call."""
        prompt_tokens = max(0, int(prompt_tokens or 0))
        completion_tokens = max(0, int(completion_tokens or 0))
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
            "temperature": temperature,
            "top_p": top_p,
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
            f"total_tokens={prompt_tokens + completion_tokens}",
            f"decode_tps={record['decode_tps']:.2f}",
            f"max_tokens={int(max_tokens or 0)}",
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

        payload = {
            "call_name": call_name,
            "elapsed_seconds": elapsed,
            "error": error,
            "response_format": response_format,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_content": raw_content,
            "reasoning_content": (
                str(
                    response["choices"][0]["message"].get(
                        "reasoning_content"
                    )
                    or ""
                )
                if isinstance(response, dict)
                else ""
            ),
            "raw_response": response,
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
    ) -> str:
        material = json.dumps(
            {
                "namespace": self._cache_namespace,
                "call_name": call_name,
                "system": system_prompt,
                "user": user_prompt,
                "schema": response_schema,
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
        candidates.append(
            DIRECTOR_KAGGLE_INPUT_ROOT / Path(DIRECTOR_MODEL_PATH).name
        )

        # Backwards-compatible search for the exact dataset directory name.
        for root in (
            self.project_root,
            DIRECTOR_KAGGLE_INPUT_ROOT,
        ):
            if not root.exists():
                continue
            try:
                candidates.extend(
                    path
                    for path in root.rglob(Path(DIRECTOR_MODEL_PATH).name)
                    if path.is_dir()
                )
            except OSError:
                continue

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

    def _wait_for_vllm(self, session: requests.Session, process: subprocess.Popen) -> None:
        health_url = self._vllm_base_url().rsplit("/v1", 1)[0] + "/health"
        models_url = self._vllm_base_url() + "/models"
        deadline = time.monotonic() + float(
            os.getenv("H3_DIRECTOR_VLLM_STARTUP_TIMEOUT", "600")
        )

        last_error = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    "vLLM Director server exited during startup "
                    f"with code {process.returncode}. "
                    f"See {getattr(self, '_vllm_log_path', 'vLLM log')}."
                )
            for url in (health_url, models_url):
                try:
                    response = session.get(url, timeout=5)
                    if response.status_code == 200:
                        if url.endswith("/models"):
                            payload = response.json()
                            data = payload.get("data", [])
                            if data and self._vllm_model_name() in {
                                str(item.get("id", "")) for item in data
                            }:
                                return
                        else:
                            return
                except Exception as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0)

        raise RuntimeError(
            "Timed out waiting for the Qwen3-14B-AWQ vLLM server. "
            f"Last error: {last_error}. "
            f"Log: {getattr(self, '_vllm_log_path', 'unknown')}"
        )

    def load(
        self,
    ) -> None:

        if not self.available:
            return

        if self._llama is not None:
            return

        if os.getenv("H3_DIRECTOR_VLLM_EXTERNAL", "").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            session = requests.Session()
            try:
                self._wait_for_vllm(
                    session,
                    subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(10**9)"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    ),
                )
            except Exception:
                session.close()
                raise
            self._llama = session
        else:
            session = requests.Session()
            host = DIRECTOR_VLLM_HOST
            port = DIRECTOR_VLLM_PORT
            model_name = self._vllm_model_name()
            log_path = Path(
                os.getenv(
                    "H3_DIRECTOR_VLLM_LOG",
                    str(self.project_root / "qwen3_vllm_server.log"),
                )
            )
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab")

            vllm_bin = shutil.which("vllm")
            if vllm_bin:
                command = [vllm_bin, "serve"]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "vllm.entrypoints.openai.api_server",
                ]

            command.extend(
                [
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
                    "1",
                    "--dtype",
                    "half",
                    "--trust-remote-code",
                    "--disable-log-requests",
                ]
            )

            try:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                self._vllm_process = process
                self._vllm_log_handle = log_handle
                self._vllm_log_path = log_path
                self._vllm_session = session

                self._wait_for_vllm(
                    session,
                    process,
                )

                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        str(self._model_path),
                        local_files_only=True,
                        trust_remote_code=True,
                        use_fast=True,
                    )
                except Exception as exc:
                    self._shutdown_vllm()
                    raise RuntimeError(
                        "Qwen3-14B-AWQ tokenizer initialization failed."
                    ) from exc

                self._llama = session

                print(
                    "[QWEN] vLLM Director ready",
                    f"model={model_name}",
                    f"tp={DIRECTOR_VLLM_TENSOR_PARALLEL_SIZE}",
                    f"context={DIRECTOR_VLLM_MAX_MODEL_LEN}",
                    f"log={log_path}",
                    flush=True,
                )

            except Exception as exc:
                try:
                    log_handle.close()
                except Exception:
                    pass
                if self._vllm_process is not None:
                    try:
                        self._shutdown_vllm()
                    except Exception:
                        pass
                raise RuntimeError(
                    "Failed to initialize Qwen3-14B-AWQ vLLM director. "
                    f"Model: {self._model_path}\nError: {exc}"
                ) from exc

    def _shutdown_vllm(self) -> None:
        process = getattr(self, "_vllm_process", None)
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
        self._vllm_process = None

        log_handle = getattr(self, "_vllm_log_handle", None)
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
        self._vllm_log_handle = None

    def unload(
        self,
    ) -> None:

        session = self._llama
        self._llama = None

        if session is not None:
            try:
                session.close()
            except Exception:
                pass

        self._shutdown_vllm()

        tokenizer = getattr(self, "_tokenizer", None)
        self._tokenizer = None
        if tokenizer is not None:
            del tokenizer

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    def _count_tokens(
        self,
        text: str,
    ) -> int:

        if self._llama is None:
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
    ) -> dict:
        if self._llama is None:
            raise RuntimeError("Qwen director model is not loaded.")

        payload = {
            "model": self._vllm_model_name(),
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
        }
        if response_format is not None:
            payload["response_format"] = response_format

        url = self._vllm_base_url() + "/chat/completions"
        response = self._llama.post(
            url,
            json=payload,
            timeout=float(os.getenv("H3_DIRECTOR_VLLM_REQUEST_TIMEOUT", "1800")),
        )

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

        if self._llama is None:
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
            decode_tps = (
                completion_tokens / elapsed
                if elapsed > 0 and completion_tokens > 0
                else 0.0
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
    ) -> str:

        if self._llama is None:
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
            decode_tps = (
                completion_tokens / elapsed
                if elapsed > 0 and completion_tokens > 0
                else 0.0
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
                response_format=None,
                finish_reason=finish_reason,
                error=error_text,
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

        content = str(
            content or ""
        ).strip()

        # Qwen3 may emit internal reasoning in a <think>...</think> block.
        # Keep narrative reasoning enabled, but never expose that block as
        # part of the story returned to the application.
        content = re.sub(
            r"<think>.*?</think>",
            "",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if re.search(r"<think>", content, flags=re.IGNORECASE):
            # A truncated reasoning block means the model spent its output
            # budget on hidden reasoning and never produced usable narrative.
            content = re.split(r"<think>", content, maxsplit=1, flags=re.IGNORECASE)[0].strip()

        if not content:
            raise RuntimeError(
                "Qwen returned an empty response."
            )

        return content

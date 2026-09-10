from __future__ import annotations

import ctypes
import faulthandler
import sys
import gc
import hashlib
import json
import os
import re
import time
from functools import wraps
from pathlib import Path


from planner.config import (
    DIRECTOR_KAGGLE_INPUT_ROOT,
    DIRECTOR_MAX_TOKENS,
    DIRECTOR_MODEL_ENV,
    DIRECTOR_MODEL_FILENAME,
    DIRECTOR_N_BATCH,
    DIRECTOR_N_CTX,
    DIRECTOR_N_GPU_LAYERS,
    DIRECTOR_TEMPERATURE,
    DIRECTOR_THREADS,
    DIRECTOR_THREADS_BATCH,
    DIRECTOR_TOP_P,
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

        explicit = os.getenv(
            DIRECTOR_MODEL_ENV,
            "",
        ).strip()

        candidates: list[Path] = []

        if explicit:

            candidates.append(
                Path(
                    explicit
                )
            )

        candidates.extend(
            [
                (
                    self.project_root
                    / "data"
                    / "models"
                    / DIRECTOR_MODEL_FILENAME
                ),
                (
                    self.project_root
                    / DIRECTOR_MODEL_FILENAME
                ),
                (
                    DIRECTOR_KAGGLE_INPUT_ROOT
                    / DIRECTOR_MODEL_FILENAME
                ),
            ]
        )

        for root in (
            self.project_root,
            DIRECTOR_KAGGLE_INPUT_ROOT,
        ):

            if not root.exists():
                continue

            try:

                candidates.extend(
                    root.rglob(
                        DIRECTOR_MODEL_FILENAME
                    )
                )

            except OSError:

                continue

        unique: list[Path] = []
        seen: set[str] = set()

        for candidate in candidates:

            candidate = Path(
                candidate
            )

            try:

                key = str(
                    candidate.resolve()
                )

            except OSError:

                key = str(
                    candidate
                )

            if key in seen:
                continue

            seen.add(
                key
            )

            unique.append(
                candidate
            )

        existing = [
            path
            for path in unique
            if path.is_file()
        ]

        if not existing:

            raise FileNotFoundError(
                "Qwen director model was not found.\n"
                f"Expected filename: "
                f"{DIRECTOR_MODEL_FILENAME}\n"
                "Attach the Kaggle director-model dataset "
                "or set H3_DIRECTOR_MODEL_PATH."
            )

        if len(existing) > 1:

            raise RuntimeError(
                "Multiple Qwen director models were found:\n"
                + "\n".join(
                    str(path)
                    for path in existing
                )
            )

        return existing[0]

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
            and self._model_path.is_file()
        )

    @staticmethod
    def _load_nvidia_cuda_libraries() -> None:

        import site

        site_roots: list[Path] = []

        try:

            site_roots.extend(
                Path(path)
                for path
                in site.getsitepackages()
                if path
            )

        except Exception:
            pass

        try:

            user_site = (
                site.getusersitepackages()
            )

            if user_site:

                site_roots.append(
                    Path(
                        user_site
                    )
                )

        except Exception:
            pass

        # The pinned llama-cpp-python wheel is cu130.
        # Never allow CUDA 12 userspace to be selected accidentally.
        cudart: list[Path] = []
        cublas: list[Path] = []
        cublaslt: list[Path] = []

        for site_root in site_roots:

            nvidia_root = (
                site_root
                / "nvidia"
            )

            if not nvidia_root.is_dir():
                continue

            try:

                cudart.extend(
                    path
                    for path
                    in nvidia_root.rglob(
                        "libcudart.so.13*"
                    )
                    if path.is_file()
                    and re.match(
                        r"libcudart\.so\.13(?:\.[0-9]+)*$",
                        path.name,
                    )
                )

                cublas.extend(
                    path
                    for path
                    in nvidia_root.rglob(
                        "libcublas.so.13*"
                    )
                    if path.is_file()
                    and re.match(
                        r"libcublas\.so\.13(?:\.[0-9]+)*$",
                        path.name,
                    )
                )

                cublaslt.extend(
                    path
                    for path
                    in nvidia_root.rglob(
                        "libcublasLt.so.13*"
                    )
                    if path.is_file()
                    and re.match(
                        r"libcublasLt\.so\.13(?:\.[0-9]+)*$",
                        path.name,
                    )
                )

            except OSError:

                continue

        if not cudart:

            raise RuntimeError(
                "No compatible libcudart.so.13 runtime was found."
            )

        if not cublas:

            raise RuntimeError(
                "No compatible libcublas.so.13 runtime was found."
            )

        if not cublaslt:

            raise RuntimeError(
                "No compatible libcublasLt.so.13 runtime was found."
            )

        cudart_lib = cudart[0]

        matching_cublas = [
            path
            for path
            in cublas
            if path.parent
            == cudart_lib.parent
        ]

        cublas_lib = (
            matching_cublas[0]
            if matching_cublas
            else cublas[0]
        )

        matching_cublaslt = [
            path
            for path
            in cublaslt
            if path.parent
            == cudart_lib.parent
        ]

        cublaslt_lib = (
            matching_cublaslt[0]
            if matching_cublaslt
            else cublaslt[0]
        )

        directories = [
            str(
                cudart_lib.parent
            ),
            str(
                cublas_lib.parent
            ),
            str(
                cublaslt_lib.parent
            ),
        ]

        old_ld = os.environ.get(
            "LD_LIBRARY_PATH",
            "",
        )

        if old_ld:

            directories.append(
                old_ld
            )

        # Put the CUDA 13 userspace directories first so a
        # pre-existing CUDA 12 path cannot win resolution.
        os.environ[
            "LD_LIBRARY_PATH"
        ] = ":".join(
            directories
        )

        try:

            ctypes.CDLL(
                str(
                    cudart_lib
                ),
                mode=ctypes.RTLD_GLOBAL,
            )

            ctypes.CDLL(
                str(
                    cublas_lib
                ),
                mode=ctypes.RTLD_GLOBAL,
            )

            ctypes.CDLL(
                str(
                    cublaslt_lib
                ),
                mode=ctypes.RTLD_GLOBAL,
            )

        except OSError as exc:

            raise RuntimeError(
                "Unable to load NVIDIA CUDA 13 libraries:\n"
                f"CUDA runtime: {cudart_lib}\n"
                f"cuBLAS: {cublas_lib}\n"
                f"cuBLASLt: {cublaslt_lib}\n"
                f"{exc}"
            ) from exc

    def load(
        self,
    ) -> None:

        if not self.available:
            return

        if self._llama is not None:
            return

        self._load_nvidia_cuda_libraries()

        try:

            from llama_cpp import Llama

        except ImportError as exc:

            raise RuntimeError(
                "llama-cpp-python is not installed."
            ) from exc

        try:

            self._llama = Llama(
                model_path=str(
                    self._model_path
                ),
                n_ctx=DIRECTOR_N_CTX,
                n_gpu_layers=DIRECTOR_N_GPU_LAYERS,
                n_batch=DIRECTOR_N_BATCH,
                n_threads=DIRECTOR_THREADS,
                n_threads_batch=DIRECTOR_THREADS_BATCH,
                flash_attn=True,
                verbose=False,
            )

        except Exception as exc:

            raise RuntimeError(
                "Failed to initialize Qwen3-14B director.\n"
                f"Model: {self._model_path}\n"
                f"Error: {exc}"
            ) from exc

    def unload(
        self,
    ) -> None:

        model = self._llama

        self._llama = None

        if model is not None:

            del model

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

        return len(
            self._llama.tokenize(
                text.encode(
                    "utf-8"
                ),
                add_bos=True,
                special=True,
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
                "type": "json_object",
                "schema": response_schema,
            }

        started = time.perf_counter()
        response = None
        error_text = ""

        print(f"[QWEN] START {call_name}", flush=True)

        try:
            response = self._llama.create_chat_completion(
                **kwargs
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
            response = (
                self._llama.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
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

from __future__ import annotations

import json
import logging
from planner.config import RUNTIME
import os
import time
import uuid

from pathlib import Path
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.parse import urlencode
from urllib.request import (
    Request,
    urlopen,
)


LOGGER = logging.getLogger(__name__)


class ComfyClient:

    def __init__(
        self,
        base_url: str,
        timeout: float | None = None,
        request_retries: int | None = None,
    ):

        self.base_url = (
            str(base_url).strip().rstrip("/")
        )

        if not self.base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                "ComfyUI base_url must start with http:// or https://."
            )

        runtime_cfg = dict(RUNTIME.get("runtime", {}) or {})
        if timeout is None:
            timeout = float(
                runtime_cfg.get(
                    "comfyui_request_timeout_seconds",
                    60,
                )
            )
        if request_retries is None:
            request_retries = int(
                runtime_cfg.get(
                    "comfyui_request_retries",
                    3,
                )
            )

        configured_timeout = os.getenv("H3_COMFY_REQUEST_TIMEOUT")
        configured_retries = os.getenv("H3_COMFY_REQUEST_RETRIES")
        if configured_timeout is not None:
            timeout = float(configured_timeout)
        if configured_retries is not None:
            request_retries = int(configured_retries)

        self.timeout = max(1, float(timeout))
        self.request_retries = max(0, int(request_retries))

        self.client_id = str(
            uuid.uuid4()
        )

    def _request(
        self,
        method,
        path,
        payload=None,
        retry=False,
        timeout=None,
    ):

        url = (
            self.base_url
            + path
        )

        data = None
        headers = {}

        if payload is not None:

            data = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode(
                "utf-8"
            )

            headers[
                "Content-Type"
            ] = "application/json"

        attempts = (
            self.request_retries + 1
            if retry
            else 1
        )

        request_timeout = (
            self.timeout
            if timeout is None
            else max(
                1,
                int(timeout),
            )
        )

        for attempt in range(
            attempts
        ):

            request = Request(
                url=url,
                method=method,
                data=data,
                headers=headers,
            )

            try:

                with urlopen(
                    request,
                    timeout=request_timeout,
                ) as response:

                    body = response.read()

                    if not body:
                        return None

                    content_type = (
                        response.headers.get(
                            "Content-Type",
                            "",
                        )
                    ).lower()

                    if (
                        "json"
                        in content_type
                        or body[:1] in {
                            b"{",
                            b"[",
                        }
                    ):

                        try:
                            return json.loads(
                                body.decode(
                                    "utf-8"
                                )
                            )
                        except json.JSONDecodeError:
                            pass

                    return body

            except HTTPError as error:

                body = (
                    error.read()
                    .decode(
                        "utf-8",
                        errors="replace",
                    )
                )

                if (
                    retry
                    and error.code
                    in {
                        408,
                        429,
                        500,
                        502,
                        503,
                        504,
                    }
                    and attempt
                    < attempts - 1
                ):

                    delay = min(
                        2 ** attempt,
                        10,
                    )
                    LOGGER.warning(
                        "ComfyUI HTTP %s for %s; retrying in %ss "
                        "(attempt %s/%s).",
                        error.code,
                        url,
                        delay,
                        attempt + 1,
                        attempts - 1,
                    )
                    time.sleep(delay)

                    continue

                raise RuntimeError(
                    f"ComfyUI HTTP {error.code}: "
                    f"{body}"
                ) from error

            except (
                URLError,
                TimeoutError,
            ) as error:

                if (
                    retry
                    and attempt
                    < attempts - 1
                ):

                    delay = min(
                        2 ** attempt,
                        10,
                    )
                    LOGGER.warning(
                        "ComfyUI connection failure for %s; "
                        "retrying in %ss (attempt %s/%s): %s",
                        url,
                        delay,
                        attempt + 1,
                        attempts - 1,
                        error,
                    )
                    time.sleep(delay)

                    continue

                raise RuntimeError(
                    f"Cannot connect to ComfyUI "
                    f"{self.base_url}: {error}"
                ) from error

        raise RuntimeError(
            "ComfyUI request failed."
        )

    def free_memory(self, *, unload_models: bool = False, free_memory: bool = True) -> bool:
        """Ask ComfyUI to release resources after a failed GPU job."""
        result = self._request(
            "POST",
            "/free",
            payload={
                "unload_models": bool(unload_models),
                "free_memory": bool(free_memory),
            },
            retry=True,
            timeout=30,
        )
        return result is None or result == b"" or isinstance(result, (dict, list))

    def free_and_wait(self, *, unload_models: bool = True, free_memory: bool = True) -> bool:
        ok = self.free_memory(unload_models=unload_models, free_memory=free_memory)
        if not ok:
            return False
        return self.health_check()

    def get_system_stats(self) -> dict:
        result = self._request("GET", "/system_stats", retry=True, timeout=min(self.timeout, 10))
        if not isinstance(result, dict):
            raise RuntimeError("Invalid /system_stats response.")
        return result

    def health_check(
        self,
    ):

        try:
            self._request(
                "GET",
                "/system_stats",
                retry=True,
            )

            return True

        except Exception as error:
            LOGGER.debug(
                "ComfyUI health check failed for %s: %s",
                self.base_url,
                error,
                exc_info=True,
            )
            return False

    def get_object_info(
        self,
    ):

        result = self._request(
            "GET",
            "/object_info",
            retry=True,
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Invalid /object_info response."
            )

        return result

    def convert_workflow(
        self,
        workflow,
    ):

        result = self._request(
            "POST",
            "/workflow/convert",
            payload=workflow,
            retry=False,
            timeout=180,
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Workflow converter returned "
                "invalid data."
            )

        if result.get(
            "success"
        ) is False:

            raise RuntimeError(
                "Workflow conversion failed: "
                + str(
                    result.get(
                        "error",
                        result,
                    )
                )
            )

        if all(
            isinstance(
                value,
                dict,
            )
            for value in result.values()
        ):

            return result

        for key in (
            "workflow",
            "prompt",
            "data",
        ):

            candidate = result.get(
                key
            )

            if (
                isinstance(
                    candidate,
                    dict,
                )
                and all(
                    isinstance(
                        value,
                        dict,
                    )
                    for value
                    in candidate.values()
                )
            ):

                return candidate

        raise RuntimeError(
            "Could not find converted API "
            "workflow in converter response."
        )

    def queue_prompt(
        self,
        workflow,
    ) -> str:

        result = self._request(
            "POST",
            "/prompt",
            payload={
                "prompt": workflow,
                "client_id": self.client_id,
            },
            retry=False,
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Invalid /prompt response."
            )

        if result.get(
            "error"
        ):

            raise RuntimeError(
                "ComfyUI rejected prompt: "
                + str(
                    result["error"]
                )
            )

        prompt_id = result.get(
            "prompt_id"
        )

        if not prompt_id:

            raise RuntimeError(
                "ComfyUI did not return prompt_id: "
                + str(result)
            )

        return str(
            prompt_id
        )

    def get_history(
        self,
        prompt_id,
    ):

        result = self._request(
            "GET",
            f"/history/{prompt_id}",
            retry=True,
        )

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                "Invalid history response."
            )

        return result

    def cancel_prompt(self, prompt_id: str, *, interrupt_running: bool = True) -> None:
        """Best-effort cancellation for one prompt on this dedicated worker.

        ProductionRunner provisions one ComfyUI process per GPU, so interrupting
        the worker is scoped to the job currently owned by this client. Queued
        work is removed through the supported /queue POST API.
        """
        prompt_id = str(prompt_id or "").strip()
        if not prompt_id:
            return
        if interrupt_running:
            try:
                self._request(
                    "POST",
                    "/interrupt",
                    payload={},
                    retry=False,
                    timeout=10,
                )
            except Exception as error:
                LOGGER.warning("ComfyUI interrupt failed for %s: %s", prompt_id, error)
        try:
            self._request(
                "POST",
                "/queue",
                payload={"delete": [prompt_id]},
                retry=False,
                timeout=10,
            )
        except Exception as error:
            LOGGER.warning("ComfyUI queue deletion failed for %s: %s", prompt_id, error)

    def wait_for_prompt(
        self,
        prompt_id,
        poll_interval=2.0,
        timeout=14400.0,
        *,
        liveness_interval=15.0,
        max_liveness_failures=3,
    ):

        prompt_id = str(
            prompt_id or ""
        ).strip()

        if not prompt_id:
            raise ValueError(
                "prompt_id cannot be empty."
            )

        delay = float(
            poll_interval
        )

        if delay <= 0:
            raise ValueError(
                "poll_interval must be greater than zero."
            )

        timeout = float(
            timeout
        )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        started = time.monotonic()
        last_liveness = started
        liveness_failures = 0

        try:
            while True:

                now = time.monotonic()
                if now - started > timeout:
                    self.cancel_prompt(prompt_id)
                    raise TimeoutError(
                        f"ComfyUI prompt {prompt_id} timed out after {timeout:.0f}s."
                    )

                if now - last_liveness >= max(1.0, float(liveness_interval)):
                    last_liveness = now
                    try:
                        self._request(
                            "GET",
                            "/system_stats",
                            retry=False,
                            timeout=min(self.timeout, 10),
                        )
                        liveness_failures = 0
                    except Exception as health_error:
                        liveness_failures += 1
                        LOGGER.warning(
                            "ComfyUI liveness check failed for %s (%s/%s): %s",
                            prompt_id,
                            liveness_failures,
                            max_liveness_failures,
                            health_error,
                        )
                        if liveness_failures >= max(1, int(max_liveness_failures)):
                            self.cancel_prompt(prompt_id)
                            raise RuntimeError(
                                f"ComfyUI worker became unreachable while prompt {prompt_id} was pending."
                            ) from health_error

                try:
                    history = self.get_history(prompt_id)
                except RuntimeError as history_error:
                    # A transient history request failure is tolerated, but a
                    # dead worker is detected immediately by the next liveness probe.
                    LOGGER.warning(
                        "ComfyUI history poll failed for %s: %s",
                        prompt_id,
                        history_error,
                    )
                    time.sleep(delay)
                    delay = min(delay * 1.5, 10.0)
                    continue

            if prompt_id in history:

                result = history[
                    prompt_id
                ]

                status = result.get(
                    "status",
                    {},
                )

                if (
                    status.get(
                        "status_str"
                    )
                    == "error"
                ):
                    LOGGER.error(
                        "ComfyUI prompt %s failed: %s",
                        prompt_id,
                        status,
                    )

                    raise RuntimeError(
                        f"ComfyUI failed {prompt_id}: "
                        f"{status}"
                    )

                if result.get("node_errors"):
                    raise RuntimeError(
                        f"ComfyUI prompt {prompt_id} reported node errors: "
                        f"{result.get('node_errors')}"
                    )

                if status.get("status_str") == "success":
                    return result

                time.sleep(
                    delay
                )

                delay = min(
                    delay * 1.5,
                    10.0,
                )

        except TimeoutError:
            raise

    def download_file(
        self,
        filename,
        subfolder="",
        file_type="output",
        destination=None,
    ) -> Path:
        """Stream a ComfyUI output to disk with bounded memory usage and atomic publication."""
        query = urlencode({"filename": filename, "subfolder": subfolder, "type": file_type})
        destination = Path(destination if destination is not None else filename).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_name(f".{destination.name}.{os.getpid()}.download")
        attempts = self.request_retries + 1
        last_error = None

        try:
            for attempt in range(attempts):
                try:
                    request = Request(self.base_url + f"/view?{query}", method="GET")
                    with urlopen(request, timeout=self.timeout) as response:
                        content_type = (response.headers.get("Content-Type", "") or "").lower()
                        if "json" in content_type:
                            body = response.read(8192).decode("utf-8", errors="replace")
                            raise RuntimeError(f"ComfyUI /view returned JSON instead of media: {body}")
                        with temp_path.open("wb") as handle:
                            while True:
                                chunk = response.read(8 * 1024 * 1024)
                                if not chunk:
                                    break
                                handle.write(chunk)
                            handle.flush()
                            os.fsync(handle.fileno())
                    if not temp_path.is_file() or temp_path.stat().st_size <= 0:
                        raise RuntimeError("ComfyUI /view returned an empty file.")
                    os.replace(temp_path, destination)
                    return destination
                except HTTPError as error:
                    body = error.read().decode("utf-8", errors="replace")[-4000:]
                    last_error = RuntimeError(f"ComfyUI HTTP {error.code}: {body}")
                    if error.code not in {408, 429, 500, 502, 503, 504} or attempt >= attempts - 1:
                        raise last_error from error
                except (URLError, TimeoutError, OSError, RuntimeError) as error:
                    last_error = error
                    if attempt >= attempts - 1:
                        raise RuntimeError(f"ComfyUI download failed for {filename}: {error}") from error
                delay = min(2 ** attempt, 10)
                LOGGER.warning(
                    "ComfyUI download failed for %s; retrying in %ss (attempt %s/%s).",
                    filename, delay, attempt + 1, attempts - 1,
                )
                time.sleep(delay)
            raise RuntimeError(f"ComfyUI download failed: {last_error}")
        finally:
            temp_path.unlink(missing_ok=True)

    @staticmethod
    def _is_video(
        filename,
    ):

        return Path(
            filename
        ).suffix.lower() in {
            ".mp4",
            ".mov",
            ".mkv",
            ".webm",
        }

    @classmethod
    def find_video_outputs(
        cls,
        history,
    ):

        results = []

        outputs = history.get(
            "outputs",
            {},
        )

        if not isinstance(
            outputs,
            dict,
        ):
            return results

        for node_output in outputs.values():

            if not isinstance(
                node_output,
                dict,
            ):
                continue

            for items in node_output.values():

                if not isinstance(
                    items,
                    list,
                ):
                    continue

                for item in items:

                    if not isinstance(
                        item,
                        dict,
                    ):
                        continue

                    filename = item.get(
                        "filename"
                    )

                    if (
                        filename
                        and cls._is_video(
                            filename
                        )
                    ):

                        results.append(
                            {
                                "filename": filename,
                                "subfolder": item.get(
                                    "subfolder",
                                    "",
                                ),
                                "type": item.get(
                                    "type",
                                    "output",
                                ),
                            }
                        )

        return results

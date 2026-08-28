from __future__ import annotations

import json
import logging
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
        timeout: int = 60,
        request_retries: int = 3,
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

        self.timeout = max(
            1,
            int(timeout),
        )

        self.request_retries = max(
            0,
            int(request_retries),
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
                "client_id": str(
                    uuid.uuid4()
                ),
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

    def wait_for_prompt(
        self,
        prompt_id,
        poll_interval=2.0,
        timeout=14400.0,
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

        while True:

            if (
                time.monotonic()
                - started
                > timeout
            ):

                raise TimeoutError(
                    f"ComfyUI prompt {prompt_id} "
                    "timed out."
                )

            history = (
                self.get_history(
                    prompt_id
                )
            )

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

                if (
                    status.get(
                        "status_str"
                    )
                    == "success"
                ):

                    return result

                if result.get(
                    "outputs"
                ):

                    return result

            time.sleep(
                delay
            )

            delay = min(
                delay * 1.5,
                10.0,
            )

    def download_file(
        self,
        filename,
        subfolder="",
        file_type="output",
        destination=None,
    ) -> Path:

        query = urlencode(
            {
                "filename": filename,
                "subfolder": subfolder,
                "type": file_type,
            }
        )

        data = self._request(
            "GET",
            f"/view?{query}",
            retry=True,
        )

        if not isinstance(
            data,
            bytes,
        ):

            raise RuntimeError(
                "ComfyUI /view did not return "
                "binary data."
            )

        if destination is None:
            destination = Path(
                filename
            )

        destination = Path(
            destination
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination.write_bytes(
            data
        )

        if (
            not destination.is_file()
            or destination.stat().st_size <= 0
        ):

            raise RuntimeError(
                "Downloaded file is missing "
                "or empty."
            )

        return destination

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

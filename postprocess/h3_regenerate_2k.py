from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from urllib.error import (
    HTTPError,
    URLError,
)
from urllib.request import (
    Request,
    urlopen,
)

from planner.config import (
    H3_API_BASE,
    H3_API_KEY,
    H3_QUERY_ENDPOINT,
    H3_REGENERATE_ENDPOINT,
)


class H3Regenerate2K:

    def __init__(
        self,
        api_key=None,
        api_base=H3_API_BASE,
        endpoint=H3_REGENERATE_ENDPOINT,
        query_endpoint=H3_QUERY_ENDPOINT,
    ):

        self.api_key = (
            api_key
            or H3_API_KEY
        )

        self.api_base = (
            api_base.rstrip("/")
        )

        self.endpoint = endpoint
        self.query_endpoint = (
            query_endpoint.rstrip("/")
        )

    def _post(
        self,
        payload,
    ):

        if not self.api_key:
            raise RuntimeError(
                "MINIMAX_API_KEY is not configured."
            )

        request = Request(
            self.api_base
            + self.endpoint,
            method="POST",
            data=json.dumps(
                payload
            ).encode(
                "utf-8"
            ),
            headers={
                "Content-Type":
                    "application/json",
                "Authorization":
                    f"Bearer {self.api_key}",
            },
        )

        try:

            with urlopen(
                request,
                timeout=300,
            ) as response:

                return json.loads(
                    response.read().decode(
                        "utf-8"
                    )
                )

        except HTTPError as error:

            body = (
                error.read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )

            raise RuntimeError(
                f"H3 Regenerate-2K HTTP "
                f"{error.code}: {body}"
            ) from error

        except URLError as error:

            raise RuntimeError(
                f"H3 Regenerate-2K connection "
                f"failed: {error}"
            ) from error

    def _get_task(
        self,
        task_id,
    ):

        request = Request(
            (
                f"{self.api_base}"
                f"{self.query_endpoint}"
                f"/{task_id}"
            ),
            method="GET",
            headers={
                "Authorization":
                    f"Bearer {self.api_key}",
            },
        )

        with urlopen(
            request,
            timeout=120,
        ) as response:

            return json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

    @staticmethod
    def _data_url(
        video_path,
    ):

        encoded = base64.b64encode(
            Path(video_path).read_bytes()
        ).decode(
            "ascii"
        )

        return (
            "data:video/mp4;base64,"
            + encoded
        )

    def regenerate(
        self,
        source_video: Path,
        destination: Path,
        prompt: str = "",
    ) -> Path:

        source_video = Path(
            source_video
        )

        destination = Path(
            destination
        )

        if not source_video.is_file():
            raise FileNotFoundError(
                source_video
            )

        content = [
            {
                "type": "video_url",
                "video_url": {
                    "url": self._data_url(
                        source_video
                    ),
                },
                "role": "base_video",
            }
        ]

        if prompt.strip():

            content.insert(
                0,
                {
                    "type": "text",
                    "text": prompt.strip(),
                },
            )

        result = self._post(
            {
                "model": "MiniMax-H3",
                "content": content,
                "resolution": "2K",
            }
        )

        task_id = (
            result.get(
                "task_id"
            )
        )

        if not task_id:

            task_id = (
                result.get(
                    "data",
                    {}
                ).get(
                    "task_id"
                )
                if isinstance(
                    result.get(
                        "data"
                    ),
                    dict,
                )
                else None
            )

        if not task_id:

            raise RuntimeError(
                "H3 regeneration did not return "
                "a task_id:\n"
                + json.dumps(
                    result,
                    indent=2,
                )
            )

        deadline = time.monotonic() + 3600

        while time.monotonic() < deadline:

            status = self._get_task(
                task_id
            )

            task = status.get(
                "task",
                status,
            )

            task_status = (
                str(
                    task.get(
                        "status",
                        ""
                    )
                ).lower()
            )

            if task_status in {
                "success",
                "succeeded",
                "completed",
            }:

                content = (
                    task.get(
                        "content"
                    )
                )

                if isinstance(
                    content,
                    dict,
                ):

                    url = (
                        content.get(
                            "url"
                        )
                    )

                else:
                    url = None

                if not url:

                    raise RuntimeError(
                        "H3 regeneration completed "
                        "without an output URL:\n"
                        + json.dumps(
                            status,
                            indent=2,
                        )
                    )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                request = Request(
                    url,
                    method="GET",
                )

                with urlopen(
                    request,
                    timeout=600,
                ) as response:

                    destination.write_bytes(
                        response.read()
                    )

                if (
                    not destination.is_file()
                    or destination.stat().st_size <= 0
                ):

                    raise RuntimeError(
                        "Downloaded H3 2K result "
                        "is empty."
                    )

                return destination

            if task_status in {
                "failed",
                "error",
                "cancelled",
            }:

                raise RuntimeError(
                    "H3 2K regeneration failed:\n"
                    + json.dumps(
                        status,
                        indent=2,
                    )
                )

            time.sleep(
                5
            )

        raise TimeoutError(
            f"H3 2K regeneration task "
            f"{task_id} timed out."
        )

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import (
    Request,
    urlopen,
)

from planner.config import (
    PLANNER_API_KEY,
    PLANNER_BASE_URL,
    PLANNER_MAX_NEW_TOKENS,
    PLANNER_MODEL,
    PLANNER_TEMPERATURE,
    PLANNER_TOP_P,
)


class QwenStoryModel:

    def __init__(
        self,
        model_id: str | None = None,
    ):
        self.model_id = (
            model_id
            or PLANNER_MODEL
        )

    def load(self) -> None:
        self._validate()

    def _validate(self) -> None:

        if not PLANNER_BASE_URL:
            raise RuntimeError(
                "PLANNER_BASE_URL is not configured."
            )

        if not self.model_id:
            raise RuntimeError(
                "PLANNER_MODEL is not configured."
            )

    def generate(
        self,
        messages: list,
        max_new_tokens: int = PLANNER_MAX_NEW_TOKENS,
        temperature: float = PLANNER_TEMPERATURE,
        top_p: float = PLANNER_TOP_P,
    ) -> str:

        self._validate()

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }

        request = Request(
            PLANNER_BASE_URL
            + "/chat/completions",
            method="POST",
            data=json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **(
                    {
                        "Authorization":
                        f"Bearer {PLANNER_API_KEY}"
                    }
                    if PLANNER_API_KEY
                    else {}
                ),
            },
        )

        try:
            with urlopen(
                request,
                timeout=600,
            ) as response:

                raw = response.read()

        except HTTPError as error:

            body = error.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"Planner API HTTP {error.code}: "
                f"{body}"
            ) from error

        except URLError as error:

            raise RuntimeError(
                f"Planner API connection failed: "
                f"{error}"
            ) from error

        try:
            result = json.loads(
                raw.decode("utf-8")
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Planner API returned invalid JSON."
            ) from error

        try:
            text = (
                result["choices"][0]
                ["message"]["content"]
            )
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise RuntimeError(
                "Planner API response is missing "
                "choices[0].message.content."
            ) from error

        text = str(text).strip()

        if not text:
            raise RuntimeError(
                "Planner returned empty content."
            )

        return text

    def unload(self) -> None:
        return None

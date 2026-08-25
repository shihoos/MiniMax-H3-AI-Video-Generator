from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from planner.config import (
    PLANNER_API_KEY,
    PLANNER_BASE_URL,
    PLANNER_MAX_NEW_TOKENS,
    PLANNER_MODEL,
    PLANNER_TEMPERATURE,
    PLANNER_TOP_P,
)


class QwenStoryModel:
    """
    Planning-model client.

    IMPORTANT:
    The locked MiniMax H3 qwen3vl text encoder is NOT used here
    as a chat-completion model. It is used by the H3 ComfyUI
    generation workflow.

    This client expects a separate OpenAI-compatible planning
    service.
    """

    def __init__(
        self,
        model_id: str | None = None,
    ):
        self.model_id = (
            model_id or PLANNER_MODEL
        )

    def _validate(self) -> None:

        if not PLANNER_BASE_URL:
            raise RuntimeError(
                "PLANNER_BASE_URL is not configured."
            )

        if not self.model_id:
            raise RuntimeError(
                "PLANNER_MODEL is not configured."
            )

    def load(self) -> None:
        self._validate()

    def generate(
        self,
        messages: list[dict],
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

        headers = {
            "Content-Type": "application/json",
        }

        if PLANNER_API_KEY:
            headers["Authorization"] = (
                f"Bearer {PLANNER_API_KEY}"
            )

        request = Request(
            f"{PLANNER_BASE_URL}/chat/completions",
            method="POST",
            headers=headers,
            data=json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8"),
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
                f"Planner HTTP {error.code}: {body}"
            ) from error

        except URLError as error:

            raise RuntimeError(
                f"Planner connection failed: {error}"
            ) from error

        try:
            result = json.loads(
                raw.decode("utf-8")
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Planner returned invalid JSON."
            ) from error

        try:
            content = (
                result["choices"][0]
                ["message"]["content"]
            )
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as error:

            raise RuntimeError(
                "Planner response is missing "
                "choices[0].message.content."
            ) from error

        content = str(
            content
        ).strip()

        if not content:
            raise RuntimeError(
                "Planner returned empty content."
            )

        return content

    def unload(self) -> None:
        """
        External planner has no local model object to unload.
        """
        return None

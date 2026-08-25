from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from planner.config import (
    PLANNER_API_BASE_URL,
    PLANNER_API_KEY,
    PLANNER_MAX_NEW_TOKENS,
    PLANNER_MODEL,
    PLANNER_TEMPERATURE,
    PLANNER_TOP_P,
)


class QwenStoryModel:
    """
    Compatibility wrapper retained for existing planner modules.

    IMPORTANT:
    The locked Qwen3-VL-32B H3 .safetensors file is an H3 encoder,
    not a standalone causal text-generation checkpoint.

    Therefore story/scene/shot planning is delegated to an
    OpenAI-compatible text-generation endpoint.

    This keeps the six-model production dataset unchanged.
    """

    def __init__(
        self,
        model_id: str | None = None,
    ):
        self.model_id = (
            model_id
            or PLANNER_MODEL
        )

    def load(self):
        self._validate_configuration()

    def _validate_configuration(self):
        if not PLANNER_API_BASE_URL:
            raise RuntimeError(
                "PLANNER_API_BASE_URL is not configured. "
                "The locked MiniMax H3 Qwen3-VL-32B file is an "
                "H3 text encoder, not a standalone story-generation "
                "model. Configure an OpenAI-compatible planner endpoint."
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

        self._validate_configuration()

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": int(
                max_new_tokens
            ),
            "temperature": float(
                temperature
            ),
            "top_p": float(
                top_p
            ),
        }

        data = json.dumps(
            payload,
            ensure_ascii=False,
        ).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }

        if PLANNER_API_KEY:
            headers["Authorization"] = (
                f"Bearer {PLANNER_API_KEY}"
            )

        request = Request(
            (
                PLANNER_API_BASE_URL
                + "/chat/completions"
            ),
            method="POST",
            data=data,
            headers=headers,
        )

        try:
            with urlopen(
                request,
                timeout=600,
            ) as response:

                body = response.read()

        except HTTPError as error:
            message = (
                error.read()
                .decode(
                    "utf-8",
                    errors="replace",
                )
            )
            raise RuntimeError(
                f"Planner API HTTP {error.code}: "
                f"{message}"
            ) from error

        except URLError as error:
            raise RuntimeError(
                f"Planner API connection failed: "
                f"{error}"
            ) from error

        try:
            result = json.loads(
                body.decode("utf-8")
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "Planner API returned invalid JSON."
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
                "Planner API response does not contain "
                "choices[0].message.content."
            ) from error

        content = str(
            content
        ).strip()

        if not content:
            raise RuntimeError(
                "Planner returned empty output."
            )

        return content

    def unload(self):
        """
        Kept for compatibility with the existing orchestration layer.
        Remote planner clients have no local model to unload.
        """
        return None

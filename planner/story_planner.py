from __future__ import annotations

from pathlib import Path

from planner.config import (
    AI_STORY_MODE,
    EXPAND_USER_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    QWEN_STORY_TEMPERATURE,
)
from planner.qwen_loader import QwenStoryModel


class StoryPlanner:
    """
    Converts the user's request/story into the canonical production story.

    The three story modes deliberately converge to one plain-text story
    representation. Scene/shot planning happens later.
    """

    def __init__(self, model=None):
        self.model = (
            model
            if model is not None
            else QwenStoryModel()
        )

        self.project_root = (
            Path(__file__).resolve().parents[1]
        )

        self.prompts_dir = (
            self.project_root
            / "prompts"
            / "qwen"
        )

    def _read_prompt(self, filename: str) -> str:
        path = self.prompts_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing story prompt: {path}"
            )

        return path.read_text(
            encoding="utf-8"
        )

    @staticmethod
    def _format_prompt(
        template: str,
        user_input: str,
    ) -> str:
        return template.replace(
            "{user_request}",
            user_input,
        ).replace(
            "{user_story}",
            user_input,
        )

    def _messages(
        self,
        prompt: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are the story-development stage "
                    "of a production-grade cinematic video "
                    "generation system. "
                    "Return only the requested story text. "
                    "Do not return JSON unless the user prompt "
                    "explicitly requires JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

    def plan(
        self,
        mode: str,
        user_input: str,
    ) -> str:

        user_input = str(
            user_input or ""
        ).strip()

        if not user_input:
            raise ValueError(
                "Story input cannot be empty."
            )

        if mode == AI_STORY_MODE:
            filename = "create_story.txt"

        elif mode == PRESERVE_USER_STORY_MODE:
            filename = "preserve_story.txt"

        elif mode == EXPAND_USER_STORY_MODE:
            filename = "expand_story.txt"

        else:
            raise ValueError(
                f"Unsupported story mode: {mode}"
            )

        template = self._read_prompt(
            filename
        )

        prompt = self._format_prompt(
            template,
            user_input,
        )

        result = self.model.generate(
            self._messages(prompt),
            temperature=QWEN_STORY_TEMPERATURE,
        )

        result = str(result).strip()

        if not result:
            raise RuntimeError(
                "Story planner returned empty output."
            )

        return result

    def unload(self):
        self.model.unload()

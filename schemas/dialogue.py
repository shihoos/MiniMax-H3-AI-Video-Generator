from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DialogueEvent:
    """Production-owned dialogue event.

    Qwen supplies speaker/text/continuation intent. Deterministic scheduling
    supplies exact timing after canonical speaker resolution.
    """

    dialogue_id: str
    speaker_id: str
    speaker_name: str
    text: str
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    pre_roll_seconds: float = 0.0
    post_roll_seconds: float = 0.0
    continues_from_previous_shot: bool = False
    continues_to_next_shot: bool = False
    expected_duration_ms: int | None = None
    duration_source: str = "wpm_estimate"

    def __post_init__(self) -> None:
        self.dialogue_id = str(self.dialogue_id or "").strip()
        self.speaker_id = str(self.speaker_id or "").strip()
        self.speaker_name = str(self.speaker_name or "").strip()
        self.text = str(self.text or "")
        if not self.dialogue_id:
            raise ValueError("dialogue_id cannot be empty.")
        if not self.speaker_id:
            raise ValueError("speaker_id cannot be empty.")
        if not self.speaker_name:
            raise ValueError("speaker_name cannot be empty.")
        if not self.text.strip():
            raise ValueError("Dialogue text cannot be empty.")
        for name in (
            "start_seconds",
            "end_seconds",
            "pre_roll_seconds",
            "post_roll_seconds",
        ):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative.")
            setattr(self, name, value)
        if self.end_seconds < self.start_seconds:
            raise ValueError("Dialogue end cannot precede start.")
        if self.expected_duration_ms is not None:
            self.expected_duration_ms = int(self.expected_duration_ms)
            if self.expected_duration_ms <= 0:
                raise ValueError("expected_duration_ms must be greater than zero.")
        self.duration_source = str(self.duration_source or "wpm_estimate").strip()

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialogue_id": self.dialogue_id,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "pre_roll_seconds": self.pre_roll_seconds,
            "post_roll_seconds": self.post_roll_seconds,
            "continues_from_previous_shot": self.continues_from_previous_shot,
            "continues_to_next_shot": self.continues_to_next_shot,
            "expected_duration_ms": self.expected_duration_ms,
            "duration_source": self.duration_source,
        }

    @staticmethod
    def json_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "speaker": {"type": "string"},
                "text": {"type": "string"},
                "continues_to_next_shot": {"type": "boolean"},
            },
            "required": ["speaker", "text", "continues_to_next_shot"],
            "additionalProperties": False,
        }

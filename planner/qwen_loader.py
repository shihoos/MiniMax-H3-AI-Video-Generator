from __future__ import annotations


class QwenStoryModel:
    """
    Compatibility facade for the project planner.

    The locked Qwen3-VL-32B checkpoint is the MiniMax H3
    conditioning encoder, not an independent chat-completion
    endpoint.

    Therefore this class intentionally does not contact an
    external LLM service and does not load another Qwen model.
    """

    def __init__(
        self,
        model_id: str | None = None,
    ):
        self.model_id = (
            model_id
            or
            "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"
        )

    def load(self) -> None:
        return None

    def generate(
        self,
        messages,
        max_new_tokens=0,
        temperature=0.0,
        top_p=1.0,
    ) -> str:
        raise RuntimeError(
            "QwenStoryModel.generate() is disabled. "
            "The locked Qwen checkpoint is the MiniMax H3 "
            "conditioning encoder, not a standalone chat model. "
            "The project planner must use deterministic local "
            "planning and must not introduce another Qwen model."
        )

    def unload(self) -> None:
        return None

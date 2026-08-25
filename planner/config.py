# ============================================================
# PLANNER MODEL
# ============================================================

PLANNER_BACKEND = os.getenv(
    "PLANNER_BACKEND",
    "openai_compatible",
).strip().lower()

PLANNER_BASE_URL = os.getenv(
    "PLANNER_BASE_URL",
    "",
).rstrip("/")

PLANNER_API_KEY = os.getenv(
    "PLANNER_API_KEY",
    "",
)

PLANNER_MODEL = os.getenv(
    "PLANNER_MODEL",
    "",
)

# IMPORTANT:
# This is deliberately NOT the locked H3 text encoder.
#
# The locked H3 Qwen model is consumed by ComfyUI as the
# MiniMax H3 conditioning encoder.
#
# The planner requires a chat-capable Qwen/LLM backend.

PLANNER_MAX_NEW_TOKENS = int(
    os.getenv(
        "PLANNER_MAX_NEW_TOKENS",
        "4096",
    )
)

PLANNER_TEMPERATURE = float(
    os.getenv(
        "PLANNER_TEMPERATURE",
        "0.25",
    )
)

PLANNER_TOP_P = float(
    os.getenv(
        "PLANNER_TOP_P",
        "0.8",
    )
)

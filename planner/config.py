from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


# ============================================================
# PLANNER API
# ============================================================

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


# Stage-specific planner temperatures.
QWEN_STORY_TEMPERATURE = float(
    os.getenv(
        "QWEN_STORY_TEMPERATURE",
        "0.35",
    )
)

QWEN_CHARACTER_DETECTION_TEMPERATURE = float(
    os.getenv(
        "QWEN_CHARACTER_DETECTION_TEMPERATURE",
        "0.10",
    )
)

QWEN_CHARACTER_PLAN_TEMPERATURE = float(
    os.getenv(
        "QWEN_CHARACTER_PLAN_TEMPERATURE",
        "0.20",
    )
)

QWEN_SCENE_PLAN_TEMPERATURE = float(
    os.getenv(
        "QWEN_SCENE_PLAN_TEMPERATURE",
        "0.20",
    )
)

QWEN_SHOT_PLAN_TEMPERATURE = float(
    os.getenv(
        "QWEN_SHOT_PLAN_TEMPERATURE",
        "0.20",
    )
)


# ============================================================
# PROMPTS
# ============================================================

QWEN_PROMPTS_DIR = (
    PROJECT_ROOT / "prompts" / "qwen"
)

SYSTEM_PROMPT_PATH = (
    QWEN_PROMPTS_DIR / "system.txt"
)

CREATE_STORY_PROMPT_PATH = (
    QWEN_PROMPTS_DIR / "create_story.txt"
)

PRESERVE_STORY_PROMPT_PATH = (
    QWEN_PROMPTS_DIR / "preserve_story.txt"
)

EXPAND_STORY_PROMPT_PATH = (
    QWEN_PROMPTS_DIR / "expand_story.txt"
)

CHARACTER_PLAN_PROMPT_PATH = (
    QWEN_PROMPTS_DIR / "character_plan.txt"
)

SCENE_PLAN_PROMPT_PATH = (
    QWEN_PROMPTS_DIR / "scene_plan.txt"
)

SHOT_PLAN_PROMPT_PATH = (
    QWEN_PROMPTS_DIR / "shot_plan.txt"
)


# ============================================================
# DATA
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

STORIES_DIR = (
    DATA_DIR / "stories"
)

CHARACTERS_DIR = (
    DATA_DIR / "characters"
)

GENERATED_CHARACTERS_DIR = (
    CHARACTERS_DIR / "generated"
)

SCENES_DIR = (
    DATA_DIR / "scenes"
)

SHOTS_DIR = (
    DATA_DIR / "shots"
)

PRODUCTION_DIR = (
    DATA_DIR / "production"
)


# ============================================================
# LOCKED H3 MODEL INVENTORY
# ============================================================

H3_REF2VA_MODEL = (
    "MiniMax_H3_Ref2VA_pruned_mixed_int4_int8_convrot.safetensors"
)

H3_TEXT_ENCODER = (
    "qwen3vl_32b_minimax_h3_int4_convrot.safetensors"
)

H3_TURBO_LORA = (
    "minimax_h3_turbo_v4_step600_ema.safetensors"
)

H3_VIDEO_VAE = (
    "minimax_h3_video_vae_fp16.safetensors"
)

H3_AUDIO_VAE = (
    "minimax_h3_audio_vae_fp32.safetensors"
)

H3_LATENT_UPSCALER_3D = (
    "minimax_h3_latent_upscaler_3d_fp16.safetensors"
)


# ============================================================
# H3 GENERATION
# ============================================================

H3_FPS = int(
    os.getenv(
        "H3_FPS",
        "24",
    )
)

H3_WIDTH = int(
    os.getenv(
        "H3_WIDTH",
        "960",
    )
)

H3_HEIGHT = int(
    os.getenv(
        "H3_HEIGHT",
        "544",
    )
)

H3_FRAMES_PER_SHOT = int(
    os.getenv(
        "H3_FRAMES_PER_SHOT",
        "125",
    )
)

H3_STEPS = int(
    os.getenv(
        "H3_STEPS",
        "14",
    )
)

H3_REF_IMAGE_SIZE = os.getenv(
    "H3_REF_IMAGE_SIZE",
    "match",
)


# ============================================================
# TURBO
# ============================================================

TURBO_ENABLED = (
    os.getenv(
        "H3_ENABLE_TURBO",
        "0",
    )
    == "1"
)

TURBO_STEPS = int(
    os.getenv(
        "H3_TURBO_STEPS",
        "8",
    )
)


# ============================================================
# UPSCALE
# ============================================================

UPSCALE_ENABLED = (
    os.getenv(
        "H3_ENABLE_UPSCALE",
        "1",
    )
    == "1"
)

UPSCALE_WIDTH = int(
    os.getenv(
        "H3_UPSCALE_WIDTH",
        "1280",
    )
)

UPSCALE_HEIGHT = int(
    os.getenv(
        "H3_UPSCALE_HEIGHT",
        "720",
    )
)

UPSCALE_ENABLE_TEMPORAL_CHUNKING = (
    os.getenv(
        "H3_UPSCALE_TEMPORAL_CHUNKING",
        "1",
    )
    == "1"
)


# ============================================================
# WORKFLOW MODES
# ============================================================

WORKFLOW_AUTO = "auto"
WORKFLOW_REF2V = "ref2v"
WORKFLOW_TURBO_REF2V = "turbo_ref2v"
WORKFLOW_UPSCALE = "upscale"


ALL_WORKFLOW_MODES = {
    WORKFLOW_AUTO,
    WORKFLOW_REF2V,
    WORKFLOW_TURBO_REF2V,
    WORKFLOW_UPSCALE,
}


# ============================================================
# STORY MODES
# ============================================================

AI_STORY_MODE = "ai_story"

PRESERVE_USER_STORY_MODE = (
    "preserve_user_story"
)

EXPAND_USER_STORY_MODE = (
    "expand_user_story"
)

VALID_STORY_MODES = {
    AI_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    EXPAND_USER_STORY_MODE,
}


def ensure_directories() -> None:
    for directory in (
        STORIES_DIR,
        CHARACTERS_DIR,
        GENERATED_CHARACTERS_DIR,
        SCENES_DIR,
        SHOTS_DIR,
        PRODUCTION_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

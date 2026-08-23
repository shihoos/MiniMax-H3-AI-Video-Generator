from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ============================================================
# QWEN PLANNER
# ============================================================

QWEN_MODEL_ID = os.getenv(
    "QWEN_MODEL_ID",
    "Qwen/Qwen3-4B-Instruct-2507",
)

QWEN_KAGGLE_PATH = Path(
    os.getenv(
        "QWEN_KAGGLE_PATH",
        "/kaggle/input/qwen3-4b-instruct-2507",
    )
)

QWEN_LOCAL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-4b"
)

QWEN_MAX_NEW_TOKENS = int(
    os.getenv(
        "QWEN_MAX_NEW_TOKENS",
        "4096",
    )
)

QWEN_STORY_TEMPERATURE = float(
    os.getenv(
        "QWEN_STORY_TEMPERATURE",
        "0.7",
    )
)

QWEN_PRESERVE_STORY_TEMPERATURE = float(
    os.getenv(
        "QWEN_PRESERVE_STORY_TEMPERATURE",
        "0.15",
    )
)

QWEN_CHARACTER_DETECTION_TEMPERATURE = float(
    os.getenv(
        "QWEN_CHARACTER_DETECTION_TEMPERATURE",
        "0.1",
    )
)

QWEN_CHARACTER_PLAN_TEMPERATURE = float(
    os.getenv(
        "QWEN_CHARACTER_PLAN_TEMPERATURE",
        "0.2",
    )
)

QWEN_SCENE_PLAN_TEMPERATURE = float(
    os.getenv(
        "QWEN_SCENE_PLAN_TEMPERATURE",
        "0.2",
    )
)

QWEN_SHOT_PLAN_TEMPERATURE = float(
    os.getenv(
        "QWEN_SHOT_PLAN_TEMPERATURE",
        "0.2",
    )
)

QWEN_TOP_P = float(
    os.getenv(
        "QWEN_TOP_P",
        "0.8",
    )
)


QWEN_PROMPTS_DIR = (
    PROJECT_ROOT
    / "prompts"
    / "qwen"
)

SYSTEM_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "system.txt"
)

CREATE_STORY_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "create_story.txt"
)

PRESERVE_STORY_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "preserve_story.txt"
)

EXPAND_STORY_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "expand_story.txt"
)

CHARACTER_PLAN_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "character_plan.txt"
)

SCENE_PLAN_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "scene_plan.txt"
)

SHOT_PLAN_PROMPT_PATH = (
    QWEN_PROMPTS_DIR
    / "shot_plan.txt"
)


# ============================================================
# DATA
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

STORIES_DIR = DATA_DIR / "stories"
CHARACTERS_DIR = DATA_DIR / "characters"
GENERATED_CHARACTERS_DIR = (
    CHARACTERS_DIR / "generated"
)
SCENES_DIR = DATA_DIR / "scenes"
SHOTS_DIR = DATA_DIR / "shots"
PRODUCTION_DIR = DATA_DIR / "production"


# ============================================================
# H3 BASE PROFILE
# ============================================================

H3_FPS = int(
    os.getenv(
        "H3_FPS",
        "24",
    )
)

# 768-short-edge 16:9 target.
# T4 fallback is controlled separately.
H3_WIDTH = int(
    os.getenv(
        "H3_WIDTH",
        "1344",
    )
)

H3_HEIGHT = int(
    os.getenv(
        "H3_HEIGHT",
        "768",
    )
)

H3_SAFE_WIDTH = int(
    os.getenv(
        "H3_SAFE_WIDTH",
        "960",
    )
)

H3_SAFE_HEIGHT = int(
    os.getenv(
        "H3_SAFE_HEIGHT",
        "544",
    )
)

# H3's trained frame grid.
# 243 = approximately 10.1 seconds at 24 FPS.
H3_FRAMES_PER_SHOT = int(
    os.getenv(
        "H3_FRAMES_PER_SHOT",
        "243",
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
# BASE Q4 MODEL FILES
# ============================================================

H3_REF2VA_MODEL = os.getenv(
    "H3_REF2VA_MODEL",
    "minimax_h3_ref2va_pruned-Q4_K_M.gguf",
)

H3_FL2VA_MODEL = os.getenv(
    "H3_FL2VA_MODEL",
    "minimax_h3_fl2va_pruned-Q4_K_M.gguf",
)

H3_TEXT_ENCODER = os.getenv(
    "H3_TEXT_ENCODER",
    "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
)

H3_VIDEO_VAE = os.getenv(
    "H3_VIDEO_VAE",
    "minimax_h3_video_vae_fp16.safetensors",
)

H3_AUDIO_VAE = os.getenv(
    "H3_AUDIO_VAE",
    "minimax_h3_audio_vae_fp32.safetensors",
)


# ============================================================
# TURBO PROFILE
# ============================================================

TURBO_ENABLED = (
    os.getenv(
        "H3_ENABLE_TURBO",
        "0",
    )
    == "1"
)

TURBO_REF2VA_MODEL = os.getenv(
    "H3_TURBO_REF2VA_MODEL",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
)

TURBO_FL2VA_MODEL = os.getenv(
    "H3_TURBO_FL2VA_MODEL",
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
)

TURBO_TEXT_ENCODER = os.getenv(
    "H3_TURBO_TEXT_ENCODER",
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
)

TURBO_REF2V_LORA = os.getenv(
    "H3_TURBO_REF2V_LORA",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
)

TURBO_FL2V_LORA = os.getenv(
    "H3_TURBO_FL2V_LORA",
    "minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
)

TURBO_REF2V_STEPS = int(
    os.getenv(
        "H3_TURBO_REF2V_STEPS",
        "4",
    )
)

TURBO_FL2V_STEPS = int(
    os.getenv(
        "H3_TURBO_FL2V_STEPS",
        "8",
    )
)


# ============================================================
# OFFICIAL 2K API
# ============================================================

H3_REGENERATE_2K_ENABLED = (
    os.getenv(
        "H3_REGENERATE_2K_ENABLED",
        "0",
    )
    == "1"
)

H3_API_BASE = os.getenv(
    "MINIMAX_API_BASE",
    "https://api.minimax.io",
)

H3_API_KEY = os.getenv(
    "MINIMAX_API_KEY",
    "",
)

H3_REGENERATE_ENDPOINT = os.getenv(
    "H3_REGENERATE_ENDPOINT",
    "/v2/video_regeneration",
)

H3_QUERY_ENDPOINT = os.getenv(
    "H3_QUERY_ENDPOINT",
    "/v2/query/video_generation",
)


# ============================================================
# FINAL DELIVERY
# ============================================================

FINAL_WIDTH = int(
    os.getenv(
        "FINAL_WIDTH",
        "1280",
    )
)

FINAL_HEIGHT = int(
    os.getenv(
        "FINAL_HEIGHT",
        "720",
    )
)


# ============================================================
# WORKFLOW MODES
# ============================================================

WORKFLOW_AUTO = "auto"

WORKFLOW_HARD_R2V = "hard_r2v"
WORKFLOW_HARD_CHAINED = "hard_chained"

WORKFLOW_SEAMLESS_V2 = "seamless_v2"
WORKFLOW_SEAMLESS_CORE = "seamless_core"

WORKFLOW_KEYFRAMES = "keyframes"
WORKFLOW_EXTEND_TAKE = "extend_take"

WORKFLOW_TURBO_I2V = "turbo_i2v"
WORKFLOW_TURBO_REF2V = "turbo_ref2v"
WORKFLOW_TURBO_T2V = "turbo_t2v"


ALL_WORKFLOW_MODES = {
    WORKFLOW_AUTO,
    WORKFLOW_HARD_R2V,
    WORKFLOW_HARD_CHAINED,
    WORKFLOW_SEAMLESS_V2,
    WORKFLOW_SEAMLESS_CORE,
    WORKFLOW_KEYFRAMES,
    WORKFLOW_EXTEND_TAKE,
    WORKFLOW_TURBO_I2V,
    WORKFLOW_TURBO_REF2V,
    WORKFLOW_TURBO_T2V,
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


def ensure_directories():

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

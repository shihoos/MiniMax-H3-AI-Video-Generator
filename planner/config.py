from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

STORIES_DIR = DATA_DIR / "stories"
CHARACTERS_DIR = DATA_DIR / "characters"
GENERATED_CHARACTERS_DIR = CHARACTERS_DIR / "generated"
SCENES_DIR = DATA_DIR / "scenes"
SHOTS_DIR = DATA_DIR / "shots"
PRODUCTION_DIR = DATA_DIR / "production"

CONTINUITY_DIR = (
    PRODUCTION_DIR / "continuity"
)

IDENTITY_DIR = (
    PRODUCTION_DIR / "identity"
)


def ensure_directories() -> None:
    for directory in (
        STORIES_DIR,
        CHARACTERS_DIR,
        GENERATED_CHARACTERS_DIR,
        SCENES_DIR,
        SHOTS_DIR,
        PRODUCTION_DIR,
        CONTINUITY_DIR,
        IDENTITY_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# LOCKED MODEL INVENTORY
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

H3_FPS = 24

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

H3_FRAMES_PER_SHOT = int(
    os.getenv(
        "H3_FRAMES_PER_SHOT",
        "124",
    )
)


# ============================================================
# NORMAL REF2V
# ============================================================

H3_STEPS = 20


# ============================================================
# REFERENCE LIMITS
# ============================================================

H3_MAX_REFERENCE_IMAGES = 9
H3_MAX_REFERENCE_VIDEOS = 3
H3_MAX_REFERENCE_AUDIO = 3
H3_MAX_REFERENCE_FILES = 12


# ============================================================
# TURBO
# ============================================================

TURBO_STEPS = 8


# ============================================================
# INTERNAL UPSCALE / REGENERATION TARGET
# ============================================================

# Important:
# H3 generation remains 1344×768.
# The MMH3 3D latent upscaler + Ultimate Upscale
# refine internally at 1920×1080.
UPSCALE_WIDTH = int(
    os.getenv(
        "H3_UPSCALE_WIDTH",
        "1920",
    )
)

UPSCALE_HEIGHT = int(
    os.getenv(
        "H3_UPSCALE_HEIGHT",
        "1080",
    )
)


# ============================================================
# FINAL DELIVERY
# ============================================================

DELIVERY_WIDTH = 1280
DELIVERY_HEIGHT = 720
DELIVERY_FPS = 24


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
# PROFILES
# ============================================================

PROFILE_BASE = "base"
PROFILE_TURBO = "turbo"
PROFILE_UPSCALE = "upscale"

ALL_PROFILES = {
    PROFILE_BASE,
    PROFILE_TURBO,
    PROFILE_UPSCALE,
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


# ============================================================
# SINGLE LOCAL QWEN INVENTORY
# ============================================================

# H3-specific Qwen3-VL conditioning encoder.
#
# It is used through the H3 multimodal workflow:
# text + reference images + reference videos + audio context.
#
# No external Qwen API.
# No second Qwen checkpoint.
PLANNER_MODEL = H3_TEXT_ENCODER

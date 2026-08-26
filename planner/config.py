from __future__ import annotations

import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

CONFIG_ROOT = (
    PROJECT_ROOT / "configs"
)

RUNTIME_CONFIG_PATH = (
    CONFIG_ROOT / "runtime_versions.yaml"
)


def _load_runtime_config() -> dict:
    if not RUNTIME_CONFIG_PATH.is_file():
        raise RuntimeError(
            f"Missing runtime configuration: "
            f"{RUNTIME_CONFIG_PATH}"
        )

    with RUNTIME_CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(
            handle
        )

    if not isinstance(data, dict):
        raise RuntimeError(
            "runtime_versions.yaml must contain a mapping."
        )

    return data


RUNTIME = _load_runtime_config()


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = (
    PROJECT_ROOT / "data"
)

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
# MODEL INVENTORY
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

DIRECTOR_MODEL_FILENAME = (
    RUNTIME[
        "director"
    ][
        "model_filename"
    ]
)


# ============================================================
# DIRECTOR MODEL
# ============================================================

DIRECTOR_MODEL_ENV = (
    "H3_DIRECTOR_MODEL_PATH"
)

DIRECTOR_ENABLED_ENV = (
    "H3_DIRECTOR_ENABLED"
)

DIRECTOR_N_CTX = int(
    os.getenv(
        "H3_DIRECTOR_N_CTX",
        str(
            RUNTIME[
                "director"
            ][
                "context"
            ]
        ),
    )
)

DIRECTOR_N_GPU_LAYERS = int(
    os.getenv(
        "H3_DIRECTOR_N_GPU_LAYERS",
        str(
            RUNTIME[
                "director"
            ][
                "gpu_layers"
            ]
        ),
    )
)

DIRECTOR_N_BATCH = int(
    os.getenv(
        "H3_DIRECTOR_N_BATCH",
        str(
            RUNTIME[
                "director"
            ][
                "batch"
            ]
        ),
    )
)

DIRECTOR_MAX_TOKENS = int(
    os.getenv(
        "H3_DIRECTOR_MAX_TOKENS",
        str(
            RUNTIME[
                "director"
            ][
                "max_tokens"
            ]
        ),
    )
)

DIRECTOR_TEMPERATURE = float(
    os.getenv(
        "H3_DIRECTOR_TEMPERATURE",
        str(
            RUNTIME[
                "director"
            ][
                "temperature"
            ]
        ),
    )
)

DIRECTOR_TOP_P = float(
    os.getenv(
        "H3_DIRECTOR_TOP_P",
        str(
            RUNTIME[
                "director"
            ][
                "top_p"
            ]
        ),
    )
)

DIRECTOR_THREADS = int(
    os.getenv(
        "H3_DIRECTOR_THREADS",
        str(
            RUNTIME[
                "director"
            ][
                "threads"
            ]
        ),
    )
)

DIRECTOR_MAX_PLAN_CHARS = 50000

DIRECTOR_KAGGLE_INPUT_ROOT = Path(
    "/kaggle/input"
)


def director_enabled() -> bool:

    value = os.getenv(
        DIRECTOR_ENABLED_ENV,
        "1",
    ).strip().lower()

    return value not in {
        "0",
        "false",
        "no",
        "off",
    }


# ============================================================
# GENERATION
# ============================================================

H3_FPS = int(
    RUNTIME[
        "generation"
    ][
        "fps"
    ]
)

H3_WIDTH = int(
    os.getenv(
        "H3_WIDTH",
        str(
            RUNTIME[
                "generation"
            ][
                "width"
            ]
        ),
    )
)

H3_HEIGHT = int(
    os.getenv(
        "H3_HEIGHT",
        str(
            RUNTIME[
                "generation"
            ][
                "height"
            ]
        ),
    )
)

H3_FRAMES_PER_SHOT = int(
    os.getenv(
        "H3_FRAMES_PER_SHOT",
        str(
            RUNTIME[
                "generation"
            ][
                "frames_per_shot"
            ]
        ),
    )
)

H3_STEPS = int(
    RUNTIME[
        "generation"
    ][
        "normal_steps"
    ]
)

TURBO_STEPS = int(
    RUNTIME[
        "generation"
    ][
        "turbo_steps"
    ]
)

H3_REF_IMAGE_SIZE = str(
    RUNTIME[
        "generation"
    ][
        "ref_image_size"
    ]
)


# ============================================================
# REFERENCES
# ============================================================

H3_MAX_REFERENCE_IMAGES = 9
H3_MAX_REFERENCE_VIDEOS = 3
H3_MAX_REFERENCE_AUDIO = 3
H3_MAX_REFERENCE_FILES = 12


# ============================================================
# UPSCALE
# ============================================================

UPSCALE_WIDTH = int(
    os.getenv(
        "H3_UPSCALE_WIDTH",
        str(
            RUNTIME[
                "upscale"
            ][
                "width"
            ]
        ),
    )
)

UPSCALE_HEIGHT = int(
    os.getenv(
        "H3_UPSCALE_HEIGHT",
        str(
            RUNTIME[
                "upscale"
            ][
                "height"
            ]
        ),
    )
)


# ============================================================
# FINAL DELIVERY
# ============================================================

DELIVERY_WIDTH = int(
    RUNTIME[
        "delivery"
    ][
        "width"
    ]
)

DELIVERY_HEIGHT = int(
    RUNTIME[
        "delivery"
    ][
        "height"
    ]
)

DELIVERY_FPS = int(
    RUNTIME[
        "delivery"
    ][
        "fps"
    ]
)


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

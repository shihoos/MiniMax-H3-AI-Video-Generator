from __future__ import annotations

import os
from pathlib import Path

import yaml


PROJECT_ROOT = (
    Path(__file__)
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
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise RuntimeError(
            "runtime_versions.yaml must contain a mapping."
        )

    return data


RUNTIME = _load_runtime_config()


# ============================================================
# DIRECTORIES
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
STORIES_DIR = DATA_DIR / "stories"
CHARACTERS_DIR = DATA_DIR / "characters"
GENERATED_CHARACTERS_DIR = CHARACTERS_DIR / "generated"
SCENES_DIR = DATA_DIR / "scenes"
SHOTS_DIR = DATA_DIR / "shots"
PRODUCTION_DIR = DATA_DIR / "production"
CONTINUITY_DIR = PRODUCTION_DIR / "continuity"
IDENTITY_DIR = PRODUCTION_DIR / "identity"


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

DIRECTOR_MODEL_FILENAME = str(
    RUNTIME["director"]["model_filename"]
).strip()

if not DIRECTOR_MODEL_FILENAME:
    raise RuntimeError(
        "runtime_versions.yaml director.model_filename is empty."
    )


# ============================================================
# DIRECTOR MODEL
# ============================================================

DIRECTOR_MODEL_ENV = "H3_DIRECTOR_MODEL_PATH"
DIRECTOR_ENABLED_ENV = "H3_DIRECTOR_ENABLED"

DIRECTOR_N_CTX = int(
    os.getenv(
        "H3_DIRECTOR_N_CTX",
        str(RUNTIME["director"]["context"]),
    )
)

DIRECTOR_N_GPU_LAYERS = int(
    os.getenv(
        "H3_DIRECTOR_N_GPU_LAYERS",
        str(RUNTIME["director"]["gpu_layers"]),
    )
)

DIRECTOR_N_BATCH = int(
    os.getenv(
        "H3_DIRECTOR_N_BATCH",
        str(RUNTIME["director"]["batch"]),
    )
)

DIRECTOR_MAX_TOKENS = int(
    os.getenv(
        "H3_DIRECTOR_MAX_TOKENS",
        str(RUNTIME["director"]["max_tokens"]),
    )
)

DIRECTOR_TEMPERATURE = float(
    os.getenv(
        "H3_DIRECTOR_TEMPERATURE",
        str(RUNTIME["director"]["temperature"]),
    )
)

DIRECTOR_TOP_P = float(
    os.getenv(
        "H3_DIRECTOR_TOP_P",
        str(RUNTIME["director"]["top_p"]),
    )
)

DIRECTOR_THREADS = int(
    os.getenv(
        "H3_DIRECTOR_THREADS",
        str(RUNTIME["director"]["threads"]),
    )
)

_configured_input_root = os.getenv("H3_INPUT_ROOT", "").strip()
if _configured_input_root:
    DIRECTOR_KAGGLE_INPUT_ROOT = Path(_configured_input_root).expanduser().resolve()
elif Path("/kaggle/input").is_dir():
    DIRECTOR_KAGGLE_INPUT_ROOT = Path("/kaggle/input").resolve()
else:
    DIRECTOR_KAGGLE_INPUT_ROOT = (
        Path(__file__).resolve().parents[1] / "input"
    ).resolve()


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

H3_FPS = int(RUNTIME["generation"]["fps"])

H3_WIDTH = int(
    os.getenv(
        "H3_WIDTH",
        str(RUNTIME["generation"]["width"]),
    )
)

H3_HEIGHT = int(
    os.getenv(
        "H3_HEIGHT",
        str(RUNTIME["generation"]["height"]),
    )
)

H3_FRAMES_PER_SHOT = int(
    os.getenv(
        "H3_FRAMES_PER_SHOT",
        str(RUNTIME["generation"]["frames_per_shot"]),
    )
)

H3_STEPS = int(RUNTIME["generation"]["normal_steps"])
TURBO_STEPS = int(RUNTIME["generation"]["turbo_steps"])
H3_REF_IMAGE_SIZE = str(
    RUNTIME["generation"]["ref_image_size"]
).strip()


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
        str(RUNTIME["upscale"]["width"]),
    )
)

UPSCALE_HEIGHT = int(
    os.getenv(
        "H3_UPSCALE_HEIGHT",
        str(RUNTIME["upscale"]["height"]),
    )
)


# ============================================================
# FINAL DELIVERY
# ============================================================

DELIVERY_WIDTH = int(RUNTIME["delivery"]["width"])
DELIVERY_HEIGHT = int(RUNTIME["delivery"]["height"])
DELIVERY_FPS = int(RUNTIME["delivery"]["fps"])


# ============================================================
# STORYBOARD SERVER
# ============================================================

STORYBOARD_HOST = str(
    os.getenv(
        "H3_STORYBOARD_HOST",
        str(RUNTIME["storyboard"]["host"]),
    )
).strip()

STORYBOARD_PORT = int(
    os.getenv(
        "H3_STORYBOARD_PORT",
        str(RUNTIME["storyboard"]["port"]),
    )
)

GRADIO_SHARE_ENV = "H3_GRADIO_SHARE"


def storyboard_share_enabled() -> bool:
    value = os.getenv(
        GRADIO_SHARE_ENV,
        "1",
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


# Optional quality/UX features. Explicit environment variables override
# manifest defaults; no Kaggle-specific detection is used.
def feature_enabled(name: str, default: bool = False) -> bool:
    env_name = "H3_" + str(name).upper()
    value = os.getenv(env_name)
    if value is None:
        features = RUNTIME.get("features", {})
        configured = features.get(name)
        return bool(default if configured is None else configured)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


H3_LIVE_PREVIEW = feature_enabled("live_preview", True)
H3_CONTEXT_IR_VERSION = int(os.getenv("H3_CONTEXT_IR_VERSION", str(RUNTIME.get("features", {}).get("context_ir_version", 2))))
H3_VLM_ENABLED = feature_enabled("vlm_enabled", True)
H3_QA_ENABLED = feature_enabled("qa_enabled", True)
H3_SELECTIVE_RETAKE = feature_enabled("selective_retake", True)
H3_DIRECTOR_CRITIC = feature_enabled("director_critic", True)
H3_VLM_REFERENCE_ANALYSIS = feature_enabled("vlm_reference_analysis", True)
H3_VLM_VISUAL_QA = feature_enabled("vlm_visual_qa", True)
H3_AUTO_RETAKE = feature_enabled("auto_retake", True)
H3_MAX_AUTO_RETAKES_PER_SHOT = max(0, int(os.getenv("H3_MAX_AUTO_RETAKES_PER_SHOT", str(RUNTIME.get("features", {}).get("max_auto_retakes_per_shot", 1)))))


# ============================================================
# BASIC RUNTIME VALIDATION
# ============================================================

def validate_runtime_values() -> None:
    positive = (
        ("H3_FPS", H3_FPS),
        ("H3_WIDTH", H3_WIDTH),
        ("H3_HEIGHT", H3_HEIGHT),
        ("H3_FRAMES_PER_SHOT", H3_FRAMES_PER_SHOT),
        ("H3_STEPS", H3_STEPS),
        ("TURBO_STEPS", TURBO_STEPS),
        ("UPSCALE_WIDTH", UPSCALE_WIDTH),
        ("UPSCALE_HEIGHT", UPSCALE_HEIGHT),
        ("DELIVERY_WIDTH", DELIVERY_WIDTH),
        ("DELIVERY_HEIGHT", DELIVERY_HEIGHT),
        ("DELIVERY_FPS", DELIVERY_FPS),
        ("DIRECTOR_N_CTX", DIRECTOR_N_CTX),
        ("DIRECTOR_N_BATCH", DIRECTOR_N_BATCH),
        ("DIRECTOR_MAX_TOKENS", DIRECTOR_MAX_TOKENS),
        ("DIRECTOR_THREADS", DIRECTOR_THREADS),
        ("STORYBOARD_PORT", STORYBOARD_PORT),
    )

    invalid = [
        f"{name}={value}"
        for name, value in positive
        if value <= 0
    ]

    if invalid:
        raise RuntimeError(
            "Runtime configuration contains non-positive values: "
            + ", ".join(invalid)
        )

    if not 0.0 <= DIRECTOR_TOP_P <= 1.0:
        raise RuntimeError(
            f"DIRECTOR_TOP_P must be between 0 and 1: {DIRECTOR_TOP_P}"
        )

    if DIRECTOR_TEMPERATURE < 0.0:
        raise RuntimeError(
            f"DIRECTOR_TEMPERATURE must be >= 0: {DIRECTOR_TEMPERATURE}"
        )

    for name, width, height in (
        ("generation", H3_WIDTH, H3_HEIGHT),
        ("upscale", UPSCALE_WIDTH, UPSCALE_HEIGHT),
        ("delivery", DELIVERY_WIDTH, DELIVERY_HEIGHT),
    ):
        if width / height <= 0:
            raise RuntimeError(
                f"{name} dimensions are invalid: {width}x{height}"
            )

    if STORYBOARD_HOST == "":
        raise RuntimeError(
            "Storyboard host cannot be empty."
        )

    comfy_startup = float(os.getenv("H3_COMFY_STARTUP_TIMEOUT", "300"))
    comfy_job = float(os.getenv("H3_COMFY_JOB_TIMEOUT", "14400"))
    comfy_poll = float(os.getenv("H3_COMFY_POLL_INTERVAL", "2"))
    comfy_retries = int(os.getenv("H3_COMFY_REQUEST_RETRIES", "3"))
    if comfy_startup <= 0 or comfy_job <= 0 or comfy_poll <= 0:
        raise RuntimeError("ComfyUI timeouts/intervals must be positive.")
    if comfy_retries < 0:
        raise RuntimeError("H3_COMFY_REQUEST_RETRIES must be >= 0.")


validate_runtime_values()


# ============================================================
# STORY MODES
# ============================================================

AI_STORY_MODE = "ai_story"
PRESERVE_USER_STORY_MODE = "preserve_user_story"
EXPAND_USER_STORY_MODE = "expand_user_story"

VALID_STORY_MODES = {
    AI_STORY_MODE,
    PRESERVE_USER_STORY_MODE,
    EXPAND_USER_STORY_MODE,
}


# ============================================================
# WORKFLOW MODES
# ============================================================

WORKFLOW_AUTO = "auto"
WORKFLOW_REF2VA = "ref2va"
WORKFLOW_TURBO_REF2VA = "turbo_ref2va"
WORKFLOW_UPSCALE = "upscale"

ALL_WORKFLOW_MODES = {
    WORKFLOW_AUTO,
    WORKFLOW_REF2VA,
    WORKFLOW_TURBO_REF2VA,
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

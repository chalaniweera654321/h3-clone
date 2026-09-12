from __future__ import annotations

"""
MiniMax H3 + ComfyUI + Gradio app for Modal notebooks.

Expected model volume mount:
    /mnt/minimax-h3-models

The model volume can contain either:
    /mnt/minimax-h3-models/
        diffusion_models/
        text_encoders/
        vae/
        loras/
or the same directories nested below another folder. The scanner searches
recursively, so the exact top-level layout is not important.

The UI is intentionally inspired by the supplied Krea 2 app:
- model selector
- prompt
- optional first/last frame
- resolution
- duration
- sampling controls
- dynamic local LoRA controls
- detailed startup/generation logging

The supplied MiniMax H3 workflow identifies these required model files:
- minimax_h3_fl2va_pruned_int8_convrot.safetensors
- qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
- minimax_h3_video_vae_fp16.safetensors
- minimax_h3_audio_vae_fp32.safetensors

H3's supplied workflow uses:
- MiniMaxH3ImageToVideo
- KSamplerSelect: res_multistep
- BasicScheduler: simple / 20 / 1
- SamplerCustomAdvanced
- VAEDecode
- VAEDecodeAudio
- CreateVideo: 24 fps / 8-bit
- SaveVideo
"""

import asyncio
import glob
import json
import os
import pathlib
import random
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from typing import Any

import gradio as gr
from PIL import Image


# ============================================================================
# CONFIG
# ============================================================================

APP_NAME = "MiniMax H3 Studio"

ROOT = pathlib.Path(__file__).resolve().parent

# Modal Volume should normally be mounted here.
MODELS = pathlib.Path(
    os.environ.get("MINIMAX_H3_MODELS", "/mnt/minimax-h3-models")
)

COMFY = pathlib.Path(
    os.environ.get("COMFY_ROOT", "/root/ComfyUI")
)

INPUT = COMFY / "input"
OUTPUT = COMFY / "output"
CUSTOM_NODES = COMFY / "custom_nodes"

DIFFUSION_DIR = MODELS / "diffusion_models"
TEXT_ENCODER_DIR = MODELS / "text_encoders"
VAE_DIR = MODELS / "vae"
LORA_ROOT = MODELS / "loras"

# H3 files from the supplied workflow.
DEFAULT_DIFFUSION = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

MAX_WIDTH = 1344
MAX_HEIGHT = 768
MIN_WIDTH = 256
MIN_HEIGHT = 256

DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 768
DEFAULT_DURATION = 5.0
DEFAULT_STEPS = 20
DEFAULT_SEED = 1

SAMPLERS = [
    "res_multistep",
]

SCHEDULERS = [
    "simple",
]

# H3 workflow uses 24 fps and rounds duration to a valid 17-frame block grid:
# max(5, round(seconds * 24)) + (5 - (frames % 17)) % 17
FPS = 24

# A single process should execute one heavy Comfy graph at a time.
_GENERATION_LOCK = threading.Lock()

_COMFY_READY = False
_NODES_READY = False

LOCAL_MODELS: list[str] = []
LOCAL_LORAS: list[str] = []


# ============================================================================
# LOGGING
# ============================================================================

def log(message: str) -> None:
    print(
        f"[{time.strftime('%H:%M:%S')}] {message}",
        flush=True,
    )


def section(title: str) -> None:
    log("")
    log("=" * 78)
    log(title)
    log("=" * 78)


# ============================================================================
# BASIC COMMAND HELPERS
# ============================================================================

def run_cmd(
    command: list[str],
    cwd: pathlib.Path | None = None,
    check: bool = True,
) -> None:
    log("$ " + " ".join(command))
    import subprocess

    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
    )


def pip_install(packages: list[str]) -> None:
    if not packages:
        return

    run_cmd(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            *packages,
        ],
        check=False,
    )


# ============================================================================
# MODEL PATH REGISTRATION
# ============================================================================

def ensure_directories() -> None:
    section("STEP 1/7 — Checking directories")

    for path in (
        COMFY,
        INPUT,
        OUTPUT,
        CUSTOM_NODES,
        MODELS,
        DIFFUSION_DIR,
        TEXT_ENCODER_DIR,
        VAE_DIR,
        LORA_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
        log(f"[dir] {path}")


def register_model_paths() -> None:
    """
    Tell ComfyUI to search the Modal Volume directly.

    We import folder_paths only after ensuring ComfyUI is on sys.path.
    """
    if str(COMFY) not in sys.path:
        sys.path.insert(0, str(COMFY))

    import folder_paths

    log(f"[models] registering diffusion_models -> {DIFFUSION_DIR}")
    folder_paths.add_model_folder_path(
        "diffusion_models",
        str(DIFFUSION_DIR),
    )

    log(f"[models] registering text_encoders -> {TEXT_ENCODER_DIR}")
    folder_paths.add_model_folder_path(
        "text_encoders",
        str(TEXT_ENCODER_DIR),
    )

    log(f"[models] registering vae -> {VAE_DIR}")
    folder_paths.add_model_folder_path(
        "vae",
        str(VAE_DIR),
    )

    log(f"[models] registering loras -> {LORA_ROOT}")
    folder_paths.add_model_folder_path(
        "loras",
        str(LORA_ROOT),
    )


# ============================================================================
# MODEL SCANNER
# ============================================================================

MODEL_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".bin",
}


def _relative_model_path(
    path: pathlib.Path,
    root: pathlib.Path,
) -> str:
    return pathlib.PurePosixPath(
        *path.relative_to(root).parts
    ).as_posix()


def scan_models() -> None:
    global LOCAL_MODELS, LOCAL_LORAS

    section("STEP 2/7 — Scanning Modal model volume")

    LOCAL_MODELS = []
    LOCAL_LORAS = []

    # Scan diffusion directory.
    if DIFFUSION_DIR.exists():
        for path in DIFFUSION_DIR.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in MODEL_EXTENSIONS
            ):
                LOCAL_MODELS.append(
                    _relative_model_path(path, DIFFUSION_DIR)
                )

    # If the expected subdirectory is empty, search the entire volume and
    # display useful diagnostics. We still prefer the standard Comfy layout.
    if not LOCAL_MODELS and MODELS.exists():
        log(
            "[models] diffusion_models is empty; "
            "performing recursive fallback scan"
        )

        for path in MODELS.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in MODEL_EXTENSIONS:
                continue

            if "loras" in path.parts:
                continue

            name = path.name.lower()
            if (
                "minimax" in name
                or "qwen" in name
            ):
                LOCAL_MODELS.append(str(path))

    if LORA_ROOT.exists():
        for path in LORA_ROOT.rglob("*"):
            if (
                path.is_file()
                and path.suffix.lower() in MODEL_EXTENSIONS
            ):
                LOCAL_LORAS.append(
                    _relative_model_path(path, LORA_ROOT)
                )

    LOCAL_MODELS = sorted(
        set(LOCAL_MODELS),
        key=str.lower,
    )

    LOCAL_LORAS = sorted(
        set(LOCAL_LORAS),
        key=str.lower,
    )

    log(f"[models] diffusion models found: {len(LOCAL_MODELS)}")
    for item in LOCAL_MODELS:
        log(f"[models]   {item}")

    log(f"[loras] LoRAs found: {len(LOCAL_LORAS)}")
    for item in LOCAL_LORAS:
        log(f"[loras]   {item}")

    log(f"[models] volume: {MODELS}")


# ============================================================================
# MODEL RESOLUTION
# ============================================================================

def _safe_relative_path(
    value: str,
) -> pathlib.PurePosixPath:
    normalized = str(value).replace("\\", "/").strip()
    path = pathlib.PurePosixPath(normalized)

    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("invalid model path")

    return path


def validate_model_name(
    model_name: str,
) -> str:
    path = _safe_relative_path(model_name)

    candidate = DIFFUSION_DIR.joinpath(*path.parts)

    if candidate.is_file():
        return path.as_posix()

    # Fallback: allow a discovered absolute path only when it is inside
    # the configured model volume. This is useful for unusual volume layouts.
    raw = pathlib.Path(model_name)

    if raw.is_absolute() and raw.is_file():
        try:
            raw.resolve().relative_to(MODELS.resolve())
            return str(raw)
        except ValueError:
            pass

    # Last chance: match by basename.
    matches = [
        p
        for p in MODELS.rglob(path.name)
        if p.is_file()
        and p.suffix.lower() in MODEL_EXTENSIONS
        and "loras" not in p.parts
    ]

    if len(matches) == 1:
        return str(matches[0])

    raise ValueError(
        "Diffusion model is not installed in the Modal volume: "
        + model_name
    )


def validate_lora_name(
    lora_name: str,
) -> str:
    path = _safe_relative_path(lora_name)

    candidate = LORA_ROOT.joinpath(*path.parts)

    if not candidate.is_file():
        raise ValueError(
            "LoRA is not installed in the Modal volume: "
            + lora_name
        )

    return path.as_posix()


def find_model_by_basename(
    filename: str,
) -> pathlib.Path | None:
    for root in (
        DIFFUSION_DIR,
        TEXT_ENCODER_DIR,
        VAE_DIR,
    ):
        candidate = root / filename
        if candidate.is_file():
            return candidate

    for candidate in MODELS.rglob(filename):
        if candidate.is_file():
            return candidate

    return None


def resolve_text_encoder() -> str:
    found = find_model_by_basename(DEFAULT_TEXT_ENCODER)
    if found is None:
        raise RuntimeError(
            f"Missing H3 text encoder: {DEFAULT_TEXT_ENCODER}\n"
            f"Expected somewhere under {MODELS}"
        )

    # The H3 node expects the filename known by Comfy's text_encoders path.
    try:
        return found.resolve().relative_to(
            TEXT_ENCODER_DIR.resolve()
        ).as_posix()
    except ValueError:
        return found.name


def resolve_video_vae() -> str:
    found = find_model_by_basename(DEFAULT_VIDEO_VAE)
    if found is None:
        raise RuntimeError(
            f"Missing H3 video VAE: {DEFAULT_VIDEO_VAE}\n"
            f"Expected somewhere under {MODELS}"
        )

    try:
        return found.resolve().relative_to(
            VAE_DIR.resolve()
        ).as_posix()
    except ValueError:
        return found.name


def resolve_audio_vae() -> str:
    found = find_model_by_basename(DEFAULT_AUDIO_VAE)
    if found is None:
        raise RuntimeError(
            f"Missing H3 audio VAE: {DEFAULT_AUDIO_VAE}\n"
            f"Expected somewhere under {MODELS}"
        )

    try:
        return found.resolve().relative_to(
            VAE_DIR.resolve()
        ).as_posix()
    except ValueError:
        return found.name


# ============================================================================
# COMFYUI SETUP
# ============================================================================

def install_comfy_requirements() -> None:
    section("STEP 3/7 — Checking ComfyUI")

    if not (COMFY / "main.py").is_file():
        raise RuntimeError(
            f"ComfyUI was not found at {COMFY}.\n"
            "Run the notebook clone step before starting this app."
        )

    requirements = COMFY / "requirements.txt"

    if not requirements.is_file():
        log("[comfy] requirements.txt not found; skipping")
        return

    # Do not replace Modal's CUDA/PyTorch stack accidentally.
    blocked = {
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "accelerate",
        "huggingface-hub",
    }

    filtered: list[str] = []

    for raw in requirements.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        item = raw.strip()

        if not item or item.startswith("#"):
            continue

        package = item.split(
            "==",
            1,
        )[0].split(
            ">=",
            1,
        )[0].split(
            "<=",
            1,
        )[0].split(
            "[",
            1,
        )[0].strip().lower().replace("_", "-")

        if package not in blocked:
            filtered.append(item)

    if filtered:
        log(
            f"[comfy] installing {len(filtered)} non-PyTorch requirements"
        )
        pip_install(filtered)
    else:
        log("[comfy] no additional requirements required")


def init_comfy_nodes() -> None:
    global _NODES_READY

    if _NODES_READY:
        return

    section("STEP 4/7 — Initializing ComfyUI nodes")

    comfy_path = str(COMFY)

    sys.path = [
        item
        for item in sys.path
        if item != comfy_path
    ]
    sys.path.insert(0, comfy_path)

    os.chdir(COMFY)

    # ComfyUI imports can depend on the current working directory.
    import execution
    import nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)

    # Initialize the same node registry used by ComfyUI.
    execution.PromptQueue(server_instance)

    loop.run_until_complete(
        nodes.init_extra_nodes()
    )

    _NODES_READY = True

    log("[comfy] node registry initialized")


def ensure_comfy() -> None:
    global _COMFY_READY

    if _COMFY_READY:
        return

    section("STEP 5/7 — Preparing H3 runtime")

    ensure_directories()
    install_comfy_requirements()

    register_model_paths()
    scan_models()

    # Make sure all required H3 assets are visible.
    resolve_text_encoder()
    resolve_video_vae()
    resolve_audio_vae()

    if not LOCAL_MODELS:
        raise RuntimeError(
            "No diffusion models were found. "
            f"Check {DIFFUSION_DIR} or {MODELS}."
        )

    init_comfy_nodes()

    _COMFY_READY = True

    log("[runtime] H3 runtime is READY")


# ============================================================================
# H3 DURATION
# ============================================================================

def duration_to_length(
    duration_seconds: float,
) -> int:
    """
    Matches the supplied workflow's Math Expression:

    max(5, round(a * 24))
      + (5 - (max(5, round(a * 24)) % 17)) % 17
    """
    seconds = max(
        0.2,
        min(
            15.0,
            float(duration_seconds),
        ),
    )

    frames = max(
        5,
        round(seconds * FPS),
    )

    length = frames + (
        5 - (frames % 17)
    ) % 17

    return int(length)


# ============================================================================
# IMAGE STAGING
# ============================================================================

def stage_image(
    image_path: str,
    prefix: str,
) -> str:
    if not image_path:
        raise ValueError("image path is empty")

    source_path = pathlib.Path(image_path)

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Uploaded image was not found: {image_path}"
        )

    with Image.open(source_path) as source:
        image = source.convert("RGB")

    filename = (
        f"{prefix}_{uuid.uuid4().hex[:12]}.png"
    )

    destination = INPUT / filename
    image.save(
        destination,
        format="PNG",
    )

    log(
        f"[input] staged {source_path.name} -> {destination.name}"
    )

    return filename


# ============================================================================
# H3 WORKFLOW
# ============================================================================

def ref(
    node_id: str,
    output: int = 0,
) -> list[Any]:
    return [
        node_id,
        output,
    ]


def inject_loras(
    workflow: dict[str, dict[str, Any]],
    loras: list[tuple[str, float]],
    *,
    model_source: list[Any],
    clip_source: list[Any],
    model_target: tuple[str, str],
    clip_target: tuple[str, str],
) -> None:
    """
    LoRA chain modeled after the supplied Krea 2 app.

    Each enabled LoRA becomes a LoraLoader:
        previous model + previous clip
                  ↓
             LoraLoader
                  ↓
        next model + next clip

    Zero-weight LoRAs are ignored.
    """
    if not loras:
        return

    previous_model = model_source
    previous_clip = clip_source

    for index, (filename, strength) in enumerate(loras):
        node_id = f"user_lora_{index}"

        workflow[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": previous_model,
                "clip": previous_clip,
                "lora_name": filename,
                "strength_model": float(strength),
                "strength_clip": float(strength),
            },
        }

        previous_model = ref(node_id, 0)
        previous_clip = ref(node_id, 1)

        log(
            f"[lora] enabled {filename} @ {float(strength):.2f}"
        )

    workflow[model_target[0]]["inputs"][model_target[1]] = previous_model
    workflow[clip_target[0]]["inputs"][clip_target[1]] = previous_clip


def build_h3_workflow(
    *,
    prompt: str,
    width: int,
    height: int,
    duration: float,
    seed: int,
    model_name: str,
    first_frame: str | None,
    last_frame: str | None,
    loras: list[tuple[str, float]],
) -> dict[str, dict[str, Any]]:
    model_name = validate_model_name(model_name)

    clip_name = resolve_text_encoder()
    video_vae = resolve_video_vae()
    audio_vae = resolve_audio_vae()

    length = duration_to_length(duration)

    # H3's supplied workflow uses these exact core nodes.
    workflow: dict[str, dict[str, Any]] = {
        # Model
        "6": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": (
                    pathlib.Path(model_name).name
                    if pathlib.Path(model_name).is_absolute()
                    else model_name
                ),
                "weight_dtype": "default",
            },
        },

        "13": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": "minimax",
                "device": "default",
            },
        },

        "11": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": video_vae,
            },
        },

        "24": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": audio_vae,
            },
        },

        # Sampling
        "17": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "res_multistep",
            },
        },

        "9": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ref("6"),
                "scheduler": "simple",
                "steps": int(DEFAULT_STEPS),
                "denoise": 1.0,
            },
        },

        "15": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": int(seed),
            },
        },

        # H3 conditioning + latent
        "104": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ref("13"),
                "vae": ref("11"),
                "first_frame": (
                    ref("114")
                    if first_frame
                    else None
                ),
                "last_frame": (
                    ref("115")
                    if last_frame
                    else None
                ),
                "prompt": prompt.strip(),
                "width": int(width),
                "height": int(height),
                "length": int(length),
            },
        },

        "16": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ref("6"),
                "conditioning": ref("104", 0),
            },
        },

        "14": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ref("15"),
                "guider": ref("16"),
                "sampler": ref("17"),
                "sigmas": ref("9"),
                "latent_image": ref("104", 1),
            },
        },

        # Decode
        "10": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ref("14", 0),
                "vae": ref("11"),
            },
        },

        "23": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ref("14", 0),
                "vae": ref("24"),
            },
        },

        "91": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ref("10"),
                "audio": ref("23"),
                "fps": 24,
                "bit_depth": 8,
            },
        },

        "92": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ref("91"),
                "filename_prefix": "MiniMax_H3",
                "format": "auto",
                "codec": "auto",
            },
        },
    }

    # Optional first frame.
    if first_frame:
        workflow["114"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": first_frame,
            },
        }

    # Optional last frame.
    if last_frame:
        workflow["115"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": last_frame,
            },
        }

    # Remove null optional inputs. Comfy's validation is happier when absent.
    h3_inputs = workflow["104"]["inputs"]

    if not first_frame:
        h3_inputs.pop("first_frame", None)

    if not last_frame:
        h3_inputs.pop("last_frame", None)

    # LoRA support follows the supplied Krea app's LoraLoader chain.
    # IMPORTANT: H3 itself is not guaranteed to support arbitrary SD/FLUX/
    # Krea LoRAs. Only use LoRAs trained for a compatible H3 checkpoint.
    if loras:
        inject_loras(
            workflow,
            loras,
            model_source=ref("6"),
            clip_source=ref("13"),
            model_target=("9", "model"),
            clip_target=("104", "clip"),
        )

        # The H3 conditioning node receives the CLIP output.
        # BasicGuider receives the LoRA-modified model.
        workflow["16"]["inputs"]["model"] = (
            workflow[
                f"user_lora_{len(loras) - 1}"
            ]
            and ref(
                f"user_lora_{len(loras) - 1}",
                0,
            )
        )

        workflow["104"]["inputs"]["clip"] = ref(
            f"user_lora_{len(loras) - 1}",
            1,
        )

        # BasicScheduler must also receive the LoRA-modified model.
        workflow["9"]["inputs"]["model"] = ref(
            f"user_lora_{len(loras) - 1}",
            0,
        )

    # The supplied workflow uses 20 steps by default.
    workflow["9"]["inputs"]["steps"] = int(
        CURRENT_STEPS.get()
        if CURRENT_STEPS is not None
        else DEFAULT_STEPS
    )

    return workflow


# A tiny holder lets build_h3_workflow remain independent from Gradio event
# objects. generate() sets it for the duration of a request.
class _StepHolder:
    value = DEFAULT_STEPS

    @classmethod
    def get(cls) -> int:
        return cls.value


CURRENT_STEPS = _StepHolder


# ============================================================================
# EXECUTION
# ============================================================================

def find_output_files(
    history_result: Any,
) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []

    outputs = (
        history_result.get("outputs", {})
        if isinstance(history_result, dict)
        else {}
    )

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("filename"):
                filename = str(value["filename"])
                subfolder = str(
                    value.get("subfolder", "")
                )

                item_type = str(
                    value.get("type", "output")
                )

                if item_type == "output":
                    base = OUTPUT
                else:
                    base = COMFY / item_type

                candidate = (
                    base
                    / subfolder
                    / filename
                )

                if candidate.exists():
                    paths.append(candidate)

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(outputs)

    # Also inspect standard Comfy output directories. This is important for
    # video nodes whose history structure can differ between Comfy versions.
    for pattern in (
        str(OUTPUT / "**" / "*.mp4"),
        str(OUTPUT / "**" / "*.webm"),
        str(OUTPUT / "**" / "*.mov"),
        str(OUTPUT / "**" / "*.mkv"),
        str(OUTPUT / "**" / "*.avi"),
    ):
        paths.extend(
            pathlib.Path(item)
            for item in glob.glob(
                pattern,
                recursive=True,
            )
        )

    unique: dict[str, pathlib.Path] = {}

    for path in paths:
        if path.is_file():
            unique[str(path.resolve())] = path

    return sorted(
        unique.values(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def execute_workflow(
    workflow: dict[str, dict[str, Any]],
) -> list[str]:
    import execution
    import server

    log("[execute] creating isolated Comfy executor")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_instance = server.PromptServer(loop)

    executor = execution.PromptExecutor(
        server_instance,
        cache_type=execution.CacheType.RAM_PRESSURE,
        cache_args={
            "lru": 0,
            "ram": 2.0,
            "ram_inactive": 8.0,
        },
    )

    prompt_id = str(uuid.uuid4())

    save_video_nodes = [
        node_id
        for node_id, node in workflow.items()
        if node.get("class_type") == "SaveVideo"
    ]

    execute_outputs = (
        save_video_nodes
        if save_video_nodes
        else None
    )

    log(
        f"[execute] prompt id: {prompt_id}"
    )

    executor.execute(
        workflow,
        prompt_id,
        extra_data={},
        execute_outputs=execute_outputs,
    )

    if not executor.success:
        messages = getattr(
            executor,
            "status_messages",
            [],
        )

        message = (
            messages[-1]
            if messages
            else "ComfyUI execution failed"
        )

        raise RuntimeError(str(message))

    paths = find_output_files(
        getattr(
            executor,
            "history_result",
            {},
        )
    )

    if not paths:
        raise RuntimeError(
            "ComfyUI finished, but no video file was found in output."
        )

    return [
        str(path)
        for path in paths
    ]


# ============================================================================
# VALIDATION
# ============================================================================

def normalize_dimensions(
    width: int,
    height: int,
) -> tuple[int, int]:
    width = max(
        MIN_WIDTH,
        min(
            MAX_WIDTH,
            int(width),
        ),
    )

    height = max(
        MIN_HEIGHT,
        min(
            MAX_HEIGHT,
            int(height),
        ),
    )

    # H3 workflow uses multiples of 32.
    width = max(
        32,
        (width // 32) * 32,
    )

    height = max(
        32,
        (height // 32) * 32,
    )

    return width, height


def validate_frames(
    first_frame: str | None,
    last_frame: str | None,
) -> None:
    if first_frame:
        if not pathlib.Path(first_frame).is_file():
            raise FileNotFoundError(
                f"First-frame image does not exist: {first_frame}"
            )

    if last_frame:
        if not pathlib.Path(last_frame).is_file():
            raise FileNotFoundError(
                f"Last-frame image does not exist: {last_frame}"
            )


# ============================================================================
# GENERATE
# ============================================================================

def generate(
    prompt: str,
    first_frame: str | None,
    last_frame: str | None,
    width: int,
    height: int,
    duration: float,
    steps: int,
    seed: int,
    randomize_seed: bool,
    base_model: str,
    lora_values: list[float],
    progress: gr.Progress = gr.Progress(track_tqdm=True),
):
    started = time.time()

    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")

    if not _GENERATION_LOCK.acquire(
        blocking=False
    ):
        raise gr.Error(
            "Another generation is already running. "
            "Please wait for it to finish."
        )

    staged: list[pathlib.Path] = []

    try:
        section("NEW GENERATION")

        progress(
            0.02,
            desc="Preparing H3 runtime",
        )

        ensure_comfy()

        if not LOCAL_MODELS:
            scan_models()

        if not LOCAL_MODELS:
            raise RuntimeError(
                "No diffusion model was found."
            )

        width, height = normalize_dimensions(
            width,
            height,
        )

        duration = max(
            0.2,
            min(
                15.0,
                float(duration),
            ),
        )

        steps = max(
            4,
            min(
                40,
                int(steps),
            ),
        )

        if randomize_seed or int(seed) < 0:
            effective_seed = random.randint(
                0,
                2**32 - 1,
            )
        else:
            effective_seed = int(seed)

        validate_frames(
            first_frame,
            last_frame,
        )

        first_name = None
        last_name = None

        if first_frame:
            first_name = stage_image(
                first_frame,
                "first",
            )
            staged.append(
                INPUT / first_name
            )

        if last_frame:
            last_name = stage_image(
                last_frame,
                "last",
            )
            staged.append(
                INPUT / last_name
            )

        enabled_loras: list[tuple[str, float]] = []

        for filename, weight in zip(
            LOCAL_LORAS,
            lora_values,
        ):
            value = float(weight or 0.0)

            if abs(value) < 1e-6:
                continue

            validate_lora_name(filename)

            if value < -3.0 or value > 3.0:
                raise ValueError(
                    f"LoRA weight out of range: {filename}"
                )

            enabled_loras.append(
                (
                    filename,
                    value,
                )
            )

        model_name = validate_model_name(
            base_model
        )

        length = duration_to_length(
            duration
        )

        log(
            f"[generate] model={model_name}"
        )
        log(
            f"[generate] resolution={width}x{height}"
        )
        log(
            f"[generate] duration={duration:.2f}s"
        )
        log(
            f"[generate] frames={length}"
        )
        log(
            f"[generate] steps={steps}"
        )
        log(
            f"[generate] seed={effective_seed}"
        )
        log(
            f"[generate] first_frame={first_name}"
        )
        log(
            f"[generate] last_frame={last_name}"
        )
        log(
            f"[generate] LoRAs={len(enabled_loras)}"
        )

        progress(
            0.15,
            desc="Building MiniMax H3 workflow",
        )

        CURRENT_STEPS.value = steps

        workflow = build_h3_workflow(
            prompt=prompt,
            width=width,
            height=height,
            duration=duration,
            seed=effective_seed,
            model_name=model_name,
            first_frame=first_name,
            last_frame=last_name,
            loras=enabled_loras,
        )

        # Save a copy for debugging/reproducibility.
        debug_dir = OUTPUT / "_workflows"
        debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        workflow_path = (
            debug_dir
            / f"h3_{uuid.uuid4().hex[:12]}.json"
        )

        workflow_path.write_text(
            json.dumps(
                workflow,
                indent=2,
            ),
            encoding="utf-8",
        )

        log(
            f"[workflow] saved {workflow_path}"
        )

        progress(
            0.25,
            desc="Running MiniMax H3",
        )

        with open(
            os.devnull,
            "w",
        ):
            result_paths = execute_workflow(
                workflow
            )

        progress(
            0.95,
            desc="Collecting video",
        )

        # Return the newest video. If multiple files are present because a
        # Comfy node emitted intermediates, use the newest matching output.
        video_candidates = [
            pathlib.Path(path)
            for path in result_paths
            if pathlib.Path(path).suffix.lower()
            in {
                ".mp4",
                ".webm",
                ".mov",
                ".mkv",
                ".avi",
            }
        ]

        if not video_candidates:
            raise RuntimeError(
                "No video output was produced."
            )

        video = max(
            video_candidates,
            key=lambda p: p.stat().st_mtime,
        )

        elapsed = time.time() - started

        status = (
            f"Done — {video.name} | "
            f"{width}×{height} | "
            f"{duration:.2f}s | "
            f"seed {effective_seed} | "
            f"{elapsed:.1f}s"
        )

        log(
            f"[done] {status}"
        )

        progress(
            1.0,
            desc="Done",
        )

        return (
            str(video),
            status,
            effective_seed,
        )

    except gr.Error:
        raise

    except Exception as exc:
        log(
            "[ERROR] "
            + traceback.format_exc()
        )

        raise gr.Error(
            "Generation failed: "
            + str(exc)[:1000]
        ) from exc

    finally:
        for path in staged:
            try:
                path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

        _GENERATION_LOCK.release()


# ============================================================================
# REFRESH
# ============================================================================

def refresh_catalog():
    try:
        ensure_comfy()
        scan_models()

        model_value = (
            LOCAL_MODELS[0]
            if LOCAL_MODELS
            else None
        )

        return (
            gr.update(
                choices=LOCAL_MODELS,
                value=model_value,
            ),
            (
                f"Found {len(LOCAL_MODELS)} model(s) and "
                f"{len(LOCAL_LORAS)} LoRA(s) in "
                f"{MODELS}"
            ),
        )

    except Exception as exc:
        log(
            "[refresh] "
            + traceback.format_exc()
        )

        return (
            gr.update(
                choices=[],
                value=None,
            ),
            f"Refresh failed: {exc}",
        )


# ============================================================================
# STARTUP
# ============================================================================

def startup() -> None:
    section("MiniMax H3 Studio — STARTUP")

    log(f"[startup] app root: {ROOT}")
    log(f"[startup] ComfyUI: {COMFY}")
    log(f"[startup] model volume: {MODELS}")

    try:
        ensure_comfy()

        log("")
        log("[startup] REQUIRED H3 ASSETS")

        for filename in (
            DEFAULT_DIFFUSION,
            DEFAULT_TEXT_ENCODER,
            DEFAULT_VIDEO_VAE,
            DEFAULT_AUDIO_VAE,
        ):
            found = find_model_by_basename(filename)

            if found:
                log(f"[startup] OK  {found}")
            else:
                log(f"[startup] MISSING  {filename}")

        log("")
        log("[startup] MiniMax H3 Studio READY")

    except Exception as exc:
        log(
            "[startup] setup incomplete: "
            + repr(exc)
        )
        log(
            "[startup] The app will retry setup when Generate is pressed."
        )


# ============================================================================
# UI
# ============================================================================

CSS = """
.gradio-container {
    max-width: 1450px !important;
}

#title {
    text-align: center;
}

#subtitle {
    text-align: center;
    opacity: 0.75;
}

#generate {
    min-height: 58px;
    font-size: 18px;
}

.video-output video {
    max-height: 720px;
}

.small-note {
    opacity: 0.75;
    font-size: 0.9em;
}
"""


def create_ui() -> gr.Blocks:
    startup()

    initial_models = list(LOCAL_MODELS)
    initial_loras = list(LOCAL_LORAS)

    default_model = (
        (
            DEFAULT_DIFFUSION
            if DEFAULT_DIFFUSION in initial_models
            else initial_models[0]
        )
        if initial_models
        else None
    )

    with gr.Blocks(
        title=APP_NAME,
        theme=gr.themes.Soft(),
        css=CSS,
    ) as demo:

        gr.Markdown(
            "# MiniMax H3 Studio",
            elem_id="title",
        )

        gr.Markdown(
            "Image-to-video / text-to-video with native H3 audio, "
            "local Modal Volume models, and LoRA controls.",
            elem_id="subtitle",
        )

        with gr.Row():

            # ---------------------------------------------------------------
            # LEFT SIDE — CONTROLS
            # ---------------------------------------------------------------

            with gr.Column(
                scale=1,
                min_width=430,
            ):

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder=(
                        "Describe the shots, camera movement, subject motion, "
                        "dialogue, sound effects and music..."
                    ),
                    lines=10,
                )

                with gr.Row():
                    first_frame = gr.Image(
                        label="First frame — optional",
                        type="filepath",
                        sources=["upload"],
                    )

                    last_frame = gr.Image(
                        label="Last frame — optional",
                        type="filepath",
                        sources=["upload"],
                    )

                gr.Markdown(
                    "Leave both frames empty for text-to-video. "
                    "Use one or both frames for first/last-frame generation.",
                    elem_classes=["small-note"],
                )

                base_model = gr.Dropdown(
                    choices=initial_models,
                    value=default_model,
                    label="MiniMax H3 diffusion model",
                    allow_custom_value=False,
                )

                with gr.Row():
                    width = gr.Slider(
                        MIN_WIDTH,
                        MAX_WIDTH,
                        value=DEFAULT_WIDTH,
                        step=32,
                        label="Width",
                    )

                    height = gr.Slider(
                        MIN_HEIGHT,
                        MAX_HEIGHT,
                        value=DEFAULT_HEIGHT,
                        step=32,
                        label="Height",
                    )

                with gr.Row():
                    duration = gr.Slider(
                        0.2,
                        15.0,
                        value=DEFAULT_DURATION,
                        step=0.1,
                        label="Duration (seconds)",
                    )

                    steps = gr.Slider(
                        4,
                        40,
                        value=DEFAULT_STEPS,
                        step=1,
                        label="Steps",
                    )

                with gr.Row():
                    seed = gr.Number(
                        value=DEFAULT_SEED,
                        precision=0,
                        label="Seed",
                    )

                    randomize_seed = gr.Checkbox(
                        value=True,
                        label="Randomize seed",
                    )

                with gr.Accordion(
                    "LoRAs",
                    open=False,
                ):
                    gr.Markdown(
                        "LoRAs are scanned from "
                        f"`{LORA_ROOT}`. "
                        "A weight of 0 disables a LoRA. Multiple LoRAs "
                        "can be chained."
                    )

                    lora_sliders: list[gr.Slider] = []

                    if initial_loras:
                        for filename in initial_loras:
                            lora_sliders.append(
                                gr.Slider(
                                    minimum=-3.0,
                                    maximum=3.0,
                                    value=0.0,
                                    step=0.05,
                                    label=filename,
                                )
                            )
                    else:
                        gr.Markdown(
                            "⚠️ No LoRAs found in the Modal Volume."
                        )

                with gr.Row():
                    refresh = gr.Button(
                        "↻ Refresh models / LoRAs",
                    )

                    generate_button = gr.Button(
                        "Generate",
                        variant="primary",
                        elem_id="generate",
                    )

            # ---------------------------------------------------------------
            # RIGHT SIDE — OUTPUT
            # ---------------------------------------------------------------

            with gr.Column(
                scale=1,
                min_width=520,
            ):
                video_output = gr.Video(
                    label="MiniMax H3 output",
                    interactive=False,
                    autoplay=True,
                    elem_classes=["video-output"],
                )

                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

                used_seed = gr.Number(
                    label="Used seed",
                    interactive=False,
                )

                gr.Markdown(
                    "### H3 workflow\n"
                    "- `MiniMaxH3ImageToVideo`\n"
                    "- `res_multistep` sampler\n"
                    "- `simple` scheduler\n"
                    "- 24 FPS\n"
                    "- video + native audio\n\n"
                    "The supplied workflow describes H3 as supporting "
                    "text-to-video when no images are connected and "
                    "first/last-frame video when keyframes are supplied.",
                    elem_classes=["small-note"],
                )

        # ---------------------------------------------------------------
        # EVENTS
        # ---------------------------------------------------------------

        generation_inputs = [
            prompt,
            first_frame,
            last_frame,
            width,
            height,
            duration,
            steps,
            seed,
            randomize_seed,
            base_model,
            *lora_sliders,
        ]

        def generation_wrapper(*values):
            base_values = values[:10]
            lora_values = values[10:]

            return generate(
                *base_values,
                list(lora_values),
            )

        generate_button.click(
            generation_wrapper,
            inputs=generation_inputs,
            outputs=[
                video_output,
                status,
                used_seed,
            ],
        )

        refresh.click(
            refresh_catalog,
            inputs=[],
            outputs=[
                base_model,
                status,
            ],
        )

    return demo


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    demo = create_ui()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )

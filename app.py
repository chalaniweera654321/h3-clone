from __future__ import annotations

import asyncio
import glob
import json
import os
import pathlib
import random
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Any
import time

import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image
from pathlib import Path


# ============================================================================
# SPACES COMPATIBILITY
# ============================================================================

try:
    import spaces
except ImportError:

    class _SpacesFallback:

        @staticmethod
        def GPU(**_kwargs):

            def decorate(function):
                return function

            return decorate

    spaces = _SpacesFallback()


# ============================================================================
# SETTINGS
# ============================================================================

from settings_utils import (
    build_settings,
    extract_image_settings,
    parse_settings_text,
    write_png_metadata,
)


# ============================================================================
# PATHS
# ============================================================================

ROOT = pathlib.Path(__file__).resolve().parent


def _detect_comfy_root() -> pathlib.Path:

    # Case 1:
    # app.py is directly inside ComfyUI.
    if (
        (ROOT / "main.py").is_file()
        and (ROOT / "models").is_dir()
    ):
        return ROOT

    # Case 2:
    # ComfyUI is a child of the application directory.
    candidate = ROOT / "ComfyUI"

    if (
        (candidate / "main.py").is_file()
        and (candidate / "models").is_dir()
    ):
        return candidate

    # Case 3:
    # /content/ComfyUI is the standard Colab location.
    candidate = pathlib.Path("/content/ComfyUI")

    if (
        (candidate / "main.py").is_file()
        and (candidate / "models").is_dir()
    ):
        return candidate

    # Last resort.
    return ROOT / "ComfyUI"


COMFY = _detect_comfy_root()

MODELS = Path("/mnt/krea2-models")

import folder_paths

folder_paths.add_model_folder_path(
    "diffusion_models",
    str(MODELS / "diffusion_models")
)

folder_paths.add_model_folder_path(
    "loras",
    str(MODELS / "loras")
)

folder_paths.add_model_folder_path(
    "vae",
    str(MODELS / "vae")
)

folder_paths.add_model_folder_path(
    "text_encoders",
    str(MODELS / "text_encoders")
)

INPUT = COMFY / "input"
OUTPUT = COMFY / "output"
CUSTOM_NODES = COMFY / "custom_nodes"


print(
    "[paths] ROOT:",
    ROOT,
    flush=True,
)

print(
    "[paths] COMFY:",
    COMFY,
    flush=True,
)

print(
    "[paths] MODELS:",
    MODELS,
    flush=True,
)


# ============================================================================
# WORKFLOW FILES
# ============================================================================

T2I_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2.json"
EDIT_SOURCE = ROOT / "lustifyWorkflowsKrea2_krea2Edit.json"


# ============================================================================
# KREA EDIT NODE
# ============================================================================

KREA_EDIT_NODES = (
    "https://github.com/lbouaraba/comfyui-krea2edit.git"
)


# ============================================================================
# IDENTITY ADAPTER
# ============================================================================

IDENTITY_REPO = "conradlocke/krea2-identity-edit"

IDENTITY_FILE = "krea2_identity_edit_v1_2.safetensors"

IDENTITY_LORA_DIR = (
    MODELS / "loras" / "krea"
)

IDENTITY_LORA_PATH = (
    IDENTITY_LORA_DIR / IDENTITY_FILE
)

IDENTITY_COMFY_NAME = pathlib.PurePosixPath(
    "krea",
    IDENTITY_FILE,
).as_posix()


# ============================================================================
# LOCAL MODEL DIRECTORIES
# ============================================================================

TEXT_ENCODER_DIR = MODELS / "text_encoders"

VAE_DIR = MODELS / "vae"

DIFFUSION_DIR = MODELS / "diffusion_models"

LORA_ROOT = MODELS / "loras"


# ============================================================================
# REQUIRED KREA FILES
# ============================================================================

TEXT_ENCODER_FILE = (
    "qwen3vl_4b_fp8_scaled.safetensors"
)

VAE_FILE = (
    "qwen_image_vae.safetensors"
)


# ============================================================================
# MODEL CATALOGS
# ============================================================================

LOCAL_BASE_MODELS: list[str] = []

LOCAL_LORAS: list[str] = []


# ============================================================================
# SAMPLERS
# ============================================================================

SAMPLERS = [
    "euler",
    "euler_ancestral",
    "euler_a",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_sde",
    "heun",
    "lms",
]


SCHEDULERS = [
    "beta",
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
]


# ============================================================================
# DEFAULTS
# ============================================================================

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

DEFAULT_TARGET_MP = 1.4

MAX_WIDTH = 2048
MAX_HEIGHT = 2048
MAX_TARGET_MP = 4.0

DEFAULT_GROUNDING = 768

DEFAULT_REF_BOOST = 1.0

DEFAULT_STEPS = 8

DEFAULT_CFG = 1.0

DEFAULT_SAMPLER = "euler"

DEFAULT_SCHEDULER = "beta"

DEFAULT_SEED = 2


MIN_GPU_SECONDS = int(
    os.environ.get(
        "MIN_GPU_SECONDS",
        "45",
    )
)


MAX_GPU_SECONDS = int(
    os.environ.get(
        "MAX_GPU_SECONDS",
        "300",
    )
)


# ============================================================================
# RUNTIME STATE
# ============================================================================

_comfy_ready = False

_nodes_ready = False

_workflow_cache: dict[str, dict[str, Any]] = {}


# ============================================================================
# COMMAND HELPERS
# ============================================================================

def _run(
    command: list[str],
    cwd: pathlib.Path | None = None,
    check: bool = True,
) -> None:

    print(
        "[setup]",
        " ".join(command),
        flush=True,
    )

    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=check,
    )


def _pip_install(
    arguments: list[str],
) -> None:

    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            *arguments,
        ],
        check=False,
    )


def _install_filtered_requirements(
    path: pathlib.Path,
) -> None:

    if not path.exists():
        return

    blocked = {
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "huggingface-hub",
        "accelerate",
    }

    requirements: list[str] = []

    for raw in path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():

        item = raw.strip()

        if not item:
            continue

        if item.startswith("#"):
            continue

        package = re.split(
            r"[<>=!~;\[\s]",
            item.lower().replace("_", "-"),
            maxsplit=1,
        )[0]

        if package not in blocked:
            requirements.append(item)

    if requirements:
        _pip_install(requirements)


def _ensure_repo(
    path: pathlib.Path,
    url: str,
) -> None:

    if path.exists():
        return

    _run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            url,
            str(path),
        ]
    )


# ============================================================================
# COMFY UTILS COMPATIBILITY
# ============================================================================

def _restore_utils_namespace() -> None:

    source = COMFY / "utils"

    target = COMFY / "utilities"

    if not source.exists() and target.exists():

        target.rename(source)

    if not source.exists():
        return

    for path in COMFY.rglob("*.py"):

        if "__pycache__" in path.parts:
            continue

        try:

            text = path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:

            continue

        updated = re.sub(
            r"\bfrom utilities\b",
            "from utils",
            text,
        )

        updated = re.sub(
            r"\bimport utilities\b",
            "import utils",
            updated,
        )

        if updated != text:

            path.write_text(
                updated,
                encoding="utf-8",
            )


# ============================================================================
# DIRECTORY SETUP
# ============================================================================

def _ensure_model_directories() -> None:

    for folder in [
        "diffusion_models",
        "text_encoders",
        "vae",
        "loras",
        "loras/krea",
    ]:

        (
            MODELS / folder
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    INPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# IDENTITY MODEL
# ============================================================================

def _download_identity_model() -> None:

    IDENTITY_LORA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if IDENTITY_LORA_PATH.exists():

        print(
            "[identity] already installed:",
            IDENTITY_LORA_PATH,
            flush=True,
        )

        return

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get(
            "HUGGINGFACE_HUB_TOKEN"
        )
    )

    print(
        "[identity] downloading:",
        IDENTITY_FILE,
        flush=True,
    )

    downloaded = pathlib.Path(
        hf_hub_download(
            repo_id=IDENTITY_REPO,
            filename=IDENTITY_FILE,
            local_dir=str(
                IDENTITY_LORA_DIR
            ),
            token=token,
        )
    )

    if (
        downloaded.resolve()
        != IDENTITY_LORA_PATH.resolve()
    ):

        shutil.move(
            str(downloaded),
            str(IDENTITY_LORA_PATH),
        )

    print(
        "[identity] ready:",
        IDENTITY_LORA_PATH,
        flush=True,
    )


# ============================================================================
# LOCAL MODEL SCANNER
# ============================================================================

def _scan_local_models() -> None:

    global LOCAL_BASE_MODELS
    global LOCAL_LORAS

    _ensure_model_directories()

    extensions = {
        ".safetensors",
        ".ckpt",
        ".pt",
        ".bin",
    }


    # ------------------------------------------------------------------------
    # DIFFUSION MODELS
    # ------------------------------------------------------------------------

    models: list[str] = []

    if DIFFUSION_DIR.exists():

        for path in DIFFUSION_DIR.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            relative = path.relative_to(
                DIFFUSION_DIR
            )

            models.append(
                pathlib.PurePosixPath(
                    *relative.parts
                ).as_posix()
            )

    LOCAL_BASE_MODELS = sorted(
        models,
        key=str.lower,
    )


    # ------------------------------------------------------------------------
    # LORAS
    # ------------------------------------------------------------------------

    loras: list[str] = []

    if LORA_ROOT.exists():

        for path in LORA_ROOT.rglob("*"):

            if not path.is_file():
                continue

            if path.suffix.lower() not in extensions:
                continue

            try:

                if (
                    path.resolve()
                    == IDENTITY_LORA_PATH.resolve()
                ):

                    continue

            except OSError:

                pass

            relative = path.relative_to(
                LORA_ROOT
            )

            loras.append(
                pathlib.PurePosixPath(
                    *relative.parts
                ).as_posix()
            )

    LOCAL_LORAS = sorted(
        loras,
        key=str.lower,
    )


    # ------------------------------------------------------------------------
    # LOG
    # ------------------------------------------------------------------------

    print(
        f"[models] found "
        f"{len(LOCAL_BASE_MODELS)} "
        f"local diffusion model(s)",
        flush=True,
    )

    for model in LOCAL_BASE_MODELS:

        print(
            "[models]   ",
            model,
            flush=True,
        )


    print(
        f"[loras] found "
        f"{len(LOCAL_LORAS)} "
        f"local LoRA(s)",
        flush=True,
    )

    for lora in LOCAL_LORAS:

        print(
            "[loras]   ",
            lora,
            flush=True,
        )


# ============================================================================
# REQUIRED ASSETS
# ============================================================================

def _validate_required_assets() -> None:

    missing: list[str] = []


    text_encoder = (
        TEXT_ENCODER_DIR
        / TEXT_ENCODER_FILE
    )

    if not text_encoder.is_file():

        missing.append(
            f"text encoder: {text_encoder}"
        )


    vae = (
        VAE_DIR
        / VAE_FILE
    )

    if not vae.is_file():

        missing.append(
            f"VAE: {vae}"
        )


    if not IDENTITY_LORA_PATH.is_file():

        missing.append(
            f"identity adapter: "
            f"{IDENTITY_LORA_PATH}"
        )


    if missing:

        raise RuntimeError(
            "Missing required local model files:\n"
            + "\n".join(
                f"  - {item}"
                for item in missing
            )
        )


# ============================================================================
# COMFY SETUP
# ============================================================================

def _ensure_comfy() -> None:

    global _comfy_ready

    if _comfy_ready:
        return


    print(
        "[comfy] using:",
        COMFY,
        flush=True,
    )


    if not (
        COMFY / "main.py"
    ).exists():

        raise RuntimeError(
            "ComfyUI was not found at:\n"
            f"{COMFY}"
        )


    # ------------------------------------------------------------------------
    # Requirements
    # ------------------------------------------------------------------------

    _install_filtered_requirements(
        COMFY / "requirements.txt"
    )


    # ------------------------------------------------------------------------
    # Custom nodes
    # ------------------------------------------------------------------------

    CUSTOM_NODES.mkdir(
        parents=True,
        exist_ok=True,
    )

    _ensure_repo(
        CUSTOM_NODES / "comfyui-krea2edit",
        KREA_EDIT_NODES,
    )


    # ------------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------------

    _restore_utils_namespace()


    # ------------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------------

    _ensure_model_directories()


    # ------------------------------------------------------------------------
    # Identity adapter only
    # ------------------------------------------------------------------------

    _download_identity_model()


    # ------------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------------

    _scan_local_models()


    _comfy_ready = True


# ============================================================================
# NODE INITIALIZATION
# ============================================================================

def _init_comfy_nodes() -> None:

    global _nodes_ready

    if _nodes_ready:
        return


    comfy_path = str(COMFY)


    # Make sure ComfyUI is first.
    sys.path = [
        item
        for item in sys.path
        if item != comfy_path
    ]

    sys.path.insert(
        0,
        comfy_path,
    )


    # Remove stale utils modules.
    for name in list(sys.modules):

        if (
            name == "utils"
            or name.startswith("utils.")
        ):

            del sys.modules[name]


    os.chdir(COMFY)


    import execution
    import nodes
    import server


    loop = asyncio.new_event_loop()

    asyncio.set_event_loop(loop)


    server_instance = server.PromptServer(
        loop
    )


    execution.PromptQueue(
        server_instance
    )


    loop.run_until_complete(
        nodes.init_extra_nodes()
    )


    _nodes_ready = True


# ============================================================================
# MODEL VALIDATION
# ============================================================================

def _validate_model_name(
    model_name: str,
) -> str:

    normalized = str(
        model_name
    ).replace(
        "\\",
        "/",
    )

    path = pathlib.PurePosixPath(
        normalized
    )


    if (
        not normalized
        or path.is_absolute()
        or any(
            part in {
                "",
                ".",
                "..",
            }
            for part in path.parts
        )
    ):

        raise ValueError(
            "invalid diffusion model path"
        )


    candidate = (
        DIFFUSION_DIR
        / pathlib.Path(
            *path.parts
        )
    )


    try:

        candidate.resolve().relative_to(
            DIFFUSION_DIR.resolve()
        )

    except ValueError as exc:

        raise ValueError(
            "invalid diffusion model path"
        ) from exc


    if not candidate.is_file():

        raise ValueError(
            "diffusion model is not installed: "
            + normalized
        )


    return path.as_posix()


# ============================================================================
# LORA VALIDATION
# ============================================================================

def _validate_lora_name(
    lora_name: str,
) -> str:

    normalized = str(
        lora_name
    ).replace(
        "\\",
        "/",
    )

    path = pathlib.PurePosixPath(
        normalized
    )


    if (
        not normalized
        or path.is_absolute()
        or any(
            part in {
                "",
                ".",
                "..",
            }
            for part in path.parts
        )
    ):

        raise ValueError(
            "invalid LoRA path"
        )


    candidate = (
        LORA_ROOT
        / pathlib.Path(
            *path.parts
        )
    )


    try:

        candidate.resolve().relative_to(
            LORA_ROOT.resolve()
        )

    except ValueError as exc:

        raise ValueError(
            "invalid LoRA path"
        ) from exc


    if not candidate.is_file():

        raise ValueError(
            "LoRA is not installed: "
            + normalized
        )


    if (
        candidate.resolve()
        == IDENTITY_LORA_PATH.resolve()
    ):

        raise ValueError(
            "identity adapter cannot be "
            "selected as a user LoRA"
        )


    return path.as_posix()


# ============================================================================
# WORKFLOW HELPERS
# ============================================================================

def _read_source_workflow(
    path: pathlib.Path,
) -> dict[str, Any]:

    if not path.exists():

        raise FileNotFoundError(
            f"workflow file is missing: "
            f"{path}"
        )


    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


    if not data.get("nodes"):

        raise ValueError(
            f"workflow file has no nodes: "
            f"{path.name}"
        )


    return data


def _ref(
    node: str,
    output: int = 0,
) -> list[Any]:

    return [
        node,
        output,
    ]


# ============================================================================
# T2I WORKFLOW
# ============================================================================

def _t2i_workflow(
    base_model: str,
) -> dict[str, Any]:

    base_model = _validate_model_name(
        base_model
    )


    cache_key = (
        f"text2image:{base_model}"
    )


    if cache_key in _workflow_cache:

        return json.loads(
            json.dumps(
                _workflow_cache[
                    cache_key
                ]
            )
        )


    _read_source_workflow(
        T2I_SOURCE
    )


    workflow = {

        "1": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": base_model,
                "weight_dtype": "default",
            },
        },

        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_FILE,
                "type": "krea2",
                "device": "default",
            },
        },

        "3": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_FILE,
            },
        },

        "4": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": _ref("1"),
                "shift": 4.0,
            },
        },

        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": _ref("2"),
                "text": "",
            },
        },

        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": _ref("5"),
            },
        },

        "7": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "batch_size": 1,
            },
        },

        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": _ref("4"),
                "positive": _ref("5"),
                "negative": _ref("6"),
                "latent_image": _ref("7"),
                "seed": DEFAULT_SEED,
                "steps": DEFAULT_STEPS,
                "cfg": DEFAULT_CFG,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER,
                "denoise": 1.0,
            },
        },

        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": _ref("8"),
                "vae": _ref("3"),
            },
        },

        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "images": _ref("9"),
                "filename_prefix": "krea2_turbo",
            },
        },
    }


    _workflow_cache[
        cache_key
    ] = workflow


    return json.loads(
        json.dumps(workflow)
    )


# ============================================================================
# EDIT WORKFLOW
# ============================================================================

def _edit_workflow(
    has_second_reference: bool,
    base_model: str,
) -> dict[str, Any]:

    base_model = _validate_model_name(
        base_model
    )


    _read_source_workflow(
        EDIT_SOURCE
    )


    workflow: dict[str, Any] = {

        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "",
            },
        },

        "3": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER_FILE,
                "type": "krea2",
                "device": "default",
            },
        },

        "4": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VAE_FILE,
            },
        },

        "5": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": base_model,
                "weight_dtype": "default",
            },
        },

        "6": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": _ref("5"),
                "lora_name": IDENTITY_COMFY_NAME,
                "strength_model": 1.0,
            },
        },

        "7": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": _ref("1"),
                "vae": _ref("4"),
            },
        },

        "8": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": DEFAULT_WIDTH,
                "height": DEFAULT_HEIGHT,
                "batch_size": 1,
            },
        },

        "9": {
            "class_type": "Krea2EditModelPatch",
            "inputs": {
                "model": _ref("6"),
                "source_latent": _ref("7"),
                "vae": _ref("4"),
                "source_image": _ref("1"),
                "target_latent": _ref("8"),
                "ref_boost": DEFAULT_REF_BOOST,
                "ref_boost_a": DEFAULT_REF_BOOST,
                "fit_mode": "fit",
            },
        },

        "10": {
            "class_type": "Krea2EditGroundedEncode",
            "inputs": {
                "clip": _ref("3"),
                "image": _ref("1"),
                "prompt": "",
                "grounding_px": DEFAULT_GROUNDING,
            },
        },

        "11": {
            "class_type": "ConditioningZeroOut",
            "inputs": {
                "conditioning": _ref("10"),
            },
        },

        "12": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {
                "model": _ref("9"),
                "shift": 4.0,
            },
        },

        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": _ref("12"),
                "positive": _ref("10"),
                "negative": _ref("11"),
                "latent_image": _ref("8"),
                "seed": DEFAULT_SEED,
                "steps": DEFAULT_STEPS,
                "cfg": DEFAULT_CFG,
                "sampler_name": DEFAULT_SAMPLER,
                "scheduler": DEFAULT_SCHEDULER,
                "denoise": 1.0,
            },
        },

        "14": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": _ref("13"),
                "vae": _ref("4"),
            },
        },

        "15": {
            "class_type": "SaveImage",
            "inputs": {
                "images": _ref("14"),
                "filename_prefix": "krea2_edit",
            },
        },
    }


    if has_second_reference:

        workflow["2"] = {
            "class_type": "LoadImage",
            "inputs": {
                "image": "",
            },
        }

        workflow["16"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": _ref("2"),
                "vae": _ref("4"),
            },
        }

        workflow["9"]["inputs"][
            "source_latent_b"
        ] = _ref("16")

        workflow["9"]["inputs"][
            "source_image_b"
        ] = _ref("2")

        workflow["10"]["inputs"][
            "image_b"
        ] = _ref("2")


    return workflow


# ============================================================================
# FIND NODE
# ============================================================================

def _find_node(
    workflow: dict[str, Any],
    class_type: str,
) -> str:

    for node_id, node in workflow.items():

        if node.get("class_type") == class_type:

            return node_id

    raise KeyError(
        f"workflow does not contain "
        f"{class_type}"
    )


# ============================================================================
# LORA CHAIN
# ============================================================================

def _inject_lora_chain(
    workflow: dict[str, Any],
    enabled_loras: list[tuple[str, float]],
    *,
    model_source: list[Any],
    clip_source: list[Any],
    model_consumers: list[tuple[str, str]],
    clip_consumers: list[tuple[str, str]],
) -> None:

    if not enabled_loras:
        return


    previous_model = model_source

    previous_clip = clip_source


    for index, (
        filename,
        strength,
    ) in enumerate(
        enabled_loras
    ):

        node_id = (
            f"user_lora_{index}"
        )


        workflow[node_id] = {

            "class_type": "LoraLoader",

            "inputs": {

                "model": previous_model,

                "clip": previous_clip,

                "lora_name": filename,

                "strength_model": float(
                    strength
                ),

                "strength_clip": float(
                    strength
                ),
            },
        }


        previous_model = _ref(
            node_id
        )

        previous_clip = _ref(
            node_id,
            1,
        )


    for node_id, input_name in (
        model_consumers
    ):

        workflow[node_id]["inputs"][
            input_name
        ] = previous_model


    for node_id, input_name in (
        clip_consumers
    ):

        workflow[node_id]["inputs"][
            input_name
        ] = previous_clip


# ============================================================================
# EDIT IMAGE
# ============================================================================

def _prepare_edit_image(
    path: str,
    target_megapixels: float,
) -> tuple[str, int, int]:

    with Image.open(path) as source:

        image = source.convert("RGB")


        megapixels = max(
            0.25,
            min(
                MAX_TARGET_MP,
                float(target_megapixels),
            ),
        )


        scale = (
            megapixels
            * 1_000_000
            / max(
                1,
                image.width
                * image.height,
            )
        ) ** 0.5


        width = max(
            64,
            int(
                round(
                    image.width
                    * scale
                    / 64
                )
                * 64
            ),
        )


        height = max(
            64,
            int(
                round(
                    image.height
                    * scale
                    / 64
                )
                * 64
            ),
        )


        width = min(
            MAX_WIDTH,
            width,
        )

        height = min(
            MAX_HEIGHT,
            height,
        )


        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )


        name = (
            f"input_"
            f"{uuid.uuid4().hex[:12]}"
            f".png"
        )


        image.save(
            INPUT / name,
            format="PNG",
        )


    return (
        name,
        width,
        height,
    )


# ============================================================================
# EXECUTION HELPER
# ============================================================================

def _execute_prompt(prompt: dict[str, Any]) -> str:
    import execution
    import server

    loop = asyncio.get_event_loop()
    server_instance = server.PromptServer.instance
    queue = server_instance.prompt_queue

    prompt_id = str(uuid.uuid4())
    valid = execution.validate_prompt(prompt)

    if not valid[0]:
        raise RuntimeError(f"Invalid prompt structure: {valid[1]}")

    queue.put((0, prompt_id, prompt, extra_data := {}, valid[2]))

    q = execution.PromptExecutor(server_instance, queue)
    loop.run_until_complete(q.execute())

    return prompt_id


# ============================================================================
# GENERATION ENGINE
# ============================================================================

@spaces.GPU
def generate_image(
    prompt_text: str,
    base_model: str,
    image_a: str | None,
    image_b: str | None,
    width: int,
    height: int,
    target_mp: float,
    grounding_px: int,
    ref_boost: float,
    steps: int,
    cfg: float,
    sampler_name: str,
    scheduler: str,
    seed: int,
    lora1_name: str,
    lora1_weight: float,
    lora2_name: str,
    lora2_weight: float,
    lora3_name: str,
    lora3_weight: float,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str, str, str]:

    start_time = time.time()
    _ensure_comfy()
    _init_comfy_nodes()
    _validate_required_assets()

    enabled_loras: list[tuple[str, float]] = []
    for l_name, l_weight in [
        (lora1_name, lora1_weight),
        (lora2_name, lora2_weight),
        (lora3_name, lora3_weight),
    ]:
        if l_name and l_name != "None":
            valid_lora = _validate_lora_name(l_name)
            enabled_loras.append((valid_lora, float(l_weight)))

    print(f"\n[gen] --- Starting New Generation Task ---", flush=True)
    print(f"[gen] Prompt: '{prompt_text}'", flush=True)
    print(f"[gen] Base Model: {base_model}", flush=True)
    print(f"[gen] Active LoRAs ({len(enabled_loras)}):", flush=True)
    for name, weight in enabled_loras:
        print(f"[gen]   - {name} (Strength: {weight})", flush=True)

    mode = "edit" if image_a else "t2i"

    if mode == "t2i":
        workflow = _t2i_workflow(base_model)
        
        # Inject LoRAs
        _inject_lora_chain(
            workflow,
            enabled_loras,
            model_source=_ref("1"),
            clip_source=_ref("2"),
            model_consumers=[("4", "model")],
            clip_consumers=[("5", "clip")],
        )

        # Set Prompt & Parameters
        clip_encode_id = _find_node(workflow, "CLIPTextEncode")
        workflow[clip_encode_id]["inputs"]["text"] = prompt_text

        empty_latent_id = _find_node(workflow, "EmptyLatentImage")
        workflow[empty_latent_id]["inputs"]["width"] = width
        workflow[empty_latent_id]["inputs"]["height"] = height

        ksampler_id = _find_node(workflow, "KSampler")
        workflow[ksampler_id]["inputs"].update({
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
        })

    else:
        has_b = bool(image_b)
        workflow = _edit_workflow(has_b, base_model)

        name_a, w_a, h_a = _prepare_edit_image(image_a, target_mp)
        workflow["1"]["inputs"]["image"] = name_a

        if has_b:
            name_b, _, _ = _prepare_edit_image(image_b, target_mp)
            workflow["2"]["inputs"]["image"] = name_b

        workflow["8"]["inputs"]["width"] = w_a
        workflow["8"]["inputs"]["height"] = h_a
        workflow["9"]["inputs"]["ref_boost"] = ref_boost
        workflow["9"]["inputs"]["ref_boost_a"] = ref_boost
        workflow["10"]["inputs"]["prompt"] = prompt_text
        workflow["10"]["inputs"]["grounding_px"] = grounding_px

        _inject_lora_chain(
            workflow,
            enabled_loras,
            model_source=_ref("5"),
            clip_source=_ref("3"),
            model_consumers=[("6", "model")],
            clip_consumers=[("10", "clip")],
        )

        ksampler_id = _find_node(workflow, "KSampler")
        workflow[ksampler_id]["inputs"].update({
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
        })

    print("[gen] Executing workflow graph...", flush=True)
    prompt_id = _execute_prompt(workflow)

    # Locating output image
    output_files = glob.glob(str(OUTPUT / "*.png"))
    if not output_files:
        raise RuntimeError("Generation failed: No output image found in output directory.")

    latest_image_path = max(output_files, key=os.path.getmtime)
    
    # Process output image to temp location
    temp_output_path = pathlib.Path(tempfile.gettempdir()) / f"out_{uuid.uuid4().hex}.png"
    shutil.copy(latest_image_path, temp_output_path)

    generation_time = time.time() - start_time
    print(f"[gen] Image generation complete in {generation_time:.2f} seconds.", flush=True)

    status_log = (
        f"Mode: {mode.upper()}\n"
        f"Base Model: {base_model}\n"
        f"Generation Time: {generation_time:.2f}s\n"
        f"Active LoRAs: {len(enabled_loras)}\n"
        f"Seed: {seed}"
    )

    settings_dump = json.dumps({
        "prompt": prompt_text,
        "base_model": base_model,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler_name,
        "scheduler": scheduler,
        "loras": enabled_loras,
    }, indent=2)

    return str(temp_output_path), status_log, settings_dump


# ============================================================================
# GRADIO INTERFACE CONSTRUCTION
# ============================================================================

def build_ui() -> gr.Blocks:
    _ensure_model_directories()
    _scan_local_models()

    lora_options = ["None"] + LOCAL_LORAS
    default_base = LOCAL_BASE_MODELS[0] if LOCAL_BASE_MODELS else ""

    with gr.Blocks(title="Krea2 ComfyUI Studio") as demo:
        gr.Markdown("# Krea2 Image Generation & Edit Studio")

        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your generation prompt here...",
                    lines=3,
                )

                base_model_dropdown = gr.Dropdown(
                    label="Base Diffusion Model",
                    choices=LOCAL_BASE_MODELS,
                    value=default_base,
                )

                with gr.Accordion("Reference Images (Edit Mode)", open=False):
                    image_a_input = gr.Image(label="Source Image A", type="filepath")
                    image_b_input = gr.Image(label="Source Image B (Optional)", type="filepath")
                    target_mp_slider = gr.Slider(0.25, MAX_TARGET_MP, value=DEFAULT_TARGET_MP, step=0.05, label="Target Megapixels")
                    ref_boost_slider = gr.Slider(0.0, 2.0, value=DEFAULT_REF_BOOST, step=0.1, label="Reference Boost")
                    grounding_slider = gr.Slider(64, 2048, value=DEFAULT_GROUNDING, step=64, label="Grounding Pixels")

                with gr.Accordion("LoRA Selection Stack", open=True):
                    with gr.Row():
                        lora1_dropdown = gr.Dropdown(label="LoRA 1", choices=lora_options, value="None")
                        lora1_weight = gr.Slider(-2.0, 2.0, value=1.0, step=0.05, label="Strength 1")
                    with gr.Row():
                        lora2_dropdown = gr.Dropdown(label="LoRA 2", choices=lora_options, value="None")
                        lora2_weight = gr.Slider(-2.0, 2.0, value=1.0, step=0.05, label="Strength 2")
                    with gr.Row():
                        lora3_dropdown = gr.Dropdown(label="LoRA 3", choices=lora_options, value="None")
                        lora3_weight = gr.Slider(-2.0, 2.0, value=1.0, step=0.05, label="Strength 3")

                with gr.Accordion("Advanced Parameters", open=False):
                    with gr.Row():
                        width_slider = gr.Slider(64, MAX_WIDTH, value=DEFAULT_WIDTH, step=64, label="Width")
                        height_slider = gr.Slider(64, MAX_HEIGHT, value=DEFAULT_HEIGHT, step=64, label="Height")
                    with gr.Row():
                        steps_slider = gr.Slider(1, 50, value=DEFAULT_STEPS, step=1, label="Steps")
                        cfg_slider = gr.Slider(0.0, 20.0, value=DEFAULT_CFG, step=0.1, label="CFG")
                    with gr.Row():
                        sampler_dropdown = gr.Dropdown(label="Sampler", choices=SAMPLERS, value=DEFAULT_SAMPLER)
                        scheduler_dropdown = gr.Dropdown(label="Scheduler", choices=SCHEDULERS, value=DEFAULT_SCHEDULER)
                    seed_number = gr.Number(label="Seed", value=DEFAULT_SEED, precision=0)

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                image_output = gr.Image(label="Generated Result")
                status_output = gr.Textbox(label="Execution Status & Time Log", interactive=False)
                settings_output = gr.Code(label="Generation Parameters JSON", language="json")

        generate_btn.click(
            fn=generate_image,
            inputs=[
                prompt_input,
                base_model_dropdown,
                image_a_input,
                image_b_input,
                width_slider,
                height_slider,
                target_mp_slider,
                grounding_slider,
                ref_boost_slider,
                steps_slider,
                cfg_slider,
                sampler_dropdown,
                scheduler_dropdown,
                seed_number,
                lora1_dropdown,
                lora1_weight,
                lora2_dropdown,
                lora2_weight,
                lora3_dropdown,
                lora3_weight,
            ],
            outputs=[
                image_output,
                status_output,
                settings_output,
            ],
        )

    return demo


if __name__ == "__main__":
    _ensure_comfy()
    app = build_ui()
    app.queue().launch()

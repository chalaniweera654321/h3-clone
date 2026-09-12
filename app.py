import os
import sys
import json
import time
import uuid
import shutil
import socket
import subprocess
from pathlib import Path

import requests
import gradio as gr


# ============================================================
# CONFIG
# ============================================================

COMFY_ROOT = Path("/root/ComfyUI")
MODEL_ROOT = Path("/mnt/minimax-h3-models")

COMFY_URL = "http://127.0.0.1:8188"

VIDEO_MODEL = "10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"

UPSCALE_REPO = (
    "https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale"
)

OUTPUT_DIR = Path("/tmp/minimax_h3_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROCESS = None


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(f"[APP] {message}", flush=True)


# ============================================================
# BASIC HELPERS
# ============================================================

def port_open(host="127.0.0.1", port=8188):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        sock.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        sock.close()


def wait_for_comfy(timeout=300):
    log("Waiting for ComfyUI...")

    end = time.time() + timeout

    while time.time() < end:
        if port_open():
            try:
                r = requests.get(
                    COMFY_URL + "/system_stats",
                    timeout=5,
                )

                if r.ok:
                    log("ComfyUI is ready.")
                    return True

            except Exception:
                pass

        time.sleep(2)

    raise RuntimeError(
        "ComfyUI did not become ready within the timeout."
    )


# ============================================================
# MODEL CHECKS
# ============================================================

def check_models():
    log("Checking models...")

    required = {
        "diffusion_models": VIDEO_MODEL,
        "text_encoders": TEXT_ENCODER,
        "vae": VIDEO_VAE,
        "vae_audio": AUDIO_VAE,
        "latent_upscale_models": UPSCALER,
    }

    missing = []

    for category, filename in required.items():

        if category == "vae_audio":
            path = MODEL_ROOT / "vae" / filename
        else:
            path = MODEL_ROOT / category / filename

        if path.exists() and path.stat().st_size > 0:
            size_gb = path.stat().st_size / (1024 ** 3)
            log(
                f"OK: {category}/{filename} "
                f"({size_gb:.2f} GB)"
            )
        else:
            missing.append(str(path))
            log(f"MISSING: {path}")

    if missing:
        raise RuntimeError(
            "Required models are missing:\n\n"
            + "\n".join(missing)
        )

    log("All required models are present.")


# ============================================================
# COMFYUI MODEL LINKING
# ============================================================

def link_models():
    """
    The Modal volume is mounted at /mnt/minimax-h3-models.
    ComfyUI expects models under /root/ComfyUI/models.
    """

    comfy_models = COMFY_ROOT / "models"
    comfy_models.mkdir(parents=True, exist_ok=True)

    categories = [
        "diffusion_models",
        "text_encoders",
        "vae",
        "latent_upscale_models",
    ]

    for category in categories:

        source = MODEL_ROOT / category
        target = comfy_models / category

        if not source.exists():
            continue

        if target.is_symlink():
            try:
                if target.resolve() == source.resolve():
                    log(f"LINK OK: {category}")
                    continue
            except Exception:
                pass

            target.unlink()

        elif target.exists():

            # If ComfyUI already has a real directory,
            # copy/link individual files instead.
            log(
                f"Using existing ComfyUI model directory: "
                f"{target}"
            )

            for file in source.iterdir():

                destination = target / file.name

                if destination.exists():
                    continue

                try:
                    destination.symlink_to(file)
                except Exception:
                    shutil.copy2(file, destination)

            continue

        target.symlink_to(source, target_is_directory=True)

        log(
            f"Linked {target} -> {source}"
        )


# ============================================================
# CUSTOM NODE INSTALLATION
# ============================================================

def install_custom_nodes():

    custom_nodes = COMFY_ROOT / "custom_nodes"
    custom_nodes.mkdir(parents=True, exist_ok=True)

    node_dir = (
        custom_nodes /
        "Comfyui-MMH3-UltimateUpscale"
    )

    if node_dir.exists():
        log("MiniMax H3 Ultimate Upscale node already installed.")
        return

    log("Installing MiniMax H3 Ultimate Upscale custom node...")

    result = subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            UPSCALE_REPO,
            str(node_dir),
        ],
        cwd=str(COMFY_ROOT),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Failed to install MiniMax H3 Ultimate Upscale.\n\n"
            + result.stdout
            + "\n"
            + result.stderr
        )

    log("Ultimate Upscale custom node installed.")


# ============================================================
# START COMFYUI
# ============================================================

def start_comfy():

    global PROCESS

    if port_open():
        log("ComfyUI is already running.")
        return

    log("Starting ComfyUI...")

    command = [
        sys.executable,
        str(COMFY_ROOT / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        "8188",
        "--lowvram",
        "--force-fp16",
        "--use-ck-attention",
    ]

    log("Command:")
    log(" ".join(command))

    PROCESS = subprocess.Popen(
        command,
        cwd=str(COMFY_ROOT),
        stdout=None,
        stderr=None,
    )

    wait_for_comfy()


# ============================================================
# STARTUP
# ============================================================

def startup():

    log("=" * 70)
    log("MiniMax H3 Gradio App Starting")
    log("=" * 70)

    check_models()
    link_models()
    install_custom_nodes()
    start_comfy()

    log("=" * 70)
    log("Startup complete.")
    log("=" * 70)


# ============================================================
# COMFY API
# ============================================================

def queue_prompt(workflow):

    log("Sending workflow to ComfyUI...")

    payload = {
        "prompt": workflow,
        "client_id": str(uuid.uuid4()),
    }

    r = requests.post(
        COMFY_URL + "/prompt",
        json=payload,
        timeout=60,
    )

    if not r.ok:
        raise RuntimeError(
            "ComfyUI rejected the workflow.\n\n"
            f"HTTP {r.status_code}\n\n"
            f"{r.text}"
        )

    data = r.json()

    if "error" in data:
        raise RuntimeError(
            "ComfyUI workflow error:\n\n"
            + json.dumps(
                data,
                indent=2,
            )
        )

    if "prompt_id" not in data:
        raise RuntimeError(
            "ComfyUI did not return a prompt_id.\n\n"
            + json.dumps(
                data,
                indent=2,
            )
        )

    prompt_id = data["prompt_id"]

    log(f"Prompt ID: {prompt_id}")

    return prompt_id


# ============================================================
# WAIT FOR HISTORY
# ============================================================

def wait_history(prompt_id):

    log("Waiting for ComfyUI generation...")

    end = time.time() + 3600

    last_status = None
    last_print = 0

    while time.time() < end:

        try:

            r = requests.get(
                COMFY_URL + f"/history/{prompt_id}",
                timeout=15,
            )

            if r.ok:

                data = r.json()

                if prompt_id in data:

                    history = data[prompt_id]

                    status = history.get(
                        "status",
                        {},
                    )

                    status_str = status.get(
                        "status_str"
                    )

                    completed = status.get(
                        "completed",
                        False,
                    )

                    if (
                        status_str != last_status
                        or time.time() - last_print > 10
                    ):

                        log(
                            f"ComfyUI status: "
                            f"{status_str}, "
                            f"completed={completed}"
                        )

                        last_status = status_str
                        last_print = time.time()

                    # ------------------------------------------------
                    # IMPORTANT:
                    # Catch ComfyUI errors here instead of waiting
                    # until fetch_video().
                    # ------------------------------------------------

                    if status_str == "error":

                        messages = status.get(
                            "messages",
                            [],
                        )

                        log("COMFYUI REPORTED AN ERROR")

                        raise RuntimeError(
                            "ComfyUI generation failed:\n\n"
                            + json.dumps(
                                messages,
                                indent=2,
                            )
                        )

                    # ComfyUI normally gives outputs once execution
                    # has completed.
                    if completed or status_str in (
                        "success",
                        "completed",
                    ):

                        log("ComfyUI reports generation complete.")

                        return history

                    # Some versions don't expose completed=True
                    # consistently. If outputs already exist, allow
                    # the next stage to inspect them.
                    if history.get("outputs"):

                        return history

        except RuntimeError:
            raise

        except Exception as e:

            log(
                f"History polling error: {type(e).__name__}: {e}"
            )

        time.sleep(2)

    raise TimeoutError(
        "Timed out waiting for ComfyUI generation."
    )


# ============================================================
# RECURSIVE OUTPUT SEARCH
# ============================================================

def find_video_items(obj):

    """
    ComfyUI custom nodes do not always return exactly the same
    output dictionary structure.

    Search recursively for video/file/image-like entries.
    """

    found = []

    def walk(value):

        if isinstance(value, dict):

            # ----------------------------------------------------
            # A normal ComfyUI file object
            # ----------------------------------------------------

            if "filename" in value:

                filename = value.get("filename")

                if filename:
                    found.append(value.copy())

            # ----------------------------------------------------
            # Continue recursively
            # ----------------------------------------------------

            for child in value.values():
                walk(child)

        elif isinstance(value, list):

            for child in value:
                walk(child)

    walk(obj)

    # Deduplicate
    unique = []
    seen = set()

    for item in found:

        key = (
            item.get("filename"),
            item.get("subfolder", ""),
            item.get("type", "output"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


# ============================================================
# FETCH VIDEO
# ============================================================

def fetch_video(history):

    log("=" * 60)
    log("Inspecting ComfyUI outputs...")
    log("=" * 60)

    outputs = history.get("outputs", {})

    # ------------------------------------------------------------
    # VERY IMPORTANT DEBUGGING
    # ------------------------------------------------------------

    if not outputs:

        log("WARNING: ComfyUI returned ZERO output nodes.")

        log("FULL HISTORY:")
        print(
            json.dumps(
                history,
                indent=2,
                default=str,
            ),
            flush=True,
        )

        raise RuntimeError(
            "ComfyUI finished, but the workflow produced no "
            "output nodes. Check the ComfyUI terminal above "
            "for the actual node error."
        )

    log(
        "Output node IDs: "
        + ", ".join(str(x) for x in outputs.keys())
    )

    # ------------------------------------------------------------
    # Print every output node
    # ------------------------------------------------------------

    for node_id, node_output in outputs.items():

        log(
            f"Output node {node_id}: "
            + json.dumps(
                node_output,
                indent=2,
                default=str,
            )
        )

    # ------------------------------------------------------------
    # Find all file objects recursively
    # ------------------------------------------------------------

    items = find_video_items(outputs)

    if not items:

        log("No filename objects found in ComfyUI outputs.")

        log(
            "FULL OUTPUT SECTION:"
        )

        print(
            json.dumps(
                outputs,
                indent=2,
                default=str,
            ),
            flush=True,
        )

        raise RuntimeError(
            "ComfyUI finished, but no video file was returned. "
            "The SaveVideo node may not have executed."
        )

    log(f"Found {len(items)} file object(s).")

    # ------------------------------------------------------------
    # Print discovered files
    # ------------------------------------------------------------

    for i, item in enumerate(items):

        log(
            f"[{i}] "
            f"filename={item.get('filename')} "
            f"subfolder={item.get('subfolder', '')} "
            f"type={item.get('type', 'output')}"
        )

    # ------------------------------------------------------------
    # Prefer the final UltimateUpscale output
    # ------------------------------------------------------------

    def is_video(item):

        filename = str(
            item.get("filename", "")
        ).lower()

        return filename.endswith(
            (
                ".mp4",
                ".webm",
                ".mov",
                ".mkv",
                ".avi",
                ".gif",
            )
        )

    upscale_items = [
        x for x in items
        if "UltimateUpscale" in
        str(x.get("filename", ""))
    ]

    if upscale_items:

        item = upscale_items[-1]

        log(
            "Selected UltimateUpscale output: "
            + str(item.get("filename"))
        )

    else:

        video_items = [
            x for x in items
            if is_video(x)
        ]

        if video_items:

            item = video_items[-1]

            log(
                "Selected video output: "
                + str(item.get("filename"))
            )

        else:

            item = items[-1]

            log(
                "No obvious video extension found; "
                "using final file object: "
                + str(item.get("filename"))
            )

    # ------------------------------------------------------------
    # Download using /view
    # ------------------------------------------------------------

    filename = item.get("filename")

    if not filename:
        raise RuntimeError(
            "ComfyUI returned an output entry without a filename."
        )

    params = {
        "filename": filename,
        "subfolder": item.get(
            "subfolder",
            "",
        ),
        "type": item.get(
            "type",
            "output",
        ),
    }

    log(
        "Downloading from ComfyUI /view:"
    )

    log(
        f"filename={params['filename']}"
    )

    log(
        f"subfolder={params['subfolder']}"
    )

    log(
        f"type={params['type']}"
    )

    r = requests.get(
        COMFY_URL + "/view",
        params=params,
        timeout=600,
    )

    if not r.ok:

        raise RuntimeError(
            "ComfyUI generated the file, but /view failed.\n\n"
            f"HTTP {r.status_code}\n\n"
            f"{r.text[:2000]}"
        )

    if not r.content:

        raise RuntimeError(
            "ComfyUI /view returned an empty file."
        )

    # ------------------------------------------------------------
    # Save locally
    # ------------------------------------------------------------

    safe_name = Path(filename).name

    output_path = (
        OUTPUT_DIR /
        f"{uuid.uuid4().hex}_{safe_name}"
    )

    output_path.write_bytes(
        r.content
    )

    log(
        f"Downloaded result: {output_path}"
    )

    log(
        f"File size: "
        f"{output_path.stat().st_size / (1024 ** 2):.2f} MB"
    )

    return str(output_path)


# ============================================================
# IMAGE UPLOAD
# ============================================================

def upload_image_to_comfy(image_path):

    if not image_path:
        return None

    path = Path(image_path)

    if not path.exists():
        raise RuntimeError(
            f"Image does not exist: {path}"
        )

    log(
        f"Uploading image to ComfyUI: {path.name}"
    )

    with open(path, "rb") as f:

        r = requests.post(
            COMFY_URL + "/upload/image",
            files={
                "image": (
                    path.name,
                    f,
                    "application/octet-stream",
                )
            },
            data={
                "overwrite": "true",
            },
            timeout=120,
        )

    if not r.ok:

        raise RuntimeError(
            "Failed to upload image to ComfyUI.\n\n"
            + r.text
        )

    data = r.json()

    log(
        "Uploaded image: "
        + json.dumps(data)
    )

    return data.get(
        "name",
        path.name,
    )


# ============================================================
# DIMENSION HELPERS
# ============================================================

def align32(value):
    return max(
        32,
        int(round(value / 32)) * 32,
    )


def calculate_dimensions(
    aspect_ratio,
    stage1_mp=0.4,
    stage2_mp=0.9,
):

    ratios = {
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "1:1": 1.0,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
    }

    ratio = ratios.get(
        aspect_ratio,
        16 / 9,
    )

    # ------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------

    h1 = int(
        ((stage1_mp * 1_000_000) / ratio) ** 0.5
    )

    w1 = int(
        h1 * ratio
    )

    w1 = align32(w1)
    h1 = align32(h1)

    # ------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------

    h2 = int(
        ((stage2_mp * 1_000_000) / ratio) ** 0.5
    )

    w2 = int(
        h2 * ratio
    )

    w2 = align32(w2)
    h2 = align32(h2)

    return w1, h1, w2, h2


# ============================================================
# WORKFLOW
# ============================================================

def build_workflow(
    prompt,
    duration,
    aspect_ratio,
    first_frame=None,
    last_frame=None,
    lora_name=None,
    lora_strength=1.0,
):

    # ------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------

    s1w, s1h, s2w, s2h = calculate_dimensions(
        aspect_ratio,
        stage1_mp=0.4,
        stage2_mp=0.9,
    )

    # ------------------------------------------------------------
    # H3 frame calculation
    #
    # Source workflow uses:
    #
    # max(5, round(a * 24))
    # + (5 - (max(5, round(a * 24)) % 17)) % 17
    #
    # ------------------------------------------------------------

    raw_frames = max(
        5,
        round(float(duration) * 24),
    )

    length = (
        raw_frames
        + (
            5
            - (
                raw_frames % 17
            )
        ) % 17
    )

    log(
        f"Stage 1: {s1w}x{s1h}"
    )

    log(
        f"Stage 2: {s2w}x{s2h}"
    )

    log(
        f"Duration: {duration}s"
    )

    log(
        f"Frames: {length}"
    )

    # ------------------------------------------------------------
    # Uploaded images
    # ------------------------------------------------------------

    first_name = None
    last_name = None

    if first_frame:
        first_name = upload_image_to_comfy(
            first_frame
        )

    if last_frame:
        last_name = upload_image_to_comfy(
            last_frame
        )

    # ------------------------------------------------------------
    # Basic source workflow
    # ------------------------------------------------------------

    workflow = {

        # --------------------------------------------------------
        # Video VAE
        # --------------------------------------------------------

        "119": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": VIDEO_VAE,
            },
        },

        # --------------------------------------------------------
        # Audio VAE
        # --------------------------------------------------------

        "120": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": AUDIO_VAE,
            },
        },

        # --------------------------------------------------------
        # Model
        # --------------------------------------------------------

        "127": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": VIDEO_MODEL,
                "weight_dtype": "default",
            },
        },

        # --------------------------------------------------------
        # Text encoder
        # --------------------------------------------------------

        "128": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": TEXT_ENCODER,
                "type": "minimax",
                "device": "default",
            },
        },

        # --------------------------------------------------------
        # Optional LoRA
        # --------------------------------------------------------

        "129": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": 0,
            },
        },

        # --------------------------------------------------------
        # Sigma shift
        # --------------------------------------------------------

        "144": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": [
                    "127",
                    0,
                ],
                "shift_video": 12,
                "shift_audio": 3,
            },
        },

        # --------------------------------------------------------
        # Attention backend
        # --------------------------------------------------------

        "173": {
            "class_type": "ModelAttentionBackend",
            "inputs": {
                "model": [
                    "144",
                    0,
                ],
                "attention": "comfy kitchen attention",
            },
        },

        # --------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------

        "124": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": [
                    "173",
                    0,
                ],
                "scheduler": "simple",
                "steps": 6,
                "denoise": 1.0,
            },
        },

        # --------------------------------------------------------
        # KSampler
        # --------------------------------------------------------

        "123": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler",
            },
        },

        # --------------------------------------------------------
        # Guider
        # --------------------------------------------------------

        "126": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": [
                    "173",
                    0,
                ],
                "conditioning": [
                    "131",
                    0,
                ],
            },
        },

        # --------------------------------------------------------
        # Image-to-video conditioning
        # --------------------------------------------------------

        "131": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": [
                    "128",
                    0,
                ],
                "vae": [
                    "119",
                    0,
                ],
                "prompt": prompt,
                "width": s1w,
                "height": s1h,
                "length": length,
            },
        },

        # --------------------------------------------------------
        # Advanced sampler
        # --------------------------------------------------------

        "125": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": [
                    "129",
                    0,
                ],
                "guider": [
                    "126",
                    0,
                ],
                "sampler": [
                    "123",
                    0,
                ],
                "sigmas": [
                    "124",
                    0,
                ],
                "latent_image": [
                    "131",
                    0,
                ],
            },
        },

        # --------------------------------------------------------
        # Upscale conditioning
        # --------------------------------------------------------

        "164": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": [
                    "128",
                    0,
                ],
                "vae": [
                    "119",
                    0,
                ],
                "prompt": prompt,
                "width": s2w,
                "height": s2h,
                "length": length,
            },
        },

        # --------------------------------------------------------
        # Upscaler
        # --------------------------------------------------------

        "160": {
            "class_type": "MMH3LatentUpscaleWithModelParams",
            "inputs": {
                "model_name": UPSCALER,
                "width": s2w,
                "height": s2h,
                "device": "cuda",
                "precision": "fp16",
            },
        },

        # --------------------------------------------------------
        # Temporal split
        # --------------------------------------------------------

        "161": {
            "class_type": "MMH3TemporalSplitParams",
            "inputs": {
                "chunk_length": 1020,
                "temporal_overlap": 34,
                "anchor_strength": 0.999,
            },
        },

        # --------------------------------------------------------
        # Spatial split
        # --------------------------------------------------------

        "162": {
            "class_type": "MMH3SpatialSplitParams",
            "inputs": {
                "tile_width": 512,
                "tile_height": 384,
                "overlap_width": 128,
                "overlap_height": 128,
                "fade_width": 32,
                "fade_height": 32,
                "min_tile_width": 256,
                "min_tile_height": 256,
                "overlap_mode": "earlier",
                "overlap_blend": "linear",
            },
        },

        # --------------------------------------------------------
        # Upscale noise
        # --------------------------------------------------------

        "165": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": 0,
            },
        },

        # --------------------------------------------------------
        # Upscale scheduler
        # --------------------------------------------------------

        "167": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": [
                    "173",
                    0,
                ],
                "scheduler": "simple",
                "steps": 6,
                "denoise": 0.18,
            },
        },

        # --------------------------------------------------------
        # Upscale sampler
        # --------------------------------------------------------

        "175": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler",
            },
        },

        # --------------------------------------------------------
        # Ultimate Upscale
        # --------------------------------------------------------

        "163": {
            "class_type": "MMH3UltimateUpscale",
            "inputs": {
                "model": [
                    "173",
                    0,
                ],
                "conditioning": [
                    "164",
                    0,
                ],
                "latent": [
                    "125",
                    0,
                ],
                "noise": [
                    "165",
                    0,
                ],
                "sampler": [
                    "175",
                    0,
                ],
                "sigmas": [
                    "167",
                    0,
                ],
                "negative": None,
                "latent_upscale_param": [
                    "160",
                    0,
                ],
                "temporal_split_param": [
                    "161",
                    0,
                ],
                "spatial_split_param": [
                    "162",
                    0,
                ],
                "cfg": 1,
            },
        },

        # --------------------------------------------------------
        # Final video VAE decode
        # --------------------------------------------------------

        "170": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [
                    "163",
                    0,
                ],
                "vae": [
                    "119",
                    0,
                ],
            },
        },

        # --------------------------------------------------------
        # Final audio decode
        # --------------------------------------------------------

        "171": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": [
                    "163",
                    0,
                ],
                "vae": [
                    "120",
                    0,
                ],
            },
        },

        # --------------------------------------------------------
        # Final CreateVideo
        # --------------------------------------------------------

        "172": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": [
                    "170",
                    0,
                ],
                "audio": [
                    "171",
                    0,
                ],
                "fps": 24,
                "bit_depth": 8,
            },
        },

        # --------------------------------------------------------
        # ORIGINAL OUTPUT
        # --------------------------------------------------------

        "155": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [
                    "130",
                    0,
                ],
                "filename_prefix": "video/MiniMax_H3_Original",
            },
        },

        # --------------------------------------------------------
        # FINAL UPSCALED OUTPUT
        # --------------------------------------------------------

        "92": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": [
                    "172",
                    0,
                ],
                "filename_prefix": "video/MiniMax_H3_UltimateUpscale",
            },
        },

        # --------------------------------------------------------
        # ORIGINAL VIDEO
        # --------------------------------------------------------

        "130": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": [
                    "122",
                    0,
                ],
                "audio": [
                    "121",
                    0,
                ],
                "fps": 24,
                "bit_depth": 8,
            },
        },

        # --------------------------------------------------------
        # Original decode
        # --------------------------------------------------------

        "122": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": [
                    "125",
                    0,
                ],
                "vae": [
                    "119",
                    0,
                ],
            },
        },

        "121": {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": [
                    "125",
                    0,
                ],
                "vae": [
                    "120",
                    0,
                ],
            },
        },
    }

    # ------------------------------------------------------------
    # Optional first-frame / last-frame
    #
    # Only add them when supplied.
    # ------------------------------------------------------------

    if first_name is not None:

        workflow["131"]["inputs"]["start_image"] = [
            first_name,
            0,
        ]

        workflow["164"]["inputs"]["start_image"] = [
            first_name,
            0,
        ]

    if last_name is not None:

        workflow["131"]["inputs"]["end_image"] = [
            last_name,
            0,
        ]

        workflow["164"]["inputs"]["end_image"] = [
            last_name,
            0,
        ]

    # ------------------------------------------------------------
    # Optional LoRA
    # ------------------------------------------------------------

    if lora_name:

        lora_path = (
            COMFY_ROOT /
            "models" /
            "loras" /
            lora_name
        )

        if not lora_path.exists():
            raise RuntimeError(
                f"LoRA not found: {lora_path}"
            )

        workflow["127"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": [
                    "127",
                    0,
                ],
                "lora_name": lora_name,
                "strength_model": float(
                    lora_strength
                ),
            },
        }

    return workflow


# ============================================================
# GENERATE
# ============================================================

def generate(
    prompt,
    duration,
    aspect_ratio,
    first_frame,
    last_frame,
    lora_name,
    lora_strength,
    progress=gr.Progress(),
):

    try:

        if not prompt or not prompt.strip():
            raise gr.Error(
                "Please enter a prompt."
            )

        progress(
            0.05,
            desc="Checking ComfyUI..."
        )

        if not port_open():
            start_comfy()

        progress(
            0.10,
            desc="Building workflow..."
        )

        log("=" * 70)
        log("NEW GENERATION")
        log("=" * 70)

        workflow = build_workflow(
            prompt=prompt.strip(),
            duration=float(duration),
            aspect_ratio=aspect_ratio,
            first_frame=first_frame,
            last_frame=last_frame,
            lora_name=lora_name or None,
            lora_strength=float(lora_strength),
        )

        # --------------------------------------------------------
        # DEBUG: print workflow before queueing
        # --------------------------------------------------------

        log(
            "Workflow node count: "
            + str(len(workflow))
        )

        progress(
            0.15,
            desc="Sending workflow to ComfyUI..."
        )

        prompt_id = queue_prompt(
            workflow
        )

        progress(
            0.20,
            desc="Generating video..."
        )

        history = wait_history(
            prompt_id
        )

        # --------------------------------------------------------
        # Print completion status
        # --------------------------------------------------------

        log(
            "FINAL HISTORY STATUS:"
        )

        print(
            json.dumps(
                history.get("status", {}),
                indent=2,
                default=str,
            ),
            flush=True,
        )

        progress(
            0.90,
            desc="Fetching generated video..."
        )

        result = fetch_video(
            history
        )

        progress(
            1.0,
            desc="Done!"
        )

        log(
            "Generation completed successfully."
        )

        return result

    except gr.Error:
        raise

    except Exception as e:

        log("=" * 70)
        log("GENERATION FAILED")
        log("=" * 70)

        log(
            f"{type(e).__name__}: {e}"
        )

        raise gr.Error(
            str(e)
        )


# ============================================================
# GRADIO UI
# ============================================================

def get_loras():

    lora_dir = (
        COMFY_ROOT /
        "models" /
        "loras"
    )

    if not lora_dir.exists():
        return []

    return sorted(
        x.name
        for x in lora_dir.iterdir()
        if x.is_file()
        and x.suffix.lower() in (
            ".safetensors",
            ".pt",
            ".ckpt",
        )
    )


with gr.Blocks(
    title="MiniMax H3 Video Generator"
) as demo:

    gr.Markdown(
        """
# MiniMax H3 Video Generator

Generate MiniMax H3 videos using the 10Eros Turbo model and
MiniMax H3 Ultimate Upscale.
"""
    )

    with gr.Row():

        with gr.Column():

            prompt = gr.Textbox(
                label="Prompt",
                placeholder=(
                    "Describe the video you want to generate..."
                ),
                lines=8,
            )

            with gr.Row():

                duration = gr.Number(
                    label="Duration (seconds)",
                    value=5,
                    minimum=1,
                    maximum=20,
                    step=1,
                )

                aspect_ratio = gr.Dropdown(
                    label="Aspect Ratio",
                    choices=[
                        "16:9",
                        "9:16",
                        "1:1",
                        "4:3",
                        "3:4",
                    ],
                    value="9:16",
                )

            with gr.Row():

                first_frame = gr.Image(
                    label="First Frame (optional)",
                    type="filepath",
                )

                last_frame = gr.Image(
                    label="Last Frame (optional)",
                    type="filepath",
                )

            with gr.Row():

                lora_name = gr.Dropdown(
                    label="LoRA (optional)",
                    choices=get_loras(),
                    value=None,
                    allow_custom_value=False,
                )

                lora_strength = gr.Slider(
                    label="LoRA Strength",
                    minimum=0,
                    maximum=2,
                    value=1,
                    step=0.05,
                )

            generate_button = gr.Button(
                "Generate Video",
                variant="primary",
            )

        with gr.Column():

            output_video = gr.Video(
                label="Generated Video",
                autoplay=True,
                interactive=False,
            )

    generate_button.click(
        fn=generate,
        inputs=[
            prompt,
            duration,
            aspect_ratio,
            first_frame,
            last_frame,
            lora_name,
            lora_strength,
        ],
        outputs=output_video,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    startup()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )

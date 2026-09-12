import os
import sys
import time
import json
import uuid
import threading
import subprocess
import requests
import gradio as gr

COMFY_HOST = "127.0.0.1:8188"
MODELS_DIR = "/root/ComfyUI/models"
VOLUME_DIR = "/root/models"  # Path where Modal Volume is mounted

# ---------------------------------------------------------
# 1. Symlink Models from Modal Volume to ComfyUI
# ---------------------------------------------------------
def setup_model_symlinks():
    """Symlinks pre-downloaded models from Modal volume into ComfyUI directory structure."""
    mappings = {
        "diffusion_models": "diffusion_models",
        "text_encoders": "text_encoders",
        "vae": "vae"
    }
    
    for vol_folder, comfy_folder in mappings.items():
        src_dir = os.path.join(VOLUME_DIR, vol_folder)
        dst_dir = os.path.join(MODELS_DIR, comfy_folder)
        
        if os.path.exists(src_dir):
            os.makedirs(dst_dir, exist_ok=True)
            for f in os.listdir(src_dir):
                src_file = os.path.join(src_dir, f)
                dst_file = os.path.join(dst_dir, f)
                if not os.path.exists(dst_file):
                    os.symlink(src_file, dst_file)
                    print(f"[Setup] Symlinked: {f} -> {comfy_folder}")

# ---------------------------------------------------------
# 2. ComfyUI Server Management
# ---------------------------------------------------------
def start_comfyui():
    """Launches ComfyUI server in a background thread."""
    def run():
        cmd = [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    
    print("[ComfyUI] Starting server...")
    while True:
        try:
            r = requests.get(f"http://{COMFY_HOST}/system_stats")
            if r.status_code == 200:
                print("[ComfyUI] Server online!")
                break
        except Exception:
            pass
        time.sleep(2)

# ---------------------------------------------------------
# 3. ComfyUI API Execution Engine
# ---------------------------------------------------------
def upload_image_to_comfy(image_path):
    if not image_path:
        return None
    url = f"http://{COMFY_HOST}/upload/image"
    with open(image_path, "rb") as f:
        response = requests.post(url, files={"image": f})
    if response.status_code == 200:
        return response.json().get("name")
    return None

def build_workflow_prompt(first_frame_name, last_frame_name, prompt, width, height, duration, seed):
    # Calculates MiniMax H3 valid frame count (snapping to 17-frame grid)[cite: 1]
    frames = max(5, round(duration * 24)) + (5 - (max(5, round(duration * 24)) % 17)) % 17
    
    graph = {
        "6": {
            "inputs": {
                "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        },
        "11": {
            "inputs": {
                "vae_name": "minimax_h3_video_vae_fp16.safetensors"
            },
            "class_type": "VAELoader"
        },
        "24": {
            "inputs": {
                "vae_name": "minimax_h3_audio_vae_fp32.safetensors"
            },
            "class_type": "VAELoader"
        },
        "13": {
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default"
            },
            "class_type": "CLIPLoader"
        },
        "15": {
            "inputs": {
                "noise_seed": seed
            },
            "class_type": "RandomNoise"
        },
        "17": {
            "inputs": {
                "sampler_name": "res_multistep"
            },
            "class_type": "KSamplerSelect"
        },
        "9": {
            "inputs": {
                "model": ["6", 0],
                "scheduler": "simple",
                "steps": 20,
                "denoise": 1.0
            },
            "class_type": "BasicScheduler"
        },
        "104": {
            "inputs": {
                "clip": ["13", 0],
                "vae": ["11", 0],
                "first_frame": ["114", 0] if first_frame_name else None,
                "last_frame": ["115", 0] if last_frame_name else None,
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": frames
            },
            "class_type": "MiniMaxH3ImageToVideo"
        },
        "16": {
            "inputs": {
                "model": ["6", 0],
                "conditioning": ["104", 0]
            },
            "class_type": "BasicGuider"
        },
        "14": {
            "inputs": {
                "noise": ["15", 0],
                "guider": ["16", 0],
                "sampler": ["17", 0],
                "sigmas": ["9", 0],
                "latent_image": ["104", 1]
            },
            "class_type": "SamplerCustomAdvanced"
        },
        "10": {
            "inputs": {
                "samples": ["14", 0],
                "vae": ["11", 0]
            },
            "class_type": "VAEDecode"
        },
        "23": {
            "inputs": {
                "samples": ["14", 0],
                "vae": ["24", 0]
            },
            "class_type": "VAEDecodeAudio"
        },
        "91": {
            "inputs": {
                "images": ["10", 0],
                "audio": ["23", 0],
                "fps": 24,
                "bit_depth": 8
            },
            "class_type": "CreateVideo"
        },
        "92": {
            "inputs": {
                "video": ["91", 0],
                "filename_prefix": "video/MiniMax_H3",
                "format": "auto",
                "codec": "auto"
            },
            "class_type": "SaveVideo"
        }
    }

    if first_frame_name:
        graph["114"] = {"inputs": {"image": first_frame_name}, "class_type": "LoadImage"}
    if last_frame_name:
        graph["115"] = {"inputs": {"image": last_frame_name}, "class_type": "LoadImage"}

    return graph

def generate_video(first_frame, last_frame, prompt, resolution_str, duration, seed):
    w, h = map(int, resolution_str.split("x"))
    
    first_name = upload_image_to_comfy(first_frame) if first_frame else None
    last_name = upload_image_to_comfy(last_frame) if last_frame else None
    
    prompt_payload = build_workflow_prompt(first_name, last_name, prompt, w, h, duration, int(seed))
    
    client_id = str(uuid.uuid4())
    res = requests.post(f"http://{COMFY_HOST}/prompt", json={"prompt": prompt_payload, "client_id": client_id})
    prompt_id = res.json().get("prompt_id")

    while True:
        history = requests.get(f"http://{COMFY_HOST}/history/{prompt_id}").json()
        if prompt_id in history:
            outputs = history[prompt_id].get("outputs", {})
            for node_id, out_data in outputs.items():
                if "gifs" in out_data or "videos" in out_data:
                    vid_info = (out_data.get("gifs") or out_data.get("videos"))[0]
                    video_url = f"http://{COMFY_HOST}/view?filename={vid_info['filename']}&subfolder={vid_info['subfolder']}&type={vid_info['type']}"
                    video_content = requests.get(video_url).content
                    
                    out_path = f"/tmp/output_{prompt_id}.mp4"
                    with open(out_path, "wb") as f:
                        f.write(video_content)
                    return out_path
        time.sleep(2)

# ---------------------------------------------------------
# 4. Gradio UI Initialization
# ---------------------------------------------------------
if __name__ == "__main__":
    setup_model_symlinks()
    start_comfyui()

    with gr.Blocks(title="MiniMax H3 Video Generator") as demo:
        gr.Markdown("# 🎬 MiniMax H3 Video Generator (ComfyUI Engine)")
        
        with gr.Row():
            with gr.Column():
                first_frame = gr.Image(type="filepath", label="First Frame (Optional)")
                last_frame = gr.Image(type="filepath", label="Last Frame (Optional)")
                prompt = gr.Textbox(
                    lines=5, 
                    label="Prompt & Audio Description",
                    value="Editorial tech product film. Dark studio background, blue and neon orange rim lighting..."
                )
                
                with gr.Row():
                    resolution = gr.Dropdown(
                        choices=["1344x768", "768x1344", "1024x1024", "864x480"],
                        value="1344x768",
                        label="Resolution"
                    )
                    duration = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Duration (s)")
                
                seed = gr.Number(value=42, label="Seed", precision=0)
                generate_btn = gr.Button("Generate Video", variant="primary")
            
            with gr.Column():
                output_video = gr.Video(label="Generated Video (Native Audio)")
        
        generate_btn.click(
            fn=generate_video,
            inputs=[first_frame, last_frame, prompt, resolution, duration, seed],
            outputs=[output_video]
        )

    # Launch Gradio with a public share link
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
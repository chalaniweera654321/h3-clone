import os, sys, time, uuid, json, subprocess
from pathlib import Path
import gradio as gr
import requests

# Simple paths for your Modal notebook.
COMFY_ROOT = Path('/root/ComfyUI')
MODEL_ROOT = Path('/mnt/minimax-h3-models')
COMFY_URL = 'http://127.0.0.1:8188'
VIDEO_MODEL = '10Eros_Max_h3_TURBO-hybrid_beta3_int8_convrot_skip_edges.safetensors'
TEXT_ENCODER = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
VIDEO_VAE = 'minimax_h3_video_vae_fp16.safetensors'
AUDIO_VAE = 'minimax_h3_audio_vae_fp32.safetensors'
UPSCALER = 'minimax_h3_latent_upscaler_3d_fp16.safetensors'
PROCESS = None
RUNNING = False

DEFAULT_PROMPT = '''integrated_multimodal_description: [Shot 1] Photorealistic cinematic shot, 85mm lens at f/1.8, shallow depth of field with creamy bokeh. A beautiful young woman in her mid-twenties stands on a weathered wooden causeway stretching over a windswept beach, leaning back casually against the salt-faded white railing with one elbow resting on it, her weight shifted onto one hip in an effortlessly confident, playful stance.

She has vibrant honey-red auburn hair catching the golden hour light, loose strands whipping gently across her face and lifting in the warm ocean breeze, sunlit edges glowing like copper filaments against the sky. Her expression is a soft, genuine half-smile with slightly squinted eyes from the sunlight, radiating warmth, mischief, and quiet self-assurance. Faint freckles dust her nose and cheekbones.

She wears a flowing ivory-and-blush chiffon sundress that billows and ripples in the wind, fabric translucent where the low sun shines through it, hugging her figure then trailing away like liquid silk.

Behind her the Atlantic horizon meets a dramatic sky painted in peach, apricot, and lavender, soft cirrus clouds streaked gold. Distant waves break in slow white foam; wet sand reflects the sunset like polished glass.

Warm late-afternoon sunlight rakes across her face from camera-left, luminous rim lighting on her hair and shoulders. Natural skin texture, Kodak Portra 400 color science, soft film grain, cinematic teal-orange grading. The camera holds a medium-wide shot along the causeway, slightly below eye level. She looks toward the camera with a relaxed smile, then turns her face toward the sea. Single continuous shot, no cuts, no on-screen text.

overall_soundscape: Soft ocean waves, distant seagulls, light wind across the wooden railing, chiffon fabric rustling.

non_diegetic_music: Gentle acoustic guitar, warm and unobtrusive.'''


def log(x): print('[H3]', x, flush=True)


def model_path(category, name):
    p = MODEL_ROOT / category / name
    if p.exists(): return p
    found = list(MODEL_ROOT.rglob(name))
    return found[0] if found else None


def setup_models():
    log('STEP 1/7 - Checking model volume')
    if not COMFY_ROOT.exists(): raise RuntimeError(f'ComfyUI not found: {COMFY_ROOT}')
    needed = [('diffusion_models', VIDEO_MODEL), ('text_encoders', TEXT_ENCODER),
              ('vae', VIDEO_VAE), ('vae', AUDIO_VAE), ('latent_upscale_models', UPSCALER)]
    missing = []
    for cat, name in needed:
        p = model_path(cat, name)
        log(('OK   ' if p else 'MISS ') + f'{cat}/{name}')
        if not p: missing.append(f'{cat}/{name}')
    if missing: raise RuntimeError('Missing models:\n' + '\n'.join('  '+x for x in missing))

    log('STEP 2/7 - Making Modal Volume visible to ComfyUI')
    root = COMFY_ROOT / 'models'; root.mkdir(exist_ok=True)
    for cat in ['diffusion_models','text_encoders','vae','latent_upscale_models','loras']:
        src = MODEL_ROOT / cat; src.mkdir(parents=True, exist_ok=True)
        dst = root / cat
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
        elif dst.is_dir() and not dst.is_symlink():
            for f in src.rglob('*'):
                if f.is_file():
                    out = dst / f.relative_to(src); out.parent.mkdir(parents=True, exist_ok=True)
                    if not out.exists():
                        try: out.symlink_to(f)
                        except Exception: pass


def install_extra_node():
    log('STEP 3/7 - Checking MMH3 Ultimate Upscale node')
    custom = COMFY_ROOT / 'custom_nodes'; custom.mkdir(exist_ok=True)
    candidates = [custom/'Comfyui-MMH3-UltimateUpscale', custom/'ComfyUI-MMH3-UltimateUpscale']
    if any(x.exists() for x in candidates): return
    log('Node pack missing; installing it...')
    subprocess.run(['git','clone','--depth','1',
                    'https://github.com/bbaudio-2025/Comfyui-MMH3-UltimateUpscale',
                    str(custom/'Comfyui-MMH3-UltimateUpscale')], check=True)


def start_comfy():
    global PROCESS
    log('STEP 4/7 - Starting ComfyUI')
    PROCESS = subprocess.Popen([
        sys.executable, str(COMFY_ROOT/'main.py'), '--listen','127.0.0.1',
        '--port','8188','--lowvram','--force-fp16','--use-ck-attention'
    ], cwd=COMFY_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    end = time.time()+180
    while time.time() < end:
        if PROCESS.poll() is not None: raise RuntimeError('ComfyUI stopped during startup.')
        try:
            if requests.get(COMFY_URL+'/system_stats', timeout=2).ok:
                log('ComfyUI is ready.'); return
        except Exception: pass
        time.sleep(1)
    raise TimeoutError('ComfyUI did not start in 180 seconds.')


def stop_comfy():
    if PROCESS is not None and PROCESS.poll() is None:
        try: PROCESS.terminate(); PROCESS.wait(10)
        except Exception:
            try: PROCESS.kill()
            except Exception: pass


def upload_image(path):
    if not path: return None
    with open(path,'rb') as f:
        r=requests.post(COMFY_URL+'/upload/image', files={'image':(Path(path).name,f)},
                        data={'overwrite':'true','type':'input'}, timeout=120)
    r.raise_for_status(); d=r.json()
    return f"{d.get('subfolder','')}/{d['name']}".lstrip('/')


def frames(seconds):
    n=max(5,round(float(seconds)*24))
    return n+(5-(n%17))%17


def size_for(aspect, mp):
    ratios={'9:16':9/16,'16:9':16/9,'1:1':1,'4:3':4/3,'3:4':3/4}
    ratio=ratios[aspect]; pixels=float(mp)*1_000_000
    w=(pixels*ratio)**0.5; h=w/ratio
    return max(32,round(w/32)*32), max(32,round(h/32)*32)


def workflow(prompt,s1w,s1h,s2w,s2h,duration,seed,first,last,loras):
    length=frames(duration)
    w={
      '119':{'class_type':'VAELoader','inputs':{'vae_name':VIDEO_VAE}},
      '120':{'class_type':'VAELoader','inputs':{'vae_name':AUDIO_VAE}},
      '127':{'class_type':'UNETLoader','inputs':{'unet_name':VIDEO_MODEL,'weight_dtype':'default'}},
      '128':{'class_type':'CLIPLoader','inputs':{'clip_name':TEXT_ENCODER,'type':'minimax','device':'default'}},
      '144':{'class_type':'MiniMaxH3SigmaShift','inputs':{'model':['127',0],'shift_video':12,'shift_audio':3}},
      '173':{'class_type':'ModelAttentionBackend','inputs':{'model':['144',0],'attention':'comfy kitchen attention'}},
      '123':{'class_type':'KSamplerSelect','inputs':{'sampler_name':'euler'}},
      '124':{'class_type':'BasicScheduler','inputs':{'model':['173',0],'scheduler':'simple','steps':6,'denoise':1.0}},
      '129':{'class_type':'RandomNoise','inputs':{'noise_seed':int(seed)}},
      '131':{'class_type':'MiniMaxH3ImageToVideo','inputs':{'clip':['128',0],'vae':['119',0],'prompt':prompt,'width':int(s1w),'height':int(s1h),'length':int(length)}},
      '126':{'class_type':'BasicGuider','inputs':{'model':['173',0],'conditioning':['131',0]}},
      '125':{'class_type':'SamplerCustomAdvanced','inputs':{'noise':['129',0],'guider':['126',0],'sampler':['123',0],'sigmas':['124',0],'latent_image':['131',1]}},
      '122':{'class_type':'VAEDecode','inputs':{'samples':['125',0],'vae':['119',0]}},
      '121':{'class_type':'VAEDecodeAudio','inputs':{'samples':['125',0],'vae':['120',0]}},
      '130':{'class_type':'CreateVideo','inputs':{'images':['122',0],'audio':['121',0],'fps':24,'bit_depth':8}},
      '164':{'class_type':'MiniMaxH3ImageToVideo','inputs':{'clip':['128',0],'vae':['119',0],'prompt':prompt,'width':int(s2w),'height':int(s2h),'length':int(length)}},
      '160':{'class_type':'MMH3LatentUpscaleWithModelParams','inputs':{'model_name':UPSCALER,'width':int(s2w),'height':int(s2h),'device':'cuda','precision':'fp16'}},
      '161':{'class_type':'MMH3TemporalSplitParams','inputs':{'chunk_length':1020,'temporal_overlap':34,'anchor_strength':0.999}},
      '162':{'class_type':'MMH3SpatialSplitParams','inputs':{'tile_width':512,'tile_height':384,'spatial_w_overlap':128,'spatial_h_overlap':128,'fade_width':32,'fade_height':32,'min_tile_size':256,'overlap_mode':'earlier','overlap_blend':'linear'}},
      '165':{'class_type':'RandomNoise','inputs':{'noise_seed':int(seed)+1}},
      '175':{'class_type':'KSamplerSelect','inputs':{'sampler_name':'euler'}},
      '167':{'class_type':'BasicScheduler','inputs':{'model':['173',0],'scheduler':'simple','steps':6,'denoise':0.18}},
      '163':{'class_type':'MMH3UltimateUpscale','inputs':{'model':['173',0],'conditioning':['164',0],'latent':['125',0],'noise':['165',0],'sampler':['175',0],'sigmas':['167',0],'negative':None,'latent_upscale_param':['160',0],'temporal_split_param':['161',0],'spatial_split_param':['162',0],'cfg':1}},
      '170':{'class_type':'VAEDecode','inputs':{'samples':['163',0],'vae':['119',0]}},
      '171':{'class_type':'VAEDecodeAudio','inputs':{'samples':['163',0],'vae':['120',0]}},
      '172':{'class_type':'CreateVideo','inputs':{'images':['170',0],'audio':['171',0],'fps':24,'bit_depth':8}},
      '155':{'class_type':'SaveVideo','inputs':{'video':['130',0],'filename_prefix':'video/MiniMax_H3_Original','format':'auto','codec':'auto'}},
      '92':{'class_type':'SaveVideo','inputs':{'video':['172',0],'filename_prefix':'video/MiniMax_H3_UltimateUpscale','format':'auto','codec':'auto'}},
    }
    if first:
        w['900']={'class_type':'LoadImage','inputs':{'image':first}}
        w['131']['inputs']['first_frame']=['900',0]; w['164']['inputs']['first_frame']=['900',0]
    if last:
        w['901']={'class_type':'LoadImage','inputs':{'image':last}}
        w['131']['inputs']['last_frame']=['901',0]; w['164']['inputs']['last_frame']=['901',0]
    previous='127'
    for i,(name,weight) in enumerate(loras):
        nid=str(180+i)
        w[nid]={'class_type':'LoraLoaderModelOnly','inputs':{'model':[previous,0],'lora_name':name,'strength_model':float(weight)}}
        previous=nid
    if loras: w['144']['inputs']['model']=[previous,0]
    return w


def submit(w):
    r=requests.post(COMFY_URL+'/prompt',json={'prompt':w,'client_id':str(uuid.uuid4())},timeout=120)
    if not r.ok: raise RuntimeError(f'ComfyUI rejected workflow:\n{r.text}')
    d=r.json()
    if d.get('error'): raise RuntimeError(json.dumps(d,indent=2))
    return d['prompt_id']


def wait_history(pid):
    end=time.time()+3600
    while time.time()<end:
        try:
            r=requests.get(COMFY_URL+f'/history/{pid}',timeout=10)
            if r.ok and pid in r.json(): return r.json()[pid]
        except Exception: pass
        time.sleep(2)
    raise TimeoutError('Timed out waiting for ComfyUI.')


def fetch_video(history):
    items=[]
    for out in history.get('outputs',{}).values():
        for key in ('videos','gifs','files'):
            items += out.get(key,[]) or []
    if not items: raise RuntimeError('ComfyUI finished but returned no video.')
    item=next((x for x in items if 'UltimateUpscale' in x.get('filename','')),items[-1])
    r=requests.get(COMFY_URL+'/view',params={'filename':item['filename'],'subfolder':item.get('subfolder',''),'type':item.get('type','output')},timeout=600)
    r.raise_for_status()
    out=Path('/tmp/minimax_h3_results'); out.mkdir(exist_ok=True)
    p=out/(uuid.uuid4().hex+'_'+item['filename']); p.write_bytes(r.content)
    return str(p)


def lora_list():
    d=MODEL_ROOT/'loras'
    if not d.exists(): return []
    return sorted(str(x.relative_to(d)).replace('\\','/') for x in d.rglob('*') if x.is_file() and x.suffix.lower() in {'.safetensors','.ckpt','.pt','.bin'})


def generate(prompt,aspect,s1mp,s2mp,duration,seed,randomize,first,last,*weights):
    global RUNNING
    if RUNNING: raise gr.Error('Another generation is already running.')
    RUNNING=True
    try:
        if not prompt.strip(): raise gr.Error('Prompt is empty.')
        seed=int.from_bytes(os.urandom(8),'big')%(2**63-1) if randomize else int(seed)
        s1w,s1h=size_for(aspect,s1mp); s2w,s2h=size_for(aspect,s2mp)
        first=upload_image(first) if first else None; last=upload_image(last) if last else None
        names=lora_list(); loras=[(n,float(v)) for n,v in zip(names,weights) if abs(float(v))>0.0001]
        log('='*70); log(f'Stage 1: {s1w}x{s1h}'); log(f'Stage 2: {s2w}x{s2h}'); log(f'Frames: {frames(duration)}'); log(f'Seed: {seed}')
        log(f'LoRAs: {len(loras)}'); log('STEP 5/7 - Building workflow')
        wf=workflow(prompt,s1w,s1h,s2w,s2h,duration,seed,first,last,loras)
        pid=submit(wf); log(f'STEP 6/7 - Running: {pid}')
        debug=COMFY_ROOT/'output'/'_h3_workflows'; debug.mkdir(parents=True,exist_ok=True)
        (debug/f'{pid}.json').write_text(json.dumps(wf,indent=2),encoding='utf-8')
        history=wait_history(pid)
        if history.get('status',{}).get('status_str')=='error': raise RuntimeError(json.dumps(history['status'].get('messages',[]),indent=2))
        log('STEP 7/7 - Getting final video')
        result=fetch_video(history)
        return result,f'Done — seed {seed} — {s1w}×{s1h} → {s2w}×{s2h}'
    except Exception as e:
        log('ERROR: '+str(e)); raise gr.Error(str(e))
    finally: RUNNING=False


def refresh_loras():
    names=lora_list(); log(f'Found {len(names)} LoRA(s).')
    return [gr.update(visible=i<len(names),label=f'LoRA {i+1}: {names[i]}' if i<len(names) else f'LoRA {i+1}',value=0) for i in range(20)]


def create_ui():
    names=lora_list()
    with gr.Blocks(title='MiniMax H3 — 10Eros Ultimate Upscale') as demo:
        gr.Markdown('# MiniMax H3 — 10Eros Ultimate Upscale\n**10Eros TURBO → H3 generation → 3D latent Ultimate Upscale → final video**')
        with gr.Row():
            with gr.Column(scale=2):
                prompt=gr.Textbox(label='Prompt',value=DEFAULT_PROMPT,lines=16)
                with gr.Row():
                    first=gr.Image(label='First frame (optional)',type='filepath')
                    last=gr.Image(label='Last frame (optional)',type='filepath')
                with gr.Row():
                    aspect=gr.Dropdown(['9:16','16:9','1:1','4:3','3:4'],value='9:16',label='Aspect ratio')
                    duration=gr.Slider(1,20,5,step=1,label='Duration (seconds)')
                with gr.Row():
                    s1mp=gr.Slider(.20,1.00,.40,step=.05,label='Stage 1 MP')
                    s2mp=gr.Slider(.30,1.20,.90,step=.05,label='Stage 2 MP')
                with gr.Row():
                    seed=gr.Number(value=757358688076805,precision=0,label='Seed')
                    randomize=gr.Checkbox(True,label='Randomize seed')
                gr.Markdown('### Style LoRAs — weight 0 disables\n**Do not use speed LoRAs. 10Eros already has turbo baked in.**')
                sliders=[]
                for i in range(20):
                    sliders.append(gr.Slider(-3,3,0,step=.05,visible=i<len(names),label=f'LoRA {i+1}: {names[i]}' if i<len(names) else f'LoRA {i+1}'))
                with gr.Row():
                    go=gr.Button('Generate',variant='primary'); refresh=gr.Button('Refresh LoRAs')
            with gr.Column(scale=1):
                out=gr.Video(label='Final Ultimate Upscale',autoplay=True)
                status=gr.Textbox(label='Status',interactive=False)
                gr.Markdown('### Workflow defaults\n- Stage 1: ~0.4 MP\n- Stage 2: ~0.9 MP\n- euler / simple\n- 6 + 6 steps\n- upscale denoise 0.18\n- tiles 512×384\n- temporal 1020 / 34 / 0.999\n- 24 FPS')
        go.click(generate,[prompt,aspect,s1mp,s2mp,duration,seed,randomize,first,last,*sliders],[out,status])
        refresh.click(refresh_loras,[],sliders)
    return demo


def main():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
    setup_models(); install_extra_node(); start_comfy()
    try:
        create_ui().launch(server_name='0.0.0.0',server_port=7860,share=True,show_error=True)
    finally: stop_comfy()

if __name__=='__main__': main()

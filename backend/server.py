"""
V6rge Local Backend Server
Runs on localhost:5000 with all AI models locally
"""

import os
import sys
# Force line buffering for stdout/stderr to ensure logs appear in Electron console
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import os
import subprocess
import shutil
import stat
import tempfile
import traceback
import gc
import uuid
import json
import re
import base64
import random
import threading
import time
import queue # For thread-safe communication
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Flask
from flask import Flask, request, send_file, jsonify, Response
from flask_cors import CORS
from flask_sock import Sock
from werkzeug.exceptions import HTTPException

# Model imports (lazy loaded)
from PIL import Image
import cv2
import scipy.io.wavfile as wavfile
import soundfile as sf

# ==================================================================================
# TQDM PATCHING - Must happen BEFORE huggingface_hub import
# ==================================================================================
# Global variables to store download progress
DOWNLOAD_PROGRESS = {}
CURRENT_DOWNLOAD_ID = None

import tqdm
import tqdm.auto

# Store originals
_original_tqdm = tqdm.tqdm
_original_auto_tqdm = tqdm.auto.tqdm

# Global checks for cancellation
CANCELLATION_REQUESTED = set()

class WrapperTqdm(tqdm.auto.tqdm):
    """Custom tqdm that reports progress to our global dict"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def update(self, n=1):
        super().update(n)
        if CURRENT_DOWNLOAD_ID and CURRENT_DOWNLOAD_ID in DOWNLOAD_PROGRESS:
            # CHECK CANCELLATION
            if CURRENT_DOWNLOAD_ID in CANCELLATION_REQUESTED:
                raise KeyboardInterrupt("Download Cancelled by User")

            try:
                info = self.format_dict
                total = self.total if self.total else 0
                downloaded = self.n
                rate = info.get('rate', 0)  # bytes/s
                remaining = (total - downloaded) / rate if rate and rate > 0 else 0
                
                DOWNLOAD_PROGRESS[CURRENT_DOWNLOAD_ID].update({
                    'status': 'downloading',
                    'progress': (downloaded / total) * 100 if total else 0,
                    'downloaded': downloaded,
                    'total': total,
                    'speed': rate,
                    'eta': remaining,
                    'message': f"{downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB"
                })
            except Exception:
                pass

# Apply the patch globally BEFORE huggingface_hub imports
tqdm.tqdm = WrapperTqdm
tqdm.auto.tqdm = WrapperTqdm

from huggingface_hub import snapshot_download, hf_hub_download, try_to_load_from_cache

# ==================================================================================
# CONFIGURATION
# ==================================================================================

# Configuration
BASE_DIR = Path.home() / '.v6rge'
CONFIG_FILE = BASE_DIR / 'config.json'

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(config):
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

config = load_config()
MODELS_DIR = Path(config.get('models_dir', str(BASE_DIR)))
MODELS_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists

UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'output'
VOICES_FOLDER = BASE_DIR / 'voices'
MODELS_3D_FOLDER = BASE_DIR / '3d_models'

# Create directories
for folder in [BASE_DIR, MODELS_DIR, UPLOAD_FOLDER, OUTPUT_FOLDER, VOICES_FOLDER, MODELS_3D_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# ==================================================================================
# ENHANCED GPU DETECTION WITH DIAGNOSTICS
# ==================================================================================

print("\n" + "="*70)
print("V6RGE BACKEND - GPU DETECTION")
print("="*70)

# Check for CUDA availability
if torch.cuda.is_available():
    DEVICE = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    compute_cap = torch.cuda.get_device_capability(0)
    
    print(f"✅ GPU DETECTED: {gpu_name}")
    print(f"   VRAM: {gpu_mem:.1f} GB")
    print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
    
    # CUDA Version Check
    try:
        cuda_version = torch.version.cuda
        print(f"   CUDA Version: {cuda_version}")
        
        # RTX 50-series requires CUDA 12.6+ (they have compute capability 10.0)
        if compute_cap[0] >= 10:
            print(f"\n⚠️  RTX 50-SERIES DETECTED")
            print(f"   Your GPU ({gpu_name}) requires CUDA 12.6 or later.")
            if cuda_version and float(cuda_version.split('.')[0] + '.' + cuda_version.split('.')[1]) < 12.6:
                print(f"   ⚠️  WARNING: Current PyTorch built with CUDA {cuda_version}")
                print(f"   You may experience performance issues or errors.")
                print(f"   Recommended: Install PyTorch with CUDA 12.6+")
                print(f"   Visit: https://pytorch.org/get-started/locally/")
        
        # General CUDA version guidance
        elif cuda_version and float(cuda_version.split('.')[0] + '.' + cuda_version.split('.')[1]) < 12.1:
            print(f"   ⚠️  Your CUDA version ({cuda_version}) is older than recommended (12.1+)")
            print(f"   Consider upgrading PyTorch for better performance.")
            
    except Exception as e:
        print(f"   Warning: Could not detect CUDA version: {e}")
    
    print(f"\n🚀 Running in GPU Mode")
    print("="*70 + "\n")
    
else:
    DEVICE = "cpu"
    print(f"ℹ️  NO CUDA GPU DETECTED - Running in CPU Mode")
    print(f"\nTroubleshooting GPU Detection:")
    print(f"  1. Ensure NVIDIA drivers are installed (Latest recommended)")
    print(f"  2. Verify PyTorch was installed with CUDA support:")
    print(f"     Run: python -c \"import torch; print(torch.version.cuda)\"")
    print(f"  3. For RTX 50-series GPUs, you need CUDA 12.6+")
    print(f"  4. Reinstall PyTorch with CUDA:")
    print(f"     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print(f"\n⚠️  CPU mode will be SIGNIFICANTLY SLOWER for:")
    print(f"     - Image generation (FLUX)")
    print(f"     - Video generation (HunyuanVideo)")
    print(f"     - Large language models (Qwen 72B)")
    print(f"     - Music generation (MusicGen)")
    print(f"\n💡 Recommended: Use smaller models like Qwen-3B for CPU")
    print("="*70 + "\n")


# ==================================================================================
# GLOBAL MODEL STORAGE
# ==================================================================================

# Models are loaded on-demand to save VRAM
MODELS = {
    'chat': None,
    'chat_tokenizer': None,
    'vision': None,
    'vision_processor': None,
    'image_gen': None,
    'image_edit': None,
    'music': None,
    'music_processor': None,
    'tts': None,
    'video': None,
    'vocal_sep': None,
    'upscaler': None,
    'bg_remover': None,
    '3d_shape': None,
    '3d_texture': None
}

CURRENT_LOADED = {} # Tracks currently loaded model IDs: {'image': 'flux-schnell', 'music': 'musicgen-large', ...}

# Download progress tracking
DOWNLOAD_PROGRESS = {}  # {model_id: {'status':..., 'progress':..., 'speed':..., 'total':..., 'downloaded':..., 'eta':...}}

class DownloadProgress:
    def __init__(self, model_id):
        self.model_id = model_id
        self.last_time = time.time()
        self.last_bytes = 0
        self.total_bytes = 0
        self.downloaded_bytes = 0
        
    def __call__(self, future_state):
        # future_state is a dict from tqdm (when using callback in snapshot_download)
        # However, huggingface_hub callbacks are simpler. 
        # We might need to wrap tqdm or use the 'tqdm_class' argument.
        pass
    
    # We will use a custom tqdm-like wrapper instead
    def update(self, n=1):
        self.downloaded_bytes += n
        current_time = time.time()
        
        # Update speed every 0.1s to avoid spam
        if current_time - self.last_time > 0.1:
            elapsed = current_time - self.last_time
            bytes_diff = self.downloaded_bytes - self.last_bytes
            speed = bytes_diff / elapsed if elapsed > 0 else 0
            
            eta = (self.total_bytes - self.downloaded_bytes) / speed if speed > 0 else 0
            
            DOWNLOAD_PROGRESS[self.model_id] = {
                'status': 'downloading',
                'progress': (self.downloaded_bytes / self.total_bytes) * 100 if self.total_bytes > 0 else 0,
                'downloaded': self.downloaded_bytes,
                'total': self.total_bytes,
                'speed': speed,
                'eta': eta,
                'message': f"{self.downloaded_bytes/1e6:.1f}MB / {self.total_bytes/1e6:.1f}MB"
            }
            
            self.last_time = current_time
            self.last_bytes = self.downloaded_bytes

CLONED_VOICES = {}
LAST_UPLOADED_IMAGE = None
TTS_SAMPLE_RATE = 24000

# ==================================================================================
# MODEL LOADING FUNCTIONS
# ==================================================================================

def load_chat_model(model_id='qwen-72b'):
    """Load Chat Model dynamically"""
    if MODELS['chat'] is not None and CURRENT_LOADED.get('chat') == model_id:
        return
        
    if model_id not in MODEL_REGISTRY:
        model_id = 'qwen-72b'
    config = MODEL_REGISTRY[model_id]

    if MODELS['chat'] is not None:
        print(f"Switching Chat Model...")
        del MODELS['chat']
        del MODELS['chat_tokenizer']
        MODELS['chat'] = None
        MODELS['chat_tokenizer'] = None
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"Loading {model_id}...")
    
    # === GGUF PRIORITY CHECK ===
    # Check for GGUF BEFORE trying to load AutoTokenizer, as GGUF repos often don't have tokenizer.json
    if config.get('filename') and config['filename'].endswith('.gguf'):
        print(f"Loading GGUF Model: {config['filename']} via llama.cpp", flush=True)
        try:
            from huggingface_hub import hf_hub_download
            import llama_cpp
            
            # Get physical path of the GGUF file
            model_path = hf_hub_download(
                repo_id=config['hf_name'],
                filename=config['filename'],
                cache_dir=MODELS_DIR
            )
            
            n_gpu_layers = -1 if DEVICE == 'cuda' else 0
            
            # Using 4096 context context window by default for Qwen 3B
            MODELS['chat'] = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=4096,
                verbose=False
            )
            # Mark as GGUF for chat endpoint
            MODELS['chat'].is_gguf = True
            
            CURRENT_LOADED['chat'] = model_id
            print(f"Chat Model (GGUF) Ready | Device: {DEVICE}\n", flush=True)
            return
            
        except Exception as e:
            print(f"❌ Failed to load GGUF: {e}", flush=True)
            traceback.print_exc()
            # CRITICAL: Do not fallthrough to transformers for GGUF files.
            raise e

    # === STANDARD TRANSFORMERS LOAD ===
    # Only if NOT GGUF or if fallthrough was desired (but we raise above)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    try:
        MODELS['chat_tokenizer'] = AutoTokenizer.from_pretrained(
            config['hf_name'],
            trust_remote_code=True,
            cache_dir=MODELS_DIR
        )
        
        # Dynamic device config
        use_cuda = (DEVICE == 'cuda')
        # Use float16 on CPU too to save RAM (Qwen 3B fits in ~6GB fp16, but ~12GB fp32)
        dtype = torch.float16 if use_cuda else torch.float16 
        
        # Enable auto device map for CPU too (requires accelerate) to handle offloading
        dmap = "auto" 
        
        print(f"DEBUG: Loading Chat Logic | Device: {DEVICE} | Dtype: {dtype} | Map: {dmap}", flush=True)

        kwargs = {
            "torch_dtype": dtype,
            "device_map": dmap,
            "trust_remote_code": True,
            "cache_dir": MODELS_DIR,
            "low_cpu_mem_usage": True
        }

        MODELS['chat'] = AutoModelForCausalLM.from_pretrained(
            config['hf_name'],
            **kwargs
        )
        MODELS['chat'].is_gguf = False
        
    except Exception as load_err:
        print(f"Standard load failed, retrying with fp32: {load_err}", flush=True)
        # Fallback to fp32 if fp16 fails on old CPUs
        MODELS['chat'] = AutoModelForCausalLM.from_pretrained(
            config['hf_name'],
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=MODELS_DIR,
            low_cpu_mem_usage=True
        )
        MODELS['chat'].is_gguf = False
    
    CURRENT_LOADED['chat'] = model_id
    
    mem_str = f"{torch.cuda.memory_allocated()/1e9:.2f}GB" if DEVICE == 'cuda' else "N/A (CPU)"
    print(f"Chat Model Ready | VRAM: {mem_str}\n", flush=True)

def load_vision_model():
    """Load Qwen2-VL Vision Model"""
    if MODELS['vision'] is not None:
        return
    
    print("Loading Qwen2-VL-7B (Vision)...")
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    # Dynamic device config
    use_cuda = (DEVICE == 'cuda')
    dtype = torch.float16 if use_cuda else torch.float32
    dmap = "auto" if use_cuda else None

    MODELS['vision'] = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=dtype,
        device_map=dmap,
        trust_remote_code=True,
        cache_dir=MODELS_DIR
    )
    
    if not use_cuda:
        MODELS['vision'].to(DEVICE)
    
    MODELS['vision_processor'] = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        cache_dir=MODELS_DIR
    )
    
    print(f"Vision Model Ready\n")

def load_image_gen(model_id='flux-schnell'):
    """Load Image Gen Model dynamically - supports FLUX and Qwen-Image"""
    # Check if already loaded
    if MODELS['image_gen'] is not None and CURRENT_LOADED.get('image') == model_id:
        return
    
    # Check registry
    if model_id not in MODEL_REGISTRY:
        print(f"Unknown model {model_id}, falling back to flux-schnell")
        model_id = 'flux-schnell'

    config = MODEL_REGISTRY[model_id]
    
    # Unload previous
    if MODELS['image_gen'] is not None:
        print(f"Unloading {CURRENT_LOADED.get('image')} to load {model_id}...")
        del MODELS['image_gen']
        if MODELS.get('image_edit'):
            del MODELS['image_edit']
        MODELS['image_gen'] = None
        MODELS['image_edit'] = None
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"Loading {model_id} ({config['hf_name']})...")
    
    try:
        # Qwen-Image uses standard DiffusionPipeline
        if model_id == 'qwen-image':
            from diffusers import DiffusionPipeline
            
            MODELS['image_gen'] = DiffusionPipeline.from_pretrained(
                config['hf_name'],
                torch_dtype=torch.bfloat16 if DEVICE == 'cuda' else torch.float32,
                cache_dir=MODELS_DIR,
                local_files_only=True
            )
            MODELS['image_gen'].to(DEVICE)
            MODELS['image_gen'].model_type = 'qwen'  # Mark for generate_image endpoint
            MODELS['image_edit'] = None  # Qwen handles editing differently
            
        # FLUX models
        else:
            from diffusers import FluxPipeline, FluxImg2ImgPipeline
            
            MODELS['image_gen'] = FluxPipeline.from_pretrained(
                config['hf_name'],
                torch_dtype=torch.bfloat16,
                cache_dir=MODELS_DIR,
                local_files_only=True
            )
            MODELS['image_gen'].enable_model_cpu_offload()
            MODELS['image_gen'].model_type = 'flux'  # Mark for generate_image endpoint
            
            # Share components for image editing
            MODELS['image_edit'] = FluxImg2ImgPipeline(
                transformer=MODELS['image_gen'].transformer,
                scheduler=MODELS['image_gen'].scheduler,
                vae=MODELS['image_gen'].vae,
                text_encoder=MODELS['image_gen'].text_encoder,
                text_encoder_2=MODELS['image_gen'].text_encoder_2,
                tokenizer=MODELS['image_gen'].tokenizer,
                tokenizer_2=MODELS['image_gen'].tokenizer_2,
            )
        
        CURRENT_LOADED['image'] = model_id
        print(f"{model_id} Ready\n")
        
    except Exception as e:
        print(f"Failed to switch to {model_id}: {e}")
        traceback.print_exc()
        raise e

def load_music_gen(model_id='musicgen-large'):
    """Load MusicGen Model dynamically"""
    if MODELS['music'] is not None and CURRENT_LOADED.get('music') == model_id:
        return
        
    if model_id not in MODEL_REGISTRY:
        model_id = 'musicgen-large'
    config = MODEL_REGISTRY[model_id]

    if MODELS['music'] is not None:
        print(f"Switching Music Model...")
        del MODELS['music']
        del MODELS['music_processor']
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"Loading {model_id}...")
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    
    MODELS['music_processor'] = AutoProcessor.from_pretrained(
        config['hf_name'],
        cache_dir=MODELS_DIR
    )
    
    MODELS['music'] = MusicgenForConditionalGeneration.from_pretrained(
        config['hf_name'],
        cache_dir=MODELS_DIR,
        local_files_only=True
    ).to(DEVICE)
    
    CURRENT_LOADED['music'] = model_id
    print(f"{model_id} Ready\n")

def load_tts():
    """Load Chatterbox Turbo TTS"""
    global TTS_SAMPLE_RATE
    
    if MODELS['tts'] is not None:
        return
    
    print("Loading Chatterbox Turbo TTS...")
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        
        # Chatterbox handles its own model downloading via HuggingFace
        device = "cuda" if DEVICE == "cuda" else "cpu"
        MODELS['tts'] = ChatterboxTurboTTS.from_pretrained(device=device)
        TTS_SAMPLE_RATE = MODELS['tts'].sr
        
        print(f"Chatterbox Turbo TTS Ready | Sample Rate: {TTS_SAMPLE_RATE}Hz\n")
    except ImportError:
        print("MISSING DEPENDENCY: chatterbox-tts. Install with: pip install chatterbox-tts")
        raise RuntimeError("Missing chatterbox-tts package. Run: pip install chatterbox-tts")
    except Exception as e:
        print(f"TTS failed to load: {e}")
        traceback.print_exc()
        MODELS['tts'] = None

def load_video_gen(model_id='hunyuan-video'):
    """Load Video Model dynamically"""
    if MODELS['video'] is not None and CURRENT_LOADED.get('video') == model_id:
        return

    if model_id not in MODEL_REGISTRY:
        model_id = 'hunyuan-video'
    config = MODEL_REGISTRY[model_id]
        
    print(f"Loading {model_id}...")
    try:
        from diffusers import HunyuanVideoPipeline
        
        if MODELS['video'] is not None:
            del MODELS['video']
            gc.collect()
            torch.cuda.empty_cache()
        
        # Load logic specific to Hunyuan (currently only one supported, but preparing structure)
        MODELS['video'] = HunyuanVideoPipeline.from_pretrained(
            config['hf_name'],
            torch_dtype=torch.bfloat16,
            transformer_dtype=torch.float8_e4m3fn,
            cache_dir=MODELS_DIR,
            local_files_only=True
        )
        MODELS['video'].vae.enable_tiling()
        MODELS['video'].enable_model_cpu_offload()
        
        CURRENT_LOADED['video'] = model_id
        print(f"{model_id} Ready\n")
    except Exception as e:
        print(f"Video model failed: {e}")
        MODELS['video'] = None
        raise e

def load_vocal_separator():
    """Load Vocal Separator"""
    if MODELS['vocal_sep'] is not None:
        return
    
    print("Loading Vocal Separator...")
    from audio_separator.separator import Separator
    
    MODELS['vocal_sep'] = Separator(
        output_dir=str(OUTPUT_FOLDER),
        model_file_dir=str(MODELS_DIR),
        mdx_params={"segment_size": 2048, "overlap": 0.95, "denoise": True}
    )
    
    # Use locally downloaded model
    # Global variable to surfacing loading errors
    global VOCAL_SEP_ERROR
    VOCAL_SEP_ERROR = None

    try:
        # 1. Download/Get path from HF Cache
        hf_path = hf_hub_download(
            repo_id="KitsuneX07/Music_Source_Sepetration_Models", 
            filename="vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            cache_dir=MODELS_DIR
        )
        
        # 2. audio_separator expects the file to be in 'model_file_dir' (MODELS_DIR)
        #    AND it expects 'load_model' to receive just the filename key.
        #    So we copy the file from the deep HF cache to the root of MODELS_DIR.
        
        target_filename = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        target_path = MODELS_DIR / target_filename
        
        if not target_path.exists():
            print(f"Copying model to {target_path}...")
            shutil.copy(hf_path, str(target_path))
            
        # 3. Load using the filename key
        MODELS['vocal_sep'].load_model(target_filename)
        print("Vocal Separator Ready\n")
    except Exception as e:
        VOCAL_SEP_ERROR = str(e)
        print(f"Failed to load BS-RoFormer: {e}")
        traceback.print_exc()
        MODELS['vocal_sep'] = None

def load_upscaler():
    """Load RealESRGAN Upscaler"""
    if MODELS['upscaler'] is not None:
        return
    
    print("Loading Image Upscaler...")
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    
    # Get local path from HF cache
    try:
        model_path = hf_hub_download(
            repo_id="lllyasviel/Annotators", 
            filename="RealESRGAN_x4plus.pth",
            cache_dir=MODELS_DIR
        )
        
        upscaler_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        MODELS['upscaler'] = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=upscaler_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=DEVICE == 'cuda',
            gpu_id=0 if DEVICE == 'cuda' else None
        )
        print("Image Upscaler Ready\n")
    except Exception as e:
        print(f"Failed to load RealESRGAN: {e}")
        MODELS['upscaler'] = None

def load_bg_remover():
    """Load Background Remover"""
    if MODELS['bg_remover'] is not None:
        return
    
    print("Loading Background Remover...")
    from rembg import new_session
    
    MODELS['bg_remover'] = new_session("u2net_human_seg")
    
    print("Background Remover Ready\n")

def load_3d_models():
    """Load Hunyuan3D 2.5"""
    if MODELS['3d_shape'] is not None:
        return
    
    print("Loading Hunyuan3D 2.5...")
    try:
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        
        MODELS['3d_shape'] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=MODELS_DIR,
            local_files_only=True
        )
        MODELS['3d_shape'].enable_flashvdm()
        MODELS['3d_shape'].to(DEVICE)
        
        MODELS['3d_texture'] = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            cache_dir=MODELS_DIR,
            local_files_only=True
        )
        print("Hunyuan3D 2.5 Ready\n")
    except Exception as e:
        print(f"Hunyuan3D failed: {e}")
        traceback.print_exc()
        MODELS['3d_shape'] = None
        MODELS['3d_texture'] = None

def load_video_gen():
    """Load HunyuanVideo"""
    if MODELS['video'] is not None:
        return
    
    print("Loading HunyuanVideo...")
    try:
        from diffusers import HunyuanVideoPipeline
        
        MODELS['video'] = HunyuanVideoPipeline.from_pretrained(
            "hunyuanvideo-community/HunyuanVideo",
            torch_dtype=torch.bfloat16,
            transformer_dtype=torch.float8_e4m3fn,
            cache_dir=MODELS_DIR,
            local_files_only=True
        )
        MODELS['video'].vae.enable_tiling()
        MODELS['video'].enable_model_cpu_offload()
        print("HunyuanVideo Ready\n")
    except Exception as e:
        print(f"HunyuanVideo failed: {e}")
        traceback.print_exc()
        MODELS['video'] = None

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================

def analyze_image(image_path, question="Describe this image in detail."):
    """Analyze image with vision model"""
    try:
        load_vision_model()
        
        from qwen_vl_utils import process_vision_info
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": question}
            ]
        }]
        
        text = MODELS['vision_processor'].apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = MODELS['vision_processor'](
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        generated_ids = MODELS['vision'].generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = MODELS['vision_processor'].batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True
        )[0]
        
        return output_text
        
    except Exception as e:
        print(f"❌ Vision error: {e}")
        traceback.print_exc()
        return f"Error analyzing image: {str(e)}"

V6RGE_SYSTEM_PROMPT = """You are V6rge, a powerful AI assistant with vision and creative tools.
## Your Capabilities:
- 👁️ SEE AND ANALYZE IMAGES
- 🖼️ Generate images (Flux.1)
- 🎵 Create music (MusicGen)
- 🗣️ Text-to-speech (Chatterbox TURBO)
- ✂️ Remove backgrounds from images
- 📈 Upscale images to 4K
- 🎤 Separate vocals from music
- 🎬 Generate videos (HunyuanVideo)
- 📦 Convert images to 3D models (Hunyuan3D 2.5)
- 📄 Read and analyze text files, code, and documents
## When to use tools:
- If user asks to CREATE/GENERATE/MAKE an image → respond with [TOOL:generate_image:prompt]
- If user asks for MUSIC/SOUNDTRACK/BEATS → respond with [TOOL:generate_music:prompt:duration]
- If user asks to READ ALOUD/SPEAK → respond with [TOOL:text_to_speech:text]
- If user asks for 3D MODEL from an uploaded image → respond with [TOOL:generate_3d:glb]
  (Use 'glb', 'obj', or 'stl' as the format parameter)
## TTS Emotion Tags: [clear throat], [sigh], [shush], [cough], [groan], [sniff], [gasp], [chuckle], [laugh]
Respond naturally and concisely. Do not constantly ask "How can I assist you?" at the end of every message. Be conversational. Only use [TOOL:...] syntax when explicitly creating content."""

# ==================================================================================
# FLASK APP
# ==================================================================================

app = Flask(__name__)
sock = Sock(app)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'message': 'V6rge Local Backend',
        'device': DEVICE,
        'models_loaded': sum(1 for v in MODELS.values() if v is not None)
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    # Return JSON for non-HTTP errors
    return jsonify({'error': str(e)}), 500

# ==================== V6RGE LENS WEBSOCKET ====================
# ==================== V6RGE LENS WEBSOCKET ====================
@sock.route('/lens')
def lens_socket(ws):
    print("New V6rge Lens Connection established", flush=True)
    
    # Ensure model is ready (using a smaller/faster model if possible, otherwise Qwen-72B/7B might be too slow for real-time)
    # We'll try to use the currently loaded chat model or load a default 'qwen-3b' if nothing is loaded.
    if MODELS['chat'] is None:
        try:
             # Pre-load a fast model for Lens. 
             # Ideally we'd valid config, here we default to what's available or qwen-2.5-3b-instruct if specifically set up.
             # For now, we rely on load_chat_model's default or current state.
             pass 
        except:
             pass

    while True:
        try:
            data = ws.receive()
            if data is None:
                break
                
            payload = json.loads(data)
            
            if payload.get('source') == 'youtube':
                items = payload['items']
                if not items:
                    continue

                # 1. OPTIMIZATION: BATCH PROCESSING
                # Instead of 1 title -> 1 AI call, we send ALL titles in one prompt.
                
                # Prepare the input list
                titles_text = "\n".join([f"- ID_{item['id']}: {item['text']}" for item in items])
                
                prompt = (
                    "You are a 'Clickbait Filter'. Your job is to rewrite sensationalist YouTube titles to be neutral, factual, and descriptive.\n"
                    "If a title is already normal, keep it exactly as is.\n"
                    "Return ONLY a JSON object mapping IDs to new titles: {\"ID_123\": \"New Title\", ...}\n"
                    "Do not output markdown code blocks. Just the raw JSON string.\n\n"
                    "TITLES TO PROCESS:\n"
                    f"{titles_text}\n\n"
                    "JSON OUTPUT:"
                )

                # Check if we have a model loaded, if not, use Mock (or better, auto-load)
                if MODELS['chat'] is not None:
                    # REAL AI PATH
                    try:
                        # Simple generation wrapper
                        # We need a proper generate function. server.py usually has one or we use the model directly.
                        # We will use the model directly here for speed.
                        
                        inputs = MODELS['chat_tokenizer'](prompt, return_tensors="pt").to(DEVICE)
                        
                        with torch.no_grad():
                            outputs = MODELS['chat'].generate(
                                **inputs, 
                                max_new_tokens=512,
                                temperature=0.3, # Low temp for factual consistency
                                do_sample=True
                            )
                        
                        output_text = MODELS['chat_tokenizer'].decode(outputs[0], skip_special_tokens=True)
                        
                        # Extract the JSON part (simple heuristics)
                        # We expect the model to output the JSON at the end.
                        # We'll look for the last pair of braces.
                        json_str = output_text.split("JSON OUTPUT:")[-1].strip()
                        
                        # Clean up potential markdown formatting
                        json_str = json_str.replace("```json", "").replace("```", "").strip()

                        result_map = json.loads(json_str)
                        
                        sanitized_items = []
                        for item in items:
                            key = f"ID_{item['id']}"
                            if key in result_map:
                                new_title = result_map[key]
                                sanitized_items.append({
                                    'original_text': item['text'],
                                    'new_text': new_title,
                                    'id': item['id']
                                })
                            else:
                                # Fallback if AI skipped it
                                sanitized_items.append({
                                    'original_text': item['text'],
                                    'new_text': item['text'], # Unblur, keep same
                                    'id': item['id']
                                })

                    except Exception as ai_err:
                        print(f"Lens AI Error: {ai_err}", flush=True)
                        # Fallback to simple logic on error
                        sanitized_items = []
                        for item in items:
                            sanitized_items.append({'original_text': item['text'], 'new_text': item['text'], 'id': item['id']})

                else:
                    # FALLBACK (No Model Loaded)
                    # Just return original to unblur
                    sanitized_items = []
                    for item in items:
                         sanitized_items.append({
                            'original_text': item['text'],
                            'new_text': item['text'],
                            'id': item['id']
                        })
                    
                    # Optional: Trigger auto-load in background?
                
                if sanitized_items:
                    response = {
                        'type': 'sanitize_response',
                        'items': sanitized_items
                    }
                    ws.send(json.dumps(response))
                    
        except Exception as e:
            print(f"Lens Socket Error: {e}", flush=True)
            break
    print("V6rge Lens Connection closed", flush=True)

# ==================== CONFIGURATION ENDPOINTS ====================
@app.route('/config/storage-path', methods=['GET', 'POST'])
def config_storage_path():
    """Get or set the model storage directory"""
    global MODELS_DIR, config
    
    if request.method == 'GET':
        return jsonify({
            'status': 'success',
            'path': str(MODELS_DIR),
            'default': str(BASE_DIR)
        })
    
    # POST - update storage path
    try:
        data = request.get_json() or {}
        new_path = data.get('path', '')
        
        if not new_path:
            return jsonify({'error': 'No path provided'}), 400
        
        # Convert to Path and validate
        new_path = Path(new_path)
        
        # Ensure we are using a 'models' subdirectory
        if new_path.name != 'models':
            new_path = new_path / 'models'
        
        # Create directory if it doesn't exist
        new_path.mkdir(parents=True, exist_ok=True)
        
        # Update global and config
        MODELS_DIR = new_path
        config['models_dir'] = str(new_path)
        save_config(config)
        
        print(f"[CONFIG] Storage path updated to: {new_path}")
        
        return jsonify({
            'status': 'success',
            'message': 'Storage path updated',
            'path': str(new_path)
        })
        
    except Exception as e:
        print(f"[CONFIG ERROR] {e}")
        return jsonify({'error': str(e)}), 500

# ==================== CHAT ====================
@app.route('/chat', methods=['POST'])
def chat():
    global LAST_UPLOADED_IMAGE
    
    model_id = request.form.get('model_id')
    
    print(f"DEBUG: Chat Request Received. Model: {model_id}", flush=True)
    
    try:
        if model_id:
            load_chat_model(model_id)
        else:
            # Default to 7B as it's safer for most hardware than 72B
            load_chat_model('qwen-7b')
    except Exception as e:
        print(f"❌ Model Load Error: {e}", flush=True)
        traceback.print_exc()
        return jsonify({'response': f"Failed to load model: {str(e)}"}), 500
    
    print("DEBUG: Model loaded successfully", flush=True)

    message = request.form.get('message', '')
    history_str = request.form.get('history', '[]')
    
    try:
        history = json.loads(history_str)
    except:
        history = []
    
    file_contents, image_analyses = [], []
    
    # Process uploaded files
    for key in request.files:
        file = request.files[key]
        filename = file.filename
        file_ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        if file_ext in ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp']:
            filepath = UPLOAD_FOLDER / f"chat_{uuid.uuid4().hex[:8]}_{filename}"
            print(f"DEBUG: Processing image {filename}", flush=True)
            file.save(str(filepath))
            LAST_UPLOADED_IMAGE = str(filepath)
            
            question = message if message else "Describe this image in detail."
            analysis = analyze_image(str(filepath), question)
            image_analyses.append(f"=== IMAGE: {filename} ===\n{analysis}")
        
        elif file_ext in ['txt', 'py', 'js', 'json', 'csv', 'md', 'html', 'css']:
            try:
                content = file.read().decode('utf-8')
                file_contents.append(f"=== FILE: {filename} ===\n```{file_ext}\n{content[:10000]}\n```")
            except Exception as e:
                file_contents.append(f"[FILE: {filename}] Error: {e}")
    
    # Build full message
    full_message = message
    if file_contents or image_analyses:
        parts = []
        if image_analyses: parts.append("\n\n".join(image_analyses))
        if file_contents: parts.append("\n\n".join(file_contents))
        parts.append(f"USER MESSAGE: {message}")
        full_message = "\n\n".join(parts)
    
    # Generate response
    print(f"DEBUG: Generating confirmation... Message length: {len(full_message)}", flush=True)
    try:
        messages = [{"role": "system", "content": V6RGE_SYSTEM_PROMPT}]
        for h in history[-10:]:
            messages.append({"role": h.get('role', 'user'), "content": h.get('content', '')})
        messages.append({"role": "user", "content": full_message})
        
        print("DEBUG: Checking generation method...", flush=True)
        if MODELS['chat'] and getattr(MODELS['chat'], 'is_gguf', False):
            print("DEBUG: Using GGUF Generation", flush=True)
            
            # Sanitize messages for llama.cpp (strict validation)
            safe_messages = []
            for m in messages:
                if m.get('role') and m.get('content'):
                    safe_messages.append({
                        'role': str(m['role']),
                        'content': str(m['content'])
                    })
            
            print(f"DEBUG: Input messages ({len(safe_messages)}): {json.dumps(safe_messages, indent=2)}", flush=True)
            
            try:
                # GGUF Generation
                response_data = MODELS['chat'].create_chat_completion(
                    messages=safe_messages,
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=2048,
                    stream=False
                )
                print(f"DEBUG: Raw Llama Response: {response_data}", flush=True)
                response = response_data['choices'][0]['message']['content']
            except Exception as llama_err:
                print(f"❌ Llama-cpp Execution Error: {llama_err}", flush=True)
                traceback.print_exc()
                raise llama_err
        
        else:
            print("DEBUG: Using Transformers Generation", flush=True)
            text = MODELS['chat_tokenizer'].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = MODELS['chat_tokenizer']([text], return_tensors="pt").to(DEVICE)
            
            outputs = MODELS['chat'].generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=MODELS['chat_tokenizer'].eos_token_id
            )
            
            response = MODELS['chat_tokenizer'].decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
        
        # Execute tool calls if present
        tool_result = None
        if '[TOOL:' in response:
            tool_result = execute_tool_call(response)
            if tool_result:
                response = re.sub(r'\[TOOL:[^\]]+\]', '', response).strip()
                if not response:
                    response = "Done! Here's what I created:"
        
        result = {'response': response}
        if tool_result:
            result['tool_result'] = tool_result
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Chat Error: {e}")
        traceback.print_exc()
        return jsonify({'response': f"Error: {str(e)}"}), 500

def execute_tool_call(response):
    """Execute tool calls from chat"""
    match = re.search(r'\[TOOL:([^:]+):([^\]]+)\]', response)
    if not match:
        return None
    
    tool_name, tool_args = match.group(1), match.group(2)
    print(f"🔧 Executing tool: {tool_name}")
    
    try:
        if tool_name == 'generate_image':
            load_image_gen('flux-schnell') # Agent uses default for now
            image = MODELS['image_gen'](
                prompt=tool_args,
                num_inference_steps=4,
                guidance_scale=0.0,
                width=1024,
                height=1024,
                max_sequence_length=256
            ).images[0]
            
            output_name = f"gen_{uuid.uuid4().hex[:8]}.png"
            image.save(str(OUTPUT_FOLDER / output_name))
            return {'type': 'image', 'url': f'/download/{output_name}'}
        
        elif tool_name == 'generate_music':
            load_music_gen('musicgen-large')
            parts = tool_args.split(':')
            prompt = parts[0]
            duration = int(parts[1]) if len(parts) > 1 else 10
            
            # Clamp duration to safe limits (MusicGen works best with 5-30 seconds)
            duration = max(5, min(duration, 30))
            
            inputs = MODELS['music_processor'](
                text=[prompt], padding=True, return_tensors="pt"
            ).to(DEVICE)
            
            # MusicGen generates ~51.2 tokens/second (sampling rate 32000 / frame size 625)
            # Use safe calculation with bounds checking
            max_new_tokens = int(duration * 51.2)
            max_new_tokens = min(max_new_tokens, 1500)  # Hard cap at 1500 tokens (~29 seconds)
            
            audio_values = MODELS['music'].generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True
            )
            
            # Safely extract audio (handle potential shape issues)
            if audio_values.dim() == 3:
                audio_data = audio_values[0, 0].cpu().numpy()
            else:
                audio_data = audio_values[0].cpu().numpy()
            
            sr = MODELS['music'].config.audio_encoder.sampling_rate
            output_name = f"track_{uuid.uuid4().hex[:8]}.wav"
            wavfile.write(
                str(OUTPUT_FOLDER / output_name),
                rate=sr,
                data=audio_data
            )
            return {'type': 'audio', 'url': f'/download/{output_name}'}
        
        elif tool_name == 'text_to_speech':
            load_tts()
            if MODELS['tts'] is None:
                return {'type': 'error', 'message': 'TTS not available'}
            
            wav = MODELS['tts'].generate(tool_args)
            audio_data = wav.cpu().squeeze().numpy()
            output_name = f"tts_{uuid.uuid4().hex[:8]}.wav"
            sf.write(
                str(OUTPUT_FOLDER / output_name),
                audio_data,
                TTS_SAMPLE_RATE
            )
            return {'type': 'audio', 'url': f'/download/{output_name}'}
        
        elif tool_name == 'generate_3d':
            load_3d_models()
            if MODELS['3d_shape'] is None:
                return {'type': 'error', 'message': '3D generation not available'}
            
            if LAST_UPLOADED_IMAGE is None or not os.path.exists(LAST_UPLOADED_IMAGE):
                return {'type': 'error', 'message': 'Please upload an image first'}
            
            output_format = tool_args.strip() if tool_args in ['glb', 'obj', 'stl'] else 'glb'
            
            pil_img = Image.open(LAST_UPLOADED_IMAGE).convert("RGB")
            
            mesh_output = MODELS['3d_shape'](
                image=pil_img,
                num_inference_steps=50,
                guidance_scale=5.5,
                octree_resolution=256,
                num_chunks=8000,
                generator=torch.Generator(DEVICE).manual_seed(42)
            )
            mesh = mesh_output[0]
            
            if MODELS['3d_texture'] is not None:
                try:
                    textured_mesh = MODELS['3d_texture'](
                        mesh=mesh,
                        image=pil_img,
                        texture_resolution=1024
                    )
                    mesh = textured_mesh.mesh
                except Exception as tex_err:
                    print(f"  ⚠️ Texture skipped: {tex_err}")
            
            output_name = f"model_{uuid.uuid4().hex[:8]}.{output_format}"
            mesh.export(str(MODELS_3D_FOLDER / output_name))
            
            return {'type': 'model', 'url': f'/download-3d/{output_name}', 'format': output_format}
            
    except Exception as e:
        print(f"❌ Tool error: {e}")
        traceback.print_exc()
        return {'type': 'error', 'message': str(e)}
    
    return None



# ==================== MUSIC GENERATION ====================
@app.route('/generate_music', methods=['POST'])
def generate_music():
    model_id = request.form.get('model', 'musicgen-large')
    load_music_gen(model_id)
    
    prompt = request.form.get('prompt', '')
    duration = int(request.form.get('duration', 10))
    
    # Advanced Params
    guidance_scale = float(request.form.get('guidance', 3.0))
    temperature = float(request.form.get('temperature', 1.0))

    try:
        # Validate and clamp duration to safe limits
        duration = max(5, min(duration, 30))  # MusicGen works best with 5-30 seconds
        
        # MusicGen generates ~51.2 tokens/second (sampling rate 32000 / frame size 625)
        # Use safe calculation with bounds checking to prevent index errors
        max_new_tokens = int(duration * 51.2)
        max_new_tokens = min(max_new_tokens, 1500)  # Hard cap at 1500 tokens (~29 seconds)
        
        print(f"[MusicGen] Generating {duration}s audio ({max_new_tokens} tokens)...", flush=True)
        
        audio_values = MODELS['music'].generate(
            prompt=[prompt],
            max_new_tokens=max_new_tokens,
            guidance_scale=guidance_scale,
            temperature=temperature,
            do_sample=True
        )
        
        # Safely extract audio data (handle potential shape variations)
        if audio_values.dim() == 3:
            audio_data = audio_values[0, 0].cpu().numpy()
        else:
            audio_data = audio_values[0].cpu().numpy()
        
        sr = MODELS['music'].config.audio_encoder.sampling_rate
        output_name = f"track_{uuid.uuid4().hex[:8]}.wav"
        wavfile.write(
            str(OUTPUT_FOLDER / output_name),
            rate=sr,
            data=audio_data
        )
        
        print(f"[MusicGen] ✅ Generated {output_name}", flush=True)
        return jsonify({'status': 'success', 'audio_url': f'/download/{output_name}'})
        
    except Exception as e:
        print(f"[MusicGen] ❌ Error: {e}", flush=True)
        traceback.print_exc()
        return jsonify({'error': f'Music generation failed: {str(e)}'}), 500

# ==================== TEXT-TO-SPEECH ====================
@app.route('/synthesize', methods=['POST'])
def synthesize():
    load_tts()
    
    if MODELS['tts'] is None:
        return jsonify({'error': 'TTS model not available'}), 500
    
    text = request.form.get('text', '')[:2000]
    if not text:
        return jsonify({'error': 'Text is required'}), 400
    
    try:
        wav = MODELS['tts'].generate(text)
        audio_data = wav.cpu().squeeze().numpy()
        
        output_name = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = OUTPUT_FOLDER / output_name
        sf.write(str(output_path), audio_data, TTS_SAMPLE_RATE)
        
        return send_file(str(output_path), mimetype='audio/wav')
        
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== VIDEO GENERATION ====================
@app.route('/generate_video', methods=['POST'])
def generate_video():
    model_id = request.form.get('model', 'hunyuan-video')
    load_video_gen(model_id)
    
    if MODELS['video'] is None:
        return jsonify({'error': 'Video generation not available'}), 500
    
    prompt = request.form.get('prompt', '')
    
    try:
        from diffusers.utils import export_to_video
        
        # Advanced Params
        num_frames = int(request.form.get('frames', 65))
        num_inference_steps = int(request.form.get('steps', 30))
        # fps is used in export
        fps = int(request.form.get('fps', 24))

        video = MODELS['video'](
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            generator=torch.Generator(DEVICE).manual_seed(42)
        ).frames[0]
        
        output_name = f"video_{uuid.uuid4().hex[:8]}.mp4"
        export_to_video(video, str(OUTPUT_FOLDER / output_name), fps=fps)
        
        return jsonify({'status': 'success', 'video_url': f'/download/{output_name}'})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== IMAGE UPSCALER ====================
@app.route('/upscale', methods=['POST'])
def upscale_image():
    load_upscaler()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    scale = int(request.form.get('scale', 4))
    
    input_path = UPLOAD_FOLDER / file.filename
    file.save(str(input_path))
    
    try:
        img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
        output, _ = MODELS['upscaler'].enhance(img, outscale=scale)
        
        output_name = f"{os.path.splitext(file.filename)[0]}_upscaled.png"
        cv2.imwrite(str(OUTPUT_FOLDER / output_name), output)
        
        return jsonify({'status': 'success', 'upscaled_url': f'/download/{output_name}'})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== BACKGROUND REMOVAL ====================
@app.route('/remove_background', methods=['POST'])
def remove_background():
    load_bg_remover()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    
    try:
        from rembg import remove
        
        img = Image.open(file).convert("RGB")
        output = remove(img, session=MODELS['bg_remover'])
        
        output_name = f"{os.path.splitext(file.filename)[0]}_nobg.png"
        output.save(str(OUTPUT_FOLDER / output_name), 'PNG')
        
        return jsonify({'status': 'success', 'image_url': f'/download/{output_name}'})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== IMAGE GENERATION ====================
@app.route('/generate_image', methods=['POST'])
def generate_image():
    model_id = request.form.get('model', 'flux-schnell')
    try:
        load_image_gen(model_id)
    except Exception as e:
        return jsonify({'error': f"Failed to load model {model_id}: {str(e)}"}), 500
    
    if MODELS['image_gen'] is None:
        return jsonify({'error': 'Image Generation model not loaded. Please download it first.'}), 500
        
    prompt = request.form.get('prompt', '')
    seed = int(request.form.get('seed', 0))
    
    generator = None
    if seed > 0:
        generator = torch.Generator(DEVICE).manual_seed(seed)

    try:
        # Check which model type is loaded
        model_type = getattr(MODELS['image_gen'], 'model_type', 'flux')
        
        if model_type == 'qwen':
            # ===== QWEN-IMAGE GENERATION =====
            aspect_ratio = request.form.get('aspect_ratio', '1:1')
            guidance_scale = float(request.form.get('guidance', 4.0))
            num_inference_steps = int(request.form.get('steps', 50))
            negative_prompt = request.form.get('negative_prompt', ' ')
            prompt_enhance = request.form.get('prompt_enhance', 'false').lower() == 'true'
            
            # Qwen aspect ratio presets
            aspect_ratios = {
                "1:1": (1328, 1328),
                "16:9": (1664, 928),
                "9:16": (928, 1664),
                "4:3": (1472, 1140),
                "3:4": (1140, 1472),
                "3:2": (1584, 1056),
                "2:3": (1056, 1584),
            }
            width, height = aspect_ratios.get(aspect_ratio, (1328, 1328))
            
            # Prompt enhancement (magic suffix)
            if prompt_enhance:
                prompt = prompt + ", Ultra HD, 4K, cinematic composition."
            
            image = MODELS['image_gen'](
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else " ",
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=guidance_scale,
                generator=generator
            ).images[0]
            
        else:
            # ===== FLUX GENERATION =====
            width = int(request.form.get('width', 1024))
            height = int(request.form.get('height', 1024))
            num_inference_steps = int(request.form.get('steps', 0))
            guidance_scale = float(request.form.get('guidance', 0.0))
            
            # Set defaults if 0 (FLUX defaults)
            if num_inference_steps == 0:
                num_inference_steps = 4 if 'schnell' in model_id else 25
            if guidance_scale == 0:
                guidance_scale = 0.0 if 'schnell' in model_id else 3.5

            image = MODELS['image_gen'](
                prompt=prompt, 
                num_inference_steps=num_inference_steps, 
                guidance_scale=guidance_scale, 
                width=width, 
                height=height, 
                max_sequence_length=256,
                generator=generator
            ).images[0]
        
        output_name = f"gen_{uuid.uuid4().hex[:8]}.png"
        image.save(str(OUTPUT_FOLDER / output_name))
        
        return jsonify({'status': 'success', 'image_url': f'/download/{output_name}'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/edit_image', methods=['POST'])
def edit_image():
    model_id = request.form.get('model', 'flux-schnell') 
    load_image_gen(model_id)
    
    if MODELS['image_edit'] is None:
        return jsonify({'error': 'Image Editing model not loaded.'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    prompt = request.form.get('prompt', '')
    strength = float(request.form.get('strength', 0.6))
    
    try:
        img_path = UPLOAD_FOLDER / f"edit_{uuid.uuid4().hex}.png"
        file.save(str(img_path))
        
        init_image = Image.open(img_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        
        image = MODELS['image_edit'](
            prompt=prompt, 
            image=init_image, 
            strength=strength, 
            num_inference_steps=4, 
            guidance_scale=0.0, 
            max_sequence_length=256
        ).images[0]
        
        output_name = f"edit_{uuid.uuid4().hex[:8]}.png"
        image.save(str(OUTPUT_FOLDER / output_name))
        
        # Cleanup
        if img_path.exists():
            img_path.unlink()
            
        return jsonify({'status': 'success', 'image_url': f'/download/{output_name}'})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== VOCAL SEPARATOR ====================
@app.route('/process', methods=['POST'])
def process_audio():
    load_vocal_separator()
    
    if MODELS['vocal_sep'] is None:
        error_msg = f"Vocal Separator model not loaded. Error: {str(globals().get('VOCAL_SEP_ERROR', 'Unknown load error'))}"
        return jsonify({'error': error_msg}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    filepath = UPLOAD_FOLDER / file.filename
    file.save(str(filepath))
    
    try:
        output_files = MODELS['vocal_sep'].separate(str(filepath))
        
        inst = next((f for f in output_files if 'Instrumental' in f), None)
        vocal = next((f for f in output_files if 'Vocals' in f), None)
        
        base_name = os.path.splitext(file.filename)[0]
        folder = OUTPUT_FOLDER / base_name
        folder.mkdir(exist_ok=True)
        
        shutil.move(inst, str(folder / 'instrumental.wav'))
        shutil.move(vocal, str(folder / 'vocals.wav'))
        
        return jsonify({
            'status': 'success',
            'vocals_url': f'/download/{base_name}/vocals.wav',
            'instrumental_url': f'/download/{base_name}/instrumental.wav'
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== 3D GENERATION ====================
@app.route('/generate-3d', methods=['POST'])
def generate_3d():
    load_3d_models()
    
    if MODELS['3d_shape'] is None:
        return jsonify({'error': '3D generation not available'}), 500
    
    images = request.files.getlist('images')
    if not images:
        return jsonify({'error': 'No images provided'}), 400
    
    
    # Advanced Params
    num_inference_steps = int(request.form.get('steps', 50))
    texture_res = int(request.form.get('texture_res', 1024))
    seed = int(request.form.get('seed', 0))
    output_format = request.form.get('format', 'glb') # Added output_format here

    try:
        img_path = UPLOAD_FOLDER / f"3d_{uuid.uuid4().hex}.png"
        images[0].save(str(img_path))
        pil_img = Image.open(img_path).convert("RGB")
        
        gen_seed = seed if seed > 0 else 42
        
        mesh_output = MODELS['3d_shape'](
            image=pil_img,
            num_inference_steps=num_inference_steps,
            guidance_scale=5.5,
            octree_resolution=256,
            num_chunks=8000,
            generator=torch.Generator(DEVICE).manual_seed(gen_seed)
        )
        mesh = mesh_output[0]
        
        if MODELS['3d_texture'] is not None:
            try:
                textured_mesh = MODELS['3d_texture'](
                    mesh=mesh,
                    image=pil_img,
                    texture_resolution=texture_res
                )
                mesh = textured_mesh.mesh
            except Exception as tex_err:
                print(f"  ⚠️ Texture skipped: {tex_err}")
        
        output_name = f"model_{uuid.uuid4().hex[:8]}.{output_format}"
        mesh.export(str(MODELS_3D_FOLDER / output_name))
        
        img_path.unlink()
        
        return jsonify({
            'status': 'success',
            'model_url': f'/download-3d/{output_name}',
            'format': output_format
        })
        
    except Exception as e:
        print(f"❌ 3D Generation Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ==================== FILE DOWNLOADS ====================
@app.route('/download/<path:p>')
def download(p):
    print(f"DEBUG: Download request for {p}", flush=True)
    path = OUTPUT_FOLDER / p
    print(f"DEBUG: Absolute path: {path}, Exists: {path.exists()}", flush=True)
    if path.exists():
        ext = p.split('.')[-1].lower()
        mimetypes_map = {
            'png': 'image/png',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'wav': 'audio/wav',
            'mp3': 'audio/mpeg',
            'mp4': 'video/mp4'
        }
        mimetype = mimetypes_map.get(ext, 'application/octet-stream')
        
        response = send_file(str(path), mimetype=mimetype)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    return jsonify({'error': 'Not found'}), 404

@app.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# ==================== CONFIGURATION ====================
@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    global MODELS_DIR
    
    if request.method == 'POST':
        data = request.json
        new_models_dir = data.get('models_dir')
        
        if new_models_dir:
            config = load_config()
            config['models_dir'] = new_models_dir
            save_config(config)
            
            # Update global var (requires restart to fully take effect for some libraries, but we update passingly)
            MODELS_DIR = Path(new_models_dir)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Re-init registry paths helper if needed
            return jsonify({'status': 'success', 'models_dir': str(MODELS_DIR)})
            
    # GET
    config = load_config()
    current_dir = config.get('models_dir', str(BASE_DIR))
    return jsonify({'models_dir': current_dir})

@app.route('/download-3d/<path:filename>')
def download_3d(filename):
    path = MODELS_3D_FOLDER / filename
    if path.exists():
        ext = filename.split('.')[-1].lower()
        mimetypes_map = {
            'glb': 'model/gltf-binary',
            'gltf': 'model/gltf+json',
            'obj': 'model/obj',
            'stl': 'model/stl'
        }
        mimetype = mimetypes_map.get(ext, 'application/octet-stream')
        
        response = send_file(str(path), mimetype=mimetype, as_attachment=True)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    
    return jsonify({'error': 'Model not found'}), 404

# ==================================================================================
# MODEL MANAGEMENT ENDPOINTS
# ==================================================================================

MODEL_REGISTRY = {
    'qwen-72b': {'name': 'Qwen 2.5 (72B) - Chat', 'hf_name': 'Qwen/Qwen2.5-72B-Instruct-AWQ', 'type': 'chat', 'size_gb': 42},
    'qwen-3b': {'name': 'Qwen 2.5 (3B) - GGUF', 'hf_name': 'Qwen/Qwen2.5-3B-Instruct-GGUF', 'filename': 'qwen2.5-3b-instruct-q4_k_m.gguf', 'type': 'chat', 'size_gb': 2.0},
    'qwen2-vl': {'name': 'Qwen2 Vision (7B)', 'hf_name': 'Qwen/Qwen2-VL-7B-Instruct', 'type': 'vision', 'size_gb': 16},
    'flux-dev': {'name': 'Flux.1 Dev', 'hf_name': 'black-forest-labs/FLUX.1-dev', 'type': 'image', 'size_gb': 24},
    'flux-schnell': {'name': 'Flux.1 Schnell (Turbo)', 'hf_name': 'black-forest-labs/FLUX.1-schnell', 'type': 'image', 'size_gb': 24},
    'qwen-image': {'name': 'Qwen-Image', 'hf_name': 'Qwen/Qwen-Image', 'type': 'image', 'size_gb': 57.7},
    'musicgen-large': {'name': 'MusicGen Large', 'hf_name': 'facebook/musicgen-large', 'type': 'music', 'size_gb': 3.3},
    'musicgen-medium': {'name': 'MusicGen Medium', 'hf_name': 'facebook/musicgen-medium', 'type': 'music', 'size_gb': 1.5},
    'chatterbox-turbo': {'name': 'Chatterbox Turbo TTS', 'hf_name': 'ResembleAI/chatterbox', 'type': 'tts', 'size_gb': 1.0},
    'hunyuan-video': {'name': 'Hunyuan Video', 'hf_name': 'hunyuanvideo-community/HunyuanVideo', 'type': 'video', 'size_gb': 26},
    'hunyuan3d': {'name': 'Hunyuan 3D (v2)', 'hf_name': 'tencent/Hunyuan3D-2', 'type': '3d', 'size_gb': 8},
    'realesrgan': {'name': 'RealESRGAN 4x', 'hf_name': 'lllyasviel/Annotators', 'filename': 'RealESRGAN_x4plus.pth', 'type': 'tool', 'size_gb': 0.064},
    'u2net': {'name': 'Background Remover (U2Net)', 'hf_name': 'rembg/u2net', 'type': 'tool', 'size_gb': 0.176, 'skip_download': True},
    'bs-roformer': {'name': 'Vocal Separator (RoFormer)', 'hf_name': 'KitsuneX07/Music_Source_Sepetration_Models', 'filename': 'vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt', 'type': 'tool', 'size_gb': 1.2}
}

@app.route('/models/status', methods=['GET'])
def get_models_status():
    """Get download status of all models"""
    status = {}
    
    for model_id, info in MODEL_REGISTRY.items():
        is_downloaded = False
        
        # Check huggingface cache
        try:
            filename = info.get('filename')
            
            if model_id == 'u2net':
                 # Special check for u2net (rembg)
                 u2net_path = Path.home() / '.u2net'
                 # We consider it downloaded if .u2net folder has content (e.g. u2net.pth)
                 if u2net_path.exists() and any(u2net_path.iterdir()):
                     is_downloaded = True
            elif filename:
                # Single file check
                filepath = try_to_load_from_cache(repo_id=info['hf_name'], filename=filename, cache_dir=MODELS_DIR)
                if filepath and os.path.exists(filepath):
                    is_downloaded = True
                else:
                    # Fallback: Manual recursive search in the repo folder
                    # This finds the file even if HF structure is complex (snapshots/blobs/refs)
                    repo_folder = f"models--{info['hf_name'].replace('/', '--')}"
                    repo_path = MODELS_DIR / repo_folder
                    
                    if repo_path.exists():
                        # Search for the specific filename anywhere in this repo folder
                        # This covers snapshots/<hash>/filename AND other variations
                        found_files = list(repo_path.rglob(filename))
                        if found_files:
                            # Verify size to avoid empty placeholder files
                            if any(f.stat().st_size > 1024 for f in found_files):
                                is_downloaded = True

            else:
                 # Repo check
                 repo_path = MODELS_DIR / f"models--{info['hf_name'].replace('/', '--')}"
                 
                 if repo_path.exists():
                     # Robust check: look for ANY model file or large file in the repo folder
                     # regardless of snapshots/blobs structure
                     for file in repo_path.rglob('*'):
                         if file.is_file():
                             # Check for model extensions OR large size (>100MB)
                             if (file.suffix in ['.safetensors', '.bin', '.pth', '.pt', '.ckpt', '.onnx', '.msgpack'] 
                                 or file.stat().st_size > 100 * 1024 * 1024):
                                 is_downloaded = True
                                 break

        except Exception:
             pass

        status[model_id] = {
            'downloaded': is_downloaded, 
            'size_gb': info['size_gb'],
            'type': info['type'],
            'name': info.get('name', model_id)
        }
    
    total_downloaded = sum(s['size_gb'] for s in status.values() if s['downloaded'])
    return jsonify({'models': status, 'total_downloaded_gb': round(total_downloaded, 2)})

# ==================================================================================
# DOWNLOAD HELPERS
# ==================================================================================

def download_model_thread(model_id, model_info):
    global CURRENT_DOWNLOAD_ID
    CURRENT_DOWNLOAD_ID = model_id
    
    try:
        print(f"Starting download for {model_id}...")
        
        cache_dir = MODELS_DIR
        print(f"DEBUG: Downloading to cache_dir: {cache_dir}")
        
        repo_id = model_info['hf_name']
        filename = model_info.get('filename')
        skip_download = model_info.get('skip_download', False)
        
        DOWNLOAD_PROGRESS[model_id]['message'] = "Fetching file list..."
        
        if skip_download:
            # Some models (like rembg) handle their own downloads
            DOWNLOAD_PROGRESS[model_id]['message'] = "Preparing..."
        elif filename:
            # Single file download for tools
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                local_files_only=False,
                resume_download=True
            )
        else:
            # Full repo snapshot
            snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=False,
                resume_download=True
            )
        

        
        DOWNLOAD_PROGRESS[model_id].update({'status': 'loading', 'progress': 99, 'message': 'Loading into VRAM...'})
        
        # Now clean load (cached)
        # Note: We duplicate logic from download_model here but it's cleaner
        if model_info['type'] == 'chat':
            load_chat_model(model_id)
        elif model_info['type'] == 'vision':
            load_vision_model()
        elif model_info['type'] == 'image':
            load_image_gen(model_id)
        elif model_info['type'] == 'music':
            load_music_gen(model_id)
        elif model_info['type'] == 'tts':
            load_tts()
        elif model_info['type'] == 'video':
            load_video_gen(model_id)
        elif model_info['type'] == '3d':
            load_3d_models()
        elif model_info['type'] == 'tool':
            if model_id == 'realesrgan':
                load_upscaler()
            elif model_id == 'u2net':
                load_bg_remover()
            elif model_id == 'bs-roformer':
                load_vocal_separator()
        
        DOWNLOAD_PROGRESS[model_id] = {'status': 'complete', 'progress': 100, 'message': 'Ready', 'speed': 0, 'eta': 0}
        CURRENT_DOWNLOAD_ID = None
        
    except Exception as e:
        if "Download Cancelled" in str(e) or isinstance(e, KeyboardInterrupt):
             print(f"Download CANCELLED for {model_id}")
             DOWNLOAD_PROGRESS[model_id] = {'status': 'cancelled', 'progress': 0, 'message': 'Cancelled'}
        else:
            print(f"Download error: {e}")
            traceback.print_exc()
            DOWNLOAD_PROGRESS[model_id] = {'status': 'error', 'progress': 0, 'message': str(e)}
        
        # Restore tqdm
        try:
            import tqdm
            import tqdm.auto
            if 'original_tqdm' in locals():
                tqdm.tqdm = original_tqdm
            if 'original_auto_tqdm' in locals():
                tqdm.auto.tqdm = original_auto_tqdm
        except:
            pass
            
        CURRENT_DOWNLOAD_ID = None
        if model_id in CANCELLATION_REQUESTED:
            CANCELLATION_REQUESTED.remove(model_id)


@app.route('/models/cancel/<model_id>', methods=['POST'])
def cancel_download(model_id):
    """Cancel a running download"""
    if model_id == CURRENT_DOWNLOAD_ID:
        CANCELLATION_REQUESTED.add(model_id)
        return jsonify({'status': 'success', 'message': 'Cancellation requested'})
    return jsonify({'status': 'ignored', 'message': 'Not currently downloading'})


@app.route('/models/download/<model_id>', methods=['POST'])
def download_model(model_id):
    """Trigger background model download"""
    if model_id not in MODEL_REGISTRY:
        return jsonify({'error': 'Unknown model'}), 404
    
    model_info = MODEL_REGISTRY[model_id]
    
    # Check if already downloading (simple lock)
    if model_id in DOWNLOAD_PROGRESS and DOWNLOAD_PROGRESS[model_id]['status'] == 'downloading':
        return jsonify({'status': 'ignored', 'message': 'Already downloading'})

    # Init progress
    DOWNLOAD_PROGRESS[model_id] = {
        'status': 'downloading', 
        'progress': 0, 
        'message': 'Starting...',
        'downloaded': 0,
        'total': 0,
        'speed': 0,
        'eta': 0
    }
    
    # Start thread
    thread = threading.Thread(target=download_model_thread, args=(model_id, model_info))
    thread.daemon = True # Kill if server stops
    thread.start()
    
    return jsonify({
        'status': 'success',
        'message': f'Started download for {model_id}'
    })

@app.route('/models/progress/<model_id>', methods=['GET'])
def get_download_progress(model_id):
    """Get download progress for a specific model"""
    if model_id in DOWNLOAD_PROGRESS:
        return jsonify(DOWNLOAD_PROGRESS[model_id])
    else:
        return jsonify({'status': 'idle', 'progress': 0, 'message': 'Not started'})

@app.route('/models/delete/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a downloaded model"""
    if model_id not in MODEL_REGISTRY:
        return jsonify({'error': 'Unknown model'}), 404
    
    model_info = MODEL_REGISTRY[model_id]
    
    # Helper for Windows read-only files
    def remove_readonly(func, path, excinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    try:
        # Special handling for u2net (rembg)
        if model_id == 'u2net':
            # rembg stores models in ~/.u2net/u2net.pth
            u2net_path = Path.home() / '.u2net'
            if u2net_path.exists():
                shutil.rmtree(str(u2net_path), onerror=remove_readonly)
                return jsonify({'status': 'success', 'message': f'Model {model_id} deleted'})
            else:
                 return jsonify({'error': 'Model not found'}), 404

        # Standard Hugging Face Cache
        # HF Cache has structure models--owner--repo
        repo_folder_name = f"models--{model_info['hf_name'].replace('/', '--')}"
        cache_path = MODELS_DIR / repo_folder_name

        if cache_path.exists():
            # onerror=remove_readonly ensures we can delete Git/read-only files on Windows
            shutil.rmtree(str(cache_path), onerror=remove_readonly)
            return jsonify({'status': 'success', 'message': f'Model {model_id} deleted'})
        else:
            return jsonify({'error': 'Model not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================================================================================
# START SERVER
# ==================================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("V6RGE LOCAL BACKEND STARTING")
    print("="*60)
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Output Directory: {OUTPUT_FOLDER}")
    print(f"Device: {DEVICE}")
    print("="*60)
    print("\nModels will be downloaded on first use")
    print("Starting server on http://localhost:5000\n")
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

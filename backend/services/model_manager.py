import os
import gc
import torch
import shutil
from pathlib import Path
from huggingface_hub import try_to_load_from_cache

# Import config (assuming it's in parent directory, but we should fix python path later)
# For now, we assume relative import or running from backend root
try:
    from config import MODELS_DIR, DEVICE
except ImportError:
    # If config isn't found (if running directly), define minimal fallback or fix path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODELS_DIR, DEVICE

class ModelManager:
    def __init__(self):
        # Global Registry of supported models
        self.registry = {
            'qwen-72b': {'name': 'Qwen 2.5 (72B) - GGUF (High Quality)', 'hf_name': 'bartowski/Qwen2.5-72B-Instruct-GGUF', 'filename': 'Qwen2.5-72B-Instruct-Q4_K_M.gguf', 'type': 'chat', 'size_gb': 44.16},
            'qwen-32b': {'name': 'Qwen 2.5 (32B) - GGUF (Balanced)', 'hf_name': 'bartowski/Qwen2.5-32B-Instruct-GGUF', 'filename': 'Qwen2.5-32B-Instruct-Q4_K_M.gguf', 'type': 'chat', 'size_gb': 18.5},
            'qwen-7b': {'name': 'Qwen 2.5 (7B) - Full', 'hf_name': 'Qwen/Qwen2.5-7B-Instruct', 'type': 'chat', 'size_gb': 14.2},
            'qwen-3b': {'name': 'Qwen 7B (Quantized) [Low End PC]', 'hf_name': 'bartowski/Qwen2.5-7B-Instruct-GGUF', 'filename': 'Qwen2.5-7B-Instruct-Q4_K_M.gguf', 'type': 'chat', 'size_gb': 4.4},
            'qwen2-vl': {'name': 'Qwen2 Vision (7B)', 'hf_name': 'Qwen/Qwen2-VL-7B-Instruct', 'type': 'vision', 'size_gb': 15.5},
            'flux-dev': {'name': 'Flux.1 Dev', 'hf_name': 'black-forest-labs/FLUX.1-dev', 'type': 'image', 'size_gb': 53.9},
            'flux-schnell': {'name': 'Flux.1 Schnell (Turbo)', 'hf_name': 'black-forest-labs/FLUX.1-schnell', 'type': 'image', 'size_gb': 53.9},
            'qwen-image': {'name': 'Qwen-Image', 'hf_name': 'Qwen/Qwen-Image', 'type': 'image', 'size_gb': 53.7},
            'musicgen-large': {'name': 'MusicGen Large', 'hf_name': 'facebook/musicgen-large', 'type': 'music', 'size_gb': 19.1},
            'musicgen-medium': {'name': 'MusicGen Medium', 'hf_name': 'facebook/musicgen-medium', 'type': 'music', 'size_gb': 11.1},
            'chatterbox-turbo': {'name': 'Chatterbox Turbo TTS', 'hf_name': 'ResembleAI/chatterbox-turbo', 'type': 'tts', 'size_gb': 9.0},
            'hunyuan-video': {'name': 'Hunyuan Video', 'hf_name': 'hunyuanvideo-community/HunyuanVideo', 'type': 'video', 'size_gb': 39.0},
            'hunyuan3d': {'name': 'Hunyuan 3D (v2)', 'hf_name': 'tencent/Hunyuan3D-2', 'type': '3d', 'size_gb': 69.7},
            'realesrgan': {'name': 'RealESRGAN 4x', 'hf_name': 'lllyasviel/Annotators', 'filename': 'RealESRGAN_x4plus.pth', 'type': 'upscaler', 'size_gb': 0.06},
            'u2net': {'name': 'Background Remover (U2Net)', 'hf_name': 'rembg/u2net', 'type': 'bg_remover', 'size_gb': 0.18, 'skip_download': True},
            'bs-roformer': {'name': 'Vocal Separator (RoFormer)', 'hf_name': 'KitsuneX07/Music_Source_Sepetration_Models', 'filename': 'vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt', 'type': 'vocal_sep', 'size_gb': 0.6}
        }

        # Runtime State
        self.loaded_models = {}  # {'chat': <obj>, 'vision': <obj>, ...}
        self.current_model_ids = {} # {'chat': 'qwen-3b', ...}
        
    def get_model_config(self, model_id):
        return self.registry.get(model_id)

    def is_downloaded(self, model_id):
        """Check if a model is downloaded (Robust Check)"""
        info = self.registry.get(model_id)
        if not info: return False
        
        filename = info.get('filename')
        
        # Special check for u2net
        if model_id == 'u2net':
             # Use configured path
             u2net_path = MODELS_DIR / 'u2net'
             return u2net_path.exists() and any(u2net_path.iterdir())

        if model_id == 'realesrgan':
            return (MODELS_DIR / "RealESRGAN_x4plus.pth").exists()

        if model_id == 'bs-roformer':
            return (MODELS_DIR / "model_bs_roformer_ep_317_sdr_12.9755.ckpt").exists()

        if filename:
            # Check HF Cache API first (Standard Path)
            filepath = try_to_load_from_cache(repo_id=info['hf_name'], filename=filename, cache_dir=MODELS_DIR)
            if filepath and isinstance(filepath, (str, Path)) and os.path.exists(filepath):
                return True

            # Check HF Cache API (Hub Subdirectory - New Default)
            filepath_hub = try_to_load_from_cache(repo_id=info['hf_name'], filename=filename, cache_dir=MODELS_DIR / 'hub')
            if filepath_hub and isinstance(filepath_hub, (str, Path)) and os.path.exists(filepath_hub):
                return True
                
            # Manual Recursive Search
            repo_folder = f"models--{info['hf_name'].replace('/', '--')}"
            repo_path = MODELS_DIR / repo_folder
            if repo_path.exists():
                found_files = list(repo_path.rglob(filename))
                if any(f.stat().st_size > 1024 for f in found_files):
                    return True
        else:
            # Directory Check (Repo based) - Check both root and hub/
            folder_name = f"models--{info['hf_name'].replace('/', '--')}"
            candidates = [
                MODELS_DIR / folder_name,
                MODELS_DIR / 'hub' / folder_name
            ]
            
            for repo_path in candidates:
                if repo_path.exists():
                    for file in repo_path.rglob('*'):
                        if file.is_file():
                            # Check for model weights
                            if (file.suffix in ['.safetensors', '.bin', '.pth', '.pt', '.ckpt', '.onnx'] 
                                or file.stat().st_size > 100 * 1024 * 1024):
                                return True
        return False

    def get_status(self):
        """Return status of all models for frontend"""
        status = {}
        for mid, info in self.registry.items():
            encoded_status = {
                'id': mid,  # Critical: Frontend needs this for filtering
                'name': info['name'],
                'type': info['type'],
                'size_gb': info['size_gb'],
                'downloaded': self.is_downloaded(mid)
            }
            status[mid] = encoded_status
            
        total_gb = sum(s['size_gb'] for s in status.values() if s['downloaded'])
        return {'models': status, 'total_downloaded_gb': round(total_gb, 2)}

    def unload_model(self, model_id):
        """Unload a specific model and force GC to release file locks"""
        found = False
        # Remove from loaded_models if present (check by ID or type mapping)
        # current_model_ids maps type -> id.
        # We need to find the type for this ID.
        type_to_unload = None
        for m_type, m_id in self.current_model_ids.items():
            if m_id == model_id:
                type_to_unload = m_type
                break
        
        # Fallback: if model_id is a key in registry, we can infer type
        if not type_to_unload and model_id in self.registry:
            type_to_unload = self.registry[model_id]['type']

        if type_to_unload and type_to_unload in self.loaded_models:
            if self.loaded_models[type_to_unload] is not None:
                print(f"[ModelManager] Unloading {type_to_unload} ({model_id})...")
                self.loaded_models[type_to_unload] = None
                found = True
                
        if found or True: # Always run GC if requested to delete, just in case
            del type_to_unload # Clear local ref
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            print("[ModelManager] GC collected.")
            
        # Also clean from current_model_ids
        # We iterate list to avoid runtime error during modification
        for k in list(self.current_model_ids.keys()):
             if self.current_model_ids[k] == model_id:
                 del self.current_model_ids[k]

    def unload_all(self, keep_types=[]):
        """Unload all models except specified types to free VRAM"""
        freed = False
        keys_to_remove = []
        
        for m_type, model in self.loaded_models.items():
            if m_type not in keep_types and model is not None:
                print(f"[ModelManager] Unloading {m_type}...")
                self.loaded_models[m_type] = None
                keys_to_remove.append(m_type)
                freed = True
                
        # Clear IDs
        for k in keys_to_remove:
            if k in self.current_model_ids:
                del self.current_model_ids[k]
                
        if freed:
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
            print("[ModelManager] VRAM Cleanup Complete")

    def register_model(self, model_type, model_obj, model_id=None):
        """Register a loaded model object"""
        # First, ensure we have room (Simple Strategy: Unload everything else if it's a heavy model)
        # Heavy types: image, video, chat (large), music
        heavy_types = ['image', 'video', 'chat', '3d']
        
        if model_type in heavy_types:
            # Unload other heavy models
            self.unload_all(keep_types=[model_type])
            
        self.loaded_models[model_type] = model_obj
        if model_id:
            self.current_model_ids[model_type] = model_id
            
    def get_loaded_model(self, model_type):
        return self.loaded_models.get(model_type)

model_manager = ModelManager()

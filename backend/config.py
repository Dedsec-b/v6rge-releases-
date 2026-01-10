import os
import sys
import json
from pathlib import Path

# Base Paths
BASE_DIR = Path.home() / '.v6rge'
CONFIG_FILE = BASE_DIR / 'config.json'

# Tokens
HF_TOKEN = "" # Token removed for security

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

# Load global config
config = load_config()

# Directories
_models_dir_candidate = Path(config.get('models_dir', str(BASE_DIR / 'models')))
if not _models_dir_candidate.exists():
    # If the configured path (e.g. on D drive) doesn't exist, fall back and warn
    print(f"[WARNING] Configured models directory '{_models_dir_candidate}' not found. Falling back to default.")
    _models_dir_candidate = BASE_DIR / 'models'

MODELS_DIR = _models_dir_candidate
UPLOAD_FOLDER = BASE_DIR / 'uploads'
OUTPUT_FOLDER = BASE_DIR / 'output'
VOICES_FOLDER = BASE_DIR / 'voices'
MODELS_3D_FOLDER = BASE_DIR / '3d_models'

# Ensure directories exist
for folder in [BASE_DIR, MODELS_DIR, UPLOAD_FOLDER, OUTPUT_FOLDER, VOICES_FOLDER, MODELS_3D_FOLDER]:
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Could not create directory {folder}: {e}")

# Hardware Detection
DEVICE = "cpu"
TORCH_DTYPE = None # Placeholder

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[OK] GPU Detected: {gpu_name} ({gpu_mem:.1f}GB VRAM)")
        except:
            print(f"[OK] GPU Detected (details unavailable)")
    else:
        print("[INFO] No CUDA GPU detected. Running in CPU mode.")
    
    # Device Map Strategy (for transformers)
    TORCH_DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

except ImportError:
    print("[WARNING] PyTorch not installed or failed to import. Running in limited mode.")
except Exception as e:
    print(f"[ERROR] Error initializing PyTorch: {e}. Running in limited mode.")

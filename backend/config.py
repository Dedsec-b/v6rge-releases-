import os
import sys
import json
import uuid
from pathlib import Path

# Load .env file if present (for development)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Force UTF-8 output for Windows console to handle emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

# Base Paths
BASE_DIR = Path.home() / '.v6rge'
CONFIG_FILE = BASE_DIR / 'config.json'

# Tokens (Load from environment variables or .env file)
# See .env.example for required keys. Get your HuggingFace token from: https://huggingface.co/settings/tokens
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Email Configuration (Optional - for feedback feature)
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
MAILERSEND_API_KEY = os.environ.get("MAILERSEND_API_KEY", "")
MAILERSEND_DOMAIN = os.environ.get("MAILERSEND_DOMAIN", "")
SENDER_EMAIL = f"info@{MAILERSEND_DOMAIN}" if MAILERSEND_DOMAIN else ""

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

def get_or_create_user_id():
    c = load_config()
    if 'user_id' not in c:
        c['user_id'] = str(uuid.uuid4())
        save_config(c)
    return c['user_id']

# Load global config
config = load_config()
# Ensure User ID exists
get_or_create_user_id()
# Reload to get the ID we just saved (if any)
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

# Debug Logger
def log_debug(msg):
    try:
        log_path = Path.home() / 'Desktop' / 'v6rge_gpu_debug.txt'
        with open(log_path, 'a') as f:
            f.write(f"{msg}\n")
    except:
        pass

try:
    import torch
    log_debug(f"Torch Version: {torch.__version__}")
    
    print("\n" + "="*70)
    print("V6RGE BACKEND - GPU DETECTION")
    print("="*70)
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            compute_cap = torch.cuda.get_device_capability(0)
            
            print(f"✅ GPU DETECTED: {gpu_name}")
            print(f"   VRAM: {gpu_mem:.1f} GB")
            print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            
            msg = f"GPU: {gpu_name} ({gpu_mem:.1f}GB, CC {compute_cap[0]}.{compute_cap[1]})"
            log_debug(msg)
            
            # CUDA Version Check
            try:
                cuda_version = torch.version.cuda
                print(f"   CUDA Version: {cuda_version}")
                log_debug(f"CUDA Version: {cuda_version}")
                
                # RTX 50-series requires CUDA 12.6+ (they have compute capability 10.0)
                if compute_cap[0] >= 10:
                    print(f"\n⚠️  RTX 50-SERIES DETECTED")
                    print(f"   Your GPU ({gpu_name}) requires CUDA 12.6 or later.")
                    log_debug("RTX 50-series detected - requires CUDA 12.6+")
                    
                    if cuda_version:
                        try:
                            major_minor = float(cuda_version.split('.')[0] + '.' + cuda_version.split('.')[1])
                            if major_minor < 12.6:
                                print(f"   ⚠️  WARNING: Current PyTorch built with CUDA {cuda_version}")
                                print(f"   You may experience performance issues or errors.")
                                print(f"   Recommended: Install PyTorch with CUDA 12.6+")
                                print(f"   Visit: https://pytorch.org/get-started/locally/")
                                log_debug(f"WARNING: CUDA {cuda_version} < 12.6 for RTX 50-series")
                        except:
                            pass
                
                # General CUDA version guidance
                elif cuda_version:
                    try:
                        major_minor = float(cuda_version.split('.')[0] + '.' + cuda_version.split('.')[1])
                        if major_minor < 12.1:
                            print(f"   ⚠️  Your CUDA version ({cuda_version}) is older than recommended (12.1+)")
                            print(f"   Consider upgrading PyTorch for better performance.")
                            log_debug(f"Old CUDA version: {cuda_version}")
                    except:
                        pass
                    
            except Exception as cuda_err:
                print(f"   Warning: Could not detect CUDA version: {cuda_err}")
                log_debug(f"CUDA version detection error: {cuda_err}")
            
            print(f"\n🚀 Running in GPU Mode")
            print("="*70 + "\n")
            
        except Exception as e:
            msg = f"[OK] GPU Detected (details error: {e})"
            print(msg)
            log_debug(msg)
    else:
        msg = f"ℹ️  NO CUDA GPU DETECTED - Running in CPU Mode"
        print(msg)
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
        log_debug(msg)
        log_debug(f"CUDA Available Check: {torch.cuda.is_available()}")
    
    # Device Map Strategy (for transformers)
    TORCH_DTYPE = torch.float16 if DEVICE == 'cuda' else torch.float32

except ImportError:
    msg = "[WARNING] PyTorch not installed or failed to import."
    print(msg)
    log_debug(msg)
except Exception as e:
    msg = f"[ERROR] Error initializing PyTorch: {e}"
    print(msg)
    log_debug(msg)


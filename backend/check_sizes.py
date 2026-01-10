import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata, login
import json

# Setup paths to import config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import HF_TOKEN
from services.model_manager import model_manager

def format_size(size_bytes):
    return round(size_bytes / (1024**3), 2)  # GB

def check_sizes():
    print("Authenticating with HuggingFace...")
    if HF_TOKEN:
        login(token=HF_TOKEN)
        
    api = HfApi()

    registry = model_manager.registry
    updates = {}
    
    print(f"\nScanning {len(registry)} models for accurate sizes...\n")
    print("-" * 60)
    print(f"{'Model ID':<20} | {'Old GB':<8} | {'New GB':<8} | {'Status'}")
    print("-" * 60)

    for model_id, info in registry.items():
        if info.get('skip_download'):
            continue
            
        repo_id = info['hf_name']
        filename = info.get('filename')
        old_size = info.get('size_gb', 0)
        
        try:
            total_bytes = 0
            if filename:
                # Single File Mode (GGUF, etc)
                # We can use HfApi to get specific file info too
                # or just use get_hf_file_metadata
                url = hf_hub_url(repo_id=repo_id, filename=filename)
                meta = get_hf_file_metadata(url)
                total_bytes = meta.size
            else:
                # Repo Mode (Diffusers, Transformers)
                # Use HfApi().list_repo_files is too simple (no size)
                # Use HfApi().model_info returns siblings but often without size in basic call
                # Use HfFileSystem for sizing
                from huggingface_hub import HfFileSystem
                fs = HfFileSystem()
                
                # List recursively
                # This returns a list of dictionaries with 'name', 'size', 'type'
                files_info = fs.ls(repo_id, detail=True, recursive=True)
                
                for f_info in files_info:
                    fname = f_info['name'].replace(f"{repo_id}/", "")
                    
                    # Filtering Logic
                    if any(x in fname for x in ['.git', 'README.md', '.txt', '.png', '.jpg']):
                        continue
                        
                    # Handle separate fp16/fp32 folders if present?
                    # Most diffusers repos (flux, hunyuan) have a structure like:
                    # transformer/diffusion_pytorch_model.safetensors
                    # text_encoder/model.safetensors
                    # We just sum all safetensors/bins that aren't obviously duplicate variant folders
                    # But usually "main" branch is just one set.
                    
                    total_bytes += f_info['size']

            new_size_gb = format_size(total_bytes)
            updates[model_id] = new_size_gb
            
            # Highlight huge discrepancies (>20% diff)
            status = "OK"
            if abs(new_size_gb - old_size) > 1.0:
                status = "UPDATE"
                
            print(f"{model_id:<20} | {old_size:<8} | {new_size_gb:<8} | {status}")

        except Exception as e:
            print(f"{model_id:<20} | {old_size:<8} | {'ERROR':<8} | {str(e)[:20]}")

    print("-" * 60)
    print("\nPYTHON DICT UPDATE:\n")
    print(json.dumps(updates, indent=4))

if __name__ == "__main__":
    check_sizes()

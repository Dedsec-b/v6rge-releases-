import sys
import os
import time
import traceback

# Force line buffering
# Force line buffering (Only if attached to console)
if sys.stdout: sys.stdout.reconfigure(line_buffering=True)
if sys.stderr: sys.stderr.reconfigure(line_buffering=True)

# === IMPORTS ===
# Add current dir to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# === TQDM PATCH (Must be first) ===
import patch_tqdm

from config import BASE_DIR, MODELS_DIR, UPLOAD_FOLDER, OUTPUT_FOLDER, VOICES_FOLDER, MODELS_3D_FOLDER, DEVICE, HF_TOKEN, load_config, save_config

# === ENV SETUP ===
# Force Hugging Face to use our models directory as cache
os.environ['HF_HOME'] = str(MODELS_DIR)
os.environ['HF_TOKEN'] = HF_TOKEN

# === U2NET PATH FIX ===
# Force rembg to use our custom models directory
os.environ['U2NET_HOME'] = str(MODELS_DIR / 'u2net')

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from pathlib import Path
import json
import uuid
import re
import threading

# Progress tracking global
download_status = {}

# V6rge System Prompt
# === SYSTEM PROMPTS ===
BASE_SYSTEM_PROMPT = """You are V6rge, a powerful AI assistant with vision capabilities.
## Your Capabilities:
- 👁️ SEE AND ANALYZE IMAGES
- 🖼️ Generate images (Flux.1 or Qwen image)
- 🎵 Create music (MusicGen)
- 🗣️ Text-to-speech (Chatterbox TURBO)
- ✂️ Remove backgrounds from images
- 📈 Upscale images to 4K
- 🎤 Separate vocals from music
- 🎬 Generate videos (HunyuanVideo)
- 📦 Convert images to 3D models (Hunyuan3D 2.5)
- 📄 Read and analyze text files, code, and documents (User Uploaded)

## When to use tools:
- If user asks to CREATE/GENERATE/MAKE an image → respond with [TOOL:generate_image:prompt]
- If user asks for MUSIC/SOUNDTRACK/BEATS → respond with [TOOL:generate_music:prompt:duration]
- If user asks to READ ALOUD/SPEAK → respond with [TOOL:text_to_speech:text]
- If user asks for 3D MODEL from an uploaded image → respond with [TOOL:generate_3d:glb]

## Important Notes:
- **TTS Emotion Tags**: [laugh], [sigh], [gasp], [chuckle], [cough], [sniff], [groan]
- Respond naturally. When action is needed, use the tools."""

AGENT_SYSTEM_PROMPT = """
## ⚠️ GOD MODE (OS AGENT) ACTIVE
You have FULL ACCESS to the user's computer via Terminal and File System.

## 🛑 CRITICAL PROTOCOL (READ CAREFULLY):
To CREATE or EDIT a file, you **MUST** use the `[WRITE_FILE]` block as shown below. 
If you simply say "I created the file" without this block, **NOTHING HAPPENS**.

### 1. WRITE FILE SYNTAX
[WRITE_FILE:filename.ext]
File content goes here...
Multiple lines supported.
[END_WRITE_FILE]

### 2. OTHER TOOLS
- `[TOOL:terminal:command]` -> Execute PowerShell command (e.g. `mkdir`, `ping`).
- `[TOOL:read_file:path]` -> Read file content.
- `[TOOL:list_dir:path]` -> List directory.

## STRATEGY:
1. **THINK**: Plan your action.
2. **ACT**: Output the EXACT tool block.
3. **VERIFY**: Read the file back (using `read_file` or `list_dir`) to confirm it exists.
"""

# === IMPORTS ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.model_manager import model_manager
from services.chat_service import chat_service
from services.image_service import image_service
from services.music_service import music_service
from services.video_service import video_service
from services.tts_service import tts_service
from services.tools_service import tools_service
from services.feedback_service import feedback_bp
from services.network_service import network_service  # [NEW] Network Service

from services.terminal_service import terminal_service
from services.memory_service import memory_service

# Start Memory Watcher
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
# Run initial scan in background
# threading.Thread(target=memory_service.scan_directory, args=(desktop_path,), daemon=True).start()
# Start watcher
# memory_service.start_watcher(desktop_path)

# Setup TQDM Callback
def update_progress(downloaded, total, rate):
    # Store progress for the generic "active" download
    global download_status
    
    status = {
        'status': 'downloading',
        'progress': (downloaded / total * 100) if total > 0 else 0,
        'downloaded': downloaded,
        'total': total,
        'rate': rate
    }
    
    # Update global tracker (keyed by the ID requested in the API)
    if hasattr(app, 'current_download_id') and app.current_download_id:
        download_status[app.current_download_id] = status

# Register callback with patcher
patch_tqdm.set_progress_callback(update_progress)

loading_progress_callback = update_progress

# === FLASK APP ===
# Serve static files from ../app directory
APP_DIR = BASE_DIR.parent / 'app'
app = Flask(__name__, static_folder=str(APP_DIR), static_url_path='')
app.register_blueprint(feedback_bp)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def serve_index():
    return send_file(str(APP_DIR / 'index.html'))

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

# ==================== CONFIGURATION ====================
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

# === ROUTES ===

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
    
    # Also return user_id if present (for telemetry)
    user_id = config.get('user_id')
    return jsonify({'models_dir': current_dir, 'user_id': user_id})

# === ROUTES ===

@app.route('/')
def home():
    loaded_count = len([k for k, v in model_manager.loaded_models.items() if v is not None])
    return jsonify({
        'status': 'running',
        'message': 'V6rge Modular Backend',
        'device': DEVICE,
        'models_loaded': loaded_count
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/system/vram', methods=['GET'])
def get_vram_info():
    """Return GPU VRAM information for compatibility checks"""
    import torch
    
    vram_info = {
        'cuda_available': False,
        'gpu_name': None,
        'vram_total_gb': 0,
        'vram_free_gb': 0,
        'vram_used_gb': 0
    }
    
    try:
        if torch.cuda.is_available():
            vram_info['cuda_available'] = True
            vram_info['gpu_name'] = torch.cuda.get_device_name(0)
            
            # Get VRAM in bytes, convert to GB
            total = torch.cuda.get_device_properties(0).total_memory
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            free = total - reserved
            
            vram_info['vram_total_gb'] = round(total / (1024**3), 2)
            vram_info['vram_free_gb'] = round(free / (1024**3), 2)
            vram_info['vram_used_gb'] = round(allocated / (1024**3), 2)
    except Exception as e:
        vram_info['error'] = str(e)
    
    return jsonify(vram_info)

# === MODEL STATUS & GENERIC MANAGEMENT ===
@app.route('/models/status', methods=['GET'])
def get_models_status():
    return jsonify(model_manager.get_status())

@app.route('/models/download/<model_id>', methods=['POST'])
def download_model(model_id):
    # This triggers the download in a non-blocking background thread
    import threading
    
    import shutil
    
    # Check Disk Space
    config = model_manager.get_model_config(model_id)
    if config and 'size_gb' in config:
        required_gb = config['size_gb']
        # Check space on the drive where MODELS_DIR is located
        try:
            # Ensure dir exists for check, or use parent
            check_dir = MODELS_DIR if MODELS_DIR.exists() else MODELS_DIR.parent
            total, used, free = shutil.disk_usage(check_dir)
            free_gb = free / (1024**3)
            
            print(f"[Disk Check] Dir: {check_dir} | Free: {free_gb:.2f} GB | Required: {required_gb} GB")

            if free_gb < (required_gb + 2): # 2GB buffer
                error_msg = f"Insufficient Disk Space. Required: {required_gb}GB, Available: {free_gb:.2f}GB"
                print(f"[Disk Check] FAILED: {error_msg}")
                return jsonify({
                    'status': 'error', 
                    'message': error_msg,
                    'error': error_msg  # Frontend expects 'error' field now too
                }), 400
        except Exception as e:
            print(f"[Disk Check] Error checking disk space: {e}")
            # Fallthrough or return error? Let's log and proceed cautiously or fail safe?
            # If we allow it, it might fill disk. If we block, we might block valid cases.
            # Let's fail safe for now but log loudly.
            pass

    # Set context for progress tracker
    app.current_download_id = model_id
    download_status[model_id] = {'status': 'pending', 'progress': 0}
    
    def run_download():
        try:
            if model_service := get_service_for_model(model_id):
                model_service.load_model(model_id)
                
                # Mark as complete
                download_status[model_id] = {
                    'status': 'ready',
                    'progress': 100.0,
                    'message': 'Model ready'
                }
        except Exception as e:
            print(f"Download Error for {model_id}: {e}")
            download_status[model_id] = {
                'status': 'error', 
                'error': str(e),
                'progress': 0
            }

    thread = threading.Thread(target=run_download)
    thread.start()
    
    return jsonify({'status': 'started', 'message': f'Download started for {model_id}'})

def get_service_for_model(model_id):
    # Helper to map IDs to services
    # ORDER MATTERS: More specific matches first!
    
    # Image models (check before 'qwen' to catch qwen-image)
    if model_id == 'qwen-image': return image_service
    if 'flux' in model_id: return image_service
    
    # Chat models
    if 'qwen' in model_id: return chat_service
    
    # Music
    if 'music' in model_id: return music_service
    
    # TTS
    if 'tts' in model_id or 'chatterbox' in model_id: return tts_service
    
    # 3D (check before video to catch hunyuan3d)
    if 'hunyuan3d' in model_id or '3d' in model_id: return tools_service
    
    # Video
    if 'hunyuan' in model_id or 'video' in model_id: return video_service
    
    # Tools
    if 'realesrgan' in model_id: return tools_service
    if 'u2net' in model_id: return tools_service
    if 'roformer' in model_id or 'vocal' in model_id: return tools_service
    
    return None

@app.route('/models/progress/<model_id>', methods=['GET'])
def get_model_progress(model_id):
    # Check if loaded first (fast path)
    if get_service_for_model(model_id):
        pass

    # Check tracker
    if model_id in download_status:
        return jsonify(download_status[model_id])
        
    return jsonify({
        'status': 'unknown',
        'progress': 0.0
    })

@app.route('/models/delete/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    try:
        # Unload from memory first (Reset GC to free file locks)
        model_manager.unload_model(model_id)
            
        # Dispatch to service for file deletion
        service = get_service_for_model(model_id)
        if service and hasattr(service, 'delete_model'):
            success = service.delete_model(model_id)
            if success:
                # Clear status
                if model_id in download_status:
                    del download_status[model_id]
                return jsonify({'status': 'deleted'})
            else:
                return jsonify({'error': 'Failed to delete file (Check logs)'}), 500
        
        # Fallback if service doesn't have specific delete logic
        return jsonify({'error': 'Delete not implemented for this model type'}), 501

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === CHAT ===
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Handle both FormData and JSON
        data = request.form if request.form else (request.get_json() or {})
        
        model_id = data.get('model_id')
        if model_id: chat_service.load_model(model_id)
        else: chat_service.load_model('qwen-3b')
            
        message = data.get('message', '')
        history_str = data.get('history', '[]')
        history = json.loads(history_str) if isinstance(history_str, str) else history_str
        
        # Check Agent Mode (str 'true' or bool True)
        agent_mode_raw = data.get('agent_mode', False)
        agent_mode = str(agent_mode_raw).lower() == 'true'
        
        # Select Prompt
        current_system_prompt = BASE_SYSTEM_PROMPT
        if agent_mode:
            current_system_prompt += AGENT_SYSTEM_PROMPT
            print("[AGENT MODE ACTIVE]")
        
        # Save uploaded files (essential for Vision/3D tools)
        latest_file_path = None
        for key in request.files:
            file = request.files[key]
            if file.filename:
                # Save to UPLOAD_FOLDER
                safe_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
                path = UPLOAD_FOLDER / safe_name
                file.save(str(path))
                latest_file_path = str(path)
                print(f"Saved chat attachment: {path}")

        # Build messages
        messages = [{"role": "system", "content": current_system_prompt}] 
        
        # Add history
        for h in history[-10:]:
             messages.append({"role": h.get('role', 'user'), "content": h.get('content', '')})
        
        if latest_file_path:
             # AUTO-SWITCH TO VISION MODEL
             # If it's an image and Qwen2-VL is available, use it.
             ext = os.path.splitext(latest_file_path)[1].lower()
             is_image = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
             
             if is_image and model_manager.is_downloaded('qwen2-vl'):
                 print("Image detected + Qwen2-VL available. Switching to Vision Model.")
                 model_id = 'qwen2-vl'
                 chat_service.load_model('qwen2-vl')
                 image_paths = [latest_file_path]
                 
                 # Append context to message
                 messages.append({"role": "user", "content": f"{message} [Attached Image: {os.path.basename(latest_file_path)}]"})
             else:
                 # Standard file attachment notification
                 messages.append({"role": "user", "content": f"{message} [Attached: {os.path.basename(latest_file_path)}]"})
                 image_paths = None
        else:
             messages.append({"role": "user", "content": message})
             image_paths = None

        response_text = chat_service.generate(messages, images=image_paths)
        
        # Tool Parsing Logic
        tool_result = None
        
        try:
            # 1. Check for SPECIAL WRITE_FILE BLOCK
            write_match = re.search(r'\[WRITE_FILE:(.*?)\](.*?)\[END_WRITE_FILE\]', response_text, re.DOTALL)
            if write_match:
                path_arg = write_match.group(1).strip()
                content_arg = write_match.group(2).strip()
                
                tool_result = {
                    'type': 'approval_required',
                    'tool': 'write_file',
                    # Combine into the path|content format the Frontend expects
                    'command': f"{path_arg}|{content_arg}"
                }
                # Mask huge output
                response_text = response_text.replace(write_match.group(0), f"[Requesting Write to {path_arg}...]").strip()

            # 2. Check for standard tools if no write_file found
            if not tool_result:
                # Regex to find [TOOL:name:args]
                # Supports arguments with colons if not strictly split? 
                # The prompt says [TOOL:generate_image:prompt] -> split by first 2 colons?
                match = re.search(r'\[TOOL:(\w+):(.*?)\]', response_text, re.DOTALL)
                if match:
                    tool_name = match.group(1)
                    # Arguments might be "prompt:duration" for music
                    # simple split by colon for args
                    raw_args = match.group(2)
                    
                    print(f"Detected Tool Call: {tool_name} with args {raw_args}")
                    
                    if tool_name == 'memory':
                        # args = query
                        query = raw_args
                        results = memory_service.search(query)
                        if results:
                            tool_result = {'type': 'text', 'content': f"Found {len(results)} files:\n" + "\n".join([f"- {r['path']} (Modified: {time.ctime(r['modified'])})" for r in results])}
                        else:
                            tool_result = {'type': 'text', 'content': "No files found matching query."}
                        
                        response_text += f"\n\n[System Memory Result]:\n{tool_result['content']}"

                    elif tool_name in ['terminal', 'read_file', 'list_dir']:
                        # GOD MODE: Pause for approval
                        tool_result = {
                            'type': 'approval_required',
                            'tool': tool_name,
                            'command': raw_args,
                            'args': raw_args
                        }
                        # STRIP the tool command from the user-facing text
                        response_text = response_text.replace(match.group(0), "").strip()
                        if not response_text:
                            readable_action = "Reading file" if tool_name == 'read_file' else "Listing directory" if tool_name == 'list_dir' else "Executing command"
                            response_text = f"Requesting system access ({readable_action})..."
                        
                        # We do NOT run the command here. We return special JSON.
                    
                    elif tool_name == 'generate_image':
                        # args = prompt
                        prompt = raw_args
                        url = image_service.generate(prompt)
                        tool_result = {'type': 'image', 'url': url}
                        
                    elif tool_name == 'generate_music':
                        # args = prompt:duration (optional)
                        parts = raw_args.split(':')
                        prompt = parts[0]
                        duration = 10
                        if len(parts) > 1:
                            try: duration = int(parts[1])
                            except: pass
                        url = music_service.generate(prompt, duration=duration)
                        tool_result = {'type': 'audio', 'url': url}
                        
                    elif tool_name == 'text_to_speech':
                        # args = text
                        text = raw_args
                        path = tts_service.generate(text)
                        # Convert local path to download URL
                        filename = os.path.basename(path)
                        tool_result = {'type': 'audio', 'url': f"/download/{filename}"}
                        
                    elif tool_name == 'generate_3d':
                        # prompt often just says "glb"
                        # This relies on the latest uploaded file
                        if latest_file_path:
                            # args might be format
                            fmt = raw_args.strip() if raw_args.strip() in ['glb', 'obj', 'stl'] else 'glb'
                            url = tools_service.generate_3d(latest_file_path, output_format=fmt)
                            tool_result = {'type': 'model', 'url': url}
                        else:
                            response_text += "\n\n(Error: Please upload an image to generate a 3D model.)"

        except Exception as tool_err:
            print(f"Tool Execution Failed: {tool_err}")
            traceback.print_exc()
            response_text += f"\n\n[System Error: Failed to execute tool {tool_name}]"

        return jsonify({'response': response_text, 'tool_result': tool_result})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/tool/execute', methods=['POST'])
def execute_tool():
    """Execute a tool after approval (Terminal Only for now)"""
    try:
        data = request.json
        tool = data.get('tool')
        command = data.get('command')
        
        if tool == 'terminal':
            output = terminal_service.execute(command)
            return jsonify({'status': 'success', 'output': output})
        elif tool == 'read_file':
            output = terminal_service.read_file(command)
            return jsonify({'status': 'success', 'output': output})
        elif tool == 'list_dir':
            output = terminal_service.list_dir(command)
            return jsonify({'status': 'success', 'output': output})
        elif tool == 'write_file':
            # Handling path|content split
            if '|' in command:
                path, content = command.split('|', 1)
                output = terminal_service.write_file(path, content)
                return jsonify({'status': 'success', 'output': output})
            else:
                return jsonify({'error': 'Invalid write_file format. Use path|content'}), 400
            
        return jsonify({'error': 'Unknown tool'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === IMAGE ===
@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        prompt = request.form.get('prompt')
        width = int(request.form.get('width', 1024))
        height = int(request.form.get('height', 1024))
        steps = int(request.form.get('steps', 4))
        guidance = float(request.form.get('guidance', 0.0))
        model_id = request.form.get('model')  # Frontend sends 'model'
        negative_prompt = request.form.get('negative_prompt', '')
        prompt_enhance = request.form.get('prompt_enhance') == 'true'
        
        url = image_service.generate(
            prompt, 
            width=width, 
            height=height, 
            steps=steps, 
            guidance=guidance, 
            model_id=model_id,
            negative_prompt=negative_prompt,
            prompt_enhance=prompt_enhance
        )
        return jsonify({'status': 'success', 'image_url': url})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/edit_image', methods=['POST'])
def edit_image():
    if 'image' not in request.files: return jsonify({'error': 'No image'}), 400
    file = request.files['image']
    path = UPLOAD_FOLDER / file.filename
    file.save(str(path))
    
    prompt = request.form.get('prompt')
    url = image_service.edit(str(path), prompt)
    return jsonify({'status': 'success', 'image_url': url})

# === MUSIC ===
@app.route('/generate_music', methods=['POST'])
def generate_music():
    try:
        prompt = request.form.get('prompt')
        duration = float(request.form.get('duration', 10))
        url = music_service.generate(prompt, duration=duration)
        return jsonify({'status': 'success', 'audio_url': url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === VIDEO ===
@app.route('/generate_video', methods=['POST'])
def generate_video():
    try:
        prompt = request.form.get('prompt')
        url = video_service.generate(prompt)
        return jsonify({'status': 'success', 'video_url': url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === TTS ===
@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.form.get('text')
    try:
        path = tts_service.generate(text)
        return send_file(path, mimetype='audio/wav')
    except Exception as e:
         return jsonify({'error': str(e)}), 500

# === TOOLS ===
@app.route('/remove_background', methods=['POST'])
def remove_bg():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    path = UPLOAD_FOLDER / file.filename
    file.save(str(path))
    url = tools_service.remove_bg(str(path))
    return jsonify({'status': 'success', 'image_url': url})

@app.route('/upscale', methods=['POST'])
def upscale():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    path = UPLOAD_FOLDER / file.filename
    file.save(str(path))
    url = tools_service.upscale(str(path))
    return jsonify({'status': 'success', 'upscaled_url': url})

@app.route('/process', methods=['POST']) # Vocal Sep
def vocal_sep():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    path = UPLOAD_FOLDER / file.filename
    file.save(str(path))
    inst, vocal = tools_service.separate_vocals(str(path))
    
    # Needs to return URLs, this assumes simplified return for now
    return jsonify({
        'status': 'success', 
        'instrumental_url': f"/download/{os.path.basename(inst)}",
        'vocals_url': f"/download/{os.path.basename(vocal)}"
    }) 

# === DOWNLOADS ===
@app.route('/download/<path:p>')
def download(p):
    path = OUTPUT_FOLDER / p
    if path.exists():
        return send_file(str(path))
    return jsonify({'error': 'Not found'}), 404

# === NETWORK / HOTSPOT ===
@app.route('/network/hotspot/start', methods=['POST'])
def start_hotspot():
    return jsonify(network_service.start_hotspot())

@app.route('/network/hotspot/stop', methods=['POST'])
def stop_hotspot():
    return jsonify(network_service.stop_hotspot())

@app.route('/network/hotspot/status', methods=['GET'])
def hotspot_status():
    return jsonify(network_service.get_status())

# === CONFIG ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

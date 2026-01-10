"""
V6rge Modular Backend Server
"""
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
V6RGE_SYSTEM_PROMPT = """You are V6rge, a powerful AI assistant with vision AND SYSTEM CONTROL capabilities.
## Your Capabilities:
- 👁️ SEE AND ANALYZE IMAGES
- � **CONTROL THE COMPUTER** via Terminal Commands
- �🖼️ Generate images (Flux.1 or Qwen image)
- 🎵 Create music (MusicGen)
- 🗣️ Text-to-speech (Chatterbox TURBO)
- ✂️ Remove backgrounds from images
- 📈 Upscale images to 4K
- 🎤 Separate vocals from music
- 🎬 Generate videos (HunyuanVideo)
- 📦 Convert images to 3D models (Hunyuan3D 2.5)
- 📄 Read and analyze text files, code, and documents
- 🧠 **MEMORY**: Instantly find files on the computer without searching folders manually.
## When to use tools:
- **SYSTEM/TERMINAL (OS AGENT MODE)**: Control the Windows computer via PowerShell/CMD.
  **Strategy for Complex Tasks**:
  1. EXPLORE first → [TOOL:terminal:dir] or [TOOL:terminal:Get-ChildItem]
  2. PLAN what to do based on results
  3. SEARCH for files if needed → [TOOL:memory:query]
  4. For multi-step tasks, write a PowerShell or Python script → [TOOL:terminal:echo "script content" > script.ps1]
  **PowerShell Examples** (preferred for complex tasks):
     - List processes: [TOOL:terminal:powershell -Command "Get-Process | Select Name,CPU | Sort CPU -Desc | Select -First 10"]
     - Find large files: [TOOL:terminal:powershell -Command "Get-ChildItem -Recurse | Sort Length -Desc | Select -First 5"]
     - Find "resume.pdf": [TOOL:memory:resume] (Much faster than terminal search)
     - Create file: [TOOL:terminal:powershell -Command "Set-Content -Path 'file.txt' -Value 'content'"]
  **CMD Examples** (for simple tasks):
     - Check IP: [TOOL:terminal:ipconfig /all]
     - List files: [TOOL:terminal:dir /b]
     - Create file: [TOOL:terminal:echo "content" > file.txt]
- If user asks to CREATE/GENERATE/MAKE an image → respond with [TOOL:generate_image:prompt]
- If user asks for MUSIC/SOUNDTRACK/BEATS → respond with [TOOL:generate_music:prompt:duration]
- If user asks to READ ALOUD/SPEAK → respond with [TOOL:text_to_speech:text]
- If user asks for 3D MODEL from an uploaded image → respond with [TOOL:generate_3d:glb]

## Important Notes:
- **TTS Emotion Tags**: When using text-to-speech, you can add emotions: [laugh], [sigh], [gasp], [chuckle], [cough], [sniff], [groan]
- **Be Autonomous**: For complex tasks, don't just give instructions—actually DO the task using multiple tool calls.
- **Plan First**: For tasks involving many files or complex logic, explore first (dir/Get-ChildItem), then act.

Respond naturally. When action is needed, use the tools."""

# === IMPORTS ===
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.model_manager import model_manager
from services.chat_service import chat_service
from services.image_service import image_service
from services.music_service import music_service
from services.video_service import video_service
from services.tts_service import tts_service
from services.tools_service import tools_service

from services.terminal_service import terminal_service
from services.memory_service import memory_service

# Start Memory Watcher
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
# Run initial scan in background
threading.Thread(target=memory_service.scan_directory, args=(desktop_path,), daemon=True).start()
# Start watcher
memory_service.start_watcher(desktop_path)

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
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response

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

# === MODEL STATUS & GENERIC MANAGEMENT ===
@app.route('/models/status', methods=['GET'])
def get_models_status():
    return jsonify(model_manager.get_status())

@app.route('/models/download/<model_id>', methods=['POST'])
def download_model(model_id):
    # This triggers the download in a non-blocking background thread
    import threading
    
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
    model_id = request.form.get('model_id')
    try:
        if model_id: chat_service.load_model(model_id)
        else: chat_service.load_model('qwen-3b')
            
        message = request.form.get('message', '')
        history_str = request.form.get('history', '[]')
        history = json.loads(history_str) if history_str else []
        
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
        # V6rge System Prompt (Already defined above)
        messages = [{"role": "system", "content": V6RGE_SYSTEM_PROMPT}] 
        
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
            # Regex to find [TOOL:name:args]
            # Supports arguments with colons if not strictly split? 
            # The prompt says [TOOL:generate_image:prompt] -> split by first 2 colons?
            match = re.search(r'\[TOOL:(\w+):(.*?)\]', response_text)
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

                elif tool_name == 'terminal':
                     # GOD MODE: Pause for approval
                     tool_result = {
                         'type': 'approval_required',
                         'tool': 'terminal',
                         'command': raw_args,
                         'args': raw_args
                     }
                     # STRIP the tool command from the user-facing text
                     response_text = response_text.replace(match.group(0), "").strip()
                     if not response_text:
                         response_text = "Requesting system access..."
                     
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

# === CONFIG ===
@app.route('/config', methods=['GET'])
def get_config():
    # Return current effective config
    return jsonify({
        'models_dir': str(MODELS_DIR),
        'base_dir': str(BASE_DIR)
    })

@app.route('/config/update', methods=['POST'])
def update_config():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        current_config = load_config()
        
        if 'models_dir' in data:
            current_config['models_dir'] = data['models_dir']
            
        save_config(current_config)
        
        return jsonify({
            'status': 'success',
            'message': 'Configuration saved. Restart required.'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)

import torch
import scipy.io.wavfile as wavfile
import uuid
from transformers import AutoProcessor, MusicgenForConditionalGeneration

try:
    from config import MODELS_DIR, DEVICE, OUTPUT_FOLDER
    from services.model_manager import model_manager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODELS_DIR, DEVICE, OUTPUT_FOLDER
    from services.model_manager import model_manager

class MusicService:
    def load_model(self, model_id='musicgen-large'):
        current_model = model_manager.get_loaded_model('music')
        current_id = model_manager.current_model_ids.get('music')
        
        if current_model is not None and current_id == model_id:
            return current_model, model_manager.get_loaded_model('music_processor')

        config = model_manager.get_model_config(model_id)
        if not config:
            model_id = 'musicgen-large'
            config = model_manager.get_model_config(model_id)
            
        print(f"Loading Music Model: {model_id}...")
        
        processor = AutoProcessor.from_pretrained(
            config['hf_name'],
            cache_dir=MODELS_DIR
        )
        
        model = MusicgenForConditionalGeneration.from_pretrained(
            config['hf_name'],
            cache_dir=MODELS_DIR
        ).to(DEVICE)
        
        model_manager.register_model('music', model, model_id)
        model_manager.loaded_models['music_processor'] = processor
        
        print(f"Music Model Ready")
        return model, processor

    def generate(self, prompt, duration=10, guidance=3.0, temperature=1.0, model_id='musicgen-large'):
        model, processor = self.load_model(model_id)
        
        # Validate and clamp duration to safe limits
        duration = max(5, min(duration, 30))  # MusicGen works best with 5-30 seconds
        
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        # MusicGen generates ~51.2 tokens/second (sampling rate 32000 / frame size 625)
        # Use safe calculation with bounds checking to prevent index errors
        max_new_tokens = int(duration * 51.2)
        max_new_tokens = min(max_new_tokens, 1500)  # Hard cap at 1500 tokens (~29 seconds)
        
        print(f"[MusicGen] Generating {duration}s audio ({max_new_tokens} tokens)...", flush=True)
        
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            guidance_scale=guidance,
            temperature=temperature,
            do_sample=True
        )
        
        # Safely extract audio data (handle potential shape variations)
        if audio_values.dim() == 3:
            audio_data = audio_values[0, 0].cpu().numpy()
        else:
            audio_data = audio_values[0].cpu().numpy()
        
        sr = model.config.audio_encoder.sampling_rate
        output_name = f"track_{uuid.uuid4().hex[:8]}.wav"
        wavfile.write(
            str(OUTPUT_FOLDER / output_name),
            rate=sr,
            data=audio_data
        )
        
        print(f"[MusicGen] ✅Generated {output_name}", flush=True)
        return f"/download/{output_name}"


    def delete_model(self, model_id):
        try:
            # Unload from memory first by ensuring manager clears it
            if model_manager.get_loaded_model('music'):
                 model_manager.loaded_models['music'] = None
                 model_manager.loaded_models['music_processor'] = None
                 
            # Find files (HF cache is hard to clean precisely by ID without full path logic)
            # But we can try to find the folder in models dir if it exists
            # For MusicGen large, it's facebook/musicgen-large
            config = model_manager.get_model_config(model_id)
            if config:
                repo_folder = f"models--{config['hf_name'].replace('/', '--')}"
                repo_path = MODELS_DIR / repo_folder
                if repo_path.exists():
                    import shutil
                    shutil.rmtree(repo_path)
                    print(f"Deleted {repo_path}")
            return True
        except Exception as e:
            print(f"Error deleting music model: {e}")
            return False

music_service = MusicService()

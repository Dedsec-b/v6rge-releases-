import torch
import soundfile as sf
import uuid
from pathlib import Path

try:
    from config import MODELS_DIR, DEVICE, OUTPUT_FOLDER
    from services.model_manager import model_manager
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODELS_DIR, DEVICE, OUTPUT_FOLDER
    from services.model_manager import model_manager

class TTSService:
    def __init__(self):
        self.sample_rate = 24000
        self.model = None

    def load_model(self, model_id=None):
        """Load Chatterbox Turbo TTS"""
        if self.model is not None:
            return self.model
            
        print("Loading Chatterbox Turbo TTS...")
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            try:
                from config import HF_TOKEN
            except ImportError:
                HF_TOKEN = None

            if HF_TOKEN:
                print(f"Logging in to Hugging Face with token: {HF_TOKEN[:4]}...")
                import huggingface_hub
                huggingface_hub.login(token=HF_TOKEN)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = ChatterboxTurboTTS.from_pretrained(device=device)
            self.sample_rate = self.model.sr
            
            model_manager.register_model('tts', self.model, 'chatterbox-turbo')
            print(f"Chatterbox TTS Ready")
            return self.model
            
        except ImportError:
            print("MISSING DEPENDENCY: chatterbox-tts. Install with: pip install chatterbox-tts")
            raise Exception("Missing dependency: chatterbox-tts")
        except Exception as e:
            import traceback
            print(f"TTS Load Error: {e}")
            traceback.print_exc()
            raise e

    def delete_model(self, model_id):
        try:
             # Unload
            if model_manager.get_loaded_model('tts'):
                model_manager.unload_model('tts')
                
            config = model_manager.get_model_config(model_id)
            if config:
                repo_folder = f"models--{config['hf_name'].replace('/', '--')}"
                
                # Check both root and hub locations
                paths_to_check = [
                    MODELS_DIR / repo_folder,
                    MODELS_DIR / 'hub' / repo_folder
                ]
                
                import shutil
                deleted_any = False
                for repo_path in paths_to_check:
                    if repo_path.exists():
                        shutil.rmtree(repo_path, ignore_errors=True)
                        print(f"Deleted {repo_path}")
                        deleted_any = True
                
                if deleted_any:
                    return True
            return True
        except Exception as e:
            print(f"Error deleting tts model: {e}")
            return False

    def generate(self, text):
        model = self.load_model()
        
        wav = model.generate(text)
        audio_data = wav.cpu().squeeze().numpy()
        
        output_name = f"tts_{uuid.uuid4().hex[:8]}.wav"
        output_path = OUTPUT_FOLDER / output_name
        sf.write(str(output_path), audio_data, self.sample_rate)
        
        return str(output_path)

tts_service = TTSService()


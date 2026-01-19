import torch
import gc
import uuid
import traceback
from diffusers import HunyuanVideoPipeline
from diffusers.utils import export_to_video

try:
    from config import MODELS_DIR, DEVICE, OUTPUT_FOLDER
    from services.model_manager import model_manager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODELS_DIR, DEVICE, OUTPUT_FOLDER
    from services.model_manager import model_manager

class VideoService:
    def load_model(self, model_id='hunyuan-video'):
        current_model = model_manager.get_loaded_model('video')
        current_id = model_manager.current_model_ids.get('video')
        
        if current_model is not None and current_id == model_id:
            return current_model

        config = model_manager.get_model_config(model_id)
        if not config:
            model_id = 'hunyuan-video'
            config = model_manager.get_model_config(model_id)
            
        print(f"Loading Video Model: {model_id}...")
        
        
        pipeline = HunyuanVideoPipeline.from_pretrained(
            config['hf_name'],
            torch_dtype=torch.bfloat16,
            transformer_dtype=torch.float8_e4m3fn,
            cache_dir=MODELS_DIR
        )
        pipeline.vae.enable_tiling()
        pipeline.enable_model_cpu_offload()
        
        model_manager.register_model('video', pipeline, model_id)
        print(f"Video Model Ready")
        return pipeline

    def generate(self, prompt, frames=65, steps=30, fps=24):
        pipeline = self.load_model()
        
        video = pipeline(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=steps,
            num_frames=frames,
            generator=torch.Generator(DEVICE).manual_seed(42)
        ).frames[0]
        
        output_name = f"video_{uuid.uuid4().hex[:8]}.mp4"
        export_to_video(video, str(OUTPUT_FOLDER / output_name), fps=fps)
        
        return f"/download/{output_name}"

    def delete_model(self, model_id):
        try:
            # Unload
            if model_manager.get_loaded_model('video'):
                model_manager.unload_model('video')

            config = model_manager.get_model_config(model_id)
            if config:
                repo_folder = f"models--{config['hf_name'].replace('/', '--')}"
                repo_path = MODELS_DIR / repo_folder
                if repo_path.exists():
                    import shutil
                    shutil.rmtree(repo_path, ignore_errors=True)
                    print(f"Deleted {repo_path}")
            return True
        except Exception as e:
            print(f"Error deleting video model: {e}")
            return False

video_service = VideoService()

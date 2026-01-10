import torch
import gc
import traceback
import uuid
from PIL import Image
from diffusers import FluxPipeline, FluxImg2ImgPipeline, DiffusionPipeline

try:
    from config import MODELS_DIR, DEVICE, OUTPUT_FOLDER
    from services.model_manager import model_manager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODELS_DIR, DEVICE, OUTPUT_FOLDER
    from services.model_manager import model_manager

class ImageService:
    def __init__(self):
        self.current_model_type = None  # 'flux' or 'qwen'
    
    def load_model(self, model_id='flux-schnell'):
        current_model = model_manager.get_loaded_model('image')
        current_id = model_manager.current_model_ids.get('image')
        
        if current_model is not None and current_id == model_id:
            return current_model, model_manager.get_loaded_model('image_edit')

        config = model_manager.get_model_config(model_id)
        if not config: 
            model_id = 'flux-schnell'
            config = model_manager.get_model_config(model_id)
            
        print(f"Loading Image Model: {model_id}...")
        
        # Unload existing model
        if current_model is not None:
            model_manager.unload_model('image')
            if model_manager.get_loaded_model('image_edit'):
                model_manager.loaded_models['image_edit'] = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Load based on model type
        if model_id == 'qwen-image':
            return self._load_qwen_image(model_id, config)
        else:
            return self._load_flux(model_id, config)
    
    def _load_flux(self, model_id, config):
        """Load FLUX pipeline"""
        pipeline = FluxPipeline.from_pretrained(
            config['hf_name'],
            torch_dtype=torch.bfloat16,
            cache_dir=MODELS_DIR
        )
        pipeline.enable_model_cpu_offload()
        
        # Create Edit Pipeline sharing components
        edit_pipeline = FluxImg2ImgPipeline(
            transformer=pipeline.transformer,
            scheduler=pipeline.scheduler,
            vae=pipeline.vae,
            text_encoder=pipeline.text_encoder,
            text_encoder_2=pipeline.text_encoder_2,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
        )
        
        self.current_model_type = 'flux'
        model_manager.register_model('image', pipeline, model_id)
        model_manager.loaded_models['image_edit'] = edit_pipeline
        
        print(f"FLUX Image Model Ready")
        return pipeline, edit_pipeline
    
    def _load_qwen_image(self, model_id, config):
        """Load Qwen-Image pipeline"""
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        pipeline = DiffusionPipeline.from_pretrained(
            config['hf_name'],
            torch_dtype=dtype,
            cache_dir=MODELS_DIR
        )
        
        if torch.cuda.is_available():
            pipeline.to("cuda")
        
        self.current_model_type = 'qwen'
        model_manager.register_model('image', pipeline, model_id)
        model_manager.loaded_models['image_edit'] = None
        
        print(f"Qwen-Image Model Ready")
        return pipeline, None

    def generate(self, prompt, width=1024, height=1024, steps=4, guidance=0.0, seed=0, 
                 model_id=None, negative_prompt='', prompt_enhance=False):
        pipeline, _ = self.load_model(model_id or 'flux-schnell')
        
        generator = None
        if seed > 0:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device).manual_seed(seed)
        
        # Qwen-Image generation
        if self.current_model_type == 'qwen':
            final_prompt = prompt
            if prompt_enhance:
                final_prompt = f"{prompt}, Ultra HD, 4K, cinematic composition, professional photography"
            
            image = pipeline(
                prompt=final_prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator
            ).images[0]
        else:
            # FLUX generation
            image = pipeline(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                max_sequence_length=256,
                generator=generator
            ).images[0]
        
        output_name = f"gen_{uuid.uuid4().hex[:8]}.png"
        image.save(str(OUTPUT_FOLDER / output_name))
        return f"/download/{output_name}"

    def edit(self, image_path, prompt, strength=0.6):
        _, edit_pipeline = self.load_model('flux-schnell')
        
        if edit_pipeline is None:
            raise RuntimeError("Image editing not supported for current model")
        
        init_image = Image.open(image_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        
        image = edit_pipeline(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=4,
            guidance_scale=0.0,
            max_sequence_length=256
        ).images[0]
        
        output_name = f"edit_{uuid.uuid4().hex[:8]}.png"
        image.save(str(OUTPUT_FOLDER / output_name))
        return f"/download/{output_name}"

    def delete_model(self, model_id):
        try:
             # Unload
            if model_manager.get_loaded_model('image'):
                model_manager.unload_model('image')
                
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
            print(f"Error deleting image model: {e}")
            return False

image_service = ImageService()

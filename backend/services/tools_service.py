import torch
import cv2
import shutil
import uuid
import os
import traceback
from PIL import Image
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import torchvision
from torchvision.transforms import functional as F

# AGGRESSIVE MONKEYPATCH: Fixes 'No module named torchvision.transforms.functional_tensor'
# This forces compatible logic into the place where basicsr expects it
try:
    if not hasattr(torchvision.transforms, 'functional_tensor'):
        torchvision.transforms.functional_tensor = F
    sys.modules['torchvision.transforms.functional_tensor'] = F
except Exception:
    pass

try:
    from config import MODELS_DIR, DEVICE, UPLOAD_FOLDER, OUTPUT_FOLDER, MODELS_3D_FOLDER
    from config import MODELS_DIR, DEVICE, UPLOAD_FOLDER, OUTPUT_FOLDER, MODELS_3D_FOLDER
    from services.model_manager import model_manager
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODELS_DIR, DEVICE, UPLOAD_FOLDER, OUTPUT_FOLDER, MODELS_3D_FOLDER
    from services.model_manager import model_manager

class ToolsService:
    def load_model(self, model_id):
        """Generic loader for download compatibility"""
        if 'realesrgan' in model_id: return self.load_upscaler()
        if 'u2net' in model_id: return self.load_bg_remover()
        if 'roformer' in model_id: return self.load_vocal_separator()
        if '3d' in model_id: return self.load_3d()
        return None

    # === VOCAL SEPARATOR ===
    def load_vocal_separator(self):
        if model_manager.get_loaded_model('vocal_sep'): return model_manager.get_loaded_model('vocal_sep')
        
        print("Loading Vocal Separator...")
        from audio_separator.separator import Separator
        
        sep = Separator(
            output_dir=str(OUTPUT_FOLDER),
            model_file_dir=str(MODELS_DIR),
            mdx_params={"segment_size": 2048, "overlap": 0.95, "denoise": True}
        )
        
        # Ensure model file exists in expected location
        target_filename = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
        target_path = MODELS_DIR / target_filename
        
        if not target_path.exists():
            hf_path = hf_hub_download(
                repo_id="KitsuneX07/Music_Source_Sepetration_Models", 
                filename="vocal_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
                cache_dir=MODELS_DIR
            )
            shutil.copy(hf_path, str(target_path))
            
        sep.load_model(target_filename)
        sep.load_model(target_filename)
        model_manager.register_model('vocal_sep', sep, model_id='bs-roformer')
        return sep

    def separate_vocals(self, file_path):
        sep = self.load_vocal_separator()
        output_files = sep.separate(str(file_path))
        
        inst = next((f for f in output_files if 'Instrumental' in f), None)
        vocal = next((f for f in output_files if 'Vocals' in f), None)
        
        return inst, vocal

    # === UPSCALER ===
    def load_upscaler(self):
        if model_manager.get_loaded_model('upscaler'): return model_manager.get_loaded_model('upscaler')
        
        print("Loading RealESRGAN...")
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
        target_filename = "RealESRGAN_x4plus.pth"
        target_path = MODELS_DIR / target_filename
        
        # Check if we already have the flat file
        model_path = target_path
        if not target_path.exists():
            # Download to cache then copy to flat file
            hf_path = hf_hub_download(
                repo_id="lllyasviel/Annotators", 
                filename="RealESRGAN_x4plus.pth",
                cache_dir=MODELS_DIR
            )
            shutil.copy(hf_path, str(target_path))
        
        model_obj = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upscaler = RealESRGANer(
            scale=4,
            model_path=str(model_path), # Ensure string
            model=model_obj,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=DEVICE == 'cuda',
            gpu_id=0 if DEVICE == 'cuda' else None
        )
        
        model_manager.register_model('upscaler', upscaler, model_id='realesrgan')
        return upscaler

    def upscale(self, image_path, scale=4):
        upscaler = self.load_upscaler()
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        output, _ = upscaler.enhance(img, outscale=scale)
        
        output_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_upscaled.png"
        cv2.imwrite(str(OUTPUT_FOLDER / output_name), output)
        return f"/download/{output_name}"

    # === BG REMOVER ===
    def load_bg_remover(self):
        if model_manager.get_loaded_model('bg_remover'): return model_manager.get_loaded_model('bg_remover')
        
        print("Loading U2Net...")
        from rembg import new_session
        
        # Smart Provider Selection
        # Try GPU if available, otherwise CPU. 
        # Crucially, catch GPU errors (missing DLLs) and fallback to CPU.
        providers = ['CPUExecutionProvider']
        if DEVICE == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
        try:
             print(f"Attempting U2Net with providers: {providers}")
             session = new_session("u2net_human_seg", providers=providers)
        except Exception as e:
             print(f"ONNX Runtime Error (likely GPU), falling back to CPU: {e}")
             session = new_session("u2net_human_seg", providers=['CPUExecutionProvider'])
             
        model_manager.register_model('bg_remover', session, model_id='u2net')
        return session

    def remove_bg(self, image_path):
        session = self.load_bg_remover()
        from rembg import remove
        
        img = Image.open(image_path).convert("RGB")
        output = remove(img, session=session)
        
        output_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_nobg.png"
        output.save(str(OUTPUT_FOLDER / output_name), 'PNG')
        return f"/download/{output_name}"

    def delete_model(self, model_id):
        """Delete tool models from disk"""
        try:
            target_file = None
            if 'realesrgan' in model_id:
                target_file = MODELS_DIR / "RealESRGAN_x4plus.pth"
            elif 'u2net' in model_id:
                # U2Net is now forced to MODELS_DIR/u2net via env var
                target_file = MODELS_DIR / 'u2net' / 'u2net_human_seg.onnx'
            elif 'roformer' in model_id or 'vocal' in model_id:
                target_file = MODELS_DIR / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
            elif '3d' in model_id:
                # 3D model (Hunyuan3D)
                # Use standard repo deletion logic via config
                config = model_manager.get_model_config(model_id)
                if config and 'hf_name' in config:
                     repo_folder = f"models--{config['hf_name'].replace('/', '--')}"
                     target_file = MODELS_DIR / repo_folder
                else:
                     # Fallback
                     target_file = MODELS_DIR / "models--tencent--Hunyuan3D-2" 
                
            if target_file and target_file.exists():
                if target_file.is_dir():
                    import shutil
                    shutil.rmtree(target_file, ignore_errors=True)
                else:
                    os.remove(target_file)
                print(f"Deleted {target_file}")
                return True
            else:
                 print(f"File not found for deletion: {target_file}")
                 # If file is gone, consider it success
                 return True
        except Exception as e:
            print(f"Error deleting {model_id}: {e}")
            return False

    # === 3D GEN ===
    def load_3d(self):
        if model_manager.get_loaded_model('3d_shape'): 
            return model_manager.get_loaded_model('3d_shape'), model_manager.get_loaded_model('3d_texture')

        print("Loading Hunyuan3D...")
        from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        
        shape_model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=MODELS_DIR
        )
        shape_model.enable_flashvdm()
        shape_model.to(DEVICE)
        
        tex_model = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            cache_dir=MODELS_DIR
        )
        
        model_manager.register_model('3d_shape', shape_model)
        model_manager.loaded_models['3d_texture'] = tex_model
        return shape_model, tex_model

    def generate_3d(self, image_path, output_format='glb', steps=50, texture_res=1024, seed=0):
        shape_model, tex_model = self.load_3d()
        
        pil_img = Image.open(image_path).convert("RGB")
        gen_seed = seed if seed > 0 else 42
        
        mesh_output = shape_model(
            image=pil_img,
            num_inference_steps=steps,
            guidance_scale=5.5,
            octree_resolution=256,
            num_chunks=8000,
            generator=torch.Generator(DEVICE).manual_seed(gen_seed)
        )
        mesh = mesh_output[0]
        
        if tex_model:
            try:
                textured_mesh = tex_model(
                    mesh=mesh,
                    image=pil_img,
                    texture_resolution=texture_res
                )
                mesh = textured_mesh.mesh
            except Exception:
                pass
        
        output_name = f"model_{uuid.uuid4().hex[:8]}.{output_format}"
        mesh.export(str(MODELS_3D_FOLDER / output_name))
        return f"/download-3d/{output_name}"

tools_service = ToolsService()

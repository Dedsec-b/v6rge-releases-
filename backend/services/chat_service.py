import torch
import gc
import traceback
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download

# Local imports
try:
    from config import MODELS_DIR, DEVICE
    from services.model_manager import model_manager
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import MODELS_DIR, DEVICE
    from services.model_manager import model_manager

class ChatService:
    def __init__(self):
        pass

    def load_model(self, model_id='qwen-72b'):
        """Load Chat Model dynamically"""
        # Check if already loaded
        current_model = model_manager.get_loaded_model('chat')
        current_id = model_manager.current_model_ids.get('chat')
        
        if current_model is not None and current_id == model_id:
            return current_model, model_manager.get_loaded_model('chat_tokenizer')

        config = model_manager.get_model_config(model_id)
        if not config:
            # Fallback
            model_id = 'qwen-72b'
            config = model_manager.get_model_config(model_id)

        print(f"Loading Chat Model: {model_id}...")
        
        # Unload previous logic is handled by ModelManager.register_model now, 
        # but we explicitly cleanup tokenizer here
        if model_manager.get_loaded_model('chat_tokenizer'):
             model_manager.loaded_models['chat_tokenizer'] = None

        # === GGUF CHECK ===
        if config.get('filename') and config['filename'].endswith('.gguf'):
            return self._load_gguf(model_id, config)
        
        # === TRANSFORMERS ===
        return self._load_transformers(model_id, config)

    def _load_gguf(self, model_id, config):
        print(f"Loading GGUF Model: {config['filename']} via llama.cpp")
        try:
            import llama_cpp
            
            model_path = hf_hub_download(
                repo_id=config['hf_name'],
                filename=config['filename'],
                cache_dir=MODELS_DIR
            )
            
            n_gpu_layers = -1 if DEVICE == 'cuda' else 0
            
            model = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=4096,
                verbose=False
            )
            model.is_gguf = True
            
            # Register
            model_manager.register_model('chat', model, model_id)
            print(f"Chat Model (GGUF) Ready")
            return model, None
            
        except Exception as e:
            print(f"[ERROR] Failed to load GGUF: {e}")
            raise e

    def _load_transformers(self, model_id, config):
        try:
            # Handle Vision Models
            if config.get('type') == 'vision':
                print(f"Loading Vision Model: {model_id}...")
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                
                # Load Processor
                processor = AutoProcessor.from_pretrained(
                    config['hf_name'],
                    trust_remote_code=True,
                    cache_dir=MODELS_DIR
                )
                
                # Determine device map based on hardware
                # On CPU, 'auto' can cause "offload whole model to disk" error
                device_map = "auto" if DEVICE == 'cuda' else "cpu"
                
                # Load Model
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    config['hf_name'],
                    torch_dtype=torch.float16, # Default to fp16
                    device_map=device_map,
                    trust_remote_code=True,
                    cache_dir=MODELS_DIR,
                    low_cpu_mem_usage=True
                )
                
                # Register
                model_manager.register_model('chat', model, model_id)
                model_manager.loaded_models['chat_tokenizer'] = processor # Store processor as tokenizer (compatible-ish interface?)
                
                print(f"Vision Model Ready")
                return model, processor

            # Standard Chat Models
            tokenizer = AutoTokenizer.from_pretrained(
                config['hf_name'],
                trust_remote_code=True,
                cache_dir=MODELS_DIR
            )
            
            # Device/Dtype logic matches server.py
            kwargs = {
                "torch_dtype": torch.float16, # Defaulting to float16 for all
                "device_map": "auto",
                "trust_remote_code": True,
                "cache_dir": MODELS_DIR,
                "low_cpu_mem_usage": True
            }

            model = AutoModelForCausalLM.from_pretrained(
                config['hf_name'],
                **kwargs
            )
            model.is_gguf = False
            
            # Register
            model_manager.register_model('chat', model, model_id)
            model_manager.loaded_models['chat_tokenizer'] = tokenizer
            
            print(f"Chat Model (Transformers) Ready")
            return model, tokenizer
            
        except Exception as load_err:
            print(f"Standard load failed: {load_err}")
            
            # If it was a Vision model failure, DO NOT use AutoModelForCausalLM fallback
            if config.get('type') == 'vision':
                raise load_err

            print(f"Retrying with fp32 config...")
            # Fallback
            model = AutoModelForCausalLM.from_pretrained(
                config['hf_name'],
                torch_dtype=torch.float32,
                device_map="auto",
                trust_remote_code=True,
                cache_dir=MODELS_DIR,
                low_cpu_mem_usage=True
            )
            model.is_gguf = False
            
            model_manager.register_model('chat', model, model_id)
            model_manager.loaded_models['chat_tokenizer'] = tokenizer
            return model, tokenizer

    def generate(self, messages, images=None, max_tokens=2048, temperature=0.7):
        model = model_manager.get_loaded_model('chat')
        tokenizer = model_manager.get_loaded_model('chat_tokenizer') # This is 'processor' for Vision models
        config = model_manager.get_model_config(model_manager.current_model_ids.get('chat'))

        if not model:
            raise Exception("Model not loaded")

        if getattr(model, 'is_gguf', False):
            # Llama.cpp generation
            safe_messages = [{'role': str(m['role']), 'content': str(m['content'])} for m in messages]
            response_data = model.create_chat_completion(
                messages=safe_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response_data['choices'][0]['message']['content']
            
        elif config and config.get('type') == 'vision':
            # === QWEN2-VL LOGIC ===
            from qwen_vl_utils import process_vision_info
            
            # Construct Qwen-VL specific message format if images present
            # We assume 'messages' coming in is standard [{"role": "user", "content": "text..."}]
            # We need to inject the image into the last user message
            
            vl_messages = []
            for msg in messages:
                if msg['role'] == 'user' and images and msg == messages[-1]:
                    content = []
                    for img_path in images:
                         content.append({"type": "image", "image": img_path})
                    content.append({"type": "text", "text": msg['content']})
                    vl_messages.append({"role": "user", "content": content})
                else:
                    vl_messages.append(msg)

            # Preparation for inference
            text = tokenizer.apply_chat_template(
                vl_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(vl_messages)
            inputs = tokenizer(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(DEVICE)

            # Inference
            generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)
            
            # Trim inputs from output
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return output_text

        else:
            # Standard Transformers generation
            if not tokenizer: raise Exception("Tokenizer not loaded")
            
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            return tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

    def delete_model(self, model_id):
        try:
            # Unload
            if model_manager.get_loaded_model('chat'):
                model_manager.unload_model('chat')
                
            config = model_manager.get_model_config(model_id)
            if config:
                # GGUF Check
                if config.get('filename') and config['filename'].endswith('.gguf'):
                     # GGUF caches differently sometimes, but hf_hub_download typically uses similiar structure
                     # or we just try standard repo delete first
                     pass
                     
                repo_folder = f"models--{config['hf_name'].replace('/', '--')}"
                repo_path = MODELS_DIR / repo_folder
                if repo_path.exists():
                    import shutil
                    shutil.rmtree(repo_path, ignore_errors=True)
                    print(f"Deleted {repo_path}")
            return True
        except Exception as e:
            print(f"Error deleting chat model: {e}")
            return False

chat_service = ChatService()

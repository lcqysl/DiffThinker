import torch
import os
import math
from PIL import Image
from tqdm import tqdm
import torch.multiprocessing as mp
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="2x2", help="Num of Number")
    parser.add_argument("--model", type=str, default="Qwen-Image-Edit")
    return parser.parse_args()

args = parse_args()
LEVEL = args.level
model_name=args.model
if model_name=="Qwen-Image-Edit-2511":
    zero_cond_t=True
else:
    zero_cond_t=False
ROOT_DIR = f"path/to/models/{model_name}"
LORA_MODEL_PATH = f"path/to/models/Gen/{model_name}/Jigsaw/epoch-4.safetensors"
INPUT_DIR = f"path/to/DiffThinker/Jigsaw/{LEVEL}_test"

num_inference_steps = 20
OUTPUT_DIR = os.path.join(INPUT_DIR, "result")

def gpu_worker(gpu_id, file_chunk):
    device_str = f"cuda:{gpu_id}"

    try:
        transformer_path = os.path.join(ROOT_DIR, "transformer", "diffusion_pytorch_model_merged.safetensors")
        text_encoder_path = os.path.join(ROOT_DIR, "text_encoder", "model_merged.safetensors")
        vae_path = os.path.join(ROOT_DIR, "vae", "diffusion_pytorch_model.safetensors")
        processor_path = os.path.join(ROOT_DIR, "processor/")
        tokenizer_path = os.path.join(ROOT_DIR, "tokenizer/")

        pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device_str,
            model_configs=[
                ModelConfig(path=transformer_path),
                ModelConfig(path=text_encoder_path),
                ModelConfig(path=vae_path),
            ],
            processor_config=ModelConfig(path=processor_path),
            tokenizer_config=ModelConfig(path=tokenizer_path),
        )

        pipe.load_lora(pipe.dit, LORA_MODEL_PATH, alpha=1)
        
    except Exception as e:
        print(f"[GPU {gpu_id}]: {e}")
        return

    prompt = "Solve this Jigsaw puzzle."

    for filename in tqdm(file_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
        input_path = os.path.join(INPUT_DIR, filename)
        save_path = os.path.join(OUTPUT_DIR, filename)

        try:
            input_image = Image.open(input_path).convert("RGB")
            input_image = input_image.resize((512, 512), resample=Image.NEAREST)

            generated_image = pipe(
                prompt, 
                edit_image=input_image, 
                seed=42, 
                num_inference_steps=num_inference_steps, 
                height=512, 
                width=512,
                edit_image_auto_resize=False,
                zero_cond_t=zero_cond_t
            )

            generated_image.save(save_path)
            
        except Exception as e:
            print(f"[GPU {gpu_id}]{e}")

    print(f"[GPU {gpu_id}] done")

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    all_files = [
        f for f in os.listdir(INPUT_DIR) 
        if f.lower().endswith(valid_extensions) and "solution" not in f
    ]
    all_files.sort()
    
    total_files = len(all_files)
    if total_files == 0:
        return

    num_gpus = 8

    chunk_size = math.ceil(total_files / num_gpus)
    chunks = [all_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]

    processes = []
    for rank in range(num_gpus):
        if rank < len(chunks):
            p = mp.Process(target=gpu_worker, args=(rank, chunks[rank]))
            p.start()
            processes.append(p)
    
    for p in processes:
        p.join()

    print("done")

if __name__ == "__main__":
    main()
import torch
import os
import math
import re
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch.multiprocessing as mp
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import argparse

BENCH_GAP = 12 
MODEL_GAP = 4   
LABEL_BOX_SIZE = 32
FONT_SIZE_RATIO = 0.8
IMG_SIZE = 512

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
INPUT_DIR = f"path/to/data/DiffThinker/Jigsaw/{LEVEL}_test"

num_inference_steps = 20
OUTPUT_DIR = os.path.join(INPUT_DIR, "result")

def get_grid_config(filename):
    try:
        match = re.search(r'(\d+)', filename)
        if match:
            idx = int(match.group(1))
        else:
            return 2, 2
            
        if 1 <= idx <= 80:
            return 1, 2
        elif 81 <= idx <= 160:
            return 2, 1
        elif 161 <= idx <= 240:
            return 1, 3
        elif 241 <= idx <= 320:
            return 3, 1
        elif 321 <= idx <= 400: 
            return 2, 2
        else:
            return 2, 2
    except:
        return 2, 2

def draw_label(block_img, text):
    draw = ImageDraw.Draw(block_img)
    draw.rectangle([(0, 0), (LABEL_BOX_SIZE, LABEL_BOX_SIZE)], fill="white")
    draw.rectangle([(0, 0), (LABEL_BOX_SIZE, LABEL_BOX_SIZE)], outline="black", width=1)
    
    try:
        font_size = int(LABEL_BOX_SIZE * FONT_SIZE_RATIO)
        font_paths = [
            "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
        ]
        font = None
        for p in font_paths:
            try:
                font = ImageFont.truetype(p, font_size)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    x = (LABEL_BOX_SIZE - text_w) / 2
    y = (LABEL_BOX_SIZE - text_h) / 2
    draw.text((x, y), text, fill="black", font=font)

def stitch_blocks(blocks, rows, cols):
    if not blocks:
        return Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")

    target_block_w = (IMG_SIZE - (cols - 1) * MODEL_GAP) // cols
    target_block_h = (IMG_SIZE - (rows - 1) * MODEL_GAP) // rows
    
    full_img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    
    for idx, block in enumerate(blocks):
        block = block.resize((target_block_w, target_block_h), Image.BICUBIC)
        
        r = idx // cols
        c = idx % cols
        
        x = c * (target_block_w + MODEL_GAP)
        y = r * (target_block_h + MODEL_GAP)
        
        full_img.paste(block, (x, y))
        
    return full_img

def preprocess_benchmark_image(img_path, rows, cols):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        
        blocks = []
        step_w = IMG_SIZE / cols
        step_h = IMG_SIZE / rows
        
        for r in range(rows):
            for c in range(cols):
                cx = (c + 0.5) * step_w
                cy = (r + 0.5) * step_h
                
                crop_w = step_w - BENCH_GAP
                crop_h = step_h - BENCH_GAP
                
                left = cx - crop_w / 2
                top = cy - crop_h / 2
                right = cx + crop_w / 2
                bottom = cy + crop_h / 2
                
                block = im.crop((left, top, right, bottom))
                blocks.append(block)
        
        labeled_blocks = []
        for i, block in enumerate(blocks):
            b = block.copy()
            draw_label(b, str(i + 1))
            labeled_blocks.append(b)
            
        processed_input = stitch_blocks(labeled_blocks, rows, cols)
        return processed_input


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

    prompt = "Solve this jigsaw puzzle."

    for filename in tqdm(file_chunk, desc=f"GPU {gpu_id}", position=gpu_id):
        input_path = os.path.join(INPUT_DIR, filename)
        save_path = os.path.join(OUTPUT_DIR, filename)

        try:
            rows, cols = get_grid_config(filename)
            
            input_image = preprocess_benchmark_image(input_path, rows, cols)
            
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
            print(f"[GPU {gpu_id}] Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()

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
        if f.lower().endswith(valid_extensions) and "solution" not in f and "problem" in f
    ]
    all_files.sort()
    
    total_files = len(all_files)
    if total_files == 0:
        print(f"No problem images found in {INPUT_DIR}")
        return

    print(f"Found {total_files} problem images to process.")

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

    print("All tasks finished.")

if __name__ == "__main__":
    main()
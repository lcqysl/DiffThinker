import os
import json
import argparse
import re
from tqdm import tqdm
from PIL import Image, ImageOps
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

CHECKPOINT_PATH = "path/to/models/Qwen3-VL-8B-Instruct"

TARGET_SIZE = 512
CELL_SIZE = 256
PADDING_RATIO = 0.1

PROMPT_TEXT = "Identify the digit in this image. Output ONLY a single digit (1-9). If the cell is empty or has no digit, output 0."

def crop_cells(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        
        if img.size != (TARGET_SIZE, TARGET_SIZE):
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
            
        step_size = TARGET_SIZE / 9
        padding = step_size * PADDING_RATIO
        
        crops = []
        
        for row in range(9):
            for col in range(9):
                left = col * step_size + padding
                top = row * step_size + padding
                right = (col + 1) * step_size - padding
                bottom = (row + 1) * step_size - padding
                
                cell = img.crop((left, top, right, bottom))
                
                cell = ImageOps.expand(cell, border=10, fill=(255, 255, 255))
                cell = cell.resize((CELL_SIZE, CELL_SIZE), Image.Resampling.BICUBIC)
                
                crops.append(cell)
        
        return crops
    except Exception as e:
        print(f"Error cropping {image_path}: {e}")
        return None

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def main():
    parser = argparse.ArgumentParser(description="MLLM-based Sudoku OCR")
    parser.add_argument("dir", type=str, help="Directory containing PNG images (e.g., 45_test/result)")
    args = parser.parse_args()

    target_dir = args.dir
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    print(f"Loading model from {CHECKPOINT_PATH} ...")
    processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH)
    
    llm = LLM(
        model=CHECKPOINT_PATH,
        mm_encoder_tp_mode="data", 
        tensor_parallel_size=8,
        seed=42,
        gpu_memory_utilization=0.9,
        max_model_len=512,       
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 1}
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=5,     
        top_k=-1,
    )

    print(f"\n{'='*20} Processing Directory: {target_dir} {'='*20}")
    
    img_files = sorted([
        f for f in os.listdir(target_dir) 
        if f.endswith(".png") and "debug" not in f
    ])

    if not img_files:
        print("No png images found in the specified directory.")
        return

    all_inputs = []
    metadata = [] # (filename, cell_index)

    print(f"Found {len(img_files)} images. Preparing crops (Target Size: {CELL_SIZE}x{CELL_SIZE})...")

    for filename in tqdm(img_files, desc="Preprocessing"):
        img_path = os.path.join(target_dir, filename)
        
        crops = crop_cells(img_path)
        if crops is None or len(crops) != 81:
            print(f"Skipping malformed image: {filename}")
            continue
        
        for idx, cell_img in enumerate(crops):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": cell_img},
                        {"type": "text", "text": PROMPT_TEXT},
                    ],
                }
            ]
            
            try:
                processed_input = prepare_inputs_for_vllm(messages, processor)
                all_inputs.append(processed_input)
                metadata.append((filename, idx))
            except Exception as e:
                print(f"Error processing cell input: {e}")

    if not all_inputs:
        print("No valid inputs generated.")
        return

    print(f"Running vLLM inference on {len(all_inputs)} cells...")
    outputs = llm.generate(all_inputs, sampling_params=sampling_params)

    print("Reconstructing Sudoku grids...")
    puzzle_results = {} # {filename: [char]*81}

    for i, output in enumerate(outputs):
        filename, cell_idx = metadata[i]
        
        generated_text = output.outputs[0].text.strip()
        
        match = re.search(r'[0-9]', generated_text)
        digit = match.group(0) if match else '0'
        
        if filename not in puzzle_results:
            puzzle_results[filename] = ['0'] * 81
        
        puzzle_results[filename][cell_idx] = digit

    final_json_output = {}
    for filename, digits_list in puzzle_results.items():
        stem = os.path.splitext(filename)[0]
        solution_str = "".join(digits_list)
        final_json_output[stem] = solution_str

    output_path = os.path.join(target_dir, "0_result.json")
    print(f"Saving results to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_output, f, indent=4, ensure_ascii=False)

    print("\n" + ">" * 40)
    print("Done.")

if __name__ == '__main__':
    main()
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

LABEL_CROP_RATIO = 0.25 

PROMPT_TEXT = "Identify the number in this image fragment. Output ONLY a single number. If no number is visible, output 0."

def crop_cells(image_path, grid_rows, grid_cols):
    try:
        img = Image.open(image_path).convert('RGB')
        
        if img.size != (TARGET_SIZE, TARGET_SIZE):
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
            
        step_x = TARGET_SIZE / grid_cols
        step_y = TARGET_SIZE / grid_rows
        
        crop_w = step_x * LABEL_CROP_RATIO
        crop_h = step_y * LABEL_CROP_RATIO
        
        crops = []
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                base_x = col * step_x
                base_y = row * step_y
                
                left = base_x
                top = base_y
                right = base_x + crop_w
                bottom = base_y + crop_h
                
                cell = img.crop((left, top, right, bottom))
                
                cell = ImageOps.expand(cell, border=2, fill=(255, 255, 255))
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
    parser = argparse.ArgumentParser(description="MLLM-based Jigsaw Puzzle Label Parser")
    parser.add_argument("dir", type=str, help="Directory containing PNG images")
    parser.add_argument("--grid", type=str, default="2x2", help="Grid size of the puzzles (e.g., 2x2, 3x3)")
    args = parser.parse_args()

    target_dir = args.dir
    if not os.path.exists(target_dir):
        print(f"Error: Directory '{target_dir}' does not exist.")
        return

    try:
        grid_rows, grid_cols = map(int, args.grid.lower().split('x'))
        total_cells = grid_rows * grid_cols
    except ValueError:
        print("Error: Invalid grid format. Use '2x2', '3x3', etc.")
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

    print(f"\n{'='*20} Processing Directory: {target_dir} ({args.grid}) {'='*20}")
    
    img_files = sorted([
        f for f in os.listdir(target_dir) 
        if f.endswith(".png") and "debug" not in f and "solution" not in f
    ])

    if not img_files:
        print("No png images found in the specified directory.")
        return

    all_inputs = []
    metadata = [] # (filename, cell_index)

    print(f"Found {len(img_files)} images. Preparing label crops...")

    for filename in tqdm(img_files, desc="Preprocessing"):
        img_path = os.path.join(target_dir, filename)
        
        crops = crop_cells(img_path, grid_rows, grid_cols)
        
        if crops is None or len(crops) != total_cells:
            print(f"Skipping malformed image (crop mismatch): {filename}")
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

    print("Reconstructing Puzzle sequences...")
    puzzle_results = {} # {filename: [digit_str]*total_cells}

    for i, output in enumerate(outputs):
        filename, cell_idx = metadata[i]
        
        generated_text = output.outputs[0].text.strip()
        
        match = re.search(r'\d+', generated_text)
        digit = match.group(0) if match else '0'
        
        if filename not in puzzle_results:
            puzzle_results[filename] = ['0'] * total_cells
        
        puzzle_results[filename][cell_idx] = digit

    final_json_output = {}
    for filename, digits_list in puzzle_results.items():
        stem = os.path.splitext(filename)[0]
        solution_str = " ".join(digits_list)
        final_json_output[stem] = solution_str

    output_path = os.path.join(target_dir, "0_result.json")
    print(f"Saving results to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json_output, f, indent=4, ensure_ascii=False)

    print("\n" + ">" * 40)
    print("Done.")

if __name__ == '__main__':
    main()
import os
import re
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

PIXEL_THRESHOLD = 0.02
TARGET_LEVELS = [3,4,5,6,7,8,16,32]

def parse_map_info(txt_path):
    if not os.path.exists(txt_path):
        return None, None, None

    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    grid_rows = []
    start_pos = None

    for line in lines:
        line = line.strip()
        if not line or 'Col' in line or '---' in line:
            continue
        
        parts = [p.strip() for p in line.split('|')]
        clean_parts = [p for p in parts if p]
        
        if len(clean_parts) < 2:
            continue
            
        row_chars = clean_parts[1:]
        
        current_row_idx = len(grid_rows)
        for c_idx, char in enumerate(row_chars):
            if char == '@':
                start_pos = (current_row_idx, c_idx)
        
        grid_rows.append(row_chars)

    if not grid_rows:
        return None, None, None

    rows = len(grid_rows)
    cols = len(grid_rows[0])
    
    return rows, cols, start_pos

def extract_actions_from_image(img_path, rows, cols, start_pos):
    if not os.path.exists(img_path):
        return []

    img = Image.open(img_path).convert("RGB")
    img = img.resize((512, 512), resample=Image.BILINEAR)
    pixels = np.array(img)

    r_ch = pixels[:, :, 0].astype(float)
    g_ch = pixels[:, :, 1].astype(float)
    b_ch = pixels[:, :, 2].astype(float)
    
    red_mask = (r_ch > 100) & (r_ch > g_ch * 1.2) & (r_ch > b_ch * 1.2)

    cell_h = 512 // rows
    cell_w = 512 // cols
    
    path_grid = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            sub_mask = red_mask[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
            if np.mean(sub_mask) > PIXEL_THRESHOLD: 
                path_grid[r, c] = True

    actions = []
    visited = set()
    visited.add(start_pos)
    
    curr_r, curr_c = start_pos
    
    max_steps = rows * cols * 2
    step_count = 0
    
    while step_count < max_steps:
        step_count += 1
        found_next = False
        
        directions = [
            (0, 1, 'R'), 
            (1, 0, 'D'), 
            (0, -1, 'L'), 
            (-1, 0, 'U')
        ]
        
        for dr, dc, act_str in directions:
            nr, nc = curr_r + dr, curr_c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                if path_grid[nr, nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    actions.append(act_str)
                    curr_r, curr_c = nr, nc
                    found_next = True
                    break 
        
        if not found_next:
            break 

    return actions

def process_single_level(level):
    if level >= 16:
        BASE_DIR = f"path/to/data/DiffThinker/FrozenLake/{level}_test"
        RESULT_DIR = f"{BASE_DIR}/result"
        TABLE_DIR = f"{BASE_DIR}/table"
    else:
        RESULT_DIR = f"path/to/data/DiffThinker/FrozenLake/VSP/maps/level{level}/img/result"
        TABLE_DIR = f"path/to/data/DiffThinker/FrozenLake/VSP/maps/level{level}/table"

    print(f"\n{'='*40}")
    print(f"Processing Level: {level}")
    print(f"Table Directory: {TABLE_DIR}")
    print(f"Result Directory: {RESULT_DIR}")
    print(f"{'='*40}")

    if not os.path.exists(RESULT_DIR):
        print(f"Error: Result directory not found: {RESULT_DIR}")
        return

    if not os.path.exists(TABLE_DIR):
        print(f"Error: Table directory not found: {TABLE_DIR}")
        return

    txt_files = [f for f in os.listdir(TABLE_DIR) if f.endswith('.txt')]
    txt_files.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

    results_data = {}
    missing_count = 0
    
    for txt_file in tqdm(txt_files, desc=f"Level {level}"):
        try:
            file_id = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(TABLE_DIR, txt_file)
            
            rows, cols, start_pos = parse_map_info(txt_path)
            
            if rows is None or start_pos is None:
                continue

            img_filename = f"{file_id}.png"
            img_path = os.path.join(RESULT_DIR, img_filename)
            
            if not os.path.exists(img_path):
                missing_count += 1
                continue

            action_sequence_list = extract_actions_from_image(img_path, rows, cols, start_pos)
            action_sequence_str = "".join(action_sequence_list)
            
            results_data[img_filename] = action_sequence_str
            
        except Exception as e:
            print(f"Error processing {txt_file}: {e}")
            continue

    print(f"Level {level} Complete.")
    print(f"Total Processed: {len(results_data)}")
    print(f"Missing Images : {missing_count}")

    json_output_path = os.path.join(RESULT_DIR, "0_result.json")
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=4)
        print(f"Results saved to: {json_output_path}")
    except Exception as e:
        print(f"Failed to save JSON: {e}")

def main():
    print(f"Starting batch processing for levels: {TARGET_LEVELS}")
    print(f"Pixel Threshold: {PIXEL_THRESHOLD}")
    
    for level in TARGET_LEVELS:
        process_single_level(level)

    print("\nAll levels processed.")

if __name__ == "__main__":
    main()
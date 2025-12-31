import os
import re
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=16, help="Level of the maze")
    return parser.parse_args()

args = parse_args()
LEVEL = args.level
BASE_DIR = f"path/to/DiffThinker/Maze/{LEVEL}_test"
TABLE_DIR = BASE_DIR
RESULT_DIR = os.path.join(BASE_DIR, "result")

WALL_MASKS = {
    'N': 1,
    'S': 2,
    'W': 4,
    'E': 8
}

MOVES = {
    'U': (-1, 0, 'N'),
    'D': (1, 0, 'S'),
    'L': (0, -1, 'W'),
    'R': (0, 1, 'E'),
}

def parse_maze_txt(txt_path):
    if not os.path.exists(txt_path):
        return None

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        n = int(lines[0])
        start_r, start_c = map(int, lines[1].split())
        end_r, end_c = map(int, lines[2].split())

        grid_data = []
        for r in range(n):
            if 3 + r >= len(lines):
                break
            row_values = list(map(int, lines[3 + r].split()))
            grid_data.append(row_values)

        return {
            "size": n,
            "start": (start_r, start_c),
            "end": (end_r, end_c),
            "grid": grid_data
        }
    except Exception as e:
        print(f"Error parsing {txt_path}: {e}")
        return None

def extract_actions_smart(img_path, maze_info):
    if not os.path.exists(img_path):
        return []

    rows = maze_info['size']
    cols = maze_info['size']
    start_pos = maze_info['start']
    grid_walls = maze_info['grid']

    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((512, 512), resample=Image.BILINEAR)
        pixels = np.array(img)
    except Exception:
        return []

    r_ch = pixels[:, :, 0].astype(float)
    g_ch = pixels[:, :, 1].astype(float)
    b_ch = pixels[:, :, 2].astype(float)
    
    red_mask = (r_ch > 100) & (r_ch > g_ch * 1.2) & (r_ch > b_ch * 1.2)

    cell_h = 512 // rows
    cell_w = 512 // cols
    
    path_grid = np.zeros((rows, cols), dtype=bool)
    
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
            x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
            
            margin = 2 
            if cell_h > 10: margin = int(cell_h * 0.2)
            
            sub_mask = red_mask[y0+margin:y1-margin, x0+margin:x1-margin]
            
            if sub_mask.size > 0 and np.mean(sub_mask) > 0.1:
                path_grid[r, c] = True

    actions = []
    visited = set()
    visited.add(start_pos)
    
    curr_r, curr_c = start_pos
    
    max_steps = rows * cols * 2 
    step_count = 0
    
    directions = [
        ('R', MOVES['R']), 
        ('D', MOVES['D']), 
        ('L', MOVES['L']), 
        ('U', MOVES['U'])
    ]
    
    while step_count < max_steps:
        step_count += 1
        found_next = False
        
        current_wall_val = grid_walls[curr_r][curr_c]
        
        for act_str, (dr, dc, wall_dir_char) in directions:
            nr, nc = curr_r + dr, curr_c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                
                wall_bit = WALL_MASKS[wall_dir_char]
                has_wall = (current_wall_val & wall_bit) != 0
                
                if has_wall:
                    continue
                
                if path_grid[nr, nc] and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    actions.append(act_str)
                    curr_r, curr_c = nr, nc
                    found_next = True
                    break 
        
        if not found_next:
            break 

    return actions

def main():
    if not os.path.exists(RESULT_DIR):
        print(f"Error: Result directory not found: {RESULT_DIR}")
        return

    txt_files = [f for f in os.listdir(TABLE_DIR) if f.endswith('.txt')]
    txt_files.sort(key=lambda x: [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', x)])

    total_count = 0
    missing_count = 0
    
    results_data = {}

    for txt_file in tqdm(txt_files):
        total_count += 1
        file_id = txt_file.replace('.txt', '')
        
        txt_path = os.path.join(TABLE_DIR, txt_file)
        maze_info = parse_maze_txt(txt_path)
        
        if not maze_info:
            continue

        img_filename = f"{file_id}.png"
        img_path = os.path.join(RESULT_DIR, img_filename)
        
        if not os.path.exists(img_path):
            missing_count += 1
            continue

        actions_list = extract_actions_smart(img_path, maze_info)
        action_str = "".join(actions_list)
        
        results_data[img_filename] = action_str
        
    processed_count = total_count - missing_count
    
    print("\n" + "="*30)
    print("Path Extraction Summary")
    print("="*30)
    print(f"Total Tasks Found: {total_count}")
    print(f"Missing Images   : {missing_count}")
    print(f"Processed Tasks  : {processed_count}")
    print("="*30)

    output_json_path = os.path.join(RESULT_DIR, "0_result.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4)
    
    print(f"All extracted paths have been saved to: {output_json_path}")


if __name__ == "__main__":
    main()
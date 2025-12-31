import os
import json
import argparse
import gymnasium as gym
from tqdm import tqdm

def parse_map_to_gym_desc(txt_path):
    if not os.path.exists(txt_path):
        return None, None

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
        
        gym_row = ""
        for c_idx, char in enumerate(row_chars):
            if char == '@':
                gym_row += "S"
                start_pos = (len(grid_rows), c_idx)
            elif char == '*':
                gym_row += "G"
            elif char == '#':
                gym_row += "H"
            elif char == '_':
                gym_row += "F"
            else:
                gym_row += "F"
        
        grid_rows.append(gym_row)

    return grid_rows, start_pos

def evaluate(table_dir, json_path):
    if not os.path.exists(table_dir):
        print(f"Error: Table directory not found: {table_dir}")
        return
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return

    print(f"Loading predictions from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    total_count = 0
    correct_count = 0
    missing_map_count = 0
    
    action_number_map = {'L': 0, 'D': 1, 'R': 2, 'U': 3}
    
    correct_paths_details = []
    for img_filename, action_sequence_str in tqdm(predictions.items()):
        keyword = "Action plan"
    
        if keyword in action_sequence_str:
            action_sequence_str = action_sequence_str.rsplit(keyword, 1)[-1]
            
        total_count += 1
        
        base_name = os.path.splitext(img_filename)[0]
        txt_filename = base_name + ".txt"
        txt_path = os.path.join(table_dir, txt_filename)
        
        gym_desc, start_pos = parse_map_to_gym_desc(txt_path)
        
        if not gym_desc or not start_pos:
            missing_map_count += 1
            continue

        rows = len(gym_desc)
        cols = len(gym_desc[0])

        try:
            env = gym.make('FrozenLake-v1', desc=gym_desc, map_name=f"{rows}x{cols}", is_slippery=False, render_mode=None)
            env.reset(seed=42)
            
            success = False
            for act_char in action_sequence_str:
                if act_char not in action_number_map:
                    continue
                    
                action = action_number_map[act_char]
                observation, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    if reward > 0:
                        success = True
                    break
            
            env.close()

            if success:
                correct_count += 1
                correct_paths_details.append({
                    "filename": img_filename,
                    "length": len(action_sequence_str),
                    "path": action_sequence_str
                })
            else:
                print(img_filename)
                pass
            
        except Exception as e:
            print(f"Error evaluating {img_filename}: {e}")

    valid_total = total_count - missing_map_count
    
    print("\n" + "="*30)
    print("="*30)
    print(f"Total Inputs : {total_count}")
    print(f"Missing Maps : {missing_map_count}")
    print(f"Valid Total  : {valid_total}")
    print(f"Total Correct: {correct_count}")
    
    accuracy = (correct_count / valid_total * 100) if valid_total > 0 else 0.0
    print(f"Accuracy     : {accuracy:.2f}%")
    print("="*30)

    if correct_paths_details:
        correct_paths_details.sort(key=lambda x: x['length'], reverse=True)
        
        print("\n" + "*"*30)
        print("🏆 Top 3 Longest Correct Paths 🏆")
        print("*"*30)
        
        top_k = correct_paths_details[:3]
        
        for i, item in enumerate(top_k):
            print(f"Rank {i+1}:")
            print(f"  File  : {item['filename']}")
            print(f"  Length: {item['length']}")
            print(f"  Path  : {item['path']}")
            print("-" * 20)
    else:
        print("\nNo correct paths found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FrozenLake UDRL predictions from JSON.")
    parser.add_argument("--table_dir", type=str, required=True, help="Path to the directory containing .txt map files")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the JSON file containing predictions")
    
    args = parser.parse_args()
    
    evaluate(args.table_dir, args.json_path)
import os
import warnings
import argparse
import random
import gymnasium as gym
import networkx as nx
from PIL import Image, ImageDraw
import multiprocessing
from multiprocessing import Pool, Manager
from tqdm import tqdm

os.environ["SDL_AUDIODRIVER"] = "dummy"
warnings.filterwarnings("ignore", category=UserWarning, module="pygame")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--num", type=int, default=10000)
    parser.add_argument("--out", type=str, default="output")
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--min_len", type=int, default=1)
    parser.add_argument("--workers", type=int, default=64)
    
    default_prev_dirs = [
        "path/to/data/DiffThinker/FrozenLake/VSP/maps/level3/table",
        "path/to/data/DiffThinker/FrozenLake/VSP/maps/level4/table",
        "path/to/data/DiffThinker/FrozenLake/VSP/maps/level5/table",
        "path/to/data/DiffThinker/FrozenLake/VSP/maps/level6/table",
        "path/to/data/DiffThinker/FrozenLake/VSP/maps/level7/table",
        "path/to/data/DiffThinker/FrozenLake/VSP/maps/level8/table",
        "path/to/data/DiffThinker/FrozenLake/data/table",
        "path/to/data/DiffThinker/FrozenLake/16/table",
        "path/to/data/DiffThinker/FrozenLake/16_test/table",
        "path/to/data/DiffThinker/FrozenLake/32/table",
        "path/to/data/DiffThinker/FrozenLake/32_test/table",
    ]
    parser.add_argument("--prev_dirs", nargs='+', default=default_prev_dirs)
    
    return parser.parse_args()

def generate_random_layout(size, p=0.8):
    map_grid = [['' for _ in range(size)] for _ in range(size)]
    all_coords = [(r, c) for r in range(size) for c in range(size)]
    start_pos, goal_pos = random.sample(all_coords, 2)
    
    for r in range(size):
        for c in range(size):
            if (r, c) == start_pos:
                map_grid[r][c] = 'S'
            elif (r, c) == goal_pos:
                map_grid[r][c] = 'G'
            else:
                map_grid[r][c] = 'F' if random.random() < p else 'H'
    return ["".join(row) for row in map_grid]

def get_shortest_path(desc):
    rows = len(desc)
    cols = len(desc[0])
    G = nx.Graph()
    start_pos = None
    goal_pos = None

    for r in range(rows):
        for c in range(cols):
            char = desc[r][c]
            if char == 'S': start_pos = (r, c)
            if char == 'G': goal_pos = (r, c)
            if char == 'H': continue
            
            G.add_node((r, c))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if desc[nr][nc] != 'H':
                        G.add_edge((r, c), (nr, nc))
    try:
        if start_pos and goal_pos:
            return nx.shortest_path(G, source=start_pos, target=goal_pos)
    except Exception:
        return None
    return None

def generate_table_content(desc, size):
    """@=Start, *=Goal, #=Hole, _=Ice"""
    char_map = {'S': '@', 'G': '*', 'H': '#', 'F': '_'}
    lines = []
    header = "| | " + " | ".join([f"Col {i+1}" for i in range(size)]) + " |"
    lines.append(header)
    
    for r in range(size):
        row_str = desc[r]
        mapped_chars = [char_map[c] for c in row_str]
        line = f"| Row {r+1} | " + " | ".join(mapped_chars) + " |"
        lines.append(line)
    return "\n".join(lines)

def draw_solution_line(image, path, grid_size):
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    cell_w = img_w / grid_size
    cell_h = img_h / grid_size
    pixel_points = []
    for r, c in path:
        center_x = (c * cell_w) + (cell_w / 2)
        center_y = (r * cell_h) + (cell_h / 2)
        pixel_points.append((center_x, center_y))
    
    line_width = int (cell_w / 4)
    draw.line(pixel_points, fill="red", width=line_width, joint="curve")
    return image

def parse_table_file(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        rev_map = {'@': 'S', '*': 'G', '#': 'H', '_': 'F'}
        grid_rows = []
        for line in lines:
            line = line.strip()
            if not line.startswith('|') or "Col" in line: continue
            parts = [p.strip() for p in line.split('|')]
            data_parts = parts[2:] 
            row_str = "".join([rev_map.get(char, '') for char in data_parts])
            if row_str: grid_rows.append(row_str)
        return "".join(grid_rows)
    except Exception:
        return None

def load_existing_layouts(dirs):
    seen = set()
    count = 0
    for d in dirs:
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if f.endswith('.txt'):
                layout_str = parse_table_file(os.path.join(d, f))
                if layout_str:
                    seen.add(layout_str)
                    count += 1
    return seen

def process_one_puzzle(args_tuple):
    (index, size, p, out_dir, min_len, shared_seen, shared_lock) = args_tuple
    random.seed() 
    TARGET_SIZE = (512, 512)
    table_dir = os.path.join(out_dir, "table")
    
    while True:
        desc_list = generate_random_layout(size, p)
        path = get_shortest_path(desc_list)
        
        if path is not None and (len(path) - 1) >= min_len:
            layout_flat = "".join(desc_list)
            is_duplicate = False
            
            with shared_lock:
                if layout_flat in shared_seen:
                    is_duplicate = True
                else:
                    shared_seen[layout_flat] = True
            
            if not is_duplicate:
                break

    file_base = f"{index+1:03d}"
    filename_core = f"{size}_{min_len}_{p}_{file_base}"
    
    try:
        env = gym.make("FrozenLake-v1", desc=desc_list, is_slippery=False, render_mode="rgb_array")
        env.reset()
        rgb_array = env.render()
        env.close()

        original_img = Image.fromarray(rgb_array).resize(TARGET_SIZE, Image.NEAREST)
        original_img.save(os.path.join(out_dir, f"{filename_core}.png"))
        
        solved_img = original_img.copy()
        draw_solution_line(solved_img, path, size).save(os.path.join(out_dir, f"{filename_core}_solution.png"))
        
        with open(os.path.join(table_dir, f"{filename_core}.txt"), "w") as f:
            f.write(generate_table_content(desc_list, size))
            
        return {"status": "success", "msg": f"Done: {file_base}"}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

def main():
    args = parse_args()
    if not os.path.exists(args.out): os.makedirs(args.out)
    table_dir = os.path.join(args.out, "table")
    if not os.path.exists(table_dir): os.makedirs(table_dir)

    manager = Manager()
    shared_seen_layouts = manager.dict()
    shared_lock = manager.Lock()
    
    initial_seen = load_existing_layouts(args.prev_dirs)
    for s in initial_seen: 
        shared_seen_layouts[s] = True

    tasks = [(i, args.size, args.p, args.out, args.min_len, shared_seen_layouts, shared_lock) for i in range(args.num)]
    
    with Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(process_one_puzzle, tasks)
        iterator = tqdm(results, total=args.num)
        for res in iterator:
            if res["status"] != "success":
                print(f"Error: {res['msg']}")


if __name__ == "__main__":
    main()
import os
import argparse
import json
import networkx as nx
from multiprocessing import Pool, cpu_count

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()

def parse_table_file(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        rev_map = {'@': 'S', '*': 'G', '#': 'H', '_': 'F'}
        grid_rows = []
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('|') or "Col" in line:
                continue
            
            parts = [p.strip() for p in line.split('|')]
            data_parts = parts[2:] 
            
            row_str = ""
            for char in data_parts:
                if char in rev_map:
                    row_str += rev_map[char]
            
            if row_str:
                grid_rows.append(row_str)
        
        return grid_rows
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None

def get_shortest_path_udrl(desc):
    if not desc: return None
    
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
        path = nx.shortest_path(G, source=start_pos, target=goal_pos)
    except Exception:
        return None

    if not path or len(path) < 2:
        return ""
    
    dirs = []
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        if r2 < r1: dirs.append("U")
        elif r2 > r1: dirs.append("D")
        elif c2 < c1: dirs.append("L")
        elif c2 > c1: dirs.append("R")
    
    return "".join(dirs)

def process_file(file_info):
    table_path, filename_base = file_info
    
    grid = parse_table_file(table_path)
    
    path_str = get_shortest_path_udrl(grid)
    
    if path_str is None:
        return None
    
    image_name = f"{filename_base}.png"
    return (image_name, path_str)

def main():
    args = parse_args()
    
    table_dir = os.path.join(args.dir, "table")
    if not os.path.exists(table_dir):
        return

    files = [f for f in os.listdir(table_dir) if f.endswith(".txt")]

    tasks = []
    for f in files:
        full_path = os.path.join(table_dir, f)
        base_name = os.path.splitext(f)[0]
        tasks.append((full_path, base_name))

    metadata_dict = {}
    with Pool(processes=args.workers) as pool:
        results = pool.imap_unordered(process_file, tasks)
        
        iterator = tqdm(results, total=len(tasks)) if HAS_TQDM else results
        
        for res in iterator:
            if res:
                key, val = res
                metadata_dict[key] = val

    json_path = os.path.join(args.dir, "metadata.json")
    
    with open(json_path, "w") as f:
        json.dump(metadata_dict, f, indent=4)

if __name__ == "__main__":
    main()
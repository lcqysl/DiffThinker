import argparse
import random
import json
import os
import multiprocessing
import math
import itertools
from pathlib import Path
from python_tsp.exact import solve_tsp_dynamic_programming
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def solve_tsp_optimal(points):
    # Held-Karp DP
    num_points = len(points)
    
    dist_matrix = np.zeros((num_points, num_points))
    
    for i in range(num_points):
        for j in range(num_points):
            dist = math.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
            dist_matrix[i][j] = dist
            
    permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
    
    optimal_path = [points[i] for i in permutation]
    
    optimal_path.append(optimal_path[0])
    
    return optimal_path

def generate_one_tsp_data(grid_size, num_points):
    valid_range = range(0, grid_size)
    candidates = [(x, y) for x in valid_range for y in valid_range]
    
    if len(candidates) < num_points:
        raise ValueError(f"{grid_size}x{grid_size} too small")

    points = random.sample(candidates, num_points)
    
    solution_path = solve_tsp_optimal(points)
    
    return points, solution_path

def render_tsp(grid_size, points, path=None, size_px=512):
    img = Image.new("RGB", (size_px, size_px), "white")
    draw = ImageDraw.Draw(img)
    
    cell_size = size_px / grid_size
    
    def to_px(p):
        return (p[0] * cell_size + cell_size / 2, 
                p[1] * cell_size + cell_size / 2)

    grid_color = (224, 224, 224) 
    line_width = max(1, int(size_px / grid_size / 20))
    
    for i in range(1, grid_size):
        pos = i * cell_size
        draw.line([(pos, 0), (pos, size_px)], fill=grid_color, width=line_width)
        draw.line([(0, pos), (size_px, pos)], fill=grid_color, width=line_width)

    border_width = line_width * 2
    draw.rectangle([(0, 0), (size_px - 1, size_px - 1)], outline="black", width=border_width)
    
    radius = max(3, int(cell_size * 0.3))
    start_node = points[0]
    
    for p in points:
        px, py = to_px(p)
        
        if p == start_node:
            color = "yellow"
        else:
            color = "blue"
        
        draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=color)
        
    if path:
        px_path = [to_px(p) for p in path]
        path_width = int(cell_size * 0.15)
        draw.line(px_path, fill="red", width=path_width, joint="curve")



    return img

def save_tsp_to_text(grid_size, points, filepath):
    with open(filepath, 'w') as f:
        f.write(f"{grid_size} {len(points)}\n")
        for p in points:
            f.write(f"{p[0]} {p[1]}\n")

def path_to_string(path):
    return ",".join([f"({p[0]},{p[1]})" for p in path])

def process_one_tsp(args_tuple):
    i, size, num_points, out_dir = args_tuple
    
    points, path = generate_one_tsp_data(size, num_points)
    
    f_base = f"{size}_{num_points}_{i:03d}"
    f_img_base = out_dir / f"{f_base}.png"
    f_img_sol  = out_dir / f"{f_base}_solution.png"
    f_txt      = out_dir / f"{f_base}.txt"
    
    img_base = render_tsp(size, points, path=None)
    img_base.save(f_img_base)
    
    img_sol = render_tsp(size, points, path=path)
    img_sol.save(f_img_sol)
    
    save_tsp_to_text(size, points, f_txt)
    path_str = path_to_string(path)
    
    return (f_base, path_str)


def main():
    p = argparse.ArgumentParser(description="Generate Grid TSP images (White BG, Black Border).")
    p.add_argument("--size", type=int, required=True, help="Grid size (e.g., 8).")
    p.add_argument("--num", type=int, default=1, help="Total number of samples to generate.")
    p.add_argument("--points", type=int, default=5, help="Fixed number of cities/points.")
    p.add_argument("--out", type=str, required=True, help="Output directory.")
    p.add_argument("--max_workers", type=int, default=64, help="Max worker processes.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    

    num_workers = min(args.max_workers, 64, args.num)
    print(f"Generating {args.num} TSP samples...")
    print(f"Config: {args.size}x{args.size}, {args.points} Points. Style: White BG, Black Border, No Dot Outline.")
    
    tasks = [(i, args.size, args.points, out) for i in range(1, args.num + 1)]
    all_solutions = {}
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_tsp, tasks), total=args.num))
        
    for f_base, sol_str in results:
        all_solutions[f_base] = sol_str
        
    path_json_file = out / "path.json"
    
    if not os.path.exists(path_json_file) or os.path.getsize(path_json_file) == 0:
        base_json = {}
    else:
        try:
            with open(path_json_file, "r", encoding="utf-8") as f:
                base_json = json.load(f)
        except:
            base_json = {}

    base_json.update(dict(sorted(all_solutions.items())))

    with open(path_json_file, "w", encoding="utf-8") as f:
        json.dump(base_json, f, indent=4, ensure_ascii=False)
        
    print(f"Done. Saved to {out.resolve()}")

if __name__ == "__main__":
    main()
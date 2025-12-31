import argparse
import random
import json
import os
import multiprocessing
from pathlib import Path
from collections import deque

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def make_empty_grid(n):
    return [[{"N": True, "E": True, "S": True, "W": True, "visited": False}
             for _ in range(n)] for _ in range(n)]

def neighbors_of(r, c, n):
    res = []
    if r > 0:     res.append(("N", r-1, c))
    if c < n-1:   res.append(("E", r, c+1))
    if r < n-1:   res.append(("S", r+1, c))
    if c > 0:     res.append(("W", r, c-1))
    return res

def remove_wall(grid, r, c, d):
    if d == "N": grid[r][c]["N"], grid[r-1][c]["S"] = False, False
    elif d == "S": grid[r][c]["S"], grid[r+1][c]["N"] = False, False
    elif d == "E": grid[r][c]["E"], grid[r][c+1]["W"] = False, False
    elif d == "W": grid[r][c]["W"], grid[r][c-1]["E"] = False, False

def gen_maze_dfs(n, rng=random):
    grid = make_empty_grid(n)
    stack = []
    sr, sc = rng.randrange(n), rng.randrange(n)
    grid[sr][sc]["visited"] = True
    stack.append((sr, sc))
    while stack:
        r, c = stack[-1]
        neighbors = [(d, nr, nc) for (d, nr, nc) in neighbors_of(r, c, n)
                     if not grid[nr][nc]["visited"]]
        if neighbors:
            d, nr, nc = rng.choice(neighbors)
            remove_wall(grid, r, c, d)
            grid[nr][nc]["visited"] = True
            stack.append((nr, nc))
        else:
            stack.pop()
    for row in grid:
        for cell in row:
            cell.pop("visited", None)
    return grid

def shortest_path_bfs(grid, start, end):
    n = len(grid)
    q = deque([(start, [start])])
    visited = {start}
    while q:
        (r, c), path = q.popleft()
        if (r, c) == end:
            return np.array(path, dtype=int)
        cell = grid[r][c]
        for dr, dc, wall_self, wall_other, nr_cell_wall in [
            (-1, 0, "N", "S", (r-1, c)), (1, 0, "S", "N", (r+1, c)),
            (0, -1, "W", "E", (r, c-1)), (0, 1, "E", "W", (r, c+1))]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not cell[wall_self] and (nr, nc) not in visited:
                visited.add((nr, nc))
                new_path = list(path)
                new_path.append((nr, nc))
                q.append(((nr, nc), new_path))
    raise ValueError("No path exists")

def convert_path_to_udrl(path):
    if len(path) < 2: return ""
    moves = []
    for i in range(len(path) - 1):
        r1, c1 = path[i]; r2, c2 = path[i+1]
        if r2 < r1: moves.append('U')
        elif r2 > r1: moves.append('D')
        elif c2 < c1: moves.append('L')
        elif c2 > c1: moves.append('R')
    return "".join(moves)

def generate_one_maze_data(size, min_len, max_tries=200):
    rng = random.Random()
    for _ in range(max_tries):
        grid = gen_maze_dfs(size, rng)
        nodes = [(r, c) for r in range(size) for c in range(size)]
        start, end = rng.sample(nodes, 2)
        try:
            path = shortest_path_bfs(grid, start, end)
        except ValueError:
            continue
        if len(path) >= min_len:
            return grid, tuple(start), tuple(end), path
    raise RuntimeError(f"Failed to generate maze with required path length >= {min_len}.")


def render_maze(grid, start, end, path=None, size_px=512):
    n = len(grid)
    img = Image.new("RGB", (size_px, size_px), "black")
    draw = ImageDraw.Draw(img)
    cell_size_f = float(size_px) / n
    wall_w_f = cell_size_f / 4.0
    half_wall_f = wall_w_f / 2.0
    grid_w = max(1, int(cell_size_f / 16.0))

    for r in range(n):
        for c in range(n):
            x1, y1 = c * cell_size_f + half_wall_f, r * cell_size_f + half_wall_f
            x2, y2 = (c + 1) * cell_size_f - half_wall_f, (r + 1) * cell_size_f - half_wall_f
            draw.rectangle([(x1, y1), (x2, y2)], fill="white")
            cell = grid[r][c]
            if not cell["S"] and r < n - 1:
                draw.rectangle([(x1, y2), (x2, y2 + wall_w_f)], fill="white")
            if not cell["E"] and c < n - 1:
                draw.rectangle([(x2, y1), (x2 + wall_w_f, y2)], fill="white")

    grid_color = (224, 224, 224)
    for r in range(n):
        for c in range(n):
            if r < n - 1 and not grid[r][c]["S"]:
                y = (r + 1) * cell_size_f
                x1, x2 = c * cell_size_f + half_wall_f, (c + 1) * cell_size_f - half_wall_f
                draw.line([(x1, y), (x2, y)], fill=grid_color, width=grid_w)
            if c < n - 1 and not grid[r][c]["E"]:
                x = (c + 1) * cell_size_f
                y1, y2 = r * cell_size_f + half_wall_f, (r + 1) * cell_size_f - half_wall_f
                draw.line([(x, y1), (x, y2)], fill=grid_color, width=grid_w)

    def draw_dot(rc, color):
        r, c = rc
        cx, cy = c * cell_size_f + cell_size_f / 2, r * cell_size_f + cell_size_f / 2
        rad = max(2, int((cell_size_f - wall_w_f) * 0.25))
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=color)

    draw_dot(start, "yellow")  # MODIFIED: Start point is now yellow
    draw_dot(end, "blue")

    if path is not None:
        pts = [(c * cell_size_f + cell_size_f / 2, r * cell_size_f + cell_size_f / 2) for r, c in path]
        draw.line(pts, fill="red", width=max(1, int(wall_w_f)), joint="curve")
    return img

def save_maze_to_text(grid, start, end, filepath):
    n = len(grid)
    with open(filepath, 'w') as f:
        f.write(f"{n}\n{start[0]} {start[1]}\n{end[0]} {end[1]}\n")
        for r in range(n):
            row_values = []
            for c in range(n):
                cell, value = grid[r][c], 0
                if cell["N"]: value |= 1
                if cell["S"]: value |= 2
                if cell["W"]: value |= 4
                if cell["E"]: value |= 8
                row_values.append(str(value))
            f.write(" ".join(row_values) + "\n")


def process_one_maze(args_tuple):
    """
    A self-contained function for a worker process to generate one maze and its files.
    Returns a tuple (base_filename, udrl_path_string).
    """
    i, size, min_len, out_dir = args_tuple
    
    # 1. Generate maze data
    grid, start, end, path = generate_one_maze_data(size, min_len)
    
    # 2. Define filenames
    f_base = f"{size}_{min_len}_{i:03d}"
    f_img_base = out_dir / f"{f_base}.png"
    f_img_sol  = out_dir / f"{f_base}_solution.png"
    f_txt      = out_dir / f"{f_base}.txt"
    
    # 3. Render and save images
    img_base = render_maze(grid, start, end, path=None)
    img_base.save(f_img_base)
    
    img_sol  = render_maze(grid, start, end, path=path)
    img_sol.save(f_img_sol)
    
    # 4. Save text data
    save_maze_to_text(grid, start, end, f_txt)
    
    # 5. Convert path to UDRL for the final answer key
    udrl_path = convert_path_to_udrl(path)
    
    # 6. Return the result for the main process to collect
    return (f_base, udrl_path)


def main():
    p = argparse.ArgumentParser(description="Generate maze images, data, and solution files concurrently.")
    p.add_argument("--size", type=int, required=True, help="Size of the maze grid (e.g., 8 for an 8x8 maze).")
    p.add_argument("--num", type=int, default=1, help="Total number of mazes to generate.")
    p.add_argument("--min_len", type=int, default=1, help="Minimum required length of the solution path.")
    p.add_argument("--out", type=str, required=True, help="Output directory for all generated files.")
    p.add_argument("--max_workers", type=int, default=64, help="Maximum number of worker processes to use. Capped at 64.")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    # Determine the number of processes to use
    num_workers = min(args.max_workers, 64, args.num)
    print(f"Starting generation of {args.num} mazes using {num_workers} worker processes...")
    
    # Prepare the list of tasks for the process pool
    tasks = [(i, args.size, args.min_len, out) for i in range(1, args.num + 1)]
    
    all_paths_data = {}
    
    # Create a process pool and map tasks to it
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use tqdm for a progress bar
        results = list(tqdm(pool.imap_unordered(process_one_maze, tasks), total=args.num, desc="Generating Mazes"))
        
    # Process the results returned by the workers
    for f_base, udrl_path in results:
        all_paths_data[f_base] = udrl_path
        
    # Save the collected answers to path.json
    path_json_file = out / "path.json"
    # Sort keys for consistent output
    sorted_paths = dict(sorted(all_paths_data.items()))

    def append_json(path_json_file, new_dict):
        if not isinstance(new_dict, dict):
            raise ValueError("not dict")

        if not os.path.exists(path_json_file) or os.path.getsize(path_json_file) == 0:
            base = {}
        else:
            with open(path_json_file, "r", encoding="utf-8") as f:
                base = json.load(f)
                if not isinstance(base, dict):
                    raise ValueError("not dict")

        base.update(new_dict)

        with open(path_json_file, "w", encoding="utf-8") as f:
            json.dump(base, f, indent=4, ensure_ascii=False)
    append_json(path_json_file, sorted_paths)

        
    print("\nGeneration complete.")
    print(f"All files saved in: {out.resolve()}")
    print(f"Answer key saved to: {path_json_file.resolve()}")


if __name__ == "__main__":
    main()
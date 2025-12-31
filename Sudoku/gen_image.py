import argparse
import random
import os
import multiprocessing
import copy
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import json

class SudokuGenerator:
    def __init__(self, size=9):
        self.size = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.solution = None

    def _get_empty_cell(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    return i, j
        return None

    def _is_valid(self, row, col, num):
        if num in self.grid[row]:
            return False
        if num in [self.grid[i][col] for i in range(self.size)]:
            return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if self.grid[start_row + i][start_col + j] == num:
                    return False
        return True

    def solve(self):
        empty_cell = self._get_empty_cell()
        if not empty_cell:
            return True

        row, col = empty_cell
        nums = list(range(1, self.size + 1))
        random.shuffle(nums)

        for num in nums:
            if self._is_valid(row, col, num):
                self.grid[row][col] = num
                if self.solve():
                    return True
                self.grid[row][col] = 0
        return False

    def count_solutions(self):
        count = 0
        
        grid_copy = copy.deepcopy(self.grid)

        def solver():
            nonlocal count
            empty = self._find_empty_in_copy(grid_copy)
            if not empty:
                count += 1
                return
            
            if count >= 2:
                return

            row, col = empty
            for num in range(1, 10):
                if self._is_valid_in_copy(grid_copy, row, col, num):
                    grid_copy[row][col] = num
                    solver()
                    grid_copy[row][col] = 0
                    if count >= 2:
                        return
        
        solver()
        return count

    def _find_empty_in_copy(self, grid):
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 0:
                    return (i, j)
        return None

    def _is_valid_in_copy(self, grid, row, col, num):
        if num in grid[row]: return False
        if num in [grid[i][col] for i in range(9)]: return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if grid[start_row + i][start_col + j] == num:
                    return False
        return True

    def generate_puzzle(self, difficulty):
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.solve()
        self.solution = copy.deepcopy(self.grid)

        cells = list(range(self.size * self.size))
        random.shuffle(cells)
        
        cells_to_remove = self.size * self.size - difficulty
        removed_count = 0
        
        for cell_idx in cells:
            if removed_count >= cells_to_remove:
                break
                
            row, col = cell_idx // self.size, cell_idx % self.size
            if self.grid[row][col] == 0:
                continue

            temp = self.grid[row][col]
            self.grid[row][col] = 0

            if self.count_solutions() != 1:
                self.grid[row][col] = temp
            else:
                removed_count += 1
        
        if removed_count < cells_to_remove:
            return self.generate_puzzle(difficulty)

        return self.grid, self.solution


def render_sudoku(grid, out_path, size_px=512):
    img = Image.new("RGB", (size_px, size_px), "white")
    draw = ImageDraw.Draw(img)
    cell_size = size_px / 9
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/lato/Lato-Medium.ttf", int(cell_size * 0.7))
    except IOError:
        font = ImageFont.load_default()

    for i in range(10):
        line_width = 3 if i % 3 == 0 else 1
        draw.line([(i * cell_size, 0), (i * cell_size, size_px)], fill="black", width=line_width)
        draw.line([(0, i * cell_size), (size_px, i * cell_size)], fill="black", width=line_width)

    for i in range(9):
        for j in range(9):
            num = grid[i][j]
            if num != 0:
                text = str(num)
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                x = (j * cell_size) + (cell_size - text_width) / 2
                y = (i * cell_size) + (cell_size - text_height) / 2
                draw.text((x, y), text, fill="black", font=font)
    
    img.save(out_path)


def save_sudoku_to_text(grid, out_path):
    puzzle_str = "".join([str(cell) for row in grid for cell in row])
    with open(out_path, 'w') as f:
        f.write(puzzle_str)


def process_one_sudoku(args_tuple):
    i, exist, out_dir = args_tuple
    
    generator = SudokuGenerator()
    puzzle, solution = generator.generate_puzzle(exist)
    
    f_base = f"{exist}_{i:03d}"
    f_img_puzzle = out_dir / f"{f_base}.png"
    f_img_solution = out_dir / f"{f_base}_solution.png"
    f_txt = out_dir / f"{f_base}.txt"
    
    render_sudoku(puzzle, f_img_puzzle)
    render_sudoku(solution, f_img_solution)
    save_sudoku_to_text(puzzle, f_txt)

    solution_str = "".join([str(cell) for row in solution for cell in row])
    return f_base, solution_str
    
    return f_base


def main():
    p = argparse.ArgumentParser(description="Generate 9x9 Sudoku puzzles with unique solutions.")
    p.add_argument("--num", type=int, default=1, help="Total number of puzzles to generate.")
    p.add_argument("--exist", type=int, required=True, help="Number of pre-filled cells (e.g., 40).")
    p.add_argument("--out", type=str, required=True, help="Output directory.")
    p.add_argument("--max_workers", type=int, default=64, help="Max worker processes.")
    args = p.parse_args()

    if not (17 <= args.exist <= 80):
        print("Warning: 'exist' is recommended to be between 17 and 80.")
        print("Puzzles with fewer than 17 clues may not have a unique solution.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    
    num_workers = min(args.max_workers, 64, args.num)
    
    print(f"Generating {args.num} Sudoku puzzles...")
    print(f"Config: 9x9, {args.exist} existing cells.")
    
    tasks = [(i, args.exist, out) for i in range(1, args.num + 1)]
    
    all_solutions = {}
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_sudoku, tasks), total=args.num))
    
    for f_base, sol_str in results:
        all_solutions[f_base] = sol_str

    path_json_file = out / "solutions.json"
    
    if not os.path.exists(path_json_file) or os.path.getsize(path_json_file) == 0:
        base_json = {}
    else:
        try:
            with open(path_json_file, "r", encoding="utf-8") as f:
                base_json = json.load(f)
        except json.JSONDecodeError:
            base_json = {}

    base_json.update(dict(sorted(all_solutions.items())))

    with open(path_json_file, "w", encoding="utf-8") as f:
        json.dump(base_json, f, indent=4, ensure_ascii=False)
        
    print(f"Done. Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
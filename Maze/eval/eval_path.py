import argparse
import json
from pathlib import Path

def parse_maze_from_text(filepath: Path):
    try:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            
            if not lines:
                return None

            n = int(lines[0])
            start_r, start_c = map(int, lines[1].split())
            end_r, end_c = map(int, lines[2].split())
            
            grid_data = []
            for r in range(n):
                row = []
                if 3 + r >= len(lines):
                    break
                row_values = map(int, lines[3 + r].split())
                for val in row_values:
                    cell = {
                        "N": (val & 1) != 0,
                        "S": (val & 2) != 0,
                        "W": (val & 4) != 0,
                        "E": (val & 8) != 0,
                    }
                    row.append(cell)
                grid_data.append(row)

            return {
                "size": n,
                "start": (start_r, start_c),
                "end": (end_r, end_c),
                "grid": grid_data
            }
    except (IOError, IndexError, ValueError) as e:
        return None

def verify_path(maze_data, path_udrl: str) -> bool:
    if not maze_data:
        return False
        
    n = maze_data["size"]
    grid = maze_data["grid"]
    current_pos = maze_data["start"]
    
    moves = {
        'U': (-1, 0, 'N'),
        'D': (1, 0, 'S'),
        'L': (0, -1, 'W'),
        'R': (0, 1, 'E'),
    }

    clean_path = path_udrl.replace(",", "").replace(" ", "").strip()

    for move_char in clean_path:
        if move_char not in moves:
            continue

        dr, dc, wall_to_check = moves[move_char]
        r, c = current_pos
        
        if grid[r][c][wall_to_check]:
            return False
            
        nr, nc = r + dr, c + dc
        
        if not (0 <= nr < n and 0 <= nc < n):
            return False
            
        current_pos = (nr, nc)
        
    return current_pos == maze_data["end"]


def main():
    parser = argparse.ArgumentParser(description="Verify maze solutions from a JSON file.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file containing solutions.")
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.is_file():
        print(f"Error: JSON file not found at '{json_path}'")
        return

    data_dir = json_path.parent.parent

    try:
        with open(json_path, 'r') as f:
            solutions = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'")
        return

    if not isinstance(solutions, dict):
        print("Error: JSON content must be a dictionary.")
        return
        
    correct_count = 0
    total_count = len(solutions)
    correctly_solved = []
    skipped_count = 0

    for name, path in solutions.items():
        clean_name = name.replace(".png", "")
        txt_filepath = data_dir / f"{clean_name}.txt"
        
        if not txt_filepath.exists():
            if skipped_count < 3:
                print(f"Warning: Skipping '{name}'. Corresponding file not found: {txt_filepath}")
            skipped_count += 1
            continue
            
        maze = parse_maze_from_text(txt_filepath)
        
        if maze is None:
            if skipped_count < 3:
                print(f"Warning: Skipping '{name}'. Failed to parse maze file: {txt_filepath}")
            skipped_count += 1
            continue
        
        if verify_path(maze, path):
            correct_count += 1
            correctly_solved.append({"name": name, "length": len(path)})

    valid_total = total_count - skipped_count

    print("\n======================================")
    print("Verification Summary")
    print("======================================")

    if skipped_count > 0:
        print(f"Skipped Files: {skipped_count} (due to missing or invalid .txt files)")

    if valid_total == 0:
        print("No valid tasks to evaluate.")
        return
        
    accuracy = (correct_count / valid_total) * 100
    print(f"Total Valid Tasks: {valid_total}")
    print(f"Correctly Solved : {correct_count}")
    print(f"Accuracy         : {accuracy:.2f}%")
    print("--------------------------------------")

    correctly_solved.sort(key=lambda x: x["length"], reverse=True)

    print("\nTop 3 Longest Correct Paths:")
    if not correctly_solved:
        print("  None")
    else:
        for i, item in enumerate(correctly_solved[:3]):
            print(f"  {i+1}. {item['name']} (Length: {item['length']})")
            
    print("======================================")

if __name__ == "__main__":
    main()
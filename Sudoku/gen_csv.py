import csv
import os
from pathlib import Path

TARGET_DIRS = [
    "30_train",
    "35_train",
    "40_train",
    "45_train",
]

OUTPUT_FILE = "metadata_edit.csv"

PROMPT = "Solve this Sudoku puzzle."

def main():
    headers = ['edit_image', 'image', 'prompt']
    rows = []
    total_pairs = 0

    print(f"Start scanning directories: {TARGET_DIRS} ...")

    for dir_name in TARGET_DIRS:
        base_dir = Path(dir_name)
        
        if not base_dir.exists():
            print(f"[Warning] Directory not found: {dir_name}, skipping.")
            continue
        
        for input_img_path in base_dir.rglob("*.png"):
            
            if "solution" in input_img_path.name:
                continue

            stem = input_img_path.stem
            solution_name = f"{stem}_solution.png"
            solution_img_path = input_img_path.parent / solution_name

            if solution_img_path.exists():
                edit_path_str = str(input_img_path).replace(os.sep, '/')
                sol_path_str = str(solution_img_path).replace(os.sep, '/')

                rows.append({
                    'edit_image': edit_path_str,
                    'image': sol_path_str,
                    'prompt': PROMPT
                })
                total_pairs += 1

    if rows:
        try:
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
            
            print(f"\nSuccess! Generated '{OUTPUT_FILE}' with {total_pairs} pairs.")
            
            print("\nPreivew (First 3 lines):")
            for i in range(min(3, len(rows))):
                print(rows[i])
                
        except Exception as e:
            print(f"Error writing CSV: {e}")
    else:
        print("\nNo valid image pairs found. Please check your directories.")

if __name__ == "__main__":
    main()
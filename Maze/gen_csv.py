import os
import csv

BASE_DIR = "path/to/DiffThinker/Maze"

SOURCE_DIRS = [
    os.path.join(BASE_DIR, "8_train"),
    os.path.join(BASE_DIR, "16_train"),
    os.path.join(BASE_DIR, "32_train")
]

OUTPUT_CSV = os.path.join(BASE_DIR, "metadata_edit.csv")

FIXED_PROMPT = "Draw a continuous red line connecting the yellow dot to the blue dot, avoiding all walls."

def main():
    records = []
    
    print(f"Start processing... Output will be saved to: {OUTPUT_CSV}")

    for data_dir in SOURCE_DIRS:
        if not os.path.exists(data_dir):
            print(f"Warning: Directory not found -> {data_dir}")
            continue
            
        subdir_name = os.path.basename(data_dir)
        print(f"Scanning: {subdir_name} ...")

        for filename in os.listdir(data_dir):
            if filename.endswith("_solution.png"):
                target_filename = filename
                source_filename = filename.replace("_solution.png", ".png")
                
                abs_source_path = os.path.join(data_dir, source_filename)
                
                if os.path.exists(abs_source_path):
                    rel_source_path = os.path.join(subdir_name, source_filename)
                    rel_target_path = os.path.join(subdir_name, target_filename)
                    
                    records.append([rel_source_path, rel_target_path, FIXED_PROMPT])

    print(f"Writing {len(records)} records to CSV...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["edit_image", "image", "prompt"])
        writer.writerows(records)
    
    print("Done.")

if __name__ == "__main__":
    main()
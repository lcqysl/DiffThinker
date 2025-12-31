import os
import csv
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=64)
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.dir
    
    output_csv = os.path.join(data_dir, "metadata_edit.csv")
    fixed_prompt = "Draw a continuous red line connecting the Start point to the Goal point, avoiding all holes."

    records = []

    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith("_solution.png"):
                target_img = filename
                source_img = filename.replace("_solution.png", ".png")
                
                if os.path.exists(os.path.join(data_dir, source_img)):
                    records.append([source_img, target_img, fixed_prompt])
    else:
        return

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["edit_image", "image", "prompt"])
        writer.writerows(records)

if __name__ == "__main__":
    main()
import argparse
import random
import os
import multiprocessing
import json
import shutil
import uuid
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

IMG_SIZE = 512
GRID_WIDTH = 4
LABEL_BOX_SIZE = 32
FONT_SIZE_RATIO = 0.9
AR_THRESHOLD = 1.35

class PuzzleGenerator:
    def __init__(self, size_str="2x2"):
        self.rows, self.cols = map(int, size_str.lower().split('x'))
        self.num_blocks = self.rows * self.cols

    def process_image(self, img_path):
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                if min(w, h) == 0: return None, None, None
                
                ratio = max(w, h) / min(w, h)
                if ratio > AR_THRESHOLD:
                    return None, None, None

                im = im.convert("RGB")
                im = im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
                
            block_w = IMG_SIZE // self.cols
            block_h = IMG_SIZE // self.rows
            
            original_raw_blocks = []
            for r in range(self.rows):
                for c in range(self.cols):
                    box = (c * block_w, r * block_h, (c + 1) * block_w, (r + 1) * block_h)
                    original_raw_blocks.append(im.crop(box))

            indices = list(range(self.num_blocks))
            random.shuffle(indices)
            visual_blocks = []
            
            for visual_idx, original_idx in enumerate(indices):
                block = original_raw_blocks[original_idx].copy()
                label_str = str(visual_idx + 1)
                self._draw_label(block, label_str)
                visual_blocks.append(block)

            puzzle_img = self._stitch_blocks(visual_blocks)
            
            solution_ordered_blocks = [None] * self.num_blocks
            for i, original_idx in enumerate(indices):
                solution_ordered_blocks[original_idx] = visual_blocks[i]
                
            solution_img = self._stitch_blocks(solution_ordered_blocks)
            
            pos_map = {}
            for visual_pos, original_idx in enumerate(indices):
                pos_map[original_idx] = visual_pos + 1
            
            solution_seq = [pos_map[i] for i in range(self.num_blocks)]
            solution_str = " ".join(map(str, solution_seq))

            return puzzle_img, solution_img, solution_str

        except Exception as e:
            return None, None, None

    def _draw_label(self, block_img, text):
        draw = ImageDraw.Draw(block_img)
        draw.rectangle([(0, 0), (LABEL_BOX_SIZE, LABEL_BOX_SIZE)], fill="white")
        draw.rectangle([(0, 0), (LABEL_BOX_SIZE, LABEL_BOX_SIZE)], outline="black", width=1)
        
        try:
            font_size = int(LABEL_BOX_SIZE * FONT_SIZE_RATIO)
            font_paths = [
                "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
            ]
            font = None
            for p in font_paths:
                try:
                    font = ImageFont.truetype(p, font_size)
                    break
                except:
                    continue
            if font is None:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        
        x = (LABEL_BOX_SIZE - text_w) / 2
        y = (LABEL_BOX_SIZE - text_h) / 2
        draw.text((x, y), text, fill="black", font=font)

    def _stitch_blocks(self, blocks):
        block_w, block_h = blocks[0].size
        total_w = self.cols * block_w + (self.cols - 1) * GRID_WIDTH
        total_h = self.rows * block_h + (self.rows - 1) * GRID_WIDTH
        full_img = Image.new("RGB", (total_w, total_h), "white")
        
        for idx, block in enumerate(blocks):
            if block is None: continue
            r = idx // self.cols
            c = idx % self.cols
            x = c * (block_w + GRID_WIDTH)
            y = r * (block_h + GRID_WIDTH)
            full_img.paste(block, (x, y))
            
        return full_img

def process_one_task(args_tuple):
    img_path, size_str, temp_dir = args_tuple
    
    generator = PuzzleGenerator(size_str)
    puzzle_img, solution_img, solution_str = generator.process_image(img_path)
    
    if puzzle_img is None:
        return None

    unique_id = str(uuid.uuid4())
    f_puzzle_temp = temp_dir / f"{unique_id}.png"
    f_solution_temp = temp_dir / f"{unique_id}_solution.png"
    
    try:
        puzzle_img.save(f_puzzle_temp)
        solution_img.save(f_solution_temp)
        return unique_id, solution_str
    except Exception as e:
        return None

def main():
    p = argparse.ArgumentParser(description="Generate labeled jigsaw puzzles.")
    p.add_argument("--source_dir", type=str, required=True, help="Directory containing source images.")
    p.add_argument("--num", type=int, default=1000, help="Target number of VALID puzzles.")
    p.add_argument("--size", type=str, default="2x2", help="Grid size (e.g., 2x2, 3x3).")
    p.add_argument("--out", type=str, default="./puzzle_train", help="Output directory.")
    p.add_argument("--max_workers", type=int, default=64, help="Max worker processes.")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = out_dir / "temp_processing"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    source_dir = Path(args.source_dir)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    print(f"Scanning {source_dir}...")
    all_images = [
        p for p in source_dir.rglob("*") 
        if p.suffix.lower() in valid_extensions
    ]
    
    if not all_images:
        print(f"Error: No images found in {args.source_dir}")
        return

    print(f"Found {len(all_images)} source images.")
    print(f"Target: {args.num} valid puzzles (AR <= {AR_THRESHOLD}).")
    
    if len(all_images) < args.num:
        print(f"Warning: Source images ({len(all_images)}) are fewer than target ({args.num}).")
        print("Will try to process all, but will definitely fail to reach target.")

    random.shuffle(all_images)
    
    tasks = [(img, args.size, temp_dir) for img in all_images]
    
    num_workers = min(args.max_workers, 64)
    all_solutions = {}
    valid_count = 0
    
    print(f"Processing tasks with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        iterator = pool.imap_unordered(process_one_task, tasks)
        
        pbar = tqdm(total=args.num)
        
        try:
            for res in iterator:
                if res is None:
                    continue
                
                unique_id, sol_str = res
                
                if valid_count < args.num:
                    valid_count += 1
                    
                    f_base = f"{args.size}_{valid_count:05d}"
                    
                    src_puzzle = temp_dir / f"{unique_id}.png"
                    src_solution = temp_dir / f"{unique_id}_solution.png"
                    
                    dst_puzzle = out_dir / f"{f_base}.png"
                    dst_solution = out_dir / f"{f_base}_solution.png"
                    dst_txt = out_dir / f"{f_base}.txt"
                    
                    shutil.move(src_puzzle, dst_puzzle)
                    shutil.move(src_solution, dst_solution)
                    
                    with open(dst_txt, 'w') as f:
                        f.write(sol_str)
                    
                    all_solutions[f_base] = sol_str
                    pbar.update(1)
                    
                    if valid_count >= args.num:
                        break
                else:
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            pool.terminate()
        finally:
            pbar.close()

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    if valid_count < args.num:
        print("\n" + "!" * 40)
        print(f"ERROR: Insufficient valid images!")
        print(f"Target: {args.num}")
        print(f"Actual Valid: {valid_count}")
        print(f"Total Source Checked: {len(all_images)}")
        print(f"Filtered out (AR>{AR_THRESHOLD} or bad file): {len(all_images) - valid_count}")
        print("!" * 40)
        
        path_json_file = out_dir / "solutions.json"
        if path_json_file.exists() and path_json_file.stat().st_size > 0:
            try:
                with open(path_json_file, "r", encoding="utf-8") as f:
                    base_json = json.load(f)
            except json.JSONDecodeError:
                base_json = {}
        else:
            base_json = {}

        base_json.update(dict(sorted(all_solutions.items())))
        with open(path_json_file, "w", encoding="utf-8") as f:
            json.dump(base_json, f, indent=4, ensure_ascii=False)
            
        sys.exit(1)

    path_json_file = out_dir / "solutions.json"
    if path_json_file.exists() and path_json_file.stat().st_size > 0:
        try:
            with open(path_json_file, "r", encoding="utf-8") as f:
                base_json = json.load(f)
        except json.JSONDecodeError:
            base_json = {}
    else:
        base_json = {}

    base_json.update(dict(sorted(all_solutions.items())))
    with open(path_json_file, "w", encoding="utf-8") as f:
        json.dump(base_json, f, indent=4, ensure_ascii=False)
        
    print(f"Done. Generated {valid_count} puzzles.")
    print(f"Saved to {out_dir.resolve()}")

if __name__ == "__main__":
    main()
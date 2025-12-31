import hashlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm 

# ================= CONFIGURATION =================
# Add the directories you want to scan here
SEARCH_DIRS = [
    "path/to/DiffThinker/Maze/8_test",
    "path/to/DiffThinker/Maze/8_train",
    "path/to/DiffThinker/Maze/16_test",
    "path/to/DiffThinker/Maze/16_train",
    "path/to/DiffThinker/Maze/32_test",
    "path/to/DiffThinker/Maze/32_train",
]
# =================================================

def get_file_hash(filepath):
    """Calculates the MD5 hash of a file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            # Read in chunks to avoid memory issues with large files
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def find_txt_files(directories):
    """Recursively finds all .txt files in the specified directories."""
    files = []
    print("Scanning directories...")
    for d in directories:
        path = Path(d)
        if not path.exists():
            print(f"Warning: Directory not found, skipping: {d}")
            continue
        
        found = list(path.rglob("*.txt"))
        print(f"  -> Found {len(found)} files in '{d}'")
        files.extend(found)
    return files

def main():
    # 1. Collect all files
    all_files = find_txt_files(SEARCH_DIRS)
    
    if not all_files:
        print("\nNo .txt files found. Please check your configuration.")
        return

    print(f"\nTotal {len(all_files)} files found. Starting hash computation...")

    # 2. Compute hashes and group by content
    content_map = defaultdict(list)
    
    # Use tqdm for progress bar
    for fpath in tqdm(all_files, desc="Computing Hashes"):
        file_hash = get_file_hash(fpath)
        if file_hash:
            content_map[file_hash].append(fpath)

    # 3. Analyze results
    total_files = len(all_files)
    unique_contents = len(content_map)
    duplicate_groups = {k: v for k, v in content_map.items() if len(v) > 1}
    redundant_files_count = total_files - unique_contents

    print("\n" + "="*40)
    print(f"Summary Report")
    print("="*40)
    print(f"Total Files Scanned : {total_files}")
    print(f"Unique Contents     : {unique_contents}")
    print(f"Redundant Files     : {redundant_files_count}")
    print(f"Duplicate Groups    : {len(duplicate_groups)}")
    print("="*40)

    if not duplicate_groups:
        print("\nPerfect! No duplicate files found.")
        return

    print("\nFound duplicates:")
    
    for i, (file_hash, paths) in enumerate(duplicate_groups.items(), 1):
        paths.sort() 
        print(f"\n[Group #{i} | Hash: {file_hash[:8]}...] - {len(paths)} copies:")
        for p in paths:
            print(f"  - {p}")

if __name__ == "__main__":
    main()
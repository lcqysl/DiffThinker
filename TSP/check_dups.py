import os
from pathlib import Path
from collections import defaultdict
import hashlib

TARGET_DIRECTORIES = [
    "path/to/data/DiffThinker/TSP/12_test",
    "path/to/data/DiffThinker/TSP/15_test",
    "path/to/data/DiffThinker/TSP/18_test",
    "path/to/data/DiffThinker/TSP/12_train",
    "path/to/data/DiffThinker/TSP/13_train",
    "path/to/data/DiffThinker/TSP/14_train",
    "path/to/data/DiffThinker/TSP/15_train",
    "path/to/data/DiffThinker/TSP/16_train",
    "path/to/data/DiffThinker/TSP/17_train",
]

def parse_map_file(filepath):
    """
    读取txt文件并生成唯一的地图指纹。
    指纹由 (grid_size, 排序后的坐标点元组) 组成。
    这样即使点的顺序不同，只要点的位置集合相同，也会被视为重复。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            
        if not lines:
            return None

        # 第一行通常是 "GridSize NumPoints"
        header = lines[0].split()
        if len(header) < 1:
            return None
        grid_size = int(header[0])

        # 读取后续的坐标行
        points = []
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                x, y = int(parts[0]), int(parts[1])
                points.append((x, y))
        
        # 关键步骤：排序
        # 对点进行排序，确保检测的是“点集”是否重复，忽略文件中的书写顺序
        sorted_points = tuple(sorted(points))
        
        return (grid_size, sorted_points)

    except Exception as e:
        print(f"[Warning] 无法读取文件 {filepath}: {e}")
        return None

def find_duplicates(directories):
    # 存储结构： { 指纹: [文件路径1, 文件路径2, ...] }
    map_registry = defaultdict(list)
    total_files = 0
    
    print(f"开始扫描目录: {directories} ...")

    for dir_path in directories:
        p = Path(dir_path)
        if not p.exists():
            print(f"[Skipped] 目录不存在: {dir_path}")
            continue
            
        # 遍历所有txt文件
        for txt_file in p.glob("*.txt"):
            total_files += 1
            fingerprint = parse_map_file(txt_file)
            
            if fingerprint:
                map_registry[fingerprint].append(txt_file)

    # 统计重复
    duplicates = {k: v for k, v in map_registry.items() if len(v) > 1}
    unique_count = len(map_registry)
    
    print("\n" + "="*40)
    print("扫描结果报告")
    print("="*40)
    print(f"扫描文件总数: {total_files}")
    print(f"唯一地图数量: {unique_count}")
    print(f"重复地图组数: {len(duplicates)}")
    
    if duplicates:
        print("\n[发现重复详情]:")
        for i, (fp, paths) in enumerate(duplicates.items(), 1):
            grid_size = fp[0]
            points_sample = fp[1]
            print(f"\n{i}. 相同地图 (Size: {grid_size}, Points: {points_sample}) 出现在 {len(paths)} 个文件中:")
            for path in paths:
                print(f"   - {path}")
    else:
        print("\n完美！没有发现重复的地图。")

    print("="*40)

if __name__ == "__main__":
    find_duplicates(TARGET_DIRECTORIES)
import os
from pathlib import Path
from collections import defaultdict

# 请根据您的实际路径修改这里
TARGET_DIRECTORIES = [
    "path/to/DiffThinker/Sudoku/30_train",
    "path/to/DiffThinker/Sudoku/35_train",
    "path/to/DiffThinker/Sudoku/40_train",
    "path/to/DiffThinker/Sudoku/45_train",
    "path/to/DiffThinker/Sudoku/25_test",
    "path/to/DiffThinker/Sudoku/35_test",
    "path/to/DiffThinker/Sudoku/45_test",
]

def parse_sudoku_file(filepath):
    """
    读取txt文件并生成唯一的数独指纹。
    指纹为清洗后的81位数字字符串。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 清洗数据：移除换行符和空格，只保留内容
        # 假设文件内容是 "530070000..." 或者是矩阵形式
        clean_content = "".join(content.split())
        
        # 简单校验：标准数独应该是81位字符
        if len(clean_content) != 81:
            # 如果不是81位，可能是格式不对，或者空文件
            return None
        
        # 对于数独，题目本身（包含0的字符串）就是唯一的指纹
        return clean_content

    except Exception as e:
        print(f"[Warning] 无法读取文件 {filepath}: {e}")
        return None

def find_duplicates(directories):
    # 存储结构： { 指纹字符串: [文件路径1, 文件路径2, ...] }
    puzzle_registry = defaultdict(list)
    total_files = 0
    
    print(f"开始扫描目录: {directories} ...")

    for dir_path in directories:
        p = Path(dir_path)
        if not p.exists():
            print(f"[Skipped] 目录不存在: {dir_path}")
            continue
            
        # 遍历所有txt文件
        # 注意：之前的生成脚本生成的题目文件是 .txt
        for txt_file in p.glob("*.txt"):
            total_files += 1
            fingerprint = parse_sudoku_file(txt_file)
            
            if fingerprint:
                puzzle_registry[fingerprint].append(txt_file)

    # 统计重复
    # 筛选出列表长度大于1的项
    duplicates = {k: v for k, v in puzzle_registry.items() if len(v) > 1}
    unique_count = len(puzzle_registry)
    
    print("\n" + "="*40)
    print("数独查重扫描报告")
    print("="*40)
    print(f"扫描文件总数: {total_files}")
    print(f"唯一题目数量: {unique_count}")
    print(f"重复题目组数: {len(duplicates)}")
    
    if duplicates:
        print("\n[发现重复详情]:")
        for i, (fp, paths) in enumerate(duplicates.items(), 1):
            # 为了显示简洁，只打印指纹的前20位和最后5位
            fp_display = f"{fp[:20]}...{fp[-5:]}"
            print(f"\n{i}. 相同题目 ({fp_display}) 出现在 {len(paths)} 个文件中:")
            for path in paths:
                print(f"   - {path}")
    else:
        print("\n完美！没有发现重复的数独题目。")

    print("="*40)

if __name__ == "__main__":
    find_duplicates(TARGET_DIRECTORIES)
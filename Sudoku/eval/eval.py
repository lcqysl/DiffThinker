import json
import argparse
import re
from pathlib import Path

def parse_sudoku_solution(raw_str):
    if not raw_str or not isinstance(raw_str, str):
        return ""
    
    target_marker = "Solution:"
    idx = raw_str.rfind(target_marker)
    if idx != -1:
        raw_str = raw_str[idx + len(target_marker):]

    digits = re.sub(r'\D', '', raw_str)
    return digits

def count_mismatches(str1, str2):
    if len(str1) != len(str2):
        return abs(len(str1) - len(str2))
    return sum(1 for a, b in zip(str1, str2) if a != b)

def normalize_key(key):
    if key.endswith(".png"):
        return key[:-4]
    return key

def evaluate(gt_path, pred_path, verbose=False):
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        raw_pred_data = json.load(f)
        
    pred_map = {}
    for k, v in raw_pred_data.items():
        norm_k = normalize_key(k)
        pred_map[norm_k] = v

    total_samples = 0
    correct_count = 0
    missing_count = 0
    invalid_format_count = 0
    wrong_value_count = 0
    
    keys = sorted(gt_data.keys())
    
    print(f"{'Sample ID':<20} | {'Status':<15} | {'Details':<20}")
    print("-" * 60)
    
    for key in keys:
        total_samples += 1
        
        gt_raw = gt_data[key]
        gt_clean = parse_sudoku_solution(gt_raw)
        
        key_norm = normalize_key(key)
        pred_raw = pred_map.get(key_norm)
        
        if pred_raw is None:
            missing_count += 1
            if verbose: print(f"{key:<20} | MISSING         | No prediction found")
            continue
            
        pred_clean = parse_sudoku_solution(pred_raw)
        
        status = ""
        details = ""
        
        if len(pred_clean) != 81:
            invalid_format_count += 1
            status = "INVALID_LEN"
            details = f"Length: {len(pred_clean)} (Expected 81)"
        
        elif pred_clean == gt_clean:
            # print(pred_clean)
            correct_count += 1
            status = "PASS"
            details = "Exact Match"
            
        else:
            print(pred_clean)
            wrong_value_count += 1
            status = "WRONG"
            mismatches = count_mismatches(pred_clean, gt_clean)
            details = f"Mismatched cells: {mismatches}"
        
        if status != "PASS":
            if verbose:
                print(f"{key:<20} | {status:<15} | {details}")
        elif verbose and total_samples <= 5:
            print(f"{key:<20} | {status:<15} | {details}")

    accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n" + "="*40)
    print(f"Sudoku Evaluation Results")
    print("="*40)
    print(f"Total Samples:      {total_samples}")
    print(f"Correct Predictions:{correct_count}")
    print(f"Accuracy:           {accuracy:.2f}%")
    print("-" * 40)
    print(f"Missing Keys:       {missing_count}")
    print(f"Invalid Length:     {invalid_format_count} (Output not 81 digits)")
    print(f"Wrong Values:       {wrong_value_count} (Length 81 but incorrect digits)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sudoku Predictions (Exact String Match)")
    parser.add_argument("gt", type=str, help="Path to Ground Truth JSON (solutions.json)")
    parser.add_argument("pred", type=str, help="Path to Prediction JSON")
    parser.add_argument("--verbose", action="store_true", help="Print details for failed cases")
    
    args = parser.parse_args()
    
    if not Path(args.gt).exists() or not Path(args.pred).exists():
        print("Error: Input files not found.")
    else:
        evaluate(args.gt, args.pred, args.verbose)
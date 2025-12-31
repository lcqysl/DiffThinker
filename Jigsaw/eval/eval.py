import json
import argparse
import re
from pathlib import Path

def parse_puzzle_solution(raw_str):
    if not raw_str or not isinstance(raw_str, str):
        return []
    
    target_marker = "Solution:"
    idx = raw_str.rfind(target_marker)
    if idx != -1:
        raw_str = raw_str[idx + len(target_marker):]

    tokens = re.findall(r'\d+', raw_str)
    return tokens

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
    print("-" * 65)
    
    for key in keys:
        total_samples += 1
        
        gt_raw = gt_data[key]
        gt_tokens = parse_puzzle_solution(gt_raw)
        
        key_norm = normalize_key(key)
        pred_raw = pred_map.get(key_norm)
        
        if pred_raw is None:
            missing_count += 1
            if verbose: print(f"{key:<20} | MISSING         | No prediction found")
            continue
            
        pred_tokens = parse_puzzle_solution(pred_raw)
        
        status = ""
        details = ""
        
        if len(pred_tokens) != len(gt_tokens):
            invalid_format_count += 1
            status = "INVALID_LEN"
            details = f"Got {len(pred_tokens)} items, expected {len(gt_tokens)}"
        
        elif pred_tokens == gt_tokens:
            correct_count += 1
            status = "PASS"
            details = "Exact Match"
            
        else:
            wrong_value_count += 1
            status = "WRONG"
            mismatches = sum(1 for p, g in zip(pred_tokens, gt_tokens) if p != g)
            details = f"Mismatched blocks: {mismatches}"
        
        if status != "PASS":
            if verbose:
                print(f"{key:<20} | {status:<15} | {details}")
                # print(f"   GT: {gt_tokens}")
                # print(f"   PR: {pred_tokens}")
        elif verbose and total_samples <= 5:
            print(f"{key:<20} | {status:<15} | {details}")

    accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n" + "="*45)
    print(f"Jigsaw Puzzle Evaluation Results")
    print("="*45)
    print(f"Total Samples:      {total_samples}")
    print(f"Correct Predictions:{correct_count}")
    print(f"Accuracy:           {accuracy:.2f}%")
    print("-" * 45)
    print(f"Missing Keys:       {missing_count}")
    print(f"Invalid Length:     {invalid_format_count} (Item count mismatch)")
    print(f"Wrong Values:       {wrong_value_count} (Correct length but wrong order)")
    print("="*45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Jigsaw Puzzle Predictions")
    parser.add_argument("gt", type=str, help="Path to Ground Truth JSON (solutions.json)")
    parser.add_argument("pred", type=str, help="Path to Prediction JSON")
    parser.add_argument("--verbose", action="store_true", help="Print details for failed cases")
    
    args = parser.parse_args()
    
    if not Path(args.gt).exists() or not Path(args.pred).exists():
        print("Error: Input files not found.")
    else:
        evaluate(args.gt, args.pred, args.verbose)
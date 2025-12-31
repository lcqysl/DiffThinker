import json
import math
import argparse
import re
from pathlib import Path

def parse_coordinate_string(coord_str):
    if not coord_str or not isinstance(coord_str, str):
        return []
    
    target_marker = "Path:"
    idx = coord_str.rfind(target_marker)
    
    if idx != -1:
        coord_str = coord_str[idx + len(target_marker):]

    matches = re.findall(r'\((\d+)\s*,\s*(\d+)\)', coord_str)
    points = [(int(x), int(y)) for x, y in matches]
    return points

def calculate_total_distance(points):
    if not points or len(points) < 2:
        return 0.0
    
    total_dist = 0.0
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        total_dist += dist
        
    return total_dist

def get_node_set(points):
    return set(points)

def normalize_key(key):
    if key.endswith(".png"):
        return key[:-4]
    return key

def evaluate(gt_path, pred_path, output_diff=False):
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
    invalid_structure_count = 0
    suboptimal_count = 0
    
    EPSILON = 1e-4
    
    keys = sorted(gt_data.keys())
    
    print(f"{'Sample ID':<20} | {'GT Dist':<10} | {'Pred Dist':<10} | {'Status':<10}")
    print("-" * 65)
    
    for key in keys:
        total_samples += 1
        
        gt_str = gt_data[key]
        
        key_norm = normalize_key(key)
        pred_str = pred_map.get(key_norm)
        
        if pred_str is None:
            missing_count += 1
            if output_diff: print(f"{key:<20} | {'-':<10} | {'-':<10} | MISSING")
            continue
            
        gt_points = parse_coordinate_string(gt_str)
        pred_points = parse_coordinate_string(pred_str)
        
        if len(pred_points) < 2:
            invalid_structure_count += 1
            if output_diff: print(f"{key:<20} | {calculate_total_distance(gt_points):.4f}     | {'Err':<10} | INVALID")
            continue

        gt_dist = calculate_total_distance(gt_points)
        pred_dist = calculate_total_distance(pred_points)
        
        gt_set = get_node_set(gt_points)
        pred_set = get_node_set(pred_points)
        
        nodes_match = (gt_set == pred_set)

        calc_pred_points = list(pred_points)
        
        if len(calc_pred_points) > 1 and calc_pred_points[0] != calc_pred_points[-1]:
            calc_pred_points.append(calc_pred_points[0]) 
            
        pred_dist = calculate_total_distance(calc_pred_points)
        
        dist_match = (pred_dist <= gt_dist + EPSILON)
        
        if nodes_match and dist_match:
            correct_count += 1
            status = "PASS"
            print(gt_points)
        else:
            status = "FAIL"
            if not nodes_match:
                status = "WRONG_NODES"
            elif not dist_match:
                suboptimal_count += 1
                status = f"LONGER (+{pred_dist - gt_dist:.2f})"
        
        if output_diff and status != "PASS":
            print(f"{key:<20} | {gt_dist:.4f}     | {pred_dist:.4f}     | {status}")
        elif output_diff and total_samples <= 5:
            print(f"{key:<20} | {gt_dist:.4f}     | {pred_dist:.4f}     | {status}")

    accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
    
    print("\n" + "="*30)
    print(f"Evaluation Results")
    print("="*30)
    print(f"Total Samples:      {total_samples}")
    print(f"Correct Predictions:{correct_count}")
    print(f"Accuracy:           {accuracy:.2f}%")
    print("-" * 30)
    print(f"Missing Keys:       {missing_count}")
    print(f"Invalid Format:     {invalid_structure_count}")
    print(f"Suboptimal Path:    {suboptimal_count} (Valid tour but longer distance)")
    print(f"Wrong Nodes:        {total_samples - correct_count - missing_count - invalid_structure_count - suboptimal_count} (Visited different cities)")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TSP Predictions based on Path Distance")
    parser.add_argument("gt", type=str, help="Path to Ground Truth JSON")
    parser.add_argument("pred", type=str, help="Path to Prediction JSON")
    parser.add_argument("--verbose", action="store_true", help="Print details for failed cases")
    
    args = parser.parse_args()
    
    if not Path(args.gt).exists() or not Path(args.pred).exists():
        print("Error: Input files not found.")
    else:
        evaluate(args.gt, args.pred, args.verbose)
import json
import argparse
import re
import sys
from pathlib import Path

def normalize_text(text):
    if not text: return ""
    text = text.strip().lower()
    if text.endswith('.'):
        text = text[:-1]
    return " ".join(text.split())

def parse_grid_type(question):
    match = re.search(r'(\d+)[x×](\d+)', question)
    if match:
        return f"{match.group(1)}x{match.group(2)}"
    return None

def parse_prediction_digits(pred_str):
    if "Solution:" in pred_str:
        pred_str = pred_str.split("Solution:")[-1]
    return re.findall(r'\d+', pred_str)

def generate_hypothesis(grid_type, digits):
    if not digits:
        return []
    
    d = [str(x) for x in digits]
    hypotheses = []

    try:
        if grid_type == "1x2":
            if len(d) >= 2:
                h1 = f"Part {d[1]} should be to the right of Part {d[0]}"
                h2 = f"Part {d[0]} should be to the left of Part {d[1]}"
                hypotheses = [h1, h2]

        elif grid_type == "2x1":
            if len(d) >= 2:
                h1 = f"Part {d[0]} should be to the top of Part {d[1]}"
                h2 = f"Part {d[1]} should be to the bottom of Part {d[0]}"
                hypotheses = [h1, h2]

        elif grid_type == "1x3":
            if len(d) >= 3:
                h = f"The left part should be Part {d[0]}; the middle part should be Part {d[1]}; and the right part should be Part {d[2]}"
                hypotheses = [h]

        elif grid_type == "3x1":
            if len(d) >= 3:
                h = f"The top part should be Part {d[0]}; the middle part should be Part {d[1]}; and the bottom part should be Part {d[2]}"
                hypotheses = [h]

        elif grid_type == "2x2":
            if len(d) >= 4:
                h = f"The top-left part should be Part {d[0]}; the top-right part should be Part {d[1]}; the bottom-left part should be Part {d[2]}; and the bottom-right part should be Part {d[3]}"
                hypotheses = [h]
                
    except Exception as e:
        print(f"Error generating hypothesis for {grid_type} with {digits}: {e}")
        return []

    return [normalize_text(h) for h in hypotheses]

def match_key(bench_key, pred_keys):
    if bench_key in pred_keys:
        return bench_key
    
    stem = Path(bench_key).stem
    if stem in pred_keys:
        return stem

    for pk in pred_keys:
        if bench_key in pk:
            return pk
            
    return None

def evaluate(bench_path, pred_path, verbose=False):
    with open(bench_path, 'r', encoding='utf-8') as f:
        try:
            bench_data = json.load(f)
            if isinstance(bench_data, dict):
                 if 'data' in bench_data:
                     bench_data = bench_data['data']
                 else:
                     bench_data = [v for k,v in bench_data.items()]
        except json.JSONDecodeError:
            f.seek(0)
            bench_data = [json.loads(line) for line in f]

    with open(pred_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    normalized_preds = {}
    for k, v in pred_data.items():
        stem = Path(k).stem
        normalized_preds[stem] = v
    
    total = 0
    correct = 0
    missing = 0
    parse_error = 0
    
    print(f"{'ID':<15} | {'Type':<5} | {'Status':<10} | {'Details'}")
    print("-" * 80)

    for item in bench_data:
        total += 1
        
        pid = item.get("pid", str(total))
        img_path = item.get("problem_image", "")
        
        pred_key = match_key(img_path, normalized_preds.keys())
        if not pred_key:
            pred_key = match_key(pid, normalized_preds.keys())
            
        if not pred_key:
            missing += 1
            if verbose: print(f"{pid:<15} | {'?':<5} | MISSING    | Pred not found for {img_path}")
            continue

        raw_pred = normalized_preds[pred_key]
        pred_digits = parse_prediction_digits(raw_pred)
        
        question = item.get("question", "")
        grid_type = parse_grid_type(question)
        
        if not grid_type:
            if len(pred_digits) == 4: grid_type = "2x2"
            elif len(pred_digits) == 2: grid_type = "1x2"
            elif len(pred_digits) == 3: grid_type = "1x3"
            else:
                if verbose: print(f"{pid:<15} | {'?':<5} | SKIP       | Unknown grid type")
                continue

        hypotheses = generate_hypothesis(grid_type, pred_digits)
        
        if not hypotheses:
            parse_error += 1
            if verbose: print(f"{pid:<15} | {grid_type:<5} | INVALID    | Digits: {pred_digits}")
            continue

        gt_answer = normalize_text(item.get("answer", ""))
        
        
        match = False
        for hyp in hypotheses:
            if hyp == gt_answer:
                match = True
                break
            if hyp.replace(';', '') in gt_answer.replace(';', ''):
                match = True
                break

        if match:
            correct += 1
            status = "PASS"
            detail = "Match"
            # print(f"{pid:<15} | {grid_type:<5} | {status:<10} | {detail}")
        else:
            status = "FAIL"
            detail = f"Pred: {pred_digits} -> Hyp: {hypotheses[0][:30]}... != GT: {gt_answer[:30]}..."
        
        # if verbose or status == "FAIL":
            print(f"{pid:<15} | {grid_type:<5} | {status:<10} | {detail}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print("\n" + "="*50)
    print(f"Puzzle VQA Evaluation")
    print("="*50)
    print(f"Total Samples:       {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy:            {accuracy:.2f}%")
    print("-" * 50)
    print(f"Missing Predictions: {missing}")
    print(f"Format Errors:       {parse_error}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bench", type=str, help="Path to Benchmark JSON/JSONL")
    parser.add_argument("pred", type=str, help="Path to Prediction JSON (from previous step)")
    parser.add_argument("--verbose", action="store_true", help="Show details")
    args = parser.parse_args()
    
    if not Path(args.bench).exists() or not Path(args.pred).exists():
        print("Error: Files not found.")
    else:
        evaluate(args.bench, args.pred, args.verbose)
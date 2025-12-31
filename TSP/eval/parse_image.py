import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import math


def get_grid_coordinates(centroid, cell_size):
    cx, cy = centroid
    grid_x = int(cx / cell_size)
    grid_y = int(cy / cell_size)
    return (grid_x, grid_y)

def get_theoretical_center(grid_pos, cell_size):
    gx, gy = grid_pos
    px_x = int((gx + 0.5) * cell_size)
    px_y = int((gy + 0.5) * cell_size)
    return (px_x, px_y)

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def parse_tsp_image(image_path, grid_size, tolerance=0.70):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, "Read Error"
    
    h, w, _ = img.shape
    cell_size = w / grid_size
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([25, 255, 255])
    lower_red2 = np.array([155, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    
    def extract_nodes(mask, node_type):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nodes = []
        for c in cnts:
            if cv2.contourArea(c) < 5: continue
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                g_pos = get_grid_coordinates((cx, cy), cell_size)
                t_center = get_theoretical_center(g_pos, cell_size)
                
                nodes.append({
                    'pos': g_pos,
                    'px': t_center,
                    'type': node_type
                })
        return nodes

    start_nodes = extract_nodes(mask_yellow, 'start')
    city_nodes = extract_nodes(mask_blue, 'city')
    
    if not start_nodes:
        return None, "No start node found"

    temp_nodes = start_nodes + city_nodes
    unique_nodes = []
    seen_pos = set()
    temp_nodes.sort(key=lambda x: x['type'] == 'start', reverse=True)
    
    for n in temp_nodes:
        if n['pos'] not in seen_pos:
            seen_pos.add(n['pos'])
            unique_nodes.append(n)
            
    all_nodes = unique_nodes
    num_nodes = len(all_nodes)
    
    for idx, node in enumerate(all_nodes):
        node['id'] = idx

    adj = {n['id']: [] for n in all_nodes}
    
    kernel = np.ones((3,3), np.uint8)
    dilated_red = cv2.dilate(mask_red, kernel, iterations=2)
    
    probe_thickness = 1
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            u = all_nodes[i]
            v = all_nodes[j]
            
            sp = u['px'] 
            ep = v['px'] 
            
            test_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(test_mask, sp, ep, 255, probe_thickness)
            
            overlap = cv2.bitwise_and(dilated_red, dilated_red, mask=test_mask)
            total_pixels = cv2.countNonZero(test_mask)
            red_pixels = cv2.countNonZero(overlap)
            
            if total_pixels == 0: continue
            
            ratio = red_pixels / total_pixels
            
            if ratio > tolerance:
                adj[u['id']].append(v['id'])
                adj[v['id']].append(u['id'])
    
    try:
        start_id = next(n['id'] for n in all_nodes if n['type'] == 'start')
    except StopIteration:
        return None, "Start node lost logic error"

    path_ids = [start_id]
    visited = {start_id}
    current_id = start_id
    
    while len(path_ids) < num_nodes:
        neighbors = adj[current_id]
        candidates = [nid for nid in neighbors if nid not in visited]
        
        if not candidates:
            return None, f"Broken path at node {all_nodes[current_id]['pos']}. Neighbors: {[all_nodes[n]['pos'] for n in neighbors]}"
        
        best_next = None
        min_dist_sq = float('inf')
        curr_px = all_nodes[current_id]['px']
        
        for cand_id in candidates:
            cand_px = all_nodes[cand_id]['px']
            d = dist_sq(curr_px, cand_px)
            if d < min_dist_sq:
                min_dist_sq = d
                best_next = cand_id
        
        current_id = best_next
        visited.add(current_id)
        path_ids.append(current_id)
    
    path_ids.append(start_id)

    coord_path = [all_nodes[pid]['pos'] for pid in path_ids]
    coord_strs = [f"({p[0]},{p[1]})" for p in coord_path]
    result_str = ",".join(coord_strs)
    
    return result_str, None

def main():
    parser = argparse.ArgumentParser(description="Parse TSP images using Snap-to-Grid logic.")
    parser.add_argument("dir", type=str, help="Directory containing PNG images")
    args = parser.parse_args()
    
    base_dir = Path(args.dir)
    if not base_dir.exists():
        print("Error: Directory does not exist.")
        return

    output_json = {}
    
    files = sorted(list(base_dir.glob("*.png")))
    valid_files = [f for f in files if "solution" not in f.name and "debug" not in f.name]
    
    print(f"Found {len(valid_files)} task images. Processing...")
    
    success_count = 0
    fail_count = 0
    
    for f in tqdm(valid_files):
        try:
            stem = f.stem
            parts = stem.split('_')
            
            if len(parts) < 1:
                print(f"Skipping malformed filename: {f.name}")
                continue
                
            grid_size = int(parts[0])
            
            path_str, error = parse_tsp_image(f, grid_size)
            
            if path_str:
                output_json[stem] = path_str
                success_count += 1
            else:
                print(f"Fail {f.name}: {error}")
                fail_count += 1
                
        except Exception as e:
            print(f"Exception on {f.name}: {e}")
            fail_count += 1

    out_file = base_dir / "0_result.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*30)
    print(f"Processing Complete")
    print(f"Success: {success_count}")
    print(f"Failed:  {fail_count}")
    print(f"Saved to: {out_file.resolve()}")
    print("="*30)

if __name__ == "__main__":
    main()
import os
from safetensors.torch import load_file, save_file

model_dir = "path/to/Qwen-Image-Edit-2511/text_encoder"
output_filename = "model_merged.safetensors"
output_path = os.path.join(model_dir, output_filename)

full_state_dict = {}


for i in range(1, 5):
    filename = f"model-0000{i}-of-00004.safetensors"
    file_path = os.path.join(model_dir, filename)
    
    if os.path.exists(file_path):
        print(f"({i}/4): {filename}")
        shard_dict = load_file(file_path, device="cpu")
        full_state_dict.update(shard_dict)
    else:
        print(f"no file {filename}")
        exit(1)

print("-" * 30)

save_file(full_state_dict, output_path)

print("success")
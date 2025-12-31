from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import os
from PIL import Image

num_inference_steps = 20
root_dir = "path/to/models/Qwen-Image-Edit"
save_folder = "path/to/DiffSynth-Studio/add/result"
lora_model_path= "path/to/models/Gen/Qwen-Image-Edit/FrozenLake/epoch-4.safetensors"
input_image_path = "path/to/input.png"

intermediate_folder = os.path.join(save_folder, "steps_visualization")
os.makedirs(intermediate_folder, exist_ok=True)

input_image = Image.open(input_image_path).convert("RGB")
input_image = input_image.resize((512, 512), resample=Image.NEAREST)
transformer_path = os.path.join(root_dir, "transformer", "diffusion_pytorch_model_merged.safetensors")
text_encoder_path = os.path.join(root_dir, "text_encoder", "model_merged.safetensors")
vae_path = os.path.join(root_dir, "vae", "diffusion_pytorch_model.safetensors")
processor_path = os.path.join(root_dir, "processor/")
tokenizer_path = os.path.join(root_dir, "tokenizer/")

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(path=transformer_path),
        ModelConfig(path=text_encoder_path),
        ModelConfig(path=vae_path),
    ],
    processor_config=ModelConfig(path=processor_path),
    tokenizer_config=ModelConfig(path=tokenizer_path),
)
pipe.load_lora(pipe.dit, lora_model_path, alpha=1)

prompt = "Draw a continuous red line connecting the Start point to the Goal point, avoiding all holes."

print("Starting inference...")
final_image, mixed_intermediates = pipe(
    prompt, 
    edit_image=input_image, 
    seed=1, 
    num_inference_steps=num_inference_steps, 
    height=512, 
    width=512,
    return_intermediates=True,
    edit_image_auto_resize=False
)

final_save_path = os.path.join(save_folder, "frozenlak.jpg")
final_image.save(final_save_path)
print(f"Final image saved to {final_save_path}")

print("Saving intermediate steps...")
step_count = 0

for item in mixed_intermediates:
    if isinstance(item, Image.Image):
        save_path = os.path.join(intermediate_folder, f"step_{step_count:02d}.png")
        item.save(save_path)
        print(f"Saved {save_path}")
        step_count += 1

print("Done.")
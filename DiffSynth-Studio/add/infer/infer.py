from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch
import os
from PIL import Image

num_inference_steps = 20
root_dir = "path/to/models/Qwen-Image-Edit"
save_folder = "path/to/DiffSynth-Studio/add/result"
lora_model_path= "path/to/models/Gen/Qwen-Image-Edit/Maze/epoch-4.safetensors"
input_image_path = "path/to/input.jpg"
input_image = Image.open(input_image_path).convert("RGB")

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
image = pipe(prompt, edit_image=input_image, seed=1, edit_image_auto_resize= False,num_inference_steps=num_inference_steps, height=512, width=512)
final_save_path = os.path.join(save_folder, "edit.jpg")
image.save(final_save_path)
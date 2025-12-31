NAME="FrozenLake" #Maze,Sudoku,TSP,Jigsaw
Data_Path="path/to/DiffThinker/${NAME}"
Model_Path="path/to/models"
Output_Path="${Model_Path}/Gen/Qwen-Image-Edit/${NAME}"

accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path "${Data_Path}" \
  --dataset_metadata_path "${Data_Path}/metadata_edit.csv" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_paths "${Model_Path}/Qwen-Image-Edit" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${Output_Path}" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --find_unused_parameters \
  --dataset_num_workers 8 >"path/to/DiffSynth-Studio/add/log/${NAME}.txt" 2>&1
export HF_HOME=/workspace/cachepath/.cache 

# include all subfolders in root dir
python train.py \
  --data_dir "<your_data_dir>"\
  --wandb_mode online --wandb_project sqlmm_main --wandb_run_name exp1 \
  --solver_model Qwen/Qwen2.5-VL-7B-Instruct \
  --proposer_model Qwen/Qwen2.5-VL-7B-Instruct \
  --use_lora_solver --use_lora_proposer \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --num_solver_samples 5 --proposer_update_freq 5 --total_steps 16180 \
  --kl_target 0.020 --kl_adapt_rate 0.10 \
  --solver_soft_gamma 0.7 \
  --clear_cache_every 10
export HF_HOME="HF_CACHE_PATH"
export HF_TOKEN="HUGGINGFACE_TOKEN"
export TRITON_CACHE_DIR="TRITON_DIR"
export OPENAI_API_KEY="OPENAI_API_KEY"



accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
  --model qwen2_5_vl_our \
  --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=<LoRA_WEIGHTS_PATH>,max_pixels=12845056,interleave_visuals=False \
  --tasks chartqa \
  --batch_size 1 \
  --output_path /workspace/lmms-eval/eval_results/exp1 \
  --use_cache /workspace/lmms-eval/eval_results/exp1
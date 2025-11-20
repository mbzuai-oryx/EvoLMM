# EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards

## Abstract
EvoLMM couples a **Proposer** and **Solver** built on the same vision-language backbone and trains them end-to-end with continuous, self-consistency rewards. The Proposer generates image-grounded questions while the Solver answers them; both are optimized via KL-regularized REINFORCE with adaptive baselines and lightweight LoRA adapters. The framework needs only raw images (no labels or external reward models) and delivers ~2–3% absolute gains on multimodal math/diagram reasoning benchmarks over the Qwen2.5-VL baseline.

## Links
<p>
  <a href="git_project_page.html" style="background:#f68946;color:#fff;padding:10px 14px;border-radius:6px;text-decoration:none;margin-right:10px;">Project Page</a>
  <a href="https://arxiv.org/abs/TBD" style="background:#008ad7;color:#fff;padding:10px 14px;border-radius:6px;text-decoration:none;">arXiv (coming soon)</a>
</p>

## Repository Layout
- `src/train.py`: core training loop, LoRA setup, adaptive KL, checkpoints, and logging.
- `src/train.sh`: example hyperparameters for Qwen2.5-VL-7B with LoRA.
- `Evaluation/lmms-eval`: evaluation harness (based on lmms-eval) with a ready-made script.
- `git_project_page.html`: project page containing full tables and figures.

## Setup
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Set cache paths/tokens, e.g.:
   ```bash
   export HF_HOME=/workspace/cache
   export HF_TOKEN=<your_hf_token>
   ```

## Data Preparation
Training only needs images (no annotations). By default the loader scans `images/train` and all first-level subfolders recursively. Expected layout:
```
images/
  train/
    split1/          # any subfolder names are accepted
      img_001.jpg
      ...
    split2/
      ...
```
- Use `--data_dir /path/to/images/train` to point to your root.
- To restrict to certain subfolders, pass `--include_subfolders=split1,split2`.
- Corrupted images are skipped; sampling is deterministic given `--seed`.

## Training
Baseline LoRA recipe (from `src/train.sh`) for Qwen2.5-VL-7B:
```bash
python src/train.py \
  --data_dir /path/to/images/train \
  --solver_model Qwen/Qwen2.5-VL-7B-Instruct \
  --proposer_model Qwen/Qwen2.5-VL-7B-Instruct \
  --use_lora_solver --use_lora_proposer \
  --lora_targets q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,mm_projector \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --num_solver_samples 5 --proposer_update_freq 5 --total_steps 16180 \
  --kl_target 0.020 --kl_adapt_rate 0.10 \
  --solver_soft_gamma 0.7 \
  --wandb_mode online --wandb_project sqlmm_main --wandb_run_name exp1 \
  --clear_cache_every 10
```
Notes:
- Set `--device`, `--dtype`, and `--device_map` for your hardware (defaults use CUDA if available).
- Checkpoints and per-iteration logs land in `runs/<run_name>/`.
- Adaptive resume is supported: keep `--wandb_run_name` fixed and checkpoints under `runs/` to auto-restore weights/optimizers/RNG.

## Evaluation
The evaluation harness in `Evaluation/lmms-eval` mirrors the training backbone. Example to evaluate a LoRA checkpoint on ChartQA:
```bash
cd Evaluation/lmms-eval
pip install -e .
export HF_HOME=/workspace/cache
export HF_TOKEN=<your_hf_token>

accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
  --model qwen2_5_vl_our \
  --model_args=pretrained=Qwen/Qwen2.5-VL-7B-Instruct,base_model=Qwen/Qwen2.5-VL-7B-Instruct,lora_path=/path/to/runs/exp1/step_xxxxx/solver,max_pixels=12845056,interleave_visuals=False \
  --tasks chartqa \
  --batch_size 1 \
  --output_path /workspace/lmms-eval/eval_results/exp1 \
```
Replace `lora_path` with the checkpoint directory you want to test. Additional tasks (MathVista, MathVision, etc.) are supported via `--tasks`.

## Results (Qwen2.5-VL-7B, zero labels)
| Model                                   | ChartQA | MathVista | MathVision | MathVerse | InfoGraphic-VQA<sub>val</sub> | AI2D  | ScienceQA | MMMU<sub>val</sub> |
|-----------------------------------------|:-------:|:---------:|:----------:|:---------:|:------------------------------:|:-----:|:---------:|:------------------:|
| Qwen2.5-VL-7B (baseline)                | 84.00   | 68.46     | 23.91      | 43.78     | 80.44                         | 82.61 | 88.30     | 51.11              |
| Qwen2.5-VL-7B + Discrete reward         | 84.62   | 68.88     | 22.52      | 42.10     | 80.52                         | 82.18 | 87.98     | 50.84              |
| **Qwen2.5-VL-7B + Continuous reward (EvoLMM)** | **86.70** | **70.52**   | **24.81**   | **44.88**   | **81.06**                     | **83.41** | **89.50**   | **52.01**          |

For additional ablations (LoRA vs. QLoRA/full fine-tune) and other backbones (InternVL3-8B, Gemma-3-12B, Llama-3.2-11B-Vision), see `arxiv` or the project website.
 
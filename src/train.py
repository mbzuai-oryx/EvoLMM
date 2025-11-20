import warnings
warnings.filterwarnings('ignore')

import re
import shutil
import pathlib
import os
import math
import time
import json
import random
import string
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
from contextlib import contextmanager
import argparse
import gc  # NEW: for CPU GC on long runs

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Optional deps
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedModel,
    set_seed,
)

# ----------------------------
# Defaults / Helpers
# ----------------------------
DEFAULT_LORA_TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "mm_projector",
)

def clip_grad_norm_multi_device(model: nn.Module, max_norm: float):
    by_dev: Dict[torch.device, List[nn.Parameter]] = {}
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            by_dev.setdefault(p.grad.device, []).append(p)
    for dev, params in by_dev.items():
        nn.utils.clip_grad_norm_(params, max_norm)

def _safe_dtype(dtype: str):
    return torch.bfloat16 if dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported() \
        else (torch.float16 if dtype == "float16" and torch.cuda.is_available() else torch.float32)

def strip_tags(text: str, tag: str) -> Optional[str]:
    lt, rt = f"<{tag}>", f"</{tag}>"
    if lt in text and rt in text:
        s = text.split(lt, 1)[1]
        s = s.split(rt, 1)[0]
        return s.strip()
    return None

def normalize_answer(ans: str) -> str:
    s = ans.strip().lower()
    s = s.replace(",", "")
    s = s.replace("\n", " ").replace("\t", " ")
    s = " ".join(s.split())
    return s.strip(string.punctuation + " ")

def majority_vote(answers: List[str]) -> Tuple[str, int]:
    counts: Dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    maj = max(counts.items(), key=lambda kv: kv[1])
    return maj[0], maj[1]

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def shannon_entropy_nats(probs: List[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(max(p, eps)) for p in probs if p > 0.0)

def pre_answer_word_count(text: str) -> int:
    idx = text.lower().find("<answer>")
    pre = text if idx == -1 else text[:idx]
    return len(pre.strip().split())

def gaussian_reward(x: float, mu: float, sigma: float) -> float:
    # NEW: bounded smooth reward around target mu
    if sigma <= 0:
        return 0.0
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma * sigma))


# ----------------------------
# Config
# ----------------------------
@dataclass
class Config:
    # Model ids
    solver_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    proposer_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Device / precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bfloat16"
    device_map: str = "auto"

    # Training
    total_steps: int = 100
    batch_size: int = 1
    lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    proposer_update_freq: int = 2
    kl_coef: float = 1e-3
    temp: float = 1.0
    top_p: float = 1.0
    max_new_tokens_solver: int = 128
    max_new_tokens_proposer: int = 128
    num_solver_samples: int = 5

    # Reward shaping (kept) + NEW params for Set A #1
    len_penalty_weight: float = 0.10
    len_penalty_target_words: int = 6
    # A1: solver softness exponent
    solver_soft_gamma: float = 0.7           # NEW: r = p(a)^gamma
    # A3: proposer Gaussian on entropy
    prop_entropy_mu: float = 0.90            # NEW: center for N=5
    prop_entropy_sigma: float = 0.35         # NEW: width

    # Adaptive KL
    kl_target: float = 0.02
    kl_adapt_rate: float = 0.10

    # Data / IO
    data_dir: str = "images/train"
    output_dir: str = "runs"
    save_every: int = 50
    max_checkpoints: int = 2
    include_subfolders: Optional[Tuple[str, ...]] = None

    # Optional GT evaluation:
    dataset: str = "chartqa"
    split: str = "validation"
    use_gt_eval: bool = False

    # Freezing
    freeze_vision: bool = True

    # Repro
    seed: int = 42

    # LoRA options
    use_lora_solver: bool = False
    use_lora_proposer: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGETS
    load_solver_adapter: Optional[str] = None
    load_proposer_adapter: Optional[str] = None
    start_step: int = 0

    # W&B options
    wandb_mode: str = "disabled"
    wandb_project: str = "sqlm"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_images_every: int = 0

    # OOM/fragmentation guard
    clear_cache_every: int = 25              # NEW: call empty_cache/ipc_collect every N outer steps


# ----------------------------
# Data loader (root -> subfolders -> images)
# ----------------------------
class ImagePool:
    DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.paths: List[str] = []

        root = os.path.abspath(cfg.data_dir)
        if not os.path.isdir(root):
            raise RuntimeError(f"[ImagePool] data_dir not found: {root}")

        # Determine which FIRST-LEVEL subfolders to scan
        if cfg.include_subfolders:
            # Exact-name filter
            chosen = []
            for name in cfg.include_subfolders:
                sub = os.path.join(root, name)
                if os.path.isdir(sub):
                    chosen.append((name, sub))
                else:
                    print(f"[ImagePool] WARNING: requested subfolder not found: {name}")
        else:
            # All first-level subfolders (skip hidden)
            chosen = []
            for name in sorted(os.listdir(root)):
                sub = os.path.join(root, name)
                if os.path.isdir(sub) and not name.startswith("."):
                    chosen.append((name, sub))

        # Fallback: if no subfolders matched, still look directly under root (just in case)
        if not chosen:
            print(f"[ImagePool] NOTE: No subfolders selected/found under {root}; "
                  f"falling back to scanning images directly under root.")
            chosen = [("", root)]

        # Walk each chosen subfolder recursively and collect images
        def _is_img(fn: str) -> bool:
            fnl = fn.lower()
            return fnl.endswith(self.DEFAULT_EXTS) and not os.path.basename(fnl).startswith(".")

        for sub_name, sub_path in chosen:
            for r, _dirs, files in os.walk(sub_path):
                for fn in files:
                    if _is_img(fn):
                        full = os.path.join(r, fn)
                        self.paths.append(full)

        if not self.paths:
            raise RuntimeError(f"[ImagePool] No images found under: {root} (subfolders={ [n for n,_ in chosen] })")

        self.paths.sort()
        print(f"[ImagePool] Found {len(self.paths)} images under: {root} "
              f"(subfolders={ [n for n,_ in chosen] })")

        # Deterministic permutation using cfg.seed
        self.indices = list(range(len(self.paths)))
        rnd = random.Random(cfg.seed)
        rnd.shuffle(self.indices)

        # Keep root to compute subfolder/relative path in meta
        self._root = root

    def __len__(self) -> int:
        return len(self.paths)

    def _build_meta(self, p: str) -> dict:
        # First-level subfolder name (relative to root)
        rel = os.path.relpath(p, self._root)
        parts = rel.split(os.sep)
        subfolder = parts[0] if len(parts) > 1 else ""
        return {
            "dataset": "folder",
            "split": "train",
            "path": p,                   # absolute path
            "rel_path": rel,             # path relative to root (includes subfolder)
            "subfolder": subfolder,      # first-level subfolder name
        }

    def sample(self) -> Tuple[Image.Image, dict]:
        p = random.choice(self.paths)
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            return self.sample()
        meta = self._build_meta(p)
        return img, meta

    # Deterministic sample by outer loop iteration (1-based 'iter_no' = step)
    def sample_by_iter(self, iter_no: int) -> Tuple[Image.Image, dict]:
        idx = self.indices[(max(1, int(iter_no)) - 1) % len(self.paths)]
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            # If corrupted, fall back to next entry deterministically
            return self.sample_by_iter(iter_no + 1)
        meta = self._build_meta(p)
        return img, meta


# ----------------------------
# Prompt templates
# ----------------------------
def build_proposer_prompt(meta: dict) -> str:
    prompt = f"""
                You are a Question Proposer.
                Given the IMAGE, generate **one** short-answer question
                that can be answered from the image alone (no external knowledge). Avoid ambiguity, avoid
                trivia, and avoid overly simple counting if possible.
                Rules:
                - Output exactly in XML with two tags:
                  <question> ... </question>
                  <rationale>Briefly explain why this question is non-trivial but solvable.</rationale>
                - Do NOT include the answer.
                - Keep the question 1–2 sentences, clear and specific.
                - If numeric, make sure units/context are clear.
                Only output the two XML tags, nothing else.
              """
    return prompt.strip()

def build_solver_prompt(question_text: str) -> str:
    prompt = f"""
                You are a precise Vision-Language Solver.
                Task: Answer the user's question using ONLY the provided IMAGE.
                - Think briefly, then provide a short final answer (number/word/phrase).
                - Wrap ONLY the final answer in <answer>...</answer>.
                - Do not include steps, just the final answer in the tag.
                Question: {question_text}
              """
    return prompt.strip()


# ----------------------------
# VLM core (shared backbone + adapters)  — (unchanged logic)
# ----------------------------
class VLMCore:
    def __init__(self, model_name: str, device: str, dtype: str, cfg: Config,
                 *, apply_lora: bool = False, make_adapters: Optional[List[str]] = None):
        self.device = device
        self.dtype = _safe_dtype(dtype)
        self.model_name = model_name
        self.cfg = cfg

        

        print(f"[Load] {model_name} on {device} ({self.dtype}), device_map={cfg.device_map}")
        self.model: PreTrainedModel = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=cfg.device_map,
            # attn_implementation="flash_attention_2"
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            self.tokenizer = getattr(self.processor, "tokenizer", None)

        if self.cfg.freeze_vision:
            for n, p in self.model.named_parameters():
                if "vision" in n.lower():
                    p.requires_grad_(False)

        self.is_lora = False
        self.adapter_names: List[str] = []

        def _apply_or_load_lora(adapter_name: str, load_path: Optional[str]):
            nonlocal targets
            if not HAS_PEFT:
                print("[LoRA] peft not installed; continuing without LoRA.")
                return False
            if load_path:
                from peft import PeftModel
                try:
                    self.model = PeftModel.from_pretrained(self.model, load_path)
                    if hasattr(self.model, "active_adapter") and self.model.active_adapter != adapter_name:
                        try:
                            self.model.load_adapter(load_path, adapter_name=adapter_name)
                        except Exception:
                            pass
                except Exception:
                    from peft import LoraConfig, get_peft_model, TaskType
                    lcfg = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                        lora_dropout=self.cfg.lora_dropout, target_modules=targets,
                    )
                    self.model = get_peft_model(self.model, lcfg)
                    self.model.load_adapter(load_path, adapter_name=adapter_name)
                self.adapter_names.append(adapter_name)
                return True
            else:
                from peft import LoraConfig, get_peft_model, TaskType
                lcfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=self.cfg.lora_r, lora_alpha=self.cfg.lora_alpha,
                    lora_dropout=self.cfg.lora_dropout, target_modules=targets,
                )
                if not hasattr(self.model, "peft_config"):
                    self.model = get_peft_model(self.model, lcfg)
                    self.adapter_names.append(adapter_name)
                    return True
                else:
                    try:
                        self.model.add_adapter(adapter_name, lcfg)
                        self.adapter_names.append(adapter_name)
                        return True
                    except Exception:
                        return False

        if apply_lora:
            targets = list(getattr(self.cfg, "lora_target_modules", DEFAULT_LORA_TARGETS))
            loaded_any = False
            loaded_any |= _apply_or_load_lora("default", getattr(self.cfg, "load_solver_adapter", None))
            if make_adapters:
                for name in make_adapters:
                    if name == "proposer":
                        loaded_any |= _apply_or_load_lora("proposer", getattr(self.cfg, "load_proposer_adapter", None))
                    elif name != "default":
                        loaded_any |= _apply_or_load_lora(name, None)
            if loaded_any:
                for n, p in self.model.named_parameters():
                    if "lora_" in n.lower():
                        p.requires_grad_(True)
                    else:
                        p.requires_grad_(False)
                self.is_lora = True
                try:
                    self.model.print_trainable_parameters()
                except Exception:
                    pass

        self.primary_device = self._infer_primary_device()
        self.device = self.primary_device

        dm = cfg.device_map
        if (dm is None) or (isinstance(dm, str) and dm.lower() == "cpu"):
            self.model.to(self.primary_device)

        self.model.eval()

    def _infer_primary_device(self) -> torch.device:
        dm = getattr(self.model, "hf_device_map", None)
        if isinstance(dm, dict):
            cuda_devs = [d for d in dm.values() if isinstance(d, str) and d.startswith("cuda")]
            if cuda_devs:
                try:
                    idx = min(int(d.split(":")[1]) for d in cuda_devs)
                    return torch.device(f"cuda:{idx}")
                except Exception:
                    pass
        try:
            return torch.device(self.cfg.device)
        except Exception:
            return torch.device("cpu")

    def _set_active_adapter(self, name: Optional[str]):
        if not HAS_PEFT: return
        if hasattr(self.model, "set_adapter") and name is not None:
            try: self.model.set_adapter(name)
            except Exception: pass

    def _disable_adapters(self):
        if not HAS_PEFT: return
        if hasattr(self.model, "disable_adapter"):
            try: self.model.disable_adapter()
            except Exception: pass

    @contextmanager
    def use_adapter(self, name: Optional[str]):
        prev = None
        if HAS_PEFT and hasattr(self.model, "active_adapter"):
            prev = getattr(self.model, "active_adapter", None)
        if name is None: self._disable_adapters()
        else: self._set_active_adapter(name)
        try: yield
        finally:
            if prev is None: self._disable_adapters()
            else: self._set_active_adapter(prev)

    def _render_chat(self, image: Image.Image, prompt: str, add_generation_prompt: bool) -> str:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
        }]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return chat_text

    @torch.no_grad()
    def generate(self, adapter: Optional[str], image: Image.Image, prompt: str,
                 max_new_tokens: int = 128, temperature: float = 1.0, top_p: float = 1.0) -> str:
        chat_text = self._render_chat(image, prompt, add_generation_prompt=True)
        inputs = self.processor(text=chat_text, images=[image], return_tensors="pt").to(self.primary_device)
        with self.use_adapter(adapter):
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=(self.tokenizer.eos_token_id if self.tokenizer is not None else None),
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = (self.tokenizer or self.processor.tokenizer).decode(gen_ids, skip_special_tokens=True)
        return text.strip()

    def logprob_of_completion(self, adapter: Optional[str], image: Image.Image, prompt: str, completion: str) -> float:
        self.model.train(False)
        prompt_chat = self._render_chat(image, prompt, add_generation_prompt=True)
        full_text = prompt_chat + completion
        with torch.no_grad():
            prompt_inputs = self.processor(text=prompt_chat, images=[image], return_tensors="pt").to(self.primary_device)
        inputs = self.processor(text=full_text, images=[image], return_tensors="pt").to(self.primary_device)
        input_ids = inputs["input_ids"]
        attn = inputs.get("attention_mask")
        labels = input_ids.clone()
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[:, :prompt_len] = -100
        with self.use_adapter(adapter), torch.enable_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attn, labels=labels)
            nll = outputs.loss.item()
        return -nll


# Thin role wrapper
class VLMRole:
    def __init__(self, core: VLMCore, adapter_name: Optional[str]):
        self.core = core
        self.adapter_name = adapter_name
        self.device = core.device
        self.processor = core.processor
        self.tokenizer = core.tokenizer
        self.model = core.model
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 128,
                 temperature: float = 1.0, top_p: float = 1.0) -> str:
        return self.core.generate(self.adapter_name, image, prompt, max_new_tokens, temperature, top_p)
    def logprob_of_completion(self, image: Image.Image, prompt: str, completion: str) -> float:
        return self.core.logprob_of_completion(self.adapter_name, image, prompt, completion)


# ----------------------------
# REINFORCE updater with token-level KL + Adaptive β
# ----------------------------
class PolicyUpdater:
    def __init__(self, policy: VLMRole, ref_policy: VLMRole, cfg: Config, *, adapter_name: Optional[str] = None):
        self.policy = policy
        self.ref_policy = ref_policy
        self.cfg = cfg
        self.kl_coef = cfg.kl_coef
        self._step = 0
        params = self._collect_trainable_params(self.policy.core.model, adapter_name)
        self.opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    @staticmethod
    def _collect_trainable_params(model: torch.nn.Module, adapter_name: Optional[str]) -> Iterable[nn.Parameter]:
        found, selected = False, []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if adapter_name is not None and f".{adapter_name}." in n:
                    selected.append(p); found = True
        if found: return selected
        return [p for p in model.parameters() if p.requires_grad]

    def _adapt_beta(self, kl_val: float):
        tgt = max(self.cfg.kl_target, 1e-8)
        delta = (kl_val - tgt) / tgt
        self.kl_coef = float(min(max(self.kl_coef * math.exp(self.cfg.kl_adapt_rate * delta), 1e-8), 1e2))

    def step(self, image: Image.Image, prompt: str, completion: str, reward: float, baseline: float = 0.0) -> Dict[str, float]:
        core = self.policy.core
        device = core.device
        self._step += 1

        chat_prompt = core._render_chat(image, prompt, add_generation_prompt=True)
        chat_full = chat_prompt + completion

        inputs_full = core.processor(text=chat_full, images=[image], return_tensors="pt").to(device)
        inputs_prompt = core.processor(text=chat_prompt, images=[image], return_tensors="pt").to(device)

        input_ids = inputs_full["input_ids"]; attn = inputs_full.get("attention_mask")
        labels = input_ids.clone()
        prompt_len = inputs_prompt["input_ids"].shape[1]
        labels[:, :prompt_len] = -100

        shift_labels = labels[:, 1:].contiguous()
        valid_mask = (shift_labels != -100)

        self.policy.core.model.train(True)
        with core.use_adapter(self.policy.adapter_name):
            out_pi = core.model(input_ids=input_ids, attention_mask=attn, labels=labels)
        ce_loss = out_pi.loss

        logp_pi = F.log_softmax(out_pi.logits, dim=-1)
        with torch.no_grad(), core.use_adapter(None):
            out_ref = core.model(input_ids=input_ids, attention_mask=attn)
            logp_ref = F.log_softmax(out_ref.logits, dim=-1)

        logp_pi_shift = logp_pi[:, :-1, :]
        logp_ref_shift = logp_ref[:, :-1, :]
        p_pi_shift = logp_pi_shift.exp()
        kl_per_tok = (p_pi_shift * (logp_pi_shift - logp_ref_shift)).sum(dim=-1)
        kl_loss = (kl_per_tok[valid_mask].mean()) if valid_mask.any() else torch.tensor(0.0, device=ce_loss.device)

        advantage = float(reward - baseline)
        beta_used = float(self.kl_coef)
        loss_total = (advantage * ce_loss) + (beta_used * kl_loss)

        self.opt.zero_grad(set_to_none=True)
        loss_total.backward()
        clip_grad_norm_multi_device(self.policy.core.model, self.cfg.grad_clip)
        self.opt.step()
        self.policy.core.model.train(False)

        # Adapt β
        kl_val = float(kl_loss.item())
        beta_before = beta_used
        self._adapt_beta(kl_val)

        # --- NEW: proactively release large tensors & clean up ---
        try:
            del inputs_full, inputs_prompt, input_ids, attn, labels
            del shift_labels, valid_mask, out_pi, logp_pi, out_ref, logp_ref
            del logp_pi_shift, logp_ref_shift, p_pi_shift, kl_per_tok
        except Exception:
            pass

        if torch.cuda.is_available() and (self.cfg.clear_cache_every > 0) and (self._step % self.cfg.clear_cache_every == 0):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()

        return {
            "ce_loss": float(ce_loss.item()),
            "kl_loss": kl_val,
            "advantage": advantage,
            "kl_coef_before": beta_before,
            "kl_coef_after": float(self.kl_coef),
            "loss_total": float(loss_total.item()),
        }


# ----------------------------
# Training loop
# ----------------------------
class SQLM_VLM_Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        random.seed(cfg.seed)

        share_backbone = (cfg.use_lora_solver and cfg.use_lora_proposer
                          and (cfg.solver_model_name == cfg.proposer_model_name))
        if share_backbone:
            print("[Models] Shared backbone with two adapters (solver=default, proposer=proposer).")
            core = VLMCore(cfg.solver_model_name, cfg.device, cfg.dtype, cfg,
                           apply_lora=True, make_adapters=["proposer"])
            self.solver = VLMRole(core, adapter_name="default")
            self.proposer = VLMRole(core, adapter_name="proposer")
            self.solver_ref = VLMRole(core, adapter_name=None)
            self.proposer_ref = VLMRole(core, adapter_name=None)
        else:
            print("[Models] Separate backbones (and frozen refs).")
            solver_core = VLMCore(cfg.solver_model_name, cfg.device, cfg.dtype, cfg, apply_lora=cfg.use_lora_solver)
            proposer_core = VLMCore(cfg.proposer_model_name, cfg.device, cfg.dtype, cfg, apply_lora=cfg.use_lora_proposer)
            self.solver = VLMRole(solver_core, adapter_name=("default" if solver_core.is_lora else None))
            self.proposer = VLMRole(proposer_core, adapter_name=("default" if proposer_core.is_lora else None))
            self.solver_ref = VLMRole(VLMCore(cfg.solver_model_name, cfg.device, cfg.dtype, cfg, apply_lora=False), adapter_name=None)
            self.proposer_ref = VLMRole(VLMCore(cfg.proposer_model_name, cfg.device, cfg.dtype, cfg, apply_lora=False), adapter_name=None)

        print("=" * 50)
        self.solver_updater = PolicyUpdater(self.solver, self.solver_ref, cfg, adapter_name=self.solver.adapter_name)
        self.proposer_updater = PolicyUpdater(self.proposer, self.proposer_ref, cfg, adapter_name=self.proposer.adapter_name)
        self.pool = ImagePool(cfg)

        self.solver_baseline = 0.0
        self.proposer_baseline = 0.0
        self.momentum = 0.9

        # Compute a stable run_dir once (shared with saver)
        self.run_name = cfg.wandb_run_name or f"{pathlib.Path(cfg.solver_model_name).name}_{int(time.time())}"
        self.run_dir = os.path.join(cfg.output_dir, self.run_name)
        os.makedirs(self.run_dir, exist_ok=True)

        # === NEW: resume training state (optimizers, β, baselines, RNG) ===
        resume_dir = getattr(cfg, "_resume_dir", None)
        if resume_dir and os.path.isdir(resume_dir):
            state_path = os.path.join(resume_dir, "trainer_state.pt")
            if os.path.isfile(state_path):
                try:
                    state = torch.load(state_path, map_location="cpu")
                    # Optimizers
                    if "solver_opt" in state:
                        self.solver_updater.opt.load_state_dict(state["solver_opt"])
                    if "proposer_opt" in state:
                        self.proposer_updater.opt.load_state_dict(state["proposer_opt"])
                    # KL β
                    if "solver_kl_coef" in state:
                        self.solver_updater.kl_coef = float(state["solver_kl_coef"])
                    if "proposer_kl_coef" in state:
                        self.proposer_updater.kl_coef = float(state["proposer_kl_coef"])
                    # Baselines
                    self.solver_baseline = float(state.get("solver_baseline", 0.0))
                    self.proposer_baseline = float(state.get("proposer_baseline", 0.0))
                    # Updater internal step counters
                    self.solver_updater._step = int(state.get("solver_updater_step", cfg.start_step))
                    self.proposer_updater._step = int(state.get("proposer_updater_step", cfg.start_step))
                    # RNG states
                    if "py_random_state" in state:
                        random.setstate(state["py_random_state"])
                    if "torch_rng_state" in state:
                        torch.set_rng_state(state["torch_rng_state"])
                    if torch.cuda.is_available() and ("torch_cuda_rng_state_all" in state):
                        try:
                            torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])
                        except Exception:
                            pass
                    print(f"[Resume] Loaded trainer state from: {state_path}")
                except Exception as e:
                    print(f"[Resume] WARNING: failed to load trainer state: {e}")

        self.wandb_run = None
        if HAS_WANDB and cfg.wandb_mode != "disabled":
            run_name = cfg.wandb_run_name or f"sqlm_vlm_{int(time.time())}"
            self.wandb_run = wandb.init(
                project=cfg.wandb_project, entity=cfg.wandb_entity,
                name=run_name, mode=cfg.wandb_mode, config=dataclasses.asdict(cfg),
            )
            try: wandb.watch(self.solver.core.model, log="gradients", log_freq=100)
            except Exception: pass

    def _append_iter_log(self, record: Dict[str, object]):
        """NEW: Append a one-line JSON record per iteration for auditability."""
        try:
            log_path = os.path.join(self.run_dir, "iter_log.jsonl")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[IterLog] WARNING: failed to append log: {e}")

    def _save_checkpoint(self, step: int):
        """Save adapters (if using LoRA) or full weights, plus tokenizer/processor, then prune to last K.
           NEW: also save optimizer states, KL β, baselines, RNG to trainer_state.pt
        """
        cfg = self.cfg

        # Each experiment under: runs/<wandb_run_name or fallback>/step_<NNNNN>/
        run_dir = self.run_dir
        os.makedirs(run_dir, exist_ok=True)

        # Atomic write: save to tmp then rename to final step dir
        final_dir = os.path.join(run_dir, f"step_{step:05d}")
        tmp_dir = final_dir + ".tmp"
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        def _save_core(core: VLMCore, subdir: str):
            sub = os.path.join(tmp_dir, subdir)
            os.makedirs(sub, exist_ok=True)

            # Save tokenizer/processor
            try:
                if core.tokenizer is not None:
                    core.tokenizer.save_pretrained(sub)
            except Exception:
                pass
            try:
                if core.processor is not None:
                    core.processor.save_pretrained(sub)
            except Exception:
                pass

            # Save adapters or full model
            try:
                if core.is_lora:
                    # Save ALL adapters on this model (solver/proposer if shared)
                    core.model.save_pretrained(sub, save_adapter=True)
                else:
                    core.model.save_pretrained(sub)
            except Exception:
                # Fallback: raw state_dict
                torch.save(core.model.state_dict(), os.path.join(sub, "pytorch_model.bin"))

            # Metadata
            meta = {
                "model_name": core.model_name,
                "is_lora": core.is_lora,
                "adapter_names": getattr(core, "adapter_names", []),
                "device_map": cfg.device_map,
                "dtype": str(core.dtype),
                "step": step,
                "time": int(time.time()),
            }
            with open(os.path.join(sub, "checkpoint_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

        # Save solver / proposer cores
        _save_core(self.solver.core, "solver")
        if self.proposer.core is not self.solver.core:
            _save_core(self.proposer.core, "proposer")
        else:
            # Shared backbone marker (optional)
            with open(os.path.join(tmp_dir, "SHARED_BACKBONE.txt"), "w") as f:
                f.write("Adapters 'default'(solver) and 'proposer' are in this checkpoint.\n")

        # === NEW: Save trainer state (optimizers, KL β, baselines, RNG) ===
        trainer_state = {
            "step": step,
            "solver_opt": self.solver_updater.opt.state_dict(),
            "proposer_opt": self.proposer_updater.opt.state_dict(),
            "solver_kl_coef": float(self.solver_updater.kl_coef),
            "proposer_kl_coef": float(self.proposer_updater.kl_coef),
            "solver_baseline": float(self.solver_baseline),
            "proposer_baseline": float(self.proposer_baseline),
            "solver_updater_step": int(self.solver_updater._step),
            "proposer_updater_step": int(self.proposer_updater._step),
            "py_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            try:
                trainer_state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        torch.save(trainer_state, os.path.join(tmp_dir, "trainer_state.pt"))

        # Write completion marker and atomically publish the folder
        with open(os.path.join(tmp_dir, "SAVE_OK"), "w") as f:
            f.write("ok\n")

        try:
            os.replace(tmp_dir, final_dir)   # atomic rename on same filesystem
        except Exception:
            # If rename failed, keep tmp dir and log
            print(f"[Checkpoint] WARNING: atomic rename failed; keeping {tmp_dir}")
            return

        print(f"[Checkpoint] Saved: {os.path.basename(final_dir)}")

        # Retain only last K (≥2) complete checkpoints for this run
        try:
            _retain_last_k_checkpoints(run_dir, k=self.cfg.max_checkpoints)
        except Exception as e:
            print(f"[Checkpoint] Retention skipped due to error: {e}")
    
    def update_baseline(self, which: str, reward: float):
        if which == "solver":
            self.solver_baseline = self.momentum * self.solver_baseline + (1 - self.momentum) * reward
        else:
            self.proposer_baseline = self.momentum * self.proposer_baseline + (1 - self.momentum) * reward

    def _wandb_log_step(
        self,
        step: int,
        image: Optional[Image.Image],
        maj_count: int,
        solver_rewards_raw: List[float],
        solver_rewards_soft: List[float],       # CHANGED: log soft rewards
        proposer_reward: float,                  # CHANGED: continuous proposer reward
        question: str,
        solver_answers_raw: List[str],
        maj_answer: str,
        proposer_out: str,
        solver_stats: Optional[Dict[str, float]],
        proposer_stats: Optional[Dict[str, float]],
        entropy_nats: float,
        maj_frac: float,
        pre_words_mean: float,
        ans_hist: Dict[str, int],
    ):
        if self.wandb_run is None:
            return
        metrics: Dict[str, object] = {
            "train/step": step,
            "train/maj_count": maj_count,
            "train/maj_frac": maj_frac,
            "train/num_solver_samples": self.cfg.num_solver_samples,
            "train/solver_reward_mean_raw": sum(solver_rewards_raw) / max(1, len(solver_rewards_raw)),
            "train/solver_reward_mean_soft": sum(solver_rewards_soft) / max(1, len(solver_rewards_soft)),
            "train/proposer_reward_gauss": proposer_reward,
            "train/solver_baseline": self.solver_baseline,
            "train/proposer_baseline": self.proposer_baseline,
            "train/entropy_nats": entropy_nats,
            "train/pre_words_mean": pre_words_mean,
            "text/question": question,
            "text/maj_answer": maj_answer,
            "text/solver_answers": ", ".join(solver_answers_raw),
            "text/proposer_out": proposer_out,
            "text/answer_hist": json.dumps(ans_hist, ensure_ascii=False),
        }
        if solver_stats:
            metrics.update({
                "solver/ce_loss_mean": solver_stats.get("ce_loss_mean"),
                "solver/kl_loss_mean": solver_stats.get("kl_loss_mean"),
                "solver/advantage_mean": solver_stats.get("advantage_mean"),
                "solver/kl_coef": solver_stats.get("kl_coef_after"),
            })
        if proposer_stats:
            metrics.update({
                "proposer/ce_loss": proposer_stats.get("ce_loss"),
                "proposer/kl_loss": proposer_stats.get("kl_loss"),
                "proposer/advantage": proposer_stats.get("advantage"),
                "proposer/kl_coef": proposer_stats.get("kl_coef_after"),
            })

        # NEW: clearer KL β labels and data path
        metrics["kl/solver_beta"] = (solver_stats.get("kl_coef_after") if solver_stats else self.solver_updater.kl_coef)
        metrics["kl/proposer_beta"] = (proposer_stats.get("kl_coef_after") if proposer_stats else self.proposer_updater.kl_coef)
        metrics["data/image_path"] = getattr(self, "_last_image_path", None)

        if self.cfg.wandb_log_images_every > 0 and (step % self.cfg.wandb_log_images_every) == 0 and image is not None:
            try: metrics["vis/image"] = wandb.Image(image, caption=f"Step {step}")
            except Exception: pass
        wandb.log(metrics, step=step)

    def maybe_empty_cache(self, step: int):
        # NEW: periodic cleanup (GPU + CPU) to mitigate fragmentation/OOM
        if torch.cuda.is_available() and self.cfg.clear_cache_every > 0 and (step % self.cfg.clear_cache_every) == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try: torch.cuda.ipc_collect()
            except Exception: pass
            gc.collect()

    def train(self):
        cfg = self.cfg
        print(f"Starting training for {cfg.total_steps} steps.")
        for step in range(cfg.start_step + 1, cfg.total_steps + 1):
            # Deterministic data selection per-step
            image, meta = self.pool.sample_by_iter(step)
            self._last_image_path = meta.get("path")

            # 1) PROPOSER question
            proposer_prompt = build_proposer_prompt(meta)
            proposer_out = self.proposer.generate(
                image=image, prompt=proposer_prompt,
                max_new_tokens=cfg.max_new_tokens_proposer, temperature=cfg.temp, top_p=cfg.top_p
            )
            question = strip_tags(proposer_out, "question") or proposer_out.strip()
            question = question.replace("\n", " ").strip()
            if not question:
                question = "What is the most prominent value shown in the image?"

            # 2) SOLVER answers N times
            solver_answers_raw, solver_answers_norm, solver_completions = [], [], []
            pre_words_list: List[int] = []
            solver_prompt = build_solver_prompt(question)
            for _ in range(cfg.num_solver_samples):
                sol_out = self.solver.generate(
                    image=image, prompt=solver_prompt,
                    max_new_tokens=cfg.max_new_tokens_solver,
                    temperature=cfg.temp, top_p=cfg.top_p
                )
                ans = strip_tags(sol_out, "answer")
                if ans is None:
                    lines = [ln.strip() for ln in sol_out.strip().splitlines() if ln.strip()]
                    ans = lines[-1] if lines else "unknown"
                ans_norm = normalize_answer(ans)
                solver_answers_raw.append(ans)
                solver_answers_norm.append(ans_norm)
                solver_completions.append(sol_out)
                pre_words_list.append(pre_answer_word_count(sol_out))

            maj_answer, maj_count = majority_vote(solver_answers_norm)
            maj_frac = maj_count / float(cfg.num_solver_samples)

            # Histogram + entropy
            hist: Dict[str, int] = {}
            for a in solver_answers_norm:
                hist[a] = hist.get(a, 0) + 1
            probs = [c / float(cfg.num_solver_samples) for c in hist.values()]
            entropy_nats = shannon_entropy_nats(probs)

            # --- 3) REWARDS (Set A #1) ---
            # Solver RAW (for logging only)
            solver_rewards_raw = [1.0 if a == maj_answer else 0.0 for a in solver_answers_norm]
            # Solver SOFT: p(answer)^gamma with length penalty
            target_w = max(1, cfg.len_penalty_target_words)
            penalties = [min(1.0, max(0.0, (pw - target_w) / float(target_w))) for pw in pre_words_list]
            # per-sample p(answer)
            prob_map = {ans: count / float(cfg.num_solver_samples) for ans, count in hist.items()}
            solver_probs_per_sample = [prob_map[a] for a in solver_answers_norm]
            solver_rewards_soft = [
                (p ** cfg.solver_soft_gamma) * (1.0 - cfg.len_penalty_weight * pen)
                for p, pen in zip(solver_probs_per_sample, penalties)
            ]
            # Proposer: Gaussian on entropy
            proposer_reward = gaussian_reward(entropy_nats, cfg.prop_entropy_mu, cfg.prop_entropy_sigma)

            # 4) Update solver (use SOFT reward!)
            solver_stats_list = []
            for sol_out, r_soft in zip(solver_completions, solver_rewards_soft):
                stats = self.solver_updater.step(
                    image=image, prompt=solver_prompt, completion=sol_out,
                    reward=r_soft, baseline=self.solver_baseline
                )
                solver_stats_list.append(stats)
                self.update_baseline("solver", r_soft)

            solver_stats = None
            if solver_stats_list:
                ce_mean = sum(s["ce_loss"] for s in solver_stats_list) / len(solver_stats_list)
                kl_mean = sum(s["kl_loss"] for s in solver_stats_list) / len(solver_stats_list)
                adv_mean = sum(s["advantage"] for s in solver_stats_list) / len(solver_stats_list)
                solver_stats = {
                    "ce_loss_mean": ce_mean,
                    "kl_loss_mean": kl_mean,
                    "advantage_mean": adv_mean,
                    "kl_coef_after": solver_stats_list[-1]["kl_coef_after"],
                }

            # 5) Update proposer (every k steps) — reward is continuous now
            proposer_stats = None
            if (step % cfg.proposer_update_freq) == 0:
                proposer_stats = self.proposer_updater.step(
                    image=image, prompt=proposer_prompt, completion=proposer_out,
                    reward=proposer_reward, baseline=self.proposer_baseline
                )
                self.update_baseline("proposer", proposer_reward)

            # 6) Logging
            pre_words_mean = sum(pre_words_list) / max(1, len(pre_words_list))
            print(f"[Step {step:03d}] maj={maj_count}/{cfg.num_solver_samples} "
                  f"maj_frac={maj_frac:.2f} H={entropy_nats:.3f} "
                  f"proposer_r_gauss={proposer_reward:.2f} "
                  f"solver_r_raw={sum(solver_rewards_raw)/len(solver_rewards_raw):.2f} "
                  f"solver_r_soft={sum(solver_rewards_soft)/len(solver_rewards_soft):.2f} "
                  f"pre_words={pre_words_mean:.1f}")
            print(f"Q: {question}")
            print(f"A: [{', '.join(solver_answers_raw)}]  | MAJ: {maj_answer}")

            self._wandb_log_step(
                step=step, image=image, maj_count=maj_count,
                solver_rewards_raw=solver_rewards_raw,
                solver_rewards_soft=solver_rewards_soft,
                proposer_reward=proposer_reward,
                question=question,
                solver_answers_raw=solver_answers_raw,
                maj_answer=maj_answer,
                proposer_out=proposer_out,
                solver_stats=solver_stats,
                proposer_stats=proposer_stats,
                entropy_nats=entropy_nats,
                maj_frac=maj_frac,
                pre_words_mean=pre_words_mean,
                ans_hist=hist,
            )

            # NEW: Append per-iteration JSONL log
            self._append_iter_log({
                "step": step,
                "image_path": meta.get("path"),
                "proposer_question": question,
                "proposer_out_xml": proposer_out,
                "solver_answers_raw": solver_answers_raw,
                "solver_answers_norm": solver_answers_norm,
                "solver_rewards_raw": solver_rewards_raw,
                "solver_rewards_soft": solver_rewards_soft,
                "maj_answer": maj_answer,
                "maj_count": maj_count,
                "maj_frac": maj_frac,
                "entropy_nats": entropy_nats,
                "proposer_reward_gauss": proposer_reward,
                "pre_answer_word_counts": pre_words_list,
                "answer_hist": hist,
            })

            # 7) Periodic memory cleanup (GPU + CPU)
            self.maybe_empty_cache(step)

            # 8) Save weights (adapters if LoRA) + trainer state periodically
            if self.cfg.save_every and (step % self.cfg.save_every) == 0:
                self._save_checkpoint(step)

        if self.wandb_run is not None:
            try: wandb.finish()
            except Exception: pass


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("SQLM VLM (self-evolving) — Set A #1 continuous rewards")

    # Data
    p.add_argument("--data_dir", type=str, default=os.environ.get("DATA_DIR", "images/train"),
                   help="Folder (recursive) containing all training images.")
    p.add_argument(
        "--include_subfolders",
        type=str,
        default=os.environ.get("INCLUDE_SUBFOLDERS", None),
        help="Comma-separated list of FIRST-LEVEL subfolder names under --data_dir to include. "
             "Default: load ALL subfolders."
    )

    # Models
    p.add_argument("--solver_model", type=str, default=os.environ.get("SOLVER_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))
    p.add_argument("--proposer_model", type=str, default=os.environ.get("PROPOSER_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct"))

    # Device / precision
    p.add_argument("--device", type=str, default=os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument("--dtype", type=str, default=os.environ.get("DTYPE", "bfloat16"), choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device_map", type=str, default=os.environ.get("DEVICE_MAP", "auto"),
                   help="HF device map (e.g., auto, balanced, sequential, cpu)")

    # Training
    p.add_argument("--total_steps", type=int, default=int(os.environ.get("TOTAL_STEPS", "100")))
    p.add_argument("--proposer_update_freq", type=int, default=int(os.environ.get("PROP_FREQ", "5")))
    p.add_argument("--num_solver_samples", type=int, default=int(os.environ.get("N_SAMPLES", "5")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-6")))
    p.add_argument("--weight_decay", type=float, default=float(os.environ.get("WEIGHT_DECAY", "0.01")))
    p.add_argument("--grad_clip", type=float, default=float(os.environ.get("GRAD_CLIP", "1.0")))
    p.add_argument("--temp", type=float, default=float(os.environ.get("TEMP", "1.0")))
    p.add_argument("--top_p", type=float, default=float(os.environ.get("TOP_P", "1.0")))
    p.add_argument("--max_new_tokens_solver", type=int, default=int(os.environ.get("MAX_NEW_TOKENS_SOLVER", "128")))
    p.add_argument("--max_new_tokens_proposer", type=int, default=int(os.environ.get("MAX_NEW_TOKENS_PROPOSER", "128")))
    p.add_argument("--freeze_vision", action="store_true", default=(os.environ.get("FREEZE_VISION", "1") != "0"))
    p.add_argument("--no-freeze-vision", dest="freeze_vision", action="store_false")

    # Reward shaping
    p.add_argument("--len_penalty_weight", type=float, default=float(os.environ.get("LEN_PENALTY_WEIGHT", "0.10")))
    p.add_argument("--len_penalty_target_words", type=int, default=int(os.environ.get("LEN_PENALTY_TARGET_WORDS", "6")))
    p.add_argument("--solver_soft_gamma", type=float, default=float(os.environ.get("SOLVER_SOFT_GAMMA", "0.7")))
    p.add_argument("--prop_entropy_mu", type=float, default=float(os.environ.get("PROP_ENTROPY_MU", "0.90")))
    p.add_argument("--prop_entropy_sigma", type=float, default=float(os.environ.get("PROP_ENTROPY_SIGMA", "0.35")))

    # Adaptive KL
    p.add_argument("--kl_target", type=float, default=float(os.environ.get("KL_TARGET", "0.02")))
    p.add_argument("--kl_adapt_rate", type=float, default=float(os.environ.get("KL_ADAPT_RATE", "0.10")))
    p.add_argument("--kl_coef", type=float, default=float(os.environ.get("KL_COEF", "1e-3")))

    # Repro
    p.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    p.add_argument("--output_dir", type=str, default=os.environ.get("OUTPUT_DIR", "runs"))
    p.add_argument("--save_every", type=int, default=int(os.environ.get("SAVE_EVERY", "50")))
    p.add_argument("--max_checkpoints", type=int, default=int(os.environ.get("MAX_CKPTS", "2")),
                help="Keep only the latest K checkpoints per run (min=2).")

    # LoRA
    p.add_argument("--use_lora_solver", action="store_true", default=(os.environ.get("USE_LORA_SOLVER", "0") == "1"))
    p.add_argument("--use_lora_proposer", action="store_true", default=(os.environ.get("USE_LORA_PROPOSER", "0") == "1"))
    p.add_argument("--lora_r", type=int, default=int(os.environ.get("LORA_R", "16")))
    p.add_argument("--lora_alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32")))
    p.add_argument("--lora_dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", "0.05")))
    p.add_argument("--lora_targets", type=str,
                   default=os.environ.get("LORA_TARGETS", ",".join(DEFAULT_LORA_TARGETS)),
                   help="Comma-separated target module names for LoRA (include 'mm_projector' for vision-language)")

    # W&B
    p.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "disabled"),
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "sqlm"))
    p.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY", None))
    p.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME", None))
    p.add_argument("--wandb_log_images_every", type=int, default=int(os.environ.get("WANDB_LOG_IMAGES_EVERY", "0")))

    # OOM guard
    p.add_argument("--clear_cache_every", type=int, default=int(os.environ.get("CLEAR_CACHE_EVERY", "25")),
                   help="Every N outer steps, call CUDA empty_cache/ipc_collect and gc.collect")

    return p.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    lora_targets = tuple([s.strip() for s in (args.lora_targets or "").split(",") if s.strip()]) or DEFAULT_LORA_TARGETS
    include_subfolders = None
    if args.include_subfolders:
        parsed = [s.strip() for s in args.include_subfolders.split(",") if s.strip()]
        include_subfolders = tuple(parsed) if parsed else None
    cfg = Config(
        solver_model_name=args.solver_model,
        proposer_model_name=args.proposer_model,
        device=args.device, dtype=args.dtype, device_map=args.device_map,
        total_steps=args.total_steps, proposer_update_freq=args.proposer_update_freq,
        num_solver_samples=args.num_solver_samples, lr=args.lr, weight_decay=args.weight_decay,
        grad_clip=args.grad_clip, temp=args.temp, top_p=args.top_p,
        max_new_tokens_solver=args.max_new_tokens_solver, max_new_tokens_proposer=args.max_new_tokens_proposer,
        data_dir=args.data_dir, freeze_vision=args.freeze_vision, seed=args.seed,
        use_lora_solver=args.use_lora_solver, use_lora_proposer=args.use_lora_proposer,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        lora_target_modules=lora_targets, wandb_mode=args.wandb_mode, wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity, wandb_run_name=args.wandb_run_name, wandb_log_images_every=args.wandb_log_images_every,
        len_penalty_weight=args.len_penalty_weight, len_penalty_target_words=args.len_penalty_target_words,
        solver_soft_gamma=args.solver_soft_gamma, prop_entropy_mu=args.prop_entropy_mu, prop_entropy_sigma=args.prop_entropy_sigma,
        kl_target=args.kl_target, kl_adapt_rate=args.kl_adapt_rate, kl_coef=args.kl_coef,
        clear_cache_every=args.clear_cache_every,
        output_dir=args.output_dir,
        save_every=args.save_every,
        max_checkpoints=max(2, int(args.max_checkpoints)),
        include_subfolders=include_subfolders
    )
    return cfg

def _parse_step_num(name: str) -> int:
    m = re.match(r"^step_(\d+)$", name)
    return int(m.group(1)) if m else -1

def _is_complete_ckpt(step_dir: str) -> bool:
    """A 'complete' checkpoint has SAVE_OK and a solver/ subdir with metadata or weights."""
    if not os.path.isdir(step_dir):
        return False
    if not os.path.isfile(os.path.join(step_dir, "SAVE_OK")):
        return False
    solver_dir = os.path.join(step_dir, "solver")
    if not os.path.isdir(solver_dir):
        return False
    # meta or weight presence
    has_meta = os.path.isfile(os.path.join(solver_dir, "checkpoint_meta.json"))
    has_any_weight = any(os.path.isfile(os.path.join(solver_dir, f))
                         for f in ("adapter_config.json", "adapter_model.bin",
                                   "adapter_model.safetensors",
                                   "pytorch_model.bin", "model.safetensors",
                                   "config.json"))
    return has_meta or has_any_weight

def _list_valid_ckpts(run_dir: str) -> list[tuple[int, str]]:
    """Return sorted [(step, path), ...] of complete step_* directories."""
    if not os.path.isdir(run_dir):
        return []
    pairs = []
    for d in os.listdir(run_dir):
        step = _parse_step_num(d)
        if step >= 0:
            full = os.path.join(run_dir, d)
            if _is_complete_ckpt(full):
                pairs.append((step, full))
    return sorted(pairs, key=lambda x: x[0])

def _retain_last_k_checkpoints(run_dir: str, k: int = 2):
    """Keep only the last k complete checkpoints; delete older ones."""
    k = max(2, int(k))
    ckpts = _list_valid_ckpts(run_dir)
    if len(ckpts) <= k:
        return
    to_delete = ckpts[:-k]  # all but the last k
    for step, path in to_delete:
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"[Checkpoint] Pruned: {os.path.basename(path)}")
        except Exception as e:
            print(f"[Checkpoint] Prune failed for {path}: {e}")

def _is_valid_step_dir(step_dir: str) -> bool:
    """Heuristic: step dir must have a solver/ subdir and either meta or weights."""
    solver_dir = os.path.join(step_dir, "solver")
    if not os.path.isdir(solver_dir):
        return False
    # meta or weight presence
    meta = os.path.join(solver_dir, "checkpoint_meta.json")
    has_meta = os.path.isfile(meta)
    has_weights = any(os.path.isfile(os.path.join(solver_dir, f))
                      for f in ("adapter_config.json", "adapter_model.bin",
                                "adapter_model.safetensors", "pytorch_model.bin",
                                "model.safetensors", "config.json"))
    return has_meta or has_weights

def _parse_step_num(name: str) -> int:
    m = re.match(r"step_(\d+)$", name)
    return int(m.group(1)) if m else -1

def _find_preferred_ckpt(run_dir: str) -> tuple[int, str] | tuple[None, None]:
    """
    Return (step_num, step_dir) where:
      - if there are ≥2 complete checkpoints: the 2nd newest
      - else if there is exactly 1 complete checkpoint: that one
      - else: (None, None)
    Uses 'complete' definition (SAVE_OK + solver/ present with weights/meta).
    """
    ckpts = _list_valid_ckpts(run_dir)  # sorted [(step, path), ...] ascending
    if not ckpts:
        return (None, None)
    if len(ckpts) >= 2:
        return ckpts[-2]  # second newest
    return ckpts[-1]      # only one: return it

def _read_meta_is_lora(dir_path: str) -> bool | None:
    meta_path = os.path.join(dir_path, "checkpoint_meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return bool(meta.get("is_lora"))
    except Exception:
        return None

def _prune_other_checkpoints(run_dir: str, keep_step: int):
    """Delete all step_* except the chosen one."""
    keep_name = f"step_{keep_step:05d}"
    for d in os.listdir(run_dir):
        if d.startswith("step_") and d != keep_name:
            shutil.rmtree(os.path.join(run_dir, d), ignore_errors=True)

def maybe_autoresume(cfg):
    run_name = cfg.wandb_run_name
    if not run_name:
        return cfg

    run_dir = os.path.join(cfg.output_dir, run_name)
    step, step_dir = _find_preferred_ckpt(run_dir)
    if step is None:
        return cfg

    solver_dir = os.path.join(step_dir, "solver")
    proposer_dir = os.path.join(step_dir, "proposer")
    if not os.path.isdir(proposer_dir):
        proposer_dir = solver_dir  # shared backbone case

    is_lora = _read_meta_is_lora(solver_dir)
    print(f"[Auto-Resume] Using checkpoint: {step_dir} (is_lora={is_lora})")
    cfg.start_step = int(step)

    if is_lora is True:
        if not getattr(cfg, "use_lora_solver", False):
            print("[Auto-Resume] Forcing --use_lora_solver to match checkpoint.")
            cfg.use_lora_solver = True
        if not getattr(cfg, "use_lora_proposer", False):
            print("[Auto-Resume] Forcing --use_lora_proposer to match checkpoint.")
            cfg.use_lora_proposer = True
        cfg.load_solver_adapter = solver_dir
        cfg.load_proposer_adapter = proposer_dir
    elif is_lora is False:
        cfg.solver_model_name = solver_dir
        cfg.proposer_model_name = proposer_dir
        cfg.use_lora_solver = False
        cfg.use_lora_proposer = False
        cfg.load_solver_adapter = None
        cfg.load_proposer_adapter = None
    else:
        print("[Auto-Resume] Unknown LoRA/full; leaving cfg flags as-is.")

    try:
        _prune_other_checkpoints(run_dir, keep_step=cfg.start_step)
        print(f"[Auto-Resume] Pruned others; kept step_{cfg.start_step:05d}")
    except Exception as e:
        print(f"[Auto-Resume] Prune skipped: {e}")

    # NEW: carry the step_dir for the trainer to load optimizer/RNG state
    cfg._resume_dir = step_dir

    return cfg

def _parse_step_num(name: str) -> int:
    m = re.match(r"^step_(\d+)$", name)
    return int(m.group(1)) if m else -1

def _is_complete_ckpt(step_dir: str) -> bool:
    """A 'complete' checkpoint has SAVE_OK and a solver/ subdir with metadata or weights."""
    if not os.path.isdir(step_dir):
        return False
    if not os.path.isfile(os.path.join(step_dir, "SAVE_OK")):
        return False
    solver_dir = os.path.join(step_dir, "solver")
    if not os.path.isdir(solver_dir):
        return False
    # meta or weight presence
    has_meta = os.path.isfile(os.path.join(solver_dir, "checkpoint_meta.json"))
    has_any_weight = any(os.path.isfile(os.path.join(solver_dir, f))
                         for f in ("adapter_config.json", "adapter_model.bin",
                                   "adapter_model.safetensors",
                                   "pytorch_model.bin", "model.safetensors",
                                   "config.json"))
    return has_meta or has_any_weight

def _list_valid_ckpts(run_dir: str) -> list[tuple[int, str]]:
    """Return sorted [(step, path), ...] of complete step_* directories."""
    if not os.path.isdir(run_dir):
        return []
    pairs = []
    for d in os.listdir(run_dir):
        step = _parse_step_num(d)
        if step >= 0:
            full = os.path.join(run_dir, d)
            if _is_complete_ckpt(full):
                pairs.append((step, full))
    return sorted(pairs, key=lambda x: x[0])

def _retain_last_k_checkpoints(run_dir: str, k: int = 2):
    """Keep only the last k complete checkpoints; delete older ones."""
    k = max(2, int(k))
    ckpts = _list_valid_ckpts(run_dir)
    if len(ckpts) <= k:
        return
    to_delete = ckpts[:-k]  # all but the last k
    for step, path in to_delete:
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"[Checkpoint] Pruned: {os.path.basename(path)}")
        except Exception as e:
            print(f"[Checkpoint] Prune failed for {path}: {e}")

# ----------------------------
# Entrypoint
# ----------------------------
def main():
    args = parse_args()
    cfg = build_config_from_args(args)
    cfg = maybe_autoresume(cfg)
    print(cfg); print("=" * 50)
    trainer = SQLM_VLM_Trainer(cfg)
    trainer.train()

if __name__ == "__main__":
    main()

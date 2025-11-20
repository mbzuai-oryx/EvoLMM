#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-evaluate multiple Qwen VLM checkpoints (LoRA) with lmms-eval.

Features:
- Discovers newest step per base model via tools/select_qwen_solver.py
- Merges LoRA -> full weights (cached; re-merge only if input changed)
- Runs lmms-eval with live, timestamped logs (no blank screen)
- Per-model log files + summary table at the end
- Lots of sanity checks (env, binaries, paths)

Usage (example):
  conda activate /share/data/drive_3/conda_envs/lmms-eval
  cd ~/self_evolving_vlm/lmms-eval
  python tools/eval_all_qwen.py \
      --runs_root /home/omkar/self_evolving_vlm/original_version/runs \
      --tasks mme,mmbench_dev_en,chartqa \
      --procs 8 --port 12346 --pixels 12845056 --dtype bfloat16 \
      --logs_dir /share/data/drive_3/self_evolving_vlm/eval_logs \
      --skip_merge_if_unchanged

Tips:
- If OOM: lower --pixels (e.g. 256*28*28) and/or use --procs 4 or 1
- If FA2 is not installed: use --attn '' to disable flash_attention_2
"""

from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Relative helper tools inside this repo
SEL = Path("tools/select_qwen_solver.py")
MERGE = Path("tools/merge_qwen_lora.py")

def ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)

def print_header(title: str):
    line = "=" * 80
    print(f"\n{line}\n[{ts()}] {title}\n{line}")

def which(bin_name: str) -> Optional[str]:
    from shutil import which as _which
    return _which(bin_name)

def check_python_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False

def run_json(cmd: List[str]) -> object:
    """Run a command that prints JSON to stdout and return parsed object."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{err}")
    try:
        return json.loads(out.strip() or "null")
    except json.JSONDecodeError as ex:
        raise RuntimeError(f"Invalid JSON from: {' '.join(cmd)}\n--- STDOUT ---\n{out}\n--- STDERR ---\n{err}") from ex

def stream_run(cmd: str, log_file: Optional[Path] = None) -> int:
    """
    Stream command output live (stdout+stderr) with timestamps.
    Also tee to log_file if provided.
    """
    print(f"[{ts()}] >> {cmd}")
    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    f = None
    try:
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            f = log_file.open("a", encoding="utf-8")
            f.write(f"===== CMD: {cmd}\n")
        for line in p.stdout:
            line_ts = f"[{ts()}] {line.rstrip()}"
            print(line_ts)
            if f:
                f.write(line_ts + "\n")
    finally:
        if f:
            f.flush()
            f.close()
    return p.wait()

def sha1_of_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()  # nosec: not for security

def model_key_from_base(base_model: str) -> str:
    """
    Map HF base to lmms-eval model key.
    """
    if "Qwen3-VL" in base_model:
        return "qwen3_vl"
    if "Qwen2.5-VL" in base_model or "Qwen2.5" in base_model:
        return "qwen2_5_vl"
    # Fallback (best effort)
    return "qwen2_5_vl"

def load_candidates(runs_root: Path) -> List[Dict]:
    """Use select_qwen_solver.py --per_model to get newest per base."""
    cmd = ["python", str(SEL), "--runs_root", str(runs_root), "--per_model"]
    return run_json(cmd) or []

def must_exist(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")

def write_stamp(merged_dir: Path, base_model: str, lora_path: str):
    stamp = {
        "base_model": base_model,
        "lora_path": lora_path,
        "lora_hash": sha1_of_text(lora_path),
        "created": ts(),
    }
    (merged_dir / "eval_merge_stamp.json").write_text(json.dumps(stamp, indent=2), encoding="utf-8")

def read_stamp(merged_dir: Path) -> Optional[Dict]:
    f = merged_dir / "eval_merge_stamp.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def needs_merge(merged_dir: Path, base_model: str, lora_path: str, skip_if_unchanged: bool) -> bool:
    """
    Decide whether we should merge again.
    - If skip_if_unchanged and stamp matches (same base + lora hash), skip.
    - If config.json missing -> merge.
    """
    cfg = merged_dir / "config.json"
    if not cfg.exists():
        return True
    if not skip_if_unchanged:
        return True
    stamp = read_stamp(merged_dir)
    if not stamp:
        return True
    if stamp.get("base_model") != base_model:
        return True
    if stamp.get("lora_hash") != sha1_of_text(lora_path):
        return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, type=Path, help="Path to your runs/ directory")
    ap.add_argument("--tasks", default="mme", help="Comma-separated task list for lmms-eval")
    ap.add_argument("--procs", type=int, default=8, help="accelerate --num_processes")
    ap.add_argument("--port", type=int, default=12346, help="accelerate --main_process_port")
    ap.add_argument("--pixels", default="12845056", help="model_args:max_pixels (lower if OOM)")
    ap.add_argument("--attn", default="flash_attention_2", help="attn_implementation value; '' to disable")
    ap.add_argument("--merged_root", default="../merged_models", help="Where to place merged models")
    ap.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"], help="Merge dtype")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--logs_dir", default="../eval_logs", help="Directory for per-model logs")
    ap.add_argument("--output_path", default="./results", help="Directory to save results")
    ap.add_argument("--skip_merge_if_unchanged", action="store_true", help="Skip merge if same base+lora as last time")
    ap.add_argument("--dry_run", action="store_true", help="Print steps but do not execute accelerate runs")
    args = ap.parse_args()

    print_header("Environment & prerequisites")

    # Check we are inside the repo (so relative tools exist)
    repo_root = Path.cwd()
    print(f"[{ts()}] Repo root: {repo_root}")
    must_exist(SEL, "Selector")
    must_exist(MERGE, "Merger")
    must_exist(args.runs_root, "runs_root directory")

    # Check binaries
    need_bins = ["python", "accelerate"]
    for b in need_bins:
        path = which(b)
        print(f"[{ts()}] which {b}: {path}")
        if not path:
            raise RuntimeError(f"Required binary not found on PATH: {b}")

    # Python modules
    mods = ["transformers", "peft", "lmms_eval"]
    for m in mods:
        ok = check_python_import(m)
        print(f"[{ts()}] import {m}: {'OK' if ok else 'MISSING'}")
        if not ok:
            raise RuntimeError(f"Python module missing: {m} (install it in this env)")

    # CUDA info (best effort)
    nvsmi = which("nvidia-smi")
    if nvsmi:
        print_header("nvidia-smi (GPU info)")
        _ = stream_run(f"{nvsmi}")
    else:
        eprint(f"[{ts()}] Warning: nvidia-smi not found; assuming CPU or non-standard CUDA setup")

    print_header("Discovering newest checkpoints (per base model)")
    items = load_candidates(args.runs_root)
    if not items:
        raise RuntimeError(f"No candidates found under {args.runs_root}")
    # Pretty print table
    colw = max(len(Path(it['run_dir']).name) for it in items)
    print(f"{'RUN DIR'.ljust(colw)}  STEP   BASE_KEY         BASE_ID")
    print("-"*colw + "  -----  ---------------  -------------------------------")
    for it in items:
        run_name = Path(it["run_dir"]).name
        print(f"{run_name.ljust(colw)}  {it['step_num']:<5}  {it['base_key']:<15}  {it['base_model']}")

    print_header("Merge & Eval plan")
    print(json.dumps(items, indent=2))

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    results_summary: List[Dict[str, str]] = []

    for it in items:
        base = it["base_model"]
        lora = it["lora_path"]
        base_key = it["base_key"]
        model_key = model_key_from_base(base)
        merged = Path(args.merged_root) / f"merged_{base_key}"
        merged.mkdir(parents=True, exist_ok=True)

        print_header(f"[{base_key}] Merge decision")
        do_merge = needs_merge(merged, base, lora, skip_if_unchanged=args.skip_merge_if_unchanged)
        print(f"[{ts()}] merged dir: {merged}")
        print(f"[{ts()}] will_merge: {do_merge} (skip_merge_if_unchanged={args.skip_merge_if_unchanged})")

        if do_merge:
            merge_cmd = f'python {shlex.quote(str(MERGE))} --base "{base}" --lora "{lora}" --out "{merged}" --dtype {args.dtype}'
            rc = stream_run(merge_cmd, log_file=logs_dir / f"{base_key}.merge.log")
            if rc != 0:
                eprint(f"[{ts()}] ERROR: merge failed for {base_key} (rc={rc})")
                results_summary.append({"base_key": base_key, "status": "merge_failed"})
                continue
            write_stamp(merged, base, lora)
        else:
            print(f"[{ts()}] Skipping merge (unchanged & cached).")

        # Build model args
        model_args = [
            f'pretrained="{merged}"',
            f"max_pixels={args.pixels}",
            "interleave_visuals=False",
        ]
        if args.attn:
            model_args.append(f"attn_implementation={args.attn}")
        eval_cmd = (
            f"accelerate launch --num_processes={args.procs} --main_process_port={args.port} -m lmms_eval "
            f"--model {model_key} "
            f'--model_args={",".join(model_args)} '
            f"--tasks {args.tasks} "
            f"--output_path {args.output_path} "
            f"--batch_size {args.batch_size}"
        )

        print_header(f"[{base_key}] Evaluate via lmms-eval")
        print(f"[{ts()}] MODEL_KEY: {model_key}")
        print(f"[{ts()}] TASKS: {args.tasks}")
        print(f"[{ts()}] CMD: {eval_cmd}")

        if args.dry_run:
            print(f"[{ts()}] DRY RUN: skipping execution.")
            results_summary.append({"base_key": base_key, "status": "dry_run"})
            continue

        rc = stream_run(eval_cmd, log_file=logs_dir / f"{base_key}.eval.log")
        status = "ok" if rc == 0 else f"eval_failed_rc_{rc}"
        if rc != 0:
            eprint(f"[{ts()}] ERROR: eval failed for {base_key} (rc={rc})")
        results_summary.append({
            "base_key": base_key,
            "status": status,
            "merged_dir": str(merged),
            "log_file": str((logs_dir / f"{base_key}.eval.log").resolve()),
        })

    print_header("Summary")
    # Compact table
    colw = max(8, max(len(x["base_key"]) for x in results_summary))
    print(f"{'BASE_KEY'.ljust(colw)}  STATUS")
    print("-"*colw + "  ------")
    for r in results_summary:
        print(f"{r['base_key'].ljust(colw)}  {r['status']}")
    print("\nFull JSON:")
    print(json.dumps(results_summary, indent=2))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        eprint(f"\n[{ts()}] Interrupted by user.")
        sys.exit(130)
    except Exception as ex:
        eprint(f"\n[{ts()}] FATAL: {ex}")
        sys.exit(1)

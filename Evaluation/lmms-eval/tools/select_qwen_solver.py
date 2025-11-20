#!/usr/bin/env python3
"""
Select Qwen-VL solver adapters from a runs/ tree.

Features:
- --list: show table of latest step for each run dir
- --model: filter by run-name substring (e.g., qwen2_5_vl_7b)
- default: pick the newest candidate across ALL runs (by latest step mtime)
- --per_model: instead of a single pick, return one pick per distinct base model
- JSON output with: run_dir, step_dir, base_model (HF id), lora_path

Examples:
  python tools/select_qwen_solver.py --runs_root /abs/path/runs
  python tools/select_qwen_solver.py --runs_root /abs/path/runs --model qwen2_5_vl_7b
  python tools/select_qwen_solver.py --runs_root /abs/path/runs --per_model
  python tools/select_qwen_solver.py --runs_root /abs/path/runs --list
"""

import argparse, json, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HF_MAP: Dict[str, str] = {
    # Qwen2.5-VL
    "qwen2_5_vl_1_5b": "Qwen/Qwen2.5-VL-1.5B-Instruct",
    "qwen2_5_vl_3b":   "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_5_vl_7b":   "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2_5_vl_14b":  "Qwen/Qwen2.5-VL-14B-Instruct",
    # Qwen3-VL
    "qwen3_vl_2b":     "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_vl_4b":     "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3_vl_7b":     "Qwen/Qwen3-VL-7B-Instruct",
}

STEP_RE = re.compile(r"^step_(\d+)$")

def infer_base_key(run_dir_name: str) -> Optional[str]:
    # strip suffixes like _lora_r16, _r8, _rs16, etc.
    name = re.sub(r"(_lora.*|_r\d+.*|_rs\d+.*)$", "", run_dir_name).rstrip("_")
    # try exact match first
    if name in HF_MAP:
        return name
    # fallback: best-effort prefix match
    for k in HF_MAP:
        if name.startswith(k):
            return k
    return None

def infer_base_id(run_dir_name: str) -> Optional[str]:
    k = infer_base_key(run_dir_name)
    return HF_MAP.get(k) if k else None

def latest_step_dir(run_path: Path) -> Optional[Tuple[int, Path]]:
    best = (-1, None)
    for d in run_path.iterdir():
        if d.is_dir():
            m = STEP_RE.match(d.name)
            if m:
                n = int(m.group(1))
                if n > best[0]:
                    best = (n, d)
    return best if best[1] is not None else None

@dataclass
class Candidate:
    run_dir: Path
    step_dir: Path
    step_num: int
    mtime: float
    base_key: str
    base_id: str
    lora_path: Path

def scan_runs(root: Path, name_filter: Optional[str]) -> List[Candidate]:
    out: List[Candidate] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        if name_filter and name_filter not in run_dir.name:
            continue
        base_id = infer_base_id(run_dir.name)
        base_key = infer_base_key(run_dir.name) or "unknown"
        step_info = latest_step_dir(run_dir)
        if not base_id or not step_info:
            continue
        step_num, step_dir = step_info
        solver = step_dir / "solver"
        if not solver.exists():
            continue
        out.append(
            Candidate(
                run_dir=run_dir,
                step_dir=step_dir,
                step_num=step_num,
                mtime=step_dir.stat().st_mtime,
                base_key=base_key,
                base_id=base_id,
                lora_path=solver,
            )
        )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, type=Path)
    ap.add_argument("--model", help="Substring to filter run dir names, e.g. qwen2_5_vl_7b")
    ap.add_argument("--list", action="store_true", help="List latest step for each run and exit")
    ap.add_argument("--per_model", action="store_true",
                    help="Return one newest candidate per base model key (JSON list)")
    args = ap.parse_args()

    root = args.runs_root
    if not root.exists():
        print(f"[ERR] runs_root not found: {root}", file=sys.stderr)
        sys.exit(2)

    cands = scan_runs(root, args.model)

    if args.list:
        if not cands:
            print("No candidates found.")
            return
        # print a table
        w = max(len(c.run_dir.name) for c in cands)
        print(f"{'RUN DIR'.ljust(w)}  STEP   BASE_KEY         BASE_ID")
        print("-"*w + "  -----  ---------------  -------------------------------")
        for c in cands:
            print(f"{c.run_dir.name.ljust(w)}  {c.step_num:<5}  {c.base_key:<15}  {c.base_id}")
        return

    if not cands:
        print("[]") if args.per_model else print("{}", end="")
        return

    if args.per_model:
        # pick newest per base_key
        best_by_key: Dict[str, Candidate] = {}
        for c in cands:
            b = best_by_key.get(c.base_key)
            if (b is None) or (c.mtime > b.mtime):
                best_by_key[c.base_key] = c
        result = []
        for key in sorted(best_by_key):
            c = best_by_key[key]
            result.append({
                "run_dir": str(c.run_dir.resolve()),
                "step_dir": str(c.step_dir.resolve()),
                "step_num": c.step_num,
                "base_key": c.base_key,
                "base_model": c.base_id,
                "lora_path": str(c.lora_path.resolve()),
            })
        print(json.dumps(result, indent=2))
        return

    # default: newest overall
    newest = max(cands, key=lambda c: c.mtime)
    out = {
        "run_dir": str(newest.run_dir.resolve()),
        "step_dir": str(newest.step_dir.resolve()),
        "step_num": newest.step_num,
        "base_key": newest.base_key,
        "base_model": newest.base_id,
        "lora_path": str(newest.lora_path.resolve()),
        "note": "Newest candidate by latest-step mtime.",
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

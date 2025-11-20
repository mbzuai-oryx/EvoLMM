#!/usr/bin/env python3
import argparse, os
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    model = AutoModelForVision2Seq.from_pretrained(args.base, torch_dtype=args.dtype if args.dtype!="auto" else "auto")
    model = PeftModel.from_pretrained(model, args.lora)
    model = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)

    proc = AutoProcessor.from_pretrained(args.base)
    proc.save_pretrained(args.out)
    print("Merged model saved to:", args.out)

if __name__ == "__main__":
    main()

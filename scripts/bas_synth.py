#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path

from pathlib import Path
import sys, os

def _try_imports():
    # 1) Try as an installed package
    try:
        from backdoor_generation.config import GenConfig
        from backdoor_generation.writer import write_balanced_jsonl
        return GenConfig, write_balanced_jsonl
    except Exception:
        pass

    # 2) Fallback: add repo root and package dir to sys.path, then import
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    pkg_dir = repo_root / "backdoor_generation"

    # Make both discoverable (repo root is the key one)
    for p in (str(repo_root), str(pkg_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Prefer package-style import if possible
    try:
        from backdoor_generation.config import GenConfig
        from backdoor_generation.writer import write_balanced_jsonl
        return GenConfig, write_balanced_jsonl
    except Exception:
        # Last resort: plain local modules at repo root (only if you had flat files)
        from config import GenConfig          # type: ignore
        from writer import write_balanced_jsonl  # type: ignore
        return GenConfig, write_balanced_jsonl

def main():
    GenConfig, write_balanced_jsonl = _try_imports()

    p = argparse.ArgumentParser("BAS synthetic dataset generator")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--val", type=int, default=500)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--outdir", type=str, default="data/bas_synth")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-enum-size", type=int, default=6)
    args = p.parse_args()

    cfg = GenConfig(seed=args.seed, max_enum_size=args.max_enum_size)
    os.makedirs(args.outdir, exist_ok=True)
    write_balanced_jsonl(os.path.join(args.outdir, "train.jsonl"), args.train, "train", cfg)
    write_balanced_jsonl(os.path.join(args.outdir, "val.jsonl"),   args.val,   "val",   cfg)
    write_balanced_jsonl(os.path.join(args.outdir, "test.jsonl"),  args.test,  "test",  cfg)
    print(f"Wrote dataset to {args.outdir}/{{train,val,test}}.jsonl")

if __name__ == "__main__":
    main()
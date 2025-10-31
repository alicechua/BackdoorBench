import os

from .config import GenConfig
from .writer import write_balanced_jsonl

def main():
    import argparse
    p = argparse.ArgumentParser("BAS synthetic dataset generator")
    p.add_argument("--train", type=int, default=2000, help="num train samples (set large later)")
    p.add_argument("--val", type=int, default=500)
    p.add_argument("--test", type=int, default=1000)
    p.add_argument("--outdir", type=str, default="data/bas_synth")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--max-enum-size", type=int, default=6,
                   help="cap search for valid sets; increase for more completeness at cost")
    args = p.parse_args()

    cfg = GenConfig(seed=args.seed, max_enum_size=args.max_enum_size)

    # # Train
    # write_jsonl(os.path.join(args.outdir, "train.jsonl"),
    #            generate_samples(args.train, "train", cfg))
    # # Val
    # write_jsonl(os.path.join(args.outdir, "val.jsonl"),
    #            generate_samples(args.val, "val", cfg))
    # # Test (uses test-degree range & can be larger N if you raise cfg.n_max)
    # write_jsonl(os.path.join(args.outdir, "test.jsonl"),
    #            generate_samples(args.test, "test", cfg))

    # Train
    write_balanced_jsonl(os.path.join(args.outdir, "train.jsonl"), args.train, "train", cfg)
    # Val
    write_balanced_jsonl(os.path.join(args.outdir, "val.jsonl"), args.val, "val", cfg)
    # Test
    write_balanced_jsonl(os.path.join(args.outdir, "test.jsonl"), args.test, "test", cfg)

    print(f"Wrote dataset to {args.outdir}/{{train,val,test}}.jsonl")
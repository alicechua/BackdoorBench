#!/usr/bin/env python3
import argparse, os, sys

def _try_imports():
    try:
        from backdoor_generation.config import GenConfig
        from backdoor_generation.writer import write_balanced_jsonl
        return GenConfig, write_balanced_jsonl
    except Exception:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from backdoor_generation.config import GenConfig
        from backdoor_generation.writer import write_balanced_jsonl
        return GenConfig, write_balanced_jsonl

def _csv_floats(s: str):
    return tuple(float(x) for x in s.split(",")) if s else None

def main():
    GenConfig, write_balanced_jsonl = _try_imports()

    p = argparse.ArgumentParser("BAS synthetic dataset generator")
    p.add_argument("--train", type=int, default=0)
    p.add_argument("--val",   type=int, default=0)
    p.add_argument("--test",  type=int, default=0)
    p.add_argument("--outdir", type=str, default="data/bas_synth")
    p.add_argument("--seed",   type=int, default=123)
    p.add_argument("--max-enum-size", type=int, default=6)

    # --- NEW: node naming controls ---
    p.add_argument("--node-name-style", default="int",
                   choices=["int","alpha3","alpha5","alnum3","varNNN","mixed","words"])
    p.add_argument("--node-name-prefix", default="")
    p.add_argument("--node-name-lower", action="store_true", help="force lowercase for alpha/alnum (not mixed)")
    # mixed-only options
    p.add_argument("--mixed-len", type=int, default=3, help="MAX length for mixed (actual 1..max)")
    p.add_argument("--mixed-policy", default="uniform", choices=["uniform","prefer_max"])
    p.add_argument("--mixed-weights", type=_csv_floats, default=None,
                   help="custom probs for lengths 1..max, e.g. '0.1,0.2,0.7'")
    p.add_argument("--mixed-no-lower", action="store_true")
    p.add_argument("--mixed-no-upper", action="store_true")
    p.add_argument("--mixed-no-digits", action="store_true")

    args = p.parse_args()

    cfg = GenConfig(
        seed=args.seed,
        max_enum_size=args.max_enum_size,
        # node naming:
        node_name_style=args.node_name_style,
        node_name_prefix=args.node_name_prefix,
        node_name_upper=not args.node_name_lower,
        node_mixed_len=args.mixed_len,
        node_mixed_len_policy=args.mixed_policy,
        node_mixed_len_weights=args.mixed_weights,
        node_mixed_allow_lower=not args.mixed_no_lower,
        node_mixed_allow_upper=not args.mixed_no_upper,
        node_mixed_allow_digits=not args.mixed_no_digits,
    )

    os.makedirs(args.outdir, exist_ok=True)
    if args.train > 0:
        write_balanced_jsonl(os.path.join(args.outdir, "train.jsonl"), args.train, "train", cfg)
    if args.val > 0:
        write_balanced_jsonl(os.path.join(args.outdir, "val.jsonl"),   args.val,   "val",   cfg)
    if args.test > 0:
        write_balanced_jsonl(os.path.join(args.outdir, "test.jsonl"),  args.test,  "test",  cfg)
    print(f"Wrote dataset to {args.outdir}/{{train,val,test}}.jsonl")

if __name__ == "__main__":
    main()
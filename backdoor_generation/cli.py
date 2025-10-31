import os

from .config import GenConfig
from .writer import write_balanced_jsonl

def _csv_floats(s: str):
    return tuple(float(x) for x in s.split(",")) if s else None

def main():
    import argparse
    p = argparse.ArgumentParser("BAS synthetic dataset generator")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--val",   type=int, default=500)
    p.add_argument("--test",  type=int, default=1000)
    p.add_argument("--outdir", type=str, default="data/bas_synth")
    p.add_argument("--seed",   type=int, default=123)
    p.add_argument("--max-enum-size", type=int, default=6)

    # Same node-naming flags as above
    p.add_argument("--node-name-style", default="int",
                   choices=["int","alpha3","alpha5","alnum3","varNNN","mixed","words"])
    p.add_argument("--node-name-prefix", default="")
    p.add_argument("--node-name-lower", action="store_true")
    p.add_argument("--mixed-len", type=int, default=3)
    p.add_argument("--mixed-policy", default="uniform", choices=["uniform","prefer_max"])
    p.add_argument("--mixed-weights", type=_csv_floats, default=None)
    p.add_argument("--mixed-no-lower", action="store_true")
    p.add_argument("--mixed-no-upper", action="store_true")
    p.add_argument("--mixed-no-digits", action="store_true")

    args = p.parse_args()

    cfg = GenConfig(
        seed=args.seed,
        max_enum_size=args.max_enum_size,
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
    write_balanced_jsonl(os.path.join(args.outdir, "train.jsonl"), args.train, "train", cfg)
    write_balanced_jsonl(os.path.join(args.outdir, "val.jsonl"),   args.val,   "val",   cfg)
    write_balanced_jsonl(os.path.join(args.outdir, "test.jsonl"),  args.test,  "test",  cfg)
    print(f"Wrote dataset to {args.outdir}/{{train,val,test}}.jsonl")

if __name__ == "__main__":
    main()


# def main():
#     import argparse
#     p = argparse.ArgumentParser("BAS synthetic dataset generator")
#     p.add_argument("--train", type=int, default=2000, help="num train samples (set large later)")
#     p.add_argument("--val", type=int, default=500)
#     p.add_argument("--test", type=int, default=1000)
#     p.add_argument("--outdir", type=str, default="data/bas_synth")
#     p.add_argument("--seed", type=int, default=123)
#     p.add_argument("--max-enum-size", type=int, default=6,
#                     help="cap search for valid sets; increase for more completeness at cost")
#     p.add_argument("--no-meta-complexity", action="store_true",
#                     help="disable graph-complexity metadata on samples")
#     p.add_argument("--complexity-fields", type=str, default="",
#                     help="comma-separated fields to keep (n_nodes,avg_degree,num_backdoorish_paths,S_size,num_minimal_sets)")
#     args = p.parse_args()

#     fields = tuple([s.strip() for s in args.complexity_fields.split(",") if s.strip()]) or None

#     cfg = GenConfig(
#         seed=args.seed,
#         max_enum_size=args.max_enum_size,
#         include_meta_complexity=not args.no_meta_complexity,
#         complexity_fields=fields or GenConfig().complexity_fields,
#     )

#     # # Train
#     # write_jsonl(os.path.join(args.outdir, "train.jsonl"),
#     #            generate_samples(args.train, "train", cfg))
#     # # Val
#     # write_jsonl(os.path.join(args.outdir, "val.jsonl"),
#     #            generate_samples(args.val, "val", cfg))
#     # # Test (uses test-degree range & can be larger N if you raise cfg.n_max)
#     # write_jsonl(os.path.join(args.outdir, "test.jsonl"),
#     #            generate_samples(args.test, "test", cfg))

#     # Train
#     write_balanced_jsonl(os.path.join(args.outdir, "train.jsonl"), args.train, "train", cfg)
#     # Val
#     write_balanced_jsonl(os.path.join(args.outdir, "val.jsonl"), args.val, "val", cfg)
#     # Test
#     write_balanced_jsonl(os.path.join(args.outdir, "test.jsonl"), args.test, "test", cfg)

#     print(f"Wrote dataset to {args.outdir}/{{train,val,test}}.jsonl")
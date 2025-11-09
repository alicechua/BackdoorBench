from __future__ import annotations
import json, os, random
import networkx as nx
from tqdm import tqdm

from .names import make_node_names
from .adjustment import is_minimal, is_ba_valid 
from .graphs import descendants

def _graph_from_edges(num_nodes: int, edges):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    return G

def _relabel_for_output(edges, X, Y, S, cfg, rng):
    # infer n from edges/X/Y/S
    node_ids = {u for (u, v) in edges for u in (u, v)} | {X, Y} | set(S)
    n = max(node_ids) + 1 if node_ids else 0

    names = make_node_names(
        n=n,
        style=cfg.node_name_style,
        rng=rng,
        upper=cfg.node_name_upper,
        prefix=cfg.node_name_prefix,
        mixed_len=cfg.node_mixed_len,                     # MAX length
        mixed_allow_lower=cfg.node_mixed_allow_lower,
        mixed_allow_upper=cfg.node_mixed_allow_upper,
        mixed_allow_digits=cfg.node_mixed_allow_digits,
        mixed_len_policy=getattr(cfg, "node_mixed_len_policy", "uniform"),
        mixed_len_weights=getattr(cfg, "node_mixed_len_weights", None),
    )
    mapping = {i: names[i] for i in range(n)}
    out_edges = [(mapping[u], mapping[v]) for (u, v) in edges]
    out_X, out_Y = mapping[X], mapping[Y]
    out_S = [mapping[s] for s in S]
    reverse = {v: k for k, v in mapping.items()}  # if you need to keep integer IDs somewhere
    style = cfg.node_name_style
    return out_edges, out_X, out_Y, out_S, reverse, style

def write_balanced_jsonl(path: str, total: int, split: str, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    pos_target = total // 2
    neg_target = total - pos_target
    pos = neg = 0
    rng = random.Random(cfg.seed ^ hash((split, path)))

    from .generate import random_dag, pick_query_pair
    from .positives import make_positive
    from .negatives import make_negative_weighted

    with open(path, "w") as f:
        # Add a progress bar for pos + neg
        pbar = tqdm(total=pos_target + neg_target, desc=f"Writing {split} set")
        while pos < pos_target or neg < neg_target:
            nmin = cfg.n_min if split != "test" else cfg.n_min + 3
            nmax = cfg.n_max if split != "test" else cfg.n_max + 3
            n = rng.randint(nmin, nmax)
            deg_rng = cfg.avg_deg_test if split == "test" else cfg.avg_deg_train
            G = random_dag(n, rng.uniform(*deg_rng), rng)
            X, Y = pick_query_pair(G, rng)
            assert X != Y

            need_pos = (pos_target - pos) >= (neg_target - neg)
            built = make_positive(G, X, Y, rng, cfg.max_enum_size, cfg) if need_pos \
                    else make_negative_weighted(G, X, Y, rng, cfg)
            if not built:
                continue

            # Recompute label if you want iron-clad truth (optional but recommended)
            Gfull = _graph_from_edges(G.number_of_nodes(), built.graph_edges)
            valid = bool(is_ba_valid(Gfull, built.X, built.Y, set(built.S)))
            minimal = valid and is_minimal(Gfull, built.X, built.Y, set(built.S), is_ba_valid)
            label = 1 if (valid and minimal) else 0
            if need_pos and label != 1: 
                continue
            if (not need_pos) and label != 0:
                continue

            # === relabel for emission ===
            out_edges, out_X, out_Y, out_S, inverse_map, style = _relabel_for_output(
                built.graph_edges, built.X, built.Y, built.S, cfg, rng
            )


            # meta updates
            built.meta["split"] = split
            built.meta["descendants_of_X"] = sorted(list(descendants(G, built.X)))
            built.meta["name_style"] = style
            # optionally keep inverse map for debugging (comment if you want slimmer files)
            # built.meta["inverse_name_map"] = inverse_map

            obj = {
                "graph": {"num_nodes": len(inverse_map), "edges": out_edges},
                "X": out_X,
                "Y": out_Y,
                "S": out_S,
                "label": built.label,
                "meta": {
                    **built.meta,
                    "split": split,
                    "descendants_of_X": built.meta.get("descendants_of_X", []),
                    "node_name_style": cfg.node_name_style,
                },
                "premise": ". ".join([f"{u} causes {v}" for (u, v) in out_edges]) + ".",
                "hypothesis": f"{sorted(out_S)} is a valid minimal backdoor adjustment set for {out_X} -> {out_Y}",
            }

            if label == 1 and pos < pos_target:
                f.write(json.dumps(obj) + "\n"); pos += 1
            elif label == 0 and neg < neg_target:
                f.write(json.dumps(obj) + "\n"); neg += 1

            pbar.n = pos + neg
            pbar.refresh()
        pbar.close()

# def _compute_label_and_meta(G: nx.DiGraph, X: int, Y: int, S: list[int]) -> tuple[int, bool, bool]:
#     Sset = set(S)
#     valid = bool(is_ba_valid(G, X, Y, Sset))
#     minimal = bool(is_minimal(G, X, Y, Sset, is_ba_valid)) if valid else False
#     label = 1 if (valid and minimal) else 0
#     return label, valid, minimal

# def _to_graph(num_nodes: int, edges: list[tuple[int,int]]) -> nx.DiGraph:
#     G = nx.DiGraph()
#     G.add_nodes_from(range(num_nodes))
#     G.add_edges_from(edges)
#     return G

# def write_balanced_jsonl(path: str, total: int, split: str, cfg):
#     os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
#     pos_target = total // 2
#     neg_target = total - pos_target
#     pos = neg = 0

#     from .generate import random_dag, pick_query_pair
#     from .positives import make_positive
#     from .negatives import make_negative_weighted
#     from .graphs import descendants

#     import random
#     rng = random.Random(cfg.seed ^ hash((split, path)))

#     with open(path, "w") as f:
#         while pos < pos_target or neg < neg_target:
#             nmax = cfg.n_max if split != "test" else max(cfg.n_max, cfg.n_min + 2)
#             n = rng.randint(cfg.n_min, nmax)
#             deg_rng = cfg.avg_deg_test if split == "test" else cfg.avg_deg_train
#             G = random_dag(n, rng.uniform(*deg_rng), rng)
#             X, Y = pick_query_pair(G, rng)

#             need_pos = (pos_target - pos) >= (neg_target - neg)

#             built = None
#             if need_pos:
#                 built = make_positive(G, X, Y, rng, cfg.max_enum_size)
#             else:
#                 built = make_negative_weighted(G, X, Y, rng, cfg)

#             if not built:
#                 continue

#             # Recompute label to guarantee correctness
#             num_nodes = G.number_of_nodes()
#             Gfull = _to_graph(num_nodes, built.graph_edges)
#             label, valid, minimal = _compute_label_and_meta(Gfull, built.X, built.Y, built.S)

#             # If we were aiming for pos/neg, ensure it matches bucket; else discard
#             if need_pos and label != 1:
#                 continue
#             if (not need_pos) and label != 0:
#                 continue

#             built.meta["split"] = split
#             built.meta["descendants_of_X"] = sorted(list(descendants(G, built.X)))
#             built.meta["computed_valid"] = valid
#             built.meta["computed_minimal"] = minimal

#             obj = {
#                 "graph": {"num_nodes": num_nodes, "edges": built.graph_edges},
#                 "X": built.X,
#                 "Y": built.Y,
#                 "S": built.S,
#                 "label": label,  # <- computed here
#                 "meta": built.meta,
#                 "hypothesis": f"{sorted(built.S)} is a valid minimal backdoor adjustment set for {built.X} -> {built.Y}",
#             }

#             if label == 1 and pos < pos_target:
#                 f.write(json.dumps(obj) + "\n"); pos += 1
#             elif label == 0 and neg < neg_target:
#                 f.write(json.dumps(obj) + "\n"); neg += 1

# def write_jsonl(path: str, samples: Iterable[Sample]) -> None:
#     os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
#     with open(path, "w") as f:
#         for s in samples:
#             edge_nodes = {u for (u, v) in s.graph_edges for u in (u, v)}
#             num_nodes = max(edge_nodes | {s.X, s.Y} | set(s.S)) + 1 if edge_nodes else 0
#             obj = {
#                 "graph": {"num_nodes": num_nodes,
#                           "edges": s.graph_edges,},
#                 "X": s.X,
#                 "Y": s.Y,
#                 "S": s.S,
#                 "label": s.label,
#                 "meta": s.meta,
#                 "hypothesis": f"{sorted(s.S)} is a valid minimal backdoor adjustment set for {s.X} -> {s.Y}",
#             }
#             f.write(json.dumps(obj) + "\n")

# from .config import GenConfig
# from .graphs import random_dag, descendants
# from .positives import make_positive  
# from .samples import pick_query_pair
# from .negatives import make_negative_weighted

# def write_balanced_jsonl(path: str, total: int, split: str, cfg: GenConfig):
#     import json, os, random
#     os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

#     pos_target = total // 2
#     neg_target = total - pos_target
#     pos = neg = 0
#     rng = random.Random(cfg.seed + hash((split, path)) % (2**31))

#     # track per-type caps if provided
#     caps = dict(cfg.neg_caps or {})
#     counts = {k: 0 for k in ["near_miss", "collider", "forbidden"]}

#     with open(path, "w") as f:
#         while pos < pos_target or neg < neg_target:
#             # sample a graph in the right regime
#             nmax = cfg.n_max if split != "test" else max(cfg.n_max, cfg.n_min + 2)
#             n = rng.randint(cfg.n_min, nmax)
#             deg_rng = cfg.avg_deg_test if split == "test" else cfg.avg_deg_train
#             G = random_dag(n, rng.uniform(*deg_rng), rng)
#             X, Y = pick_query_pair(G, rng)

#             # Decide which class we still need more of
#             need_pos = (pos_target - pos) >= (neg_target - neg)

#             built = None
#             if need_pos:
#                 # Try a few times to find a positive on this graph
#                 for _ in range(cfg.neg_retry_per_graph):
#                     built = make_positive(G, X, Y, rng, cfg.max_enum_size)
#                     if built:
#                         break
#             else:
#                 # Try a few weighted negatives; honor caps if set
#                 for _ in range(cfg.neg_retry_per_graph):
#                     cand = make_negative_weighted(G, X, Y, rng, cfg)
#                     if not cand:
#                         continue
#                     # infer type for cap accounting
#                     neg_type = cand.meta.get("neg_type")
#                     if not neg_type:
#                         # infer from builder heuristics
#                         if any(s in descendants(G, X) for s in cand.S):
#                             neg_type = "forbidden"
#                         else:
#                             # quick check via proper backdoor graph
#                             neg_type = "near_miss" if cand.label == 0 else "collider"
#                     if caps and neg_type in caps and counts[neg_type] >= caps[neg_type]:
#                         continue
#                     counts[neg_type] += 1
#                     built = cand
#                     break

#             if not built:
#                 continue

#             built.meta["split"] = split
#             built.meta["descendants_of_X"] = sorted(list(descendants(G, built.X)))

#             obj = {
#                 "graph": {"num_nodes": G.number_of_nodes(), "edges": built.graph_edges},
#                 "X": built.X,
#                 "Y": built.Y,
#                 "S": built.S,
#                 "label": built.label,
#                 "meta": built.meta,
#                 "hypothesis": f"{sorted(built.S)} is a valid minimal backdoor adjustment set for {built.X} -> {built.Y}",
#             }

#             if built.label == 1 and pos < pos_target:
#                 f.write(json.dumps(obj) + "\n")
#                 pos += 1
#             elif built.label == 0 and neg < neg_target:
#                 f.write(json.dumps(obj) + "\n")
#                 neg += 1
#             # else: discard and keep sampling
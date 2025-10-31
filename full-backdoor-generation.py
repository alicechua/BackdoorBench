# bas_synth.py
from __future__ import annotations
import itertools, json, math, os, random
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple, Iterable, Optional

import networkx as nx

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.base import DAG as PGMPY_DAG  # for d-separation checks
    HAVE_PGMPY = True
except Exception:
    HAVE_PGMPY = False

try:
    import dowhy  # noqa: F401
    HAVE_DOWHY = True
except Exception:
    HAVE_DOWHY = False


# -------------------------------
# Core graph & BAS utilities
# -------------------------------

def random_dag(n: int, avg_deg: float, rng: random.Random) -> nx.DiGraph:
    """
    Generate a random DAG by sampling a random topological order and adding forward edges
    with probability chosen to achieve approximately avg_deg.
    """
    assert n >= 3
    order = list(range(n))
    rng.shuffle(order)
    pos = {node: i for i, node in enumerate(order)}

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    # number of possible forward edges
    m = n * (n - 1) / 2
    p = min(max(avg_deg * n / (2 * m), 0.0), 1.0)  # crude calibration

    for i in range(n):
        for j in range(i + 1, n):
            u, v = order[i], order[j]
            if rng.random() < p:
                G.add_edge(u, v)

    # ensure weak connectivity-ish: add a few random forward edges if sparse
    if nx.number_of_edges(G) < n - 1:
        # chain along order
        for i in range(n - 1):
            G.add_edge(order[i], order[i + 1])
    return G


def descendants(G: nx.DiGraph, X: int) -> Set[int]:
    return nx.descendants(G, X)


def to_pgmpy_dag(G: nx.DiGraph) -> PGMPY_DAG:
    dag = PGMPY_DAG()
    dag.add_nodes_from(G.nodes())
    dag.add_edges_from(G.edges())
    return dag


def proper_backdoor_graph(G: nx.DiGraph, X: int) -> nx.DiGraph:
    """
    Proper back-door graph for X: remove ALL outgoing edges from X.
    (Per Perković et al. / 'complete adjustment criterion' DAG case)
    """
    H = G.copy()
    for _, v in list(H.out_edges(X)):
        H.remove_edge(X, v)
    return H


# def is_ba_valid(G: nx.DiGraph, X: int, Y: int, S: Set[int]) -> bool:
#     """
#     Basic (general/complete) adjustment validity in DAGs via proper back-door graph:
#     1) S contains no descendants of X in original G.
#     2) S d-separates X and Y in the proper back-door graph for X.
#     """
#     if X == Y: 
#         return False
#     if any(s in descendants(G, X) for s in S):
#         return False

#     H = proper_backdoor_graph(G, X)

#     if HAVE_PGMPY:
#         dag = to_pgmpy_dag(H)
#         return dag.is_dconnected(X, Y, observed=set()) is False if len(S) == 0 \
#                else dag.is_dseparated(set([X]), set([Y]), set(S))
#     else:
#         # Fallback: moralize+undirect d-separation approximation (conservative)
#         # Convert to BayesianNetwork for pgmpy-like behavior if available
#         # If pgmpy not available, approximate with active trail in DAG using networkx
#         # We'll do a light-weight active-trail check using pgmpy's algorithm idea:
#         # For reliability, strongly recommend pgmpy installed.
#         try:
#             from pgmpy.base import DAG as _D  # may not be present if imports failed earlier
#             dag = to_pgmpy_dag(H)
#             return dag.is_dseparated({X}, {Y}, set(S))
#         except Exception:
#             # Very conservative fallback: treat undirected separation as proxy.
#             UG = H.to_undirected()
#             UG.remove_nodes_from(S)
#             return not nx.has_path(UG, X, Y)

def is_ba_valid(G: nx.DiGraph, X: int, Y: int, S: Set[int]) -> bool:
    if X == Y:
        return False
    if any(s in descendants(G, X) for s in S):
        return False

    H = proper_backdoor_graph(G, X)

    if HAVE_PGMPY:
        dag = to_pgmpy_dag(H)
        # d-separated  <=>  NOT d-connected given observed S
        return not dag.is_dconnected(X, Y, observed=set(S))
    else:
        # Conservative fallback without pgmpy
        UG = H.to_undirected()
        UG.remove_nodes_from(S)
        return not nx.has_path(UG, X, Y)


def minimalize_sets(G: nx.DiGraph, X: int, Y: int, sets: Iterable[Set[int]]) -> List[Set[int]]:
    mins: List[Set[int]] = []
    for S in sets:
        if any(T < S for T in sets if T != S):  # quick precheck
            continue
        # true minimality: no proper subset valid
        is_min = True
        for k in range(len(S)):
            for subset in itertools.combinations(S, k):
                subset = set(subset)
                if subset != S and is_ba_valid(G, X, Y, subset):
                    is_min = False
                    break
            if not is_min:
                break
        if is_min:
            mins.append(S)
    # remove duplicates
    uniq = []
    for s in mins:
        if not any(s == t for t in uniq):
            uniq.append(s)
    return uniq


def enumerate_valid_adjustment_sets(
    G: nx.DiGraph, X: int, Y: int, max_size: Optional[int] = None
) -> List[Set[int]]:
    """
    Enumerate all valid adjustment sets by brute force over candidate nodes
    excluding X, Y. Optionally cap by max_size for speed.
    """
    nodes = [v for v in G.nodes() if v not in (X, Y)]
    all_sets = []
    upper = len(nodes) if max_size is None else min(max_size, len(nodes))
    for r in range(0, upper + 1):
        for combo in itertools.combinations(nodes, r):
            S = set(combo)
            if is_ba_valid(G, X, Y, S):
                all_sets.append(S)
    return minimalize_sets(G, X, Y, all_sets)


def list_backdoor_paths_crude(G: nx.DiGraph, X: int, Y: int) -> List[List[int]]:
    """
    (Metadata only) Enumerate simple paths in the *proper back-door graph* BEFORE conditioning.
    Not a full active-trail enumeration, but a useful proxy for 'how many backdoor-ish routes exist'.
    """
    H = proper_backdoor_graph(G, X)
    UG = H.to_undirected()
    paths = []
    try:
        for path in nx.all_simple_paths(UG, source=X, target=Y, cutoff=len(G.nodes()) - 1):
            paths.append(path)
    except nx.NetworkXNoPath:
        pass
    return paths


# -------------------------------
# Sample construction (positives + adversarial negatives)
# -------------------------------

@dataclass
class Sample:
    graph_edges: List[Tuple[int, int]]
    X: int
    Y: int
    S: List[int]
    label: int  # 1 = valid minimal BAS, 0 = not
    meta: Dict

def pick_query_pair(G: nx.DiGraph, rng: random.Random) -> Tuple[int, int]:
    nodes = list(G.nodes())
    X = rng.choice(nodes)
    Ys = [v for v in nodes if v != X and nx.has_path(G, X, v) or nx.has_path(G, v, X)]
    if not Ys:
        # fallback: ensure at least distinct
        Y = rng.choice([v for v in nodes if v != X])
    else:
        Y = rng.choice(Ys)
    return X, Y


def make_positive(G: nx.DiGraph, X: int, Y: int, rng: random.Random,
                  max_enum_size: Optional[int] = 6) -> Optional[Sample]:
    mins = enumerate_valid_adjustment_sets(G, X, Y, max_size=max_enum_size)
    if not mins:
        return None
    S = list(rng.choice(mins))
    meta = compute_metadata(G, X, Y, S, mins)
    return Sample(list(G.edges()), X, Y, S, 1, meta)


def make_negative_forbidden_desc(G: nx.DiGraph, X: int, Y: int, rng: random.Random,
                                 base_S: Optional[Set[int]] = None) -> Optional[Sample]:
    """Add a forbidden adjuster (descendant of X) to otherwise-valid-ish S."""
    desc = list(descendants(G, X))
    if not desc:
        return None
    if base_S is None:
        base_S = set()
    S = set(base_S)
    S.add(rng.choice(desc))
    if is_ba_valid(G, X, Y, S):
        return None  # failed to become negative; skip
    meta = compute_metadata(G, X, Y, S, [])
    return Sample(list(G.edges()), X, Y, sorted(S), 0, meta)


def make_negative_collider_trap(G: nx.DiGraph, X: int, Y: int, rng: random.Random) -> Optional[Sample]:
    """
    Try to include a collider or descendant of collider on an X-Y path to open it.
    Heuristic: pick Z with two parents not connected -> collider candidate.
    """
    for z in rng.sample(list(G.nodes()), k=len(G.nodes())):
        parents = list(G.predecessors(z))
        if len(parents) >= 2:
            # typical collider z <- a, z <- b
            S = {z}
            if not is_ba_valid(G, X, Y, S):
                meta = compute_metadata(G, X, Y, S, [])
                meta['neg_type'] = 'collider_trap'
                return Sample(list(G.edges()), X, Y, sorted(S), 0, meta)
    return None


def make_negative_near_miss(G: nx.DiGraph, X: int, Y: int, rng: random.Random,
                            max_enum_size: Optional[int] = 6) -> Optional[Sample]:
    """
    Take a valid minimal set and perturb it so exactly one backdoor path stays unblocked (remove a key node).
    """
    mins = enumerate_valid_adjustment_sets(G, X, Y, max_size=max_enum_size)
    if not mins:
        return None
    base = rng.choice(mins)
    if not base:
        return None
    # remove one element to (likely) leave a path open
    z = rng.choice(list(base))
    S = set(base) - {z}
    if is_ba_valid(G, X, Y, S):
        return None  # still valid -> skip
    meta = compute_metadata(G, X, Y, S, mins)
    meta['neg_type'] = 'near_miss'
    return Sample(list(G.edges()), X, Y, sorted(S), 0, meta)


def compute_metadata(G: nx.DiGraph, X: int, Y: int, S: Iterable[int], minimal_sets: List[Set[int]]) -> Dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = 0.0 if n == 0 else m * 2.0 / n
    bpaths = list_backdoor_paths_crude(G, X, Y)
    meta = {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": avg_deg,
        "num_backdoorish_paths": len(bpaths),
        "S_size": len(list(S)),
        "num_minimal_sets": len(minimal_sets),
        "minimal_sets": [sorted(list(ms)) for ms in minimal_sets][:10],  # cap
    }
    return meta

def make_negative_weighted(G, X, Y, rng, cfg) -> Optional[Sample]:
    # Prepare candidates by weight (filter out disabled types)
    options = []
    for name, w in cfg.neg_weights.items():
        if w <= 0: 
            continue
        options.append((name, w))
    if not options:
        return None

    # weighted draw
    tot = sum(w for _, w in options)
    r = rng.random() * tot
    acc = 0.0
    chosen = options[-1][0]
    for name, w in options:
        acc += w
        if r <= acc:
            chosen = name
            break

    if chosen == "near_miss":
        return make_negative_near_miss(G, X, Y, rng, max_enum_size=None)  # allow full set within cap
    elif chosen == "collider":
        return make_negative_collider_trap(G, X, Y, rng)
    elif chosen == "forbidden":
        return make_negative_forbidden_desc(G, X, Y, rng)
    else:
        return None

# -------------------------------
# Dataset driver
# -------------------------------

@dataclass
class GenConfig:
    seed: int = 42
    n_min: int = 6
    n_max: int = 8
    avg_deg_train: Tuple[float, float] = (1.4, 1.9)
    avg_deg_test: Tuple[float, float] = (2.6, 3.2)
    max_enum_size: int = 7     # cap subset enumeration for speed
    positives_frac: float = 0.6
    # edge cases toggles
    include_forbidden_neg: bool = True
    include_collider_trap: bool = True
    include_near_miss: bool = True
    # Negative sampling mix
    neg_weights: Dict[str, float] = None   # e.g., {"near_miss":0.5, "collider":0.3, "forbidden":0.2}
    neg_caps: Dict[str, int] = None        # optional per-split caps; leave None to disable
    neg_retry_per_graph: int = 3           # try a few times before resampling graph

    def __post_init__(self):
        if self.neg_weights is None:
            self.neg_weights = {"near_miss": 0.5, "collider": 0.3, "forbidden": 0.2}


def generate_samples(
    total: int,
    split: str,
    cfg: GenConfig,
    rng: Optional[random.Random] = None
) -> Iterable[Sample]:
    """
    Yields Samples. For large totals, prefer streaming to JSONL.
    """
    rng = rng or random.Random(cfg.seed + hash(split) % (2**16))
    gen_count = 0
    while gen_count < total:
        n = rng.randint(cfg.n_min, cfg.n_max if split != "test" else max(cfg.n_max, cfg.n_min + 2))
        avg_rng = cfg.avg_deg_test if split == "test" else cfg.avg_deg_train
        avg_deg = rng.uniform(*avg_rng)
        G = random_dag(n, avg_deg, rng)
        X, Y = pick_query_pair(G, rng)

        want_pos = rng.random() < cfg.positives_frac

        sample = None
        if want_pos:
            sample = make_positive(G, X, Y, rng, cfg.max_enum_size)
        else:
            # try a few negative constructors
            neg_builders = []
            if cfg.include_near_miss:
                neg_builders.append(lambda: make_negative_near_miss(G, X, Y, rng, cfg.max_enum_size))
            if cfg.include_forbidden_neg:
                neg_builders.append(lambda: make_negative_forbidden_desc(G, X, Y, rng))
            if cfg.include_collider_trap:
                neg_builders.append(lambda: make_negative_collider_trap(G, X, Y, rng))
            rng.shuffle(neg_builders)
            for builder in neg_builders:
                sample = builder()
                if sample is not None:
                    break

        if sample is None:
            continue  # try another graph

        # mark split & any extra meta
        sample.meta["split"] = split
        sample.meta["descendants_of_X"] = sorted(list(descendants(G, sample.X)))
        gen_count += 1
        yield sample


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

def write_balanced_jsonl(path: str, total: int, split: str, cfg: GenConfig):
    import json, os, random
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

    pos_target = total // 2
    neg_target = total - pos_target
    pos = neg = 0
    rng = random.Random(cfg.seed + hash((split, path)) % (2**31))

    # track per-type caps if provided
    caps = dict(cfg.neg_caps or {})
    counts = {k: 0 for k in ["near_miss", "collider", "forbidden"]}

    with open(path, "w") as f:
        while pos < pos_target or neg < neg_target:
            # sample a graph in the right regime
            nmax = cfg.n_max if split != "test" else max(cfg.n_max, cfg.n_min + 2)
            n = rng.randint(cfg.n_min, nmax)
            deg_rng = cfg.avg_deg_test if split == "test" else cfg.avg_deg_train
            G = random_dag(n, rng.uniform(*deg_rng), rng)
            X, Y = pick_query_pair(G, rng)

            # Decide which class we still need more of
            need_pos = (pos_target - pos) >= (neg_target - neg)

            built = None
            if need_pos:
                # Try a few times to find a positive on this graph
                for _ in range(cfg.neg_retry_per_graph):
                    built = make_positive(G, X, Y, rng, cfg.max_enum_size)
                    if built:
                        break
            else:
                # Try a few weighted negatives; honor caps if set
                for _ in range(cfg.neg_retry_per_graph):
                    cand = make_negative_weighted(G, X, Y, rng, cfg)
                    if not cand:
                        continue
                    # infer type for cap accounting
                    neg_type = cand.meta.get("neg_type")
                    if not neg_type:
                        # infer from builder heuristics
                        if any(s in descendants(G, X) for s in cand.S):
                            neg_type = "forbidden"
                        else:
                            # quick check via proper backdoor graph
                            neg_type = "near_miss" if cand.label == 0 else "collider"
                    if caps and neg_type in caps and counts[neg_type] >= caps[neg_type]:
                        continue
                    counts[neg_type] += 1
                    built = cand
                    break

            if not built:
                continue

            built.meta["split"] = split
            built.meta["descendants_of_X"] = sorted(list(descendants(G, built.X)))

            obj = {
                "graph": {"num_nodes": G.number_of_nodes(), "edges": built.graph_edges},
                "X": built.X,
                "Y": built.Y,
                "S": built.S,
                "label": built.label,
                "meta": built.meta,
                "hypothesis": f"{sorted(built.S)} is a valid minimal backdoor adjustment set for {built.X} -> {built.Y}",
            }

            if built.label == 1 and pos < pos_target:
                f.write(json.dumps(obj) + "\n")
                pos += 1
            elif built.label == 0 and neg < neg_target:
                f.write(json.dumps(obj) + "\n")
                neg += 1
            # else: discard and keep sampling


# -------------------------------
# Optional: lightweight validators
# -------------------------------

def validate_with_pgmpy(G: nx.DiGraph, X: int, Y: int, S: Set[int]) -> bool:
    if not HAVE_PGMPY:
        return True  # skip if not available
    dag = to_pgmpy_dag(proper_backdoor_graph(G, X))
    ok1 = dag.is_dseparated({X}, {Y}, set(S))
    ok2 = not any(s in descendants(G, X) for s in S)
    return ok1 and ok2


# (Optional) Sketch for DoWhy numeric sanity check: simulate linear-Gaussian SCM and
# compare backdoor-adjusted estimate using S to the true ATE. Left as a placeholder
# to keep this file lean. You can add it when you want empirical validation.


# -------------------------------
# CLI entry
# -------------------------------

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

if __name__ == "__main__":
    main()

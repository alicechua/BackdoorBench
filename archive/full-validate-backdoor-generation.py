# validate_bas.py
from __future__ import annotations
import argparse, json, math, sys, itertools, collections, os
from typing import Set, List, Dict, Iterable, Tuple, Optional
import networkx as nx

# --- Optional pgmpy ---
HAVE_PGMPY = True
try:
    from pgmpy.base import DAG as PGMPY_DAG
except Exception:
    HAVE_PGMPY = False

# Ensure these helpers are used everywhere you build the DAG
def node_set_from_example(graph_dict, X, Y, S):
    # Nodes explicitly referenced by the example
    edge_nodes = {u for (u, v) in graph_dict["edges"] for u in (u, v)}
    edge_nodes |= {X, Y} | set(S)

    n = graph_dict.get("num_nodes")
    # Only trust num_nodes if it’s large enough to cover all referenced ids
    if isinstance(n, int) and (len(edge_nodes) == 0 or max(edge_nodes) < n):
        return set(range(n))
    else:
        # Fall back to the explicitly referenced nodes
        return edge_nodes

def to_pgmpy_dag(edges, nodes):
    from pgmpy.base import DAG as PGMPY_DAG
    dag = PGMPY_DAG()
    dag.add_nodes_from(sorted(nodes))   # <- add ALL nodes explicitly
    dag.add_edges_from(edges)           # edges can reference a subset; that's fine
    return dag

def proper_backdoor_graph(edges, X):
    # remove edges where u == X (do NOT remove node X)
    return [(u, v) for (u, v) in edges if u != X]

def descendants(G: nx.DiGraph, X: int) -> Set[int]:
    if X not in G:
        return set()  # isolated X ⇒ no descendants
    return set(nx.descendants(G, X))

def is_ba_valid(
    edges: List[Tuple[int, int]], X: int, Y: int, S: Set[int], nodes: Set[int]
) -> bool:
    # 1) forbid descendants of X in the ORIGINAL graph
    G = nx.DiGraph()
    G.add_nodes_from(nodes)          # <-- add ALL nodes
    G.add_edges_from(edges)
    if any(s in descendants(G, X) for s in S):
        return False

    # 2) d-sep in the proper backdoor graph (remove outgoing edges from X)
    pbd_edges = proper_backdoor_graph(edges, X)
    if HAVE_PGMPY:
        dag = to_pgmpy_dag(pbd_edges, nodes)  # <-- include ALL nodes
        return not dag.is_dconnected(X, Y, observed=set(S))
    else:
        UG = nx.DiGraph()
        UG.add_nodes_from(nodes)     # <-- include ALL nodes
        UG.add_edges_from(pbd_edges)
        UG = UG.to_undirected()
        UG.remove_nodes_from(S)
        return not nx.has_path(UG, X, Y)

def is_minimal(edges, X, Y, S: Set[int], nodes: Set[int]) -> bool:
    if not is_ba_valid(edges, X, Y, S, nodes):
        return False
    for k in range(len(S)):
        for sub in itertools.combinations(S, k):
            sub = set(sub)
            if sub != S and is_ba_valid(edges, X, Y, sub, nodes):
                return False
    return True

def load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def dag_checks(edges):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    # No self-loops
    if any(u == v for u, v in edges):
        return False, "self_loop"
    # Acyclic
    try:
        _ = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return False, "cycle"
    return True, ""

def analyze(path: str, strict: bool = True):
    total = 0
    correct = 0
    pos_total = neg_total = 0
    pos_minimal_ok = 0
    neg_reason_counts = collections.Counter()
    splits = collections.Counter()
    meta_stats = collections.defaultdict(list)

    for ex in load_jsonl(path):
        total += 1
        graph = ex["graph"]
        edges = [tuple(e) for e in graph["edges"]]
        X, Y = ex["X"], ex["Y"]
        S = set(ex["S"])
        nodes_full = node_set_from_example(graph, X, Y, S)  # <-- get full node set
        label = int(ex["label"])
        meta = ex.get("meta", {})
        split = meta.get("split", "unknown")
        splits[split] += 1

        ok_dag, reason = dag_checks(edges)
        if not ok_dag:
            print(f"[error] line {total}: invalid DAG ({reason})")
            if strict: return 1

        # recompute validity
        recomputed = 1 if is_ba_valid(edges, X, Y, S, nodes_full) else 0
        if recomputed == label:
            correct += 1

        if label == 1:
            pos_total += 1
            if is_minimal(edges, X, Y, S, nodes_full):
                pos_minimal_ok += 1
            elif strict:
                print(f"[error] line {total}: labeled positive but not minimal S={sorted(S)}")
                return 1
        else:
            neg_total += 1
            # try to diagnose why it's negative (for visibility)
            # 1) forbidden descendant?
            G = nx.DiGraph(); G.add_nodes_from(nodes_full); G.add_edges_from(edges)
            forb = any(s in descendants(G, X) for s in S)
            if forb:
                neg_reason_counts["forbidden_descendant"] += 1
            else:
                # 2) still d-connected?
                pbd = proper_backdoor_graph(edges, X)
                still_conn = None
                if HAVE_PGMPY:
                    dag = to_pgmpy_dag(pbd, nodes_full)
                    still_conn = dag.is_dconnected(X, Y, observed=set(S))
                else:
                    UG = nx.DiGraph(); UG.add_nodes_from(node_set_from_example(graph, X, Y, S))
                    UG.add_edges_from(pbd); UG = UG.to_undirected()
                    UG.remove_nodes_from(S)
                    still_conn = nx.has_path(UG, X, Y)
                if still_conn:
                    neg_reason_counts["still_dconnected"] += 1
                else:
                    # 3) not minimal (i.e., valid but non-minimal) — rare as “negative” type
                    if is_ba_valid(edges, X, Y, S, nodes_full) and not is_minimal(edges, X, Y, S, nodes_full):
                        neg_reason_counts["non_minimal"] += 1
                    else:
                        neg_reason_counts["other"] += 1

        # record a few metas
        for k in ["n_nodes", "n_edges", "avg_degree", "S_size", "num_minimal_sets", "num_backdoorish_paths"]:
            if k in meta:
                meta_stats[k].append(meta[k])

    # summary
    print("=== Verification Summary ===")
    print(f"File: {path}")
    print(f"Total samples: {total}")
    print(f"Label agreement (recomputed vs. stored): {correct}/{total} = {correct/total:.3f}")
    print(f"Positives: {pos_total} | Negatives: {neg_total}")
    if pos_total:
        print(f"Minimality OK on positives: {pos_minimal_ok}/{pos_total} = {pos_minimal_ok/pos_total:.3f}")
    print(f"Splits: {dict(splits)}")
    if neg_total:
        print(f"Negative reasons (diagnosed): {dict(neg_reason_counts)}")

    # balance check
    if total:
        frac_pos = pos_total / total
        print(f"Class balance (pos fraction): {frac_pos:.3f} (target ≈ 0.50)")
        if strict and not (0.4 <= frac_pos <= 0.6):
            print("[warn] Class balance outside [0.4, 0.6] for a smoke set; fine for small runs, but watch at scale.")

    # basic meta ranges
    for k, vals in meta_stats.items():
        if not vals: 
            continue
        mn, mx = min(vals), max(vals)
        avg = sum(vals) / len(vals)
        print(f"meta.{k}: min={mn}, max={mx}, mean={avg:.3f}")

    # OOD intuition (if both train/test present)
    if "train" in splits and "test" in splits and meta_stats.get("avg_degree"):
        # This is a soft check; you can harden with thresholds
        print("OOD check: ensure test tends to be larger/denser than train by config (inspect meta above).")

    return 0

def main():
    ap = argparse.ArgumentParser("Validate BAS JSONL")
    ap.add_argument("jsonl", help="Path to JSONL (e.g., data/bas_synth_smoke/train.jsonl)")
    ap.add_argument("--loose", action="store_true", help="Do not fail on non-minimal positives or imbalance")
    args = ap.parse_args()
    sys.exit(analyze(args.jsonl, strict=not args.loose))

if __name__ == "__main__":
    main()
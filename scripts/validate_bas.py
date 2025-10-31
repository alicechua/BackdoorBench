#!/usr/bin/env python3
"""
Standalone validator for BAS JSONL files.

- Recomputes label agreement using your BAS rule (forbidden descendants + d-sep
  in the proper back-door graph) via backdoor_generation.adjustment.is_ba_valid.
- Reports class balance, minimality on positives, and simple diagnostics for negatives.

This script assumes your code lives in the package folder:
  backdoor_generation/
    __init__.py
    adjustment.py
    graphs.py
    ...

It imports as a package; on failure it adds the repo root to sys.path,
then imports again as a package (to keep relative imports working).
"""

import argparse, json, sys, os, collections, itertools
from typing import Set, List, Tuple
from pathlib import Path
import networkx as nx

def _wire_imports():
    # 1) Try normal package import (works if PYTHONPATH includes repo root or package is installed)
    try:
        from backdoor_generation.adjustment import is_ba_valid  # type: ignore
        from backdoor_generation.graphs import proper_backdoor_graph  # type: ignore
        return is_ba_valid, proper_backdoor_graph
    except Exception:
        pass

    # 2) Fallback: add repo root and package dir to sys.path, then import VIA PACKAGE NAME
    #    (so that relative imports inside your package continue to work)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    pkg_dir = repo_root / "backdoor_generation"
    for p in (str(repo_root), str(pkg_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from backdoor_generation.adjustment import is_ba_valid  # type: ignore
    from backdoor_generation.graphs import proper_backdoor_graph  # type: ignore
    return is_ba_valid, proper_backdoor_graph

is_ba_valid, proper_backdoor_graph = _wire_imports()

def load_jsonl(path: str):
    with open(path, "r") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[error] JSON decode at line {i}: {e}", file=sys.stderr)

def dag_checks(edges: List[Tuple[int,int]]):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    if any(u == v for u, v in edges):
        return False, "self_loop"
    try:
        _ = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return False, "cycle"
    return True, ""

def node_set_from_example(graph_dict, X, Y, S):
    edge_nodes = {u for (u, v) in graph_dict["edges"] for u in (u, v)}
    edge_nodes |= {X, Y} | set(S)
    n = graph_dict.get("num_nodes")
    if isinstance(n, int) and (len(edge_nodes) == 0 or max(edge_nodes) < n):
        return set(range(n))
    else:
        return edge_nodes

def descendants(G: nx.DiGraph, X: int) -> Set[int]:
    if X not in G:
        return set()
    return set(nx.descendants(G, X))

def analyze(path: str, strict: bool = True) -> int:
    total = correct = 0
    pos_total = neg_total = 0
    pos_min_ok = 0
    neg_reason_counts = collections.Counter()
    splits = collections.Counter()
    meta_stats = collections.defaultdict(list)

    for lineno, ex in load_jsonl(path):
        total += 1
        graph = ex["graph"]
        edges = [tuple(e) for e in graph["edges"]]
        X, Y = ex["X"], ex["Y"]
        S = set(ex["S"])
        label = int(ex["label"])
        meta = ex.get("meta", {})
        split = meta.get("split", "unknown")
        splits[split] += 1

        ok_dag, reason = dag_checks(edges)
        if not ok_dag:
            print(f"[error] line {lineno}: invalid DAG ({reason})")
            if strict: return 1

        nodes_full = node_set_from_example(graph, X, Y, S)

        # recompute validity using your package helper
        G = nx.DiGraph(); G.add_nodes_from(nodes_full); G.add_edges_from(edges)
        recomputed = 1 if is_ba_valid(G, X, Y, S) else 0
        if recomputed == label:
            correct += 1
        else:
            print(f"[mismatch] line {lineno}: X={X} Y={Y} S={sorted(S)} label={label} recomputed={recomputed}")

        if label == 1:
            pos_total += 1
            # minimality check
            minimal = True
            for k in range(len(S)):
                for sub in itertools.combinations(S, k):
                    sub = set(sub)
                    if sub != S and is_ba_valid(G, X, Y, sub):
                        minimal = False; break
                if not minimal: break
            if minimal:
                pos_min_ok += 1
            elif strict:
                print(f"[error] line {lineno}: labeled positive but not minimal S={sorted(S)}")
                return 1
        else:
            neg_total += 1
            # Diagnose negative
            forb = any(s in descendants(G, X) for s in S)
            if forb:
                neg_reason_counts["forbidden_descendant"] += 1
            else:
                H = proper_backdoor_graph(G, X)

                # Ensure H contains all nodes (some implementations drop isolated nodes)
                H.add_nodes_from(G.nodes())

                UG = H.to_undirected()
                UG.remove_nodes_from(S)

                # Guard: if X or Y are missing (e.g., removed or were isolated and dropped), treat as no path
                if (X not in UG) or (Y not in UG):
                    still_conn = False
                else:
                    still_conn = nx.has_path(UG, X, Y)

                if still_conn:
                    neg_reason_counts["still_dconnected"] += 1
                else:
                    neg_reason_counts["other"] += 1


        # collect metas if present
        for k in ["n_nodes", "n_edges", "avg_degree", "S_size", "num_minimal_sets", "num_backdoorish_paths"]:
            if k in meta:
                meta_stats[k].append(meta[k])

    # summary
    print("=== Verification Summary ===")
    print(f"File: {path}")
    print(f"Total samples: {total}")
    print(f"Label agreement (recomputed vs. stored): {correct}/{total} = {correct/total if total else 0:.3f}")
    print(f"Positives: {pos_total} | Negatives: {neg_total}")
    if pos_total:
        print(f"Minimality OK on positives: {pos_min_ok}/{pos_total} = {pos_min_ok/pos_total:.3f}")
    print(f"Splits: {dict(splits)}")
    if neg_total:
        print(f"Negative reasons (diagnosed): {dict(neg_reason_counts)}")
    if total:
        frac_pos = pos_total / total
        print(f"Class balance (pos fraction): {frac_pos:.3f} (target ≈ 0.50)")
    for k, vals in meta_stats.items():
        if not vals: continue
        mn, mx = min(vals), max(vals)
        avg = sum(vals) / len(vals)
        print(f"meta.{k}: min={mn}, max={mx}, mean={avg:.3f}")
    return 0

def main():
    ap = argparse.ArgumentParser("Validate BAS JSONL")
    ap.add_argument("jsonl", help="Path to JSONL (e.g., data/bas_synth/train.jsonl)")
    ap.add_argument("--loose", action="store_true", help="Do not fail on non-minimal positives or imbalance")
    args = ap.parse_args()
    sys.exit(analyze(args.jsonl, strict=not args.loose))

if __name__ == "__main__":
    main()
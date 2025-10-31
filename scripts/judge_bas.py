#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import networkx as nx

# --- import from your package, robustly ---
def _wire():
    try:
        from backdoor_generation.adjustment import is_ba_valid
        return is_ba_valid
    except Exception:
        # add repo root if running from scripts/
        here = Path(__file__).resolve().parent
        root = here.parent
        pkg = root / "backdoor_generation"
        for p in (str(root), str(pkg)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from backdoor_generation.adjustment import is_ba_valid
        return is_ba_valid

is_ba_valid = _wire()

def _to_digraph(ex: Dict) -> nx.DiGraph:
    edges = [tuple(e) for e in ex["graph"]["edges"]]
    X, Y, S = ex["X"], ex["Y"], set(ex["S"])
    n_decl = ex["graph"].get("num_nodes")
    edge_nodes = {u for (u,v) in edges for u in (u,v)}
    nodes = set(range(n_decl)) if isinstance(n_decl, int) else edge_nodes | {X, Y} | S
    G = nx.DiGraph()
    G.add_nodes_from(sorted(nodes))
    G.add_edges_from(edges)
    return G

def _is_minimal(G: nx.DiGraph, X: int, Y: int, S: Set[int]) -> bool:
    if not is_ba_valid(G, X, Y, S): 
        return False
    for k in range(len(S)):
        from itertools import combinations
        for sub in map(set, combinations(S, k)):
            if sub != S and is_ba_valid(G, X, Y, sub):
                return False
    return True

def check_one(ex: Dict) -> Dict:
    G = _to_digraph(ex)
    X, Y, S = ex["X"], ex["Y"], set(ex["S"])
    valid = bool(is_ba_valid(G, X, Y, S))
    minimal = _is_minimal(G, X, Y, S) if valid else False
    out = {
        "X": X, "Y": Y, "S": sorted(S),
        "computed_label": int(valid and minimal),
        "valid": valid, "minimal": minimal,
    }
    return out

def main():
    ap = argparse.ArgumentParser("Compute BAS truth for JSONL")
    ap.add_argument("jsonl", help="Path to JSONL")
    ap.add_argument("--head", type=int, default=10, help="Print first K judgments")
    ap.add_argument("--write", type=str, default="", help="Optional path to write JSONL with computed_label")
    args = ap.parse_args()

    rows: List[Dict] = []
    with open(args.jsonl, "r") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            rows.append(json.loads(s))

    k = min(args.head, len(rows))
    pos = neg = 0
    for i, ex in enumerate(rows, 1):
        judg = check_one(ex)
        if i <= k:
            print(f"[{i}] X={judg['X']} Y={judg['Y']} S={judg['S']}  "
                  f"valid={judg['valid']} minimal={judg['minimal']}  "
                  f"computed_label={judg['computed_label']}")
        if judg["computed_label"] == 1: pos += 1
        else: neg += 1
        ex["computed_label"] = judg["computed_label"]
        ex.setdefault("meta", {})["computed_valid"] = judg["valid"]
        ex["meta"]["computed_minimal"] = judg["minimal"]

    print(f"\nSummary: positives={pos}, negatives={neg}, total={len(rows)}")

    if args.write:
        outp = Path(args.write); outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as g:
            for ex in rows:
                g.write(json.dumps(ex) + "\n")
        print(f"Wrote with computed_label → {outp}")

if __name__ == "__main__":
    main()
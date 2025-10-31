#!/usr/bin/env python3
"""
Visualize BAS dataset samples.

For each chosen example:
- Left: original DAG with special coloring for X (red), Y (blue), S (orange)
- Right: proper back-door graph for X with S removed; we annotate whether X~Y
        are still connected (a d-connection hint before full d-sep).

Usage:
  python scripts/visualize_bas.py data/bas_synth_smoke_2/train.jsonl --n 6 --seed 123 --outdir viz/train

Requires: networkx, matplotlib
Optional: pygraphviz or pydot for nicer layouts (if installed)
"""

from __future__ import annotations
import argparse, json, random, os, sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

import networkx as nx
import matplotlib.pyplot as plt

# ---------- Imports from your package (with robust fallback) ----------

def _wire_imports():
    # Try the package form first
    try:
        from backdoor_generation.graphs import proper_backdoor_graph  # type: ignore
        from backdoor_generation.adjustment import is_ba_valid        # type: ignore
        return proper_backdoor_graph, is_ba_valid
    except Exception:
        pass

    # Fallback: add repo root & package dir, then import via package so relative imports work
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    pkg_dir = repo_root / "backdoor_generation"
    for p in (str(repo_root), str(pkg_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from backdoor_generation.graphs import proper_backdoor_graph  # type: ignore
    from backdoor_generation.adjustment import is_ba_valid        # type: ignore
    return proper_backdoor_graph, is_ba_valid

proper_backdoor_graph, is_ba_valid = _wire_imports()

# ---------- JSONL utilities ----------

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            data.append(json.loads(s))
    return data

def to_digraph(example: Dict) -> nx.DiGraph:
    """Build a DiGraph that includes *all* nodes declared/needed, not just those on edges."""
    ginfo = example["graph"]
    edges = [tuple(e) for e in ginfo["edges"]]
    X, Y, S = example["X"], example["Y"], list(example["S"])

    # Infer node set robustly
    edge_nodes = {u for (u, v) in edges for u in (u, v)}
    n_declared = ginfo.get("num_nodes")
    if isinstance(n_declared, int):
        nodes = set(range(n_declared))
    else:
        nodes = edge_nodes | {X, Y} | set(S)

    G = nx.DiGraph()
    G.add_nodes_from(sorted(nodes))
    G.add_edges_from(edges)
    return G

# ---------- Layout helpers ----------

def _has_graphviz():
    try:
        import pygraphviz  # noqa
        return "agraph"
    except Exception:
        try:
            import pydot  # noqa
            return "pydot"
        except Exception:
            return None

def dag_layout(G: nx.DiGraph):
    """
    Prefer graphviz (dot) layered layout for DAGs when available.
    Fallback to networkx shell / spring layouts.
    """
    engine = _has_graphviz()
    if engine:
        try:
            return nx.nx_agraph.graphviz_layout(G, prog="dot") if engine == "agraph" else nx.nx_pydot.graphviz_layout(G, prog="dot")
        except Exception:
            pass
    # Fallbacks
    try:
        return nx.shell_layout(G)
    except Exception:
        return nx.spring_layout(G, seed=0)

# ---------- Drawing ----------

def draw_dag(ax, G: nx.DiGraph, X: int, Y: int, S: Set[int], title: str):
    pos = dag_layout(G)

    # Colors / sizes
    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if n == X:
            node_colors.append("#d62728")  # red
            node_sizes.append(600)
        elif n == Y:
            node_colors.append("#1f77b4")  # blue
            node_sizes.append(600)
        elif n in S:
            node_colors.append("#ff7f0e")  # orange
            node_sizes.append(500)
        else:
            node_colors.append("#aaaaaa")
            node_sizes.append(350)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax, linewidths=1.0, edgecolors="k")
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes()}, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.2, ax=ax, connectionstyle="arc3,rad=0.05")

    ax.set_title(title, fontsize=11)
    ax.axis("off")

def draw_backdoor_view(ax, G: nx.DiGraph, X: int, Y: int, S: Set[int], subtitle: str):
    # Build proper backdoor graph and ensure all nodes are present
    H = proper_backdoor_graph(G, X)
    H.add_nodes_from(G.nodes())  # keep isolated nodes
    # Remove S to visualize conditioning
    H2 = H.copy()
    H2.remove_nodes_from(S)

    pos = dag_layout(H2)

    # Presence flags for safety
    hasX = X in H2
    hasY = Y in H2

    # Colors
    node_colors = []
    node_sizes = []
    for n in H2.nodes():
        if n == X:
            node_colors.append("#d62728")
            node_sizes.append(600)
        elif n == Y:
            node_colors.append("#1f77b4")
            node_sizes.append(600)
        else:
            node_colors.append("#bbbbbb")
            node_sizes.append(350)

    # Undirected connectivity hint
    UG = H2.to_undirected()
    if hasX and hasY:
        try:
            still_conn = nx.has_path(UG, X, Y)
        except Exception:
            still_conn = False
    else:
        still_conn = False

    nx.draw_networkx_nodes(H2, pos, node_color=node_colors, node_size=node_sizes, ax=ax, linewidths=1.0, edgecolors="k")
    nx.draw_networkx_labels(H2, pos, labels={n: str(n) for n in H2.nodes()}, font_size=9, ax=ax)
    nx.draw_networkx_edges(H2, pos, arrows=True, arrowstyle="-|>", arrowsize=12, width=1.2, ax=ax, connectionstyle="arc3,rad=0.05")

    # Annotate connectivity
    status = "X–Y connected" if still_conn else "X–Y separated"
    ax.set_title(f"{subtitle}\nAfter removing S in proper back-door: {status}", fontsize=11)
    ax.axis("off")

# ---------- Main visualization ----------

def render_example(example: Dict, outpath: Path, idx: int):
    label = int(example["label"])
    X, Y, S = example["X"], example["Y"], set(example["S"])
    meta: Dict = example.get("meta", {})
    hyp = example.get("hypothesis", "")
    G = to_digraph(example)

    # Meta strings
    sS = sorted(list(S))
    m_n = meta.get("n_nodes", G.number_of_nodes())
    m_e = meta.get("n_edges", G.number_of_edges())
    m_deg = meta.get("avg_degree", (2.0 * m_e / m_n) if m_n else 0.0)
    m_nb = meta.get("num_backdoorish_paths", None)
    m_nms = meta.get("num_minimal_sets", None)
    split = meta.get("split", "?")

    fig = plt.figure(figsize=(11, 5.8))
    gs = fig.add_gridspec(1, 2, wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    title_left = f"[{split}] Original DAG | X={X}, Y={Y}, S={sS} | label={label}"
    draw_dag(ax1, G, X, Y, S, title_left)

    subtitle_right = f"n={m_n}, |E|={m_e}, avg_deg={m_deg:.2f}"
    if m_nms is not None:
        subtitle_right += f", #minimal_sets={m_nms}"
    if m_nb is not None:
        subtitle_right += f", #backdoorish_paths={m_nb}"
    draw_backdoor_view(ax2, G, X, Y, S, subtitle_right)

    # Footer text (wrap hypothesis if present)
    footer = hyp or ""
    if footer:
        fig.text(0.02, 0.02, f"Hypothesis: {footer}", fontsize=9)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"Sample #{idx}", fontsize=12, y=0.98)
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser("Visualize BAS JSONL samples")
    ap.add_argument("jsonl", help="Path to JSONL (e.g., data/bas_synth_smoke/train.jsonl)")
    ap.add_argument("--n", type=int, default=6, help="Number of random samples to render")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--outdir", type=str, default="viz", help="Output directory for PNGs")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = load_jsonl(args.jsonl)
    if not data:
        print(f"[warn] No data in {args.jsonl}")
        sys.exit(0)

    # Sample without replacement (or all if fewer than n)
    k = min(args.n, len(data))
    idxs = rng.sample(range(len(data)), k)

    outdir = Path(args.outdir)
    for i, idx in enumerate(idxs, 1):
        outpath = outdir / f"{Path(args.jsonl).stem}_sample_{i:02d}.png"
        render_example(data[idx], outpath, idx=i)

    print(f"Saved {k} figures to {outdir}/")

if __name__ == "__main__":
    main()

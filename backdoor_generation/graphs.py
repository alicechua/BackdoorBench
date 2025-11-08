# -------------------------------
# Core graph & Backdoor Adjustment Set utilities
# -------------------------------
from __future__ import annotations
import networkx as nx
import random
from typing import Dict, Iterable, List, Set, Tuple

try:
    from pgmpy.base import DAG as PGMPY_DAG  # for d-separation checks
    HAVE_PGMPY = True
except Exception:
    HAVE_PGMPY = False

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
    p = min(max(avg_deg * n / m, 0.0), 1.0)  # crude calibration

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
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    H.remove_edges_from(list(G.out_edges(X)))
    return H

def list_backdoor_paths_crude(G: nx.DiGraph, X: int, Y: int) -> List[List[int]]:
    H = proper_backdoor_graph(G, X)
    UG = H.to_undirected()
    try:
        return list(nx.all_simple_paths(UG, source=X, target=Y, cutoff=max(0, len(G.nodes()) - 1)))
    except nx.NetworkXNoPath:
        return []

def compute_complexity_meta(
    G: nx.DiGraph, X: int, Y: int, S: Iterable[int], minimal_sets: List[Set[int]]
) -> Dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    avg_deg = (2.0 * m / n) if n else 0.0
    bpaths = list_backdoor_paths_crude(G, X, Y)
    return {
        "n_nodes": n,
        "n_edges": m,                     # optional but useful
        "avg_degree": avg_deg,
        "num_backdoorish_paths": len(bpaths),
        "S_size": len(list(S)),
        "num_minimal_sets": len(minimal_sets),
    }

def filter_meta(meta: Dict, keep: Tuple[str, ...]) -> Dict:
    return {k: v for k, v in meta.items() if k in set(keep)}

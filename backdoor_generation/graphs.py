# -------------------------------
# Core graph & BAS utilities
# -------------------------------

import networkx as nx
import random
from typing import Set

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
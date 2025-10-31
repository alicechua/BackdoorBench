# -------------------------------
# Optional: lightweight validators
# -------------------------------

from typing import Set
import networkx as nx

from .graphs import proper_backdoor_graph, to_pgmpy_dag, descendants, HAVE_PGMPY

def validate_with_pgmpy(G: nx.DiGraph, X: int, Y: int, S: Set[int]) -> bool:
    if not HAVE_PGMPY:
        return True  # skip if not available
    dag = to_pgmpy_dag(proper_backdoor_graph(G, X))
    ok1 = dag.is_dseparated({X}, {Y}, set(S))
    ok2 = not any(s in descendants(G, X) for s in S)
    return ok1 and ok2
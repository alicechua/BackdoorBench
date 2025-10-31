import networkx as nx
from typing import List, Set, Optional, Iterable
import itertools

from .graphs import (
    HAVE_PGMPY,
    to_pgmpy_dag,
    proper_backdoor_graph,
    descendants,
)

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

def is_minimal(G: nx.DiGraph, X: int, Y: int, S: set, is_ba_valid) -> bool:
    if not is_ba_valid(G, X, Y, S):
        return False
    for k in range(len(S)):
        for sub in itertools.combinations(S, k):
            sub = set(sub)
            if sub != S and is_ba_valid(G, X, Y, sub):
                return False
    return True
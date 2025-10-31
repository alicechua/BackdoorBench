# -------------------------------
# Sample construction (positives + adversarial negatives)
# -------------------------------

from typing import List, Tuple, Set, Dict, Iterable, Optional
import networkx as nx
from dataclasses import dataclass
import random

from .adjustment import list_backdoor_paths_crude

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
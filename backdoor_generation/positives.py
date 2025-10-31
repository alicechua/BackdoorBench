from typing import Optional
import random
import networkx as nx

from .samples import Sample, compute_metadata
from .adjustment import enumerate_valid_adjustment_sets

def make_positive(G: nx.DiGraph, X: int, Y: int, rng: random.Random,
                  max_enum_size: Optional[int] = 6) -> Optional[Sample]:
    mins = enumerate_valid_adjustment_sets(G, X, Y, max_size=max_enum_size)
    if not mins:
        return None
    S = list(rng.choice(mins))
    meta = compute_metadata(G, X, Y, S, mins)
    return Sample(list(G.edges()), X, Y, S, 1, meta)
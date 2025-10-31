from typing import Optional
import random
import networkx as nx

from .adjustment import enumerate_valid_adjustment_sets
from .graphs import compute_complexity_meta, filter_meta
from .samples import Sample

def make_positive(G: nx.DiGraph, X: int, Y: int, rng: random.Random,
                  max_enum_size: Optional[int], cfg=None) -> Optional[Sample]:
    mins = enumerate_valid_adjustment_sets(G, X, Y, max_size=max_enum_size)
    if not mins:
        return None
    S = list(rng.choice(mins))
    meta = {}
    if cfg is None or getattr(cfg, "include_meta_complexity", True):
        raw = compute_complexity_meta(G, X, Y, S, mins)
        keep = getattr(cfg, "complexity_fields", tuple(raw.keys()))
        meta.update(filter_meta(raw, keep))
    return Sample(list(G.edges()), X, Y, S, 1, meta)

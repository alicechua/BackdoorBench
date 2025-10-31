from typing import Set, Optional
import networkx as nx
import random

from .samples import Sample
from .adjustment import is_ba_valid, enumerate_valid_adjustment_sets
from .graphs import descendants
from .samples import compute_metadata

def make_negative_forbidden_desc(G: nx.DiGraph, X: int, Y: int, rng: random.Random,
                                 base_S: Optional[Set[int]] = None) -> Optional[Sample]:
    """Add a forbidden adjuster (descendant of X) to otherwise-valid-ish S."""
    desc = list(descendants(G, X))
    if not desc:
        return None
    if base_S is None:
        base_S = set()
    S = set(base_S)
    S.add(rng.choice(desc))
    if is_ba_valid(G, X, Y, S):
        return None  # failed to become negative; skip
    meta = compute_metadata(G, X, Y, S, [])
    return Sample(list(G.edges()), X, Y, sorted(S), 0, meta)


def make_negative_collider_trap(G: nx.DiGraph, X: int, Y: int, rng: random.Random) -> Optional[Sample]:
    """
    Try to include a collider or descendant of collider on an X-Y path to open it.
    Heuristic: pick Z with two parents not connected -> collider candidate.
    """
    for z in rng.sample(list(G.nodes()), k=len(G.nodes())):
        parents = list(G.predecessors(z))
        if len(parents) >= 2:
            # typical collider z <- a, z <- b
            S = {z}
            if not is_ba_valid(G, X, Y, S):
                meta = compute_metadata(G, X, Y, S, [])
                meta['neg_type'] = 'collider'
                return Sample(list(G.edges()), X, Y, sorted(S), 0, meta)
    return None


def make_negative_near_miss(G: nx.DiGraph, X: int, Y: int, rng: random.Random,
                            max_enum_size: Optional[int] = 6) -> Optional[Sample]:
    """
    Take a valid minimal set and perturb it so exactly one backdoor path stays unblocked (remove a key node).
    """
    mins = enumerate_valid_adjustment_sets(G, X, Y, max_size=max_enum_size)
    if not mins:
        return None
    base = rng.choice(mins)
    if not base:
        return None
    # remove one element to (likely) leave a path open
    z = rng.choice(list(base))
    S = set(base) - {z}
    if is_ba_valid(G, X, Y, S):
        return None  # still valid -> skip
    meta = compute_metadata(G, X, Y, S, mins)
    meta['neg_type'] = 'near_miss'
    return Sample(list(G.edges()), X, Y, sorted(S), 0, meta)




def make_negative_weighted(G, X, Y, rng, cfg) -> Optional[Sample]:
    # Prepare candidates by weight (filter out disabled types)
    options = []
    for name, w in cfg.neg_weights.items():
        if w <= 0: 
            continue
        options.append((name, w))
    if not options:
        return None

    # weighted draw
    tot = sum(w for _, w in options)
    r = rng.random() * tot
    acc = 0.0
    chosen = options[-1][0]
    for name, w in options:
        acc += w
        if r <= acc:
            chosen = name
            break

    if chosen == "near_miss":
        return make_negative_near_miss(G, X, Y, rng, max_enum_size=None)  # allow full set within cap
    elif chosen == "collider":
        return make_negative_collider_trap(G, X, Y, rng)
    elif chosen == "forbidden":
        return make_negative_forbidden_desc(G, X, Y, rng)
    else:
        return None
import random
from typing import Iterable, Optional

from .samples import Sample, pick_query_pair
from .positives import make_positive
from .negatives import (
    make_negative_near_miss,
    make_negative_forbidden_desc,
    make_negative_collider_trap,
)
from .graphs import random_dag, descendants
from .config import GenConfig

def generate_samples(
    total: int,
    split: str,
    cfg: GenConfig,
    rng: Optional[random.Random] = None
) -> Iterable[Sample]:
    """
    Yields Samples. For large totals, prefer streaming to JSONL.
    """
    rng = rng or random.Random(cfg.seed + hash(split) % (2**16))
    gen_count = 0
    while gen_count < total:
        n = rng.randint(cfg.n_min, cfg.n_max if split != "test" else max(cfg.n_max, cfg.n_min + 2))
        avg_rng = cfg.avg_deg_test if split == "test" else cfg.avg_deg_train
        avg_deg = rng.uniform(*avg_rng)
        G = random_dag(n, avg_deg, rng)
        X, Y = pick_query_pair(G, rng)

        want_pos = rng.random() < cfg.positives_frac

        sample = None
        if want_pos:
            sample = make_positive(G, X, Y, rng, cfg.max_enum_size)
        else:
            # try a few negative constructors
            neg_builders = []
            if cfg.include_near_miss:
                neg_builders.append(lambda: make_negative_near_miss(G, X, Y, rng, cfg.max_enum_size))
            if cfg.include_forbidden_neg:
                neg_builders.append(lambda: make_negative_forbidden_desc(G, X, Y, rng))
            if cfg.include_collider_trap:
                neg_builders.append(lambda: make_negative_collider_trap(G, X, Y, rng))
            rng.shuffle(neg_builders)
            for builder in neg_builders:
                sample = builder()
                if sample is not None:
                    break

        if sample is None:
            continue  # try another graph

        # mark split & any extra meta
        sample.meta["split"] = split
        sample.meta["descendants_of_X"] = sorted(list(descendants(G, sample.X)))
        gen_count += 1
        yield sample

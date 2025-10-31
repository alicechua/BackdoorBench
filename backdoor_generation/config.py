from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class GenConfig:
    seed: int = 42
    n_min: int = 6
    n_max: int = 8
    avg_deg_train: Tuple[float, float] = (1.4, 1.9)
    avg_deg_test: Tuple[float, float] = (2.6, 3.2)
    max_enum_size: int = 7     # cap subset enumeration for speed
    positives_frac: float = 0.6
    # edge cases toggles
    include_forbidden_neg: bool = True
    include_collider_trap: bool = True
    include_near_miss: bool = True
    # Negative sampling mix
    neg_weights: Dict[str, float] = None   # e.g., {"near_miss":0.5, "collider":0.3, "forbidden":0.2}
    neg_caps: Dict[str, int] = None        # optional per-split caps; leave None to disable
    neg_retry_per_graph: int = 3           # try a few times before resampling graph

    def __post_init__(self):
        if self.neg_weights is None:
            self.neg_weights = {"near_miss": 0.5, "collider": 0.3, "forbidden": 0.2}
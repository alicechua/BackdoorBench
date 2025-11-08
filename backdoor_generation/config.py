from dataclasses import dataclass
from typing import Tuple, Dict, Optional

@dataclass
class GenConfig:
    seed: int = 42
    n_min: int = 4
    n_max: int = 7
    avg_deg_train: Tuple[float, float] = (1.0, 1.4)
    avg_deg_test: Tuple[float, float] = (2.0, 2.4)
    max_enum_size: int = 7     # cap subset enumeration for speed
    positives_frac: float = 0.6

    # === negative examples ===
    # edge cases toggles
    include_forbidden_neg: bool = True
    include_collider_trap: bool = True
    include_near_miss: bool = True
    # Negative sampling mix
    neg_weights: Dict[str, float] = None   # e.g., {"near_miss":0.5, "collider":0.3, "forbidden":0.2}
    neg_caps: Dict[str, int] = None        # optional per-split caps; leave None to disable
    neg_retry_per_graph: int = 3           # try a few times before resampling graph
    
    # === node naming ===
    node_name_style: str = "int"   # "int","alpha3","alpha5","alnum3","varNNN","mixed","words"
    node_name_upper: bool = True
    node_name_prefix: str = ""

    # Mixed-case alnum (variable-length) options
    node_mixed_len: int = 3                   # MAX length; actual ∈ [1..node_mixed_len]
    node_mixed_allow_lower: bool = True
    node_mixed_allow_upper: bool = True
    node_mixed_allow_digits: bool = True
    node_mixed_len_policy: str = "uniform"    # "uniform" | "prefer_max"
    node_mixed_len_weights: Optional[Tuple[float, ...]] = None  # e.g., (0.1,0.2,0.7) for lengths 1,2,3

    def __post_init__(self):
        if getattr(self, "neg_weights", None) is None:
            self.neg_weights = {"near_miss": 0.5, "collider": 0.3, "forbidden": 0.2}
from __future__ import annotations
import random, string
from typing import List, Sequence

def _sample_unique(rng: random.Random, pool: str, length: int, n: int) -> List[str]:
    """Fixed-length unique strings from `pool`."""
    out, seen = [], set()
    while len(out) < n:
        s = "".join(rng.choice(pool) for _ in range(length))
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def _sample_unique_varlen(
    rng: random.Random,
    pool: str,
    n: int,
    max_len: int,
    policy: str = "uniform",
    weights: Sequence[float] | None = None,
) -> List[str]:
    """
    Variable-length unique strings with length in [1, max_len].
    - policy: "uniform" (each length equally likely) or "prefer_max" (bias to longer).
    - weights: optional custom probs per length; len(weights) == max_len and
      weights[i] applies to length (i+1).
    """
    if max_len < 1:
        raise ValueError("max_len must be >= 1")

    lens = list(range(1, max_len + 1))
    if weights is not None:
        if len(weights) != max_len:
            raise ValueError("node_mixed_len_weights must have length == max_len")
        probs = [max(0.0, float(w)) for w in weights]
    else:
        if policy == "prefer_max":
            probs = [float(L) for L in lens]          # 1,2,...,max_len
        else:
            probs = [1.0] * max_len                   # uniform
    tot = sum(probs)
    if tot <= 0:
        raise ValueError("Length weights must sum to > 0")
    probs = [p / tot for p in probs]

    # Precompute cumulative to sample lengths quickly
    cum = []
    acc = 0.0
    for p in probs:
        acc += p
        cum.append(acc)

    def draw_len() -> int:
        r = rng.random()
        for i, c in enumerate(cum):
            if r <= c:
                return lens[i]
        return lens[-1]

    # Soft capacity check (not fatal; this can be strict if you prefer)
    capacity = sum((len(pool) ** L) for L in lens)
    if n > capacity * 0.95:
        # High collision risk if n is very close to capacity for tiny pools
        pass

    out, seen = [], set()
    while len(out) < n:
        L = draw_len()
        s = "".join(rng.choice(pool) for _ in range(L))
        if s in seen:
            continue
        seen.add(s); out.append(s)
    return out

def make_node_names(
    n: int,
    style: str,
    rng: random.Random,
    upper: bool = True,
    prefix: str = "",
    mixed_len: int = 3,                     # MAX length for mixed
    mixed_allow_lower: bool = True,
    mixed_allow_upper: bool = True,
    mixed_allow_digits: bool = True,
    mixed_len_policy: str = "uniform",      # "uniform" | "prefer_max"
    mixed_len_weights: Sequence[float] | None = None,
) -> List[str]:
    """
    Return `n` unique node labels for the requested style.

    Styles:
      - "int"         : "0","1","2",...
      - "alpha3"      : AAA/aaa etc (fixed length=3; controlled by `upper`)
      - "alpha5"      : AAAAA/aaaaa (fixed length=5)
      - "alnum3"      : fixed length=3, letters+digits (case by `upper`)
      - "varNNN"      : Var001, Var002, ...
      - "mixed"       : variable length in [1..mixed_len], with allow_* pools
      - "words"       : small Greek list; falls back to w### if n is large
    """
    if style == "int":
        return [str(i) for i in range(n)]

    if style in {"alpha3", "alpha5", "alnum3"}:
        pool = string.ascii_uppercase if upper else string.ascii_lowercase
        if style == "alnum3":
            pool += string.digits
        length = 5 if style == "alpha5" else 3
        names = _sample_unique(rng, pool, length, n)

    elif style == "varNNN":
        base = (prefix or "Var")
        return [f"{base}{i:03d}" for i in range(1, n + 1)]

    elif style == "mixed":
        parts = []
        if mixed_allow_lower: parts.append(string.ascii_lowercase)
        if mixed_allow_upper: parts.append(string.ascii_uppercase)
        if mixed_allow_digits: parts.append(string.digits)
        pool = "".join(parts)
        if not pool:
            raise ValueError("mixed style requires at least one of lower/upper/digits")

        names = _sample_unique_varlen(
            rng=rng,
            pool=pool,
            n=n,
            max_len=max(1, mixed_len),
            policy=mixed_len_policy,
            weights=mixed_len_weights,
        )

    elif style == "words":
        base = ["alpha","beta","gamma","delta","zeta","eta","theta","iota","kappa","lambda",
                "mu","nu","xi","omicron","pi","rho","sigma","tau","upsilon","phi","chi","psi","omega"]
        names = base[:n] if n <= len(base) else [f"w{i:03d}" for i in range(n)]
        if upper: names = [s.upper() for s in names]
        if prefix: names = [prefix + s for s in names]
        return names

    else:
        raise ValueError(f"unknown node_name_style: {style}")

    if prefix:
        names = [prefix + s for s in names]
    return names
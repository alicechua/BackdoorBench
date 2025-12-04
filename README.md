# BackdoorBench — BAS Synthetic Dataset Generator

Generate **synthetic DAGs** and labeled examples for the **Backdoor Adjustment Set (BAS)** rule, validate them, and visualize a few samples.  
This repo is organized as a small package (`backdoor_generation/`) plus CLI scripts (`scripts/`).

---

## TL;DR

```bash
# 0) (Recommended) Create a fresh venv (Python 3.10–3.12)
python -m venv .venv && source .venv/bin/activate

# 1) Install deps
pip install -r requirements.txt

# 2) Generate a small, balanced smoke set (override sizes as needed)
python scripts/bas_synth.py --train 50 --val 10 --test 20 --outdir data/bas_smoke

# 3) Validate labels + minimality
python scripts/validate_bas.py data/bas_smoke/train.jsonl
python scripts/validate_bas.py data/bas_smoke/val.jsonl
python scripts/validate_bas.py data/bas_smoke/test.jsonl

# 4) Visualize random samples to PNGs
python scripts/visualize_bas.py data/bas_smoke/test.jsonl --n 6 --outdir viz/test
```

---

## Folder structure

```
.
├─ backdoor_generation/           # library code
│  ├─ __init__.py
│  ├─ adjustment.py               # BAS rule (forbidden descendants + d-sep in proper backdoor graph)
│  ├─ config.py                   # GenConfig dataclass (generation knobs)
│  ├─ generate.py                 # dataset streaming/generation
│  ├─ graphs.py                   # DAG utilities, backdoor graph, enumeration, metadata
│  ├─ names.py                    # random node label generators (int, short strings, mixed etc.)
│  ├─ negatives.py                # adversarial negative constructors (forbidden, collider, near-miss)
│  ├─ positives.py                # positive constructor (pick minimal BAS)
│  ├─ samples.py                  # Sample dataclass (schema)
│  ├─ validators.py               # (optional) cross-check helpers
│  └─ writer.py                   # balanced writer (50/50), JSONL schema, relabeling
│
├─ scripts/
│  ├─ bas_synth.py                # CLI to generate datasets → JSONL
│  ├─ validate_bas.py             # CLI to re-check labels/minimality & print stats
│  └─ visualize_bas.py            # CLI to render side-by-side DAG views to PNG
│
├─ data/                          # (created) output datasets
└─ viz/                           # (created) rendered figures
│
└─ CLEAR/                         # files used to evaluate on CLEAR dataset
│
└─ finetune_llama/                # filed used to finetune llama on BackdoorBench
└─ gpt_baseline/                  # filed used to evaluate GPT-4o-mini on BackdoorBench
└─ transformer/                   # filed used to train BERT transformer on BackdoorBench
```

---

## Installation

> **NumPy / PyTorch note**  
> Some third-party wheels compiled against NumPy 1.x may fail on NumPy 2.x. If you hit “module compiled against NumPy 1.x…”, pin `numpy<2` in `requirements.txt`, or upgrade the affected lib.

```bash
python -m venv .venv
source .venv/bin/activate

# If you saw NumPy issues previously:
# pip install "numpy<2"

pip install -r requirements.txt
```

**Minimal runtime deps:** `networkx`, `matplotlib`  
**Optional (better d-sep & layouts):** `pgmpy` and `pygraphviz`/`pydot` (Graphviz system binary required for those layouts)

---

## Dataset format (JSONL)

Each line is one example:

```json
{
  "graph": {
    "num_nodes": 7,
    "edges": [["a","b"],["u","a"],["u","y"]]
  },
  "X": "a",
  "Y": "y",
  "S": ["u"],
  "label": 1,
  "meta": {
    "split": "train",
    "n_nodes": 7,
    "n_edges": 9,
    "avg_degree": 2.57,
    "num_backdoorish_paths": 4,
    "S_size": 1,
    "num_minimal_sets": 2,
    "descendants_of_X": ["..."]
  },
  "hypothesis": "['u'] is a valid minimal backdoor adjustment set for a -> y"
}
```

- **`label`**: `1` iff **S** is a **valid minimal BAS** for `X → Y` (forbidden-descendant check + d-sep in proper backdoor graph, **and** minimality).  
- Node labels can be **integers or strings** (depending on config).

---
## Steps
## 1) Generate data

**Basic:**
```bash
python scripts/bas_synth.py \
  --train 2000 --val 500 --test 1000 \
  --outdir data/bas_synth
```
This writes **balanced** splits (≈50/50 pos/neg) via `writer.write_balanced_jsonl`.

### Useful flags (from `GenConfig`)

You can pass these via env or edit `backdoor_generation/config.py`.

**Graph size/density**
- `n_min`, `n_max` (train); test internally allows larger *n*
- `avg_deg_train=(1.4, 1.9)`
- `avg_deg_test=(2.6, 3.2)`

**Enumeration cap**
- `max_enum_size` — max set size when enumerating valid sets (speed/recall tradeoff)

**Class mix**
- Balanced by writer; internally `positives_frac` steers sampling

**Negative types & weights**
- `include_near_miss`, `include_forbidden_neg`, `include_collider_trap`
- `neg_weights={"near_miss":0.5,"collider":0.3,"forbidden":0.2}`

**Node naming**
- `node_name_style`: `"int" | "short" | "mixed"`
- `node_short_len`: fixed length for short names (e.g., `2`)
- `node_mixed_len`: **maximum** length for mixed names (**1..max**, unique)
- `node_mixed_dist`: `"uniform"` or `"prefer_max"` (or custom weights)

**Example (mixed labels up to length 3):**
```bash
python scripts/bas_synth.py \
  --outdir data/bas_mixed \
  --train 2000 --val 500 --test 1000 \
  --max-enum-size 6
```
In `config.py` (or via env):
```python
node_name_style="mixed"
node_mixed_len=3
node_mixed_dist="prefer_max"
```

---

## 2) Validate

Recomputes labels using the same BAS rule and reports quality & meta-summary.

```bash
python scripts/validate_bas.py data/bas_synth/train.jsonl
python scripts/validate_bas.py data/bas_synth/val.jsonl
python scripts/validate_bas.py data/bas_synth/test.jsonl
```

**Output includes:**
- Label agreement
- Minimality check on positives
- Class balance
- Negative diagnostics: `forbidden_descendant`, `still_dconnected`, `other`
- Meta ranges: `n_nodes`, `n_edges`, `avg_degree`, `S_size`, `num_minimal_sets`, `num_backdoorish_paths`

---

## 3) Visualize

Renders side-by-side:

- **Left:** original DAG (X=red, Y=blue, S=orange)  
- **Right:** **proper backdoor graph** with S removed and an X–Y separation hint

```bash
python scripts/visualize_bas.py data/bas_synth/test.jsonl --n 6 --seed 123 --outdir viz/test
```

PNG files are saved under `viz/test/`.

> The visualizer **does not invent nodes**: it builds the node set from `edges ∪ {X,Y} ∪ S` so string labels work and no stray `0..n-1` nodes appear.

---

## Advanced: package import vs. local

Scripts first try:
```python
from backdoor_generation.config import GenConfig
from backdoor_generation.writer import write_balanced_jsonl
```
If that fails, they add the repo root to `sys.path` and import again—so you can run scripts without `pip install -e .`.

To make it installable:
```bash
pip install -e .
```

---

## Make it your own

- **OOD knobs:** increase `n_max` or test `avg_deg_test` to create structure-level shifts.  
- **Adversarial negatives:** tweak `neg_weights`/`neg_caps` to emphasize collider traps vs. forbidden descendants.  
- **Metadata:** the dataset logs the 5 core fields: `n_nodes`, `avg_degree`, `num_backdoorish_paths`, `S_size`, `num_minimal_sets`.

---

## Troubleshooting

**ModuleNotFoundError for local modules**  
Run from repo root, e.g.:
```bash
python scripts/bas_synth.py --help
```
Scripts auto-patch `sys.path` to include the repo.

**Stray numeric nodes in figures**  
Ensure node set is built from `edges ∪ {X,Y} ∪ S` (the provided `visualize_bas.py` already does this).

**NumPy 1.x vs 2.x ABI error**  
Pin or upgrade:
```bash
pip install "numpy<2"
```

---

## Typical commands

```bash
# Small smoke set (balanced), mixed labels
python scripts/bas_synth.py --train 100 --val 20 --test 40 --outdir data/bas_smoke

# Validate all splits
for s in train val test; do
  python scripts/validate_bas.py data/bas_smoke/$s.jsonl
done

# Visualize 8 samples from test
python scripts/visualize_bas.py data/bas_smoke/test.jsonl --n 8 --outdir viz/test
```

#!/usr/bin/env python
"""
Shared data utilities for the causal DAG synthetic dataset.

Assumes JSONL files by default:
    ./data/train.jsonl
    ./data/val.jsonl
    ./data/test.jsonl

Each row / item is a dict with keys:
    "graph", "premise", "hypothesis", "label", and optionally "meta".
"""

import os
import json
from typing import Callable, Dict, List, Tuple

from datasets import Dataset

DATA_DIR = "./data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH   = os.path.join(DATA_DIR, "val.jsonl")
TEST_PATH  = os.path.join(DATA_DIR, "test.jsonl")


def load_json_mixed(path: str) -> List[Dict]:
    """
    Load either:
      - a single JSON list from file, or
      - JSONL (one JSON object per line).
    """
    with open(path, "r", encoding="utf-8") as f:
        pos = f.tell()
        first = f.read(1)
        f.seek(pos)

        if first == "[":
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]

    return data


def load_datasets(
    fmt_example: Callable[[Dict], Dict],
    train_path: str = TRAIN_PATH,
    val_path: str   = VAL_PATH,
    test_path: str  = TEST_PATH,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dataset, Dataset]:
    """
    Load raw train/val/test splits, then wrap train/val in HF Datasets.

    fmt_example: function mapping a raw dict -> {"text": prompt, ...}
    """
    print(f"Loading data from {os.path.dirname(train_path) or '.'} ...")

    train_raw = load_json_mixed(train_path)
    val_raw   = load_json_mixed(val_path)
    test_raw  = load_json_mixed(test_path)

    print(f"train_raw: {len(train_raw)}")
    print(f"val_raw:   {len(val_raw)}")
    print(f"test_raw:  {len(test_raw)}")

    train_ds = Dataset.from_list([fmt_example(x) for x in train_raw])
    val_ds   = Dataset.from_list([fmt_example(x) for x in val_raw])

    # test_raw remains a plain Python list; easier for custom evaluation.
    return train_raw, val_raw, test_raw, train_ds, val_ds
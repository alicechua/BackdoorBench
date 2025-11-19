#!/usr/bin/env python
"""
Shared evaluation utilities:
- predict_labels_batch: batched generation
- evaluate: compute binary metrics + dump JSON
"""

import json
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE = 16

# If you still sometimes use <answer>Yes</answer>, we keep this:
ANSWER_TAG_RE = re.compile(r"<answer>\s*(yes|no)\s*</answer>", re.IGNORECASE)
YESNO_RE      = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Batched prediction helper
# ---------------------------------------------------------------------------

def predict_labels_batch(
    model,
    tokenizer,
    batch_ex: List[Dict],
    build_prompt_fn: Callable[[Dict, bool], str],
    max_new_tokens: int = 4,
) -> Tuple[List[Optional[int]], List[str]]:
    """
    Run model.generate on a batch of examples and return predicted labels
    and raw texts.

    Returns:
      batch_preds: list of 0/1/None for each example
      batch_texts: list of raw decoded strings
    """
    # 1) Build prompts
    prompts = [build_prompt_fn(ex, with_label=False) for ex in batch_ex]

    # 2) Tokenize as a batch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 3) Generate for whole batch
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # 4) Decode & parse yes/no for each
    batch_preds: List[Optional[int]] = []
    batch_texts: List[str] = []

    # All prompts share the same padded length
    prompt_len = inputs["input_ids"].shape[1]

    for i in range(len(batch_ex)):
        gen_ids = out[i, prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # First try <answer>Yes/No</answer>
        m = ANSWER_TAG_RE.search(text)
        if m:
            ans = m.group(1).strip().lower()
        else:
            # Fallback: bare "yes"/"no"
            m2 = YESNO_RE.search(text)
            ans = m2.group(1).strip().lower() if m2 else None

        if ans == "yes":
            pred = 1
        elif ans == "no":
            pred = 0
        else:
            pred = None

        batch_preds.append(pred)
        batch_texts.append(text)

    return batch_preds, batch_texts


# ---------------------------------------------------------------------------
# Batched evaluation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    test_raw: List[Dict],
    build_prompt_fn: Callable[[Dict, bool], str],
    max_examples: Optional[int] = None,
    batch_size: int = BATCH_SIZE,
    output_json_path: Optional[str] = None,
    max_new_tokens: int = 4,
) -> Tuple[Dict, List[Dict]]:
    """
    Run evaluation on test_raw using *batched* generation.

    Returns:
      metrics, outputs
    """
    if max_examples is not None:
        test_raw = test_raw[:max_examples]

    model.eval()

    TP = TN = FP = FN = 0
    no_pred = 0
    total = len(test_raw)
    all_outputs: List[Dict] = []

    for start in tqdm(
        range(0, total, batch_size),
        desc="Evaluating on test",
        unit="batch",
    ):
        batch = test_raw[start:start + batch_size]
        preds, texts = predict_labels_batch(
            model,
            tokenizer,
            batch_ex=batch,
            build_prompt_fn=build_prompt_fn,
            max_new_tokens=max_new_tokens,
        )

        for offset, (ex, pred, raw_text) in enumerate(zip(batch, preds, texts)):
            idx = start + offset
            true_label = int(ex["label"])

            if pred not in (0, 1):
                no_pred += 1
                pred_clean = None
            else:
                pred_clean = int(pred)

            # Confusion matrix (treat "no prediction" as wrong)
            if true_label == 1:
                if pred_clean == 1:
                    TP += 1
                else:
                    FN += 1
            else:  # true_label == 0
                if pred_clean == 0:
                    TN += 1
                else:
                    FP += 1

            all_outputs.append(
                {
                    "index": idx,
                    "premise": ex.get("premise"),
                    "hypothesis": ex.get("hypothesis"),
                    "true_label": true_label,
                    "pred_label": pred_clean,
                    "raw_output": raw_text,
                    "valid_prediction": pred_clean is not None,
                }
            )

    def safe_div(num, denom):
        return float(num) / float(denom) if denom else 0.0

    accuracy = safe_div(TP + TN, total)
    precision_1 = safe_div(TP, TP + FP)
    recall_1    = safe_div(TP, TP + FN)
    f1_1        = safe_div(2 * precision_1 * recall_1,
                           precision_1 + recall_1) if (precision_1 + recall_1) else 0.0
    support_1   = TP + FN

    precision_0 = safe_div(TN, TN + FN)
    recall_0    = safe_div(TN, TN + FP)
    f1_0        = safe_div(2 * precision_0 * recall_0,
                           precision_0 + recall_0) if (precision_0 + recall_0) else 0.0
    support_0   = TN + FP

    metrics = {
        "total_examples": total,
        "no_prediction": no_pred,
        "accuracy": accuracy,
        "class_0": {
            "precision": precision_0,
            "recall": recall_0,
            "f1": f1_0,
            "support": support_0,
        },
        "class_1": {
            "precision": precision_1,
            "recall": recall_1,
            "f1": f1_1,
            "support": support_1,
        },
        "confusion_matrix": {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
        },
    }

    if output_json_path is not None:
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics,
                    "outputs": all_outputs,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    return metrics, all_outputs
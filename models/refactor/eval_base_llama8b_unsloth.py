#!/usr/bin/env python
"""
Evaluate the *base* Unsloth Meta-Llama-3.1-8B-bnb-4bit model
on the test set, without any finetuning.
"""

import json
import os
from typing import Dict

from common_data import load_datasets
from common_model import create_model, prepare_for_inference
from common_eval import evaluate

# Reuse the non-contextual SYSTEM / RESPONSE_TAG style
RESPONSE_TAG = "### Answer:\n"
SYSTEM = (
    "You are a precise reasoning assistant. "
    "Given a textual premise and a hypothesis, decide whether the hypothesis "
    "is logically entailed by the premise. "
    "Explain your reasoning briefly, then give the final answer as "
    "Yes or No inside <answer></answer> tags."
)


def build_prompt(ex: Dict, with_label: bool) -> str:
    """
    Same non-contextual prompt as in finetune_llama8b_unsloth.py,
    but ignoring the gold label when with_label=False.
    """
    premise    = (ex.get("premise", "") or "").strip()
    hypothesis = (ex.get("hypothesis", "") or "").strip()
    label_int  = int(ex["label"])
    gold_ans   = "Yes" if label_int == 1 else "No"

    user_block = (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Question: Is the hypothesis true under the premise? "
        "Concisely explain and answer 'Yes' or 'No'. "
        f"{RESPONSE_TAG}"
    )

    full_prompt = f"{SYSTEM}\n\n{user_block}"
    if with_label:
        full_prompt += gold_ans
    return full_prompt


# We only need fmt_example to satisfy the load_datasets API,
# but train/val datasets are not actually used here.
def _dummy_fmt_example(ex: Dict) -> Dict:
    return {"text": ""}


def main():
    max_seq_length = 2048

    # 1) Load data (we only really care about test_raw)
    _, _, test_raw, _, _ = load_datasets(_dummy_fmt_example)

    # 2) Load base model WITHOUT LoRA
    model, tokenizer = create_model(
        max_seq_length=max_seq_length,
        use_lora=False,
    )
    model = prepare_for_inference(model)

    # 3) Evaluate
    metrics, test_outputs = evaluate(
        model,
        tokenizer,
        test_raw=test_raw,
        build_prompt_fn=build_prompt,
        max_examples=None,
        output_json_path="results/test_outputs_base_model.json",
        max_new_tokens=128,
    )

    # 4) Save metrics
    os.makedirs("results", exist_ok=True)
    with open("results/metrics_base_model.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Optional: save detailed JSONL as well
    with open("results/test_results_base_model.jsonl", "w", encoding="utf-8") as f:
        for row in test_outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Base model evaluation complete.")


if __name__ == "__main__":
    main()
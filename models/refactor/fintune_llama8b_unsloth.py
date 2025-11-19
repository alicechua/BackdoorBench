#!/usr/bin/env python
"""
Finetune Llama 3.1 8B with Unsloth on causal DAG dataset (non-contextual).
"""

import json
from typing import Dict

from common_data import load_datasets
from common_model import create_model, finetune, prepare_for_inference
from common_eval import evaluate

# ---------------------------------------------------------------------------
# Prompt configuration (non-contextual)
# ---------------------------------------------------------------------------

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
    Turn one raw example into a plain text prompt.

    If with_label=True, append the gold Yes/No answer.
    If with_label=False, leave the answer blank (for inference).
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


def fmt_example(ex: Dict) -> Dict:
    """Format one example for SFTTrainer (expects a 'text' field)."""
    return {"text": build_prompt(ex, with_label=True)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    max_seq_length = 2048

    # 1) Data
    train_raw, val_raw, test_raw, train_ds, val_ds = load_datasets(fmt_example)

    # 2) Model + LoRA
    model, tokenizer = create_model(max_seq_length=max_seq_length, use_lora=True)
    model, tokenizer = finetune(
        model,
        tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        max_seq_length=max_seq_length,
        output_dir="output_base",
    )

    # 3) Inference and evaluation
    model = prepare_for_inference(model)

    metrics, test_outputs = evaluate(
        model,
        tokenizer,
        test_raw=test_raw,
        build_prompt_fn=build_prompt,
        max_examples=None,
        output_json_path="results/test_outputs_base.json",
        max_new_tokens=128,
    )

    # 4) Save metrics and per-example outputs (JSONL)
    os.makedirs("results", exist_ok=True)
    with open("results/metrics_base.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open("results/test_results_base.jsonl", "w", encoding="utf-8") as f:
        for row in test_outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 5) Save LoRA-merged model
    model.save_pretrained_merged("model_lora_base", tokenizer, save_method="lora")


if __name__ == "__main__":
    import os
    main()
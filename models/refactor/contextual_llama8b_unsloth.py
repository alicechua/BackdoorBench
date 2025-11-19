#!/usr/bin/env python
"""
Contextual finetune of Llama 3.1 8B with Unsloth on causal DAG dataset.

Uses contextual_prompts.jsonl, with:
  - kind: "general" or "negative"
  - text: the contextual text
  - neg_type: for negative prompts, matches ex["meta"]["neg_type"]
"""

import json
import os
import random
from typing import Dict, List, Tuple

from common_data import load_datasets
from common_model import create_model, finetune, prepare_for_inference
from common_eval import evaluate

# ---------------------------------------------------------------------------
# Prompt configuration (contextual)
# ---------------------------------------------------------------------------

CONTEXT_PATH = "./contextual_prompts.json"
rng = random.Random(0)

RESPONSE_TAG = "### Answer:\n"
SYSTEM = (
    "You are a precise reasoning assistant. "
    "Given a textual premise and a hypothesis, decide whether the hypothesis "
    "is logically entailed by the premise. "
    "Explain your reasoning briefly, then give the final answer as "
    "Yes or No inside <answer></answer> tags."
)


def load_contextual_prompts(path: str) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Load contextual prompts from a JSONL file.

    Each line is a JSON object with keys:
      - "kind": "general" or "negative"
      - "text": the contextual string
      - "neg_type": for negative prompts, the negative type it applies to
    """
    prompts: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    general_prompts = [p for p in prompts if p.get("kind") == "general"]
    negative_by_type: Dict[str, List[Dict]] = {}
    for p in prompts:
        if p.get("kind") == "negative":
            nt = p.get("neg_type")
            if nt:
                negative_by_type.setdefault(nt, []).append(p)

    if not general_prompts:
        raise ValueError("No general prompts found in contextual_prompts.json")

    return general_prompts, negative_by_type


general_prompts, negative_prompts_by_type = load_contextual_prompts(CONTEXT_PATH)


def pick_context_prompt(ex: Dict) -> str:
    """
    Positive example -> randomly choose a general contextual prompt.
    Negative example -> choose matching neg_type prompt if available.
    """
    label = int(ex["label"])
    meta = ex.get("meta", {}) or {}
    neg_type = meta.get("neg_type")

    # Positive example or no neg_type → any general contextual prompt
    if label == 1 or not neg_type:
        return rng.choice(general_prompts)["text"]

    # Negative example → try to match neg_type
    candidates = negative_prompts_by_type.get(neg_type)
    if candidates:
        return rng.choice(candidates)["text"]

    # Fallback if neg_type not found
    return rng.choice(general_prompts)["text"]


def build_prompt(ex: Dict, with_label: bool) -> str:
    """
    Build a contextual prompt, including a relevant contextual snippet.
    """
    premise    = (ex.get("premise", "") or "").strip()
    hypothesis = (ex.get("hypothesis", "") or "").strip()
    label_int  = int(ex["label"])
    gold_ans   = "Yes" if label_int == 1 else "No"

    ctx_text = pick_context_prompt(ex)

    user_block = (
        f"Context: {ctx_text}\n"
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
        output_dir="output_contextual",
    )

    # 3) Inference and evaluation
    model = prepare_for_inference(model)

    metrics, test_outputs = evaluate(
        model,
        tokenizer,
        test_raw=test_raw,
        build_prompt_fn=build_prompt,
        max_examples=None,
        output_json_path="results/test_outputs_cft.json",
        max_new_tokens=128,
    )

    os.makedirs("results", exist_ok=True)
    with open("results/metrics_cft.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open("results/test_results_cft.jsonl", "w", encoding="utf-8") as f:
        for row in test_outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 4) Save LoRA-merged model
    model.save_pretrained_merged("model_lora_cft", tokenizer, save_method="lora")


if __name__ == "__main__":
    main()

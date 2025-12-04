#!/usr/bin/env python
"""
Finetune Llama 3.1 8B with Unsloth on causal DAG dataset.

Assumptions:
- You already have the data files:
    ./data/train.json
    ./data/val.json
    ./data/test.json
- Each JSON file is either:
    * a list of dicts, or
    * JSONL (one JSON object per line)
- Each example dict has keys: "graph", "premise", "hypothesis", "label".
"""

import os
import json
import re
from typing import Dict, List

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth import train_on_responses_only

import torch
from datasets import Dataset
from tqdm.auto import tqdm
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import PeftModel

RUN_ID = "instruct_finetune_prompt_complete"

# ---------------------------------------------------------------------------
# 1. Data loading & prompt formatting
# ---------------------------------------------------------------------------

DATA_DIR = "./data"
TEST_PATH  = os.path.join(DATA_DIR, "test.jsonl")
MODEL_CHECKPOINT_PATH = "./checkpoints/instruct_cft_general/model_lora_instruct_cft_general"

SYSTEM = (
    "You are a precise reasoning assistant. "
    "Given a textual premise and a hypothesis, decide whether the hypothesis "
    "is logically entailed by the premise. "
    "Explain your reasoning briefly, then give the final answer as "
    "Yes or No"
)


def load_json_mixed(path: str) -> List[Dict]:
    """
    Load either:
      - a JSON list of dicts, or
      - JSONL (one JSON per line).
    """
    with open(path, "r", encoding="utf-8") as f:
        # Peek first non-whitespace char
        pos = f.tell()
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        f.seek(pos)

        if first == "[":
            # Plain JSON list
            data = json.load(f)
        else:
            # JSONL
            data = [json.loads(line) for line in f if line.strip()]
    return data

def to_messages(ex):
    # Build the user prompt (no answer)
    user_text = (
        f"Premise: {ex.get('premise','').strip()}\n"
        f"Hypothesis: {ex.get('hypothesis','').strip()}\n"
        "Question: Is the hypothesis true under the premise? "
        "Concisely explain and answer 'Yes' or 'No'."
    )
    gold_ans = "Yes" if int(ex['label']) == 1 else "No"
    messages = [
        {"role": "system",    "content": SYSTEM.strip()},
        {"role": "user",      "content": user_text},
        {"role": "assistant", "content": gold_ans},
    ]
    return messages

def make_formatting_func(tokenizer):
    def formatting_func(examples):
        texts = []
        for premise, hyp, lbl in zip(examples['premise'], examples['hypothesis'], examples['label']):
            ex = {'premise': premise, 'hypothesis': hyp, 'label': lbl}
            messages = to_messages(ex)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {'text': texts}
    return formatting_func

# ---------------------------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------------------------

def load_datasets(tokenizer):
    print(f"Loading data from {DATA_DIR} ...")
    test_raw  = load_json_mixed(TEST_PATH)

    print(f"test_raw:  {len(test_raw)}")

    fmt = make_formatting_func(tokenizer)
    # We keep test_raw as a plain Python list for easier custom evaluation
    return test_raw


# ---------------------------------------------------------------------------
# 3. Model loading & finetuning
# ---------------------------------------------------------------------------
def load_from_checkpoint(max_seq_length: int = 2048, checkpoint_dir: str = MODEL_CHECKPOINT_PATH):
    print("Loading base model (Llama 3.1 8B 4-bit) ...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        max_seq_length = max_seq_length,
        load_in_4bit   = True,
        dtype          = None,  # let Unsloth pick
    )

    print(f"Loading LoRA adapter from {checkpoint_dir} ...")
    model = PeftModel.from_pretrained(
        base_model,
        checkpoint_dir,
        is_trainable = False,   # inference only
    )

    # Optimize for inference (e.g. sets requires_grad=False, maybe hooks, etc.)
    model = FastLanguageModel.for_inference(model)
    model.eval()

    return model, tokenizer

# ---------------------------------------------------------------------------
# 4. Evaluation on test set
# ---------------------------------------------------------------------------
def predict(model, tokenizer, ex):
    # build the chat messages without answer
    user_prompt = (
        f"Premise: {ex.get('premise','').strip()}\n"
        f"Hypothesis: {ex.get('hypothesis','').strip()}\n"
        "Question: Is the hypothesis true under the premise? "
        "Concisely explain and answer 'Yes' or 'No'."
    )
    messages = [
        {"role": "system", "content": SYSTEM.strip()},
        {"role": "user",   "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=False,
        )
    # Remove the prompt part and decode the answer
    generated_ids = out[0, input_ids.shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def predict_label(model, tokenizer, ex: Dict, max_new_tokens: int = 512):
    """
    Run the finetuned model on a single example and try to extract a 0/1 label.
    """
    text = predict(model, tokenizer, ex)

    # Find the first "yes" or "no" (case-insensitive, word boundary)
    m = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if m:
        ans = m.group(1).strip().lower()
        if ans == "yes":
            return 1, text
        elif ans == "no":
            return 0, text

    # If we can't parse a clean yes/no, return None
    return None, text

def evaluate(model, tokenizer, test_raw, max_examples: int | None = None):
    """
    Evaluate binary classification metrics on the test set and
    return (metrics_dict, results_list).

    results_list is a list of dicts like:
      {
        "index": int,
        "premise": str,
        "hypothesis": str,
        "true_label": 0/1,
        "pred_label": 0/1 or None,
        "raw_output": str,
        "valid_prediction": bool,
      }
    """
    if max_examples is not None:
        test_raw = test_raw[:max_examples]

    model.eval()

    TP = TN = FP = FN = 0
    no_pred = 0
    total = len(test_raw)
    all_outputs = []

    results = []

    for idx, ex in enumerate(tqdm(test_raw, desc="Evaluating on test", unit="example")):
        true_label = int(ex["label"])
        pred, raw_text = predict_label(model, tokenizer, ex)

        if pred not in (0, 1):
            no_pred += 1
            pred_clean = None
        else:
            pred_clean = int(pred)

        # Confusion matrix (treat None as incorrect)
        if true_label == 1:
            if pred_clean == 1:
                TP += 1
            else:
                FN += 1
        elif true_label == 0:
            if pred_clean == 0:
                TN += 1
            else:
                FP += 1

        # ---- Collect for JSON dump ----
        all_outputs.append({
            "index": idx,
            "premise": ex.get("premise"),
            "hypothesis": ex.get("hypothesis"),
            "true_label": true_label,
            "pred_label": pred_clean,
            "raw_output": raw_text,
        })

    # Metrics
    def safe_div(num, denom):
        return num / denom if denom > 0 else 0.0

    accuracy = safe_div(TP + TN, total)
    precision_1 = safe_div(TP, TP + FP)
    recall_1    = safe_div(TP, TP + FN)
    f1_1        = safe_div(2 * precision_1 * recall_1, precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0
    support_1   = TP + FN

    precision_0 = safe_div(TN, TN + FN)
    recall_0    = safe_div(TN, TN + FP)
    f1_0        = safe_div(2 * precision_0 * recall_0, precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0
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
            "FP": FP,
            "FN": FN,
            "TN": TN,
        },
    }

    # ---- Dump to JSON file ----
    os.makedirs("results", exist_ok=True)
    test_oupts = RUN_ID+"/test_outputs_"+RUN_ID+".json"
    with open(test_oupts, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "outputs": all_outputs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved detailed outputs to {test_oupts}")
    return metrics, all_outputs

# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    # --- Save everything ---
    os.makedirs(RUN_ID, exist_ok=True)
    max_seq_length = 2048
    
    model, tokenizer = load_from_checkpoint(max_seq_length=2048, checkpoint_dir=MODEL_CHECKPOINT_PATH)

    test_raw = load_datasets(tokenizer)
    

    # Prepare model for inference
    FastLanguageModel.for_inference(model)

    # Evaluate on full test set (change max_examples if you want a quick check)
    metrics, test_results = evaluate(model, tokenizer, test_raw, max_examples=None)

    # 1) Save metrics as JSON
    with open(f"{RUN_ID}/metrics_{RUN_ID}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2) Save per-example test results as JSONL (one example per line)
    with open(f"{RUN_ID}/test_results_{RUN_ID}.jsonl", "w", encoding="utf-8") as f:
        for row in test_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3) Save the model (LoRA)
    model.save_pretrained_merged(f"{RUN_ID}/model_lora_{RUN_ID}", tokenizer, save_method="lora")  # depending on version


if __name__ == "__main__":
    main()
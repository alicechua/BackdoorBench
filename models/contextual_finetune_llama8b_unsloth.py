#!/usr/bin/env python
"""
Contextual Finetune Llama 3.1 8B with Unsloth on causal DAG dataset.

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

import json, random, os, re
from typing import Dict, List

from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from trl import SFTTrainer
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# 1. Data loading & prompt formatting
# ---------------------------------------------------------------------------

DATA_DIR = "./data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH   = os.path.join(DATA_DIR, "val.jsonl")
TEST_PATH  = os.path.join(DATA_DIR, "test.jsonl")

CONTEXT_PATH = "./contextual_prompts.json"  # change if needed
rng = random.Random(0)

# RESPONSE_TAG = "### Answer (0 or 1 only):\n"
# SYSTEM = (
#     "You are a precise reasoning assistant. "
#     "Given a textual premise and a hypothesis, decide whether the hypothesis "
#     "is logically entailed by the premise. "
#     "Only output a single digit: 0 if it is NOT entailed, or 1 if it IS entailed."
# )
RESPONSE_TAG = "### Answer:\n"
SYSTEM = (
    "You are a precise reasoning assistant. "
    "Given a textual premise and a hypothesis, decide whether the hypothesis "
    "is logically entailed by the premise. "
    "Explain your reasoning briefly, then give the final answer as "
    "Yes or No inside <answer></answer> tags."
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

def load_contextual_prompts(path: str):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    general_prompts = [p for p in prompts if p.get("kind") == "general"]
    negative_by_type = {}
    for p in prompts:
        if p.get("kind") == "negative":
            nt = p.get("neg_type")
            if nt:
                negative_by_type.setdefault(nt, []).append(p)

    return general_prompts, negative_by_type

general_prompts, negative_prompts_by_type = load_contextual_prompts(CONTEXT_PATH)

def pick_context_prompt(ex: Dict) -> str:
    label = int(ex["label"])
    meta  = ex.get("meta", {}) or {}
    neg_type = meta.get("neg_type")

    # Positive example → any general contextual prompt
    if label == 1 or not neg_type:
        return rng.choice(general_prompts)["text"]

    # Negative example → use matching neg_type if available
    candidates = negative_prompts_by_type.get(neg_type)
    if candidates:
        return rng.choice(candidates)["text"]

    # Fallback if neg_type not found
    return rng.choice(general_prompts)["text"]

def build_prompt(ex: Dict, with_label: bool) -> str:
    """
    Turn one raw example into a plain text prompt.

    If with_label=True, append the gold 0/1 label (for supervised finetuning).
    If with_label=False, leave the answer blank (for inference).
    """
    graph_str = json.dumps(ex["graph"], sort_keys=True)
    premise   = (ex.get("premise", "") or "").strip()
    hypothesis= (ex.get("hypothesis", "") or "").strip()
    label_int = int(ex["label"])
    label_str = str(label_int)  # "0" or "1"
    gold_ans   = "Yes" if label_int == 1 else "No"

    # user_block = (
    #     "### Task:\n"
    #     "Decide if the hypothesis logically follows from the premise.\n\n"
    #     f"### Premise:\n{premise}\n\n"
    #     f"### Hypothesis:\n{hypothesis}\n\n"
    #     "Respond with '0' if the hypothesis is NOT entailed by the premise, "
    #     "or '1' if it IS entailed.\n\n"
    #     f"{RESPONSE_TAG}"
    # )

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
        # full_prompt += label_str
        full_prompt += gold_ans
    return full_prompt


def fmt_example(ex: Dict) -> Dict:
    """Format one example for SFTTrainer (expects a 'text' field)."""
    return {"text": build_prompt(ex, with_label=True)}


# ---------------------------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------------------------

def load_datasets():
    print(f"Loading data from {DATA_DIR} ...")
    train_raw = load_json_mixed(TRAIN_PATH)
    val_raw   = load_json_mixed(VAL_PATH)
    test_raw  = load_json_mixed(TEST_PATH)

    print(f"train_raw: {len(train_raw)}")
    print(f"val_raw:   {len(val_raw)}")
    print(f"test_raw:  {len(test_raw)}")

    train_ds = Dataset.from_list([fmt_example(x) for x in train_raw])
    val_ds   = Dataset.from_list([fmt_example(x) for x in val_raw])
    # We keep test_raw as a plain Python list for easier custom evaluation
    return train_raw, val_raw, test_raw, train_ds, val_ds


# ---------------------------------------------------------------------------
# 3. Model loading & finetuning
# ---------------------------------------------------------------------------

def create_model(max_seq_length: int = 2048):
    print("Loading base model (Llama 3.1 8B 4-bit) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        max_seq_length  = max_seq_length,
        load_in_4bit    = True,
        dtype           = None,  # let Unsloth pick (bf16/fp16) based on GPU
    )

    print("Wrapping with LoRA (PEFT) ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj",
            "up_proj", "down_proj", "o_proj", "gate_proj",
        ],
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
    )
    print(model.print_trainable_parameters())
    return model, tokenizer


def finetune(model, tokenizer, train_ds, val_ds, max_seq_length: int = 2048):
    print("Starting finetuning ...")

    training_args = TrainingArguments(
        output_dir="output",
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,          # change if you want to train longer
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        report_to="none",
        seed=0,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=training_args,
    )

    trainer.train()
    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. Evaluation on test set
# ---------------------------------------------------------------------------

def predict_label(model, tokenizer, ex: Dict, max_new_tokens: int = 512):
    """
    Run the finetuned model on a single example and try to extract a 0/1 label.
    """
    prompt = build_prompt(ex, with_label=False)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # # Grab first 0/1 we see
    # m = re.search(r"[01]", text)
    # if m:
    #     return int(m.group(0)), text
    # return None, text

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
    with open("results/test_outputs_cft.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "outputs": all_outputs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved detailed outputs to results/test_outputs_cft.json")
    return metrics, all_outputs

# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    _, _, test_raw, train_ds, val_ds = load_datasets()

    max_seq_length = 2048

    model, tokenizer = create_model(max_seq_length=max_seq_length)
    model, tokenizer = finetune(
        model,
        tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        max_seq_length=max_seq_length,
    )

    # Prepare model for inference
    FastLanguageModel.for_inference(model)

    # Evaluate on full test set (change max_examples if you want a quick check)
    metrics, test_results = evaluate(model, tokenizer, test_raw, max_examples=None)

    # --- Save everything ---
    os.makedirs("results", exist_ok=True)

    # 1) Save metrics as JSON
    with open("results/metrics_cft.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 2) Save per-example test results as JSONL (one example per line)
    with open("results/test_results_cft.jsonl", "w", encoding="utf-8") as f:
        for row in test_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 3) Save the model (LoRA)
    model.save_pretrained_merged("model_cft_lora_", tokenizer, save_method="lora")  # depending on version


if __name__ == "__main__":
    main()
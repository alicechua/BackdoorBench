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
import random

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from unsloth import train_on_responses_only

import torch
import torch.nn as nn
from datasets import Dataset
from tqdm.auto import tqdm
from trl import SFTTrainer
from transformers import TrainingArguments

RUN_ID = "learnable_tokens_cft_general"
N_SOFT_TOKENS = 10

# ---------------------------------------------------------------------------
# 1. Data loading & prompt formatting
# ---------------------------------------------------------------------------

DATA_DIR = "./data"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_PATH   = os.path.join(DATA_DIR, "val.jsonl")
TEST_PATH  = os.path.join(DATA_DIR, "test.jsonl")

CONTEXT_PATH = "./models/contextual_prompts.json"  # change if needed
rng = random.Random(0)

SYSTEM = (
    "You are a precise reasoning assistant. "
    "Given a textual premise and a hypothesis, decide whether the hypothesis "
    "is logically entailed by the premise. "
    "Explain your reasoning briefly, then give the final answer as "
    "Yes or No"
)

INFIX_MARKER = "<INFIX>"


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
    return rng.choice(general_prompts)["text"]

def to_messages(ex):
    ctx_text = pick_context_prompt(ex)
    # Build the user prompt (no answer)
    user_text = (
        f"Context: {ctx_text}\n{INFIX_MARKER}"
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

class SoftPromptWrapper(nn.Module):
    def __init__(self, base_model, tokenizer, n_tokens=10):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.n_tokens = n_tokens
        self.config = base_model.config
        
        embed_dim = base_model.config.hidden_size
        self.soft_prompt = nn.Parameter(torch.randn(n_tokens, embed_dim) * 0.01)
        
        self.infix_marker = INFIX_MARKER
        
    @property
    def device(self):
        return self.base_model.device
    
    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()
    
    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)
    
    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)
    
    def is_loaded_in_4bit(self):
        return hasattr(self.base_model, 'is_loaded_in_4bit') and self.base_model.is_loaded_in_4bit()
    
    def is_loaded_in_8bit(self):
        return hasattr(self.base_model, 'is_loaded_in_8bit') and self.base_model.is_loaded_in_8bit()
    
    def named_parameters(self, prefix='', recurse=True):
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
            yield name, param
    
    def parameters(self, recurse=True):
        for param in super().parameters(recurse=recurse):
            yield param
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_model, name)
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is not None:
            return self.base_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        batch_size = input_ids.shape[0]
        embed_layer = self.base_model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)
        
        marker_token_id = self.tokenizer.encode(self.infix_marker, add_special_tokens=False)[0]
        
        new_embeds_list = []
        new_attention_mask_list = []
        new_labels_list = []
        
        for i in range(batch_size):
            marker_positions = (input_ids[i] == marker_token_id).nonzero(as_tuple=True)[0]
            
            if len(marker_positions) == 0:
                new_embeds_list.append(input_embeds[i])
                if attention_mask is not None:
                    new_attention_mask_list.append(attention_mask[i])
                if labels is not None:
                    new_labels_list.append(labels[i])
            else:
                marker_pos = marker_positions[0].item()
                
                before_embeds = input_embeds[i, :marker_pos]
                after_embeds = input_embeds[i, marker_pos+1:]
                
                new_embed = torch.cat([
                    before_embeds,
                    self.soft_prompt.to(input_embeds.device),
                    after_embeds
                ], dim=0)
                new_embeds_list.append(new_embed)
                
                if attention_mask is not None:
                    before_mask = attention_mask[i, :marker_pos]
                    after_mask = attention_mask[i, marker_pos+1:]
                    soft_mask = torch.ones(self.n_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_mask = torch.cat([before_mask, soft_mask, after_mask], dim=0)
                    new_attention_mask_list.append(new_mask)
                
                if labels is not None:
                    before_labels = labels[i, :marker_pos]
                    after_labels = labels[i, marker_pos+1:]
                    soft_labels = torch.full((self.n_tokens,), -100, dtype=labels.dtype, device=labels.device)
                    new_label = torch.cat([before_labels, soft_labels, after_labels], dim=0)
                    new_labels_list.append(new_label)
        
        max_len = max(emb.shape[0] for emb in new_embeds_list)
        embed_dim = new_embeds_list[0].shape[1]
        
        padded_embeds = torch.zeros(batch_size, max_len, embed_dim, device=input_embeds.device, dtype=input_embeds.dtype)
        padded_attention_mask = torch.zeros(batch_size, max_len, device=input_ids.device, dtype=torch.long) if attention_mask is not None else None
        padded_labels = torch.full((batch_size, max_len), -100, device=input_ids.device, dtype=torch.long) if labels is not None else None
        
        for i, emb in enumerate(new_embeds_list):
            seq_len = emb.shape[0]
            padded_embeds[i, :seq_len] = emb
            if attention_mask is not None:
                padded_attention_mask[i, :seq_len] = new_attention_mask_list[i]
            if labels is not None:
                padded_labels[i, :seq_len] = new_labels_list[i]
        
        outputs = self.base_model(
            inputs_embeds=padded_embeds,
            attention_mask=padded_attention_mask,
            labels=padded_labels,
            **kwargs
        )
        
        return outputs

# ---------------------------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------------------------

def load_datasets(tokenizer):
    print(f"Loading data from {DATA_DIR} ...")
    train_raw = load_json_mixed(TRAIN_PATH)
    val_raw   = load_json_mixed(VAL_PATH)
    test_raw  = load_json_mixed(TEST_PATH)

    print(f"train_raw: {len(train_raw)}")
    print(f"val_raw:   {len(val_raw)}")
    print(f"test_raw:  {len(test_raw)}")

    fmt = make_formatting_func(tokenizer)

    # train_ds = Dataset.from_list([fmt_example(x) for x in train_raw])
    # val_ds   = Dataset.from_list([fmt_example(x) for x in val_raw])
    train_ds_formatted = Dataset.from_list(train_raw).map(fmt, batched=True)
    val_ds_formatted   = Dataset.from_list(val_raw).map(fmt, batched=True)
    # We keep test_raw as a plain Python list for easier custom evaluation
    return train_raw, val_raw, test_raw, train_ds_formatted, val_ds_formatted#train_ds, val_ds


# ---------------------------------------------------------------------------
# 3. Model loading & finetuning
# ---------------------------------------------------------------------------

def create_model(max_seq_length: int = 2048):
    print("Loading base model (Llama 3.1 8B 16-bit) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = "unsloth/Meta-Llama-3.1-8B-Instruct",
        max_seq_length  = max_seq_length,
        load_in_4bit    = False,
        dtype           = torch.float16 if torch.cuda.is_available() else None,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    
    if INFIX_MARKER not in tokenizer.get_vocab():
        tokenizer.add_tokens([INFIX_MARKER])
        model.resize_token_embeddings(len(tokenizer))

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
    
    print("Checking LoRA setup...")
    lora_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA params (before wrapping): {lora_params_before:,}")
    
    if lora_params_before == 0:
        print("WARNING: No trainable parameters found in base model!")
        print("Checking all parameters:")
        for name, param in model.named_parameters():
            print(f"  {name}: requires_grad={param.requires_grad}, shape={param.shape}")
    
    print("Wrapping with soft prompt...")
    wrapped_model = SoftPromptWrapper(model, tokenizer, n_tokens=N_SOFT_TOKENS)
    
    wrapped_model.soft_prompt.requires_grad = True
    
    n_trainable = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in wrapped_model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.4f}%)")
    
    lora_params = sum(p.numel() for n, p in wrapped_model.base_model.named_parameters() if p.requires_grad)
    print(f"LoRA params (after wrapping): {lora_params:,}")
    print(f"Soft prompt params: {wrapped_model.soft_prompt.numel():,}")
    
    return wrapped_model, tokenizer


def finetune(model, tokenizer, train_ds, val_ds, max_seq_length: int = 2048):
    print("Starting finetuning ...")

    training_args = TrainingArguments(
        output_dir=f"{RUN_ID}/output_{RUN_ID}",
        learning_rate=1e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        warmup_steps=10,
        weight_decay=0.01,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        bf16=False,
        optim="adamw_torch",
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
        packing=False,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer.train()
    return model, tokenizer

# ---------------------------------------------------------------------------
# 4. Evaluation on test set
# ---------------------------------------------------------------------------
def predict(model, tokenizer, ex):
    user_prompt = (
        f"{INFIX_MARKER}Premise: {ex.get('premise','').strip()}\n"
        f"Hypothesis: {ex.get('hypothesis','').strip()}\n"
        "Question: Is the hypothesis true under the premise? "
        "Concisely explain and answer 'Yes' or 'No'."
    )
    messages = [
        {"role": "system", "content": SYSTEM.strip()},
        {"role": "user",   "content": user_prompt},
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    
    # Get model device
    model_device = next(model.base_model.parameters()).device
    input_ids = input_ids.to(model_device)
    
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(input_ids)
        
        marker_token_id = tokenizer.encode(INFIX_MARKER, add_special_tokens=False)[0]
        marker_positions = (input_ids[0] == marker_token_id).nonzero(as_tuple=True)[0]
        
        if len(marker_positions) > 0:
            marker_pos = marker_positions[0].item()
            before_embeds = input_embeds[0, :marker_pos]
            after_embeds = input_embeds[0, marker_pos+1:]
            
            input_embeds = torch.cat([
                before_embeds.unsqueeze(0),
                model.soft_prompt.to(input_embeds.device).unsqueeze(0),
                after_embeds.unsqueeze(0)
            ], dim=1)
        
        # Simple autoregressive generation with forward pass
        generated_ids = []
        current_embeds = input_embeds
        
        for _ in range(128):  # max_new_tokens
            outputs = model.base_model(inputs_embeds=current_embeds)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)
            
            # Stop if EOS token
            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
            generated_ids.append(next_token_id.item())
            
            # Get embedding for next token and append
            next_token_embed = model.get_input_embeddings()(next_token_id.unsqueeze(0))
            current_embeds = torch.cat([current_embeds, next_token_embed], dim=1)
        
        # Combine input_ids with generated_ids for decoding
        out = torch.cat([input_ids[0], torch.tensor(generated_ids, device=input_ids.device)])
    
    return tokenizer.decode(out, skip_special_tokens=True)

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
    
    model, tokenizer = create_model(max_seq_length=max_seq_length)

    _, _, test_raw, train_ds, val_ds = load_datasets(tokenizer)
    
    model, tokenizer = finetune(
        model,
        tokenizer,
        train_ds=train_ds,
        val_ds=val_ds,
        max_seq_length=max_seq_length,
    )

    model.eval()

    # Save first

    # Save only LoRA adapters
    model.base_model.save_pretrained(f"{RUN_ID}/lora_adapters")
    tokenizer.save_pretrained(f"{RUN_ID}/lora_adapters")
    
    # Save soft prompt separately
    torch.save({
        'soft_prompt': model.soft_prompt.data,
        'n_tokens': model.n_tokens,
        'infix_marker': model.infix_marker,
    }, f"{RUN_ID}/soft_prompt.pt")
    
    print(f"\nSaved LoRA adapters to {RUN_ID}/lora_adapters/")
    print(f"Saved soft prompt to {RUN_ID}/soft_prompt.pt")

    # --- Evaluate on test set ---
    metrics, test_results = evaluate(model, tokenizer, test_raw, max_examples=None)

    with open(f"{RUN_ID}/metrics_{RUN_ID}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(f"{RUN_ID}/test_results_{RUN_ID}.jsonl", "w", encoding="utf-8") as f:
        for row in test_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
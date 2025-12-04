import os
import torch
import math
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import (
    AutoTokenizer,
    BertConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    BertForSequenceClassification,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import json
import random
from typing import List, Dict

#-------#Code taken from ChatGPT #-------#
BertConfig.use_sdpa = False

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x, sin, cos):
    return (x * cos) + (rotate_half(x) * sin)


class RoPEBertSelfAttention(BertSelfAttention):
    def forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    past_key_values=None,
    **kwargs,
    ):
        mixed_query_layer = self.query(hidden_states)

        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            query_layer = self.transpose_for_scores(mixed_query_layer)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            query_layer = self.transpose_for_scores(mixed_query_layer)

        B, H, L, D = query_layer.size()
        if D % 2 != 0:
            raise ValueError("RoPE requires even head_dim")

        device = query_layer.device
        dtype = query_layer.dtype

        # ---- build RoPE frequencies ----
        pos = torch.arange(L, device=device, dtype=dtype)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, D, 2, device=device, dtype=dtype) / D))

        freqs = torch.einsum("i,j->ij", pos, inv_freq)  # [L, D/2]
        sin = freqs.sin()
        cos = freqs.cos()

        # reshape to broadcast to [B,H,L,D]
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)

        # ---- apply RoPE ----
        query_layer = apply_rope(query_layer, sin, cos)
        key_layer = apply_rope(key_layer, sin, cos)

        # ---- continue normal BERT attention ----
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= math.sqrt(D)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(B, L, self.all_head_size)

        return (context_layer, attention_probs) if output_attentions else (context_layer,)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) 
        return x.permute(0, 2, 1, 3)

def patch_bert_with_rope(model):
    if not hasattr(model, "bert"):
        raise ValueError("Model must have model.bert attribute")

    for layer in model.bert.encoder.layer:
        old_self = layer.attention.self

        # Case 1: Old style BERT (has .config)
        if hasattr(old_self, "config"):
            cfg = old_self.config

        else:
            # Case 2: SDPA injected BERT (no .config)
            # Extract dimensions directly from the module
            num_heads = old_self.num_attention_heads
            head_size = old_self.attention_head_size
            hidden_size = num_heads * head_size

            # Extract dropout probability safely
            if isinstance(old_self.dropout, torch.nn.Dropout):
                dropout_p = old_self.dropout.p
            else:
                dropout_p = 0.1  # default

            # Rebuild minimal BertConfig
            from transformers import BertConfig
            cfg = BertConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                attention_probs_dropout_prob=dropout_p,
            )

        # Create new attention layer using correct config
        new_self = RoPEBertSelfAttention(cfg)

        # Load any matching weights (strict=False needed)
        new_self.load_state_dict(old_self.state_dict(), strict=False)

        # Replace attention module
        layer.attention.self = new_self

    return model

# ------ #End of ChatGPT code # -------# 

path = os.getcwd() + "/models/contextual_prompts_transformer.json"
#taken from llama CFT files
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
            if line.strip():
                prompts.append(json.loads(line.rstrip("\n")))

    general_prompts = [p for p in prompts if p.get("kind") == "general"]
    negative_by_type = {}
    for p in prompts:
        if p.get("kind") == "negative":
            nt = p.get("neg_type")
            if nt:
                negative_by_type.setdefault(nt, []).append(p)

    return general_prompts, negative_by_type

general_prompts, negative_prompts_by_type = load_contextual_prompts(path)



def pick_context_prompt(ex,mode='nocontext'):
    label = int(ex["label"])
    meta = ex.get("meta", {}) or {}
    neg_type = meta.get("neg_type")

    if mode == "nocontext":
        return ''
    elif mode == "general":
        return rng.choice(general_prompts)["text"]
    elif mode == "both":
        general = rng.choice(general_prompts)["text"]
        all_negative_lists = list(negative_prompts_by_type.values())
        flat_negative = [p for group in all_negative_lists for p in group]
        negative = rng.choice(flat_negative)["text"]

        return general + "\n" + negative
    elif mode == "random":
        if rng.random() < 0.5:
            return rng.choice(general_prompts)["text"]

        if neg_type and neg_type in negative_prompts_by_type:
            return rng.choice(negative_prompts_by_type[neg_type])["text"]

        return rng.choice(general_prompts)["text"]
    else:
        raise ValueError(f"Unknown CONTEXT_MODE: {mode}")
    
def build_prompt(ex: Dict, with_label: bool=False,mode='nocontext') -> str:
    """
    Turn one raw example into a plain text prompt.

    If with_label=True, append the gold 0/1 label (for supervised finetuning).
    If with_label=False, leave the answer blank (for inference).
    """
    premise   = (ex.get("premise", "") or "").strip()
    hypothesis= (ex.get("hypothesis", "") or "").strip()
    label_int = int(ex["label"])

    if mode == "nocontext":
        user_block = (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
    )
    else: 
        ctx_text = pick_context_prompt(ex,mode)
        context_block = ""
        if mode != "nocontext":
            context_block = f"Context: {ctx_text}\n"

        user_block = (
            f"{context_block}"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            "Question: Is the hypothesis true under the premise? "
            "Concisely explain and answer 'Yes' or 'No'. "
            f"{RESPONSE_TAG}"
        )

    return user_block




rng = random.Random(0)
RESPONSE_TAG = "### Answer:\n"
EXPERIMENTS = ["both","nocontext", "general","random"]

dataset = load_dataset(
    "json",
    data_files={
        "train": os.getcwd()+"/data/bas_smoke/train.jsonl",
        "val": os.getcwd()+"/data/bas_smoke/val.jsonl",
        "test": os.getcwd()+"/data/bas_smoke/test.jsonl",
    },
)


special_tokens = [
    "<s>", "</s>", "<pad>", "<unk>", "<mask>",
    "->", "<-", "[", "]", "{", "}", "'", ",",
    "backdoor", "adjustment set", "valid", "minimal"
]
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
hf_tokenizer.add_tokens(special_tokens)
print("vocab size:", len(hf_tokenizer))

BASE_INIT_DIR = "checkpoints/_base_init"
os.makedirs(BASE_INIT_DIR, exist_ok=True)

cls_config = BertConfig(
    hidden_size=512,
    num_attention_heads=8,
    intermediate_size=2048,
    num_hidden_layers=8,
    vocab_size=len(hf_tokenizer),
    num_labels=2
)

base_model = BertForSequenceClassification(cls_config)
base_model = patch_bert_with_rope(base_model)

base_model.save_pretrained(BASE_INIT_DIR)
hf_tokenizer.save_pretrained(BASE_INIT_DIR)


# -----------------------------
# Preprocessing for BERT
# -----------------------------
def preprocess(ex,mode):
    text = build_prompt(ex, with_label=False,mode=mode)
    enc = hf_tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )
    enc["labels"] = ex["label"]
    return enc


def preprocess_length(ex,mode):
    text = build_prompt(ex, with_label=False, mode = mode)
    tokens = hf_tokenizer.tokenize(text)  # no truncation or padding
    return {"length": len(tokens), "tokens": tokens}

def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        print("Counts:", np.bincount(preds))
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "support": len(labels)}

for mode in EXPERIMENTS:
    tokenized = dataset.map(lambda ex: preprocess(ex, mode=mode))
    lengths = dataset.map(lambda ex: preprocess_length(ex, mode=mode))

    # log lengths
    all_lengths = [ex["length"] for ex in lengths["train"]]
    print("Average length:", sum(all_lengths)/len(all_lengths))
    print("Max length:", max(all_lengths))
    example_tokens = lengths["train"][0]["tokens"]
    print("Example token string:", hf_tokenizer.convert_tokens_to_string(example_tokens))

    # -----------------------------
    # LOAD THE SAME BASE INIT FOR EACH EXPERIMENT
    # -----------------------------
    model = BertForSequenceClassification.from_pretrained(BASE_INIT_DIR)
    model = patch_bert_with_rope(model)

    # tokenizer must match base
    tokenizer = tokenizer = hf_tokenizer 
    model.resize_token_embeddings(len(tokenizer))

    # -----------------------------
    # TRAIN (overwrite each time)
    # -----------------------------
    sdir = f"checkpoints/cft_classifier_{mode}"
    os.makedirs(sdir, exist_ok=True)

    print(f"\nTRAINING {mode}...\n")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=sdir,
            eval_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=32,
            num_train_epochs=4,
            load_best_model_at_end=True,
            save_strategy="epoch",
        ),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(sdir)
    tokenizer.save_pretrained(sdir)

    preds_output = trainer.predict(tokenized["test"])
    test_metrics = compute_metrics(preds_output)

    print(f"\nTEST METRICS ({mode})")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall:    {test_metrics['recall']:.4f}")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")


    
import os
import torch
import math
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import (
    PreTrainedTokenizerFast,
    AutoModelForSequenceClassification,
    BertConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    BertForMaskedLM,
    AutoTokenizer,
    BertForSequenceClassification
    
)
from tokenizers import ByteLevelBPETokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


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

# LOAD DATASET
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/bas_smoke/train.jsonl",
        "val": "data/bas_smoke/val.jsonl",
        "test": "data/bas_smoke/test.jsonl",
    },
)

# TRAIN TOKENIZER
corpus = []
for line in dataset["train"]:
    corpus.append(line["premise"])
    corpus.append(line["hypothesis"])
for line in dataset["val"]:
    corpus.append(line["premise"])
    corpus.append(line["hypothesis"])


# test normal autotokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(tokenizer.tokenize("['4'] is a valid minimal backdoor adjustment set for 1 -> 4"))

# use custom tokenizer with 20000 vocab size
tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    corpus,
    vocab_size=20000,
    min_frequency=2,
    special_tokens=[
    "<s>", "<pad>", "</s>", "<unk>", "<mask>",
    "[", "]", ",", "->", "<-", "{", "}", "'", 
    ],
)
tokenizer.save_model("tokenizer/")
tokenizer.save("tokenizer/tokenizer.json")

# load as HF tokenzizer
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer/tokenizer.json",
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)
print("vocab size: ", len(hf_tokenizer))
print(hf_tokenizer.tokenize("X -> Y, Z1"))

# MLM PRETRAINING
mlm = []
for x in dataset["train"]:
    mlm.append(x["premise"])
    mlm.append(x["hypothesis"])
for x in dataset["val"]:
    mlm.append(x["premise"])
    mlm.append(x["hypothesis"])

mlm_dataset = Dataset.from_dict({"text": mlm})


def preprocess_mlm(batch):
    return hf_tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


tokenized_mlm = mlm_dataset.map(preprocess_mlm, batched=True, remove_columns=["text"])

mlm_data_collator = DataCollatorForLanguageModeling(
    tokenizer=hf_tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# mlm config
mlm_config = BertConfig(
    hidden_size=384,
    num_attention_heads=6,
    intermediate_size=1536,
    num_hidden_layers=6,
    vocab_size=len(hf_tokenizer),
)

# load existing MLM if exists, else train
if os.path.exists("mlm_checkpoint"):
    print("Loading MLM...")
    mlm_model = BertForMaskedLM.from_pretrained("mlm_checkpoint")
    mlm_model.resize_token_embeddings(len(hf_tokenizer))
    try:
        patch_bert_with_rope(mlm_model)
    except Exception as e:
        print("Warning patching mlm model:", e)
else:
    print("Training MLM")
    #mlm_model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    mlm_model = BertForMaskedLM(mlm_config)
    mlm_trainer = Trainer(
        model=mlm_model,
        args=TrainingArguments(
            output_dir="scratch_mlm",
            per_device_train_batch_size=16,
            num_train_epochs=15,
            logging_steps=500,
            save_strategy="no",
            eval_strategy="no"
        ),
        train_dataset=tokenized_mlm,
        data_collator=mlm_data_collator,
    )
    print("\n=STARTING MLM PRETRAINING \n")
    mlm_trainer.train()
    mlm_model.save_pretrained("mlm_checkpoint")
    hf_tokenizer.save_pretrained("mlm_checkpoint")


# CLASSIFICATION DATASET
def preprocess(line):
    enc = hf_tokenizer(
        line["premise"], line["hypothesis"],truncation=True, padding="max_length", max_length=256)
    enc["labels"] = line["label"]
    return enc


tokenized = dataset.map(preprocess)

cls_config = mlm_model.config
cls_config.num_labels = 2

model = BertForSequenceClassification(cls_config)

# Load MLM weights into BERT backbone
model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)

# Patch RoPE after weight loading
model = patch_bert_with_rope(model)



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    print("Counts:", np.bincount(preds))
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    support = len(labels)
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }

sdir = '/Users/darpalpatel/Projects/BackdoorBench/final_classifier'
if os.path.exists(sdir):
    hf_tokenizer = AutoTokenizer.from_pretrained(sdir)
    model = BertForSequenceClassification.from_pretrained(sdir)    
    model.resize_token_embeddings(len(hf_tokenizer))
    model = patch_bert_with_rope(model)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="./scratch_eval", per_device_eval_batch_size=8),
        data_collator=DataCollatorWithPadding(hf_tokenizer),
        compute_metrics=compute_metrics,
    )

    preds_output = trainer.predict(tokenized["test"])
    test_metrics = compute_metrics(preds_output)

else:
    trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./scratch",
        eval_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        load_best_model_at_end=True,
        save_strategy="epoch",
        logging_steps=50,
    ),
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    data_collator=DataCollatorWithPadding(hf_tokenizer),
    compute_metrics=compute_metrics,
)

    print("\nSTARTING CLASSIFICATION\n")
    trainer.train()
    trainer.save_model("final_classifier")
    hf_tokenizer.save_pretrained("final_classifier")


    # TEST SET EVAALUATION
    preds_output = trainer.predict(tokenized["test"])
    test_metrics = compute_metrics(preds_output)

print("TEST METRICS")
print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"Precision: {test_metrics['precision']:.4f}")
print(f"Recall:    {test_metrics['recall']:.4f}")
print(f"F1 Score:  {test_metrics['f1']:.4f}")
print(f"Support:   {test_metrics['support']}")


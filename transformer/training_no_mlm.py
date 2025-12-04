import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# ----------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/bas_smoke/train.jsonl",
        "val":   "data/bas_smoke/val.jsonl",
        "test":  "data/bas_smoke/test.jsonl",
    },
)

# ----------------------------------------------------
# TOKENIZER (standard BERT, no custom vocab)
# ----------------------------------------------------
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# ----------------------------------------------------
# BASIC PREPROCESSING (no special tokens)
# ----------------------------------------------------
def preprocess(example):
    enc = hf_tokenizer(
        example["premise"],
        example["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    enc["labels"] = example["label"]
    return enc

tokenized = dataset.map(preprocess, batched=False)


# ----------------------------------------------------
# SIMPLE BERT CONFIG
# ----------------------------------------------------
cls_config = BertConfig(
    hidden_size=384,
    num_attention_heads=6,
    intermediate_size=1536,
    num_hidden_layers=6,
    num_labels=2,
)

model = BertForSequenceClassification(cls_config)


# ----------------------------------------------------
# METRICS
# ----------------------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(labels),
    }


# ----------------------------------------------------
# TRAIN
# ----------------------------------------------------
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./scratch",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=50,
        save_strategy="no",
    ),
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["val"],
    data_collator=DataCollatorWithPadding(hf_tokenizer),
    compute_metrics=compute_metrics,
)

print("\nTraining...\n")
trainer.train()


# ----------------------------------------------------
# FINAL TEST EVAL
# ----------------------------------------------------
preds = trainer.predict(tokenized["test"])
metrics = compute_metrics(preds)

print("\nTEST METRICS")
print(metrics)

#!/usr/bin/env python
"""
Shared model + training utilities for Llama 3.1 8B (Unsloth).
"""

from typing import Tuple

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments


DEFAULT_MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"


def create_model(
    max_seq_length: int = 2048,
    base_model_name: str = DEFAULT_MODEL_NAME,
    use_lora: bool = True,
):
    """
    Load the 4-bit base model, optionally wrap with LoRA for finetuning.
    """
    print(f"Loading base model ({base_model_name}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = base_model_name,
        max_seq_length = max_seq_length,
        load_in_4bit   = True,
        dtype          = None,   # Let Unsloth pick bf16/fp16 based on GPU
    )

    if use_lora:
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
            bias="none",
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
        )
        print(model.print_trainable_parameters())
    else:
        print("Using base model WITHOUT LoRA adapters (no finetuning).")

    return model, tokenizer


def finetune(
    model,
    tokenizer,
    train_ds,
    val_ds,
    max_seq_length: int = 2048,
    output_dir: str = "output",
):
    """
    Run SFTTrainer on train/val datasets.
    """
    print("Starting finetuning ...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,          # tweak for longer training
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


def prepare_for_inference(model):
    """
    Put the (possibly LoRA-wrapped) model into Unsloth inference mode.
    """
    FastLanguageModel.for_inference(model)
    return model

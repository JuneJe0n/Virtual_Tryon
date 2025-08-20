"""
Training script with train/validation split
"""
import os
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from datasets import load_dataset
from datasets import Image as HFImage
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from utils import (
    FormatReward,
    AccuracyReward,
    DuplicateShapeGuardReward,
    weighted,
    set_completions_file,
)

import wandb
wandb.init(project="lviton_grpo", name="qwen2.5-vl-run0")

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_JSONL = "/home/jiyoon/data/jsonl/training_data/all_looks.jsonl"  # {"image","solution","prompt"}
OUTPUT_DIR = "/home/jiyoon/data/ckpts/Qwen2.5-VL-3B-Instruct-GRPO"
COMPLETIONS_FILE = "/home/jiyoon/data/jsonl/completions/run_0.jsonl"
VAL_RATIO = 0.05
SEED = 42

# Set completions file for logging
set_completions_file(COMPLETIONS_FILE)

# --- load base model & processor
processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True, padding_side="left")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Enable grad checkpointing & TF32
model.config.use_cache = False
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
torch.backends.cuda.matmul.allow_tf32 = True

# LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    model.get_input_embeddings().requires_grad_(True)


# --- Load & split dataset
raw_ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")
raw_ds = raw_ds.cast_column("image", HFImage()) # lazy decoding

splits = raw_ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
train_dataset, eval_dataset = splits["train"], splits["test"]


# --- Training config 
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    remove_unused_columns=False,   # keep 'solution' for reward fns
    num_train_epochs=1,
    bf16=True,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    max_prompt_length=2048,       
    max_completion_length=512,
    num_generations=4,

    # logging/saving
    save_strategy="steps",
    save_steps=50,
    report_to=["wandb"],
    run_name="Qwen2.5-VL-GRPO",
    logging_steps=10,
    log_completions=True,              
    num_completions_to_print=4,

    eval_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=False,

    temperature=0.7,
    top_p=0.95,
    generation_kwargs={"do_sample": True},
)


# --- Reward functions 
fmt_reward = FormatReward(w_tags=0.3, w_json=0.3, w_schema=0.4)
acc_reward = AccuracyReward(reference_key="solution")
dup_reward = DuplicateShapeGuardReward(dup_penalty=0.5)

L_F, L_A, L_D = 0.5, 1.0, 0.2
reward_fns = [
    weighted(fmt_reward, L_F),
    weighted(acc_reward, L_A),
    weighted(dup_reward, L_D),
]

# --- Trainer 
trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=reward_fns,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,   
)


# --- Train & Save
trainer.train()
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
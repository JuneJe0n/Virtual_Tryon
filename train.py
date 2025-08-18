"""
Training script 
"""
import os, json, re, math
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from utils import (
    format_reward,
    make_accuracy_reward,
    length_guard_reward,
    duplicate_shape_guard_reward
)


# =========================
# Paths / IDs  
# =========================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_JSONL = "/home/jiyoon/data/jsonl/all_looks.jsonl"  # each line: {"image","solution","prompt"}
OUTPUT_DIR = "/home/jiyoon/LViton_GRPO/model/Qwen2.5-VL-3B-Instruct-GRPO"


# =========================
# Processor & Model
# =========================
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


# =========================
# Dataset
# =========================
# JSONL must have columns: image (path), solution (list[dict]), prompt (string)
train_dataset = load_dataset("json", data_files=TRAIN_JSONL, split="train")


# =========================
# Data collator (image + text -> tensors)
# =========================
def make_vl_collator(processor):
    def collate(batch):
        prompts = [ex["prompt"] for ex in batch]
        images = [Image.open(ex["image"]).convert("RGB") for ex in batch]
        proc = processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        # Keep ground-truth solutions for rewards
        proc["solution"] = [ex["solution"] for ex in batch]
        return proc
    return collate

data_collator = make_vl_collator(processor)



# =========================
# Training config
# =========================

# Reward functions
accuracy_reward = make_accuracy_reward(reference_key="solution")

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-5,
    remove_unused_columns=False,     # keep 'solution' for rewards
    num_train_epochs=1,
    bf16=True,

    per_device_train_batch_size=1,   # VLMs are heavy; start small
    gradient_accumulation_steps=8,
    max_prompt_length=2048,
    max_completion_length=512,
    num_generations=4,               # GRPO: number of samples per prompt

    logging_steps=10,
    save_strategy="steps",
    save_steps=50,
    report_to=["tensorboard"],

    # (optional) generation settings
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
)

# (optional) simple reweighing of auxiliary rewards
def weighted(rf, w):
    def f(completions, **kw):
        return [w*x for x in rf(completions, **kw)]
    return f

reward_fns = [
    weighted(format_reward, 0.5),
    weighted(accuracy_reward, 1.0),
    weighted(length_guard_reward, 0.2),
    weighted(duplicate_shape_guard_reward, 0.2),
]


# =========================
# Trainer
# =========================
trainer = GRPOTrainer(
    model=model,
    processing_class=processor,     # tokenizer+image processor
    reward_funcs=reward_fns,
    args=training_args,
    train_dataset=train_dataset,    # provides image path, solution, prompt
    data_collator=data_collator,    # turns (prompt,image) -> tensors + passes 'solution'
)

# =========================
# Train / Save
# =========================
trainer.train()
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)

# (optional) push to hub
# trainer.push_to_hub()

"""
Training script 
Base model : Qwen/Qwen2.5-VL-3B-Instruct
"""

import os
import re
import wandb
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from datasets import load_dataset
from datasets import Image as HFImage
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, TrainerCallback
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer

from utils import (
    FormatReward,
    AccuracyReward,
    weighted,
    set_completions_dir,
)


# --- Config
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_JSONL = "/home/jiyoon/data/jsonl/training_data/lips/all_looks.jsonl"  # {"image","solution","prompt"}
OUTPUT_DIR = "/home/jiyoon/data/ckpts/Qwen2.5-VL-3B-Instruct-GRPO-v7-run2"
VAL_RATIO = 0.05
SEED = 42

WANDB_NAME = "qwen2.5-vl-v7-run2"

COMPLETIONS_BASE = "/home/jiyoon/data/jsonl/completions"
VERSION = "v7"
RUN = "run2"
ROTATE = 500


# --- Helper Functions
def ensure_gradients_for_vl_model(model):
    """Ensure all necessary components require gradients for VL model training"""
    
    # Enable gradients for input embeddings
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        print("[INFO] Used enable_input_require_grads()")
    else:
        # Fallback for text embeddings
        if hasattr(model, "get_input_embeddings"):
            model.get_input_embeddings().requires_grad_(True)
            print("[INFO] Enabled gradients for input embeddings")
        
        # For vision embeddings in VL models
        if hasattr(model, "visual"):
            for param in model.visual.parameters():
                param.requires_grad_(True)
            print("[INFO] Enabled gradients for visual module")
        
        # Alternative paths for vision components
        vision_modules_found = 0
        for name, module in model.named_modules():
            if any(vision_term in name.lower() for vision_term in ['vision', 'visual', 'image', 'patch']):
                for param in module.parameters():
                    param.requires_grad_(True)
                vision_modules_found += 1
        
        if vision_modules_found > 0:
            print(f"[INFO] Enabled gradients for {vision_modules_found} vision-related modules")


def setup_gradient_checkpointing(model):
    """Setup gradient checkpointing with proper checks"""
    
    # Check if any parameters require gradients
    has_grad_params = any(p.requires_grad for p in model.parameters())
    
    if has_grad_params and hasattr(model, "gradient_checkpointing_enable"):
        print("[INFO] Enabling gradient checkpointing - found parameters requiring gradients")
        model.gradient_checkpointing_enable()
        
        # Set gradient checkpointing kwargs to suppress warnings
        if hasattr(model.config, 'gradient_checkpointing_kwargs'):
            model.config.gradient_checkpointing_kwargs = {"use_reentrant": False}
    else:
        print("[WARNING] Skipping gradient checkpointing - no parameters require gradients")


# --- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO training with optional checkpoint resume")
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint directory to resume from. If omitted, starts fresh."
    )
    return p.parse_args()


# --- Train 🚀
def main():
    args = parse_args()

    # --- wandb
    wandb.init(project="lviton_grpo", name=WANDB_NAME, resume="allow", dir='/home/jiyoon/data/wandb')
    set_completions_dir(
        base_dir=COMPLETIONS_BASE,
        version=VERSION,
        run=RUN,
        rotate_every=ROTATE
        )

    # --- load base model & processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True, padding_side="left")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Enable grad checkpointing & TF32
    model.config.use_cache = False
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

    # ENHANCED GRADIENT SETUP - This is the new part!
    ensure_gradients_for_vl_model(model)
    setup_gradient_checkpointing(model)

    # --- Load & split dataset
    raw_ds = load_dataset("json", data_files=TRAIN_JSONL, split="train")
    raw_ds = raw_ds.cast_column("image", HFImage())  # lazy decoding

    splits = raw_ds.train_test_split(test_size=VAL_RATIO, seed=SEED)
    train_dataset, eval_dataset = splits["train"], splits["test"]

    # --- Training config
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,      
        remove_unused_columns=False,   
        num_train_epochs=1,
        bf16=True,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,

        max_prompt_length=2048,
        max_completion_length=512,
        num_generations=8,

        # logging/saving
        save_strategy="steps",
        save_steps=100,
        report_to=["wandb"],
        run_name="Qwen2.5-VL-GRPO",
        logging_steps=10,
        log_completions=True,
        num_completions_to_print=4,

        eval_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=False,

        temperature=0.9,
        top_p=0.9,
        generation_kwargs={"do_sample": True},

        lr_scheduler_type="constant_with_warmup",
        warmup_ratio=0.05,               
    )

    class GradNormCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            model = kw["model"]
            sq = 0.0
            for _, p in model.named_parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    sq += float(g.norm(2).item() ** 2)
            grad_norm = sq ** 0.5
            try:
                import wandb; wandb.log({"debug/grad_norm": grad_norm})
            except Exception:
                pass


    # --- Reward functions
    fmt_reward = FormatReward(w_tags=0.3, w_json=0.3, w_schema=0.4)
    acc_reward = AccuracyReward(reference_key="solution")

    L_F, L_A = 0.3, 1.0
    reward_fns = [
        weighted(fmt_reward, L_F),
        weighted(acc_reward, L_A),
    ]
    
    # --- Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[GradNormCallback()]
    )

    # --- Train (optionally resume) & Save
    if args.resume_from_checkpoint is not None:
        ckpt_path = Path(args.resume_from_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume_from_checkpoint not found: {ckpt_path}")
        print(f"[INFO] Resuming from checkpoint: {ckpt_path}")
        trainer.train(resume_from_checkpoint=str(ckpt_path))
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
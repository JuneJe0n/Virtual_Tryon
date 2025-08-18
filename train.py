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
# Rewards
# =========================
TAG_RE = re.compile(r"^<answer>\s*(\[.*\])\s*</answer>\s*$", re.DOTALL)

def _extract_json_block(text: str) -> Optional[str]:
    m = TAG_RE.match(text.strip())
    return m.group(1) if m else None

def _is_int(v) -> bool:
    try:
        return float(v).is_integer()
    except Exception:
        return isinstance(v, int)

def _validate_item(d: Dict[str, Any]) -> bool:
    """Schema check with gamma-always-present rule (gamma integer, 0 if shape doesn’t define it)."""
    if not isinstance(d, dict): return False
    if "shape" not in d or not isinstance(d["shape"], str): return False
    c = d.get("color", {})
    if not isinstance(c, dict): return False
    for k in ("r","g","b"):
        if k not in c or not _is_int(c[k]) or not (0 <= int(c[k]) <= 255): return False
    if "alpha" not in d or not _is_int(d["alpha"]) or not (0 <= int(d["alpha"]) <= 255): return False
    if "sigma" not in d or not _is_int(d["sigma"]): return False
    if "gamma" not in d or not _is_int(d["gamma"]): return False
    return True

def _safe_load_json(arr_str: str) -> Optional[List[Dict[str, Any]]]:
    try:
        obj = json.loads(arr_str)
        if isinstance(obj, list) and len(obj) >= 1 and all(isinstance(x, dict) for x in obj):
            return obj
        return None
    except Exception:
        return None

def _color_score(a: Dict[str,int], b: Dict[str,int]) -> float:
    d = (abs(int(a["r"])-int(b["r"])) + abs(int(a["g"])-int(b["g"])) + abs(int(a["b"])-int(b["b"]))) / (3*255)
    return max(0.0, 1.0 - d)

def _param_score_int(a: int, b: int, span: int) -> float:
    err = abs(int(a) - int(b)) / max(1, span)
    return max(0.0, 1.0 - err)

def _match_and_score(pred_list: List[Dict[str,Any]], ref_list: List[Dict[str,Any]]) -> float:
    used = [False]*len(ref_list)
    per_item_scores = []
    ALPHA_SPAN, SIGMA_SPAN, GAMMA_SPAN = 255, 255, 100

    for p in pred_list:
        best, best_idx = 0.0, -1
        for j, r in enumerate(ref_list):
            if used[j] or p["shape"] != r["shape"]:
                continue
            c_score = _color_score(p["color"], r["color"])
            a_score = _param_score_int(p["alpha"], r["alpha"], ALPHA_SPAN)
            s_score = _param_score_int(p["sigma"], r["sigma"], SIGMA_SPAN)
            g_score = _param_score_int(p["gamma"], r.get("gamma", 0), GAMMA_SPAN)
            score = (1/3)*c_score + (1/6)*a_score + (1/6)*s_score + (1/3)*g_score
            if score > best:
                best, best_idx = score, j
        if best_idx >= 0:
            used[best_idx] = True
            per_item_scores.append(best)
        else:
            per_item_scores.append(0.0)

    missing = (len(ref_list) - sum(used))
    miss_penalty = 0.15 * max(0, missing)

    base = sum(per_item_scores) / max(1, len(pred_list))
    final = max(0.0, base - miss_penalty)
    return final

def format_reward(completions: List[str], **kwargs) -> List[float]:
    """0..1 score for tags + JSON parse + schema (gamma must be present)."""
    rewards = []
    for content in completions:
        tags_ok = 1.0 if TAG_RE.match(content.strip()) else 0.0
        arr = _extract_json_block(content) if tags_ok else None
        json_ok = 1.0 if (arr and _safe_load_json(arr) is not None) else 0.0
        schema_ok = 0.0
        if json_ok:
            items = _safe_load_json(arr)
            if items and all(_validate_item(it) for it in items):
                schema_ok = 1.0
        rewards.append(0.3*tags_ok + 0.3*json_ok + 0.4*schema_ok)
    return rewards

def make_accuracy_reward(reference_key: str = "solution"):
    def accuracy_reward(completions: List[str], **kwargs) -> List[float]:
        refs = kwargs.get(reference_key, None)
        scores: List[float] = []
        for i, content in enumerate(completions):
            ref_list = refs[i] if refs is not None else []
            arr = _extract_json_block(content)
            if not arr:
                scores.append(0.0); continue
            pred_list = _safe_load_json(arr)
            if not pred_list:
                scores.append(0.0); continue
            if not all(_validate_item(x) for x in pred_list):
                scores.append(0.0); continue
            scores.append(_match_and_score(pred_list, ref_list))
        return scores
    return accuracy_reward

def length_guard_reward(completions: List[str], **kwargs) -> List[float]:
    outs = []
    for c in completions:
        n = len(c)
        if n < 10: outs.append(0.0)
        elif n > 8192: outs.append(0.2)
        else: outs.append(1.0)
    return outs

def duplicate_shape_guard_reward(completions: List[str], **kwargs) -> List[float]:
    outs = []
    for c in completions:
        arr = _extract_json_block(c)
        if not arr:
            outs.append(0.0); continue
        pred = _safe_load_json(arr)
        if not pred:
            outs.append(0.0); continue
        seen = set(); ok = True
        for o in pred:
            key = (o.get("shape"),
                   o.get("color",{}).get("r"), o.get("color",{}).get("g"), o.get("color",{}).get("b"),
                   o.get("alpha"), o.get("sigma"), o.get("gamma"))
            if key in seen:
                ok = False; break
            seen.add(key)
        outs.append(1.0 if ok else 0.5)
    return outs

accuracy_reward = make_accuracy_reward(reference_key="solution")


# =========================
# Training config
# =========================
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

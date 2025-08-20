import re, json
import os
from typing import List, Dict, Any, Optional
from .prompts import ALLOWED_SHAPES, SHAPE_FAMILY
import numpy as np
from scipy.optimize import linear_sum_assignment


# --- WandB logging
def _wandb_log(data: Dict[str, Any]):
    try:
        import wandb
        if getattr(wandb, "run", None) is not None:
            wandb.log(data)
    except Exception:
        pass

# Global variable to store completions file path
_COMPLETIONS_FILE = None

def set_completions_file(filepath: str):
    """Set the global completions file path"""
    global _COMPLETIONS_FILE
    _COMPLETIONS_FILE = filepath


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


# --- Completion logging
def _log_completions(completions: List[str], completions_file: str = None, **kwargs):
    """Log completions to file"""
    global _COMPLETIONS_FILE
    if not completions_file:
        completions_file = _COMPLETIONS_FILE
    if not completions_file:
        return
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(completions_file), exist_ok=True)
        
        # Save as readable text format
        with open(completions_file, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"BATCH: {len(completions)} completions\n")
            f.write("=" * 80 + "\n")
            
            for i, completion in enumerate(completions):
                f.write(f"\n--- COMPLETION {i+1} ---\n")
                f.write(completion)
                f.write("\n")
            
            f.write("\n")
    except Exception as e:
        print(f"Warning: Could not save completions to {completions_file}: {e}")


# --- Helper functions
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
    """ Schema validator """
    if not isinstance(d, dict):
        return False

    if "shape" not in d or not isinstance(d["shape"], str):
        return False
    if d["shape"] not in ALLOWED_SHAPES:
        return False

    c = d.get("color", {})
    if not isinstance(c, dict):
        return False
    for k in ("r", "g", "b"):
        if k not in c or not _is_int(c[k]) or not (0 <= int(c[k]) <= 255):
            return False

    if "alpha" not in d or not _is_int(d["alpha"]) or not (0 <= int(d["alpha"]) <= 255):
        return False

    if "sigma" not in d or not _is_int(d["sigma"]):
        return False

    if "gamma" not in d or not _is_int(d["gamma"]):
        return False

    return True

def _safe_load_json(arr_str: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse JSON str to check :
    - Top-level object is list
    - Each element is dict
    """
    try:
        obj = json.loads(arr_str)
        if isinstance(obj, list) and len(obj) >= 1 and all(isinstance(x, dict) for x in obj):
            return obj
        return None
    except Exception:
        return None

def _color_score(a: Dict[str, int], b: Dict[str, int], thresh: float = 0.3) -> float:
    """
    Normalized L1 distance in RGB space (0-1).
    1.0 if colors are identical.
    Returns 0.0 if distance > thresh.
    """
    d = (abs(int(a["r"]) - int(b["r"])) +
         abs(int(a["g"]) - int(b["g"])) +
         abs(int(a["b"]) - int(b["b"]))) / (3 * 255)
    if d > thresh:
        return 0.0
    return 1.0 - d

def _param_score_int(a: int, b: int, span: int, thresh: float = 0.3) -> float:
    """
    Compare ints like alpha, sigma, gamma, relative to their allowed span.
    """
    err = abs(int(a) - int(b)) / max(1, span)  # normalized [0,1]
    if err > thresh:
        return 0.0
    return 1.0 - err

def _shape_factor(pred_shape: str, ref_shape: str) -> float:
    """
    1.0 exact shape, 0.6 same family, 0.2 cross-family.
    """
    if pred_shape == ref_shape:
        return 1.0
    pf, rf = SHAPE_FAMILY.get(pred_shape, ""), SHAPE_FAMILY.get(ref_shape, "")
    if pf and rf and pf == rf:
        return 0.6
    return 0.2

def _match_and_score_hungarian(pred_list: List[Dict[str,Any]], ref_list: List[Dict[str,Any]]) -> float:
    """
    Optimal global assignment with soft shape factors and dummy padding.
    
    - Similarity(p,r) = shape_factor * param_similarity
    - Cost = 1 - Similarity
    - Score = mean(matched_similarities) normalized by max(n_pred, n_ref) minus 0.15 per unmatched ref.
    """
    n, m = len(pred_list), len(ref_list)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0  # clamp to 0; no matches possible

    ALPHA_SPAN, SIGMA_SPAN, GAMMA_SPAN = 255, 255, 100

    # Similarity matrix S (n x m)
    S = np.zeros((n, m), dtype=float)
    for i, p in enumerate(pred_list):
        for j, r in enumerate(ref_list):
            c_score = _color_score(p["color"], r["color"])
            a_score = _param_score_int(p["alpha"], r["alpha"], ALPHA_SPAN)
            s_score = _param_score_int(p["sigma"], r["sigma"], SIGMA_SPAN)
            g_score = _param_score_int(p["gamma"], r.get("gamma", 0), GAMMA_SPAN)
            param_sim = 0.25 * (c_score + a_score + s_score + g_score)
            sf = _shape_factor(p["shape"], r["shape"])
            S[i, j] = sf * param_sim

    # Convert to score to cost 
    C = 1.0 - S
    dim = max(n, m)

    # Pad to square w dummies
    C_pad = np.ones((dim, dim), dtype=float) 
    C_pad[:n, :m] = C

    row_ind, col_ind = linear_sum_assignment(C_pad)

    sims = []
    matched_real_refs = set()
    for i, j in zip(row_ind, col_ind):
        if i < n and j < m:
            sims.append(1.0 - C[i, j]) 
            matched_real_refs.add(j)
        else:
            sims.append(0.0)      # dummy match -> sim=0

    unmatched_refs = m - len(matched_real_refs) 
    base = sum(sims) / float(dim)
    miss_penalty = 0.15 * max(0, unmatched_refs)
    final = max(0.0, base - miss_penalty)
    return final

def weighted(reward_callable, weight: float):
    """
    Helper function for weighted sum.
    Also logs weighted reward means to wandb as 'reward_weighted/{name}' when available.
    """
    name = getattr(reward_callable, "name", reward_callable.__class__.__name__.lower())

    def f(completions: List[str], **kw) -> List[float]:
        raw = reward_callable(completions, **kw)  
        out = [weight * x for x in raw]
        # log weighted mean
        _wandb_log({f"reward_weighted/{name}": _mean(out)})
        return out
    return f


# --- Reward classes
class FormatReward:
    """
    Format reward (0-1):
      - tags present (0.3)
      - JSON parses (0.3)
      - schema valid (0.4)
    """
    def __init__(self, w_tags=0.3, w_json=0.3, w_schema=0.4):
        self.w_tags = w_tags
        self.w_json = w_json
        self.w_schema = w_schema
        self.name = "format"

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        rewards = []
        tags_flags, json_flags, schema_flags = [], [], []
        
        # Log completions to file
        _log_completions(completions, **kwargs)
        
        for content in completions:
            tags_ok = 1.0 if TAG_RE.match(content.strip()) else 0.0
            arr = _extract_json_block(content) if tags_ok else None
            json_ok = 1.0 if (arr and _safe_load_json(arr) is not None) else 0.0
            schema_ok = 0.0
            if json_ok:
                items = _safe_load_json(arr)
                if items and all(_validate_item(it) for it in items):
                    schema_ok = 1.0

            score = self.w_tags*tags_ok + self.w_json*json_ok + self.w_schema*schema_ok
            rewards.append(score)
            tags_flags.append(tags_ok); json_flags.append(json_ok); schema_flags.append(schema_ok)

        # wandb logging (means + hist)
        _wandb_log({
            "reward/format": _mean(rewards),
            "reward/format_tags": _mean(tags_flags),
            "reward/format_json": _mean(json_flags),
            "reward/format_schema": _mean(schema_flags),
        })
        return rewards


class AccuracyReward:
    """
    Accuracy reward (0-1) using global Hungarian assignment
    """
    def __init__(self, reference_key: str = "solution"):
        self.reference_key = reference_key
        self.name = "accuracy"

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        refs = kwargs.get(self.reference_key, None)
        scores: List[float] = []
        for i, content in enumerate(completions):
            ref_list = refs[i] if refs is not None else []
            arr = _extract_json_block(content)
            if not arr:
                scores.append(0.0); continue
            pred_list = _safe_load_json(arr)
            if not pred_list or not all(_validate_item(x) for x in pred_list):
                scores.append(0.0); continue

            scores.append(_match_and_score_hungarian(pred_list, ref_list))

        _wandb_log({
            "reward/accuracy": _mean(scores),
        })
        return scores


class DuplicateShapeGuardReward:
    """
    Penalize duplicate identical items (shape + color + alpha + sigma + gamma)
      - no duplicates -> 1.0
      - duplicates    -> dup_penalty (default 0.5)
    """
    def __init__(self, dup_penalty: float = 0.5):
        self.dup_penalty = dup_penalty
        self.name = "duplicate"

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        outs = []
        dup_flags = []  # 1 if duplicate detected, else 0
        for c in completions:
            arr = _extract_json_block(c)
            if not arr:
                outs.append(0.0); dup_flags.append(0.0); continue
            pred = _safe_load_json(arr)
            if not pred:
                outs.append(0.0); dup_flags.append(0.0); continue
            seen = set(); ok = True
            for o in pred:
                key = (
                    o.get("shape"),
                    o.get("color",{}).get("r"), o.get("color",{}).get("g"), o.get("color",{}).get("b"),
                    o.get("alpha"), o.get("sigma"), o.get("gamma"),
                )
                if key in seen:
                    ok = False; break
                seen.add(key)
            outs.append(1.0 if ok else self.dup_penalty)
            dup_flags.append(0.0 if ok else 1.0)

        _wandb_log({
            "reward/duplicate": _mean(outs),
            "reward/duplicate_rate": _mean(dup_flags),  
        })
        return outs

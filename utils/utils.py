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

def _mean(xs: List[float]) -> float:
    return float(np.mean(xs)) if len(xs) else 0.0


# --- Completion logging
_COMPLETIONS_DIR: Optional[str] = None   
_ROTATE_EVERY: int = 50
_CALLS: int = 0

def set_completions_dir(base_dir: str, version: str = "v1", run: str = "run0", rotate_every: int = 50):
    """
    Set completions directory for a specific version/run.
    Files will be saved as:
      <base_dir>/<version>/<run>/part000.jsonl
    """
    global _COMPLETIONS_DIR, _ROTATE_EVERY, _CALLS
    _COMPLETIONS_DIR = os.path.join(base_dir, version, run)
    _ROTATE_EVERY = max(1, int(rotate_every))
    _CALLS = 0
    os.makedirs(_COMPLETIONS_DIR, exist_ok=True)

def _step_from(kwargs: Dict[str, Any]) -> Optional[int]:
    for k in ("global_step", "step", "iteration"):
        v = kwargs.get(k)
        if v is None: 
            continue
        try:
            vi = int(v)
            if vi >= 0:
                return vi
        except Exception:
            pass
    try:
        import wandb
        if getattr(wandb, "run", None) is not None and hasattr(wandb.run, "step"):
            s = wandb.run.step
            if s is not None:
                return int(s)
    except Exception:
        pass
    return None

def _rotated_path(step: Optional[int]) -> Optional[str]:
    if not _COMPLETIONS_DIR:
        return None
    part = (step if step is not None else _CALLS) // _ROTATE_EVERY
    return os.path.join(_COMPLETIONS_DIR, f"part{part:03d}.jsonl")

def _log_completions(completions: List[str], completions_file: str = None, **kwargs):
    global _CALLS
    if not completions:
        return

    target = completions_file or _rotated_path(_step_from(kwargs))
    if not target:
        return

    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"BATCH: {len(completions)} completions\n")
            s = _step_from(kwargs)
            if s is not None:
                f.write(f"STEP: {s}\n")
            f.write("=" * 80 + "\n")
            for i, c in enumerate(completions, 1):
                f.write(f"\n--- COMPLETION {i} ---\n")
                f.write(c)
                f.write("\n")
            f.write("\n")
    except Exception as e:
        print(f"Warning: Could not save completions to {target}: {e}")
    finally:
        _CALLS += 1

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

def _color_score(a: Dict[str, int], b: Dict[str, int]) -> float:
    """
    Normalized L1 distance in RGB space (0-1).
    1.0 if colors are identical.
    Returns 0.0 if distance > thresh.
    """
    d = (abs(int(a["r"]) - int(b["r"])) +
         abs(int(a["g"]) - int(b["g"])) +
         abs(int(a["b"]) - int(b["b"]))) / (3 * 255)
    return 1.0 - d

def _param_score_int(a: int, b: int, span: int) -> float:
    """
    Compare ints like alpha, sigma, gamma, relative to their allowed span.
    """
    err = abs(int(a) - int(b)) / max(1, span)  # normalized [0,1]
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
    return 0.0

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
            param_sim = 0.7*c_score + 0.15*a_score + 0.1*s_score + 0.05*g_score
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
            "fmt_reward/mean": _mean(rewards),
            "fmt_reward/tags": _mean(tags_flags),
            "fmt_reward/json": _mean(json_flags),
            "fmt_reward/schema": _mean(schema_flags),
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
            "acc_reward/mean": _mean(scores),
        })
        return scores



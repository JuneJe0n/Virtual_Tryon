# rewards.py
import re, json
from typing import List, Dict, Any, Optional
from prompts import ALLOWED_SHAPES  # your allowed shape list

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
    """Schema validator"""
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

def _color_score(a: Dict[str,int], b: Dict[str,int]) -> float:
    """
    Normalized L1 distance in RGB space (0-1)
    1.0 if colors are identical
    """
    d = (abs(int(a["r"])-int(b["r"])) + abs(int(a["g"])-int(b["g"])) + abs(int(a["b"])-int(b["b"]))) / (3*255)
    return max(0.0, 1.0 - d)

def _param_score_int(a: int, b: int, span: int) -> float:
    """
    Compare ints like alpha, sigma, gamma, relative to their allowed span
    """
    err = abs(int(a) - int(b)) / max(1, span)
    return max(0.0, 1.0 - err)

def _match_and_score(pred_list: List[Dict[str,Any]], ref_list: List[Dict[str,Any]]) -> float:
    """
    Greedy shape-aware matching; per-item score weights:
      color 1/3, alpha 1/6, sigma 1/6, gamma 1/3.
    Penalizes missing reference items (0.15 each).
    """
    used = [False]*len(ref_list) # Keeps track of which reference items have alr been matched
    per_item_scores = []
    ALPHA_SPAN, SIGMA_SPAN, GAMMA_SPAN = 255, 255, 100

    for p in pred_list:
        best, best_idx = 0.0, -1
        for j, r in enumerate(ref_list): # Skip if that reference is alr matched
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

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
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
            rewards.append(self.w_tags*tags_ok + self.w_json*json_ok + self.w_schema*schema_ok)
        return rewards


class AccuracyReward:
    """
    Accuracy reward (0-1)
    """
    def __init__(self, reference_key: str = "solution"):
        self.reference_key = reference_key

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        refs = kwargs.get(self.reference_key, None)
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


class LengthGuardReward:
    """
    Length guard:
      - < 10 chars -> 0.0
      - > 8192 chars -> 0.2 (soft penalty)
      - else -> 1.0
    """
    def __init__(self, min_len: int = 10, max_len: int = 8192, long_penalty: float = 0.2):
        self.min_len = min_len
        self.max_len = max_len
        self.long_penalty = long_penalty

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        outs = []
        for c in completions:
            n = len(c)
            if n < self.min_len:
                outs.append(0.0)
            elif n > self.max_len:
                outs.append(self.long_penalty)
            else:
                outs.append(1.0)
        return outs


class DuplicateShapeGuardReward:
    """
    Penalize duplicate identical items (shape + color + alpha + sigma + gamma).
      - no duplicates -> 1.0
      - duplicates    -> 0.5
    """
    def __init__(self, dup_penalty: float = 0.5):
        self.dup_penalty = dup_penalty

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
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
                key = (
                    o.get("shape"),
                    o.get("color",{}).get("r"), o.get("color",{}).get("g"), o.get("color",{}).get("b"),
                    o.get("alpha"), o.get("sigma"), o.get("gamma"),
                )
                if key in seen:
                    ok = False; break
                seen.add(key)
            outs.append(1.0 if ok else self.dup_penalty)
        return outs


# ---------- Optional: simple weighting wrapper ----------
def weighted(reward_callable, weight: float):
    """Return a callable that scales another reward's outputs by `weight`."""
    def f(completions: List[str], **kw) -> List[float]:
        return [weight * x for x in reward_callable(completions, **kw)]
    return f

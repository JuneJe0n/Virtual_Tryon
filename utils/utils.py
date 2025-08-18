import re

TAG_RE = re.compile(r"^<answer>\s*(\[.*\])\s*</answer>\s*$", re.DOTALL)

def _extract_json_block(text: str) -> Optional[str]:
    m = TAG_RE.match(text.strip())
    return m.group(1) if m else None

# For validating fields like alpha, sigma, gamma are int
def _is_int(v) -> bool:
    try:
        return float(v).is_integer()
    except Exception:
        return isinstance(v, int)

# Schema validator
def _validate_item(d: Dict[str, Any]) -> bool:
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
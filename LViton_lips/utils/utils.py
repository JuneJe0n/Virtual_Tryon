import re, json
import os
from typing import List, Dict, Any, Optional
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
_ROTATE_EVERY: int = 500
_CALLS: int = 0

def set_completions_dir(base_dir: str, version: str = "v1", run: str = "run0", rotate_every: int = 500):
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
TAG_RE = re.compile(r"^<answer>\s*(\{.*\})\s*</answer>\s*$", re.DOTALL)

def _extract_json_block(text: str) -> Optional[str]:
    text = text.strip()
    # Remove code block markers if present
    if text.startswith('```') and text.endswith('```'):
        lines = text.split('\n')
        if len(lines) >= 3:
            text = '\n'.join(lines[1:-1])
    elif text.startswith('```json') and text.endswith('```'):
        lines = text.split('\n')
        if len(lines) >= 3:
            text = '\n'.join(lines[1:-1])
    
    m = TAG_RE.match(text.strip())
    return m.group(1) if m else None

def _validate_item(d: Dict[str, Any]) -> float:
    """ Schema validator with binary validation """
    if not isinstance(d, dict):
        return 0.2

    c = d.get("color", "")
    if not isinstance(c, str) or not c.startswith("#") or len(c) != 7:
        return 0.2

    return 1.0

def _safe_load_json(obj_str: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON str to dict with 'color'
    """
    try:
        obj = json.loads(obj_str)
        if isinstance(obj, dict) and "color" in obj:
            return obj
        return None
    except Exception:
        return None

def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return r, g, b
    except ValueError:
        raise ValueError(f"Invalid hex color: {hex_color}")

def _rgb_to_xyz(r: int, g: int, b: int) -> tuple:
    """Convert RGB to XYZ color space"""
    # Normalize RGB to 0-1
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    # Apply gamma correction
    def gamma_correct(c):
        return ((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92
    
    r, g, b = gamma_correct(r), gamma_correct(g), gamma_correct(b)
    
    # Convert to XYZ using sRGB matrix
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    return x, y, z

def _xyz_to_lab(x: float, y: float, z: float) -> tuple:
    """Convert XYZ to LAB color space"""
    # D65 illuminant
    xn, yn, zn = 0.95047, 1.0, 1.08883
    
    x, y, z = x / xn, y / yn, z / zn
    
    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)
    
    fx, fy, fz = f(x), f(y), f(z)
    
    l = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    
    return l, a, b

def _rgb_to_lab(r: int, g: int, b: int) -> tuple:
    """Convert RGB directly to LAB"""
    x, y, z = _rgb_to_xyz(r, g, b)
    return _xyz_to_lab(x, y, z)

def _color_score(a: str, b: str) -> float:
    """
    Perceptual color distance using Delta E in CIELAB space.
    1.0 if colors are identical.
    0.0 if Delta E >= 100 (very different colors).
    """
    # Validate color strings
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.2  # Invalid color
    
    try:
        r_a, g_a, b_a = _hex_to_rgb(a)
        r_b, g_b, b_b = _hex_to_rgb(b)
        
        lab_a = _rgb_to_lab(r_a, g_a, b_a)
        lab_b = _rgb_to_lab(r_b, g_b, b_b)
        
        # Calculate Euclidean distance in LAB space
        delta_e = ((lab_a[0] - lab_b[0]) ** 2 + 
                   (lab_a[1] - lab_b[1]) ** 2 + 
                   (lab_a[2] - lab_b[2]) ** 2) ** 0.5
        
        base_score = max(0.0, 1.0 - (delta_e / 100.0))
        
        return base_score
    except Exception as e:
        print(f"🚨Fallback to original RGB distance if conversion fails: {e}")
        try:
            r_a, g_a, b_a = _hex_to_rgb(a)
            r_b, g_b, b_b = _hex_to_rgb(b)
            d = (abs(r_a - r_b) + abs(g_a - g_b) + abs(b_a - b_b)) / (3 * 255)
            base_score = 1.0 - d
            return base_score
        except:
            return 0.2  # If fallback also fails, return 0.2


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
        rewards, tags_flags, json_flags, schema_flags = [], [], [], []
        
        for i, content in enumerate(completions):
            tags_ok = 1.0 if TAG_RE.match(content.strip()) else 0.2
            obj_str = _extract_json_block(content) if tags_ok else None
            json_ok, schema_ok = 0.2, 0.2
            
            if obj_str:
                parsed = _safe_load_json(obj_str)
                if parsed:
                    json_ok = 1.0
                    schema_ok = _validate_item(parsed)

            score = self.w_tags*tags_ok + self.w_json*json_ok + self.w_schema*schema_ok
            rewards.append(score)
            tags_flags.append(tags_ok); json_flags.append(json_ok); schema_flags.append(schema_ok)

        _wandb_log({
            "fmt_reward/mean": _mean(rewards),
            "fmt_reward/tags": _mean(tags_flags),
            "fmt_reward/json": _mean(json_flags),
            "fmt_reward/schema": _mean(schema_flags),
        })
        return rewards


class AccuracyReward:
    """
    Accuracy reward (0-1) for lip color
    """
    def __init__(self, reference_key: str = "solution"):
        self.reference_key = reference_key
        self.name = "accuracy"

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        refs = kwargs.get(self.reference_key, None)
        scores: List[float] = []
        for i, content in enumerate(completions):
            ref_obj = refs[i] if refs is not None else {}
            if isinstance(ref_obj, list):
                ref_obj = ref_obj[0] if len(ref_obj) > 0 else {}
            ref_color = ref_obj.get("color", "") if isinstance(ref_obj, dict) else ""
            obj_str = _extract_json_block(content)
            if not obj_str:
                scores.append(0.2); continue
            
            pred = _safe_load_json(obj_str)
            if not pred:
                scores.append(0.2); continue

            schema_score = _validate_item(pred)
            if schema_score < 1.0:
                scores.append(0.2); continue

            pred_color = pred.get("color", "")
            if not pred_color or not ref_color:
                scores.append(0.2); continue

            # main color similarity
            score = _color_score(pred_color, ref_color)
            scores.append(score)

        _wandb_log({
            "acc_reward/mean": _mean(scores),
        })
        return scores



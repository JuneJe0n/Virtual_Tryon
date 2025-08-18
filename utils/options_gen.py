"""
Create random makeup looks (no overlapping classes) from product-level JSON.

Input JSON (products): list of {"code","name","options":[{"shape","color",...}, ...]}

Outputs:
- classified_products.json  -> { "LIP": [...], "EYESHADOW": [...], ... }
- random_looks.json         -> [{ "id":"A0", "products":[{"code","name","options":[...]}, ...] }, ...]
"""

from __future__ import annotations
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ─── SETTINGS ─────────────────────────────────────────────────────────────────
PRODUCTS_JSON = Path("/home/jiyoon/LViton/data/json/lviton-options.json")   # products json
OUT_DIR        = Path("/home/jiyoon/LViton/data/json")           # output path
N_LOOKS        = 9913                                    # how many random looks to make
SEED           = 1234       

# Option count per look (picked randomly in this inclusive range, limited by available classes)
MIN_CLASSES    = 3
MAX_CLASSES    = 6
# ─────────────────────────────────────────────────────────────────────────────


def shape_to_class(shape: str) -> str:
    # Examples:
    #  "LIP_FULL_BASIC"                 -> "LIP"
    #  "EYESHADOW_OVEREYE_FULL_BASIC"  -> "EYESHADOW"
    #  "EYELINER_TAIL_DOWN_SHORT_BASIC"-> "EYELINER"
    #  "EYEBROW_BASIC"                  -> "EYEBROW"
    #  "BLUSHER_SIDE_WIDE_BASIC"       -> "BLUSHER"
    #  "HIGHLIGHTER_INNEREYE_BASIC"    -> "HIGHLIGHTER"
    #  "NOSE_SHADING_LONG_BASIC"       -> "NOSE_SHADING"
    #  "FACE_BASIC"                     -> "FACE"
    #  "FACEMESH_TESSELATION"          -> "SKIP" (ignore)
    if shape == "FACE_BASIC":
        return "FACE"
    if shape.startswith("NOSE_SHADING"):
        return "NOSE_SHADING"
    if shape.startswith("EYESHADOW"):
        return "EYESHADOW"
    if shape.startswith("EYELINER"):
        return "EYELINER"
    if shape.startswith("EYEBROW"):
        return "EYEBROW"
    if shape.startswith("BLUSHER"):
        return "BLUSHER"
    if shape.startswith("HIGHLIGHTER"):
        return "HIGHLIGHTER"
    if shape.startswith("LIP"):
        return "LIP"
    if shape.startswith("FACEMESH"):
        return "SKIP"  # not a cosmetic effect
    
    return shape.split("_", 1)[0] if "_" in shape else shape


def clamp_u8(x: Any) -> int:
    v = int(x)
    return 0 if v < 0 else 255 if v > 255 else v


def load_products(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of products at top-level.")
    return data


def classify_by_shape(products: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns:
      {
        "<CLASS>": [
           {
             "code": ..., "name": ...,
             "option": { ...full option dict... }  # for this specific shape/option
           },
           ...
        ],
        ...
      }
    Each product may appear multiple times across different classes if it has multiple option shapes.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for prod in products:
        code = prod.get("code")
        name = prod.get("name")
        for opt in prod.get("options", []):
            shape = str(opt.get("shape", ""))
            cls = shape_to_class(shape)
            if cls == "SKIP" or not shape:
                continue
            # Copy & clamp color/params safely
            color = opt.get("color", {})
            entry = {
                "code": code,
                "name": name,
                "option": {
                    "shape": shape,
                    "color": {
                        "r": clamp_u8(color.get("r", 0)),
                        "g": clamp_u8(color.get("g", 0)),
                        "b": clamp_u8(color.get("b", 0)),
                    },
                    "alpha": clamp_u8(opt.get("alpha", 255)),
                    "sigma": clamp_u8(opt.get("sigma", 0)),
                    "gamma": clamp_u8(opt.get("gamma", 128)),
                    "split": clamp_u8(opt.get("split", 0)),
                }
            }
            buckets[cls].append(entry)
    
    # Sort entries within class for stable output
    for cls in buckets:
        buckets[cls].sort(key=lambda e: (e["code"], e["option"]["shape"]))
    return buckets


def pick_random_classes(buckets: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Pick a random subset of classes that actually have candidates."""
    available = [cls for cls, items in buckets.items() if items]
    if not available:
        return []
    k_min = max(1, min(MIN_CLASSES, len(available)))
    k_max = max(k_min, min(MAX_CLASSES, len(available)))
    k = random.randint(k_min, k_max)
    random.shuffle(available)
    return available[:k]


def build_random_look(
    buckets: Dict[str, List[Dict[str, Any]]],
    look_id: str
) -> Dict[str, Any]:
    """
    Pick <=1 option per class (no overlaps), and aggregate by product code in the output format:
      { "id": look_id,
        "products": [
          { "code": ..., "name": ..., "options": [opt1, opt2, ...] }, ...
        ]
      }
    """
    classes = pick_random_classes(buckets)
    random.shuffle(classes)

    # Aggregate options by product code for the final "products" array
    by_code: Dict[str, Dict[str, Any]] = {}  # code -> {"code","name","options":[...]}
    for cls in classes:
        choices = buckets.get(cls, [])
        if not choices:
            continue
        chosen = random.choice(choices)  # one option in this class
        code = chosen["code"]
        name = chosen["name"]
        opt  = chosen["option"]

        if code not in by_code:
            by_code[code] = {"code": code, "name": name, "options": []}
        by_code[code]["options"].append(opt)

    # Stable order: sort products by code; options keep their chosen order
    products_out = sorted(by_code.values(), key=lambda p: p["code"])
    return {"id": look_id, "products": products_out}


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load product-level JSON
    products = load_products(PRODUCTS_JSON)

    # 2) Classify by shape class
    buckets = classify_by_shape(products)
    (OUT_DIR / "classified_products.json").write_text(
        json.dumps(buckets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved classification → {OUT_DIR/'classified_products.json'}")

    # 3) Build random looks with IDs A0..A{N-1}
    looks: List[Dict[str, Any]] = []
    for i in range(N_LOOKS):
        look_id = f"A{i}"
        look = build_random_look(buckets, look_id)
        # filter out empty looks (e.g., if no classes available)
        if look.get("products"):
            looks.append(look)
        else:
            print(f"[WARN] Skipped {look_id}: no available classes/options.")

    # 4) Save looks
    (OUT_DIR / "random_looks.json").write_text(
        json.dumps(looks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(looks)} looks → {OUT_DIR/'random_looks.json'}")


if __name__ == "__main__":
    main()

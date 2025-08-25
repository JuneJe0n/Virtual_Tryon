"""
Code that makes random makeup looks with products

Input : json file for makeup products (only lip)
Outputs : random_looks.json         
"""

from __future__ import annotations
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ─── SETTINGS ─────────────────────────────────────────────────────────────────
PRODUCTS_JSON = Path("/home/jiyoon/data/json/options/lviton-options_lips.json")   # products json
OUT_FILE       = Path("/home/jiyoon/data/json/makeup_looks_lips_hex/random_looks.json")        # output file path
N_LOOKS        = 10000                               # how many random looks to make
SEED           = 1230       
# ─────────────────────────────────────────────────────────────────────────────

def load_products(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a list of products at top-level.")
    return data


def main():
    random.seed(SEED)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load product-level JSON
    products = load_products(PRODUCTS_JSON)
    
    # 2) Collect all lip options from all products
    all_lip_options = []
    for product in products:
        code = product.get("code")
        name = product.get("name")
        for option in product.get("options", []):
            all_lip_options.append({
                "code": code,
                "name": name,
                "option": option
            })
    
    print(f"Found {len(all_lip_options)} total lip options")

    # 3) Build random looks with IDs A0..A{N-1} - each with 1 random option
    looks: List[Dict[str, Any]] = []
    for i in range(N_LOOKS):
        look_id = f"A{i}"
        # Pick one random lip option
        chosen = random.choice(all_lip_options)
        
        look = {
            "id": look_id,
            "products": [{
                "code": chosen["code"],
                "name": chosen["name"], 
                "options": [chosen["option"]]
            }]
        }
        looks.append(look)

    # 4) Save looks
    OUT_FILE.write_text(
        json.dumps(looks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(looks)} looks → {OUT_FILE}")

if __name__ == "__main__":
    main()

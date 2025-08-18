"""
Code that makes a paired dataset with the format :

{
  "image": "makeup_results/3316_02487.png",
  "solution": [...options list...],
  "prompt": "<system+user prompt string from chat template>"
}

"""

import os, re, json
from pathlib import Path
from prompts import SYSTEM_PROMPT, QUESTION

JSON_PATH   = "/home/jiyoon/data/json/makeup_looks/lviton-makeups.json"   # makeup looks json
IMAGES_DIR  = "/home/jiyoon/data/LVtion_results/makeup_results"    # files named {makeupId}_{ffhqId}.png
OUT_JSONL   = "/home/jiyoon/data/jsonl/ameli_makeuplooks.jsonl"           # output jsonl


def load_makeup_solutions(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    looks = data if isinstance(data, list) else data.get("data") or data.get("looks") or []
    id2opts = {}
    for look in looks:
        mk_id = look.get("id")
        if mk_id is None: 
            continue
        merged = []
        for prod in look.get("products", []):
            for opt in prod.get("options", []):
                out = {
                    "shape": opt.get("shape"),
                    "color": {
                        "r": opt.get("color", {}).get("r"),
                        "g": opt.get("color", {}).get("g"),
                        "b": opt.get("color", {}).get("b"),
                    },
                    "alpha": opt.get("alpha"),
                    "sigma": opt.get("sigma"),
                }
                if "gamma" in opt and opt["gamma"] is not None:
                    out["gamma"] = opt["gamma"]
                merged.append(out)
        # dedup
        uniq, seen = [], set()
        for o in merged:
            key = (
                o["shape"], o["color"]["r"], o["color"]["g"], o["color"]["b"],
                o["alpha"], o["sigma"], o.get("gamma", None),
            )
            if key not in seen:
                seen.add(key); uniq.append(o)
        id2opts[int(mk_id)] = uniq
    return id2opts

def parse_makeup_id(fname):
    m = re.match(r"(\d+)_", os.path.basename(fname))
    if not m:
        return None
    return int(m.group(1))

def main():
    id2opts = load_makeup_solutions(JSON_PATH)
    paths = sorted(list(Path(IMAGES_DIR).glob("*.png")) + 
                   list(Path(IMAGES_DIR).glob("*.jpg")) +
                   list(Path(IMAGES_DIR).glob("*.jpeg")))

    n_written = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for p in paths:
            mk_id = parse_makeup_id(p.name)
            if mk_id is None or mk_id not in id2opts:
                continue

            # assistant answer text: wrap options in <answer>…</answer>
            answer_obj_list = id2opts[mk_id]          # list[dict] (no split, gamma optional)
            answer_text = "<answer>\n" + json.dumps(answer_obj_list, ensure_ascii=False) + "\n</answer>"

            record = {
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "user",   "value": "<image>\n" + QUESTION},
                    {"from": "assistant", "value": answer_text},
                ],
                "images": [str(p)],   # single image path
                # (optional) keep raw labels for debugging
                "solution": answer_obj_list,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} samples to {OUT_JSONL}")

if __name__ == "__main__":
    main()

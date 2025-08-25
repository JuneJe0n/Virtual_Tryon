"""

Input : makeup json file, makeup imgs
Output : paired dataset with the format
{
  "image": "makeup_results/3316_02487.png",
  "solution": [...options list...],
  "prompt": "<system+user prompt string from chat template>"
}

"""

import os, re, json
from pathlib import Path
from transformers import AutoProcessor
from prompts import SYSTEM_PROMPT, QUESTION

# Paths
JSON_PATH   = "/home/jiyoon/data/json/makeup_looks_lips_hex/random_looks.json"      # makeup looks json
IMAGES_DIR  = "/home/jiyoon/data/imgs/lips_looks/random"            # files named {makeupId}_{ffhqId}.png
OUT_JSONL   = "/home/jiyoon/data/jsonl/training_data/lips/random_looks.jsonl"           # output jsonl

# Qwen model id for chat template building
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def load_makeup_solutions(json_path):
    """
    Parse looks JSON -> {makeup_id(str): [option_dict, ...]}.
    - Keeps: color (hex format)
    - Drops: shape. split, alpha, sigma, gamma
    - Deduplicates identical option dicts
    - Keys in the map are STRINGS so they can be "A0" or "7906".
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    looks = data if isinstance(data, list) else data.get("data") or data.get("looks") or []
    id2opts = {}

    for look in looks:
        mk_id = look.get("id")
        if mk_id is None:
            continue
        mk_id_str = str(mk_id)  # normalize to string

        merged = []
        for prod in look.get("products", []):
            for opt in prod.get("options", []):
                out = {
                    "color": opt.get("color")  # now expects hex string like "#946338"
                }
                merged.append(out)

        # Deduplicate
        uniq, seen = [], set()
        for o in merged:
            key = (
                o["color"],
            )
            if key not in seen:
                seen.add(key)
                uniq.append(o)

        id2opts[mk_id_str] = uniq

    return id2opts


def parse_makeup_id(fname):
    """
    Expect filenames like {makeupId}_{ffhqId}.png
    Returns makeupId as STRING (supports alphanumeric like 'A0' or '7906').
    """
    # match anything up to the first underscore
    m = re.match(r"([^_]+)_", os.path.basename(fname))
    return m.group(1) if m else None


def main():
    # Build chat template processor
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True, padding_side="left")

    # Load solutions map
    id2opts = load_makeup_solutions(JSON_PATH)

    # Collect image files
    paths = sorted(
        list(Path(IMAGES_DIR).glob("*.png"))
        + list(Path(IMAGES_DIR).glob("*.jpg"))
        + list(Path(IMAGES_DIR).glob("*.jpeg"))
    )

    n_written = 0
    n_skipped = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for p in paths:
            mk_id = parse_makeup_id(p.name)  # returns string
            if mk_id is None or mk_id not in id2opts:
                n_skipped += 1
                continue

            # Ground-truth solution list (gamma always present)
            solution_list = id2opts[mk_id]

            # Build prompt via chat template (system + user(image + QUESTION))
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,   # ready for generation
            )

            record = {
                "image": str(p),            # store path; loader can open it later
                "solution": solution_list,  # list of dicts
                "prompt": prompt,           # single string from chat template
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} samples to {OUT_JSONL} (skipped {n_skipped} without matching IDs)")


if __name__ == "__main__":
    main()

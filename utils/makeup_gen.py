
"""
Code for generating imgs based on makeup looks json
---
Inputs : Makeup look json file, Face imgs folder
Ouput : Face imgs w the makeup look put on
"""

import json
import random
from pathlib import Path
from PIL import Image
import numpy as np

from lviton import LViton, MakeupOptions, MakeupShape

# ─── PATH SETTINGS ──────────────────────────────────────────────
JSON_FILE = Path("/home/jiyoon/data/json/makeup_looks/lviton-makeups.json")  # makeup look json
BARE_DIR = Path("/home/jiyoon/data/imgs/test_face")  # face imgs
OUT_DIR = Path("/home/jiyoon/data/imgs/test_results")  # output path
LIB_PATH = Path("/home/jiyoon/LViton_GRPO/LViton/lib/liblviton-x86_64-linux-3.0.3.so")  # compiled LViton shared library
FACE_LANDMARKER = Path("/home/jiyoon/LViton_GRPO/LViton/model/face_landmarker.task")  # mediapipe model
RANDOM_SEED = 42
MAX_TRIES = 20
# ────────────────────────────────────────────────────────────────


def pil_to_rgba_array(img: Image.Image) -> np.ndarray:
    """Convert PIL image to RGBA numpy array (uint8)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def load_random_bare_face(bare_faces: list[Path]) -> tuple[Path, np.ndarray] | None:
    """Pick a random bare-face image and return (path, RGBA array)."""
    path = random.choice(bare_faces)
    try:
        with Image.open(path) as im:
            return path, pil_to_rgba_array(im)
    except Exception:
        return None


def build_makeup_options(product_list: list[dict]) -> list[MakeupOptions]:
    """Build list of MakeupOptions from JSON product definitions."""
    options: list[MakeupOptions] = []
    for product in product_list:
        for opt in product.get("options", []):
            shape_name = opt.get("shape")
            if not shape_name:
                continue
            try:
                shape = getattr(MakeupShape, shape_name)
            except AttributeError:
                # Unknown shape in JSON — skip it.
                continue
            color = opt.get("color", {})
            r, g, b = int(color.get("r", 0)), int(color.get("g", 0)), int(color.get("b", 0))
            alpha = int(opt.get("alpha", 255))
            sigma = int(opt.get("sigma", 0))
            gamma = int(opt.get("gamma", 128))
            clamp = lambda x: max(0, min(255, int(x)))
            options.append(
                MakeupOptions(
                    shape=shape,
                    color=(clamp(r), clamp(g), clamp(b)),
                    alpha=clamp(alpha),
                    sigma=clamp(sigma),
                    gamma=clamp(gamma),
                )
            )
    return options


def main():
    random.seed(RANDOM_SEED)

    # Load JSON looks
    looks = json.loads(JSON_FILE.read_text(encoding="utf-8"))

    # Collect all bare-face image paths
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    bare_faces = [p for p in BARE_DIR.rglob("*") if p.suffix.lower() in exts]
    if not bare_faces:
        raise RuntimeError(f"No images found in {BARE_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Init LViton
    lviton = LViton(lib_path=str(LIB_PATH), face_landmarker_path=str(FACE_LANDMARKER))
    lviton.print_version()

    for look in looks:
        look_id = look["id"]
        options = build_makeup_options(look.get("products", []))
        if not options:
            continue

        # Try up to MAX_TRIES random faces until one detects a face
        for _ in range(MAX_TRIES):
            pick = load_random_bare_face(bare_faces)
            if pick is None:
                continue
            src_path, img_rgba = pick
            if not lviton.set_image(img_rgba):
                continue

            result_rgb = lviton.apply_makeup(options)
            out_name = f"{look_id}_{src_path.stem}.png"  # <makeup_id>_<ffhq_stem>.png
            out_path = OUT_DIR / out_name
            lviton.save_png(result_rgb, str(out_path))
            print(f"Saved {out_name}  (src: {src_path.name})")
            break  # move to next look


if __name__ == "__main__":
    main()

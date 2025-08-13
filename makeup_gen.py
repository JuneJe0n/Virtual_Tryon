#!/usr/bin/env python
# build_dataset.py
import json
import random
from pathlib import Path
from PIL import Image
import numpy as np

from lviton import LViton, MakeupOptions, MakeupShape


# ─── PATH SETTINGS ──────────────────────────────────────────────
JSON_FILE = Path("/home/jiyoon/LViton/data/ameli_data/lviton-makeups.json")  # makeup look json
BARE_DIR = Path("/home/jiyoon/data/FFHQ")  # face imgs
OUT_DIR = Path("/home/jiyoon/LViton/data/test/results/FFHQ_results")  # output path
LIB_PATH = Path("/home/jiyoon/LViton/lib/liblviton-x86_64-linux-3.0.3.so")  # compiled LViton shared library
FACE_LANDMARKER = Path("/home/jiyoon/LViton/model/face_landmarker.task")  # mediapipe model 
RANDOM_SEED = 42
# ────────────────────────────────────────────────────────────────


def pil_to_rgba_array(img: Image.Image) -> np.ndarray:
    """Convert PIL image to RGBA numpy array (uint8)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return np.array(img, dtype=np.uint8)


def load_random_bare_face(bare_faces: list[Path]) -> np.ndarray | None:
    """Pick a random bare-face image and return RGBA numpy array."""
    path = random.choice(bare_faces)
    try:
        with Image.open(path) as im:
            return pil_to_rgba_array(im)
    except Exception:
        return None


def build_makeup_options(product_list: list[dict]) -> list[MakeupOptions]:
    """Build list of MakeupOptions from JSON product definitions."""
    options = []
    for product in product_list:
        for opt in product.get("options", []):
            try:
                shape = getattr(MakeupShape, opt["shape"])
            except AttributeError:
                continue
            color = opt["color"]
            options.append(
                MakeupOptions(
                    shape=shape,
                    color=(color["r"], color["g"], color["b"]),
                    alpha=opt["alpha"],
                    sigma=opt["sigma"],
                    gamma=opt["gamma"],
                )
            )
    return options


def main():
    random.seed(RANDOM_SEED)

    # Load JSON looks
    looks = json.loads(JSON_FILE.read_text(encoding="utf-8"))

    # Collect all bare-face image paths
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    bare_faces = [p for p in BARE_DIR.rglob("*") if p.suffix.lower() in exts]
    if not bare_faces:
        raise RuntimeError(f"No images found in {BARE_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Init LViton
    lviton = LViton(lib_path=str(LIB_PATH), face_landmarker_path=str(FACE_LANDMARKER))
    lviton.print_version()

    for look in looks:
        look_id = look["id"]
        options = build_makeup_options(look["products"])
        if not options:
            continue

        # Pick a bare face and apply makeup
        for _ in range(20):  # try up to 20 random faces if detection fails
            img_rgba = load_random_bare_face(bare_faces)
            if img_rgba is None:
                continue
            if not lviton.set_image(img_rgba):
                continue

            result = lviton.apply_makeup(options)
            lviton.save_png(result, str(OUT_DIR / f"{look_id}.png"))
            print(f"Saved {look_id}.png")
            break


if __name__ == "__main__":
    main()

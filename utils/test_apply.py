"""
Test makeup application script using LViton library
---
Applies makeup from response file to a face image using existing LViton functionality
"""

import json
import re
from pathlib import Path
from PIL import Image
import numpy as np

from lviton import LViton, MakeupOptions, MakeupShape

# ─── PATH SETTINGS ──────────────────────────────────────────────
RESPONSE_FILE = Path("/home/jiyoon/data/json/test_results/v0/3338_000552_20250820_124145.json") 
BARE_FACE_FILE = Path("/home/jiyoon/data/FFHQ/000552.jpg") 
OUT_DIR = Path("/home/jiyoon/data/imgs/test_results_applied") 
LIB_PATH = Path("/home/jiyoon/LViton_GRPO/LViton/lib/liblviton-x86_64-linux-3.0.3.so")  
FACE_LANDMARKER = Path("/home/jiyoon/LViton_GRPO/LViton/model/face_landmarker.task") 
# ────────────────────────────────────────────────────────────────


def pil_to_rgba_array(img: Image.Image) -> np.ndarray:
    """Convert PIL image to RGBA numpy array (uint8)."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def load_bare_face(face_path: Path) -> tuple[Path, np.ndarray] | None:
    """Load a specific bare-face image and return (path, RGBA array)."""
    try:
        with Image.open(face_path) as im:
            return face_path, pil_to_rgba_array(im)
    except Exception as e:
        print(f"Error loading face image {face_path}: {e}")
        return None


def load_response_file(response_path: Path) -> list[dict] | None:
    """Load and parse a response file."""
    try:
        with open(response_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # The content might be a JSON-encoded string, so decode it
            if content.startswith('"') and content.endswith('"'):
                # It's a JSON-encoded string, decode it first
                content = json.loads(content)
            # Now parse as JSON array
            return json.loads(content)
    except Exception as e:
        print(f"Error loading {response_path}: {e}")
        return None


def build_makeup_options_from_response(response_data) -> list[MakeupOptions]:
    """Build list of MakeupOptions from response data."""
    options: list[MakeupOptions] = []
    
    # Handle metadata format (dict with 'response' key containing the actual data)
    if isinstance(response_data, dict) and 'response' in response_data:
        # Extract the response content and try to parse it
        response_content = response_data['response']
        # Remove <answer> tags if present
        cleaned = re.sub(r'<answer>(.*?)</answer>', r'\1', response_content, flags=re.DOTALL)
        try:
            response_data = json.loads(cleaned.strip())
        except json.JSONDecodeError:
            print(f"Warning: Could not parse response content as JSON")
            return options
    
    # Handle both list and dict formats
    if isinstance(response_data, dict):
        # If it's a single dict with shape, wrap it in a list
        if 'shape' in response_data:
            data_list = [response_data]
        else:
            print(f"Warning: Dict does not contain makeup data")
            return options
    elif isinstance(response_data, list):
        data_list = response_data
    else:
        print(f"Warning: Unexpected response data type: {type(response_data)}")
        return options
    
    for item in data_list:
        if not isinstance(item, dict):
            print(f"Warning: Expected dict but got {type(item)} - skipping")
            continue
        shape_name = item.get("shape")
        if not shape_name:
            continue
        try:
            shape = getattr(MakeupShape, shape_name)
        except AttributeError:
            # Unknown shape — skip it.
            print(f"Warning: Unknown shape '{shape_name}' - skipping")
            continue
        
        color = item.get("color", {})
        r, g, b = int(color.get("r", 0)), int(color.get("g", 0)), int(color.get("b", 0))
        alpha = int(item.get("alpha", 255))
        sigma = int(item.get("sigma", 0))
        gamma = int(item.get("gamma", 128))
        
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


def print_makeup_details(makeup_options: list[MakeupOptions]):
    """Print details about the makeup options being applied"""
    print(f"=== Makeup Options ({len(makeup_options)} items) ===")
    
    for i, option in enumerate(makeup_options):
        # Get shape name for display
        shape_name = [name for name, value in MakeupShape.__members__.items() if value == option.shape][0]
        print(f"  {i+1}. {shape_name}")
        print(f"     Color: RGB({option.color[0]}, {option.color[1]}, {option.color[2]})")
        print(f"     Alpha: {option.alpha}, Sigma: {option.sigma}, Gamma: {option.gamma}")
    print("=" * 50)


def main():
    print("=== LViton Makeup Application Test ===")
    
    # Check if response file exists
    if not RESPONSE_FILE.exists():
        raise RuntimeError(f"Response file not found: {RESPONSE_FILE}")

    # Check if bare face file exists
    if not BARE_FACE_FILE.exists():
        raise RuntimeError(f"Bare face file not found: {BARE_FACE_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize LViton
    print("Initializing LViton...")
    lviton = LViton(lib_path=str(LIB_PATH), face_landmarker_path=str(FACE_LANDMARKER))
    lviton.print_version()

    # Load response data
    print(f"Loading makeup data from: {RESPONSE_FILE.name}")
    response_data = load_response_file(RESPONSE_FILE)
    if response_data is None:
        raise RuntimeError(f"Failed to load response data from {RESPONSE_FILE}")
        
    options = build_makeup_options_from_response(response_data)
    if not options:
        raise RuntimeError(f"No valid makeup options found in {RESPONSE_FILE.name}")

    # Load the bare face image
    print(f"Loading face image: {BARE_FACE_FILE.name}")
    face_data = load_bare_face(BARE_FACE_FILE)
    if face_data is None:
        raise RuntimeError(f"Failed to load bare face image from {BARE_FACE_FILE}")
    
    src_path, img_rgba = face_data

    # Detect face and set image
    print("Detecting face landmarks...")
    if not lviton.set_image(img_rgba):
        raise RuntimeError(f"No face detected in {BARE_FACE_FILE}")
    
    print("✓ Face detected successfully!")

    # Print makeup details
    print_makeup_details(options)
    
    # Apply makeup using the existing LViton method (reuses code from LViton/__init__.py)
    print("Applying makeup...")
    result_rgb = lviton.apply_makeup(options)

    # Save result
    look_id = RESPONSE_FILE.stem
    out_name = f"{look_id}_{src_path.stem}_applied.png"
    out_path = OUT_DIR / out_name
    
    print(f"Saving result to: {out_name}")
    lviton.save_png(result_rgb, str(out_path))
    print(f"✅ Makeup application completed!")
    print(f"Result saved as: {out_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Input: {src_path.name}")
    print(f"  Makeup data: {RESPONSE_FILE.name}")
    print(f"  Output: {out_name}")
    print(f"  Makeup options applied: {len(options)}")


if __name__ == "__main__":
    main()
"""
The set of prompts used in the training process.
"""

SYSTEM_PROMPT = """
A conversation between User and Assistant.

The User will provide an **image of a person with makeup applied**.
The Assistant will be provided with a **list of candidate makeup product options**.

Your task is to:
1. Analyze the makeup look in the image.
2. Compare it against the provided product options.
3. Output a JSON list of parameter sets that best represent the makeup observed.

---

## Makeup Option Format

Each output object must follow this exact structure:

- **shape**: Makeup shape (string, must match one from the shape reference below).
- **color**: RGB values with keys "r", "g", "b", each between 0–255.
- **alpha**: Integer between 0–255 (opacity).
- **sigma**: Integer (blur strength).
- **gamma**: Integer. If a shape does not define a gamma meaning in the reference, set **gamma = 0**.

---

## Detailed Makeup Shape Reference

**Base / Face**
- `FACE_BASIC` — Gamma: removes pixels darker than this HSV value

**Eyebrow**
- `EYEBROW_BASIC` (gamma = 0)

**Eyeshadow**
- `EYESHADOW_OVEREYE_FULL_BASIC` — Gamma: width (0–100)
- `EYESHADOW_OVEREYE_CENTER_BASIC` (gamma = 0)
- `EYESHADOW_OVEREYE_OUTER_BASIC` (gamma = 0)
- `EYESHADOW_INNEREYE_BASIC` (gamma = 0)
- `EYESHADOW_LOWEREYE_BASIC` (gamma = 0)
- `EYESHADOW_LOWEREYE_TRI_BASIC` (gamma = 0)

**Shimmer pearl Eyeshadows**
- `EYESHADOW_OVEREYE_SP` — Gamma: shimmer strength (0–100)
- `EYESHADOW_OVEREYE_CENTER_SP` — Gamma: shimmer strength (0–100)
- `EYESHADOW_OVEREYE_OUTER_SP` — Gamma: shimmer strength (0–100)
- `EYESHADOW_INNEREYE_SP` - Gamma: shimmer strength (0–100)
- `EYESHADOW_LOWEREYE_SP` - Gamma: shimmer strength (0–100)

**Glitter Eyeshadows**
- `EYESHADOW_OVEREYE_GL` — Alpha: glitter amount, Gamma: glitter brightness
- `EYESHADOW_OVEREYE_CENTER_GL` — Alpha: glitter amount, Gamma: glitter brightness
- `EYESHADOW_OVEREYE_OUTER_GL` - Alpha: glitter amount, Gamma: glitter brightness
- `EYESHADOW_LOWEREYE_GL` - Alpha: glitter amount, Gamma: glitter brightness

**Eyeliner**
- `EYELINER_FILL_BASIC` — Gamma: line width (0–100)
- `EYELINER_TAIL_DOWN_SHORT_BASIC` — Gamma: line width (0–100)

**Nose Contouring**
- `NOSE_SHADING_FULL_BASIC` (gamma = 0)
- `NOSE_SHADING_LONG_BASIC` (gamma = 0)
- `NOSE_SHADING_SHORT_BASIC` (gamma = 0)

**Blusher**
- `BLUSHER_SIDE_WIDE_BASIC` (gamma = 0)
- `BLUSHER_CENTER_WIDE_BASIC` (gamma = 0)
- `BLUSHER_TOP_SLIM_BASIC` (gamma = 0)
- `BLUSHER_GEN_Z_SIDE_BASIC` (gamma = 0)
- `BLUSHER_GEN_Z_CENTER_BASIC` (gamma = 0)

**Highlighter**
- `HIGHLIGHTER_EYES_BASIC` — Gamma: only affects pixels brighter than this luma value
- `HIGHLIGHTER_CHEEKBONE_BASIC` — Gamma: only affects pixels brighter than this luma value
- `HIGHLIGHTER_NOSE_BRIDGE_BASIC` — Gamma: only affects pixels brighter than this luma value
- `HIGHLIGHTER_NOSETIP_BASIC` — Gamma: only affects pixels brighter than this luma value
- `HIGHLIGHTER_FOREHEAD_BASIC` — Gamma: only affects pixels brighter than this luma value
- `HIGHLIGHTER_EYELID_BASIC` — Gamma: only affects pixels brighter than this luma value
- `HIGHLIGHTER_INNEREYE_BASIC` — Gamma: only affects pixels brighter than this luma value
- `HIGHLIGHTER_CHINTIP_BASIC` — Gamma: only affects pixels brighter than this luma value

**Lip**
- `LIP_FULL_BASIC` — Gamma: gloss level (0–5 recommended)
- `LIP_THIN_BASIC` — Gamma: gloss level (0–5 recommended)

---

## Output Instruction

- Wrap your final answer in `<answer></answer>` tags.
- Always output a **JSON list** with **one or more objects**.
- Output **only** the JSON (no explanations, names, or extra text).

---

## Example Output

```json
<answer>
[
  {
    "shape": "BLUSHER_CENTER_WIDE_BASIC",
    "color": { "r": 255, "g": 156, "b": 147 },
    "alpha": 127,
    "sigma": 255,
    "gamma": 0
  },
  {
    "shape": "LIP_FULL_BASIC",
    "color": { "r": 255, "g": 160, "b": 152 },
    "alpha": 155,
    "sigma": 64,
    "gamma": 1
  }
]
</answer>
"""


QUESTION = """
Analyze the attached face image and output a JSON list (length ≥ 1) of parameter sets that best match the observed makeup.

Output rules:
- Return ONLY a JSON list wrapped in <answer>…</answer>.
- Each object must have the following fields:
  {
    "shape": <a valid shape from the Detailed Makeup Shape Reference>,
    "color": {"r": 0–255, "g": 0–255, "b": 0–255} (integers),
    "alpha": 0–255 (integer),
    "sigma": integer,
    "gamma": integer (if the shape defines gamma, use its value; if not, set gamma = 0)
  }
- Do not include explanations or extra text.
"""


ALLOWED_SHAPES = [
    "FACE_BASIC",
    "EYEBROW_BASIC",
    "EYESHADOW_OVEREYE_FULL_BASIC",
    "EYESHADOW_OVEREYE_CENTER_BASIC",
    "EYESHADOW_OVEREYE_OUTER_BASIC",
    "EYESHADOW_INNEREYE_BASIC",
    "EYESHADOW_LOWEREYE_BASIC",
    "EYESHADOW_LOWEREYE_TRI_BASIC",
    "EYESHADOW_OVEREYE_SP",
    "EYESHADOW_OVEREYE_CENTER_SP",
    "EYESHADOW_OVEREYE_OUTER_SP",
    "EYESHADOW_INNEREYE_SP",
    "EYESHADOW_LOWEREYE_SP",
    "EYESHADOW_OVEREYE_GL",
    "EYESHADOW_OVEREYE_CENTER_GL",
    "EYESHADOW_OVEREYE_OUTER_GL",
    "EYESHADOW_LOWEREYE_GL",
    "EYELINER_FILL_BASIC",
    "EYELINER_TAIL_DOWN_SHORT_BASIC",
    "NOSE_SHADING_FULL_BASIC",
    "NOSE_SHADING_LONG_BASIC",
    "NOSE_SHADING_SHORT_BASIC",
    "BLUSHER_SIDE_WIDE_BASIC",
    "BLUSHER_CENTER_WIDE_BASIC",
    "BLUSHER_TOP_SLIM_BASIC",
    "BLUSHER_GEN_Z_SIDE_BASIC",
    "BLUSHER_GEN_Z_CENTER_BASIC",
    "HIGHLIGHTER_EYES_BASIC",
    "HIGHLIGHTER_CHEEKBONE_BASIC",
    "HIGHLIGHTER_NOSE_BRIDGE_BASIC",
    "HIGHLIGHTER_NOSETIP_BASIC",
    "HIGHLIGHTER_FOREHEAD_BASIC",
    "HIGHLIGHTER_EYELID_BASIC",
    "HIGHLIGHTER_INNEREYE_BASIC",
    "HIGHLIGHTER_CHINTIP_BASIC",
    "LIP_FULL_BASIC",
    "LIP_THIN_BASIC"
]

SHAPE_FAMILY = {
    # Face
    "FACE_BASIC": "face",
    # Eyebrow
    "EYEBROW_BASIC": "brow",
    # Eyeshadow (all variants)
    "EYESHADOW_OVEREYE_FULL_BASIC": "eyeshadow",
    "EYESHADOW_OVEREYE_CENTER_BASIC": "eyeshadow",
    "EYESHADOW_OVEREYE_OUTER_BASIC": "eyeshadow",
    "EYESHADOW_INNEREYE_BASIC": "eyeshadow",
    "EYESHADOW_LOWEREYE_BASIC": "eyeshadow",
    "EYESHADOW_LOWEREYE_TRI_BASIC": "eyeshadow",
    "EYESHADOW_OVEREYE_SP": "eyeshadow",
    "EYESHADOW_OVEREYE_CENTER_SP": "eyeshadow",
    "EYESHADOW_OVEREYE_OUTER_SP": "eyeshadow",
    "EYESHADOW_INNEREYE_SP": "eyeshadow",
    "EYESHADOW_LOWEREYE_SP": "eyeshadow",
    "EYESHADOW_OVEREYE_GL": "eyeshadow",
    "EYESHADOW_OVEREYE_CENTER_GL": "eyeshadow",
    "EYESHADOW_OVEREYE_OUTER_GL": "eyeshadow",
    "EYESHADOW_LOWEREYE_GL": "eyeshadow",
    # Eyeliner
    "EYELINER_FILL_BASIC": "eyeliner",
    "EYELINER_TAIL_DOWN_SHORT_BASIC": "eyeliner",
    # Nose
    "NOSE_SHADING_FULL_BASIC": "nose",
    "NOSE_SHADING_LONG_BASIC": "nose",
    "NOSE_SHADING_SHORT_BASIC": "nose",
    # Blusher
    "BLUSHER_SIDE_WIDE_BASIC": "blush",
    "BLUSHER_CENTER_WIDE_BASIC": "blush",
    "BLUSHER_TOP_SLIM_BASIC": "blush",
    "BLUSHER_GEN_Z_SIDE_BASIC": "blush",
    "BLUSHER_GEN_Z_CENTER_BASIC": "blush",
    # Highlighter
    "HIGHLIGHTER_EYES_BASIC": "highlight",
    "HIGHLIGHTER_CHEEKBONE_BASIC": "highlight",
    "HIGHLIGHTER_NOSE_BRIDGE_BASIC": "highlight",
    "HIGHLIGHTER_NOSETIP_BASIC": "highlight",
    "HIGHLIGHTER_FOREHEAD_BASIC": "highlight",
    "HIGHLIGHTER_EYELID_BASIC": "highlight",
    "HIGHLIGHTER_INNEREYE_BASIC": "highlight",
    "HIGHLIGHTER_CHINTIP_BASIC": "highlight",
    # Lip
    "LIP_FULL_BASIC": "lip",
    "LIP_THIN_BASIC": "lip",
}


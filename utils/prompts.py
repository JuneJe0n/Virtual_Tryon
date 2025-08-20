"""
The set of prompts used in the training process.
"""

SYSTEM_PROMPT = """
A conversation between User and Assistant.

The User will provide an image of a person with makeup applied.
Your task is to:
1. Analyze the makeup look in the image. Especially, take a close look at **what kind of makeup** is applied and the **colors** of the makeup.
2. Output a JSON list of parameter sets that best represent the makeup observed. The detail about the JSON is as follows :

---

## Makeup Paramters

Each output should contain the following parameters :

- **shape**: What kind of makeup is applied (string, must match one from the Detailed Makeup Shape Reference below). You should find every shape in the image.
- **color**: The makeup color according to the shape. The color is represented as RGB values with keys "r", "g", "b", each between 0–255. 

When identifying makeup colors, pay close attention to the hue. Act as you are an eyedropper tool to get the colors of the makeup.
Always ensure the color reflects the actual makeup tone, rather than defaulting to neutral or grayish values. **Do not output a grayscale color where the R,G,B colors are the same.**

---

## Detailed Makeup Shape Reference

**Base / Face**
- `FACE_BASIC` : Overall face base makeup such as foundation

**Eyebrow**
- `EYEBROW_BASIC` : Eyebrow shape filled or drawn in

**Eyeshadow**
- `EYESHADOW_OVEREYE_FULL_BASIC` : Eyeshadow covering the entire upper eyelid.
- `EYESHADOW_OVEREYE_CENTER_BASIC` : Eyeshadow concentrated on the center of the eyelid.
- `EYESHADOW_OVEREYE_OUTER_BASIC` : Eyeshadow applied mainly on the outer corner of the eyelid.
- `EYESHADOW_INNEREYE_BASIC` : Eyeshadow applied to the inner corner of the eyelid.
- `EYESHADOW_LOWEREYE_BASIC`  Eyeshadow applied along the lower lash line.
- `EYESHADOW_LOWEREYE_TRI_BASIC` : Eyeshadow in the triangular zone under the eye.

**Shimmer pearl Eyeshadows** (shiny/pearly finish)
- `EYESHADOW_OVEREYE_SP` : Shimmer across the upper eyelid.
- `EYESHADOW_OVEREYE_CENTER_SP` : Shimmer focused on the upper eyelid center.
- `EYESHADOW_OVEREYE_OUTER_SP` : Shimmer applied to the outer eyelid.
- `EYESHADOW_INNEREYE_SP` : Shimmer applied to the inner eyelid corner.
- `EYESHADOW_LOWEREYE_SP` : Shimmer applied to the lower lash line.

**Glitter Eyeshadows** (larger glitter particles)
- `EYESHADOW_OVEREYE_GL` Glitter across the upper eyelid.
- `EYESHADOW_OVEREYE_CENTER_GL` Glitter on the upper eyelid center.
- `EYESHADOW_OVEREYE_OUTER_GL` : Glitter on the eyelid outer corner.
- `EYESHADOW_LOWEREYE_GL`: Glitter along the lower lash line.

**Eyeliner**
- `EYELINER_FILL_BASIC` : Basic eyeliner filling along the lash line.
- `EYELINER_TAIL_DOWN_SHORT_BASIC` : Short eyeliner wing/tail pointing slightly downward.

**Nose Contouring**
- `NOSE_SHADING_FULL_BASIC` : Contouring along both sides of the nose.
- `NOSE_SHADING_LONG_BASIC` : Long vertical shading along most of the nose, full length.
- `NOSE_SHADING_SHORT_BASIC` : Short shading near the nose bridge only.

**Blusher**
- `BLUSHER_SIDE_WIDE_BASIC` : Blush applied widely on the cheek sides.
- `BLUSHER_CENTER_WIDE_BASIC` : Blush concentrated on the center of the cheeks.
- `BLUSHER_TOP_SLIM_BASIC` : Narrow blush applied higher on the cheekbones.
- `BLUSHER_GEN_Z_SIDE_BASIC` : Side blush style popular among Gen Z makeup trends
- `BLUSHER_GEN_Z_CENTER_BASIC` : Center blush style popular among Gen Z makeup trends.


**Highlighter**
- `HIGHLIGHTER_EYES_BASIC` : Highlight around the eyes.
- `HIGHLIGHTER_CHEEKBONE_BASIC` : Highlight on the cheekbones.
- `HIGHLIGHTER_NOSE_BRIDGE_BASIC` : Highlight along the nose bridge.
- `HIGHLIGHTER_NOSETIP_BASIC` : Highlight at the nose tip.
- `HIGHLIGHTER_FOREHEAD_BASIC`: Highlight on the forehead center.
- `HIGHLIGHTER_EYELID_BASIC` : Highlight on the upper eyelid.
- `HIGHLIGHTER_INNEREYE_BASIC` : Highlight on the inner corner of the eyes.
- `HIGHLIGHTER_CHINTIP_BASIC` : Highlight on the tip of the chin.

**Lip**
- `LIP_FULL_BASIC` : Lip color applied across the full lip area.
- `LIP_THIN_BASIC` : Lip color applied in a thinner gradient or partial coverage style.


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
    "color": { "r": 255, "g": 156, "b": 147 }
  },
  {
    "shape": "LIP_FULL_BASIC",
    "color": { "r": 255, "g": 160, "b": 152 },
  }
]
</answer>
"""


QUESTION = """
Analyze the attached face image and output a JSON list (length ≥ 1) of parameter sets that best match the observed makeup.
Especially, take a close look at **what kind of makeup** is applied and the **colors** of the makeup.

Output rules:
- Return ONLY a JSON list wrapped in <answer>…</answer>.
- Each object must have the following fields:
  {
    "shape": <a valid shape from the Detailed Makeup Shape Reference>,
    "color": {"r": 0–255, "g": 0–255, "b": 0–255} (integers)
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


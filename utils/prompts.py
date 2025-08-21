"""
The set of prompts used in the training process.
"""

SYSTEM_PROMPT = """
A conversation between User and Assistant.
The User will provide you an image of a person with makeup applied.
Your task is to:
1. Analyze the makeup, concentrating on the makeup **shape** and **color**. 
  - **shape**: What kind of makeup is applied (string, must be one from the Allowed Shapes below). You should find every shape in the image.
  - **color**: The makeup color according to the shape.The color is represented as a hex color code (e.g., #FF9C93). When identifying makeup colors, act as you are an eyedropper tool to get the colors of the makeup. 
---
## Allowed Shapes
"FACE_BASIC",
"EYEBROW_BASIC",
"EYESHADOW_OVEREYE_FULL_BASIC",
"EYESHADOW_OVEREYE_CENTER_BASIC",
"EYESHADOW_OVEREYE_OUTER_BASIC",
"EYESHADOW_INNEREYE_BASIC",
"EYESHADOW_LOWEREYE_BASIC",
"EYESHADOW_LOWEREYE_TRI_BASIC",
"BLUSHER_SIDE_WIDE_BASIC",
"BLUSHER_CENTER_WIDE_BASIC",
"BLUSHER_TOP_SLIM_BASIC",
"BLUSHER_GEN_Z_SIDE_BASIC",
"BLUSHER_GEN_Z_CENTER_BASIC",
"LIP_FULL_BASIC",
---
## Detailed Output Instruction
- Wrap your final answer in `<answer></answer>` tags.
- Always output a **JSON list** with **one or more objects**.
- Output **only** the JSON (no explanations, names, or extra text).
- Example Output :
```json
<answer>
[
  {
    "shape": "BLUSHER_CENTER_WIDE_BASIC",
    "color": "#FF9C93"
  },
  {
    "shape": "LIP_FULL_BASIC",
    "color": #951541
  }
]
</answer>
"""


QUESTION = """
Analyze the attached face image and output a JSON list (length ≥ 1) of parameter sets that best match the observed makeup.
Especially, take a close look at the makeup **shape** and **color**. 
When identifying makeup colors, act as you are an eyedropper tool to get the colors of the makeup. 
Output rules:
- Return ONLY a JSON list wrapped in <answer>…</answer>.
- Each object must have the following fields:
  {
    "shape": <a valid shape from the Allowed Shapes>,
    "color": "<hex color code>" (e.g., "#FF9C93")
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
    "BLUSHER_SIDE_WIDE_BASIC",
    "BLUSHER_CENTER_WIDE_BASIC",
    "BLUSHER_TOP_SLIM_BASIC",
    "BLUSHER_GEN_Z_SIDE_BASIC",
    "BLUSHER_GEN_Z_CENTER_BASIC",
    "LIP_FULL_BASIC",
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
    # Blusher
    "BLUSHER_SIDE_WIDE_BASIC": "blush",
    "BLUSHER_CENTER_WIDE_BASIC": "blush",
    "BLUSHER_TOP_SLIM_BASIC": "blush",
    "BLUSHER_GEN_Z_SIDE_BASIC": "blush",
    "BLUSHER_GEN_Z_CENTER_BASIC": "blush",
    # Lip
    "LIP_FULL_BASIC": "lip"
}



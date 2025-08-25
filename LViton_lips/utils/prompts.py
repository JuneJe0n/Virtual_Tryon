"""
The set of prompts used in the training process.
"""

SYSTEM_PROMPT = """
A conversation between User and Assistant.
The User will provide you an image of a person with makeup applied.

Your task is to:
Analyze the lip color of the makeup. The color is represented as a hex color code (e.g., #FF9C93). 
When identifying the lip color, act as you are an eyedropper tool targeting the lip region.

## Detailed Output Instruction
- Wrap your final answer in `<answer></answer>` tags.
- Always output a **JSON**
- Output **only** the JSON (no explanations, names, or extra text).
- Example Output :
<answer>
{
  "color": "#FF9C93"
}
</answer>
"""


QUESTION = """
Analyze the attached face image and output a JSON that contains the hex color code that best matches the observed lip color.
When identifying the lip color, act as you are an eyedropper tool targeting the lip region.

Output rules:
- Return ONLY a JSON wrapped in <answer>…</answer>.
- Each object must have the following fields:
  {
    "color": "<hex color code>" (e.g., "#FF9C93")
  }
- Do not include explanations or extra text.
"""


ALLOWED_SHAPES = [
    "LIP_FULL_BASIC",
]





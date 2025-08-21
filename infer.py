"""
Full inference pipeline
Input : model ckpt, GT makeup imgs
Output : generated options JSON, generated makeup applied img
"""
import os
import json
import re
import torch
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from utils import SYSTEM_PROMPT, QUESTION
from utils import pil_to_rgba_array, build_makeup_options
from lviton import LViton


def hex_to_rgb(hex_color: str) -> dict:
    """Convert hex color to RGB dict"""
    hex_color = hex_color.lstrip('#')
    return {
        "r": int(hex_color[0:2], 16),
        "g": int(hex_color[2:4], 16),
        "b": int(hex_color[4:6], 16)
    }


def get_gamma_value(shape: str) -> int:
    """Get gamma value based on shape"""
    gamma_map = {
        "FACE_BASIC": 70,
        "EYEBROW_BASIC": 0,
        "EYESHADOW_OVEREYE_FULL_BASIC": 60,
        "EYESHADOW_OVEREYE_CENTER_BASIC": 0,
        "EYESHADOW_OVEREYE_OUTER_BASIC": 0,
        "EYESHADOW_INNEREYE_BASIC": 0,
        "EYESHADOW_LOWEREYE_BASIC": 0,
        "EYESHADOW_LOWEREYE_TRI_BASIC": 0,
        "BLUSHER_SIDE_WIDE_BASIC": 0,
        "BLUSHER_CENTER_WIDE_BASIC": 0,
        "BLUSHER_TOP_SLIM_BASIC": 0,
        "BLUSHER_GEN_Z_SIDE_BASIC": 0,
        "BLUSHER_GEN_Z_CENTER_BASIC": 0,
        "LIP_FULL_BASIC": 1
    }
    return gamma_map.get(shape, 0)


def format_makeup_options(response: str) -> list:
    """Convert model response to formatted makeup options"""
    try:
        data = json.loads(response)
        formatted_options = []
        
        for item in data:
            if isinstance(item, dict) and "shape" in item and "color" in item:
                shape = item["shape"]
                hex_color = item["color"]
                
                rgb_color = hex_to_rgb(hex_color)
                
                formatted_option = {
                    "shape": shape,
                    "color": rgb_color,
                    "alpha": 50,
                    "sigma": 60,
                    "gamma": get_gamma_value(shape),
                    "split": 0
                }
                formatted_options.append(formatted_option)
        
        return formatted_options
        
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def load_model_and_processor(checkpoint_path: str, base_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """ Load the fine-tuned model and processor """
    print(f"Loading base model: {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True, padding_side="left")
    
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Loading LoRA weights from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    return model, processor


def generate_response(model, processor, image_path: str, max_new_tokens: int = 512):
    """Generate response for given image and prompt"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")

    messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION},
                    ],
                },
            ]
    
    text_input = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=text_input,
        images=image,
        return_tensors="pt",
        padding=True
    )
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    return response


def clean_response(response: str) -> str:
    """Clean the response by removing escape characters and extracting content from <answer> tags"""
    # Remove common escape sequences
    cleaned = response.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace("\\'", "'")
    
    # Extract content between <answer> tags if they exist
    answer_match = re.search(r'<answer>(.*?)</answer>', cleaned, re.DOTALL)
    if answer_match:
        cleaned = answer_match.group(1).strip()
    
    return cleaned


def save_result_to_json(response: str, image_path: str, output_dir: str, base_model_id: str, checkpoint_path: str):
    """ Save the generated response to a JSON file """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean the response
    cleaned_response = clean_response(response)
    
    # Format the response to the required structure
    formatted_options = format_makeup_options(cleaned_response)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.basename(image_path).split('.')[0]
    output_filename = f"{image_name}_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_options, f, indent=2, ensure_ascii=False)
    
    return output_path


def apply_makeup_to_bare_face(json_path: str, bare_face_path: str, output_dir: str):
    """Apply makeup from JSON to bare face image and save result."""
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load bare face image
    if not os.path.exists(bare_face_path):
        raise FileNotFoundError(f"Bare face image not found: {bare_face_path}")
    
    bare_face = Image.open(bare_face_path).convert("RGB")
    img_rgba = pil_to_rgba_array(bare_face)
    
    # Initialize LViton
    lib_path = "/home/jiyoon/LViton_GRPO/LViton/lib/liblviton-x86_64-linux-3.0.3.so"
    face_landmarker_path = "/home/jiyoon/LViton_GRPO/LViton/model/face_landmarker.task"
    lviton = LViton(lib_path=lib_path, face_landmarker_path=face_landmarker_path)
    
    # Set image and check if face is detected
    if not lviton.set_image(img_rgba):
        raise RuntimeError("No face detected in the bare face image")
    
    # Build makeup options from JSON - wrap in product structure expected by build_makeup_options
    products = [{"options": json_data}]
    makeup_options = build_makeup_options(products)
    if not makeup_options:
        raise ValueError("No valid makeup options found in JSON")
    
    # Apply makeup
    result_rgb = lviton.apply_makeup(makeup_options)
    
    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = os.path.basename(json_path).split('.')[0]
    bare_face_filename = os.path.basename(bare_face_path).split('.')[0]
    output_filename = f"{json_filename}_{bare_face_filename}_applied.png"
    output_path = os.path.join(output_dir, output_filename)
    
    lviton.save_png(result_rgb, output_path)
    print(f"Applied makeup saved to: {output_path}")
    
    return output_path


def main():
    BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    CKPT_PATH = "/home/jiyoon/data/ckpts/Qwen2.5-VL-3B-Instruct-GRPO-v6/checkpoint-150"
    IMG_PATH = "/home/jiyoon/data/imgs/test/makeup_face/v6/7905_june.png"
    JSON_OUTPUT_DIR = "/home/jiyoon/data/json/test_results/v6/ckpt-150"
    BARE_FACE_PATH = "/home/jiyoon/data/imgs/test/bare_face/june.JPG"
    APPLIED_OUTPUT_DIR = "/home/jiyoon/data/imgs/test/test_results_applied/v6"
    max_tokens = 512

    print(f"✅ GT imf in: {IMG_PATH}")
    
    print("Loading model...")
    model, processor = load_model_and_processor(CKPT_PATH, BASE_MODEL)

    print("Generating response...")
    response = generate_response(model, processor, IMG_PATH, max_tokens)
    
    print("\nSaving result to JSON...")
    json_path = save_result_to_json(response, IMG_PATH, JSON_OUTPUT_DIR, BASE_MODEL, CKPT_PATH)
    print(f"✅ JSON result saved to: {json_path}")
    
    print(f"\nApplying makeup to bare face image...")
    applied_path = apply_makeup_to_bare_face(json_path, BARE_FACE_PATH, APPLIED_OUTPUT_DIR)
    print(f"✅ Applied makeup image saved to: {applied_path}")


if __name__ == "__main__": 
    main()
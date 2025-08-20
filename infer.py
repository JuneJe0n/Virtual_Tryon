"""
Inference script for LViton GRPO model
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.basename(image_path).split('.')[0]
    output_filename = f"{image_name}_{timestamp}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_response)
    
    return output_path


def main():
    BASE_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    CKPT_PATH = "/home/jiyoon/data/ckpts/Qwen2.5-VL-3B-Instruct-GRPO/checkpoint-400"
    IMG_PATH = "/home/jiyoon/data/imgs/test_results/3338_000552.png"
    OUTPUT_DIR = "/home/jiyoon/data/json/test_results/v0"
    max_tokens = 512
    
    print("Loading model...")
    model, processor = load_model_and_processor(CKPT_PATH, BASE_MODEL)
    
    print("Generating response...")
    response = generate_response(model, processor, IMG_PATH, max_tokens)
    
    print("\nSaving result to JSON...")
    json_path = save_result_to_json(response, IMG_PATH, OUTPUT_DIR, BASE_MODEL, CKPT_PATH)
    print(f"Result saved to: {json_path}")


if __name__ == "__main__":
    main()
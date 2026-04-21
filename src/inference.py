"""Rung-1 sanity check: load Qwen2.5-VL-7B in 4-bit and answer one MCQ image.

Run from project root:
    python -m src.inference
or equivalently:
    python src/inference.py

Expects the model to be pre-downloaded to weights/qwen2.5-vl-7b/ (see README).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "weights" / "qwen2.5-vl-7b"

SYSTEM_PROMPT = (
    "You are a deep learning expert. Solve the multiple-choice question shown in the image.\n"
    "Think step by step: identify what is being asked, work out the answer, then select the best option.\n\n"
    "After your reasoning, on a new line output exactly one digit:\n"
    "  1 for option A\n"
    "  2 for option B\n"
    "  3 for option C\n"
    "  4 for option D\n"
    "  5 if you cannot determine the answer.\n"
    "Output nothing else after that digit."
)


def load_model(model_dir: Path = MODEL_DIR):
    if not model_dir.exists():
        sys.exit(
            f"Model directory not found: {model_dir}\n"
            "Download first with:\n"
            '    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "weights/qwen2.5-vl-7b"'
        )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor


def answer_image(image_path: Path, model, processor) -> tuple[int, str]:
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=512, do_sample=False
        )
    generated = output_ids[:, inputs.input_ids.shape[1] :]
    decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_answer(decoded), decoded


def parse_answer(generation: str) -> int:
    """Iron-clad parser: return the last digit in [1,5] present in the output, else 5."""
    digits = re.findall(r"[1-5]", generation)
    return int(digits[-1]) if digits else 5


def main() -> None:
    model, processor = load_model()
    for name in ("image_1", "image_2"):
        img_path = PROJECT_ROOT / "images" / f"{name}.png"
        answer, full = answer_image(img_path, model, processor)
        print(f"=== {name} ===")
        print(full)
        print(f"--> parsed answer: {answer}\n")


if __name__ == "__main__":
    main()

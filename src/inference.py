"""Qwen2.5-VL inference helpers for GNR-638 Project 2.

Loads the model from local weights (no internet), builds VLM inputs, runs
greedy or self-consistent decoding, and parses answers into {1..5}.

Expects the model to be pre-downloaded to weights/qwen2.5-vl-7b/ (see README).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
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
    "\n"
    "Think step by step. First, describe what is being asked. Then work through the math or logic carefully, rejecting wrong options and explaining why the correct option is correct. Use full sentences for your reasoning.\n"
    "\n"
    "After your reasoning, on a new final line, output exactly one digit (a single digit, not a letter):\n"
    "  1 for option A\n"
    "  2 for option B\n"
    "  3 for option C\n"
    "  4 for option D\n"
    "  5 if you cannot determine the answer.\n"
    "\n"
    "Output nothing else after that digit."
)




def _detect_quantization(model_dir: Path) -> str:
    """Infer quant method from the model's config.json.

    Returns "awq" if the model is already quantized on disk (e.g. the 32B-AWQ
    checkpoint we use at contest eval), "bnb_nf4" otherwise (bf16 weights we
    apply bitsandbytes to at load time — the local 7B dev path).
    """
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with config_path.open(encoding="utf-8") as f:
                cfg = json.load(f)
            qc = cfg.get("quantization_config") or {}
            if "awq" in str(qc.get("quant_method", "")).lower():
                return "awq"
        except (json.JSONDecodeError, OSError):
            pass
    return "bnb_nf4"


def load_model(model_dir: Path = MODEL_DIR, quantization: str = "auto"):
    """Load Qwen2.5-VL from a local dir. Picks bnb-NF4 (dev) or AWQ (contest) by default."""
    if not model_dir.exists():
        sys.exit(
            f"Model directory not found: {model_dir}\n"
            "Download first with e.g.:\n"
            '    huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "weights/qwen2.5-vl-7b"\n'
            '    huggingface-cli download Qwen/Qwen2.5-VL-32B-Instruct-AWQ --local-dir "weights/qwen2.5-vl-32b-awq"'
        )
    if quantization == "auto":
        quantization = _detect_quantization(model_dir)

    kwargs = {"device_map": "auto"}
    if quantization == "bnb_nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "awq":
        pass
    elif quantization == "none":
        pass
    else:
        sys.exit(f"Unknown quantization: {quantization!r}. Use bnb_nf4, awq, none, or auto.")

    print(f"[load_model] dir={model_dir.name}  quantization={quantization}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_dir), **kwargs)
    processor = AutoProcessor.from_pretrained(str(model_dir))
    return model, processor


def _build_inputs(image_path: Path, processor, device):
    """Build VLM inputs for one image. Shared by greedy and self-consistent paths."""
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "image", "image": image}]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)


def answer_image(image_path: Path, model, processor) -> tuple[int, str]:
    """Greedy single-sample answer. Smoke test / debugging path."""
    inputs = _build_inputs(image_path, processor, model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=1024, do_sample=False
        )
    generated = output_ids[:, inputs.input_ids.shape[1] :]
    decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return parse_answer(decoded), decoded


def answer_image_sc(
    image_path: Path,
    model,
    processor,
    n_samples: int = 5,
    temperature: float = 0.7,
) -> tuple[int, float, list[int], list[str]]:
    """Self-consistent decoding: sample N times, majority-vote.

    Returns (majority_answer, confidence, per_sample_answers, per_sample_generations).
    confidence = votes_for_majority / n_samples. n_samples == 1 degrades to greedy
    so the N=1 path matches answer_image bit-for-bit.
    """
    inputs = _build_inputs(image_path, processor, model.device)
    prompt_len = inputs.input_ids.shape[1]

    generations: list[str] = []
    for _ in range(n_samples):
        with torch.no_grad():
            if n_samples == 1:
                output_ids = model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False
                )
            else:
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=temperature,
                )
        generated = output_ids[:, prompt_len:]
        generations.append(
            processor.batch_decode(generated, skip_special_tokens=True)[0]
        )

    answers = [parse_answer(g) for g in generations]
    majority, count = _majority_vote(answers)
    confidence = count / n_samples
    return majority, confidence, answers, generations




def _majority_vote(answers: list[int]) -> tuple[int, int]:
    """Majority vote with skip-averse tiebreak.

    Why: when the top count is tied between a skip (5) and an option (1-4),
    the option has EV = p - 0.25(1-p) > 0 for any p > 0.2. Since a tied
    option already has >= 40% vote support, that EV is solidly positive and
    strictly beats skip's 0. Ties between two options resolve in insertion
    order (deterministic for a given sample sequence).
    """
    counter = Counter(answers)
    top_count = counter.most_common(1)[0][1]
    tied = [a for a, c in counter.items() if c == top_count]
    non_skip = [a for a in tied if a != 5]
    winner = non_skip[0] if non_skip else tied[0]
    return winner, top_count





_LETTER_TO_DIGIT = {"A": 1, "B": 2, "C": 3, "D": 4}
_VALID_LAST_LINE = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, **_LETTER_TO_DIGIT}


def parse_answer(generation: str) -> int:
    """Parse the model's final answer. Always returns an int in {1..5}.

    Strategy (the *last non-empty line* is authoritative — avoids latching
    onto stray digits inside CoT reasoning):
      1. Strict: last line is a single digit/letter, optionally wrapped in
         markdown or punctuation — e.g. "1", "**B**", "B.", "C:".
      2. Loose: if the last line contains a digit 1-5, take the last one.
      3. Letter fallback: if the last line contains a letter A-D, map it.
      4. Give up → 5 (skip; scores 0, never -1).
    """
    lines = [ln.strip() for ln in generation.strip().splitlines() if ln.strip()]
    if not lines:
        return 5
    last = lines[-1]

    stripped = re.sub(r"[\*\.\s:,`\-]+", "", last).upper()
    if stripped in _VALID_LAST_LINE:
        return _VALID_LAST_LINE[stripped]

    digit_matches = re.findall(r"\b[1-5]\b", last)
    if digit_matches:
        return int(digit_matches[-1])

    letter_matches = re.findall(r"\b[ABCD]\b", last.upper())
    if letter_matches:
        return _LETTER_TO_DIGIT[letter_matches[-1]]

    return 5



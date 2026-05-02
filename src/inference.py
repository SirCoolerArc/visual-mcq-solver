"""Rung-1 sanity check: load Qwen2.5-VL-7B in 4-bit and answer one MCQ image.

Run from project root:
    python -m src.inference
or equivalently:
    python src/inference.py

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


# --- AWQ config patcher (kept commented out; the submission ships 7B-bnb only) -------
# We validated this on Kaggle: the Qwen2.5-VL-32B-Instruct-AWQ checkpoint stores the
# vision tower and lm_head as plain bf16 (`.weight` keys), but its config.json's
# `modules_to_not_convert` list doesn't include `visual` and `lm_head`. Without the
# patch, transformers' AwqQuantizer tries to AWQ-convert those layers, can't find
# the expected qweight/qzeros/scales, and leaves them randomly initialized. Model
# "loads" but emits garbage from the vision encoder and the final logits.
#
# If we ever revisit the 32B-AWQ path, restore this function and call it from the
# `quantization == "awq"` branch in load_model below.
#
# def _ensure_awq_config_patched(model_dir: Path) -> None:
#     config_path = model_dir / "config.json"
#     if not config_path.exists():
#         return
#     try:
#         with config_path.open(encoding="utf-8") as f:
#             cfg = json.load(f)
#     except (json.JSONDecodeError, OSError):
#         return
#     qc = cfg.get("quantization_config")
#     if not qc or "awq" not in str(qc.get("quant_method", "")).lower():
#         return
#     existing = list(qc.get("modules_to_not_convert") or [])
#     needed = [n for n in ("visual", "lm_head") if n not in existing]
#     if not needed:
#         return
#     qc["modules_to_not_convert"] = existing + needed
#     cfg["quantization_config"] = qc
#     try:
#         with config_path.open("w", encoding="utf-8") as f:
#             json.dump(cfg, f, indent=2)
#         print(f"[load_model] patched config.json: added {needed} to modules_to_not_convert")
#     except OSError as e:
#         print(f"[load_model] WARN: could not patch config.json ({e}); model output may be garbage")


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
        # 32B-AWQ path is not used by the current submission. If we revisit it,
        # uncomment the call below and the _ensure_awq_config_patched function above.
        # _ensure_awq_config_patched(model_dir)
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


def answer_image_sc_logprob(
    image_path: Path,
    model,
    processor,
    n_samples: int = 5,
    temperature: float = 0.7,
) -> tuple[int, float, int, float, list[int], list[float], list[str]]:
    """SC decoding that returns BOTH equal-weight and logprob-weighted aggregations.

    Generates N samples just like `answer_image_sc` but additionally captures
    the logprob of each sample's answer token. Computes both vote rules from
    the same generation set so they can be compared apples-to-apples on a
    single eval run.

    Returns:
        (eq_majority, eq_confidence,
         lp_majority, lp_confidence,
         per_sample_answers, per_sample_answer_logprobs, per_sample_generations)

    where eq_* are equal-weight (count-based) and lp_* are logprob-weighted.
    Both rules use the same skip-averse tiebreak.

    Experimental — not used by the production submission path. Keeps
    `answer_image_sc` byte-identical so we can A/B them without affecting the
    validated submission.
    """
    inputs = _build_inputs(image_path, processor, model.device)
    prompt_len = inputs.input_ids.shape[1]

    generations: list[str] = []
    logprobs: list[float] = []

    for _ in range(n_samples):
        with torch.no_grad():
            if n_samples == 1:
                output = model.generate(
                    **inputs, max_new_tokens=1024, do_sample=False,
                    output_scores=True, return_dict_in_generate=True,
                )
            else:
                output = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=temperature,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

        output_ids = output.sequences
        scores = output.scores  # tuple of (1, vocab) tensors, one per generated token
        generated = output_ids[:, prompt_len:]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
        generations.append(decoded)

        # Extract logprob of the parsed answer token from this sample.
        sample_answer = parse_answer(decoded)
        sample_lp = _extract_answer_token_logprob(
            output_ids, scores, prompt_len, processor, sample_answer
        )
        logprobs.append(sample_lp)

    answers = [parse_answer(g) for g in generations]

    # Equal-weight aggregation (matches answer_image_sc exactly).
    eq_majority, eq_count = _majority_vote(answers)
    eq_confidence = eq_count / n_samples

    # Logprob-weighted aggregation.
    lp_majority, lp_confidence = _logprob_weighted_vote(answers, logprobs)

    return (
        eq_majority, eq_confidence,
        lp_majority, lp_confidence,
        answers, logprobs, generations,
    )


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


def _logprob_weighted_vote(
    answers: list[int], logprobs: list[float]
) -> tuple[int, float]:
    """Logprob-weighted vote with the same skip-averse tiebreak as `_majority_vote`.

    Each sample's answer contributes `exp(logprob)` to its option's score. The
    option with the highest total wins. Returns (winner, normalized_confidence)
    where confidence = winner_score / sum_of_all_scores. Ties (within float
    epsilon) prefer non-skip answers for the same EV reason as the equal-weight
    rule.

    Skip handling: a `5` answer with high logprob still increases skip's tally,
    so the model can self-skip on confident "I don't know" outputs. The
    skip-averse tiebreak only kicks in when totals are numerically equal.
    """
    import math
    from collections import defaultdict

    scores: dict[int, float] = defaultdict(float)
    for ans, lp in zip(answers, logprobs):
        scores[ans] += math.exp(lp)

    if not scores:
        return 5, 0.0

    top_score = max(scores.values())
    tied = [a for a, s in scores.items() if abs(s - top_score) < 1e-9]
    non_skip = [a for a in tied if a != 5]
    winner = non_skip[0] if non_skip else tied[0]
    total = sum(scores.values())
    confidence = scores[winner] / total if total > 0 else 0.0
    return winner, confidence


def _extract_answer_token_logprob(
    output_ids,        # tensor (batch=1, seq_len) - full sequence including prompt
    scores,            # tuple of (1, vocab_size) tensors - one per generated token
    prompt_len: int,   # number of prompt tokens to skip
    processor,         # for decoding individual tokens
    parsed_answer: int,
) -> float:
    """Walk the generated tokens backwards and return the logprob of the last
    token matching the parsed answer.

    Matches a token if its decoded text (after stripping whitespace and common
    punctuation, case-normalized) equals the parsed digit ("1"-"5") or its
    letter equivalent ("A"-"D" for 1-4). Returns -10.0 (~4.5e-5 probability)
    if no matching token is found, which is a sensible "very low confidence"
    fallback for cases where the parser had to guess.
    """
    import torch.nn.functional as F

    new_tokens = output_ids[0, prompt_len:]
    target_chars: set[str] = {str(parsed_answer)}
    if 1 <= parsed_answer <= 4:
        target_chars.add("ABCD"[parsed_answer - 1])

    strip_re = re.compile(r"[\s\*\.\,\:\;`\-]+")
    for i in range(len(new_tokens) - 1, -1, -1):
        token_id = int(new_tokens[i].item())
        token_text = processor.tokenizer.decode([token_id], skip_special_tokens=True)
        cleaned = strip_re.sub("", token_text).upper()
        if cleaned in target_chars:
            logits = scores[i][0]
            log_probs = F.log_softmax(logits, dim=-1)
            return float(log_probs[token_id])

    return -10.0


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

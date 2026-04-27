# GNR 638 Project 2 — Visual MCQ Solver

Author: Rishabh Kumar (24b2419)
Partner: Dhruva Reddy

A pipeline that reads a PNG image of a deep-learning multiple-choice question
(question + four labelled options) and predicts the correct option as an
integer in `{1, 2, 3, 4, 5}`:

| Output | Meaning |
| --- | --- |
| `1` | option A |
| `2` | option B |
| `3` | option C |
| `4` | option D |
| `5` | abstain (skip) |

Output is guaranteed to be in `{1..5}` — the parser is iron-clad. No
hallucinated tokens reach the submission file.

## Quick start (TA)

```bash
# 1. (one-time, with internet) install dependencies
pip install -r requirements.txt

# 2. drop the test data alongside this submission, OR edit PARENT_DIR in the
#    notebook to point at it. Layout expected:
#      <parent>/images/*.png
#      <parent>/test.csv             (column: image_name)
#      <parent>/sample_submission.csv (columns: image_name, option)

# 3. run the notebook (offline; no network needed once weights are present)
jupyter nbconvert --to notebook --execute notebook.ipynb --output executed.ipynb

# 4. submission.csv is written next to the test data (same dir as test.csv).
```

If `PARENT_DIR` in the notebook is set to `.` (the default), the notebook
assumes the test data sits next to itself. Adjust the variable in the
configuration cell if your test data is elsewhere.

## Hardware requirements

- 1× NVIDIA GPU with **≥ 20 GB VRAM** (eval target: L40s, 48 GB). The
  bundled configuration loads the model at native bf16 precision — about
  15 GB for the weights plus a few GB for KV cache and activations.
- ~16 GB free disk space for the bundled 7B weights.

If running on a smaller GPU, see "Troubleshooting" in the notebook for
the one-flag downgrade to NF4 4-bit (~5 GB VRAM, small accuracy cost).

The pipeline runs **offline** at inference time — internet is required only
for the one-time `pip install -r requirements.txt` step.

## Submission layout

```
submission/
├── README.md                # this file
├── requirements.txt         # pinned dependencies
├── notebook.ipynb           # offline inference entrypoint
├── src/
│   ├── __init__.py
│   ├── inference.py         # model loading + self-consistent decoding
│   └── submit.py            # batch inference + submission.csv writer
└── weights/
    └── qwen2.5-vl-7b/       # bf16 weights, ~16 GB
```

## Pipeline

```text
PNG image
   ↓
Qwen2.5-VL-7B-Instruct  (loaded from weights/qwen2.5-vl-7b/ at native bf16,
                         ~15 GB VRAM on the eval L40s)
   ↓
System prompt: chain-of-thought reasoning, final line = digit only
   ↓
Self-consistency: N=7 samples at temperature 0.7
   ↓
Iron-clad parser: each sample → answer ∈ {1, 2, 3, 4, 5}
   ↓
Majority vote (skip-averse tiebreak: option votes win 50/50 ties against skip)
   ↓
submission.csv (image_name, option)
```

**Decisions worth noting**

- **Model choice (7B not 32B)**: the 32B AWQ checkpoint has a higher
  accuracy ceiling, but its loader has a known bug requiring an on-disk
  `config.json` patch that we couldn't validate end-to-end with actual
  inference on the GPU classes we had access to (Kaggle T4×2 hit a separate
  AutoAWQ-Triton-kernel issue, and there's no L40s available outside the
  eval). Shipping a stack we have never produced a single output token from
  would be reckless on a graded submission, especially given the 1 hr
  runtime cap which 32B may not reliably fit. The 7B path is fast,
  end-to-end validated, and has comfortable margin on every dimension.

- **bf16 precision (not NF4 4-bit) on the eval L40s.** Local development
  used `bitsandbytes` NF4 4-bit so the model could fit on an 8 GB consumer
  GPU. The L40s has 48 GB and no such constraint, so the submitted notebook
  loads the model at native bf16 — the precision the model was trained at.
  NF4 → bf16 is a strict precision upgrade, never a regression on average.
  The fallback to NF4 is one CLI flag away if the submission is ever
  re-targeted at a smaller GPU.

- **N = 7 self-consistency samples**. Each additional sample makes the
  majority vote more robust to single-sample reasoning slips. The Wang
  et al. self-consistency paper shows diminishing returns past ~5–11
  samples; N = 7 is the cheap-but-real bump that fits comfortably under
  the runtime cap on L40s.

- **τ = 0** (no confidence threshold). Empirically the 7B confidence
  distribution is too noisy for a positive τ to net points — raising τ
  above 0 cost more in skipped-correct than it saved in skipped-wrong.
  We rely on the vote itself to produce skip outputs (the model emits `5`
  when its samples disagree).

- **Skip-averse tiebreak**: when the top vote count is tied between an
  option and skip, we pick the option. EV math: a tied option has ≥ 40%
  sample support, which clears the +0.20 break-even threshold for
  guess-vs-skip.

- **No RAG, no OCR**: the VLM handles MCQ images natively. RAG would add
  context-pollution risk; OCR corrupts code-block indentation and
  subscripted math.

## Smoke test

The first thing `submit.py` does after loading the model is run a single
greedy generation on one test image and verify the parsed answer is in
`{1..5}`. If the smoke test fails (load error or out-of-range output) the
script errors out before processing all 50 questions. This guards against
catastrophic-failure modes (the kind that silently writes a useless
`submission.csv`); failing fast is better than scoring zero.

## Expected runtime

| Phase | Time on L40s |
| --- | --- |
| Model load (one-time) | ~30–45 s |
| Per question (N=7 SC, bf16) | ~25–40 s |
| Total for 50 questions | ~25–35 min |

Well inside the 1 hour cap specified in the project brief. Estimates
extrapolate from local 4060 Laptop measurements scaled by the L40s'
~3–4× memory-bandwidth advantage and the absence of dequantization
overhead at bf16.

## Scoring

Per the contest rubric: `score = #correct − 0.25·#wrong − 1·#hallucinated`.
The iron-clad parser ensures `#hallucinated = 0`; the threshold-free skip
mechanism keeps `#wrong` modest by emitting `5` only when the model's own
samples disagree. The pipeline was developed against an internally authored
set of practice MCQs; final grading is on the held-out contest test set.

## References

The implementation builds on the following open-source projects and papers:

- **Qwen2.5-VL** — Bai et al., 2025. <https://github.com/QwenLM/Qwen2.5-VL> (Apache 2.0)
- **Self-Consistency Improves Chain of Thought Reasoning in Language Models** — Wang et al., 2022. <https://arxiv.org/abs/2203.11171>
- **bitsandbytes (NF4 quantization)** — Dettmers et al., 2023. <https://github.com/bitsandbytes-foundation/bitsandbytes>
- **Hugging Face Transformers** — Wolf et al., 2020. <https://github.com/huggingface/transformers>

## Contact

Issues building or running this submission — Rishabh Kumar
(`24b2419@iitb.ac.in`).

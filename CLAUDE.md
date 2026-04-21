# GNR 638 Project 2 — Deep Learning Visual MCQ Solver

Session context anchor. Read this before doing any work on this project so we don't re-derive decisions from scratch each session.

## What we're building

A pipeline that takes a PNG image of a deep learning MCQ (question + four options, exactly one correct) and outputs an integer:

- `1–4` → selected option (A/B/C/D)
- `5` → skip / unanswered
- Anything else → treated as hallucinated (−1 penalty)

Final output: `submission.csv` with columns `id, image_name, option`, matching `test.csv` row-for-row.

**The real test set is hidden.** We only have 2 sample images (see [images/](images/)). At evaluation time the `images/` folder is swapped with an unseen set. The pipeline must generalize — don't hardcode to the samples.

## Hard constraints (from [COMPETITION_BRIEF.md](COMPETITION_BRIEF.md))

- **Deadline**: 2 May 2026 23:59 UTC (3 May 05:30 IST). No late days.
- **Eval compute**: single 48 GB L40s GPU.
- **Runtime**: ≤ 1 hour for ≤ 50 questions.
- **Offline at eval**: no internet on the submitted notebook. All weights, tokenizers, helper assets must be pre-downloaded during the setup phase (where internet *is* allowed) and loaded from local paths.
- **Packaging**: Jupyter notebook or `.py` file + weights + `environment.yml` or `requirements.txt` + `README.md`. **Missing/broken README → direct 0, no TA communication.** Take README seriously.
- **Cite sources** used in the final notebook.
- **Dev compute (ours)**: one RTX 4060 Laptop, 8 GB VRAM + 16 GB system RAM. No other GPU.

## Scoring math

`score = #correct − 0.25 × #wrong − 1 × #hallucinated`

- Break-even for guessing: `p ≥ 0.20`. If confidence > ~20 %, guess; else skip.
- Hallucination penalty dominates. The output parser must be iron-clad: anything not strictly in `{1, 2, 3, 4, 5}` is coerced to `5`.
- Expected perfect score is 50. Margin between winning and losing is likely 2–3 questions — avoiding hallucinations and skipping strategically are as important as accuracy.

## Pipeline architecture (locked)

```
PNG image
  ↓
Qwen2.5-VL-32B-Instruct-AWQ  (4-bit, ~18 GB on L40s)
  ↓
Prompt: system (CoT + strict output contract) + image as user turn
  ↓
Self-consistency: N samples at temperature ~0.7  (default N = 5)
  ↓
Parse each sample's final answer digit → majority vote
  ↓
Confidence = (majority count) / N
  ↓
If confidence ≥ τ  → emit majority answer (1/2/3/4)
Else                → emit 5 (skip)
  ↓
Iron-clad regex parser: final output ∈ {1,2,3,4,5}, else 5
  ↓
submission.csv
```

**No RAG, no OCR in v1.** Sample questions are fundamental DL concepts a 32B VLM already knows; RAG adds distraction risk; OCR corrupts code-block indentation and subscripted math. VLM-direct avoids both failure modes.

### Prompt (baseline)

System:

```
You are a deep learning expert. Solve the multiple-choice question shown in the image.
Think step by step: identify what is being asked, work out the answer, then select
the best option.

After your reasoning, on a new line output exactly one digit:
  1 for option A
  2 for option B
  3 for option C
  4 for option D
  5 if you cannot determine the answer.
Output nothing else after that digit.
```

User turn: image only.

Answer extraction: take the last `[1-5]` match in the generation. If none, emit `5`.

## Models

| Role                            | Model                              | Size       | Where                |
|---------------------------------|------------------------------------|-----------:|----------------------|
| **Submission (primary)**        | Qwen2.5-VL-32B-Instruct-AWQ        | ~18 GB     | L40s 48 GB (eval)    |
| **Local dev iteration**         | Qwen2.5-VL-7B-Instruct-AWQ         | ~5–6 GB    | RTX 4060 8 GB        |
| **Fallback (if 32B blocks)**    | Qwen2.5-VL-14B-AWQ                 | ~8 GB      | Kaggle / L40s        |

**Why Qwen2.5-VL**: strong VL reasoning on MCQ-style benchmarks, mature AWQ quantization, known-good on code-in-image and math notation.

**Not chosen: 72B.** Can't validate anywhere we have access to (even Kaggle free / Colab Pro won't fit 72B AWQ comfortably), runtime margin tight, marginal accuracy gain uncertain given self-consistency already recovers much of the gap.

## Development ladder

Three environments, promote upward, do not skip rungs:

1. **Local (RTX 4060 Laptop, 8 GB)** — Qwen2.5-VL-7B-AWQ. ~90 % of dev happens here: prompt engineering, parser shakedown, practice-set evaluation loop, self-consistency logic, offline-mode verification.
2. **Kaggle free GPU (P100/T4, 16 GB) or Colab Pro (L4, 24 GB)** — Qwen2.5-VL-32B-AWQ. **Mandatory**: one end-to-end pass on the practice set at 32B scale before submission. **This is where we calibrate τ** — confidence distributions differ across model sizes, so τ tuned on 7B will be miscalibrated at 32B.
3. **Submission (L40s 48 GB)** — Qwen2.5-VL-32B-AWQ, identical prompt and params to step 2.

## Practice set — evaluation & calibration prerequisite

We cannot (a) validate the pipeline end-to-end, (b) calibrate τ, (c) detect regressions, or (d) decide whether the OCR fallback is worth shipping, without a practice set. The pipeline skeleton and inference loop can be built *in parallel* with practice-set authoring, but no meaningful measurement happens until the practice set exists.

- **Size**: 50–100 MCQs in the same visual format as the samples (LaTeX-rendered PNG, 4 options).
- **Source**: course slides, tutorials, past quizzes (user has all PDFs).
- **Format diversity**: include questions spanning the modalities visible in the samples (prose + LaTeX math, code-block understanding, shape/size computation) so the pipeline is exercised against each failure mode. The actual distribution in the hidden test set is unknown — don't over-fit our practice set to a guessed mix.
- **Ground truth**: we author the questions, so we know the answers.
- **Gate**: must exist and be visually-validated (renders look like the samples) before any serious prompt tuning or τ calibration.

## Runtime budget

32B-AWQ on L40s ≈ 30–50 tok/s. Per question: ~300-token CoT × 5 samples ≈ 1 500 tokens ≈ 30–50 s. For 50 questions: ~25–42 min. Under the 1 hr cap with headroom.

If budget slips: reduce `N` from 5 → 3 (≈ 40 % time saved). Raise to 7 only if eval shows accuracy ceiling is from CoT variance, not model capability.

## Open / deferred decisions (resolved with data, not a priori)

1. **τ (skip threshold)** — calibrated on practice set at **32B scale** (step 2 of the ladder). Expected range 3/5 (0.6) to 4/5 (0.8). Objective: maximize expected practice-set score = `#correct − 0.25 × #wrong`.
2. **N (self-consistency samples)** — default 5. Drop to 3 if runtime tight. Raise to 7 only with headroom.
3. **OCR fallback path** — build and ship *only* if practice-set eval shows VLM consistently failing on code-heavy questions. If shipped: Nougat OCR → DeepSeek-R1-Distill-Qwen-32B 4-bit with CoT → ensemble-vote with VLM path. YAGNI until data demands it.
4. **Model downshift (14B / 7B as final)** — only if Kaggle/Colab 32B validation is blocked. Last resort.

## Submission package layout

```
submission/
├── notebook.ipynb          # offline inference entry point
├── environment.yml         # or requirements.txt
├── README.md               # clear setup instructions (this is graded!)
├── weights/
│   └── qwen2.5-vl-32b-awq/ # pre-downloaded; loaded from local path only
└── src/                    # any helpers imported by the notebook
```

Input structure at eval (parent directory path passed at runtime):

```
parent_dir/
├── images/
│   ├── image_1.png
│   └── ...
├── test.csv
└── sample_submission.csv
```

The notebook must write `submission.csv` with columns `id, image_name, option` exactly matching `sample_submission.csv` structure.

## Session kickoff checklist

When starting a new Claude session on this project:

1. This file is auto-loaded — context restored, do not re-ask what the project is.
2. Check project status: does the practice set exist? Has τ been calibrated? Has the 32B validation pass been run?
3. Don't tune τ on 7B — calibration happens at 32B scale only.
4. Before claiming done: full pipeline run on practice set → verify offline mode (disconnect internet, run) → verify `submission.csv` columns match `sample_submission.csv` exactly → check total runtime < 1 hr.
5. Don't break the iron-clad parser invariant: final output always ∈ `{1,2,3,4,5}`.

## References to cite in the final notebook

- Qwen2.5-VL — https://github.com/QwenLM/Qwen2.5-VL (Apache 2.0)
- AWQ — https://github.com/mit-han-lab/llm-awq
- Self-consistency decoding — Wang et al. 2022, https://arxiv.org/abs/2203.11171

(Add others — e.g. Nougat, DeepSeek-R1-Distill — only if they end up shipped.)

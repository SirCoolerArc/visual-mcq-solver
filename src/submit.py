"""Offline MCQ inference pipeline -> submission.csv. Single source of truth for
both local-calibration and contest-eval paths.

At contest eval, the TAs point us at a parent directory containing:
    <parent>/images/*.png
    <parent>/test.csv                 # column: image_name
    <parent>/sample_submission.csv    # columns: image_name, option  (schema reference)

This script reads test.csv, runs self-consistent decoding on each image, and
writes submission.csv with columns [image_name, option]. Option is guaranteed
to be in {1, 2, 3, 4, 5} - the iron-clad invariant is enforced at write time.

If <parent>/ground_truth.csv exists (our practice set), it additionally scores
the run and writes eval_log.csv.

Run:
    # local calibration against our practice set
    python -m src.submit --parent-dir practice_set

    # Kaggle/contest (32B AWQ weights mounted as a dataset)
    python -m src.submit \\
        --parent-dir /kaggle/input/gnr638-test \\
        --model-dir /kaggle/input/qwen25-vl-32b-awq \\
        --output /kaggle/working/submission.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import answer_image, answer_image_sc, load_model  # noqa: E402

VALID_ANSWERS = {1, 2, 3, 4, 5}
OPTION_ANSWERS = {1, 2, 3, 4}


def score_one(predicted: int, truth: int) -> float:
    if predicted not in VALID_ANSWERS:
        return -1.0
    if predicted == 5:
        return 0.0
    return 1.0 if predicted == truth else -0.25


def _free_gpu_memory(model) -> None:
    """Best-effort release of a loaded model's GPU memory before reloading a fallback."""
    try:
        del model
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _load_with_fallback(primary_dir, fallback_dir, smoke_image, quantization):
    """Load primary model + smoke-test on one image. On any failure, try fallback.

    Smoke test: greedy single-sample on smoke_image, assert parsed answer in {1..5}.
    Catches both load-time errors (exceptions in from_pretrained) and runtime errors
    (exceptions or garbage output from the first forward pass). The fallback is the
    safety net for catastrophic surprises on the contest L40s.
    """
    primary_kwargs = {"quantization": quantization}
    if primary_dir is not None:
        primary_kwargs["model_dir"] = primary_dir

    def _try(label, kwargs):
        t0 = time.time()
        print(f"[submit] loading {label} from {kwargs.get('model_dir', 'default')} ...")
        model, processor = load_model(**kwargs)
        print(f"[submit] {label} loaded in {time.time() - t0:.1f}s")
        # Smoke test
        print(f"[submit] {label} smoke-test on {smoke_image.name} ...")
        ans, _ = answer_image(smoke_image, model, processor)
        if ans not in VALID_ANSWERS:
            raise RuntimeError(f"{label} smoke test produced out-of-range answer: {ans}")
        print(f"[submit] {label} smoke-test OK (got {ans})")
        return model, processor

    try:
        return _try("primary", primary_kwargs)
    except Exception as e:
        print(f"[submit] !! primary model failed: {type(e).__name__}: {e}")
        if fallback_dir is None:
            raise
        print(f"[submit] falling back to {fallback_dir}")
        # If primary partially loaded, reclaim VRAM before trying fallback.
        # (best-effort - ignored on first-token errors)
        try:
            _free_gpu_memory(locals().get("model"))
        except Exception:
            pass
        fallback_kwargs = {"quantization": "auto", "model_dir": fallback_dir}
        return _try("fallback", fallback_kwargs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parent-dir", type=Path, required=True,
                   help="Dir containing images/ and test.csv")
    p.add_argument("--model-dir", type=Path, default=None,
                   help="Primary VLM weights dir (defaults to weights/qwen2.5-vl-7b)")
    p.add_argument("--fallback-model-dir", type=Path, default=None, dest="fallback_model_dir",
                   help="Fallback weights dir, used if primary load or smoke test fails")
    p.add_argument("--output", type=Path, default=None,
                   help="submission.csv path (default: <parent>/submission.csv)")
    p.add_argument("--eval-log", type=Path, default=None, dest="eval_log",
                   help="eval_log.csv path when ground_truth is present (default: alongside submission)")
    p.add_argument("--n-samples", type=int, default=5, dest="n_samples",
                   help="Self-consistency samples per question")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--tau", type=float, default=0.0,
                   help="Skip threshold: emit 5 if confidence < tau")
    p.add_argument("--quantization", default="auto",
                   choices=["auto", "bnb_nf4", "awq", "none"])
    args = p.parse_args()

    parent = args.parent_dir.resolve()
    images_dir = parent / "images"
    test_csv = parent / "test.csv"
    gt_csv = parent / "ground_truth.csv"
    output = args.output or (parent / "submission.csv")

    if not images_dir.is_dir():
        sys.exit(f"No images dir: {images_dir}")
    if not test_csv.exists():
        sys.exit(f"No test.csv: {test_csv}")

    with test_csv.open(encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))
    if not test_rows or "image_name" not in test_rows[0]:
        sys.exit(f"test.csv must have an 'image_name' column. Got: {list(test_rows[0].keys()) if test_rows else 'empty file'}")

    # Optional scoring path (practice set).
    truth_by_name: dict[str, int] = {}
    meta_by_name: dict[str, dict] = {}
    if gt_csv.exists():
        with gt_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                truth_by_name[r["image_name"]] = int(r["correct_option"])
                meta_by_name[r["image_name"]] = r
        print(f"[submit] ground_truth.csv detected - will score ({len(truth_by_name)} entries)")
    else:
        print("[submit] no ground_truth.csv - submission-only mode")

    print(f"[submit] parent={parent}")
    print(f"[submit] n_samples={args.n_samples}  temperature={args.temperature}  tau={args.tau}")
    print(f"[submit] test questions: {len(test_rows)}")

    # Pick the smoke-test image: first image_name in test.csv that exists on disk.
    smoke_image = None
    for r in test_rows:
        candidate = images_dir / f"{r['image_name']}.png"
        if candidate.exists():
            smoke_image = candidate
            break
    if smoke_image is None:
        sys.exit(f"No images found under {images_dir} for any name in test.csv")

    model, processor = _load_with_fallback(
        primary_dir=args.model_dir,
        fallback_dir=args.fallback_model_dir,
        smoke_image=smoke_image,
        quantization=args.quantization,
    )
    print()

    sub_rows: list[dict] = []
    eval_rows: list[dict] = []
    total_score = 0.0
    run_start = time.time()

    for i, row in enumerate(test_rows, 1):
        image_name = row["image_name"]
        img_path = images_dir / f"{image_name}.png"

        if not img_path.exists():
            # Defensive: don't crash the whole run on a missing image. Emit skip.
            print(f"[{i:>3}/{len(test_rows)}] {image_name} MISSING image -> emit 5")
            sub_rows.append({"image_name": image_name, "option": 5})
            continue

        t_q = time.time()
        predicted, confidence, sample_answers, generations = answer_image_sc(
            img_path, model, processor,
            n_samples=args.n_samples, temperature=args.temperature,
        )
        elapsed = time.time() - t_q

        # Apply tau skip threshold - only coerces options to skip, never the reverse.
        if predicted in OPTION_ANSWERS and confidence < args.tau:
            predicted = 5

        # Iron-clad final invariant: anything weird -> 5.
        if predicted not in VALID_ANSWERS:
            predicted = 5

        sub_rows.append({"image_name": image_name, "option": predicted})

        if image_name in truth_by_name:
            truth = truth_by_name[image_name]
            s = score_one(predicted, truth)
            total_score += s
            meta = meta_by_name[image_name]
            eval_rows.append({
                "image_name": image_name,
                "topic": meta.get("topic", ""),
                "subtopic": meta.get("subtopic", ""),
                "difficulty": meta.get("difficulty", ""),
                "truth": truth,
                "predicted": predicted,
                "is_correct": predicted == truth,
                "score": s,
                "confidence": round(confidence, 4),
                "n_samples": args.n_samples,
                "sample_answers": ",".join(str(a) for a in sample_answers),
                "elapsed_sec": round(elapsed, 2),
                "generation": generations[0],
            })
            print(
                f"[{i:>3}/{len(test_rows)}] {image_name:<38} "
                f"truth={truth} pred={predicted} conf={confidence:.2f} "
                f"score={s:+.2f} ({elapsed:5.1f}s)"
            )
        else:
            print(
                f"[{i:>3}/{len(test_rows)}] {image_name:<38} "
                f"pred={predicted} conf={confidence:.2f} ({elapsed:5.1f}s)"
            )

    run_elapsed = time.time() - run_start

    # Iron-clad invariant check BEFORE writing.
    bad = [r for r in sub_rows if r["option"] not in VALID_ANSWERS]
    if bad:
        print(f"\n!! {len(bad)} predictions outside {{1..5}} - invariant broken")
        for r in bad[:5]:
            print(f"   {r['image_name']} -> {r['option']}")
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "option"])
        w.writeheader()
        w.writerows(sub_rows)
    print(f"\n[ok] wrote {len(sub_rows)} predictions to {output}")

    if eval_rows:
        n = len(eval_rows)
        correct = sum(1 for r in eval_rows if r["is_correct"])
        wrong = sum(1 for r in eval_rows if r["predicted"] in OPTION_ANSWERS and not r["is_correct"])
        skipped = sum(1 for r in eval_rows if r["predicted"] == 5)
        print("\n" + "=" * 60)
        print(f"[summary] n={n}   wall={run_elapsed:.1f}s   avg={run_elapsed/n:.1f}s/q")
        print(f"  correct      : {correct:>3}  (+{correct:.2f})")
        print(f"  wrong        : {wrong:>3}  ({-0.25 * wrong:+.2f})")
        print(f"  skipped      : {skipped:>3}  (+0.00)")
        print(f"  total score  : {total_score:+.2f} / {n}")
        print(f"  raw accuracy : {correct / n:.1%}")

        # Confidence + tau sweep, same shape as src/eval.py.
        if args.n_samples > 1:
            confs_correct = [r["confidence"] for r in eval_rows if r["is_correct"]]
            confs_wrong = [r["confidence"] for r in eval_rows if r["predicted"] in OPTION_ANSWERS and not r["is_correct"]]
            def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
            print("\n[confidence]")
            print(f"  mean correct : {mean(confs_correct):.3f}  (n={len(confs_correct)})")
            print(f"  mean wrong   : {mean(confs_wrong):.3f}  (n={len(confs_wrong)})")

            print("\n[tau sweep - score if skip when confidence < tau]")
            print(f"  {'tau':>6} {'answered':>9} {'correct':>8} {'wrong':>6} {'skip':>5} {'score':>7}")
            for tau in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
                a = c = w_ = s_ = 0
                sc = 0.0
                for r in eval_rows:
                    if r["confidence"] < tau:
                        s_ += 1; continue
                    a += 1
                    if r["is_correct"]:
                        c += 1; sc += 1.0
                    elif r["predicted"] in OPTION_ANSWERS:
                        w_ += 1; sc -= 0.25
                print(f"  {tau:>6.2f} {a:>9} {c:>8} {w_:>6} {s_:>5} {sc:>7.2f}")

        eval_log = args.eval_log or (output.parent / "eval_log.csv")
        eval_log.parent.mkdir(parents=True, exist_ok=True)
        with eval_log.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
            w.writeheader()
            w.writerows(eval_rows)
        print(f"[ok] wrote eval_log to {eval_log.name}")


if __name__ == "__main__":
    main()

"""Score the practice set end-to-end and write a per-question log.

Loops over every image in practice_set/images/, runs the inference pipeline,
scores the parsed answer against ground_truth.csv using the contest rubric
(+1 correct, 0 skip, -0.25 wrong, -1 hallucinated), and prints an aggregate
scorecard + breakdowns by topic / difficulty. Also writes a per-question
CSV log (eval_log.csv) for offline failure analysis.

Run from project root:
    python -m src.eval                     # default: N=5 self-consistency
    python -m src.eval --n-samples 1       # greedy single-sample baseline
    python -m src.eval --limit 5           # first 5 only
    python -m src.eval --only cnn_shape_01 # single question
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Allow `python src/eval.py` and `python -m src.eval` to both work.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import answer_image_sc, load_model  # noqa: E402

IMAGES_DIR = PROJECT_ROOT / "practice_set" / "images"
GROUND_TRUTH_CSV = PROJECT_ROOT / "practice_set" / "ground_truth.csv"
EVAL_LOG_CSV = PROJECT_ROOT / "practice_set" / "eval_log.csv"

VALID_ANSWERS = {1, 2, 3, 4, 5}
OPTION_ANSWERS = {1, 2, 3, 4}


def load_ground_truth() -> list[dict]:
    with GROUND_TRUTH_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_one(predicted: int, truth: int) -> float:
    """Contest rubric: +1 correct, 0 skip, -0.25 wrong option, -1 hallucinated."""
    if predicted not in VALID_ANSWERS:
        return -1.0
    if predicted == 5:
        return 0.0
    return 1.0 if predicted == truth else -0.25


def confidence_report(rows: list[dict]) -> None:
    """Confidence diagnostics + tau-sweep preview.

    Shows mean confidence among correct vs wrong answers (good calibration =
    correct >> wrong), a distribution of questions by confidence, and the
    contest score that each candidate tau would yield if applied as a skip
    threshold. This is the pre-calibration table — no tau is applied yet.
    """
    confs = [r["confidence"] for r in rows]
    correct_confs = [r["confidence"] for r in rows if r["is_correct"]]
    wrong_confs = [
        r["confidence"] for r in rows
        if r["predicted"] in OPTION_ANSWERS and not r["is_correct"]
    ]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n[confidence]")
    print(f"  mean overall : {mean(confs):.3f}")
    print(f"  mean correct : {mean(correct_confs):.3f}  (n={len(correct_confs)})")
    print(f"  mean wrong   : {mean(wrong_confs):.3f}  (n={len(wrong_confs)})")

    print("\n[tau sweep - score if we skip (emit 5) when confidence < tau]")
    print(f"  {'tau':>6} {'answered':>9} {'correct':>8} {'wrong':>6} {'skip':>5} {'score':>7}")
    for tau in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        answered = correct = wrong = skipped = 0
        score = 0.0
        for r in rows:
            if r["confidence"] < tau:
                skipped += 1
                continue
            answered += 1
            if r["is_correct"]:
                correct += 1
                score += 1.0
            elif r["predicted"] in OPTION_ANSWERS:
                wrong += 1
                score -= 0.25
        marker = ""
        print(f"  {tau:>6.2f} {answered:>9} {correct:>8} {wrong:>6} {skipped:>5} {score:>7.2f}{marker}")


def bucket_report(rows: list[dict], key: str) -> None:
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        buckets.setdefault(r[key], []).append(r)

    print(f"\n[by {key}]")
    print(f"  {'bucket':<22} {'n':>3} {'correct':>7} {'wrong':>5} {'skip':>4} {'hall':>4} {'score':>7}")
    for bucket, rs in sorted(buckets.items()):
        n = len(rs)
        correct = sum(1 for r in rs if r["is_correct"])
        wrong = sum(1 for r in rs if r["predicted"] in OPTION_ANSWERS and not r["is_correct"])
        skip = sum(1 for r in rs if r["predicted"] == 5)
        halluc = sum(1 for r in rs if r["predicted"] not in VALID_ANSWERS)
        total = sum(r["score"] for r in rs)
        print(f"  {bucket:<22} {n:>3} {correct:>7} {wrong:>5} {skip:>4} {halluc:>4} {total:>7.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Run only this image_name")
    parser.add_argument("--limit", type=int, help="Stop after this many questions")
    parser.add_argument("--n-samples", type=int, default=5, dest="n_samples",
                        help="Self-consistency samples per question (1 = greedy)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature when n-samples > 1")
    args = parser.parse_args()

    gt = load_ground_truth()
    if args.only:
        gt = [r for r in gt if r["image_name"] == args.only]
        if not gt:
            sys.exit(f"No ground-truth row with image_name '{args.only}'.")
    if args.limit:
        gt = gt[: args.limit]

    print(f"[eval] n_samples={args.n_samples} temperature={args.temperature}")
    print("[eval] loading model ...")
    t0 = time.time()
    model, processor = load_model()
    print(f"[eval] model loaded in {time.time() - t0:.1f}s\n")

    rows: list[dict] = []
    total_score = 0.0
    run_start = time.time()

    for i, gt_row in enumerate(gt, 1):
        img_name = gt_row["image_name"]
        truth = int(gt_row["correct_option"])
        img_path = IMAGES_DIR / f"{img_name}.png"
        if not img_path.exists():
            sys.exit(f"Missing image: {img_path}")

        t_start = time.time()
        predicted, confidence, sample_answers, generations = answer_image_sc(
            img_path, model, processor,
            n_samples=args.n_samples, temperature=args.temperature,
        )
        elapsed = time.time() - t_start

        score = score_one(predicted, truth)
        is_correct = predicted == truth
        total_score += score

        print(
            f"[{i:>2}/{len(gt)}] {img_name:<38} "
            f"truth={truth} pred={predicted} conf={confidence:.2f} "
            f"score={score:+.2f} ({elapsed:5.1f}s)"
        )

        rows.append(
            {
                "image_name": img_name,
                "topic": gt_row["topic"],
                "subtopic": gt_row["subtopic"],
                "difficulty": gt_row["difficulty"],
                "source": gt_row["source"],
                "truth": truth,
                "predicted": predicted,
                "is_correct": is_correct,
                "score": score,
                "confidence": round(confidence, 4),
                "n_samples": args.n_samples,
                "sample_answers": ",".join(str(a) for a in sample_answers),
                "elapsed_sec": round(elapsed, 2),
                "generation": generations[0],
            }
        )

    run_elapsed = time.time() - run_start
    n = len(rows)
    correct = sum(1 for r in rows if r["is_correct"])
    wrong = sum(1 for r in rows if r["predicted"] in OPTION_ANSWERS and not r["is_correct"])
    skipped = sum(1 for r in rows if r["predicted"] == 5)
    halluc = sum(1 for r in rows if r["predicted"] not in VALID_ANSWERS)
    attempted = n - skipped

    print("\n" + "=" * 60)
    print(f"[summary] n={n}   wall={run_elapsed:.1f}s   avg={run_elapsed/max(n,1):.1f}s/q")
    print(f"  correct      : {correct:>3}  (+{float(correct):.2f})")
    print(f"  wrong        : {wrong:>3}  ({-0.25 * wrong:+.2f})")
    print(f"  skipped      : {skipped:>3}  (+0.00)")
    print(f"  hallucinated : {halluc:>3}  ({-float(halluc):+.2f})")
    print(f"  total score  : {total_score:+.2f} / {n}")
    print(f"  raw accuracy : {correct / n:.1%}  (correct / n)")
    if attempted:
        print(f"  attempted acc: {correct / attempted:.1%}  (correct / attempted)")
    if halluc:
        print("  !! parser emitted out-of-range answers — iron-clad invariant violated")

    if args.n_samples > 1:
        confidence_report(rows)

    bucket_report(rows, "topic")
    bucket_report(rows, "difficulty")

    # Only overwrite the full log when running the full practice set.
    if not args.only and not args.limit:
        with EVAL_LOG_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[ok] wrote per-question log to {EVAL_LOG_CSV.name}")


if __name__ == "__main__":
    main()

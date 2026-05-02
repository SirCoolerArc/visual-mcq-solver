"""Logprob-weighted vote experiment.

Runs N=5 self-consistency on the practice set with answer-token logprobs
captured per sample, then computes BOTH equal-weight (count-based) and
logprob-weighted (exp-logprob-weighted) aggregations from the same
generations. Reports both scores side-by-side so we can pick the winner
for the actual submission.

Usage (from project root):
    python -m experiments.logprob --limit 5      # smoke test on first 5 questions
    python -m experiments.logprob                # full 50-question eval
    python -m experiments.logprob --only cnn_shape_01

Writes practice_set/logprob_eval_log.csv on full runs.
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

from src.inference import answer_image_sc_logprob, load_model  # noqa: E402

PRACTICE_ROOT = PROJECT_ROOT / "practice_set"
VALID = {1, 2, 3, 4, 5}
OPTIONS = {1, 2, 3, 4}


def score_one(predicted: int, truth: int) -> float:
    if predicted not in VALID:
        return -1.0
    if predicted == 5:
        return 0.0
    return 1.0 if predicted == truth else -0.25


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--only", help="Run only this image_name")
    p.add_argument("--n-samples", type=int, default=5, dest="n_samples")
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    with (PRACTICE_ROOT / "ground_truth.csv").open(encoding="utf-8") as f:
        gt = list(csv.DictReader(f))
    if args.only:
        gt = [r for r in gt if r["image_name"] == args.only]
        if not gt:
            sys.exit(f"No question {args.only!r} in ground_truth.csv")
    if args.limit:
        gt = gt[: args.limit]

    print(f"[logprob] evaluating on {len(gt)} questions")
    print(f"[logprob] n_samples={args.n_samples}  temperature={args.temperature}\n")

    print("[logprob] loading model ...")
    t0 = time.time()
    model, processor = load_model()
    print(f"[logprob] model loaded in {time.time() - t0:.1f}s\n")

    rows: list[dict] = []
    eq_total = 0.0
    lp_total = 0.0
    run_start = time.time()

    for i, gtr in enumerate(gt, 1):
        img_name = gtr["image_name"]
        truth = int(gtr["correct_option"])
        img_path = PRACTICE_ROOT / "images" / f"{img_name}.png"

        t_q = time.time()
        eq_pred, eq_conf, lp_pred, lp_conf, sample_answers, sample_lps, gens = (
            answer_image_sc_logprob(
                img_path, model, processor,
                n_samples=args.n_samples, temperature=args.temperature,
            )
        )
        elapsed = time.time() - t_q

        eq_score = score_one(eq_pred, truth)
        lp_score = score_one(lp_pred, truth)
        eq_total += eq_score
        lp_total += lp_score

        flag = "DIFFER" if eq_pred != lp_pred else ""
        print(
            f"[{i:>2}/{len(gt)}] {img_name:<38} truth={truth} "
            f"eq={eq_pred}({eq_conf:.2f}/{eq_score:+.2f}) "
            f"lp={lp_pred}({lp_conf:.2f}/{lp_score:+.2f}) "
            f"({elapsed:5.1f}s) {flag}"
        )
        rows.append({
            "image_name": img_name,
            "topic": gtr.get("topic", ""),
            "subtopic": gtr.get("subtopic", ""),
            "difficulty": gtr.get("difficulty", ""),
            "truth": truth,
            "eq_predicted": eq_pred,
            "eq_confidence": round(eq_conf, 4),
            "eq_score": eq_score,
            "lp_predicted": lp_pred,
            "lp_confidence": round(lp_conf, 4),
            "lp_score": lp_score,
            "n_samples": args.n_samples,
            "sample_answers": ",".join(str(a) for a in sample_answers),
            "sample_logprobs": ",".join(f"{x:.4f}" for x in sample_lps),
            "elapsed_sec": round(elapsed, 2),
            "generation": gens[0],
        })

    n = len(rows)
    eq_correct = sum(1 for r in rows if r["eq_predicted"] == r["truth"])
    lp_correct = sum(1 for r in rows if r["lp_predicted"] == r["truth"])
    eq_wrong = sum(1 for r in rows if r["eq_predicted"] in OPTIONS and r["eq_predicted"] != r["truth"])
    lp_wrong = sum(1 for r in rows if r["lp_predicted"] in OPTIONS and r["lp_predicted"] != r["truth"])
    eq_skip = sum(1 for r in rows if r["eq_predicted"] == 5)
    lp_skip = sum(1 for r in rows if r["lp_predicted"] == 5)

    print()
    print("=" * 70)
    print(f"[summary] n={n}  wall={time.time() - run_start:.1f}s")
    print(f"  {'metric':<20} {'equal-weight':>14} {'logprob-weight':>16}")
    print(f"  {'total score':<20} {eq_total:>+14.2f} {lp_total:>+16.2f}")
    print(f"  {'correct':<20} {eq_correct:>14} {lp_correct:>16}")
    print(f"  {'wrong':<20} {eq_wrong:>14} {lp_wrong:>16}")
    print(f"  {'skipped':<20} {eq_skip:>14} {lp_skip:>16}")
    print(f"  {'delta (lp - eq)':<20}                {lp_total - eq_total:>+16.2f}")

    differences = [r for r in rows if r["eq_predicted"] != r["lp_predicted"]]
    if differences:
        print(f"\n[questions where rules disagree: {len(differences)}]")
        for r in differences:
            eq_correct_str = "✓" if r["eq_predicted"] == r["truth"] else "✗"
            lp_correct_str = "✓" if r["lp_predicted"] == r["truth"] else "✗"
            print(
                f"  {r['image_name']:<36} truth={r['truth']} "
                f"eq={r['eq_predicted']}{eq_correct_str} lp={r['lp_predicted']}{lp_correct_str}  "
                f"votes={r['sample_answers']}  lps={r['sample_logprobs']}"
            )

    if not args.only and not args.limit:
        out = PRACTICE_ROOT / "logprob_eval_log.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[logprob] wrote {out.name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GNR-638 Project 2 contest entrypoint.

Reads MCQ images from `--test_dir`, runs Qwen2.5-VL-7B-Instruct with N=7
self-consistency decoding, and writes `submission.csv` (columns: id,
image_name, option) to the current working directory.

Production configuration (validated on a 50-question authored practice set,
+42.75/50 measured at NF4 + N=5; bf16 + N=7 is a strict precision/sample
upgrade):
    - Model precision : bf16 (no quantization)
    - Self-consistency: N = 7 samples at temperature 0.7
    - Aggregation     : majority vote with skip-averse tiebreak
    - Threshold (tau) : 0  (skip only when the majority itself produces 5)
    - Iron-clad parser: every output coerced into {1, 2, 3, 4, 5}

Usage (the grader runs):
    cd ./<repo>
    bash setup.bash
    conda activate gnr_project_env
    python inference.py --test_dir <absolute_path_to_test_dir>

Expected `--test_dir` layout:
    test_dir/
    ├── images/
    │   ├── image_1.png
    │   └── ...
    ├── test.csv               (column: id  OR  image_name)
    └── sample_submission.csv  (reference schema only)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="GNR-638 Project 2 inference")
    parser.add_argument(
        "--test_dir",
        type=Path,
        required=True,
        help="Absolute path to dir containing images/ and test.csv",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    test_dir = args.test_dir.resolve()
    weights_dir = project_root / "weights" / "qwen2.5-vl-7b"
    output_csv = Path.cwd() / "submission.csv"

    # Pre-flight checks (fail fast with a clear message rather than crashing
    # mid-load with a stack trace).
    if not test_dir.is_dir():
        sys.exit(f"--test_dir does not exist or is not a directory: {test_dir}")
    if not (test_dir / "images").is_dir():
        sys.exit(f"Missing 'images/' subdirectory under {test_dir}")
    if not (test_dir / "test.csv").exists():
        sys.exit(f"Missing test.csv under {test_dir}")
    if not weights_dir.is_dir():
        sys.exit(
            f"Model weights not found at {weights_dir}.\n"
            "Run setup.bash first to download them from Hugging Face."
        )

    print(f"[inference] test_dir   : {test_dir}")
    print(f"[inference] weights    : {weights_dir}")
    print(f"[inference] output     : {output_csv}")
    print(f"[inference] config     : bf16, N=7 SC, T=0.7, tau=0\n")

    # Hand off to src.submit which owns the actual inference + CSV-writing logic.
    # Subprocess call so we get clean output streaming; no shared state to worry about.
    cmd = [
        sys.executable, "-u", "-m", "src.submit",
        "--parent-dir",   str(test_dir),
        "--model-dir",    str(weights_dir),
        "--output",       str(output_csv),
        "--quantization", "none",      # bf16 native precision
        "--n-samples",    "7",
        "--temperature",  "0.7",
        "--tau",          "0.0",
    ]
    rc = subprocess.run(cmd, cwd=str(project_root)).returncode
    if rc != 0:
        sys.exit(f"src.submit exited with code {rc}")

    if not output_csv.exists():
        sys.exit(f"submission.csv was not produced at {output_csv}")
    print(f"\n[inference] done. submission.csv written to {output_csv}")


if __name__ == "__main__":
    main()

"""Render practice-set MCQs from questions.yaml to PNGs matching competition sample style.

Pipeline per question:
    YAML body + options  -->  .tex (one-page) --pdflatex-->  .pdf  --PyMuPDF-->  .png
Also writes ground_truth.csv (image_name, correct_option, topic, subtopic, difficulty).

Usage:
    python practice_set/render.py
    python practice_set/render.py --only cnn_shape_01
    python practice_set/render.py --dpi 200
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import yaml

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
TEMPLATE = ROOT / "template.tex"
QUESTIONS_YAML = ROOT / "questions.yaml"
IMAGES_DIR = ROOT / "images"
GROUND_TRUTH_CSV = ROOT / "ground_truth.csv"

LETTER_TO_DIGIT = {"A": 1, "B": 2, "C": 3, "D": 4}


def find_pdflatex() -> str:
    """Locate pdflatex.exe. Prefer PATH; fall back to known MiKTeX user-install path."""
    found = shutil.which("pdflatex")
    if found:
        return found
    fallback = Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe"
    if fallback.exists():
        return str(fallback)
    sys.exit(
        "pdflatex not found. Install MiKTeX (`winget install MiKTeX.MiKTeX`) "
        "or add its bin directory to PATH."
    )


def render_option(letter: str, opt: dict) -> str:
    """Render a single option (A/B/C/D) as LaTeX. Opt is {prose|math|code: str}."""
    if "code" in opt:
        code = opt["code"].rstrip("\n")
        return (
            f"\\item[{letter}.] \\mbox{{}}\\vspace{{-0.8em}}\n"
            f"\\begin{{lstlisting}}\n{code}\n\\end{{lstlisting}}"
        )
    if "math" in opt:
        return f"\\item[{letter}.] $\\displaystyle {opt['math']}$"
    if "prose" in opt:
        return f"\\item[{letter}.] {opt['prose']}"
    raise ValueError(f"Option {letter} must have one of: prose, math, code.")


def build_body(q: dict) -> str:
    """Assemble the LaTeX body for one question."""
    title = q["title"]
    body = q["body"].rstrip()
    options_tex = "\n\n".join(
        render_option(letter, q["options"][letter]) for letter in ["A", "B", "C", "D"]
    )
    return (
        f"\\section*{{Ques: {title}}}\n\n"
        f"{body}\n\n"
        f"\\subsection*{{Options}}\n"
        f"\\begin{{description}}[leftmargin=1.5em, labelindent=0pt, itemsep=0.6em]\n"
        f"{options_tex}\n"
        f"\\end{{description}}\n"
    )


def wrap_in_template(body: str) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    return template.replace("%% BODY %%", body)


def compile_to_png(tex_source: str, out_png: Path, pdflatex: str, dpi: int) -> None:
    """Compile LaTeX to PDF in a temp dir, then rasterize page 1 to PNG."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tex_file = tmp / "q.tex"
        tex_file.write_text(tex_source, encoding="utf-8")
        for _ in range(2):  # two passes for stable page numbers
            result = subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "q.tex"],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
        if result.returncode != 0:
            log = (tmp / "q.log").read_text(encoding="utf-8", errors="replace")
            sys.stderr.write(log[-3000:])
            raise RuntimeError(f"pdflatex failed for {out_png.stem}")
        pdf = tmp / "q.pdf"
        doc = fitz.open(pdf)
        pix = doc[0].get_pixmap(dpi=dpi)
        pix.save(str(out_png))
        doc.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="Render only this question id")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    pdflatex = find_pdflatex()
    IMAGES_DIR.mkdir(exist_ok=True)

    with QUESTIONS_YAML.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)

    questions = data["questions"]
    if args.only:
        questions = [q for q in questions if q["id"] == args.only]
        if not questions:
            sys.exit(f"No question with id '{args.only}'.")

    rows = []
    for q in questions:
        qid = q["id"]
        print(f"[render] {qid}")
        body = build_body(q)
        tex = wrap_in_template(body)
        out_png = IMAGES_DIR / f"{qid}.png"
        compile_to_png(tex, out_png, pdflatex, args.dpi)
        rows.append(
            {
                "image_name": qid,
                "correct_option": LETTER_TO_DIGIT[q["correct"]],
                "correct_letter": q["correct"],
                "topic": q["topic"],
                "subtopic": q["subtopic"],
                "difficulty": q["difficulty"],
                "source": q.get("source", ""),
            }
        )

    if not args.only:  # only rewrite full CSV when rendering everything
        with GROUND_TRUTH_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[ok] wrote {len(rows)} rows to {GROUND_TRUTH_CSV.name}")
    print(f"[ok] PNGs in {IMAGES_DIR}")


if __name__ == "__main__":
    main()

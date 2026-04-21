# Visual MCQ Solver

GNR 638 Project 2 — a pipeline that reads PNG images of deep-learning multiple-choice questions and predicts the correct option (or abstains).

See [CLAUDE.md](CLAUDE.md) for the full design, constraints, and development game plan.

## Requirements

- Python 3.12
- NVIDIA GPU with a CUDA 12.x-compatible driver (tested with driver 581.x, CUDA 13 reported by `nvidia-smi`)

## Setup

From the project root, in PowerShell:

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

If PowerShell blocks activation with a "scripts is disabled on this system" error, run this once (current user only, safe default):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

and then re-run the activate command.

Deactivate the venv when done with `deactivate`.

## Verifying the install

```powershell
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Expected: a torch version ≥ 2.5, `cuda True`, and your GPU name.

## Project structure

```
visual-mcq-solver/
├── CLAUDE.md           # full design doc & project game plan
├── README.md           # this file (setup & orientation)
├── requirements.txt    # Python dependencies
├── images/             # sample MCQ images from the competition
├── src/                # inference & evaluation helpers (wip)
├── notebooks/          # exploratory + final submission notebooks (wip)
└── practice_set/       # our authored practice MCQs (wip)
```

## Status

Early scaffolding stage. No inference pipeline yet — see `CLAUDE.md` for the plan and next steps.

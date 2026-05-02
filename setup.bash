#!/bin/bash
# GNR-638 Project 2 — setup script.
#
# Clones the public repo into the current directory, creates a conda env,
# installs Python dependencies, and downloads the Qwen2.5-VL-7B-Instruct
# model weights from Hugging Face. After this script completes, the
# inference script can be run with:
#
#     conda activate gnr_project_env
#     python inference.py --test_dir <absolute_path_to_test_dir>
#
# Internet is required during setup. No internet is required at inference time.

set -euo pipefail

REPO_URL="https://github.com/SirCoolerArc/visual-mcq-solver.git"
ENV_NAME="gnr_project_env"
PYTHON_VERSION="3.11"

# ----------------------------------------------------------------------------
# 1. Clone repo into current directory
# ----------------------------------------------------------------------------
# `git clone <url> .` refuses to clone into a non-empty directory, and our cwd
# already contains setup.bash. So we clone into a temp subdir and move contents up.

echo "[setup] cloning repo from ${REPO_URL} ..."
TMP_CLONE="_gnr_clone_tmp"
rm -rf "${TMP_CLONE}"
git clone --depth 1 "${REPO_URL}" "${TMP_CLONE}"

# Move all files (incl. hidden) from temp clone up to cwd, skipping setup.bash.
shopt -s dotglob
for item in "${TMP_CLONE}"/*; do
    name="$(basename "$item")"
    [[ "${name}" == "setup.bash" ]] && continue
    [[ "${name}" == "." || "${name}" == ".." ]] && continue
    mv "$item" .
done
shopt -u dotglob
rm -rf "${TMP_CLONE}"

# ----------------------------------------------------------------------------
# 2. Initialize conda for this shell, then create the environment
# ----------------------------------------------------------------------------

echo "[setup] initializing conda ..."
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found in PATH" >&2
    exit 1
fi
# This makes `conda activate` work inside the script.
eval "$(conda shell.bash hook)"

echo "[setup] creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION}) ..."
# Defensive: remove the env if a previous run left one behind.
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"

# ----------------------------------------------------------------------------
# 3. Install Python dependencies
# ----------------------------------------------------------------------------

echo "[setup] installing python dependencies ..."
pip install --upgrade pip
pip install -r requirements.txt

# ----------------------------------------------------------------------------
# 4. Download Qwen2.5-VL-7B-Instruct weights
# ----------------------------------------------------------------------------

echo "[setup] downloading Qwen2.5-VL-7B-Instruct weights (~16 GB) ..."
python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir="weights/qwen2.5-vl-7b",
    ignore_patterns=["*.msgpack", "*.h5", "original/*"],
    max_workers=4,
)
print("weights downloaded.")
PYEOF

echo "[setup] done. Run: conda activate ${ENV_NAME} && python inference.py --test_dir <abs_path>"

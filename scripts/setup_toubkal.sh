#!/bin/bash
# =============================================================================
# SERVIS environment setup on Toubkal (run as a SLURM batch job on a GPU node).
#
#   sbatch scripts/setup_toubkal.sh
#
# It creates the conda env, installs a CUDA-matched torch, and compiles the
# three CUDA extensions. The build MUST run on a GPU node with nvcc, which is
# why this is an sbatch job and not something to run on the login node.
# =============================================================================
#SBATCH --job-name=servis-setup
#SBATCH --output=servis-setup-%j.log
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu                  # A100 nodes. For H100 use: gpu_h100

set -euo pipefail

# ----------------------------- TOUBKAL SETTINGS ------------------------------
# Configured for the A100 'gpu' partition. Toubkal's newest CUDA module is
# CUDA/12.8.0, so torch is pinned to the cu128 build (local cu130 has no match
# here). For the H100 'gpu_h100' partition, change ARCH_LIST to 9.0 and the
# #SBATCH --partition above to gpu_h100.
#   A100 -> 8.0   H100 -> 9.0   V100 -> 7.0
CUDA_MODULE="CUDA/12.8.0"
TORCH_CUDA="cu128"
TORCH_VERSION="2.11.0"
ARCH_LIST="8.0"

ENV_NAME="servis"
CONDA_BASE="$HOME/miniconda3"
REPO="$HOME/lustre/med_img-z2y8h4a967e/code_Haytam/SERVIS"
# -----------------------------------------------------------------------------

echo "[setup] node=$(hostname)  repo=$REPO"

module purge
module load "$CUDA_MODULE"   # provides nvcc; the host compiler comes from conda (step 2b)

command -v nvcc >/dev/null || { echo "ERROR: nvcc not found after 'module load $CUDA_MODULE'"; exit 1; }
echo "[setup] nvcc: $(nvcc --version | grep release)"

source "$CONDA_BASE/etc/profile.d/conda.sh"

# 1) conda env --------------------------------------------------------------
if conda env list | grep -qE "^[[:space:]]*${ENV_NAME}[[:space:]]"; then
  echo "[setup] env '$ENV_NAME' exists, reusing"
else
  conda env create -n "$ENV_NAME" -f "$REPO/environment.yml"
fi
conda activate "$ENV_NAME"

# 2) torch matching the cluster CUDA ---------------------------------------
echo "[setup] installing torch==$TORCH_VERSION ($TORCH_CUDA)"
pip install --no-cache-dir "torch==${TORCH_VERSION}" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# 2b) self-contained compiler toolchain ------------------------------------
# Toubkal's EasyBuild binutils 'as' crashes (SIGILL: "Internal error (Illegal
# instruction)") on the compute-node CPU, breaking every nvcc/c++ compile.
# Build with conda-forge's own CPU-baseline gcc/binutils and force nvcc to use
# it as the host compiler (-ccbin) so it goes through conda's assembler.
echo "[setup] installing conda-forge compiler toolchain"
conda install -y -c conda-forge gxx_linux-64=12 sysroot_linux-64=2.17 ninja
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"

# 3) build the CUDA extensions on this GPU node -----------------------------
# --no-build-isolation so they compile against the torch we just installed.
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-4}"
echo "[setup] building extensions for arch $ARCH_LIST"
# Use the gaussian-splatting submodule's rasterizer — it supports the
# `antialiasing` setting that scenes/gs.py passes (the top-level
# third_party/diff-gaussian-rasterization is an older fork without it).
pip install --no-build-isolation "$REPO/SRC/third_party/gaussian-splatting/submodules/diff-gaussian-rasterization"
pip install --no-build-isolation "$REPO/SRC/third_party/gaussian-splatting/submodules/simple-knn"
pip install --no-build-isolation "$REPO/SRC/third_party/gaussian-splatting/submodules/fused-ssim"

# 4) smoke test -------------------------------------------------------------
python - <<'PY'
import torch
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
import diff_gaussian_rasterization, simple_knn, fused_ssim  # noqa: F401
print("CUDA extensions import OK")
PY

echo "[setup] DONE. Activate with:  conda activate $ENV_NAME"

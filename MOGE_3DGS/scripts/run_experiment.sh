#!/usr/bin/env bash
# Orchestrate the MoGe-supervised 3DGS comparison study for one scene.
#
# Steps:
#   0. (assumed done) COLMAP run → <scene>/{images,sparse/0/...}
#   1. Cache MoGe depth      → <scene>/depths_moge/
#   2. Align to COLMAP       → <scene>/sparse/0/depth_params.json
#   3. Train each variant    → <runs_root>/<scene_name>/<variant>/
#
# Usage:
#   scripts/run_experiment.sh <scene_dir> [runs_root]
#
# Env overrides (defaults in []):
#   ITERS=30000                training iters
#   DEPTHS_DIR=depths_moge     subdir under <scene>
#   VARIANTS=...               space-separated subset of the variants below
#   LAMBDA_D=0.1               initial depth-loss weight
#   LAMBDA_D_FINAL=0.01        final  depth-loss weight
#   DEPTH_LOSS_UNTIL=20000     iter at which depth loss is shut off
#   SKIP_PREPROCESS=0          set to 1 to skip MoGe cache + align
#   MOGE_FP16=1                run MoGe in fp16
#   PY=python                  python executable

set -euo pipefail

SCENE="${1:?scene dir required}"
RUNS_ROOT="${2:-runs}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MOGE_3DGS="$REPO_ROOT/MOGE_3DGS"

PY="${PY:-python}"
ITERS="${ITERS:-30000}"
DEPTHS_DIR="${DEPTHS_DIR:-depths_moge}"
LAMBDA_D="${LAMBDA_D:-0.1}"
LAMBDA_D_FINAL="${LAMBDA_D_FINAL:-0.01}"
DEPTH_LOSS_UNTIL="${DEPTH_LOSS_UNTIL:-20000}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
MOGE_FP16="${MOGE_FP16:-1}"

ALL_VARIANTS=(
    "none"
    "l1_inv"
    "l1_inv_aligned"
    "l1_inv_solve_align"
    "pearson_inv"
    "pearson_depth"
    "masked_l1_inv_aligned"
)
VARIANTS="${VARIANTS:-${ALL_VARIANTS[*]}}"

scene_name="$(basename "$(realpath "$SCENE")")"
out_root="$RUNS_ROOT/$scene_name"
mkdir -p "$out_root"

echo "[run_experiment] scene = $SCENE"
echo "[run_experiment] outputs -> $out_root"
echo "[run_experiment] variants: $VARIANTS"

# --- 1 + 2: preprocess (MoGe cache + COLMAP alignment) ---------------------
if [[ "$SKIP_PREPROCESS" != "1" ]]; then
    fp16_flag=""
    [[ "$MOGE_FP16" == "1" ]] && fp16_flag="--fp16"
    echo "[run_experiment] caching MoGe depth -> $SCENE/$DEPTHS_DIR"
    $PY "$MOGE_3DGS/preprocess/cache_moge_depth.py" \
        --scene "$SCENE" --out-dir "$DEPTHS_DIR" $fp16_flag
    echo "[run_experiment] aligning to COLMAP sparse points"
    $PY "$MOGE_3DGS/preprocess/align_to_colmap.py" \
        --scene "$SCENE" --depths-dir "$DEPTHS_DIR"
else
    echo "[run_experiment] SKIP_PREPROCESS=1; using existing $SCENE/$DEPTHS_DIR"
fi

# --- 3: train each variant -------------------------------------------------
for v in $VARIANTS; do
    out_dir="$out_root/$v"
    if [[ -d "$out_dir" && -n "$(ls -A "$out_dir" 2>/dev/null)" ]]; then
        echo "[run_experiment] skipping $v (output exists at $out_dir)"
        continue
    fi
    mkdir -p "$out_dir"
    log="$out_dir/train.log"
    echo "[run_experiment] === training variant: $v ==="
    extra_args=()
    if [[ "$v" != "none" ]]; then
        extra_args=(
            "-d" "$DEPTHS_DIR"
            "--lambda-d" "$LAMBDA_D"
            "--lambda-d-final" "$LAMBDA_D_FINAL"
            "--depth-loss-until" "$DEPTH_LOSS_UNTIL"
        )
    fi
    $PY "$MOGE_3DGS/train/train.py" \
        -s "$SCENE" \
        -m "$out_dir" \
        --iterations "$ITERS" \
        --depth-loss "$v" \
        "${extra_args[@]}" \
        2>&1 | tee "$log"
done

echo "[run_experiment] all variants complete."
echo "[run_experiment] outputs under: $out_root"

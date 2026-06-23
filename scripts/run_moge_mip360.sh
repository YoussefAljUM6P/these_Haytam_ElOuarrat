#!/bin/bash
# run_moge_mip360.sh — MoGe-2 depth-supervised twin of run_pipeline_mip360.sh.
#
# Line-for-line identical to the Mip-360 baseline (same poses/split, same
# -i images_4 --eval --antialiasing train/render/metrics flags) with ONE extra
# step: MoGe-2 monocular depth generation + per-image scale fit, then training
# with -d depths. Mip-360 is RGB-only, so the baseline has NO depth — this is a
# MoGe-depth vs no-depth comparison. Writes output_moge/ + metrics_moge.csv so
# the baseline (output/, metrics.csv) is left intact.
#
# Usage:
#   bash run_moge_mip360.sh <datasets_root> [N] [scene]
#       <datasets_root>  e.g. 360_v2  (or SERVIS/DATA)
#       N                train = 1-of-N frames, rest test. Default 2.
#       scene            optional: process ONLY this scene.
#                        OMIT to process EVERY scene found in <datasets_root>.
#
# Output: <datasets_root>/metrics_moge.csv   (scene,PSNR,LPIPS,SSIM)
#
# REQUIRES the same reader edits as the baseline:
#   1) llffhold default 8 → 0           (setup_reader.sh)
#   2) the `if "360" in path: llffhold=8` override DISABLED.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export QT_QPA_PLATFORM=offscreen

REPO_ROOT="/home/haytam.elouarrat/lustre/med_img-z2y8h4a967e/code_Haytam"
GS_DIR="$REPO_ROOT/gaussian-splatting"
READER="$GS_DIR/scene/dataset_readers.py"
PIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_360="$REPO_ROOT/gs_benchmark_360"
MOGE_SCRIPT="$REPO_ROOT/SERVIS/SRC/make_moge_depths.py"

GS_ENV="gaussian_splatting"
SERVIS_ENV="servis"         # has MoGe-2 + joblib (for make_depth_scale.py)
CUDA_MODULE="CUDA/12.1.1"

IMAGES_DIR="images_4"       # SAME as the baseline
ITERS=30000
DENSIFY_UNTIL=20000
GRAD_THRESHOLD=0.00015
DEPTHS_NAME="depths"        # <scene>/depths, passed as -d depths

NTFY="ntfy.sh/HPC"
# ─────────────────────────────────────────────────────────────────────────────

DATASETS_ROOT="${1:?Usage: bash run_moge_mip360.sh <datasets_root> [N] [scene]}"
DATASETS_ROOT="$(realpath "${DATASETS_ROOT%/}")"
N="${2:-2}"
[[ "$N" =~ ^[0-9]+$ ]] && (( N >= 2 )) || { echo "❌ N must be int >= 2 (got $N)"; exit 1; }
shift || true; shift || true
SCENES=("$@")
# No explicit scene → ALL scenes in the dataset (every dir with a sparse/0).
if (( ! ${#SCENES[@]} )); then
    for d in "$DATASETS_ROOT"/*/; do
        [ -f "${d}sparse/0/images.bin" ] && SCENES+=("$(basename "$d")")
    done
    (( ${#SCENES[@]} )) || { echo "❌ no scenes with sparse/0 under $DATASETS_ROOT"; exit 1; }
    echo "ℹ️  no scene given → all ${#SCENES[@]}: ${SCENES[*]}"
fi

CSV="$DATASETS_ROOT/metrics_moge.csv"
SPLITTER="$BENCH_360/make_split_mip360.py"
COLLECTOR="$BENCH_360/collect_metrics.py"

conda_on()  { set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate "$1"; set -u; }
conda_off() { set +u; conda deactivate; set -u; }

# ─── Preflight: reader must NOT force llffhold=8 on a "360" path ─────────────
if grep -qE '^[[:space:]]*if "360" in path:' "$READER"; then
    echo "❌ Reader still forces llffhold=8 when the path contains '360'."
    echo "   Your dataset dir is 360_v2, so this silently overrides your 1-of-$N split."
    echo "   Fix it once, then re-run:"
    echo "     sed -i 's/if \"360\" in path:/if False:  # disabled for 360_v2/' $READER"
    exit 1
fi
if ! grep -qE 'def readColmapSceneInfo\(.*llffhold=0\):' "$READER"; then
    echo "❌ Reader default is not llffhold=0 — run setup_reader.sh first."
    exit 1
fi

module load "$CUDA_MODULE" 2>/dev/null || true
[ -f "$CSV" ] || echo "scene,PSNR,LPIPS,SSIM" > "$CSV"

trap 'curl -d "❌ moge mip360 FAILED at: ${SCENE:-unknown}" '"$NTFY"' || true' ERR

conda_on "$GS_ENV"

for SCENE in "${SCENES[@]}"; do
    SCENE_DIR="$DATASETS_ROOT/$SCENE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🗂️  $SCENE  (MoGe variant)"
    [ -f "$SCENE_DIR/sparse/0/images.bin" ] || { echo "    ⚠ no sparse/0 — skipping"; continue; }
    curl -d "🚀 moge mip360 start: $SCENE" "$NTFY" || true

    OUTPUT="$SCENE_DIR/output_moge"
    DEPTHS_DIR="$SCENE_DIR/$DEPTHS_NAME"

    # 1) split (skip if present) — reuse the baseline's exact split
    if [ -f "$SCENE_DIR/sparse/0/test.txt" ]; then
        echo "    [split] ✅ exists"
    else
        echo "    [split] writing 1-of-$N"
        python "$SPLITTER" "$SCENE_DIR" "$N" "$GS_DIR"
    fi

    # 1b) MoGe depth + per-image scale fit (servis env: MoGe + joblib) ─── NEW ──
    if [ -f "$OUTPUT/point_cloud/iteration_$ITERS/point_cloud.ply" ]; then
        : # already trained; depth maps no longer needed
    else
        conda_on "$SERVIS_ENV"
        if [ -d "$DEPTHS_DIR" ] && [ -n "$(ls -A "$DEPTHS_DIR" 2>/dev/null)" ]; then
            echo "    [moge]  ✅ depths exist"
        else
            echo "    [moge]  generating inverse-depth maps from $IMAGES_DIR"
            HF_HUB_OFFLINE=1 python "$MOGE_SCRIPT" \
                --img-path "$SCENE_DIR/$IMAGES_DIR" --outdir "$DEPTHS_DIR"
        fi
        if [ -f "$SCENE_DIR/sparse/0/depth_params.json" ]; then
            echo "    [scale] ✅ depth_params.json exists"
        else
            echo "    [scale] fitting per-image depth scales"
            python "$GS_DIR/utils/make_depth_scale.py" \
                --base_dir "$SCENE_DIR" --depths_dir "$DEPTHS_DIR"
        fi
        conda_off
    fi

    # 2) train (skip if done) — baseline flags + depth supervision
    if [ -f "$OUTPUT/point_cloud/iteration_$ITERS/point_cloud.ply" ]; then
        echo "    🔥 already trained (iter $ITERS)"
    else
        echo "    🔥 training ($ITERS iters, -i $IMAGES_DIR -d $DEPTHS_NAME)"
        python "$GS_DIR/train.py" \
            -s "$SCENE_DIR" -i "$IMAGES_DIR" -m "$OUTPUT" \
            --eval --antialiasing \
            --iterations "$ITERS" \
            --save_iterations 7000 "$ITERS" \
            --test_iterations 7000 "$ITERS" \
            --densify_until_iter "$DENSIFY_UNTIL" \
            --densify_grad_threshold "$GRAD_THRESHOLD" \
            -d "$DEPTHS_NAME" --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01
    fi

    # 3) render held-out test + metrics
    echo "    🖼️  rendering held-out test views"
    python "$GS_DIR/render.py" -m "$OUTPUT" --antialiasing --skip_train --iteration "$ITERS"
    echo "    📊 metrics"
    python "$GS_DIR/metrics.py" -m "$OUTPUT"

    # 4) CSV
    python "$COLLECTOR" --scene "$SCENE" --results "$OUTPUT/results.json" --csv "$CSV"
    curl -d "✅ moge mip360 done: $SCENE" "$NTFY" || true
done

conda_off

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📄 MoGe metrics → $CSV"
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
echo "Compare against the no-depth baseline in $DATASETS_ROOT/metrics.csv"
curl -d "🎉 moge mip360 done: ${SCENES[*]}" "$NTFY" || true

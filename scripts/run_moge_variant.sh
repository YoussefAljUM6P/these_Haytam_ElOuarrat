#!/bin/bash
# run_moge_variant.sh — MoGe-depth variant of the GS benchmark, for A/B vs the
# captured-depth baseline produced by gs_benchmark/run_pipeline.sh.
#
# It changes EXACTLY ONE thing vs the baseline: the depth source. Same joint
# reconstruction, same 1-of-N test.txt split, same train/render/metrics flags.
# So the captured-vs-MoGe PSNR/LPIPS/SSIM difference reflects depth quality and
# nothing else.
#
# Per scene (must already have sparse/0/{cameras,test}.txt from the baseline run):
#   1. MoGe-2 inverse-depth maps  -> <scene>/depths_moge        [servis env, GPU]
#   2. fit per-image scales       -> sparse/0/depth_params.json [gaussian_splatting]
#   3. train with -d depths_moge  -> <scene>/output_moge        [gaussian_splatting]
#   4. render held-out test + metrics
#   5. append a "<scene>_moge" row to metrics_moge.csv
# The captured depth_params.json is backed up and restored, so the baseline
# stays reproducible (depth_params.json is only used by the depth LOSS at train
# time; render/metrics ignore it, so the existing baseline output is unaffected).
#
# Usage:
#   bash run_moge_variant.sh <datasets_root> [scene_name]
#       <datasets_root>  same as the baseline, e.g. SERVIS/DATA
#       scene_name       optional: process only this one scene (e.g. room)
#
# Resumable: a scene whose output_moge/.../point_cloud.ply exists is skipped.
set -euo pipefail
export QT_QPA_PLATFORM=offscreen

# ─── Config (must match run_pipeline.sh) ─────────────────────────────────────
REPO_ROOT="/home/haytam.elouarrat/lustre/med_img-z2y8h4a967e/code_Haytam"
GS_DIR="$REPO_ROOT/gaussian-splatting"
SERVIS_DIR="$REPO_ROOT/SERVIS"
MOGE_SCRIPT="$SERVIS_DIR/SRC/make_moge_depths.py"
COLLECTOR="$REPO_ROOT/gs_benchmark/collect_metrics.py"
SERVIS_ENV="servis"                 # has MoGe-2 installed
GS_ENV="gaussian_splatting"         # baseline training env
CUDA_MODULE="CUDA/12.1.1"

# Identical to the baseline (run_pipeline.sh) so the comparison is fair.
ITERS=30000
DENSIFY_UNTIL=20000
GRAD_THRESHOLD=0.00015
COLOR_GLOB="*.color.jpg"            # only feed colour frames to MoGe
NTFY="ntfy.sh/HPC"
# ─────────────────────────────────────────────────────────────────────────────

DATASETS_ROOT="${1:?Usage: bash run_moge_variant.sh <datasets_root> [scene_name]}"
DATASETS_ROOT="$(realpath "${DATASETS_ROOT%/}")"
ONLY_SCENE="${2:-}"                 # optional: restrict to a single scene
CSV="$DATASETS_ROOT/metrics_moge.csv"

conda_on()  { set +u; source ~/miniconda3/etc/profile.d/conda.sh; conda activate "$1"; set -u; }
conda_off() { set +u; conda deactivate; set -u; }

module load "$CUDA_MODULE" 2>/dev/null || true
[ -f "$CSV" ] || echo "scene,PSNR,LPIPS,SSIM" > "$CSV"

trap 'curl -d "❌ moge variant FAILED at: ${SCENE:-unknown}" '"$NTFY"' || true' ERR
FAILED=()

for SCENE_DIR in "$DATASETS_ROOT"/*/; do
    SCENE_DIR="${SCENE_DIR%/}"
    SCENE=$(basename "$SCENE_DIR")
    DATA="$SCENE_DIR/data"
    SP0="$SCENE_DIR/sparse/0"
    OUT_MOGE="$SCENE_DIR/output_moge"
    DEPTHS_MOGE="$SCENE_DIR/depths_moge"

    [ -d "$DATA" ] || continue
    # Optional single-scene filter.
    if [ -n "$ONLY_SCENE" ] && [ "$SCENE" != "$ONLY_SCENE" ]; then
        continue
    fi
    # Require the baseline artefacts so we reuse the SAME recon + split.
    if [ ! -f "$SP0/cameras.bin" ] || [ ! -f "$SP0/test.txt" ]; then
        echo "⏭️  $SCENE: no sparse/0 + test.txt (run the baseline first) — skipping"
        continue
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🗂️  $SCENE  (MoGe variant)"
    curl -d "🚀 moge start: $SCENE" "$NTFY" || true

    if [ -f "$OUT_MOGE/point_cloud/iteration_$ITERS/point_cloud.ply" ]; then
        echo "    ⏭️  output_moge already trained (iter $ITERS)"
    else
        # 1) MoGe-2 depth maps (servis env, GPU, offline — weights pre-cached) ──
        if [ -d "$DEPTHS_MOGE" ] && [ -n "$(ls -A "$DEPTHS_MOGE" 2>/dev/null)" ]; then
            echo "    [moge]  ✅ depths_moge exists"
        else
            echo "    [moge]  generating inverse-depth maps"
            conda_on "$SERVIS_ENV"
            HF_HUB_OFFLINE=1 python "$MOGE_SCRIPT" \
                --img-path "$DATA" --outdir "$DEPTHS_MOGE" --pattern "$COLOR_GLOB"
            conda_off
        fi

        conda_on "$GS_ENV"

        # 2) fit per-image scales for the MoGe maps. make_depth_scale.py writes a
        #    fixed sparse/0/depth_params.json, so back up the captured one first.
        if [ -f "$SP0/depth_params.json" ] && [ ! -f "$SP0/depth_params.captured.json" ]; then
            cp "$SP0/depth_params.json" "$SP0/depth_params.captured.json"
            echo "    [scale] backed up captured depth_params.json"
        fi
        echo "    [scale] fitting MoGe depth scales"
        python "$GS_DIR/utils/make_depth_scale.py" \
            --base_dir "$SCENE_DIR" --depths_dir "$DEPTHS_MOGE"
        cp "$SP0/depth_params.json" "$SP0/depth_params.moge.json"

        # 3) train — identical flags to the baseline, only -d/-m differ ─────────
        echo "    🔥 training MoGe variant ($ITERS iters)"
        python "$GS_DIR/train.py" \
            -s "$SCENE_DIR" -i data -m "$OUT_MOGE" \
            --eval --antialiasing \
            --resolution 2 \
            --data_device cpu \
            --iterations "$ITERS" \
            --save_iterations 7000 "$ITERS" \
            --test_iterations 7000 "$ITERS" \
            --densify_until_iter "$DENSIFY_UNTIL" \
            --densify_grad_threshold "$GRAD_THRESHOLD" \
            -d depths_moge --depth_l1_weight_init 1.0 --depth_l1_weight_final 0.01

        # restore the captured depth_params.json so the baseline stays intact
        if [ -f "$SP0/depth_params.captured.json" ]; then
            cp "$SP0/depth_params.captured.json" "$SP0/depth_params.json"
            echo "    [scale] restored captured depth_params.json"
        fi
        conda_off
    fi

    # 4) render held-out test views + metrics (same flags as baseline) ─────────
    conda_on "$GS_ENV"
    echo "    🖼️  rendering held-out test views"
    python "$GS_DIR/render.py" -m "$OUT_MOGE" --antialiasing --resolution 2 \
        --skip_train --iteration "$ITERS"
    echo "    📊 computing metrics"
    python "$GS_DIR/metrics.py" -m "$OUT_MOGE"
    conda_off

    # 5) append a tagged row so it sits next to the captured baseline ──────────
    echo "    [csv]   collecting"
    conda_on "$GS_ENV"
    python "$COLLECTOR" --scene "${SCENE}_moge" \
        --results "$OUT_MOGE/results.json" --csv "$CSV"
    conda_off

    curl -d "✅ moge done: $SCENE" "$NTFY" || true
    echo "    ✅ $SCENE MoGe variant complete"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📄 MoGe metrics → $CSV"
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
echo
echo "Compare against the captured-depth baseline in $DATASETS_ROOT/metrics.csv"

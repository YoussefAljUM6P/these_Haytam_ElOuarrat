#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-nsys}"
shift || true

CONDA_ENV="${SERVIS_CONDA_ENV:-viservo}"
RUNNER="${SERVIS_RUNNER:-servo-frames}"
CONFIG="${SERVIS_CONFIG:-CONFIGS/servo_kitchen_mesh.json}"
OUT_DIR="${SERVIS_PROFILE_DIR:-profiles}"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_DIR"

CMD=(
  conda run --no-capture-output -n "$CONDA_ENV"
  python SRC/cli.py "$RUNNER" --config "$CONFIG" "$@"
)

case "$MODE" in
  nsys)
    nsys profile \
      --force-overwrite=true \
      --sample=process-tree \
      --cpuctxsw=process-tree \
      --trace=cuda,nvtx,osrt,cudnn,cublas \
      --output="$OUT_DIR/servis-$STAMP" \
      "${CMD[@]}"
    if [[ -f "$OUT_DIR/servis-$STAMP.nsys-rep" ]]; then
      nsys stats "$OUT_DIR/servis-$STAMP.nsys-rep"
    else
      echo "Nsight report importer unavailable; kept $OUT_DIR/servis-$STAMP.qdstrm"
    fi
    ;;
  ncu)
    ncu \
      --target-processes=all \
      --set=full \
      --kernel-name="${NCU_KERNEL:-regex:renderCUDA|preprocessCUDA}" \
      --launch-skip="${NCU_LAUNCH_SKIP:-0}" \
      --launch-count="${NCU_LAUNCH_COUNT:-20}" \
      --export="$OUT_DIR/servis-$STAMP" \
      "${CMD[@]}"
    ;;
  torch)
    conda run --no-capture-output -n "$CONDA_ENV" \
      python -m torch.utils.bottleneck \
      SRC/cli.py "$RUNNER" --config "$CONFIG" "$@" \
      | tee "$OUT_DIR/torch-bottleneck-$STAMP.txt"
    ;;
  perf)
    perf stat -d -d -d -o "$OUT_DIR/perf-$STAMP.txt" -- "${CMD[@]}"
    cat "$OUT_DIR/perf-$STAMP.txt"
    ;;
  monitor)
    telemetry="$OUT_DIR/nvidia-smi-$STAMP.csv"
    nvidia-smi \
      --query-gpu=timestamp,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current \
      --format=csv -l 1 > "$telemetry" &
    monitor_pid=$!
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    "${CMD[@]}"
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    trap - EXIT
    echo "Wrote $telemetry"
    ;;
  *)
    cat >&2 <<EOF
Usage: $0 {nsys|ncu|torch|perf|monitor} [SERVIS CLI overrides]

Example:
  SERVIS_CONFIG=CONFIGS/servo_kitchen_mesh.json \\
    $0 nsys --set renderer=gs --set iterations=10 --set viz_iter=0
EOF
    exit 2
    ;;
esac

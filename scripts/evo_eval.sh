#!/usr/bin/env bash
set -e

RUN_PATH="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_ROOT="${SERVIS_RUNS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)/RUNS}"

if [ -z "$RUN_PATH" ]; then
  echo "Usage:"
  echo "  ./eval_evo_run.sh gs/20260511-135731_mesh_BF_sift_intrinsic/kitchen"
  exit 1
fi

if [[ "$RUN_PATH" = /* ]]; then
  RUN_DIR="${RUN_PATH%/}/"
else
  RUN_DIR="$BASE_ROOT/${RUN_PATH%/}/"
fi

GT="$RUN_DIR/gt_traj.tum"
SIM="$RUN_DIR/sim_traj.tum"

echo "======================================================"
echo "Evaluating:"
echo "$RUN_DIR"
echo "======================================================"

if [ ! -f "$GT" ]; then
  echo "Missing file:"
  echo "$GT"
  exit 1
fi

if [ ! -f "$SIM" ]; then
  echo "Missing file:"
  echo "$SIM"
  exit 1
fi

echo
echo "[1/4] APE Translation (Sim(3) aligned)"
evo_ape tum "$GT" "$SIM" \
  -a -s -va \
  --pose_relation trans_part

echo
echo "[2/4] APE Rotation (Sim(3) aligned)"
evo_ape tum "$GT" "$SIM" \
  -a -s -va \
  --pose_relation angle_deg

echo
echo "[3/4] RPE Translation (Sim(3) aligned)"
evo_rpe tum "$GT" "$SIM" \
  -a -s -va \
  --pose_relation trans_part

echo
echo "[4/4] RPE Rotation (Sim(3) aligned)"
evo_rpe tum "$GT" "$SIM" \
  -a -s -va \
  --pose_relation angle_deg

echo
echo "======================================================"
echo "Done."
echo "======================================================"

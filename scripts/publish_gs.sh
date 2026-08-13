#!/bin/bash
# Publish trained gaussian-splatting outputs into the SERVIS DATA/ layout as
# <scene>/gs.ply (or <scene>/gs_<variant>.ply for variant runs).
#
# For every trained model dir found under --output-root (any directory that
# contains a point_cloud/ subdir), it:
#   1. picks the highest-iteration point_cloud.ply (override with --iteration),
#   2. resolves the scene name from the model's cfg_args `source_path`
#      (falling back to the model dir name), and
#   3. copies it to DATA/<scene>/<dst> (or symlinks with --link), where <dst>
#      is derived from the model dir name so variant runs land beside the
#      baseline instead of clobbering it:
#        output        -> gs.ply        (standard 3DGS)
#        output_moge   -> gs_moge.ply   (MoGe depth-supervised variant)
#        output_<v>    -> gs_<v>.ply    (any other variant)
#      The SERVIS renderer selects between gs.ply and gs_moge.ply via the
#      gs_model config key ("standard" / "moge").
#
# Usage:
#   scripts/publish_gs.sh [--output-root DIR] [--data-root DIR]
#                         [--iteration N] [--link] [--dry-run]
#
# Examples:
#   scripts/publish_gs.sh --output-root output      # gaussian-splatting ./output/<run>/...
#   scripts/publish_gs.sh                           # default: DATA/<scene>/output/...
#   scripts/publish_gs.sh --output-root /path/to/trained --link
#   scripts/publish_gs.sh --dry-run                 # show what would happen, change nothing

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$REPO/DATA"
OUTPUT_ROOT=""
ITERATION=""        # empty => highest available
LINK=0
DRYRUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2;;
    --data-root)   DATA_ROOT="$2";   shift 2;;
    --iteration)   ITERATION="$2";   shift 2;;
    --link)        LINK=1;  shift;;
    --dry-run)     DRYRUN=1; shift;;
    -h|--help)     sed -n 's/^# \{0,1\}//p' "$0" | sed '/^!/d'; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done

# Default search root: the per-scene DATA/<scene>/output/ layout.
ROOT="${OUTPUT_ROOT:-$DATA_ROOT}"
[[ -d "$ROOT" ]] || { echo "search root not found: $ROOT" >&2; exit 1; }

# Every dir that holds a point_cloud/ subtree is a trained model dir.
mapfile -t models < <(find "$ROOT" -type d -name point_cloud -printf '%h\n' 2>/dev/null | sort -u)
if [[ ${#models[@]} -eq 0 ]]; then
  echo "No trained models (a point_cloud/ dir) found under: $ROOT" >&2
  exit 1
fi

published=0
for model in "${models[@]}"; do
  # 1) pick the iteration
  if [[ -n "$ITERATION" ]]; then
    iter="$ITERATION"
  else
    iter=$(ls -d "$model"/point_cloud/iteration_* 2>/dev/null \
           | sed 's/.*iteration_//' | sort -n | tail -1)
  fi
  ply="$model/point_cloud/iteration_${iter}/point_cloud.ply"
  if [[ -z "${iter}" || ! -f "$ply" ]]; then
    echo "[skip] $model — no point_cloud.ply (iteration ${iter:-none})" >&2
    continue
  fi

  # 2) resolve scene name
  scene=""
  # (a) model under DATA/<scene>/output/... -> use the <scene> path component
  case "$model/" in
    "$DATA_ROOT"/*) rel="${model#"$DATA_ROOT"/}"; scene="${rel%%/*}";;
  esac
  # (b) else the trained model's cfg_args source_path basename
  if [[ -z "$scene" && -f "$model/cfg_args" ]]; then
    src=$(sed -n "s/.*source_path='\([^']*\)'.*/\1/p" "$model/cfg_args")
    [[ -n "$src" ]] && scene="$(basename "$src")"
  fi
  # (c) else the model dir name
  [[ -z "$scene" ]] && scene="$(basename "$model")"

  dst_dir="$DATA_ROOT/$scene"
  if [[ ! -d "$dst_dir" ]]; then
    echo "[skip] scene '$scene' has no DATA dir ($dst_dir)" >&2
    continue
  fi

  # Derive the destination filename from the model dir name so variant runs
  # (output_moge, output_<v>) land beside the baseline instead of clobbering it.
  base="$(basename "$model")"
  case "$base" in
    output_*) out_name="gs_${base#output_}.ply";;
    *)        out_name="gs.ply";;
  esac
  dst="$dst_dir/$out_name"

  # 3) publish
  if [[ $DRYRUN -eq 1 ]]; then
    echo "[dry-run] iteration_${iter}: $ply  ->  $dst"
    continue
  fi
  rm -f "$dst"
  if [[ $LINK -eq 1 ]]; then
    ln -s "$ply" "$dst"
    echo "[link] $scene/$out_name -> iteration_${iter}"
  else
    cp -f "$ply" "$dst"
    echo "[copy] $scene/$out_name  (iteration_${iter}, $(du -h "$dst" | cut -f1))"
  fi
  published=$((published + 1))
done

echo "Done: published ${published} scene(s) under $DATA_ROOT."

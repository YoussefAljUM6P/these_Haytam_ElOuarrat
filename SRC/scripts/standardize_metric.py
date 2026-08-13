#!/usr/bin/env python3
"""Convert a run's translation values from COLMAP scene scale into metric units.

12-Scenes ("Stanford") scenes ship a metric ground truth: one 4x4 camera-to-world
pose per frame in ``DATA/<scene>/data/frame-XXXXXX.pose.txt``, in **metres**. The
servoing runs, however, live in the arbitrary scale of the scene's COLMAP
reconstruction, so a ``translation_gap`` of ``0.01`` in one scene is not the same
physical distance as ``0.01`` in another and cross-scene averages are meaningless.

This tool recovers, per scene, the single scalar ``s`` (metres per COLMAP unit)
that maps the reconstruction to the metric ground truth, then multiplies every
translation-unit value in a run by ``s``. Rotations are already in degrees
(scale-invariant) and are left untouched; pixel/timing columns are left untouched.

Scale recovery (per scene, robust and gauge-free):
  * COLMAP camera centres come from ``DATA/<scene>/transforms.json`` when present,
    otherwise from the run's own ``gt_traj.tum`` (proven identical in scale).
  * Metric camera centres come from the ``.pose.txt`` files, matched by frame id.
  * ``s`` is the scale term of a scale-only Umeyama similarity fit over all matched
    centres, with three rounds of worst-10%-residual rejection. This extracts the
    map's scale; it is NOT the estimate->GT alignment we reject for reporting error.

Only 12-Scenes scenes are handled: a scene is treated as metric iff it has at
least one ``data/*.pose.txt`` file. Non-metric scenes are
skipped with a message.

Outputs (non-destructive) land in ``<run>/metric_standardized/``:
    metric_scale.json          scene, scale, unit, fit residual, #inliers, source
    per_task_errors.csv        translation columns x s
    trajectory_summary.json    ape/rpe translation metrics x s (sse x s^2)
    sim_traj.tum, gt_traj.tum   positions x s

Usage:
    python SRC/scripts/standardize_metric.py RUNS/LOUNGE-GS-IBVS-INTRINSIC-XFEAT
    python SRC/scripts/standardize_metric.py --all
    python SRC/scripts/standardize_metric.py --all --unit mm
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "DATA"
RUNS_ROOT = PROJECT_ROOT / "RUNS"

FRAME_RE = re.compile(r"frame-(\d+)")

# per_task_errors.csv columns that are distances in COLMAP world units.
TASK_TRANSLATION_COLS = (
    "initial_translation_gap",
    "translation_gap",
    "pre_cut_translation_gap",
    "motion_max_translation_step",
)

# trajectory_summary metric families that are distances (leave *_rotation_deg).
SUMMARY_TRANSLATION_FAMILIES = ("ape_translation", "rpe_translation")
# keys inside a family scaled by s (sse scales by s**2, handled separately).
SUMMARY_LINEAR_KEYS = ("rmse", "mean", "median", "std", "min", "max")


def frame_id(text):
    """Zero-padded 6-digit frame id from any string containing frame-<n>."""
    m = FRAME_RE.search(text or "")
    if not m:
        return None
    return f"{int(m.group(1)):06d}"


def umeyama_scale(X, Y):
    """Scale term of the similarity Y ~ s R X + t (Umeyama 1991), plus fit RMS.

    X: source (COLMAP) centres, N x 3.  Y: target (metric) centres, N x 3.
    """
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    sigma = (Yc.T @ Xc) / len(X)
    U, D, Vt = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_x = (Xc**2).sum() / len(X)
    s = float(np.trace(np.diag(D) @ S) / var_x)
    t = my - s * R @ mx
    resid = Y - (s * (X @ R.T) + t)
    rms = float(np.sqrt((resid**2).sum(1).mean()))
    return s, R, t, rms


def robust_scale(X, Y, rounds=3, keep_frac=0.9):
    """Umeyama scale with iterative worst-residual rejection."""
    s, R, t, rms = umeyama_scale(X, Y)
    keep = np.ones(len(X), bool)
    for _ in range(rounds):
        resid = np.linalg.norm(Y - (s * (X @ R.T) + t), axis=1)
        keep = resid <= np.quantile(resid, keep_frac)
        if keep.sum() < 4:
            break
        s, R, t, rms = umeyama_scale(X[keep], Y[keep])
    return s, rms, int(keep.sum())


def metric_centers(scene_dir):
    """{frame_id: metric camera centre} from data/*.pose.txt (cam-to-world, metres)."""
    data_dir = Path(scene_dir) / "data"
    if not data_dir.is_dir():
        return {}
    out = {}
    for pose_path in data_dir.glob("*.pose.txt"):
        fid = frame_id(pose_path.name)
        if fid is None:
            continue
        M = np.loadtxt(pose_path)
        if M.shape == (4, 4) and np.isfinite(M).all():
            out[fid] = M[:3, 3]
    return out


def colmap_centers_from_transforms(scene_dir):
    """{frame_id: COLMAP camera centre} from transforms.json, or {} if absent."""
    tj = Path(scene_dir) / "transforms.json"
    if not tj.exists():
        return {}
    t = json.loads(tj.read_text())
    out = {}
    for fr in t.get("frames", []):
        fid = frame_id(fr.get("file_path", ""))
        if fid is None:
            continue
        M = np.array(fr["transform_matrix"], float)
        out[fid] = M[:3, 3]
    return out


def ordered_frame_ids(run_dir):
    """Frame ids aligned to gt_traj.tum rows: [task0.src] + [task_k.target ...]."""
    csv_path = run_dir / "per_task_errors.csv"
    if not csv_path.exists():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    ids = [frame_id(rows[0].get("src_frame", ""))]
    ids += [frame_id(r.get("target_frame", "")) for r in rows]
    return ids


def colmap_centers_from_run(run_dir):
    """{frame_id: COLMAP centre} from the run's own gt_traj.tum (fallback)."""
    gt_path = run_dir / "gt_traj.tum"
    ids = ordered_frame_ids(run_dir)
    if not gt_path.exists() or ids is None:
        return {}
    gt = np.loadtxt(gt_path)
    if gt.ndim != 2 or gt.shape[0] != len(ids):
        return {}
    out = {}
    for fid, row in zip(ids, gt):
        if fid is not None:
            out[fid] = row[1:4]  # tx ty tz
    return out


# Per-scene scale cache, so every run of a scene shares one physical scale
# (a scene's scale is a constant; it must not vary run-to-run). Keyed by scene
# name; value is the result tuple or None (non-metric scene).
_SCENE_SCALE_CACHE = {}
_RUNS_BY_SCENE = None


def build_runs_by_scene():
    """{scene_name: [run_dir, ...]} scanned once from RUNS/."""
    global _RUNS_BY_SCENE
    if _RUNS_BY_SCENE is not None:
        return _RUNS_BY_SCENE
    mapping = {}
    for d in RUNS_ROOT.iterdir():
        if not d.is_dir():
            continue
        scene, _ = resolve_scene_dir(d)
        if scene is not None:
            mapping.setdefault(scene, []).append(d)
    _RUNS_BY_SCENE = mapping
    return mapping


def pooled_colmap_centers(scene):
    """Union of gt_traj centres across every run of a scene (fallback source).

    Pooling all runs' frames gives one stable, scene-wide estimate instead of a
    per-run one computed from whatever few frames a single run happened to visit.
    """
    out = {}
    for run_dir in build_runs_by_scene().get(scene, []):
        out.update(colmap_centers_from_run(run_dir))
    return out


def compute_scale(scene, scene_dir, run_dir):
    """Per-scene (scale, fit_rms_m, n_inliers, source, fit_pct); None if non-metric.

    Cached per scene so all runs of a scene get the identical physical scale.
    """
    if scene in _SCENE_SCALE_CACHE:
        return _SCENE_SCALE_CACHE[scene]

    metric = metric_centers(scene_dir)
    if not metric:
        _SCENE_SCALE_CACHE[scene] = None  # not a 12-Scenes / Stanford scene
        return None

    source = "transforms.json"
    colmap = colmap_centers_from_transforms(scene_dir)
    if not colmap:
        # No transforms.json: pool gt_traj across all runs of this scene so the
        # scale is scene-consistent (fixes the 5a 0.254-vs-0.208 split).
        colmap = pooled_colmap_centers(scene)
        source = "gt_traj.tum(pooled)"
    if not colmap:
        raise RuntimeError("no COLMAP centres (transforms.json and gt_traj.tum both unusable)")

    common = sorted(set(colmap) & set(metric))
    if len(common) < 4:
        raise RuntimeError(f"only {len(common)} frame(s) shared between COLMAP and metric poses")
    X = np.array([colmap[k] for k in common])
    Y = np.array([metric[k] for k in common])
    s, rms, n = robust_scale(X, Y)
    # Fit residual relative to the metric scene extent is the trust signal: a
    # clean COLMAP<->metric similarity sits near ~1%. A large value means the
    # correspondences are wrong (e.g. gt_traj frame ids that don't share the
    # 12-Scenes numbering), so the recovered scale must not be trusted blindly.
    diameter = float(np.linalg.norm(Y.max(0) - Y.min(0)))
    fit_pct = 100.0 * rms / diameter if diameter > 0 else float("inf")
    result = (s, rms, n, source, fit_pct)
    _SCENE_SCALE_CACHE[scene] = result
    return result


def scale_number(value, factor):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value  # blank / non-numeric cell -> leave as-is
    if f != f:  # NaN
        return value
    return f * factor


def standardize_task_csv(run_dir, out_dir, factor):
    src = run_dir / "per_task_errors.csv"
    if not src.exists():
        return None
    rows = list(csv.DictReader(src.open()))
    if not rows:
        return None
    fields = list(rows[0].keys())
    cols = [c for c in TASK_TRANSLATION_COLS if c in fields]
    for r in rows:
        for c in cols:
            if r[c] not in ("", None):
                r[c] = f"{scale_number(r[c], factor):.9g}"
    out = out_dir / "per_task_errors.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return cols


def standardize_summary(run_dir, out_dir, factor):
    src = run_dir / "trajectory_summary.json"
    if not src.exists():
        return None
    summary = json.loads(src.read_text())
    for scene_summary in summary.values():
        if not isinstance(scene_summary, dict):
            continue
        metrics = scene_summary.get("metrics", {})
        for fam in SUMMARY_TRANSLATION_FAMILIES:
            block = metrics.get(fam)
            if not isinstance(block, dict):
                continue
            for k in SUMMARY_LINEAR_KEYS:
                if k in block and block[k] is not None:
                    block[k] = scale_number(block[k], factor)
            if "sse" in block and block["sse"] is not None:
                block["sse"] = scale_number(block["sse"], factor * factor)
        scene_summary["metric_unit"] = "mm" if abs(factor) >= 100 else "m"
    (out_dir / "trajectory_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# Families/keys emitted in the evo-metrics text block, in order.
EVO_FAMILIES = ("ape_translation", "ape_rotation_deg", "rpe_translation", "rpe_rotation_deg")
EVO_KEYS = ("rmse", "mean", "median", "std", "min", "max")


def format_evo_block(run_name, summary, scene, unit):
    """Render one run's standardized metrics in the evo-metrics text format."""
    scene_summary = summary.get(scene)
    if not isinstance(scene_summary, dict):
        scene_summary = next((v for v in summary.values() if isinstance(v, dict)), {})
    metrics = scene_summary.get("metrics", {})
    n = metrics.get("num_poses")
    n_part = f"{n} poses" if n is not None else "n/a poses"
    lines = [f"=== {run_name} evo metrics ({n_part}, translation in {unit}) ==="]

    has_data = any(
        isinstance(metrics.get(fam), dict) and metrics[fam].get("rmse") is not None
        for fam in EVO_FAMILIES
    )
    if not has_data:
        lines.append("  (no evo metrics recorded -- run incomplete or aborted)")
        return "\n".join(lines)

    for fam in EVO_FAMILIES:
        block = metrics.get(fam, {})
        vals = " ".join(
            f"{k}={float(block[k]):.6f}"
            for k in EVO_KEYS
            if block.get(k) is not None
        )
        lines.append(f"  {fam}: {vals or 'n/a'}")

    # Control-usability line: success rate, render latency, closed-loop FPS.
    num_tasks = scene_summary.get("num_tasks")
    num_conv = scene_summary.get("num_converged")
    timing = scene_summary.get("timing", {}) or {}
    parts = []
    if num_tasks:
        if num_conv is not None:
            parts.append(f"success={100.0 * num_conv / num_tasks:.1f}% ({num_conv}/{num_tasks})")
        else:
            parts.append(f"tasks={num_tasks}")
    if timing.get("render_ms_mean") is not None:
        parts.append(f"render_ms={float(timing['render_ms_mean']):.2f}")
    if timing.get("fps") is not None:
        parts.append(f"fps={float(timing['fps']):.2f}")
    if parts:
        lines.append("  control: " + " ".join(parts))

    # A run where nothing converged reports ~0 error only because failed tasks
    # are cut to the desired pose; the error stats above are meaningless.
    if num_tasks and num_conv == 0:
        lines.append("  [FAILED -- 0% converged; error stats are cut-to-GT artifacts, ignore]")
    return "\n".join(lines)


def standardize_tum(run_dir, out_dir, factor):
    for name in ("sim_traj.tum", "gt_traj.tum"):
        src = run_dir / name
        if not src.exists():
            continue
        arr = np.loadtxt(src)
        if arr.ndim == 1:
            arr = arr[None, :]
        arr[:, 1:4] *= factor  # scale positions; quaternion untouched
        np.savetxt(out_dir / name, arr, fmt="%.9f")


def resolve_scene_dir(run_dir):
    """(scene_name, scene_dir) from trajectory_summary.json, falling back to DATA/<key>."""
    summ = run_dir / "trajectory_summary.json"
    if summ.exists():
        data = json.loads(summ.read_text())
        for key, val in data.items():
            if isinstance(val, dict):
                sd = val.get("scene_dir")
                scene = val.get("scene", key)
                if sd and Path(sd).exists():
                    return scene, Path(sd)
                cand = DATA_ROOT / str(scene)
                if cand.exists():
                    return scene, cand
    # last resort: guess from run dir name prefix
    return None, None


def standardize_run(run_dir, unit, max_fit_pct):
    run_dir = Path(run_dir)
    scene, scene_dir = resolve_scene_dir(run_dir)
    if scene_dir is None:
        return {"run": run_dir.name, "status": "skip", "reason": "scene dir not resolved"}

    try:
        result = compute_scale(scene, scene_dir, run_dir)
    except RuntimeError as e:
        return {"run": run_dir.name, "scene": scene, "status": "error", "reason": str(e)}
    if result is None:
        return {"run": run_dir.name, "scene": scene, "status": "skip",
                "reason": "not a Stanford/12-Scenes scene (no metric poses)"}

    s, rms, n, source, fit_pct = result
    factor = s * (1000.0 if unit == "mm" else 1.0)
    low_confidence = fit_pct > max_fit_pct

    out_dir = run_dir / "metric_standardized"
    out_dir.mkdir(exist_ok=True)
    scale_info = {
        "scene": scene,
        "scale_m_per_colmap_unit": s,
        "unit": unit,
        "applied_factor": factor,
        "fit_rms_m": rms,
        "fit_pct_of_scene": fit_pct,
        "n_inliers": n,
        "colmap_center_source": source,
        "low_confidence": low_confidence,
        "max_fit_pct": max_fit_pct,
    }
    (out_dir / "metric_scale.json").write_text(json.dumps(scale_info, indent=2))
    scaled_cols = standardize_task_csv(run_dir, out_dir, factor)
    scaled_summary = standardize_summary(run_dir, out_dir, factor)
    standardize_tum(run_dir, out_dir, factor)

    evo_block = None
    if scaled_summary is not None:
        evo_block = format_evo_block(run_dir.name, scaled_summary, scene, unit)
        (out_dir / "evo_metrics.txt").write_text(evo_block + "\n")

    return {"run": run_dir.name, "scene": scene,
            "status": "warn" if low_confidence else "ok", "unit": unit,
            "scale_m_per_colmap_unit": s, "fit_rms_m": rms, "fit_pct": fit_pct,
            "n_inliers": n, "source": source, "scaled_columns": scaled_cols,
            "evo_block": evo_block, "out_dir": str(out_dir)}


def iter_run_dirs(args):
    if args.all:
        for d in sorted(RUNS_ROOT.iterdir()):
            if d.is_dir() and (d / "trajectory_summary.json").exists():
                yield d
    else:
        for p in args.runs:
            yield Path(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="Run directories under RUNS/ to standardize.")
    ap.add_argument("--all", action="store_true", help="Process every run in RUNS/.")
    ap.add_argument("--unit", choices=("m", "mm"), default="m",
                    help="Output translation unit. Default: metres.")
    ap.add_argument("--max-fit-pct", type=float, default=3.0,
                    help="Flag a run low-confidence if the COLMAP<->metric fit "
                         "residual exceeds this %% of scene extent. Default: 3.")
    args = ap.parse_args()
    if not args.runs and not args.all:
        ap.error("provide run directories or --all")

    ok = warned = skipped = errored = 0
    blocks = []
    for run_dir in iter_run_dirs(args):
        r = standardize_run(run_dir, args.unit, args.max_fit_pct)
        status = r["status"]
        if status in ("ok", "warn"):
            tag = "[ok]  " if status == "ok" else "[WARN]"
            note = "" if status == "ok" else "  <-- low confidence, do not trust"
            ok += status == "ok"
            warned += status == "warn"
            print(f"{tag} {r['run']:<45} scene={r['scene']:<20} "
                  f"scale={r['scale_m_per_colmap_unit']:.6f} m/unit  "
                  f"fit={r['fit_rms_m']*1000:.1f}mm ({r['fit_pct']:.1f}%)/n={r['n_inliers']}  "
                  f"src={r['source']}{note}")
            if r.get("evo_block"):
                blocks.append(r["evo_block"] if status == "ok"
                              else r["evo_block"] + "\n  [LOW CONFIDENCE - scale unreliable]")
        elif status == "skip":
            skipped += 1
            print(f"[skip] {r['run']:<45} {r['reason']}")
        else:
            errored += 1
            print(f"[ERR]  {r['run']:<45} {r['reason']}")

    if blocks:
        report = "\n\n".join(blocks) + "\n"
        report_path = RUNS_ROOT / f"metric_standardized_report_{args.unit}.txt"
        report_path.write_text(report)
        print(f"\n--- standardized evo metrics (translation in {args.unit}) ---\n")
        print(report)
        print(f"Combined report written to {report_path}")

    print(f"Done: {ok} standardized, {warned} low-confidence, "
          f"{skipped} skipped, {errored} errored.")
    return 0 if errored == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

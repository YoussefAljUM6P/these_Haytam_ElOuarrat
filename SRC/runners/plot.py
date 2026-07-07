"""Regenerate detailed evo plots from a previous run's saved trajectories.

This never touches the servo loop: it only reads the ``sim_traj.tum`` and
``gt_traj.tum`` that every trajectory run writes, re-runs the evo APE/RPE
evaluation, and emits a richer plot set than the live runner does. Because the
trajectories are already on disk it is fast and safe to re-run -- e.g. to try a
different ``--rpe-delta`` -- without re-servoing anything.

    python cli.py plot --run RUNS/<dir> [--rpe-delta 1] [--out <dir>]

By default plots land in ``<run>/evo_plots/`` so the live runner's originals
(trajectory_xyz.png, ape_translation.png, ...) are left untouched.
"""

from pathlib import Path

from runners.servo_frames import PROJECT_ROOT, RUNS_ROOT
from runners.trajectory import poses_to_evo_trajectory, read_tum, stat_dict


def add_arguments(parser):
    parser.add_argument(
        "--run",
        required=True,
        help=(
            "Run directory (or path to a sim_traj.tum) of a previous trajectory "
            "run. Resolved relative to RUNS/ if not found as given."
        ),
    )
    parser.add_argument(
        "--rpe-delta",
        type=int,
        default=1,
        help="RPE delta in frames (default 1).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for plots (default <run>/evo_plots).",
    )
    parser.add_argument(
        "--take",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Plot only take N (1-based) of a multi-take run instead of the "
            "whole combined trajectory. Default: all takes together."
        ),
    )
    parser.add_argument(
        "--list-takes",
        action="store_true",
        help="List the takes detected in the run and exit (no plotting).",
    )


def resolve_run_dir(run_arg):
    """Accept a run dir, a RUNS-relative name, or a path to sim_traj.tum."""
    candidate = Path(run_arg)
    if candidate.is_file() and candidate.name.endswith(".tum"):
        candidate = candidate.parent
    if not candidate.exists():
        candidate = RUNS_ROOT / run_arg
    if not candidate.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_arg}")
    return candidate.resolve()


def read_take_windows(run_dir):
    """Detect the takes of a finished trajectory run from per_task_errors.csv.

    A take is a smooth capture pass; the runner marks the first task of each
    take with ``sequence_reset=1``. Every task appends exactly one pose to
    sim/gt_traj.tum at timestamp ``task_index + 1`` (pose 0 is the initial
    camera at timestamp 0), so each take maps to a contiguous timestamp window.

    Returns a list of dicts ``{take, task_start, task_end, t_lo, t_hi,
    n_tasks}`` (1-based ``take``), or ``[]`` when there is no CSV / no rows.
    A single-take run yields one entry spanning the whole trajectory.
    """
    import csv

    csv_path = Path(run_dir) / "per_task_errors.csv"
    if not csv_path.is_file():
        return []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    def _is_reset(row):
        return str(row.get("sequence_reset", "0")).strip() not in ("", "0", "False", "false")

    starts = [int(r["task_index"]) for r in rows if _is_reset(r)]
    first_task = int(rows[0]["task_index"])
    if not starts or starts[0] != first_task:
        starts = [first_task] + starts  # take 1 always begins at the first task
    last_task = int(rows[-1]["task_index"])

    windows = []
    for k, start in enumerate(starts):
        nxt = starts[k + 1] if k + 1 < len(starts) else last_task + 1
        windows.append(
            {
                "take": k + 1,
                "task_start": start,
                "task_end": nxt - 1,
                "t_lo": float(start + 1),   # pose timestamp of the take's first task
                "t_hi": float(nxt),          # pose timestamp of the take's last task
                "n_tasks": nxt - start,
            }
        )
    return windows


def _filter_by_window(ts, poses, window):
    """Keep only (timestamp, pose) pairs whose timestamp is inside window."""
    if window is None:
        return ts, poses
    t_lo, t_hi = window
    kept = [(t, p) for t, p in zip(ts, poses) if t_lo <= t <= t_hi]
    if not kept:
        return [], []
    ks, kp = zip(*kept)
    return list(ks), list(kp)


def detailed_evo_plots(sim_tum, gt_tum, out_dir, rpe_delta, label, take_window=None):
    """Read two TUM files and write the detailed evo APE/RPE plot set.

    ``take_window`` optionally restricts evaluation to a ``(t_lo, t_hi)``
    timestamp range (inclusive) so a single take can be plotted in isolation;
    pass None to use the whole trajectory.

    Returns the metrics dict (same shape the trajectory runner records).
    """
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evo.core import metrics as evo_metrics
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.tools import plot

    sim_ts, sim_poses = read_tum(sim_tum)
    gt_ts, gt_poses = read_tum(gt_tum)

    sim_ts, sim_poses = _filter_by_window(sim_ts, sim_poses, take_window)
    gt_ts, gt_poses = _filter_by_window(gt_ts, gt_poses, take_window)
    if len(sim_ts) < 2 or len(gt_ts) < 2:
        raise ValueError(
            "not enough poses to evaluate"
            + (f" for take window {take_window}" if take_window else "")
            + f" (sim={len(sim_ts)}, gt={len(gt_ts)})"
        )

    traj_sim = poses_to_evo_trajectory(sim_ts, sim_poses)
    traj_gt = poses_to_evo_trajectory(gt_ts, gt_poses)
    traj_gt, traj_sim = sync.associate_trajectories(traj_gt, traj_sim, max_diff=1e-3)
    pair = (traj_gt, traj_sim)

    ape_t = evo_metrics.APE(PoseRelation.translation_part)
    ape_t.process_data(pair)
    ape_r = evo_metrics.APE(PoseRelation.rotation_angle_deg)
    ape_r.process_data(pair)
    rpe_t = evo_metrics.RPE(
        PoseRelation.translation_part,
        delta=float(rpe_delta), delta_unit=Unit.frames,
        rel_delta_tol=0.0, all_pairs=False,
    )
    rpe_t.process_data(pair)
    rpe_r = evo_metrics.RPE(
        PoseRelation.rotation_angle_deg,
        delta=float(rpe_delta), delta_unit=Unit.frames,
        rel_delta_tol=0.0, all_pairs=False,
    )
    rpe_r.process_data(pair)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 3D + top-down trajectory overlays ---------------------------------
    for mode, suffix, title in (
        (plot.PlotMode.xyz, "trajectory_xyz", "sim vs GT (3D)"),
        (plot.PlotMode.xy, "trajectory_xy", "sim vs GT (top-down xy)"),
    ):
        fig = plt.figure(figsize=(9, 8))
        ax = plot.prepare_axis(fig, mode)
        plot.traj(ax, mode, traj_gt, style="-", color="green", label="GT")
        plot.traj(ax, mode, traj_sim, style="--", color="red", label="sim")
        ax.legend()
        ax.set_title(f"{label}: {title}")
        fig.savefig(out_dir / f"{suffix}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # --- sim trajectory colored by APE translation error -------------------
    fig = plt.figure(figsize=(9, 8))
    ax = plot.prepare_axis(fig, plot.PlotMode.xy)
    plot.traj(ax, plot.PlotMode.xy, traj_gt, style="--", color="grey", alpha=0.6)
    plot.traj_colormap(
        ax, traj_sim, ape_t.error, plot.PlotMode.xy,
        min_map=float(ape_t.error.min()), max_map=float(ape_t.error.max()),
    )
    ax.set_title(f"{label}: trajectory colored by APE translation")
    fig.savefig(out_dir / "trajectory_ape_colormap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- per-task error curves with stat overlays (rmse/mean/median/std) ----
    curves = (
        ("ape_translation", ape_t, "APE translation (m)"),
        ("ape_rotation", ape_r, "APE rotation (deg)"),
        ("rpe_translation", rpe_t, f"RPE translation (m), delta={rpe_delta}"),
        ("rpe_rotation", rpe_r, f"RPE rotation (deg), delta={rpe_delta}"),
    )
    for suffix, metric, title in curves:
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        plot.error_array(
            ax, metric.error,
            statistics={
                s: metric.get_statistic(getattr(evo_metrics.StatisticsType, s))
                for s in ("rmse", "mean", "median", "std")
            },
            name=title, title=f"{label}: {title}", xlabel="task index",
        )
        fig.savefig(out_dir / f"{suffix}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    metrics_out = {
        "ape_translation": stat_dict(ape_t),
        "ape_rotation_deg": stat_dict(ape_r),
        "rpe_translation": stat_dict(rpe_t),
        "rpe_rotation_deg": stat_dict(rpe_r),
        "rpe_delta": int(rpe_delta),
        "num_poses": int(traj_gt.num_poses),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    return metrics_out


def run(args):
    run_dir = resolve_run_dir(args.run)
    sim_tum = run_dir / "sim_traj.tum"
    gt_tum = run_dir / "gt_traj.tum"
    for path in (sim_tum, gt_tum):
        if not path.is_file():
            raise FileNotFoundError(f"missing trajectory file: {path}")

    windows = read_take_windows(run_dir)

    if getattr(args, "list_takes", False):
        if len(windows) <= 1:
            print(f"{run_dir.name}: single take (no take split).")
        else:
            print(f"{run_dir.name}: {len(windows)} takes")
            for w in windows:
                print(
                    f"  take {w['take']}: tasks {w['task_start']}-{w['task_end']} "
                    f"({w['n_tasks']} tasks)"
                )
        return

    take = getattr(args, "take", None)
    take_window = None
    out_dir = Path(args.out) if args.out else run_dir / "evo_plots"
    label = run_dir.name
    if take is not None:
        match = next((w for w in windows if w["take"] == take), None)
        if match is None:
            n = len(windows)
            raise ValueError(
                f"take {take} not found in {run_dir.name}; this run has "
                f"{n} take{'s' if n != 1 else ''} (valid: 1..{n})."
            )
        take_window = (match["t_lo"], match["t_hi"])
        label = f"{run_dir.name} (take {take}/{len(windows)})"
        if args.out is None:
            out_dir = run_dir / "evo_plots" / f"take_{take}"

    metrics = detailed_evo_plots(
        sim_tum, gt_tum, out_dir, args.rpe_delta, label, take_window=take_window
    )

    try:
        rel = out_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = out_dir
    print(f"\n=== {label} evo metrics ({metrics['num_poses']} poses) ===")
    for name in ("ape_translation", "ape_rotation_deg",
                 "rpe_translation", "rpe_rotation_deg"):
        stats = metrics[name]
        print(
            f"  {name}: rmse={stats.get('rmse', float('nan')):.6f} "
            f"mean={stats.get('mean', float('nan')):.6f} "
            f"median={stats.get('median', float('nan')):.6f} "
            f"std={stats.get('std', float('nan')):.6f}"
        )
    print(f"\nwrote detailed plots -> {rel}")

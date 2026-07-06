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


def detailed_evo_plots(sim_tum, gt_tum, out_dir, rpe_delta, label):
    """Read two TUM files and write the detailed evo APE/RPE plot set.

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

    out_dir = Path(args.out) if args.out else run_dir / "evo_plots"
    label = run_dir.name

    metrics = detailed_evo_plots(sim_tum, gt_tum, out_dir, args.rpe_delta, label)

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

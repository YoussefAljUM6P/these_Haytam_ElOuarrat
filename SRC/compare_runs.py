"""Overlay several trajectory runs (sim) against a shared GT.

Usage:
    python SRC/compare_runs.py <run1_scene_dir> <run2_scene_dir> [...] \\
        [--labels mesh,gs] \\
        [--ref <dir>] \\
        [--out <dir>] \\
        [--align] \\
        [--rpe-delta 1]

Each positional argument must be a directory containing `sim_traj.tum` and
`gt_traj.tum` (the layout produced by `cli.py trajectory`).

Outputs in --out (default: ./compare_<timestamp> under cwd):
    trajectory_xyz.png      sim vs GT, 3D
    trajectory_xy.png       sim vs GT, top-down
    ape_translation.png     APE translation per task, all runs overlaid
    ape_rotation.png        APE rotation per task, all runs overlaid
    metrics.txt             APE/RPE rmse|mean|median|std|min|max per run
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from runners.trajectory import (
    poses_to_evo_trajectory,
    read_tum,
    stat_dict,
)


_DEFAULT_COLORS = [
    "red", "blue", "orange", "purple", "brown", "magenta", "cyan", "olive",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path,
                   help="Run-scene directories (each holds sim_traj.tum + gt_traj.tum).")
    p.add_argument("--labels", type=str, default=None,
                   help="Comma-separated labels matching the run order (default: dir names).")
    p.add_argument("--ref", type=Path, default=None,
                   help="Directory whose gt_traj.tum is used as the reference (default: first run).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: ./compare_<timestamp>).")
    p.add_argument("--align", action="store_true",
                   help="Apply Umeyama SE3 alignment of each sim to GT before metrics + plots.")
    p.add_argument("--rpe-delta", type=int, default=1,
                   help="RPE delta in frames (default 1).")
    return p.parse_args()


def derive_label(run_dir):
    # <runs_root>/.../<timestamp_tag>/<scene>/ -> "<timestamp_tag>/<scene>"
    parts = run_dir.resolve().parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return run_dir.name


def main():
    args = parse_args()

    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(args.runs):
            raise ValueError(
                f"--labels has {len(labels)} entries, but {len(args.runs)} runs given"
            )
    else:
        labels = [derive_label(d) for d in args.runs]

    for d in args.runs:
        if not (d / "sim_traj.tum").exists():
            raise FileNotFoundError(f"{d}: missing sim_traj.tum")
        if not (d / "gt_traj.tum").exists():
            raise FileNotFoundError(f"{d}: missing gt_traj.tum")

    ref_dir = args.ref if args.ref is not None else args.runs[0]
    if not (ref_dir / "gt_traj.tum").exists():
        raise FileNotFoundError(f"--ref dir missing gt_traj.tum: {ref_dir}")

    out_dir = args.out or Path(
        f"./compare_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reference GT: {ref_dir / 'gt_traj.tum'}")
    ts_gt, gt_poses = read_tum(ref_dir / "gt_traj.tum")
    traj_gt = poses_to_evo_trajectory(ts_gt, gt_poses)

    sim_trajs = []
    for run_dir, label in zip(args.runs, labels):
        ts, poses = read_tum(run_dir / "sim_traj.tum")
        traj = poses_to_evo_trajectory(ts, poses)
        sim_trajs.append((label, run_dir, traj))
        print(f"Loaded {label}: {len(poses)} poses from {run_dir / 'sim_traj.tum'}")

    # -- metrics ------------------------------------------------------------
    from evo.core import metrics as evo_metrics
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit

    per_run_metrics = {}
    per_run_ape_t_err = {}  # for plotting
    per_run_ape_r_err = {}
    per_run_aligned = {}    # aligned sim trajectories (for plot if --align)

    for label, run_dir, traj in sim_trajs:
        traj_gt_synced, traj_sim_synced = sync.associate_trajectories(
            traj_gt, traj, max_diff=1e-3,
        )

        if args.align:
            traj_sim_synced.align(traj_gt_synced, correct_scale=False)

        pair = (traj_gt_synced, traj_sim_synced)

        ape_t = evo_metrics.APE(PoseRelation.translation_part)
        ape_t.process_data(pair)
        ape_r = evo_metrics.APE(PoseRelation.rotation_angle_deg)
        ape_r.process_data(pair)

        rpe_t = evo_metrics.RPE(
            PoseRelation.translation_part,
            delta=float(args.rpe_delta),
            delta_unit=Unit.frames,
            rel_delta_tol=0.0,
            all_pairs=False,
        )
        rpe_t.process_data(pair)
        rpe_r = evo_metrics.RPE(
            PoseRelation.rotation_angle_deg,
            delta=float(args.rpe_delta),
            delta_unit=Unit.frames,
            rel_delta_tol=0.0,
            all_pairs=False,
        )
        rpe_r.process_data(pair)

        per_run_metrics[label] = {
            "ape_translation": stat_dict(ape_t),
            "ape_rotation_deg": stat_dict(ape_r),
            "rpe_translation": stat_dict(rpe_t),
            "rpe_rotation_deg": stat_dict(rpe_r),
            "num_poses": int(traj_gt_synced.num_poses),
            "run_dir": str(run_dir),
        }
        per_run_ape_t_err[label] = (traj_sim_synced.timestamps, ape_t.error)
        per_run_ape_r_err[label] = (traj_sim_synced.timestamps, ape_r.error)
        per_run_aligned[label] = traj_sim_synced

    # -- plots --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from evo.tools import plot

    color_map = {lab: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
                 for i, lab in enumerate(labels)}

    def plot_modes(mode, figsize, fname, title):
        fig = plt.figure(figsize=figsize)
        ax = plot.prepare_axis(fig, mode)
        plot.traj(ax, mode, traj_gt, "-", "green", label="GT")
        for label, _run_dir, traj in sim_trajs:
            traj_to_plot = per_run_aligned[label] if args.align else traj
            plot.traj(ax, mode, traj_to_plot, "--",
                      color_map[label], label=label)
        ax.legend()
        ax.set_title(title)
        fig.savefig(out_dir / fname, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_dir / fname}")

    plot_modes(plot.PlotMode.xyz, (10, 8),
               "trajectory_xyz.png", "sim vs GT (3D)")
    plot_modes(plot.PlotMode.xy, (8, 8),
               "trajectory_xy.png", "sim vs GT (top-down xy)")

    fig_err = plt.figure(figsize=(11, 4))
    ax_err = fig_err.add_subplot(111)
    for label, (ts, err) in per_run_ape_t_err.items():
        ax_err.plot(ts, err, color=color_map[label], label=label)
    ax_err.set_xlabel("task index")
    ax_err.set_ylabel("APE trans")
    ax_err.set_title("APE translation per task")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()
    fig_err.savefig(out_dir / "ape_translation.png", dpi=120, bbox_inches="tight")
    plt.close(fig_err)
    print(f"wrote {out_dir / 'ape_translation.png'}")

    fig_rot = plt.figure(figsize=(11, 4))
    ax_rot = fig_rot.add_subplot(111)
    for label, (ts, err) in per_run_ape_r_err.items():
        ax_rot.plot(ts, err, color=color_map[label], label=label)
    ax_rot.set_xlabel("task index")
    ax_rot.set_ylabel("APE rot (deg)")
    ax_rot.set_title("APE rotation per task")
    ax_rot.grid(True, alpha=0.3)
    ax_rot.legend()
    fig_rot.savefig(out_dir / "ape_rotation.png", dpi=120, bbox_inches="tight")
    plt.close(fig_rot)
    print(f"wrote {out_dir / 'ape_rotation.png'}")

    # -- metrics.txt --------------------------------------------------------
    lines = [
        f"compare_runs.py output",
        f"ref GT: {ref_dir / 'gt_traj.tum'}",
        f"align:  {args.align}",
        f"rpe_delta: {args.rpe_delta} frame(s)",
        "",
    ]
    for label, m in per_run_metrics.items():
        lines.append(f"[{label}]  ({m['num_poses']} poses)  run_dir={m['run_dir']}")
        for k in ("ape_translation", "ape_rotation_deg",
                  "rpe_translation", "rpe_rotation_deg"):
            s = m[k]
            lines.append(
                f"  {k}: "
                f"rmse={s.get('rmse', float('nan')):.6f} "
                f"mean={s.get('mean', float('nan')):.6f} "
                f"median={s.get('median', float('nan')):.6f} "
                f"std={s.get('std', float('nan')):.6f} "
                f"min={s.get('min', float('nan')):.6f} "
                f"max={s.get('max', float('nan')):.6f}"
            )
        lines.append("")
    text = "\n".join(lines)
    (out_dir / "metrics.txt").write_text(text)
    print(f"wrote {out_dir / 'metrics.txt'}")
    print()
    print(text)


if __name__ == "__main__":
    main()
